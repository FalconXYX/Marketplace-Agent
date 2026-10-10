"""`agent serve`: everything in one process.

| Part | Runs as |
|---|---|
| Runner | A thread: ticks every tick_seconds until its stop event is set |
| Web intake | uvicorn on the asyncio loop, at web_host:web_port |
| Discord bot | On the same loop when discord_token is set; else notices print to the console |

SIGINT or SIGTERM stops the parts in order: the runner first, after its current tick, so the
notices of that tick still reach Discord; then the web server; then the bot. A second signal
stops at once, without waiting for the tick.
"""

import asyncio
import contextlib
import logging
import signal
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Protocol

import uvicorn

from marketplace_agent.config import Settings
from marketplace_agent.core.runtime import Runtime, build_runtime

if TYPE_CHECKING:
    from marketplace_agent.notify.discord_bot import DiscordBot, DiscordNotifier

RUNNER_THREAD = "marketplace-runner"
JOIN_POLL_S = 0.1
_SIGNALS = (signal.SIGINT, signal.SIGTERM)

logger = logging.getLogger("marketplace_agent.serve")


class ServeError(RuntimeError):
    """serve cannot start or a part of it stopped; the message says what to do."""


class WebServer(Protocol):
    should_exit: bool

    async def serve(self) -> None: ...


class Bot(Protocol):
    async def run_until_closed(self) -> None: ...

    async def shutdown(self) -> None: ...


class _Server(uvicorn.Server):
    """serve owns SIGINT and SIGTERM, so uvicorn must neither take nor re-raise them."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def serve(settings: Settings | None = None) -> None:
    """Run the runner, the web app, and (with a token) the Discord bot until a signal."""
    settings = settings or Settings()
    _check_discord(settings)
    notifier = _discord_notifier(settings)
    runtime = build_runtime(settings, notifier=notifier)
    if not runtime.router.providers:
        logger.warning("No model is configured (set GEMINI_API_KEY): buyers get escalated to you.")
    try:
        bot = _discord_bot(runtime, notifier) if notifier is not None else None
        web = build_web_server(runtime)
        logger.info(
            "Serving: web on http://%s:%s, Discord %s, platforms %s. Ctrl+C stops.",
            settings.web_host,
            settings.web_port,
            "on" if bot is not None else "off (notices print here)",
            ", ".join(runtime.platforms) or "none",
        )
        asyncio.run(run_services(runtime, web=web, bot=bot))
    finally:
        runtime.engine.dispose()


def build_web_server(runtime: Runtime) -> uvicorn.Server:
    from marketplace_agent.web.app import create_app
    from marketplace_agent.web.commands import redact_token

    settings = runtime.settings
    config = uvicorn.Config(create_app(runtime), host=settings.web_host, port=settings.web_port)
    logging.getLogger("uvicorn.access").addFilter(redact_token)  # ?token= stays out of the log
    return _Server(config)


async def run_services(
    runtime: Runtime,
    *,
    web: WebServer | None,
    bot: Bot | None,
    stop: asyncio.Event | None = None,
    force: asyncio.Event | None = None,
) -> None:
    """Until stop is set (by a signal, or by the caller) or a part ends on its own; then shut
    down in order. force (a second signal) skips waiting for the runner's tick. Raises
    ServeError, or the part's own error, when a part ended without being asked to."""
    stop = stop or asyncio.Event()
    force = force or asyncio.Event()
    loop = asyncio.get_running_loop()
    handled = _handle_signals(loop, stop, force)
    runner_stop = threading.Event()
    runner = threading.Thread(
        target=runtime.runner.run_forever,
        kwargs={"stop": runner_stop, "sleep": runner_stop.wait},
        name=RUNNER_THREAD,
        daemon=True,
    )
    runner.start()
    parts: dict[str, asyncio.Task[None]] = {}
    if web is not None:
        parts["web server"] = asyncio.create_task(_serve_web(web, runtime.settings), name="web")
    if bot is not None:
        parts["Discord bot"] = asyncio.create_task(bot.run_until_closed(), name="discord")
    ended: set[str] = set()
    try:
        await _first_of(stop, parts.values())
        ended = {name for name, task in parts.items() if task.done()}
    finally:
        await _shut_down(runner, runner_stop, force, web, bot, parts)
        for sig in handled:
            loop.remove_signal_handler(sig)
    _raise_unexpected(parts, ended, stopped=stop.is_set())


async def _serve_web(web: WebServer, settings: Settings) -> None:
    try:
        await web.serve()
    except SystemExit as exc:  # uvicorn exits the process when it cannot start
        raise ServeError(
            f"The web server could not start on {settings.web_host}:{settings.web_port} (see the "
            "error above). If the port is taken, stop the other server or set MA_WEB_PORT."
        ) from exc


def _handle_signals(
    loop: asyncio.AbstractEventLoop, stop: asyncio.Event, force: asyncio.Event
) -> list[signal.Signals]:
    """First signal: stop gracefully; second: stop now. Only the main thread can take signals."""
    if threading.current_thread() is not threading.main_thread():
        return []

    def on_signal() -> None:
        if stop.is_set():
            force.set()
            return
        logger.info("Stopping after the current tick. Press Ctrl+C again to stop now.")
        stop.set()

    for sig in _SIGNALS:
        loop.add_signal_handler(sig, on_signal)
    return list(_SIGNALS)


async def _first_of(stop: asyncio.Event, parts: Iterable[asyncio.Task[None]]) -> None:
    waiter: asyncio.Task[object] = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait([waiter, *parts], return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()


async def _shut_down(
    runner: threading.Thread,
    runner_stop: threading.Event,
    force: asyncio.Event,
    web: WebServer | None,
    bot: Bot | None,
    parts: Mapping[str, asyncio.Task[None]],
) -> None:
    runner_stop.set()
    while runner.is_alive() and not force.is_set():
        await asyncio.sleep(JOIN_POLL_S)
    if runner.is_alive():
        logger.warning("Stopping without waiting for the runner to finish its tick.")
    if web is not None:
        web.should_exit = True
    if bot is not None:
        await bot.shutdown()
    await asyncio.gather(*parts.values(), return_exceptions=True)


def _raise_unexpected(
    parts: Mapping[str, asyncio.Task[None]], ended: set[str], *, stopped: bool
) -> None:
    """Raise for a part that ended before the shutdown; only log one that failed during it."""
    for name, task in parts.items():
        error = None if task.cancelled() else task.exception()
        if name not in ended:
            if error is not None:
                logger.warning("The %s failed while stopping: %s", name, error)
            continue
        if error is not None:
            raise error
        if not stopped:
            raise ServeError(f"The {name} stopped on its own; see the log above for why.")


def _check_discord(settings: Settings) -> None:
    if settings.discord_token is None:
        return
    missing = [
        env
        for env, value in (
            ("MA_DISCORD_CHANNEL_ID", settings.discord_channel_id),
            ("MA_DISCORD_OWNER_ID", settings.discord_owner_id),
        )
        if value is None
    ]
    if missing:
        raise ServeError(
            f"MA_DISCORD_TOKEN is set, so the Discord bot needs {' and '.join(missing)} too: the "
            "channel it posts in and your own Discord user id (only you may press its buttons)."
        )


def _discord_notifier(settings: Settings) -> "DiscordNotifier | None":
    if settings.discord_token is None:
        return None
    from marketplace_agent.notify.discord_bot import DiscordNotifier

    return DiscordNotifier()


def _discord_bot(runtime: Runtime, notifier: "DiscordNotifier") -> "DiscordBot":
    from marketplace_agent.notify.discord_bot import DiscordBot

    return DiscordBot(runtime, notifier)
