"""core.serve: wiring (no bot without a token, a stub web app), shutdown order, signals, and the
`agent serve` command. No socket is opened and nothing talks to Discord."""

import asyncio
import os
import signal
import sys
import threading
import types

import pytest
import typer
from pydantic import SecretStr
from typer.testing import CliRunner

from marketplace_agent.core import serve as serve_module
from marketplace_agent.core.runtime import build_runtime
from marketplace_agent.core.serve import ServeError, run_services
from marketplace_agent.notify.base import ConsoleNotifier, RecordingNotifier
from marketplace_agent.notify.commands import register
from marketplace_agent.platforms.fake import FakeMarketplace


async def stub_app(scope, receive, send) -> None:
    raise AssertionError("the stub web app is never served in these tests")


@pytest.fixture
def web_app(monkeypatch) -> list:
    """Stands in for marketplace_agent.web.app; records the runtime create_app got."""
    created: list = []
    module = types.ModuleType("marketplace_agent.web.app")

    def create_app(runtime):
        created.append(runtime)
        return stub_app

    module.create_app = create_app
    monkeypatch.setitem(sys.modules, "marketplace_agent.web.app", module)
    return created


@pytest.fixture
def started(monkeypatch) -> dict:
    """serve() up to run_services, which records what it was given instead of running."""
    got: dict = {}

    async def fake_run_services(runtime, *, web, bot):
        got.update(runtime=runtime, web=web, bot=bot)

    monkeypatch.setattr(serve_module, "run_services", fake_run_services)
    return got


def test_serve_without_a_token_prints_notices_and_runs_no_bot(settings, web_app, started):
    serve_module.serve(settings)

    runtime, web = started["runtime"], started["web"]
    assert started["bot"] is None
    assert isinstance(runtime.notifier, ConsoleNotifier)
    assert web_app == [runtime]
    assert (web.config.app, web.config.host, web.config.port) == (stub_app, "127.0.0.1", 8765)
    before = signal.getsignal(signal.SIGINT)
    with web.capture_signals():
        assert signal.getsignal(signal.SIGINT) is before


def test_the_web_server_keeps_the_token_out_of_the_access_log(settings, web_app, started):
    import logging

    from marketplace_agent.web.commands import redact_token

    access = logging.getLogger("uvicorn.access")
    access.removeFilter(redact_token)  # `agent web` tests may have added it already

    serve_module.serve(settings)

    assert redact_token in access.filters


def test_serve_with_a_token_wires_the_bot_to_one_discord_notifier(settings, web_app, started):
    from marketplace_agent.notify.discord_bot import DiscordBot, DiscordNotifier

    settings.discord_token = SecretStr("token")
    settings.discord_channel_id, settings.discord_owner_id = 1, 2

    serve_module.serve(settings)

    runtime, bot = started["runtime"], started["bot"]
    assert isinstance(bot, DiscordBot) and isinstance(runtime.notifier, DiscordNotifier)
    assert bot.notifier is runtime.notifier is runtime.control.notifier is runtime.runner.notifier
    assert bot.runtime is runtime


def test_serve_with_a_token_needs_the_channel_and_the_owner(settings, web_app, started):
    settings.discord_token = SecretStr("token")

    with pytest.raises(ServeError, match="MA_DISCORD_CHANNEL_ID and MA_DISCORD_OWNER_ID"):
        serve_module.serve(settings)
    assert started == {}


# ---------------------------------------------------------------- run_services


class Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def add(self, event: str) -> None:
        self.events.append(event)


def _runner_alive() -> bool:
    return any(t.name == serve_module.RUNNER_THREAD and t.is_alive() for t in threading.enumerate())


class FakeWeb:
    """uvicorn.Server's surface: serve() returns once should_exit is set."""

    def __init__(self, log: Recorder, *, crash_at_start: bool = False) -> None:
        self.log = log
        self.crash_at_start = crash_at_start
        self._exit = False

    @property
    def should_exit(self) -> bool:
        return self._exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self.log.add(f"web asked to stop, runner alive={_runner_alive()}")
        self._exit = value

    async def serve(self) -> None:
        if self.crash_at_start:
            return
        while not self._exit:
            await asyncio.sleep(0.005)
        self.log.add("web stopped")


class FakeBot:
    def __init__(self, log: Recorder, web: FakeWeb, error: Exception | None = None) -> None:
        self.log = log
        self.web = web
        self.error = error
        self.closed = asyncio.Event()

    async def run_until_closed(self) -> None:
        if self.error is not None:
            raise self.error
        await self.closed.wait()

    async def shutdown(self) -> None:
        self.log.add(f"bot shutdown, runner alive={_runner_alive()}, web exit={self.web._exit}")
        self.closed.set()


@pytest.fixture
def runtime(settings, engine, clock):
    return build_runtime(
        settings,
        clock=clock,
        engine=engine,
        notifier=RecordingNotifier(),
        platforms={"fake": FakeMarketplace(clock)},
    )


@pytest.fixture
def ticks(runtime) -> threading.Event:
    """Set after the runner's first real tick."""
    ticked = threading.Event()
    real = runtime.runner.tick

    def tick():
        report = real()
        ticked.set()
        return report

    runtime.runner.tick = tick
    return ticked


async def test_runs_until_stopped_then_stops_runner_web_and_bot_in_order(runtime, ticks):
    log = Recorder()
    web = FakeWeb(log)
    stop = asyncio.Event()

    task = asyncio.create_task(run_services(runtime, web=web, bot=FakeBot(log, web), stop=stop))
    assert await asyncio.to_thread(ticks.wait, 5)
    assert not task.done()
    stop.set()
    await asyncio.wait_for(task, 5)

    assert log.events == [
        "web asked to stop, runner alive=False",
        "bot shutdown, runner alive=False, web exit=True",
        "web stopped",
    ]


async def test_sigterm_stops_everything_gracefully(runtime, ticks):
    log = Recorder()
    web = FakeWeb(log)

    task = asyncio.create_task(run_services(runtime, web=web, bot=FakeBot(log, web)))
    assert await asyncio.to_thread(ticks.wait, 5)
    assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, None)  # ours, not the default
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, 5)

    assert log.events[0] == "web asked to stop, runner alive=False"
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


async def test_a_second_stop_does_not_wait_for_the_runners_tick(runtime):
    in_tick, release = threading.Event(), threading.Event()

    def slow_tick():
        in_tick.set()
        release.wait(5)
        raise RuntimeError("tick cut short")

    runtime.runner.tick = slow_tick
    stop, force = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(run_services(runtime, web=None, bot=None, stop=stop, force=force))
    assert await asyncio.to_thread(in_tick.wait, 5)

    stop.set()
    await asyncio.sleep(0.2)
    assert not task.done()  # waiting for the tick
    force.set()
    await asyncio.wait_for(task, 5)

    assert _runner_alive()
    release.set()
    await asyncio.to_thread(_join_runner)


def _join_runner() -> None:
    for thread in threading.enumerate():
        if thread.name == serve_module.RUNNER_THREAD:
            thread.join(5)


async def test_a_part_that_stops_on_its_own_stops_everything_and_fails(runtime):
    log = Recorder()
    web = FakeWeb(log, crash_at_start=True)

    with pytest.raises(ServeError, match="The web server stopped on its own"):
        await asyncio.wait_for(run_services(runtime, web=web, bot=FakeBot(log, web)), 5)

    assert not _runner_alive()
    assert log.events[-1].startswith("bot shutdown")


async def test_a_web_server_that_cannot_bind_its_port_explains_itself(runtime):
    class ExitingWeb(FakeWeb):
        async def serve(self) -> None:
            raise SystemExit(3)  # what uvicorn does when the port is taken

    log = Recorder()
    web = ExitingWeb(log)

    with pytest.raises(ServeError, match="could not start on 127.0.0.1:8765"):
        await asyncio.wait_for(run_services(runtime, web=web, bot=FakeBot(log, web)), 5)

    assert not _runner_alive()


async def test_a_part_that_crashes_raises_its_own_error(runtime):
    log = Recorder()
    web = FakeWeb(log)
    bad_token = ServeError("Discord rejected the bot token")

    with pytest.raises(ServeError, match="rejected the bot token"):
        await asyncio.wait_for(run_services(runtime, web=web, bot=FakeBot(log, web, bad_token)), 5)

    assert "web stopped" in log.events and not _runner_alive()


# ---------------------------------------------------------------- agent serve


def _cli() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def main() -> None:
        """Test root, so `serve` is a subcommand."""

    register(app)
    return app


def test_agent_serve_runs_serve_with_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(serve_module, "serve", lambda settings: calls.append(settings))

    result = CliRunner().invoke(_cli(), ["serve"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and result.output.endswith("Stopped.\n")


def test_agent_serve_reports_setup_problems_without_a_traceback(monkeypatch):
    def refuse(settings):
        raise ServeError("MA_DISCORD_TOKEN is set, so the Discord bot needs MA_DISCORD_OWNER_ID")

    monkeypatch.setattr(serve_module, "serve", refuse)

    result = CliRunner().invoke(_cli(), ["serve"])

    assert result.exit_code == 1
    assert "needs MA_DISCORD_OWNER_ID" in result.output and "Traceback" not in result.output
