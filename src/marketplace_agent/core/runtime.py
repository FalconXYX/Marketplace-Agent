"""Wires one running agent from Settings. Anything passed in is used as is (tests, simulation).

| Part | Default |
|---|---|
| engine | make_engine(settings.database_url); tables created, pattern cache seeded |
| clock | SystemClock |
| router | ai.factory.build_router |
| notifier | ConsoleNotifier |
| platforms | build_platforms: one adapter per enabled platform |
"""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import Engine

from marketplace_agent.agent.loop import Agent
from marketplace_agent.ai import pattern_cache
from marketplace_agent.ai.factory import build_router
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.control.service import ControlService
from marketplace_agent.core.clock import Clock, SystemClock
from marketplace_agent.core.db import init_db, make_engine, session_scope
from marketplace_agent.core.migrate import upgrade_schema
from marketplace_agent.notify.base import ConsoleNotifier, Notifier
from marketplace_agent.platforms.base import PlatformAdapter, PlatformError
from marketplace_agent.runner.tick import Runner
from marketplace_agent.tools.registry import ToolRegistry, default_registry

PLATFORM_PACKAGES = {
    "fake": "marketplace_agent.platforms.fake",
    "kijiji": "marketplace_agent.platforms.kijiji",
    "facebook": "marketplace_agent.platforms.facebook",
}


class PlatformSetupError(RuntimeError):
    """An enabled platform cannot be built (unknown, not importable, no build_adapter, or its
    build_adapter raised PlatformError, e.g. missing credentials); the message says what to
    change."""


@dataclass
class Runtime:
    settings: Settings
    engine: Engine
    clock: Clock
    router: ProviderRouter
    notifier: Notifier
    platforms: dict[str, PlatformAdapter]
    registry: ToolRegistry
    agent: Agent
    runner: Runner
    control: ControlService


def build_platforms(settings: Settings, clock: Clock) -> dict[str, PlatformAdapter]:
    """One adapter per enabled platform, from build_adapter(settings, clock) in its package.
    Packages are imported only when enabled, so an unused adapter's dependencies never load.
    One PlatformSetupError names every platform that cannot be built, so one fix round does."""
    adapters: dict[str, PlatformAdapter] = {}
    problems: list[PlatformSetupError] = []
    for name in settings.enabled_platforms:
        try:
            adapters[name] = _build_adapter(name, settings, clock)
        except PlatformSetupError as exc:
            problems.append(exc)
    if not problems:
        return adapters
    for adapter in adapters.values():
        close = getattr(adapter, "close", None)  # e.g. Facebook's browser thread
        if close is not None:
            close()
    if len(problems) == 1:
        raise problems[0]
    raise PlatformSetupError("\n".join(map(str, problems))) from problems[0]


def build_runtime(
    settings: Settings | None = None,
    *,
    clock: Clock | None = None,
    notifier: Notifier | None = None,
    platforms: Mapping[str, PlatformAdapter] | None = None,
    router: ProviderRouter | None = None,
    engine: Engine | None = None,
) -> Runtime:
    settings = settings or Settings()
    clock = clock or SystemClock()
    if engine is None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        engine = make_engine(settings.database_url)
    init_db(engine)
    upgrade_schema(engine)
    with session_scope(engine) as session:
        pattern_cache.seed(session, clock)
    if router is None:
        router = build_router(settings, engine, clock)
    if notifier is None:
        notifier = ConsoleNotifier()
    adapters = dict(platforms) if platforms is not None else build_platforms(settings, clock)
    registry = default_registry()
    agent = Agent(router, registry, settings, clock, notifier, adapters)
    return Runtime(
        settings=settings,
        engine=engine,
        clock=clock,
        router=router,
        notifier=notifier,
        platforms=adapters,
        registry=registry,
        agent=agent,
        runner=Runner(engine, clock, settings, adapters, agent, router, notifier, registry),
        control=ControlService(engine, clock, settings, adapters, notifier, registry),
    )


def _build_adapter(name: str, settings: Settings, clock: Clock) -> PlatformAdapter:
    package = PLATFORM_PACKAGES.get(name)
    if package is None:
        known = ", ".join(PLATFORM_PACKAGES)
        raise PlatformSetupError(
            f"Unknown platform {name!r} in enabled_platforms. Use any of: {known}."
        )
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise PlatformSetupError(
            f"The {name} adapter could not load ({exc}). Install what it needs, or remove "
            f"{name!r} from enabled_platforms."
        ) from exc
    factory = getattr(module, "build_adapter", None)
    if factory is None:
        raise PlatformSetupError(
            f"The {name} adapter is not built yet. Remove {name!r} from enabled_platforms "
            "(MA_ENABLED_PLATFORMS or marketplace_agent.toml)."
        )
    try:
        adapter: PlatformAdapter = factory(settings, clock)
    except PlatformError as exc:  # missing credentials or session: the message says what to do
        raise PlatformSetupError(str(exc)) from exc
    return adapter
