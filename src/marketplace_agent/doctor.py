"""`agent doctor`: what is set up and what is not. Makes no model call and changes nothing.

| Check | Fails when |
|---|---|
| config | unknown timezone or platform, no platform, no pickup area or address, no caps for an enabled platform, an empty availability window |
| data dir | missing or not writable |
| database | missing, unreadable, or missing tables |
| models | no provider configured |
| quota | the router's quota state is exhausted |
| platform NAME | not built, cannot be built, or its healthcheck finds problems; when healthy, an adapter with validate_selectors (Facebook) also checks its locators and one with probe (Kijiji) reads every endpoint it uses, both read only |
| discord | a token without a channel id or an owner id |
"""

import importlib
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlmodel import SQLModel

from marketplace_agent.ai.base import Capability, QuotaState
from marketplace_agent.ai.factory import build_router
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core import models  # noqa: F401  registers tables
from marketplace_agent.core.clock import Clock, SystemClock
from marketplace_agent.core.db import make_engine
from marketplace_agent.core.runtime import PLATFORM_PACKAGES
from marketplace_agent.platforms.base import PlatformError

NO_PROVIDER = "no model provider configured; set GEMINI_API_KEY (free key at aistudio.google.com)"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(settings: Settings, clock: Clock | None = None) -> list[Check]:
    clock = clock or SystemClock()
    checks = [
        _safely("config", lambda: _config(settings)),
        _safely("data dir", lambda: _data_dir(settings)),
    ]
    database = _safely("database", lambda: _database(settings))
    checks.append(database)
    checks += _model_checks(settings, clock, database_ok=database.ok)
    checks += [
        _safely(f"platform {name}", partial(_platform, name, settings, clock))
        for name in settings.enabled_platforms
    ]
    checks.append(_safely("discord", lambda: _discord(settings)))
    return checks


def format_checks(checks: list[Check]) -> str:
    rows = [("CHECK", "RESULT", "DETAIL")]
    rows += [(c.name, "ok" if c.ok else "FAIL", c.detail) for c in checks]
    widths = [max(len(row[i]) for row in rows) for i in range(2)]
    return "\n".join(
        f"{row[0].ljust(widths[0])}  {row[1].ljust(widths[1])}  {row[2]}".rstrip() for row in rows
    )


def _safely(name: str, check: Callable[[], Check]) -> Check:
    try:
        return check()
    except Exception as exc:  # a broken check is a failed check, not a crashed doctor
        return Check(name, False, f"{type(exc).__name__}: {exc}")


def _config(settings: Settings) -> Check:
    problems = []
    try:
        ZoneInfo(settings.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        problems.append(f"timezone {settings.timezone!r} is unknown")
    enabled = settings.enabled_platforms
    if not enabled:
        problems.append("no platform is enabled")
    problems += [f"unknown platform {name!r}" for name in enabled if name not in PLATFORM_PACKAGES]
    for name in enabled:
        for field in ("daily_outbound_cap", "daily_publish_cap"):
            if getattr(settings, field).get(name, 0) <= 0:
                problems.append(f"{field} has no limit above 0 for {name}")
    if not settings.pickup_area.strip():
        problems.append(
            "pickup_area is empty: buyers get no meeting area (set MA_PICKUP_AREA to a cross "
            "street)"
        )
    if not settings.pickup_address.strip():
        problems.append(
            "pickup_address is empty: a buyer with a confirmed deal gets no address (set "
            "MA_PICKUP_ADDRESS)"
        )
    if not settings.availability:
        problems.append("no availability windows: no pickup can be booked")
    problems += [
        f"availability window on weekday {w.weekday} ends before it starts"
        for w in settings.availability
        if w.end <= w.start
    ]
    return Check("config", not problems, "; ".join(problems) or f"platforms: {', '.join(enabled)}")


def _data_dir(settings: Settings) -> Check:
    path = settings.data_dir.resolve()
    if not path.is_dir():
        return Check("data dir", False, f"{path} does not exist; run `agent init`")
    try:
        with tempfile.NamedTemporaryFile(dir=path):
            pass
    except OSError as exc:
        return Check("data dir", False, f"{path} is not writable ({exc.strerror})")
    return Check("data dir", True, str(path))


def _database(settings: Settings) -> Check:
    shown = make_url(settings.database_url).render_as_string(hide_password=True)
    file = _sqlite_file(settings.database_url)
    if file is not None and not file.exists():
        return Check("database", False, f"no database at {file}; run `agent init`")
    engine = make_engine(settings.database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        missing = sorted(set(SQLModel.metadata.tables) - set(inspect(engine).get_table_names()))
    except Exception as exc:
        return Check("database", False, f"{shown} cannot be read ({type(exc).__name__})")
    finally:
        engine.dispose()
    if missing:
        return Check(
            "database", False, f"{shown} lacks tables {', '.join(missing)}; run `agent init`"
        )
    return Check("database", True, shown)


def _sqlite_file(database_url: str) -> Path | None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or url.database in (None, "", ":memory:"):
        return None
    return Path(url.database)


def _model_checks(settings: Settings, clock: Clock, *, database_ok: bool) -> list[Check]:
    """Providers need no database; quota state reads today's counts from it."""
    engine = make_engine(settings.database_url)  # connects only when quota is read
    try:
        router = build_router(settings, engine, clock)
        models = _safely("models", lambda: _models(router))
        if not database_ok:
            return [models, Check("quota", False, "skipped: the database is not usable")]
        return [models, _safely("quota", lambda: _quota(router))]
    except Exception as exc:
        failed = f"{type(exc).__name__}: {exc}"
        return [Check("models", False, failed), Check("quota", False, failed)]
    finally:
        engine.dispose()


def _models(router: ProviderRouter) -> Check:
    if not router.providers:
        return Check("models", False, NO_PROVIDER)
    vision = "with" if router.has_capability(Capability.VISION) else "without"
    names = ", ".join(p.name for p in router.providers)
    return Check("models", True, f"{names}; {vision} vision (photo drafts, selector repair)")


def _quota(router: ProviderRouter) -> Check:
    if not router.providers:
        return Check("quota", False, "no model provider")
    left = ", ".join(
        f"{p.name} {_left(router.quota.remaining_today(p.name))}" for p in router.providers
    )
    state = router.state()
    return Check("quota", state is not QuotaState.EXHAUSTED, f"{state.value}; left today: {left}")


def _left(remaining: int | None) -> str:
    return "no limit" if remaining is None else str(remaining)


def _platform(name: str, settings: Settings, clock: Clock) -> Check:
    label = f"platform {name}"
    package = PLATFORM_PACKAGES.get(name)
    if package is None:
        return Check(label, False, "unknown platform")
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        return Check(label, False, f"not built ({exc})")
    factory = getattr(module, "build_adapter", None)
    if factory is None:
        return Check(label, False, "not built yet; remove it from enabled_platforms")
    try:
        adapter = factory(settings, clock)
    except PlatformError as exc:
        return Check(label, False, str(exc))
    try:
        problems = list(adapter.healthcheck())
        validate = getattr(adapter, "validate_selectors", None)
        if validate is not None and not problems:
            problems = [f"selector {problem}" for problem in validate()]
            if not problems:
                return Check(label, True, "signed in; create form and inbox locators match")
        probe = getattr(adapter, "probe", None)
        if probe is not None and not problems:
            return _probe_summary(label, probe())
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()
    return Check(label, not problems, "; ".join(problems) or "healthy")


class ProbeResult(Protocol):
    name: str
    ok: bool | None  # None: skipped
    detail: str


def _probe_summary(label: str, results: Sequence[ProbeResult]) -> Check:
    passed = [r.name for r in results if r.ok is True]
    failed = [f"FAIL {r.name}: {r.detail}" for r in results if r.ok is False]
    skipped = [f"skipped {r.name}: {r.detail}" for r in results if r.ok is None]
    parts = ([f"probe ok: {', '.join(passed)}"] if passed else []) + failed + skipped
    return Check(label, not failed, "; ".join(parts) or "probe found nothing")


def _discord(settings: Settings) -> Check:
    token = settings.discord_token
    if token is None:
        return Check("discord", True, "not configured: notices print to the console")
    if not token.get_secret_value().strip():
        return Check("discord", False, "discord_token is empty")
    ids = ("discord_channel_id", "discord_owner_id")
    missing = [name for name in ids if getattr(settings, name) is None]
    if missing:
        return Check("discord", False, f"token set, but {' and '.join(missing)} missing")
    return Check("discord", True, "token, channel id and owner id set")
