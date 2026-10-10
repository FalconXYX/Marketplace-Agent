"""core.runtime: wiring one running agent from Settings."""

import re
import sys
import types

import pytest
from sqlmodel import Session, func, select

from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core import runtime as runtime_module
from marketplace_agent.core.clock import SystemClock
from marketplace_agent.core.items import create_item, set_goal
from marketplace_agent.core.models import Item, PatternEntry, Photo
from marketplace_agent.core.runtime import PlatformSetupError, build_platforms, build_runtime
from marketplace_agent.notify.base import ConsoleNotifier
from marketplace_agent.platforms.base import PlatformAuthError
from marketplace_agent.platforms.facebook import FacebookAdapter
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.platforms.kijiji import KijijiAdapter
from marketplace_agent.tools.impl import TOOL_CLASSES

KEY_VARS = (
    "MA_GEMINI_API_KEY",
    "GEMINI_API_KEY",
    "MA_GROQ_API_KEY",
    "GROQ_API_KEY",
    "MA_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture
def offline(monkeypatch):
    """No provider keys from the developer's shell, so nothing can reach a real model."""
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)


def _settings(tmp_path, **overrides) -> Settings:
    fields = {
        "_env_file": None,
        "data_dir": tmp_path / "data",
        "db_url": "sqlite://",
        "pickup_area": "Bloor & Ossington",
        "pickup_address": "123 Example St",
    }
    return Settings(**{**fields, **overrides})


def _patterns(engine) -> int:
    with Session(engine) as session:
        return session.exec(select(func.count()).select_from(PatternEntry)).one()


def test_defaults_build_every_part(tmp_path, offline):
    settings = _settings(tmp_path)

    runtime = build_runtime(settings)

    assert runtime.settings is settings
    assert isinstance(runtime.clock, SystemClock)
    assert isinstance(runtime.notifier, ConsoleNotifier)
    assert isinstance(runtime.router, ProviderRouter) and runtime.router.providers == []
    assert list(runtime.platforms) == ["fake"]
    assert isinstance(runtime.platforms["fake"], FakeMarketplace)
    assert runtime.registry.names() == [cls.name for cls in TOOL_CLASSES]
    assert (tmp_path / "data").is_dir()
    assert _patterns(runtime.engine) > 0
    runtime.engine.dispose()


def test_given_parts_are_used_and_shared(tmp_path, engine, clock, notifier):
    router = ProviderRouter([ScriptedProvider([])], QuotaTracker(engine, clock, {}))
    market = FakeMarketplace(clock)

    runtime = build_runtime(
        _settings(tmp_path),
        clock=clock,
        notifier=notifier,
        platforms={"fake": market},
        router=router,
        engine=engine,
    )

    assert (runtime.engine, runtime.clock, runtime.router) == (engine, clock, router)
    assert runtime.platforms == {"fake": market}
    for part in (runtime.agent, runtime.runner, runtime.control):
        assert part.platforms is runtime.platforms
        assert part.notifier is notifier
        assert part.clock is clock
    assert runtime.agent.registry is runtime.registry is runtime.runner.registry
    assert runtime.runner.agent is runtime.agent and runtime.runner.router is router
    assert runtime.control.engine is engine


def test_building_twice_on_one_database_seeds_once(tmp_path, engine, clock, notifier):
    build_runtime(_settings(tmp_path), clock=clock, notifier=notifier, engine=engine)
    seeded = _patterns(engine)

    build_runtime(_settings(tmp_path), clock=clock, notifier=notifier, engine=engine)

    assert _patterns(engine) == seeded


def test_runtime_runs_a_tick_end_to_end(tmp_path, clock, notifier, offline):
    runtime = build_runtime(_settings(tmp_path), clock=clock, notifier=notifier)
    with Session(runtime.engine) as session:
        item = create_item(session, clock, "desk $120")
        item.title, item.description = "IKEA desk", "Solid."
        session.add(Photo(item_id=item.id, path="p.jpg", width=1, height=1, sha256="x"))
        set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
        session.commit()
        item_id = item.id
    [published] = runtime.control.confirm_and_publish(item_id)
    market = runtime.platforms["fake"]
    [listing] = market.listings
    market.buyer_say(market.add_buyer("scripted", listing), "Is this still available?")

    report = runtime.runner.tick()

    assert published.ok
    assert (report.ingested, report.cache_replies, report.wakes) == (1, 1, 0)
    assert market.sent[0][1] == "Hi! Yes, it's still available. Pickup is near Bloor & Ossington."
    with Session(runtime.engine) as session:
        assert session.get(Item, item_id).status.value == "listed"


def test_build_platforms_makes_one_adapter_per_enabled_platform(tmp_path, clock):
    platforms = build_platforms(_settings(tmp_path, enabled_platforms=["fake"]), clock)

    assert list(platforms) == ["fake"]
    assert platforms["fake"].clock is clock


@pytest.mark.parametrize("name", ["kijiji", "facebook"])
def test_unbuilt_adapter_is_a_clear_error(tmp_path, clock, name, monkeypatch):
    """A platform package that exists but has no build_adapter (both are built now, so an
    empty module stands in for the package)."""
    package = runtime_module.PLATFORM_PACKAGES[name]
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))

    with pytest.raises(PlatformSetupError, match=f"The {name} adapter is not built yet"):
        build_platforms(_settings(tmp_path, enabled_platforms=[name]), clock)


@pytest.mark.parametrize(
    ("name", "says"),
    [
        ("kijiji", "set MA_KIJIJI_EMAIL and MA_KIJIJI_PASSWORD"),
        ("facebook", "run `agent facebook login` once"),
    ],
)
def test_adapter_without_credentials_is_a_clear_error(tmp_path, clock, offline, name, says):
    no_account = {"kijiji_email": None, "kijiji_password": None}
    settings = _settings(tmp_path, enabled_platforms=[name], **no_account)

    with pytest.raises(PlatformSetupError, match=re.escape(says)) as raised:
        build_platforms(settings, clock)

    assert isinstance(raised.value.__cause__, PlatformAuthError)
    assert f"remove '{name}' from enabled_platforms" in str(raised.value)
    assert not settings.fb_profile_dir.exists()  # no browser was started to find out


def test_every_platform_that_cannot_be_built_is_named_at_once(tmp_path, clock, offline):
    no_account = {"kijiji_email": None, "kijiji_password": None}
    settings = _settings(tmp_path, enabled_platforms=["fake", "kijiji", "facebook"], **no_account)

    with pytest.raises(PlatformSetupError) as raised:
        build_platforms(settings, clock)

    lines = str(raised.value).splitlines()
    assert len(lines) == 2
    assert "MA_KIJIJI_EMAIL" in lines[0] and "agent facebook login" in lines[1]


def test_kijiji_with_an_account_builds_without_the_network(tmp_path, clock):
    settings = _settings(
        tmp_path, enabled_platforms=["kijiji"], kijiji_email="me@example.com", kijiji_password="pw"
    )

    adapter = build_platforms(settings, clock)["kijiji"]

    assert isinstance(adapter, KijijiAdapter) and adapter.name == "kijiji"


def test_facebook_with_a_profile_builds_without_starting_a_browser(tmp_path, clock, offline):
    settings = _settings(tmp_path, enabled_platforms=["facebook"])
    settings.fb_profile_dir.mkdir(parents=True)

    adapter = build_platforms(settings, clock)["facebook"]

    assert isinstance(adapter, FacebookAdapter) and adapter.name == "facebook"
    assert list(settings.fb_profile_dir.iterdir()) == []  # Chromium never touched the profile
    adapter.close()


def test_unknown_platform_is_a_clear_error(tmp_path, clock):
    with pytest.raises(PlatformSetupError, match="Unknown platform 'ebay'.*fake, kijiji"):
        build_platforms(_settings(tmp_path, enabled_platforms=["ebay"]), clock)


def test_adapter_that_cannot_import_is_a_clear_error(tmp_path, clock, monkeypatch):
    monkeypatch.setitem(runtime_module.PLATFORM_PACKAGES, "kijiji", "marketplace_agent.nope")

    with pytest.raises(PlatformSetupError, match="The kijiji adapter could not load"):
        build_platforms(_settings(tmp_path, enabled_platforms=["kijiji"]), clock)


def test_runtime_with_an_unbuilt_platform_raises(tmp_path, clock, offline):
    settings = _settings(tmp_path, enabled_platforms=["fake", "kijiji"])

    with pytest.raises(PlatformSetupError, match="kijiji"):
        build_runtime(settings, clock=clock)
