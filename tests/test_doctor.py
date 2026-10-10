"""run_doctor: config, data dir, database, models and quota, platforms, Discord. No network."""

import sys
import types
from datetime import time

import httpx
import pytest
from fixtures.facebook.fake_driver import FakeDriver, FakeSite
from test_kijiji_support import EMAIL, PASSWORD, FakeKijiji, make_client

from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.config import AvailabilityWindow, Settings
from marketplace_agent.core.db import init_db, make_engine
from marketplace_agent.doctor import NO_PROVIDER, Check, format_checks, run_doctor
from marketplace_agent.platforms.base import PlatformAuthError
from marketplace_agent.platforms.facebook.adapter import FacebookAdapter
from marketplace_agent.platforms.kijiji.adapter import KijijiAdapter

KIJIJI = "marketplace_agent.platforms.kijiji"
FACEBOOK = "marketplace_agent.platforms.facebook"
KEY_VARS = (
    "MA_GEMINI_API_KEY",
    "GEMINI_API_KEY",
    "MA_GROQ_API_KEY",
    "GROQ_API_KEY",
    "MA_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "MA_DISCORD_TOKEN",
)


@pytest.fixture
def make_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no marketplace_agent.toml from the developer
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)

    def make(**changes: object) -> Settings:
        fields: dict = dict(
            _env_file=None,
            data_dir=tmp_path / "data",
            pickup_area="Bloor St W & Ossington Ave",
            pickup_address="123 Example St",
            enabled_platforms=["fake"],
        )
        fields.update(changes)
        return Settings(**fields)

    return make


def _initialized(settings: Settings) -> Settings:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.database_url)
    init_db(engine)
    engine.dispose()
    return settings


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


def _module(monkeypatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_a_fresh_install_says_to_run_init_and_creates_nothing(make_settings):
    settings = make_settings()
    checks = _by_name(run_doctor(settings))
    assert not settings.data_dir.exists()

    assert checks["config"].ok
    assert not checks["data dir"].ok and "agent init" in checks["data dir"].detail
    assert not checks["database"].ok and "agent init" in checks["database"].detail
    assert checks["models"] == Check("models", False, NO_PROVIDER)
    assert "skipped" in checks["quota"].detail
    assert checks["platform fake"].ok
    assert checks["discord"] == Check(
        "discord", True, "not configured: notices print to the console"
    )


def test_an_initialized_install_without_a_model(make_settings):
    checks = _by_name(run_doctor(_initialized(make_settings())))
    assert checks["data dir"].ok and checks["database"].ok
    assert not checks["models"].ok and "GEMINI_API_KEY" in checks["models"].detail
    assert not checks["quota"].ok


def test_a_missing_table_says_to_run_init(make_settings):
    settings = make_settings()
    settings.data_dir.mkdir(parents=True)
    engine = make_engine(settings.database_url)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE unrelated (id INTEGER)")
    engine.dispose()
    database = _by_name(run_doctor(settings))["database"]
    assert not database.ok and "lacks tables" in database.detail


def test_models_and_quota_without_calling_a_model(make_settings, clock):
    settings = _initialized(make_settings(gemini_api_key="test-key"))
    checks = _by_name(run_doctor(settings, clock))
    assert checks["models"].ok and "gemini; with vision" in checks["models"].detail
    assert checks["quota"].ok and checks["quota"].detail.startswith(
        "healthy; left today: gemini 250"
    )

    engine = make_engine(settings.database_url)
    tracker = QuotaTracker(engine, clock, settings.provider_limits)
    for _ in range(settings.provider_limits["gemini"].rpd):
        tracker.record("gemini", settings.reasoning_model)
    engine.dispose()
    quota = _by_name(run_doctor(settings, clock))["quota"]
    assert not quota.ok and quota.detail.startswith("exhausted")


def test_config_problems_are_listed(make_settings):
    settings = make_settings(
        timezone="Mars/Olympus_Mons",
        enabled_platforms=["fake", "myspace"],
        pickup_area="",
        daily_outbound_cap={"fake": 0},
        availability=[AvailabilityWindow(weekday=0, start=time(18), end=time(17))],
    )
    checks = _by_name(run_doctor(settings))
    detail = checks["config"].detail
    for problem in (
        "timezone 'Mars/Olympus_Mons' is unknown",
        "unknown platform 'myspace'",
        "daily_outbound_cap has no limit above 0 for fake",
        "pickup_area is empty",
        "weekday 0 ends before it starts",
    ):
        assert problem in detail
    assert "123 Example St" not in format_checks(list(checks.values()))
    assert checks["platform myspace"] == Check("platform myspace", False, "unknown platform")


def test_kijiji_not_built_yet(make_settings, monkeypatch):
    monkeypatch.setitem(sys.modules, KIJIJI, None)  # import raises ImportError
    check = _by_name(run_doctor(make_settings(enabled_platforms=["kijiji"])))["platform kijiji"]
    assert not check.ok and check.detail.startswith("not built")

    _module(monkeypatch, KIJIJI)
    check = _by_name(run_doctor(make_settings(enabled_platforms=["kijiji"])))["platform kijiji"]
    assert check.detail == "not built yet; remove it from enabled_platforms"


def test_kijiji_build_and_health_problems(make_settings, monkeypatch):
    def no_credentials(settings, clock):
        raise PlatformAuthError("Set kijiji_email and kijiji_password.")

    _module(monkeypatch, KIJIJI, build_adapter=no_credentials)
    settings = make_settings(enabled_platforms=["kijiji"])
    assert (
        _by_name(run_doctor(settings))["platform kijiji"].detail
        == "Set kijiji_email and kijiji_password."
    )

    class Unhealthy:
        def healthcheck(self) -> list[str]:
            return ["kijiji: login rejected"]

    _module(monkeypatch, KIJIJI, build_adapter=lambda settings, clock: Unhealthy())
    check = _by_name(run_doctor(settings))["platform kijiji"]
    assert check == Check("platform kijiji", False, "kijiji: login rejected")


def test_a_crashing_check_fails_without_stopping_the_doctor(make_settings, monkeypatch):
    class Broken:
        def healthcheck(self) -> list[str]:
            raise RuntimeError("boom")

    _module(monkeypatch, KIJIJI, build_adapter=lambda settings, clock: Broken())
    checks = _by_name(run_doctor(make_settings(enabled_platforms=["kijiji", "fake"])))
    assert checks["platform kijiji"] == Check("platform kijiji", False, "RuntimeError: boom")
    assert checks["platform fake"].ok and "discord" in checks


@pytest.fixture
def kijiji(monkeypatch):
    """The kijiji package's build_adapter over a FakeKijiji; yields the fake."""
    fake = FakeKijiji()
    package = __import__(KIJIJI, fromlist=["build_adapter"])

    def build(settings, clock):
        client, _ = make_client(fake)
        return KijijiAdapter(
            client,
            clock,
            location_id=settings.kijiji_location_id,
            postal_code=settings.kijiji_postal_code,
        )

    monkeypatch.setattr(package, "build_adapter", build)
    return fake


def _kijiji_check(make_settings, **changes: object) -> Check:
    fields: dict = dict(
        enabled_platforms=["kijiji"], kijiji_postal_code="M6J 1Z1", kijiji_location_id=1700273
    )
    fields.update(changes)
    return _by_name(run_doctor(make_settings(**fields)))["platform kijiji"]


def test_kijiji_probe_summary_reads_every_endpoint_and_writes_nothing(make_settings, kijiji):
    check = _kijiji_check(make_settings)

    assert check == Check(
        "platform kijiji",
        True,
        "probe ok: search (comps), location, default category, sign in, my ads, conversations, "
        "messages, post form",
    )
    assert kijiji.sign_ins == 1  # the probe reuses the healthcheck's session
    read = {
        "GetSearchResultsPageByUrl",
        "getLocationFromId",
        "GetCategories",
        "GetPosterOtherAds",
        "GetListing",
        "GetConversations",
        "GetConversation",
    }
    assert read <= set(kijiji.graphql_calls)
    assert kijiji.calls("/p-post-ad.html") == 1
    assert kijiji.submitted == [] and kijiji.sent == [] and kijiji.deleted == []
    assert kijiji.uploads == []


def test_kijiji_probe_failures_fail_the_check_and_skips_do_not(make_settings, kijiji):
    check = _kijiji_check(make_settings, kijiji_location_id=None)
    assert check.ok
    assert check.detail.endswith(
        "; skipped location: MA_KIJIJI_LOCATION_ID not set; search covers all of Canada"
    )
    assert "location," not in check.detail

    kijiji.queue("/p-post-ad.html", httpx.Response(200, html="<form><input name='q'></form>"))
    check = _kijiji_check(make_settings)
    assert not check.ok
    assert "post form" not in check.detail.split(";")[0]
    assert check.detail.endswith(
        "; FAIL post form: kijiji: no ad form on /p-post-ad.html; Kijiji changed the page. Run "
        "`agent kijiji probe`."
    )


def test_kijiji_probe_waits_for_a_healthy_account(make_settings, kijiji):
    kijiji.login_outcome = "bad_password"
    check = _kijiji_check(make_settings)

    assert not check.ok and "rejected" in check.detail
    assert "probe" not in check.detail
    assert "GetSearchResultsPageByUrl" not in kijiji.graphql_calls  # no probe after a failure
    assert kijiji.password_posts == 1  # and no second sign in attempt
    assert PASSWORD not in check.detail and EMAIL not in check.detail

    kijiji.login_outcome = "ok"
    check = _kijiji_check(make_settings, kijiji_postal_code=None, kijiji_location_id=None)
    assert check == Check(
        "platform kijiji",
        False,
        "kijiji: MA_KIJIJI_POSTAL_CODE is not set, so ads cannot be posted",
    )


@pytest.fixture
def facebook(monkeypatch):
    """The facebook package's build_adapter over a FakeDriver; yields the driver."""
    driver = FakeDriver(FakeSite())
    package = __import__(FACEBOOK, fromlist=["build_adapter"])
    monkeypatch.setattr(
        package,
        "build_adapter",
        lambda settings, clock: FacebookAdapter(settings, clock, driver, None, find_timeout_s=0.1),
    )
    return driver


def test_facebook_session_and_read_only_selector_check(make_settings, facebook):
    check = _by_name(run_doctor(make_settings(enabled_platforms=["facebook"])))["platform facebook"]
    assert check == Check(
        "platform facebook", True, "signed in; create form and inbox locators match"
    )
    assert set(facebook.kinds()) <= {"goto", "wait_for"}  # nothing typed, clicked, or submitted
    assert facebook.closed


def test_facebook_signed_out(make_settings, facebook):
    facebook.site.logged_in = False
    check = _by_name(run_doctor(make_settings(enabled_platforms=["facebook"])))["platform facebook"]
    assert not check.ok and "agent facebook login" in check.detail


def test_facebook_stale_selector(make_settings, facebook):
    from marketplace_agent.platforms.facebook.selectors import SelectorRegistry

    facebook.break_target(SelectorRegistry.load(), "create.price")
    check = _by_name(run_doctor(make_settings(enabled_platforms=["facebook"])))["platform facebook"]
    assert not check.ok and check.detail.startswith("selector create.price")


@pytest.mark.parametrize(
    ("changes", "ok", "detail"),
    [
        (
            {"discord_token": "abc"},
            False,
            "token set, but discord_channel_id and discord_owner_id missing",
        ),
        ({"discord_token": "  "}, False, "discord_token is empty"),
        (
            {"discord_token": "abc", "discord_channel_id": 1, "discord_owner_id": 2},
            True,
            "token, channel id and owner id set",
        ),
    ],
)
def test_discord(make_settings, changes, ok, detail):
    assert _by_name(run_doctor(make_settings(**changes)))["discord"] == Check("discord", ok, detail)


def test_format_checks():
    text = format_checks([Check("config", True, "fine"), Check("data dir", False, "missing")])
    assert text.splitlines() == [
        "CHECK     RESULT  DETAIL",
        "config    ok      fine",
        "data dir  FAIL    missing",
    ]
