"""Stage 1 wired the way the CLI wires it: file SQLite, real photo ingest, real router and quota,
with ScriptedProvider standing in for Gemini."""
import io
import re
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from PIL import Image
from sqlmodel import Session, select
from typer.testing import CliRunner

from marketplace_agent import cli
from marketplace_agent.ai import factory, pattern_cache
from marketplace_agent.ai.base import (
    Capability,
    ProviderError,
    QuotaExhausted,
    QuotaState,
    RateLimited,
)
from marketplace_agent.ai.drafting import DRAFT_WIRE_SCHEMA, draft_item
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.ai.templates import availability_reply
from marketplace_agent.config import ProviderLimit, Settings
from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.db import make_engine, session_scope
from marketplace_agent.core.items import create_item, get_attributes, get_item, item_photos
from marketplace_agent.core.models import PatternEntry
from marketplace_agent.intake.photos import ingest_photos, photo_file

HINT = "IKEA desk $120"
GEMINI_MODEL = "gemini-flash-latest"
KEY_VARS = tuple(
    prefix + name
    for prefix in ("", "MA_")
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")
)
runner = CliRunner()


def _phone_jpeg(path):
    """Stored sideways with orientation 6, plus GPS and camera EXIF, like a phone photo."""
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x010F] = "Apple"
    exif[0x8825] = {1: "N", 2: (43.0, 39.0, 12.5), 3: "W", 4: (79.0, 23.0, 1.2)}
    Image.new("RGB", (400, 300), (150, 110, 70)).save(path, "JPEG", exif=exif.tobytes())
    return path


def _reply(**overrides):
    return {
        "title": "IKEA Micke Desk, White",
        "description": "White IKEA desk with one drawer. Light wear on the top.",
        "category": "Furniture",
        "condition": "good",
        "attributes": [{"key": "brand", "value": "IKEA"}, {"key": "color", "value": "white"}],
        "suggested_price_low_dollars": 95,
        "suggested_price_high_dollars": 119.99,
        "price_reasoning": "Used IKEA desks list for $90 to $130.",
        "uncertain": ["model", "dimensions"],
        **overrides,
    }


def _gemini(script):
    return ScriptedProvider(script, name="gemini", model=GEMINI_MODEL)


def _router(settings, engine, clock, *providers):
    """What factory.build_router assembles, with scripted providers in place of real ones."""
    quota = QuotaTracker(
        engine, clock, settings.provider_limits, settings.quota_constrained_fraction
    )
    return ProviderRouter(providers, quota)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """A file database, as in production, so quota.record commits on its own connection."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return Settings(
        _env_file=None,
        data_dir=data_dir,
        db_url=None,
        pickup_area="Bloor St W & Ossington Ave",
        pickup_address="123 Example St",
    )


@pytest.fixture
def item_id(settings, engine, clock, tmp_path):
    photo = _phone_jpeg(tmp_path / "desk.jpg")
    with session_scope(engine) as session:
        item = create_item(session, clock, HINT)
        ingest_photos(session, settings, item.id, [photo])
    clock.advance(minutes=5)
    return item.id


# drafting through router and quota


def test_draft_end_to_end_on_ingested_photo(settings, engine, clock, item_id):
    gemini = _gemini([_reply()])
    router = _router(settings, engine, clock, gemini)

    with session_scope(engine) as session:
        draft = draft_item(session, router, settings, clock, item_id)

    assert (draft.suggested_price_low_cents, draft.suggested_price_high_cents) == (9500, 11999)
    with Session(engine) as session:
        item = get_item(session, item_id)
        [photo] = item_photos(session, item_id)
    assert (item.title, item.category, item.condition) == (
        "IKEA Micke Desk, White",
        "Furniture",
        "good",
    )
    assert get_attributes(item) == {"brand": "IKEA", "color": "white"}
    assert item.updated_at == clock.now() > item.created_at

    [request] = gemini.requests
    assert (request.capability, request.purpose) == (Capability.VISION, "draft")
    assert request.json_schema == DRAFT_WIRE_SCHEMA
    assert HINT in request.prompt
    assert settings.pickup_address not in request.system + request.prompt
    sent = request.images[0].data
    assert sent == photo_file(settings, photo).read_bytes()
    assert b"Exif" not in sent and b"Apple" not in sent
    with Image.open(io.BytesIO(sent)) as image:
        assert image.size == (300, 400)  # rotated upright before the orientation tag was dropped
        assert not image.getexif()

    [usage] = router.quota.usage_today()
    assert (usage.provider, usage.model, usage.requests) == ("gemini", GEMINI_MODEL, 1)
    assert router.quota.remaining_today("gemini") == settings.provider_limits["gemini"].rpd - 1


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ("Sorry, I can't help with that.", "invalid JSON"),
        ({"title": "Desk"}, "invalid draft"),
    ],
)
def test_bad_reply_spends_quota_and_leaves_item_unchanged(
    settings, engine, clock, item_id, reply, error
):
    router = _router(settings, engine, clock, _gemini([reply]))

    with pytest.raises(ProviderError, match=error), session_scope(engine) as session:
        draft_item(session, router, settings, clock, item_id)

    with Session(engine) as session:
        assert get_item(session, item_id).title == ""
    assert router.quota.requests_today("gemini") == 1


def test_usage_survives_a_rolled_back_draft(settings, engine, clock, item_id):
    router = _router(settings, engine, clock, _gemini([_reply()]))

    with pytest.raises(RuntimeError), session_scope(engine) as session:
        draft_item(session, router, settings, clock, item_id)
        raise RuntimeError("crash before commit")

    with Session(engine) as session:
        assert get_item(session, item_id).title == ""
    assert router.quota.requests_today("gemini") == 1


def test_rate_limit_falls_back_then_cools_down(settings, engine, clock, item_id):
    limited = RateLimited("slow down")
    limited.retry_after = 30  # set by the real providers; RateLimited does not declare it
    backup = ScriptedProvider(
        [_reply(title="Desk via backup")], name="backup", capabilities=frozenset({Capability.VISION})
    )
    router = _router(settings, engine, clock, _gemini([limited]), backup)

    with session_scope(engine) as session:
        draft = draft_item(session, router, settings, clock, item_id)

    assert draft.title == "Desk via backup"
    assert [usage.provider for usage in router.quota.usage_today()] == ["backup"]
    assert not router.quota.can_request("gemini")
    clock.advance(seconds=31)
    assert router.quota.can_request("gemini")


def test_daily_quota_stops_drafting_until_pacific_midnight(settings, engine, clock, item_id):
    settings = settings.model_copy(
        update={"provider_limits": {"gemini": ProviderLimit(rpm=10, rpd=1)}}
    )
    gemini = _gemini(lambda request: _reply())
    router = _router(settings, engine, clock, gemini)

    with session_scope(engine) as session:
        draft_item(session, router, settings, clock, item_id)
    assert router.state(Capability.VISION) is QuotaState.EXHAUSTED
    with pytest.raises(QuotaExhausted), session_scope(engine) as session:
        draft_item(session, router, settings, clock, item_id)

    assert len(gemini.requests) == 1
    pacific = ZoneInfo("America/Los_Angeles")
    tomorrow = clock.now().astimezone(pacific).date() + timedelta(days=1)
    midnight = datetime.combine(tomorrow, time(), tzinfo=pacific)
    clock.set(midnight - timedelta(minutes=1))
    assert clock.now().astimezone(UTC).date() == tomorrow  # past UTC midnight already
    assert router.state(Capability.VISION) is QuotaState.EXHAUSTED
    clock.set(midnight)
    assert router.state(Capability.VISION) is QuotaState.HEALTHY


# pattern cache


def test_pattern_cache_seed_match_and_learn(settings, engine, clock):
    with session_scope(engine) as session:
        added = pattern_cache.seed(session, clock)
    with session_scope(engine) as session:
        assert added == len(session.exec(select(PatternEntry)).all()) > 0
        assert pattern_cache.seed(session, clock) == 0

    def first(session, text):
        return pattern_cache.match(session, text, first_contact=True)

    with session_scope(engine) as session:
        assert first(session, "Hi, is this still available?") == "availability"
        assert first(session, "is this stil availble") == "availability"
        assert first(session, "Is this still available? Would you take $80?") is None
        assert pattern_cache.learn(session, clock, "Is this item still for sale?", "availability")

    with Session(engine) as session:
        assert first(session, "is this item still for sale") == "availability"
        seeded = session.exec(
            select(PatternEntry).where(PatternEntry.normalized == "is this still available")
        ).one()
    assert seeded.hits == 2

    reply = availability_reply(committed=False, pickup_area=settings.pickup_area)
    assert settings.pickup_area in reply and settings.pickup_address not in reply


# CLI


@pytest.fixture
def cli_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no developer .env or toml
    monkeypatch.setenv("MA_DATA_DIR", str(tmp_path / "data"))
    for name in ("MA_DB_URL", *KEY_VARS):
        monkeypatch.delenv(name, raising=False)
    photo = _phone_jpeg(tmp_path / "desk.jpg")
    result = runner.invoke(cli.app, ["add-item", "--hint", HINT, "--photo", str(photo)])
    assert result.exit_code == 0, result.output
    return tmp_path


@pytest.fixture
def scripted_factory(monkeypatch):
    def build_router(settings, engine, clock):
        return _router(settings, engine, clock, _gemini([_reply()]))

    monkeypatch.setattr(factory, "build_router", build_router)


def _fails_cleanly(result) -> bool:
    return result.exit_code == 1 and isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("env", [{}, {"GROQ_API_KEY": "test-key"}], ids=["no-keys", "groq-only"])
def test_cli_draft_without_vision_prints_key_guidance(cli_dir, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    result = runner.invoke(cli.app, ["draft", "1"])

    assert _fails_cleanly(result), result.output
    assert "GEMINI_API_KEY" in result.output


def test_cli_draft_show_and_quota(cli_dir, scripted_factory):
    drafted = runner.invoke(cli.app, ["draft", "1"])

    assert drafted.exit_code == 0, drafted.output
    assert "Title:     IKEA Micke Desk, White" in drafted.output
    assert "Suggested: $95.00 to $119.99" in drafted.output
    assert "Uncertain: model, dimensions" in drafted.output
    shown = runner.invoke(cli.app, ["show", "1"]).output
    assert "Condition: good" in shown and "  brand: IKEA" in shown
    quota = runner.invoke(cli.app, ["quota"]).output
    assert re.search(rf"^gemini\s+{GEMINI_MODEL}\s+1\s+249\s", quota, re.MULTILINE), quota
    assert "State: healthy" in quota


@pytest.mark.parametrize(("item", "error"), [(2, "no photos"), (9, "not found")])
def test_cli_draft_bad_item_fails_cleanly(cli_dir, scripted_factory, item, error):
    engine = make_engine(f"sqlite:///{cli_dir / 'data' / 'agent.db'}")
    with session_scope(engine) as session:
        create_item(session, FakeClock(), "lamp, no photos yet")
    engine.dispose()

    result = runner.invoke(cli.app, ["draft", str(item)])

    assert _fails_cleanly(result), result.output
    assert error in result.output


def test_cli_draft_with_missing_photo_file_fails_cleanly(cli_dir, scripted_factory):
    for stored in (cli_dir / "data" / "photos" / "1").glob("*.jpg"):
        stored.unlink()

    result = runner.invoke(cli.app, ["draft", "1"])

    assert _fails_cleanly(result), result.output
    assert "Cannot read photo photos/1/" in result.output
    assert "No such file" in result.output
