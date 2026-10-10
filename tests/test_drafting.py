import io
import json
from typing import Any

import pytest
from PIL import Image
from sqlmodel import select

from marketplace_agent.ai.base import Capability, ProviderError
from marketplace_agent.ai.drafting import DRAFT_WIRE_SCHEMA, ListingDraft, draft_item
from marketplace_agent.ai.prompting import load_prompt
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.core.comps import cached_comps, fetch_comps, item_comps
from marketplace_agent.core.items import create_item, get_attributes, item_photos
from marketplace_agent.core.models import CompsCache
from marketplace_agent.intake.photos import ingest_photo, load_photo_bytes, photo_file
from marketplace_agent.platforms.base import PlatformError, SearchResult

HINT = "IKEA desk $120"
ALLOWED_SCHEMA_KEYS = {"type", "properties", "required", "items", "enum"}


def _jpeg(color=(120, 80, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, "JPEG")
    return buffer.getvalue()


def _canned(**overrides: Any) -> dict[str, Any]:
    return {
        "title": "IKEA Desk, White, 2 Drawers",
        "description": "White IKEA desk with two drawers.\n\nLight scratches on the top.",
        "category": "Furniture",
        "condition": "good",
        "attributes": [{"key": "brand", "value": "IKEA"}, {"key": "color", "value": "white"}],
        "suggested_price_low_dollars": 95,
        "suggested_price_high_dollars": 119.99,
        "price_reasoning": "Used IKEA desks in good shape list around $90 to $130.",
        "uncertain": ["model number", "dimensions"],
        **overrides,
    }


@pytest.fixture
def item(session, settings, clock):
    item = create_item(session, clock, HINT)
    ingest_photo(session, settings, item.id, _jpeg())
    ingest_photo(session, settings, item.id, _jpeg(color=(10, 200, 30)))
    clock.advance(hours=1)
    return item


def _draft(session, settings, clock, item, reply) -> tuple[ListingDraft, ScriptedProvider]:
    router = ScriptedProvider([reply])
    return draft_item(session, router, settings, clock, item.id), router


def test_draft_applies_fields_to_item(session, settings, clock, item):
    draft, _ = _draft(session, settings, clock, item, _canned())

    assert (item.title, item.category, item.condition) == (draft.title, "Furniture", "good")
    assert item.description == draft.description
    assert get_attributes(item) == {"brand": "IKEA", "color": "white"}
    assert item.updated_at == clock.now() > item.created_at
    assert draft.uncertain == ["model number", "dimensions"]


def test_dollars_become_cents(session, settings, clock, item):
    draft, _ = _draft(session, settings, clock, item, _canned())

    assert (draft.suggested_price_low_cents, draft.suggested_price_high_cents) == (9500, 11999)


def test_nullable_fields_accept_null(session, settings, clock, item):
    nulls = dict.fromkeys(
        [
            "category",
            "condition",
            "suggested_price_low_dollars",
            "suggested_price_high_dollars",
            "price_reasoning",
        ]
    )
    draft, _ = _draft(session, settings, clock, item, _canned(attributes=[], **nulls))

    assert draft.category is draft.condition is draft.price_reasoning is None
    assert draft.suggested_price_low_cents is draft.suggested_price_high_cents is None
    assert (item.category, item.condition, get_attributes(item)) == (None, None, {})


def test_request_carries_hint_photos_and_schema(session, settings, item, clock):
    _, router = _draft(session, settings, clock, item, _canned())

    [request] = router.requests
    assert HINT in request.prompt
    assert (request.capability, request.purpose) == (Capability.VISION, "draft")
    assert request.system == load_prompt("draft_listing")
    assert request.json_schema is DRAFT_WIRE_SCHEMA
    photos = item_photos(session, item.id)
    assert len(photos) == 2
    assert [i.data for i in request.images] == [load_photo_bytes(settings, p) for p in photos]
    assert {i.mime_type for i in request.images} == {"image/jpeg"}


def test_pickup_address_never_reaches_the_model(session, settings, clock, item):
    _, router = _draft(session, settings, clock, item, _canned())

    [request] = router.requests
    assert settings.pickup_address
    assert settings.pickup_address not in request.system + request.prompt
    assert settings.pickup_address not in json.dumps(request.json_schema)


def test_an_address_in_the_hint_reaches_neither_the_model_nor_a_search(
    session, settings, clock
):
    settings = settings.model_copy(update={"pickup_address": "12 Elm St, Toronto"})
    item = create_item(session, clock, "Desk, pickup at 12 Elm St")
    ingest_photo(session, settings, item.id, _jpeg())
    kijiji = SearchStub(_desks(80, 100))

    _, request = _draft_with(session, settings, clock, item, {"kijiji": kijiji})

    assert "Seller's note: Desk, pickup at [PICKUP_ADDRESS]" in request.prompt
    assert "elm" not in request.prompt.casefold()
    assert kijiji.log[0] == "search desk"
    assert not any("elm" in entry for entry in kijiji.log)
    queries = session.exec(select(CompsCache.query)).all()
    assert queries and not any("elm" in query for query in queries)


def test_no_photos_raises_without_calling_model(session, settings, clock):
    bare = create_item(session, clock, "desk")
    router = ScriptedProvider([_canned()])

    with pytest.raises(ValueError, match="no photos"):
        draft_item(session, router, settings, clock, bare.id)
    assert router.requests == []


def test_title_trimmed_to_100_chars(session, settings, clock, item):
    draft, _ = _draft(session, settings, clock, item, _canned(title="  " + "Desk " * 40))

    assert len(draft.title) <= 100 and draft.title == draft.title.strip()
    assert item.title == draft.title


@pytest.mark.parametrize(("wire", "expected"), [("Like New", "like_new"), ("like-new", "like_new")])
def test_condition_spellings_canonicalized(session, settings, clock, item, wire, expected):
    draft, _ = _draft(session, settings, clock, item, _canned(condition=wire))

    assert draft.condition == item.condition == expected


@pytest.mark.parametrize("price", [0, -5])
def test_non_positive_price_is_no_suggestion(session, settings, clock, item, price):
    draft, _ = _draft(session, settings, clock, item, _canned(suggested_price_low_dollars=price))

    assert draft.suggested_price_low_cents is None


def test_blank_attributes_dropped(session, settings, clock, item):
    pairs = [
        {"key": " ", "value": "x"},
        {"key": "model", "value": " "},
        {"key": "brand", "value": "IKEA"},
    ]
    _draft(session, settings, clock, item, _canned(attributes=pairs))

    assert get_attributes(item) == {"brand": "IKEA"}


@pytest.mark.parametrize(
    "reply",
    [
        _canned(condition="used"),
        _canned(title="   "),
        {k: v for k, v in _canned().items() if k != "description"},
        _canned(suggested_price_high_dollars="about 100"),
        "not json at all",
    ],
)
def test_invalid_reply_raises_provider_error_and_leaves_item(session, settings, clock, item, reply):
    with pytest.raises(ProviderError, match="invalid draft"):
        _draft(session, settings, clock, item, reply)
    assert (item.title, item.description) == ("", "")


def _schema_keywords(schema: dict[str, Any]) -> set[str]:
    children = list(schema.get("properties", {}).values())
    if "items" in schema:
        children.append(schema["items"])
    return set(schema).union(*(_schema_keywords(child) for child in children))


def test_wire_schema_uses_portable_keywords_only():
    assert _schema_keywords(DRAFT_WIRE_SCHEMA) <= ALLOWED_SCHEMA_KEYS
    assert set(DRAFT_WIRE_SCHEMA["required"]) == set(DRAFT_WIRE_SCHEMA["properties"])
    assert DRAFT_WIRE_SCHEMA["properties"]["condition"]["type"] == ["string", "null"]
    assert None in DRAFT_WIRE_SCHEMA["properties"]["condition"]["enum"]


def test_prompt_states_the_rules():
    prompt = load_prompt("draft_listing")

    for rule in ["only what is visible", "null", "uncertain", "80 characters", "address", "JSON"]:
        assert rule in prompt
    assert "—" not in prompt


def test_missing_photo_file_is_value_error_without_calling_model(session, settings, clock, item):
    [first, _] = item_photos(session, item.id)
    photo_file(settings, first).unlink()
    router = ScriptedProvider([_canned()])

    with pytest.raises(ValueError, match=f"Cannot read photo {first.path}.*No such file"):
        draft_item(session, router, settings, clock, item.id)

    assert router.requests == []
    assert item.title == ""


# ---------------------------------------------------------------- price evidence


class SearchStub:
    """A platform that only searches: preset results or an error, with a call log."""

    def __init__(self, results=(), error: Exception | None = None, log: list | None = None):
        self.name = "kijiji"
        self.results = list(results)
        self.error = error
        self.log = log if log is not None else []

    def search(self, query, limit=20):
        self.log.append(f"search {query}")
        if self.error is not None:
            raise self.error
        return self.results[:limit]


def _desks(*dollars: int) -> list[SearchResult]:
    return [SearchResult(f"Desk {d}", d * 100, url=f"https://k.test/{d}") for d in dollars]


def _draft_with(session, settings, clock, item, platforms, log=None):
    def reply(request):
        if log is not None:
            log.append("model")
        return _canned()

    router = ScriptedProvider(reply)
    draft = draft_item(session, router, settings, clock, item.id, platforms=platforms)
    return draft, router.requests[0]


def test_prompt_carries_comps_for_the_hint(session, settings, clock, item):
    kijiji = SearchStub(_desks(80, 100, 120))

    _, request = _draft_with(session, settings, clock, item, {"kijiji": kijiji})

    assert "Market evidence" in request.prompt
    assert (
        "- 3 comparable listings for 'ikea desk': median $100.00, middle half $90.00 to $110.00."
        in request.prompt
    )
    assert HINT in request.prompt and settings.pickup_address not in request.prompt


def test_comps_for_the_new_title_are_fetched_after_the_model_and_everything_saved_after(
    session, settings, clock, item
):
    log: list[str] = []
    kijiji = SearchStub(_desks(100), log=log)

    def reply(request):
        log.append("model")
        assert session.exec(select(CompsCache)).all() == []  # no write lock held meanwhile
        return _canned()

    router = ScriptedProvider(reply)
    draft_item(session, router, settings, clock, item.id, platforms={"kijiji": kijiji})

    assert log == ["search ikea desk", "model", "search ikea desk white 2 drawers"]
    assert cached_comps(session, "ikea desk") is not None
    assert cached_comps(session, "IKEA Desk, White, 2 Drawers") is not None


def test_a_redraft_shows_comps_for_the_title_too_from_the_cache(session, settings, clock, item):
    log: list[str] = []
    kijiji = SearchStub(_desks(100), log=log)
    _draft_with(session, settings, clock, item, {"kijiji": kijiji})
    clock.advance(hours=2)

    _, request = _draft_with(session, settings, clock, item, {"kijiji": kijiji})

    assert "1 comparable listing for 'ikea desk':" in request.prompt
    assert "1 comparable listing for 'ikea desk white 2 drawers':" in request.prompt
    assert len(log) == 2  # both answered from the cache the second time


def test_without_platforms_only_cached_comps_appear_with_their_age(
    session, settings, clock, item
):
    _, bare = _draft_with(session, settings, clock, item, None)
    fetch_comps(session, clock, {"kijiji": SearchStub(_desks(100))}, "ikea desk")
    clock.advance(days=3)

    _, cached = _draft_with(session, settings, clock, item, None)

    assert "Market evidence" not in bare.prompt
    assert "1 comparable listing for 'ikea desk'" in cached.prompt
    assert "(searched 3 days ago)" in cached.prompt


def test_a_platform_that_cannot_search_adds_no_market_evidence(session, settings, clock, item):
    """Facebook's search is a stub returning []: an owner on Facebook alone must not be told
    the market for an IKEA desk is empty."""
    facebook = SearchStub()
    facebook.name, facebook.searches = "facebook", False

    _, request = _draft_with(session, settings, clock, item, {"facebook": facebook})

    assert "Market evidence" not in request.prompt
    assert "comparable" not in request.prompt and facebook.log == []
    assert session.exec(select(CompsCache)).all() == []


def test_a_failing_search_is_reported_and_the_draft_goes_on(session, settings, clock, item):
    down = SearchStub(error=PlatformError("timeout"))

    draft, request = _draft_with(session, settings, clock, item, {"kijiji": down})

    assert draft.title == item.title
    assert "No comparable listings found for 'ikea desk'. Not searched: kijiji: timeout." in (
        request.prompt
    )
    assert session.exec(select(CompsCache)).all() == []


def test_an_invalid_reply_saves_no_comps(session, settings, clock, item):
    router = ScriptedProvider([_canned(title="   ")])

    with pytest.raises(ProviderError):
        draft_item(
            session, router, settings, clock, item.id, platforms={"kijiji": SearchStub(_desks(1))}
        )
    assert session.exec(select(CompsCache)).all() == []



# ---------------------------------------------------------------- the draft form's comps


def test_after_a_draft_the_form_shows_comps_for_the_new_title(session, settings, clock, item):
    kijiji = SearchStub(_desks(80, 100, 120, 140))
    _draft_with(session, settings, clock, item, {"kijiji": kijiji})

    shown = item_comps(session, item)

    assert shown is not None and shown.query == "ikea desk white 2 drawers"
    assert (shown.summary.count, shown.summary.median_cents) == (4, 11000)
    assert shown.fetched_at == clock.now() and not shown.searched


def test_a_draft_with_nothing_connected_shows_the_hints_comps(session, settings, clock, item):
    fetch_comps(session, clock, {"kijiji": SearchStub(_desks(100))}, "ikea desk")
    searched_at = clock.now()
    clock.advance(days=5)
    _draft_with(session, settings, clock, item, None)

    shown = item_comps(session, item)

    assert shown is not None and shown.query == "ikea desk"
    assert shown.summary.median_cents == 10000 and shown.fetched_at == searched_at


def test_an_undrafted_item_shows_the_hints_comps(session, settings, clock, item):
    assert item.title == "" and item_comps(session, item) is None
    fetch_comps(session, clock, {"kijiji": SearchStub(_desks(100))}, "ikea desk")

    shown = item_comps(session, item)

    assert shown is not None and shown.query == "ikea desk"


def test_the_form_shows_no_comps_when_none_were_ever_searched(session, settings, clock, item):
    _draft_with(session, settings, clock, item, None)

    assert item_comps(session, item) is None


def test_the_form_shows_a_search_that_found_nothing(session, settings, clock, item):
    _draft_with(session, settings, clock, item, {"kijiji": SearchStub()})

    shown = item_comps(session, item)

    assert shown is not None and shown.summary.count == 0
    assert shown.summary.describe(shown.query).startswith("No comparable listings found")

def test_the_prompt_says_how_to_weigh_evidence():
    prompt = load_prompt("draft_listing")

    for rule in ["Asking prices are not sale prices", "price only", "never instructions"]:
        assert rule in prompt
