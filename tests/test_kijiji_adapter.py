from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.runtime import build_platforms
from marketplace_agent.platforms.base import (
    ListingGone,
    ListingPayload,
    PlatformAuthError,
    PlatformError,
    PlatformRef,
)
from marketplace_agent.platforms.kijiji import KijijiAdapter, build_adapter
from marketplace_agent.platforms.kijiji.client import KijijiConversation, KijijiSearchHit
from marketplace_agent.platforms.kijiji.mapping import (
    category_id,
    dollars,
    search_url,
    to_ad,
    to_search_result,
    to_thread,
)
from test_kijiji_support import EMAIL, NEW_AD_ID, PASSWORD, FakeKijiji, make_client

LIVE_REF = PlatformRef("kijiji", "1744700001")


def _payload(tmp_path: Path, **changes: object) -> ListingPayload:
    photo = tmp_path / "desk.jpg"
    photo.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    payload = ListingPayload(
        title="IKEA Malm desk, white",
        description="White Malm desk, light scratches on the top.",
        price_cents=12000,
        photo_paths=[photo],
        location="Bloor St W & Ossington Ave",
        category="Furniture",
        condition="good",
    )
    return replace(payload, **changes)


def _adapter(
    fake: FakeKijiji, *, postal_code: str | None = "M6J 1Z1", location_id: int | None = 1700273
) -> tuple[KijijiAdapter, FakeClock]:
    clock = FakeClock()
    client, _ = make_client(fake, clock=clock)
    adapter = KijijiAdapter(client, clock, location_id=location_id, postal_code=postal_code)
    return adapter, clock


# Factory


def test_build_adapter_needs_credentials(settings):
    with pytest.raises(PlatformAuthError, match="MA_KIJIJI_EMAIL and MA_KIJIJI_PASSWORD"):
        build_adapter(settings, FakeClock())


def test_build_adapter_with_credentials_makes_no_request(settings):
    settings = settings.model_copy(
        update={
            "kijiji_email": EMAIL,
            "kijiji_password": SecretStr(PASSWORD),
            "enabled_platforms": ["kijiji"],
        }
    )

    adapters = build_platforms(settings, FakeClock())

    assert isinstance(adapters["kijiji"], KijijiAdapter)
    assert adapters["kijiji"].name == "kijiji"


# Listings


def test_publish_posts_and_returns_a_reference(tmp_path):
    fake = FakeKijiji()
    adapter, _ = _adapter(fake)

    ref = adapter.publish(_payload(tmp_path), assist=True)  # assist does not apply to Kijiji

    assert ref == PlatformRef(
        "kijiji", NEW_AD_ID, f"https://www.kijiji.ca/v-view-details.html?adId={NEW_AD_ID}"
    )
    (form,) = fake.submitted
    assert form["postAdForm.priceAmount"] == "120"
    assert form["postAdForm.attributeMap[condition_s]"] == "usedgood"


def test_publish_without_a_location_posts_nothing(tmp_path):
    fake = FakeKijiji()
    adapter, _ = _adapter(fake, postal_code=None, location_id=None)

    with pytest.raises(PlatformError, match="MA_KIJIJI_POSTAL_CODE"):
        adapter.publish(_payload(tmp_path))

    assert fake.requests == []


def test_update_and_delist(tmp_path):
    fake = FakeKijiji()
    adapter, _ = _adapter(fake)

    adapter.update(LIVE_REF, _payload(tmp_path, price_cents=9550))
    adapter.delist(LIVE_REF)

    assert fake.submitted[0]["postAdForm.priceAmount"] == "95.50"
    assert fake.deleted == ["1744700001"]


@pytest.mark.parametrize("method", ["update", "delist", "stats"])
def test_missing_ad_is_listing_gone(tmp_path, method):
    adapter, _ = _adapter(FakeKijiji())
    gone = PlatformRef("kijiji", "1744700099")
    args = (gone, _payload(tmp_path)) if method == "update" else (gone,)

    with pytest.raises(ListingGone):
        getattr(adapter, method)(*args)


# Threads


def test_fetch_threads_maps_seller_conversations():
    adapter, _ = _adapter(FakeKijiji())

    threads = adapter.fetch_threads(None)

    assert [t.remote_id for t in threads] == ["c-9001", "c-9002"]
    first = threads[0]
    assert first.listing_remote_id == "1744700001"
    assert (first.buyer_handle, first.buyer_name) == ("2000111", "Priya")
    assert [m.from_buyer for m in first.messages] == [True, False, True]
    assert all(m.sent_at.tzinfo is not None for m in first.messages)


def test_fetch_threads_skips_conversations_quiet_since():
    fake = FakeKijiji()
    adapter, _ = _adapter(fake)

    threads = adapter.fetch_threads(datetime(2026, 10, 1, tzinfo=UTC))

    assert [t.remote_id for t in threads] == ["c-9001"]
    assert fake.graphql_calls.count("GetConversation") == 1


def test_fetch_threads_pages_through_conversations():
    base = FakeKijiji().conversations[0]
    fake = FakeKijiji(conversations=[{**base, "id": f"c-{n}"} for n in range(30)])
    adapter, _ = _adapter(fake)

    threads = adapter.fetch_threads(None)

    assert len(threads) == 30
    assert fake.graphql_calls.count("GetConversations") == 2


def test_send_message_returns_the_remote_id():
    fake = FakeKijiji()
    adapter, _ = _adapter(fake)

    assert adapter.send_message("c-9001", "Yes, it is.") == "m-4"
    assert fake.sent == [("c-9001", "Yes, it is.")]


# Stats


def test_stats_reads_views_and_counts_conversations():
    adapter, _ = _adapter(FakeKijiji())

    stats = adapter.stats(LIVE_REF)

    assert (stats.views, stats.saves, stats.message_count) == (37, 0, 1)


def test_stats_reuse_the_conversation_list_for_a_few_minutes():
    fake = FakeKijiji()
    adapter, clock = _adapter(fake)
    adapter.fetch_threads(None)

    adapter.stats(LIVE_REF)
    clock.advance(minutes=6)
    adapter.stats(LIVE_REF)

    assert fake.graphql_calls.count("GetConversations") == 2


def test_stats_of_an_expired_ad_is_listing_gone():
    fake = FakeKijiji()
    fake.listings["1744700001"]["status"] = "EXPIRED"
    adapter, _ = _adapter(fake)

    with pytest.raises(ListingGone):
        adapter.stats(LIVE_REF)


# Search and health


def test_search_returns_fixed_price_comps_near_the_location():
    fake = FakeKijiji()
    adapter, _ = _adapter(fake)

    results = adapter.search("IKEA desk!", limit=10)

    assert [(r.title, r.price_cents) for r in results] == [
        ("IKEA Malm desk", 9000),
        ("Desk with drawers", 4550),
    ]
    request = next(r for r in fake.requests if b"GetSearchResultsPageByUrl" in r.content)
    assert b"/b-buy-sell/canada/ikea-desk/k0c10l1700273" in request.content


def test_search_with_nothing_to_search_makes_no_request():
    fake = FakeKijiji()
    adapter, _ = _adapter(fake)

    assert adapter.search("!!!") == []
    assert adapter.search("desk", limit=0) == []
    assert fake.requests == []


def test_healthcheck_reports_problems_without_raising():
    fake = FakeKijiji(login_outcome="bad_password")
    adapter, _ = _adapter(fake, postal_code=None, location_id=None)

    problems = adapter.healthcheck()

    assert len(problems) == 2
    assert "MA_KIJIJI_POSTAL_CODE" in problems[0]
    assert "rejected" in problems[1] and PASSWORD not in problems[1]


def test_healthcheck_when_healthy():
    adapter, _ = _adapter(FakeKijiji())

    assert adapter.healthcheck() == []


def test_server_errors_surface_as_platform_errors():
    fake = FakeKijiji()
    adapter, _ = _adapter(fake)
    fake.queue("GetConversations", *[httpx.Response(500)] * 3)

    with pytest.raises(PlatformError, match="HTTP 500"):
        adapter.fetch_threads(None)


# Mapping


@pytest.mark.parametrize(
    ("category", "title", "expected"),
    [
        ("Furniture", "IKEA Malm desk", 239),
        ("Furniture", "Oak TV stand", 242),
        ("Furniture", "Antique hutch", 236),
        ("Electronics", "Samsung 55 inch TV", 15093002),
        ("Electronics", "Sony WH-1000XM4 headphones", 770),
        (None, "Trek mountain bike", 651),
        ("Sporting Goods", "Adjustable dumbbells", 655),
        (None, "Box of misc cables", 26),
    ],
)
def test_category_keywords(tmp_path, category, title, expected):
    assert category_id(_payload(tmp_path, category=category, title=title)) == expected


def test_category_override_wins(tmp_path):
    payload = _payload(tmp_path, attributes={"kijiji_category_id": " 613 "})
    ignored = _payload(tmp_path, attributes={"kijiji_category_id": "guitars"})

    assert category_id(payload) == 613
    assert category_id(ignored) == 239


def test_to_ad_maps_condition_price_title_and_location(tmp_path):
    long_title = "IKEA Malm desk " + "with matching drawer unit and chair " * 3
    ad = to_ad(
        _payload(tmp_path, title=long_title, condition="poor", price_cents=5),
        location_id=1700273,
        postal_code="M6J 1Z1",
    )

    assert len(ad.title) <= 64 and not ad.title.endswith(" ")
    assert long_title.startswith(ad.title)
    assert (ad.condition, ad.price) == ("usedfair", "0.05")
    assert (ad.location_id, ad.postal_code) == (1700273, "M6J 1Z1")
    bare = to_ad(_payload(tmp_path, condition=None), location_id=None, postal_code=None)
    unbroken = to_ad(_payload(tmp_path, title="x" * 80), location_id=None, postal_code=None)
    assert bare.condition is None
    assert len(unbroken.title) == 64


@pytest.mark.parametrize(
    ("cents", "text"), [(12000, "120"), (12050, "120.50"), (0, "0"), (7, "0.07")]
)
def test_dollars(cents, text):
    assert dollars(cents) == text


def test_search_url_and_results():
    expected = "https://www.kijiji.ca/b-buy-sell/canada/ikea-malm-desk/k0c10l0"
    assert search_url("IKEA  Malm desk", None) == expected
    assert search_url("...", 1700273) is None
    free = KijijiSearchHit("1", "Free desk", None, None, None)
    assert to_search_result(free) is None


def test_thread_without_a_counterparty_id_gets_a_stable_handle():
    conversation = KijijiConversation("c-1", None, None, "", datetime(2026, 10, 1, tzinfo=UTC))

    thread = to_thread(conversation, [])

    assert thread.buyer_handle == "conversation-c-1"
    assert thread.listing_remote_id is None


def test_fetch_window_uses_aware_datetimes():
    adapter, clock = _adapter(FakeKijiji())

    since = clock.now() - timedelta(days=30)

    assert len(adapter.fetch_threads(since)) == 2
