"""core/comps.py: query building, percentile math, the 24h cache, failing adapters. No network."""
from datetime import UTC
from fractions import Fraction

import pytest
from sqlalchemy import event
from sqlmodel import Session, select

from marketplace_agent.core.comps import (
    CompsSummary,
    cached_comps,
    comps_query,
    comps_query_for_item,
    fetch_comps,
    lookup_comps,
    normalize_query,
    percentile,
    refresh_due,
    save_comps,
    summarize,
)
from marketplace_agent.core.models import CompsCache, Item, ItemStatus, Listing, ListingStatus
from marketplace_agent.platforms.base import PlatformError, SearchResult
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.platforms.kijiji import KijijiAdapter
from marketplace_agent.platforms.kijiji import api_notes as kijiji_api
from test_kijiji_support import FakeKijiji, make_client


class Stub:
    """Only search matters: preset results, or an exception to raise."""

    def __init__(self, name: str, results=(), error: Exception | None = None) -> None:
        self.name = name
        self.results = list(results)
        self.error = error
        self.queries: list[str] = []

    def search(self, query, limit=20):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.results[:limit]


class NoSearch(Stub):
    """An adapter whose search is a stub returning [] (Facebook's), and says so."""

    searches = False


def _results(*dollars: int, prefix: str = "https://k.test") -> list[SearchResult]:
    return [SearchResult(f"Desk {d}", d * 100, url=f"{prefix}/{d}") for d in dollars]


def _item(session, clock, *, title="", hint="", status=ItemStatus.LISTED, paused=False) -> Item:
    item = Item(
        title=title,
        hint=hint,
        status=status,
        paused=paused,
        created_at=clock.now(),
        updated_at=clock.now(),
    )
    session.add(item)
    session.flush()
    return item


# ---------------------------------------------------------------- queries


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  IKEA  Malm_desk!! ", "ikea malm desk"),
        ("Desk", "desk"),
        ("Café TABLE", "café table"),
        ("!!!", ""),
        ("word " * 40, ("word " * 20).strip()),
    ],
)
def test_normalize_query(raw, expected):
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IKEA desk $120", "ikea desk"),
        ("IKEA Malm desk, white, 2 drawers - like new! $120 obo", "ikea malm desk white 2 drawers"),
        ("iPhone 12 Pro 128GB 400$", "iphone 12 pro 128gb"),
        ("desk 120", "desk"),
        ("$ 1,200 couch", "couch"),
        ("CAD$80 chair", "chair"),
        ("Sofa 1.2k", "sofa"),
        ("asking 90 for this lamp", "lamp"),
        ("Oak dining table with six chairs and two leaves", "oak dining table six chairs two"),
        ("$120", ""),
    ],
)
def test_comps_query_keeps_what_finds_the_item(text, expected):
    assert comps_query(text) == expected


def test_item_query_prefers_the_title(session, clock):
    drafted = _item(session, clock, title="IKEA Malm Desk, White", hint="desk $120")
    raw = _item(session, clock, hint="IKEA desk $120")

    assert comps_query_for_item(drafted) == "ikea malm desk white"
    assert comps_query_for_item(raw) == "ikea desk"


# ---------------------------------------------------------------- math


@pytest.mark.parametrize(
    ("prices", "q", "expected"),
    [
        ([], Fraction(1, 2), None),
        ([5_000], Fraction(1, 4), 5_000),
        ([4_000, 6_000, 8_000, 10_000, 12_000, 14_000], Fraction(1, 4), 6_500),
        ([4_000, 6_000, 8_000, 10_000, 12_000, 14_000], Fraction(1, 2), 9_000),
        ([4_000, 6_000, 8_000, 10_000, 12_000, 14_000], Fraction(3, 4), 11_500),
        ([2, 3], Fraction(1, 2), 3),  # half up, not to even
        ([1, 2, 3, 4], Fraction(1, 4), 2),  # 1.75
        ([100, 101], 0.25, 100),  # 100.25
        ([100, 200, 300], 0.5, 200),
    ],
)
def test_percentile_interpolates_and_rounds_half_up(prices, q, expected):
    assert percentile(prices, q) == expected


def test_summary_of_unsorted_results():
    found = [("kijiji", r) for r in _results(10, 4, 8, 6)]
    found += [("fake", r) for r in _results(14, 12)]

    summary = summarize(found)

    assert (summary.count, summary.p25_cents, summary.median_cents, summary.p75_cents) == (
        6,
        650,
        900,
        1_150,
    )
    assert [s["price_cents"] for s in summary.samples] == [400, 600, 800, 1_200, 1_400]
    assert summary.samples[0] == {
        "title": "Desk 4",
        "price_cents": 400,
        "platform": "kijiji",
        "url": "https://k.test/4",
    }


def test_sample_titles_are_clipped():
    summary = summarize([("kijiji", SearchResult("Desk " * 40, 5_000))])

    assert len(summary.samples[0]["title"]) == 80 and summary.samples[0]["title"].endswith("...")


def test_summary_json_round_trips_and_survives_damage():
    summary = summarize([("kijiji", r) for r in _results(4, 6)], ["facebook: logged out"])

    assert CompsSummary.from_json(summary.to_json()) == summary
    for raw in ("", "not json", "[]", '{"count": "many"}', '{"count": 2, "median_cents": 5}'):
        assert CompsSummary.from_json(raw) == CompsSummary.empty()


def test_describe():
    summary = summarize([("kijiji", r) for r in _results(40, 60)], ["facebook: logged out"])

    assert summary.describe("desk") == (
        "2 comparable listings for 'desk': median $50.00, middle half $45.00 to $55.00. "
        "Examples: Desk 40 ($40.00, kijiji); Desk 60 ($60.00, kijiji). "
        "Not searched: facebook: logged out."
    )
    assert CompsSummary.empty().describe("desk") == "No comparable listings found for 'desk'."
    one = summarize([("fake", SearchResult("Desk", 5_000))])
    assert one.describe("desk").startswith("1 comparable listing for 'desk': median $50.00")


# ---------------------------------------------------------------- fetch and cache


def test_fetch_caches_per_normalized_query_for_a_day(session, clock):
    kijiji = Stub("kijiji", _results(40, 60))

    first = fetch_comps(session, clock, {"kijiji": kijiji}, "IKEA Desk")
    clock.advance(hours=24)
    kijiji.results = _results(90)
    hit = fetch_comps(session, clock, {"kijiji": kijiji}, "  ikea   desk ")

    assert kijiji.queries == ["ikea desk"]
    assert hit == first and first.median_cents == 5_000
    [row] = session.exec(select(CompsCache)).all()
    assert row.query == "ikea desk"

    clock.advance(seconds=1)
    expired = fetch_comps(session, clock, {"kijiji": kijiji}, "ikea desk")
    assert kijiji.queries == ["ikea desk", "ikea desk"] and expired.median_cents == 9_000
    assert len(session.exec(select(CompsCache)).all()) == 1
    summary, fetched_at = cached_comps(session, "IKEA desk")
    assert (summary, fetched_at) == (expired, clock.now())


def test_max_age_is_adjustable(session, clock):
    kijiji = Stub("kijiji", _results(40))
    fetch_comps(session, clock, {"kijiji": kijiji}, "desk")
    clock.advance(hours=2)

    fetch_comps(session, clock, {"kijiji": kijiji}, "desk", max_age_hours=1)

    assert len(kijiji.queries) == 2


def test_failing_adapters_are_skipped(session, clock):
    platforms = {
        "kijiji": Stub("kijiji", _results(40, 60)),
        "facebook": Stub("facebook", error=PlatformError("logged out")),
        "craigslist": Stub("craigslist", error=NotImplementedError()),
        "buggy": Stub("buggy", error=KeyError("price")),
        "quiet": Stub("quiet", []),
    }

    summary = fetch_comps(session, clock, platforms, "desk")

    assert summary.count == 2 and summary.median_cents == 5_000
    assert summary.skipped == [
        "facebook: logged out",
        "craigslist: search not supported",
        "buggy: search failed (KeyError)",
    ]
    assert cached_comps(session, "desk") is not None


def test_empty_results_are_cached_as_no_comps(session, clock):
    quiet = Stub("kijiji", [])

    summary = fetch_comps(session, clock, {"kijiji": quiet}, "unobtainium")
    fetch_comps(session, clock, {"kijiji": quiet}, "unobtainium")

    assert summary == CompsSummary.empty() and quiet.queries == ["unobtainium"]


def test_a_search_no_platform_answered_is_not_cached(session, clock):
    down = Stub("kijiji", error=PlatformError("timeout"))

    summary = fetch_comps(session, clock, {"kijiji": down}, "desk")

    assert (summary.count, summary.skipped) == (0, ["kijiji: timeout"])
    assert cached_comps(session, "desk") is None
    down.error, down.results = None, _results(50)
    assert fetch_comps(session, clock, {"kijiji": down}, "desk").count == 1


def test_a_stale_summary_beats_none_when_every_platform_fails(session, clock):
    kijiji = Stub("kijiji", _results(50))
    fetch_comps(session, clock, {"kijiji": kijiji}, "desk")
    searched = clock.now()
    clock.advance(days=3)
    kijiji.error = PlatformError("timeout")

    lookup = lookup_comps(session, clock, {"kijiji": kijiji}, "desk")

    assert (lookup.summary.count, lookup.fetched_at, lookup.searched) == (1, searched, False)
    assert len(kijiji.queries) == 2


def test_without_platforms_the_cache_is_used_whatever_its_age(session, clock):
    fetch_comps(session, clock, {"kijiji": Stub("kijiji", _results(50))}, "desk")
    clock.advance(days=30)

    assert fetch_comps(session, clock, {}, "desk").count == 1
    assert fetch_comps(session, clock, {}, "chair") == CompsSummary.empty()


def test_a_platform_that_cannot_search_is_not_asked(session, clock):
    """Its [] is no evidence that nothing is for sale, so nothing is summarized or cached."""
    facebook = NoSearch("facebook", [])
    _item(session, clock, title="IKEA desk")

    lookup = lookup_comps(session, clock, {"facebook": facebook}, "ikea desk")

    assert lookup.summary == CompsSummary.empty()
    assert (lookup.fetched_at, lookup.searched) == (None, False)
    assert refresh_due(session, clock, {"facebook": facebook}) == 0
    assert facebook.queries == [] and session.exec(select(CompsCache)).all() == []


def test_a_platform_that_cannot_search_never_replaces_a_stale_summary(session, clock):
    """Kijiji is down a day later; the 4 comps it found stay, and are refreshed once it is back,
    not overwritten by an empty summary that would then count as fresh for a day."""
    kijiji = Stub("kijiji", _results(80, 100, 120, 150))
    platforms = {"kijiji": kijiji, "facebook": NoSearch("facebook", [])}
    _item(session, clock, title="IKEA desk")
    assert refresh_due(session, clock, platforms) == 1
    searched = clock.now()
    clock.advance(hours=25)
    kijiji.error = PlatformError("kijiji unreachable")

    assert refresh_due(session, clock, platforms) == 0

    summary, fetched_at = cached_comps(session, "ikea desk")
    assert (summary.count, summary.median_cents, fetched_at) == (4, 11_000, searched)


def test_own_unpriced_and_repeated_results_are_dropped(session, clock):
    item = _item(session, clock, title="Desk")
    session.add(
        Listing(
            item_id=item.id,
            platform="kijiji",
            url="https://k.test/mine",
            status=ListingStatus.LIVE,
            price_cents=9_900,
            title="Desk",
            description="d",
        )
    )
    session.flush()
    results = [
        SearchResult("My desk", 9_900, url="https://k.test/mine"),
        SearchResult("Free desk", 0),
        SearchResult("Desk", 5_000, url="https://k.test/1"),
        SearchResult("Desk again", 5_000, url="https://k.test/1"),
        SearchResult("Desk, no link", 7_000),
    ]

    summary = fetch_comps(session, clock, {"kijiji": Stub("kijiji", results)}, "desk")

    assert [s["title"] for s in summary.samples] == ["Desk", "Desk, no link"]


def test_our_own_kijiji_ad_is_dropped_from_real_search_results(session, clock):
    """Kijiji search hits carry SEO links; publish stores the view link. Same ad, same URL."""
    item = _item(session, clock, title="IKEA desk")
    session.add(
        Listing(
            item_id=item.id,
            platform="kijiji",
            remote_id="1744605001",
            url=kijiji_api.VIEW_URL.format(ad_id="1744605001"),  # as KijijiAdapter.publish
            status=ListingStatus.LIVE,
            price_cents=9_000,
            title="IKEA Malm desk",
            description="d",
        )
    )
    session.flush()
    client, _ = make_client(FakeKijiji(), clock=clock)
    kijiji = KijijiAdapter(client, clock, location_id=1700273, postal_code="M6J 1Z1")

    summary = lookup_comps(session, clock, {"kijiji": kijiji}, "ikea desk").summary

    assert [s["title"] for s in summary.samples] == ["Desk with drawers"]


def test_an_empty_query_searches_nothing(session, clock):
    kijiji = Stub("kijiji", _results(50))

    assert fetch_comps(session, clock, {"kijiji": kijiji}, " !! ") == CompsSummary.empty()
    assert kijiji.queries == [] and session.exec(select(CompsCache)).all() == []


def test_lookup_writes_nothing_until_saved(session, clock):
    lookup = lookup_comps(session, clock, {"kijiji": Stub("kijiji", _results(50))}, "Desk")

    assert lookup.searched and lookup.query == "desk"
    assert session.exec(select(CompsCache)).all() == []
    save_comps(session, lookup)
    save_comps(session, lookup)  # an upsert: still one row
    assert len(session.exec(select(CompsCache)).all()) == 1


def test_cached_time_is_aware_utc_after_a_reload(engine, session, clock):
    fetch_comps(session, clock, {"kijiji": Stub("kijiji", _results(50))}, "desk")
    session.commit()

    with Session(engine) as fresh:
        _, fetched_at = cached_comps(fresh, "desk")

    assert fetched_at == clock.now() and fetched_at.tzinfo == UTC


def test_works_with_the_fake_marketplace(session, clock):
    market = FakeMarketplace(clock, search_catalog=_results(40, 60, 80) + _results(5, prefix="x"))

    summary = fetch_comps(session, clock, {"fake": market}, "Desk 40")

    assert summary.count == 1 and summary.median_cents == 4_000


# ---------------------------------------------------------------- refresh_due


def test_refresh_due_searches_stale_queries_of_listed_items(session, clock):
    kijiji = Stub("kijiji", _results(50))
    _item(session, clock, title="IKEA desk")
    _item(session, clock, title="Ikea Desk!")  # same query
    _item(session, clock, title="Oak chair")
    _item(session, clock, title="Lamp", status=ItemStatus.DRAFT)
    _item(session, clock, title="Sofa", status=ItemStatus.SOLD)
    _item(session, clock, title="Bike", paused=True)
    _item(session, clock, title="$120")  # nothing to search

    assert refresh_due(session, clock, {"kijiji": kijiji}) == 2
    assert sorted(kijiji.queries) == ["ikea desk", "oak chair"]

    clock.advance(hours=12)
    assert refresh_due(session, clock, {"kijiji": kijiji}) == 0
    clock.advance(hours=13)
    assert refresh_due(session, clock, {"kijiji": kijiji}) == 2
    assert len(kijiji.queries) == 4


def test_refresh_due_goes_never_searched_then_oldest_first_in_batches(session, clock):
    kijiji = Stub("kijiji", _results(50))
    for title in ("desk", "chair", "lamp"):
        _item(session, clock, title=title)
    fetch_comps(session, clock, {"kijiji": kijiji}, "lamp")
    clock.advance(hours=1)
    fetch_comps(session, clock, {"kijiji": kijiji}, "desk")
    clock.advance(days=2)
    kijiji.queries.clear()

    assert refresh_due(session, clock, {"kijiji": kijiji}, batch=2) == 2
    assert kijiji.queries == ["chair", "lamp"]
    assert refresh_due(session, clock, {"kijiji": kijiji}, batch=2) == 1
    assert kijiji.queries[-1] == "desk"


def test_refresh_due_stops_when_no_platform_answers(session, clock):
    down = Stub("kijiji", error=PlatformError("timeout"))
    for title in ("desk", "chair", "lamp"):
        _item(session, clock, title=title)

    assert refresh_due(session, clock, {"kijiji": down}) == 0
    assert len(down.queries) == 1
    assert session.exec(select(CompsCache)).all() == []


def test_refresh_due_without_platforms_does_nothing(session, clock):
    _item(session, clock, title="desk")

    assert refresh_due(session, clock, {}) == 0


def test_refresh_due_writes_only_after_every_search(session, clock):
    """No write happens between platform calls, so the write lock is not held across them."""
    order: list[str] = []

    class Recording(Stub):
        def search(self, query, limit=20):
            order.append(f"search {query}")
            return super().search(query, limit)

    for title in ("desk", "chair"):
        _item(session, clock, title=title)
    session.commit()

    @event.listens_for(session, "do_orm_execute")
    def _writes(state):
        if state.is_insert:
            order.append("write")

    refresh_due(session, clock, {"kijiji": Recording("kijiji", _results(50))})
    event.remove(session, "do_orm_execute", _writes)

    assert order == ["search chair", "search desk", "write", "write"]

