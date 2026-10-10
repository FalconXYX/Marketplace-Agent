"""escalate, remember, get_stats, query_comps, and the registry."""
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import event
from sqlmodel import select

from marketplace_agent.core import wakes
from marketplace_agent.core.models import (
    Buyer,
    Escalation,
    Item,
    ItemStatus,
    Listing,
    ListingStatsSnapshot,
    ListingStatus,
    Memory,
    MemoryScope,
    Thread,
)
from marketplace_agent.guard.invariants import ADDRESS_PLACEHOLDER
from marketplace_agent.guard.validate import validate
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import PlatformError, SearchResult
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext
from marketplace_agent.tools.impl.info import GetStatsTool, NoArgs, QueryCompsArgs, QueryCompsTool
from marketplace_agent.tools.impl.meta import (
    EscalateArgs,
    EscalateTool,
    RememberArgs,
    RememberTool,
    ScheduleWakeArgs,
    ScheduleWakeTool,
)
from marketplace_agent.tools.registry import ToolRegistry, default_registry

ESCALATE, REMEMBER, STATS, COMPS = EscalateTool(), RememberTool(), GetStatsTool(), QueryCompsTool()


class SearchStub:
    """Only search matters here; it returns preset results or raises."""

    def __init__(self, name: str, results=(), error: Exception | None = None) -> None:
        self.name = name
        self.results = list(results)
        self.error = error
        self.queries: list[tuple[str, int]] = []

    def search(self, query, limit=20):
        self.queries.append((query, limit))
        if self.error is not None:
            raise self.error
        return self.results[:limit]


@pytest.fixture
def item(session, clock) -> Item:
    return _item(session, clock)


@pytest.fixture
def ctx(session, clock, settings, notifier, item) -> ToolContext:
    return _ctx(session, clock, settings, notifier, item, {})


def _ctx(session, clock, settings, notifier, item, platforms, actor="agent") -> ToolContext:
    return ToolContext(
        session=session,
        clock=clock,
        settings=settings,
        platforms=platforms,
        notifier=notifier,
        item_id=item.id,
        actor=actor,
    )


def _item(session, clock, status: ItemStatus = ItemStatus.LISTED) -> Item:
    item = Item(title="IKEA desk", status=status, created_at=clock.now(), updated_at=clock.now())
    session.add(item)
    session.commit()
    return item


def _thread(session, clock, item: Item, handle: str) -> Thread:
    buyer = Buyer(platform="fake", handle=handle)
    session.add(buyer)
    session.flush()
    thread = Thread(
        platform="fake",
        remote_id=f"rt-{item.id}-{handle}",
        item_id=item.id,
        buyer_id=buyer.id,
        created_at=clock.now(),
    )
    session.add(thread)
    session.commit()
    return thread


def _reject(tool: Tool, ctx: ToolContext, args) -> str:
    verdict = tool.check(ctx, args)
    assert isinstance(verdict, Reject), verdict
    return verdict.reason


def _run(tool: Tool, ctx: ToolContext, args):
    assert tool.check(ctx, args) == Allow()
    result = tool.execute(ctx, args)
    ctx.session.commit()
    return result


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (ESCALATE, EscalateArgs(reason="Buyer wants delivery")),
        (REMEMBER, RememberArgs(scope="buyer", lesson="Haggles hard", buyer_id=1)),
        (STATS, NoArgs()),
        (COMPS, QueryCompsArgs(query="ikea desk")),
    ],
)
def test_check_writes_nothing(ctx, session, clock, item, tool, args):
    _thread(session, clock, item, "ann")
    flushes = []
    event.listen(session, "after_flush", lambda *_: flushes.append(1))
    tool.check(ctx, args)
    assert flushes == [] and not session.new and not session.dirty


# ---------------------------------------------------------------- schedule_wake

WAKE = ScheduleWakeTool()


def test_schedule_wake_stores_one_wake_per_item(ctx, session, clock):
    first = ScheduleWakeArgs(at="2026-10-02T12:30", reason="Follow up with Lucas")
    result = _run(WAKE, ctx, first)
    assert result.summary == "You will be woken Fri Oct 2 12:30: Follow up with Lucas"
    later = ScheduleWakeArgs(at="2026-10-02T19:00", reason="Check Aisha came")
    _run(WAKE, ctx, later)

    [pending] = wakes.pending_scheduled(session, ctx.item_id)
    assert pending.reason == "[scheduled] Check Aisha came"
    assert pending.created_at == datetime(2026, 10, 2, 23, tzinfo=UTC)


@pytest.mark.parametrize(
    ("at", "reason", "error"),
    [
        ("2026-10-01T11:00", "x", "is not in the future"),
        ("2026-10-09T12:00", "x", "more than 7 days ahead"),
        ("tomorrow", "x", "YYYY-MM-DDTHH:MM"),
        ("2026-10-02T12:00", "  ", "Say in reason"),
    ],
)
def test_schedule_wake_rejects(ctx, at, reason, error):
    assert error in _reject(WAKE, ctx, ScheduleWakeArgs(at=at, reason=reason))


# ---------------------------------------------------------------- registry


def test_default_registry_has_every_tool_with_its_trust_key():
    registry = default_registry()
    expected = {
        "send_message": "send_message",
        "update_price": "update_price",
        "rewrite_listing": "rewrite_listing",
        "publish": "publish",
        "relist": "relist",
        "accept_offer": "accept_offer",
        "accept_bundle": "accept_offer",
        "book_pickup": "book_pickup",
        "release_commitment": "release_commitment",
        "complete_sale": "complete_sale",
        "get_stats": None,
        "query_comps": None,
        "escalate": None,
        "remember": None,
        "schedule_wake": None,
    }
    assert registry.names() == list(expected)
    for name, action in expected.items():
        tool = registry.get(name)
        assert tool is not None and tool.trust_action == action
        assert tool.read_only == (name in ("get_stats", "query_comps"))
    assert registry.get("delete_everything") is None


def test_specs_are_compact_and_model_facing():
    for spec in default_registry().specs():
        assert set(spec) == {"name", "description", "args_schema"}
        assert 40 < len(spec["description"]) < 400
        assert "—" not in spec["description"]
        assert spec["args_schema"]["additionalProperties"] is False


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate tool name: escalate"):
        ToolRegistry([EscalateTool(), EscalateTool()])


# ---------------------------------------------------------------- escalate


def test_escalate_records_and_notifies(ctx, session, clock, notifier, item):
    thread = _thread(session, clock, item, "ann")
    result = _run(ESCALATE, ctx, EscalateArgs(reason="Buyer asks to ship", thread_id=thread.id))

    escalation = session.get(Escalation, result.data["escalation_id"])
    assert (escalation.item_id, escalation.thread_id) == (item.id, thread.id)
    assert escalation.reason == "Buyer asks to ship" and escalation.created_at == clock.now()
    [notice] = notifier.notices
    assert notice.kind == NoticeKind.ESCALATION and notice.ref_id == escalation.id
    assert notice.body == "Buyer asks to ship" and notice.title == "Needs you: IKEA desk"
    assert result.summary == "Escalated to the owner: Buyer asks to ship"


def test_escalate_is_allowed_on_a_sold_item(session, clock, settings, notifier):
    sold = _item(session, clock, ItemStatus.SOLD)
    ctx = _ctx(session, clock, settings, notifier, sold, {})
    assert validate(ctx, ESCALATE, EscalateArgs(reason="Buyer says it broke")) == Allow()
    assert validate(ctx, REMEMBER, RememberArgs(scope="item", lesson="Sold fast")) == Allow()


def test_escalate_needs_a_reason(ctx):
    assert _reject(ESCALATE, ctx, EscalateArgs(reason=" ")) == (
        "Say in reason what you need from the owner."
    )


def test_escalate_rejects_a_thread_of_another_item(ctx, session, clock):
    other = _thread(session, clock, _item(session, clock), "zed")
    reason = _reject(ESCALATE, ctx, EscalateArgs(reason="x", thread_id=other.id))
    assert "not a conversation about this item" in reason


def test_escalate_once_until_resolved(ctx, session, clock, item):
    thread = _thread(session, clock, item, "ann")
    _run(ESCALATE, ctx, EscalateArgs(reason="Wants delivery", thread_id=thread.id))

    reason = _reject(ESCALATE, ctx, EscalateArgs(reason="  wants   DELIVERY ", thread_id=thread.id))
    assert "already escalated this (escalation 1, Thu Oct 1 12:00)" in reason
    assert "Do not escalate it again; carry on with what you can do." in reason
    assert ESCALATE.check(ctx, EscalateArgs(reason="Wants delivery")) == Allow()
    assert ESCALATE.check(ctx, EscalateArgs(reason="Other", thread_id=thread.id)) == Allow()

    session.get(Escalation, 1).resolved_at = clock.now()
    session.commit()
    assert (
        ESCALATE.check(ctx, EscalateArgs(reason="Wants delivery", thread_id=thread.id)) == Allow()
    )


def test_escalate_never_stores_the_address(ctx, session, notifier):
    result = _run(ESCALATE, ctx, EscalateArgs(reason="Buyer knows 123 Example St already"))
    escalation = session.get(Escalation, result.data["escalation_id"])
    assert escalation.reason == f"Buyer knows {ADDRESS_PLACEHOLDER} already"
    assert "123 Example" not in notifier.notices[0].body and "123 Example" not in result.summary


# ---------------------------------------------------------------- remember


@pytest.mark.parametrize("scope", ["item", "global"])
def test_remember_item_and_global(ctx, session, item, scope):
    result = _run(REMEMBER, ctx, RememberArgs(scope=scope, lesson="Dimensions in title help"))
    memory = session.get(Memory, result.data["memory_id"])
    assert memory.scope == MemoryScope(scope) and memory.item_id == item.id
    assert memory.buyer_id is None and memory.source == "agent"
    assert result.summary == f"Remembered ({scope}): Dimensions in title help"


def test_remember_buyer(ctx, session, clock, item):
    thread = _thread(session, clock, item, "ann")
    args = RememberArgs(scope="buyer", lesson="Haggles, then pays", buyer_id=thread.buyer_id)
    memory = session.get(Memory, _run(REMEMBER, ctx, args).data["memory_id"])
    assert memory.scope == MemoryScope.BUYER and memory.buyer_id == thread.buyer_id


def test_remember_a_duplicate_is_ok_and_already_known(ctx, session):
    _run(REMEMBER, ctx, RememberArgs(scope="item", lesson="Lowballers walk at $100"))
    result = _run(REMEMBER, ctx, RememberArgs(scope="item", lesson="lowballers walk at $100."))
    assert result.ok and result.summary == "Already known; nothing new saved."
    assert len(session.exec(select(Memory)).all()) == 1


def test_remember_ignores_buyer_id_outside_buyer_scope(ctx, session):
    result = _run(REMEMBER, ctx, RememberArgs(scope="global", lesson="x y z", buyer_id=99))
    assert session.get(Memory, result.data["memory_id"]).buyer_id is None


@pytest.mark.parametrize(
    ("lesson", "expected"),
    [
        ("   ", "The lesson is empty. Write what you learned."),
        ("x" * 301, "The lesson is 301 characters; the limit is 300. Say it shorter."),
        ("Ann's number is 416-555-0199", "Describe the pattern instead."),
        ("Ann lives near 123 Example St", "Describe the pattern instead."),
    ],
)
def test_remember_rejects_bad_lessons(ctx, lesson, expected):
    assert expected in _reject(REMEMBER, ctx, RememberArgs(scope="item", lesson=lesson))


def test_remember_buyer_needs_a_buyer_of_this_item(ctx, session, clock, item):
    ann = _thread(session, clock, item, "ann")
    stranger = _thread(session, clock, _item(session, clock), "zed")

    reason = _reject(REMEMBER, ctx, RememberArgs(scope="buyer", lesson="Reliable"))
    assert reason == (
        "No buyer_id was given. A buyer lesson needs the buyer_id of a buyer in this "
        f"item's threads: {ann.buyer_id}."
    )
    reason = _reject(
        REMEMBER, ctx, RememberArgs(scope="buyer", lesson="Reliable", buyer_id=stranger.buyer_id)
    )
    assert f"Buyer {stranger.buyer_id} has no thread here" in reason


def test_remember_scope_is_in_the_schema():
    with pytest.raises(ValidationError):
        REMEMBER.parse_args({"scope": "everyone", "lesson": "x"})


# ---------------------------------------------------------------- get_stats


def _live(session, clock, item: Item, platform: str, days_ago: float) -> Listing:
    listing = Listing(
        item_id=item.id,
        platform=platform,
        remote_id=f"{platform}-1",
        url=f"https://{platform}.test/1",
        status=ListingStatus.LIVE,
        price_cents=12_000,
        title=item.title,
        description="",
        published_at=clock.now() - timedelta(days=days_ago),
    )
    session.add(listing)
    session.commit()
    return listing


def _snap(session, clock, listing: Listing, hours_ago: float, views: int, saves: int, msgs: int):
    session.add(
        ListingStatsSnapshot(
            listing_id=listing.id,
            captured_at=clock.now() - timedelta(hours=hours_ago),
            views=views,
            saves=saves,
            message_count=msgs,
        )
    )
    session.commit()


def test_get_stats_reports_latest_and_deltas(ctx, session, clock, item):
    listing = _live(session, clock, item, "kijiji", days_ago=9)
    _snap(session, clock, listing, 24 * 8, 10, 0, 0)
    _snap(session, clock, listing, 30, 90, 3, 2)
    _snap(session, clock, listing, 1, 120, 5, 3)

    result = _run(STATS, ctx, NoArgs())

    [row] = result.data["listings"]
    assert row["listing_id"] == listing.id and row["days_live"] == 9.0
    assert (row["views"], row["saves"], row["messages"]) == (120, 5, 3)
    assert (row["views_24h"], row["saves_24h"], row["messages_24h"]) == (30, 2, 1)
    assert (row["views_7d"], row["saves_7d"], row["messages_7d"]) == (110, 5, 3)
    assert result.summary == (
        "kijiji listing 1 (live, $120.00, 9.0 days): views 120 (+30 in 24h, +110 in 7d); "
        "saves 5 (+2 in 24h, +5 in 7d); messages 3 (+1 in 24h, +3 in 7d)"
    )


def test_get_stats_counts_across_a_relist(ctx, session, clock, item):
    """A relist restarts the platform's counts while snapshots continue on the same row."""
    listing = _live(session, clock, item, "fake", days_ago=3)
    _snap(session, clock, listing, 60, 40, 2, 1)
    _snap(session, clock, listing, 30, 68, 3, 2)
    _snap(session, clock, listing, 3, 105, 4, 3)
    listing.last_relisted_at = clock.now() - timedelta(hours=2)
    _snap(session, clock, listing, 1, 3, 0, 0)

    [row] = STATS.execute(ctx, NoArgs()).data["listings"]

    assert (row["views"], row["views_24h"], row["views_7d"]) == (3, 40, 108)
    assert (row["saves_24h"], row["saves_7d"]) == (1, 4)
    assert (row["messages_24h"], row["messages_7d"]) == (1, 3)


def test_get_stats_young_listing_counts_from_zero(ctx, session, clock, item):
    listing = _live(session, clock, item, "fake", days_ago=0.5)
    _snap(session, clock, listing, 1, 40, 2, 1)
    [row] = STATS.execute(ctx, NoArgs()).data["listings"]
    assert row["views_24h"] == 40 and row["views_7d"] == 40 and row["days_live"] == 0.5


def test_get_stats_missing_history_is_none(ctx, session, clock, item):
    old = _live(session, clock, item, "fake", days_ago=10)
    _snap(session, clock, old, 2, 50, 1, 0)
    _live(session, clock, item, "kijiji", days_ago=3)

    first, second = STATS.execute(ctx, NoArgs()).data["listings"]

    assert first["views"] == 50 and first["views_24h"] is None and first["views_7d"] is None
    assert second["views"] is None and second["views_24h"] is None
    assert "views not measured yet" in STATS.execute(ctx, NoArgs()).summary


def test_get_stats_delisted_stops_the_clock_and_drafts_are_skipped(ctx, session, clock, item):
    listing = _live(session, clock, item, "fake", days_ago=5)
    listing.status = ListingStatus.DELISTED
    listing.delisted_at = clock.now() - timedelta(days=2)
    session.add(
        Listing(item_id=item.id, platform="kijiji", price_cents=1, title="", description="")
    )
    session.commit()
    [row] = STATS.execute(ctx, NoArgs()).data["listings"]
    assert row["days_live"] == 3.0 and row["status"] == "delisted"


def test_get_stats_with_nothing_posted(ctx):
    result = STATS.execute(ctx, NoArgs())
    assert result.ok and result.summary == "No listing has been posted yet."


def test_get_stats_is_allowed_on_a_sold_item(session, clock, settings, notifier):
    sold = _item(session, clock, ItemStatus.SOLD)
    ctx = _ctx(session, clock, settings, notifier, sold, {})
    assert validate(ctx, STATS, NoArgs()) == Allow()


# ---------------------------------------------------------------- query_comps


def test_query_comps_summarizes_every_platform(ctx, session, clock, item):
    own = _live(session, clock, item, "fake", days_ago=1)
    kijiji = SearchStub(
        "kijiji",
        [SearchResult(f"Desk {n}", n * 1_000, url=f"https://k.test/{n}") for n in (4, 6, 8, 10)]
        + [SearchResult("Free desk", 0), SearchResult("My desk", 12_000, url=own.url)],
    )
    fake = SearchStub("fake", [SearchResult("Desk 12", 12_000), SearchResult("Desk 14", 14_000)])
    broken = SearchStub("facebook", error=PlatformError("logged out"))
    unsupported = SearchStub("craigslist", error=NotImplementedError())
    platforms = {"kijiji": kijiji, "fake": fake, "facebook": broken, "craigslist": unsupported}
    ctx = _ctx(session, clock, ctx.settings, ctx.notifier, item, platforms)

    result = _run(COMPS, ctx, QueryCompsArgs(query="  ikea desk "))

    data = result.data
    assert kijiji.queries == [("ikea desk", 20)]
    assert data["count"] == 6
    assert (data["p25_cents"], data["median_cents"], data["p75_cents"]) == (6_500, 9_000, 11_500)
    assert [s["price_cents"] for s in data["samples"]] == [4_000, 6_000, 8_000, 12_000, 14_000]
    assert data["samples"][-1] == {"title": "Desk 14", "price_cents": 14_000, "platform": "fake"}
    assert data["skipped"] == ["facebook: logged out", "craigslist: search not supported"]
    assert result.summary.startswith(
        "6 comparable listings for 'ikea desk': median $90.00, middle half $65.00 to $115.00."
    )
    assert "Not searched: facebook: logged out" in result.summary


def test_query_comps_with_no_results(session, clock, settings, notifier, item):
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": SearchStub("fake")})
    result = COMPS.execute(ctx, QueryCompsArgs(query="unobtainium"))
    assert result.data["count"] == 0 and result.data["median_cents"] is None
    assert result.summary == "No comparable listings found for 'unobtainium'."


def test_query_comps_single_result(session, clock, settings, notifier, item):
    ctx = _ctx(
        session,
        clock,
        settings,
        notifier,
        item,
        {"fake": SearchStub("fake", [SearchResult("D", 5_000)])},
    )
    data = COMPS.execute(ctx, QueryCompsArgs(query="desk")).data
    assert data["p25_cents"] == data["median_cents"] == data["p75_cents"] == 5_000


def test_query_comps_needs_a_platform(ctx):
    reason = _reject(COMPS, ctx, QueryCompsArgs(query="desk"))
    assert reason == "No platform is connected, so there is nothing to search. Skip comps."


def test_query_comps_needs_a_platform_that_can_search(session, clock, settings, notifier, item):
    facebook = SearchStub("facebook")
    facebook.searches = False
    ctx = _ctx(session, clock, settings, notifier, item, {"facebook": facebook})

    reason = _reject(COMPS, ctx, QueryCompsArgs(query="desk"))

    assert reason == (
        "facebook cannot search other sellers' listings, so there is nothing to search. "
        "Skip comps."
    )


@pytest.mark.parametrize(
    "raw", [{"query": "  ab "}, {"query": "x" * 101}, {"query": "desk", "limit": 10}]
)
def test_query_comps_argument_bounds(raw):
    with pytest.raises(ValidationError):
        COMPS.parse_args(raw)


def test_query_comps_answers_from_the_cache_for_a_day(session, clock, settings, notifier, item):
    stub = SearchStub("fake", [SearchResult("Desk", 5_000, url="https://f.test/1")])
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": stub})

    first = COMPS.execute(ctx, QueryCompsArgs(query="IKEA Desk"))
    clock.advance(hours=23)
    stub.results = [SearchResult("Desk", 9_000, url="https://f.test/2")]
    again = COMPS.execute(ctx, QueryCompsArgs(query="  ikea   desk!"))

    assert stub.queries == [("ikea desk", 20)]
    assert again.data["median_cents"] == first.data["median_cents"] == 5_000
    assert "Searched " in again.summary and "Searched " not in first.summary

    clock.advance(hours=2)
    later = COMPS.execute(ctx, QueryCompsArgs(query="ikea desk"))
    assert len(stub.queries) == 2 and later.data["median_cents"] == 9_000


def test_query_comps_uses_the_cache_with_no_platform_connected(
    session, clock, settings, notifier, item
):
    stub = SearchStub("fake", [SearchResult("Desk", 5_000)])
    online = _ctx(session, clock, settings, notifier, item, {"fake": stub})
    COMPS.execute(online, QueryCompsArgs(query="desk"))
    clock.advance(days=3)
    offline = _ctx(session, clock, settings, notifier, item, {})

    assert validate(offline, COMPS, QueryCompsArgs(query="Desk")) == Allow()
    result = COMPS.execute(offline, QueryCompsArgs(query="Desk"))
    assert result.data["median_cents"] == 5_000
    assert result.data["searched_at"] == (clock.now() - timedelta(days=3)).isoformat()


def test_query_comps_rejects_a_query_with_no_words(ctx):
    ctx.platforms = {"fake": SearchStub("fake")}
    assert "no words" in _reject(COMPS, ctx, QueryCompsArgs(query="!!! ???"))


def test_query_comps_failing_everywhere_is_not_cached(session, clock, settings, notifier, item):
    stub = SearchStub("fake", error=PlatformError("timeout"))
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": stub})

    result = COMPS.execute(ctx, QueryCompsArgs(query="desk"))
    stub.error, stub.results = None, [SearchResult("Desk", 4_000)]
    retry = COMPS.execute(ctx, QueryCompsArgs(query="desk"))

    assert result.data["count"] == 0 and result.data["skipped"] == ["fake: timeout"]
    assert result.data["searched_at"] is None
    assert retry.data["count"] == 1 and len(stub.queries) == 2
