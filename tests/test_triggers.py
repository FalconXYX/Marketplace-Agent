"""runner.triggers: which items need the agent, and why. Pure reads."""

from datetime import timedelta

import pytest
from sqlmodel import Session

from marketplace_agent.core import wakes
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    ApprovalStatus,
    Buyer,
    Direction,
    Item,
    ItemStatus,
    Listing,
    ListingStatsSnapshot,
    ListingStatus,
    Message,
    PendingApproval,
    Thread,
    ThreadState,
    WakeRequest,
)
from marketplace_agent.guard import ledger
from marketplace_agent.runner.triggers import PRIORITY_KINDS, compute_triggers, total_views


def _item(session: Session, clock, *, status=ItemStatus.LISTED, published_days_ago=0.0) -> int:
    now = clock.now()
    item = Item(title="IKEA desk", status=status, created_at=now, updated_at=now)
    session.add(item)
    session.flush()
    set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
    _listing(session, item.id, now - timedelta(days=published_days_ago))
    session.commit()
    return item.id


def _listing(session: Session, item_id: int, published_at) -> Listing:
    listing = Listing(
        item_id=item_id,
        platform="fake",
        remote_id=f"fake-{item_id}-{published_at.timestamp()}",
        status=ListingStatus.LIVE,
        price_cents=12_000,
        title="IKEA desk",
        description="Solid.",
        published_at=published_at,
    )
    session.add(listing)
    session.flush()
    return listing


def _thread(session: Session, clock, item_id: int, handle: str = "maya") -> Thread:
    buyer = Buyer(platform="fake", handle=handle, display_name=handle.title())
    session.add(buyer)
    session.flush()
    thread = Thread(
        platform="fake",
        remote_id=f"t-{handle}",
        item_id=item_id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    return thread


def _inbound(session: Session, clock, thread: Thread, body: str, handled=False) -> int:
    message = Message(
        thread_id=thread.id,
        direction=Direction.INBOUND,
        body=body,
        sent_at=clock.now(),
        handled=handled,
        via="buyer",
    )
    session.add(message)
    session.flush()
    return message.id


def _snapshot(session: Session, clock, listing_id: int, views: int) -> None:
    session.add(ListingStatsSnapshot(listing_id=listing_id, captured_at=clock.now(), views=views))
    session.flush()


def _ran(session: Session, item_id: int, at) -> None:
    item = session.get(Item, item_id)
    item.last_agent_run_at = at
    session.commit()


def _kinds(session: Session, clock, settings) -> dict[int, list[str]]:
    found = compute_triggers(session, clock, settings)
    return {item_id: [t.kind for t in triggers] for item_id, triggers in found.items()}


def _deal(session: Session, clock, item_id: int, thread: Thread, *, pickup_in=None) -> int:
    accepted = ledger.accept(
        session,
        clock,
        item_id=item_id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=11_000,
        undo_minutes=10,
    )
    session.flush()
    if pickup_in is not None:
        start = clock.now() + pickup_in
        ledger.book_pickup(
            session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(hours=1)
        )
    session.commit()
    return accepted.id


def test_priority_kinds():
    assert PRIORITY_KINDS == {
        "new_message",
        "pickup_elapsed",
        "human_override",
        "approval_decided",
        "deadline",
    }


def test_fresh_listing_needs_no_wake(session, clock, settings):
    _item(session, clock)

    assert compute_triggers(session, clock, settings) == {}


@pytest.mark.parametrize("status", [ItemStatus.DRAFT, ItemStatus.SOLD, ItemStatus.WITHDRAWN])
def test_only_listed_or_committed_items_wake(session, clock, settings, status):
    item_id = _item(session, clock, status=status, published_days_ago=30)
    _inbound(session, clock, _thread(session, clock, item_id), "Hi")
    session.add(WakeRequest(item_id=item_id, reason="owner", created_at=clock.now()))
    session.commit()

    assert compute_triggers(session, clock, settings) == {}


def test_paused_item_never_wakes(session, clock, settings):
    item_id = _item(session, clock, published_days_ago=30)
    _inbound(session, clock, _thread(session, clock, item_id), "Hi")
    session.get(Item, item_id).paused = True
    session.commit()

    assert compute_triggers(session, clock, settings) == {}


# ---------------------------------------------------------------- new_message


def test_new_messages_one_trigger_per_thread(session, clock, settings):
    item_id = _item(session, clock)
    maya = _thread(session, clock, item_id, "maya")
    omar = _thread(session, clock, item_id, "omar")
    first = _inbound(session, clock, maya, "Hi")
    second = _inbound(session, clock, maya, "Would you take 100?")
    _inbound(session, clock, maya, "old", handled=True)
    third = _inbound(session, clock, omar, "Dimensions?")
    session.commit()

    [a, b] = compute_triggers(session, clock, settings)[item_id]

    assert (a.kind, a.thread_id, a.message_ids) == ("new_message", maya.id, (first, second))
    assert (b.thread_id, b.message_ids) == (omar.id, (third,))
    assert a.detail == "2 new message(s) from Maya"


def test_frozen_thread_messages_do_not_wake(session, clock, settings):
    item_id = _item(session, clock)
    thread = _thread(session, clock, item_id)
    thread.state = ThreadState.FROZEN
    _inbound(session, clock, thread, "send me the code")
    session.commit()

    assert compute_triggers(session, clock, settings) == {}


# ---------------------------------------------------------------- stats_moved


def test_stats_moved_needs_the_configured_view_delta(session, clock, settings):
    item_id = _item(session, clock)
    listing = session.get(Listing, 1)
    _snapshot(session, clock, listing.id, settings.stats_wake_view_delta - 1)
    session.commit()
    assert _kinds(session, clock, settings) == {}

    _snapshot(session, clock, listing.id, settings.stats_wake_view_delta)
    session.commit()
    [trigger] = compute_triggers(session, clock, settings)[item_id]
    assert trigger.kind == "stats_moved"
    assert trigger.detail == f"views up {settings.stats_wake_view_delta} to 25 since your last run"


def test_stats_moved_counts_from_last_wake_views(session, clock, settings):
    item_id = _item(session, clock)
    _snapshot(session, clock, 1, 60)
    session.get(Item, item_id).last_wake_views = 40
    session.commit()
    assert _kinds(session, clock, settings) == {}

    _snapshot(session, clock, 1, 65)
    session.commit()
    assert _kinds(session, clock, settings) == {item_id: ["stats_moved"]}


def test_total_views_sums_latest_snapshot_of_each_listing(session, clock):
    item_id = _item(session, clock)
    second = _listing(session, item_id, clock.now())
    _snapshot(session, clock, 1, 10)
    _snapshot(session, clock, 1, 30)
    _snapshot(session, clock, second.id, 5)
    session.commit()

    assert total_views(session, item_id) == 35
    assert total_views(session, 999) == 0


# ---------------------------------------------------------------- idle


def test_idle_after_idle_wake_days_since_first_publish(session, clock, settings):
    item_id = _item(session, clock, published_days_ago=settings.idle_wake_days - 0.1)
    assert _kinds(session, clock, settings) == {}

    clock.advance(hours=3)

    assert _kinds(session, clock, settings) == {item_id: ["idle"]}


def test_idle_counts_from_the_last_run(session, clock, settings):
    item_id = _item(session, clock, published_days_ago=10)
    _ran(session, item_id, clock.now() - timedelta(days=1))

    assert _kinds(session, clock, settings) == {}


def test_never_published_never_run_item_is_not_idle(session, clock, settings):
    now = clock.now()
    session.add(Item(title="x", status=ItemStatus.LISTED, created_at=now, updated_at=now))
    session.commit()
    clock.advance(days=30)

    assert compute_triggers(session, clock, settings) == {}


def test_committed_item_idles_only_without_a_booked_pickup(session, clock, settings):
    item_id = _item(session, clock, published_days_ago=10)
    thread = _thread(session, clock, item_id)
    _inbound(session, clock, thread, "Deal at 110", handled=True)
    _deal(session, clock, item_id, thread)
    assert _kinds(session, clock, settings) == {item_id: ["idle"]}

    other = _item(session, clock, published_days_ago=10)
    other_thread = _thread(session, clock, other, "omar")
    _deal(session, clock, other, other_thread, pickup_in=timedelta(days=1))
    assert other not in compute_triggers(session, clock, settings)


# ---------------------------------------------------------------- pickup_elapsed


def test_pickup_elapsed_after_end_plus_grace_until_the_agent_runs(session, clock, settings):
    item_id = _item(session, clock)
    thread = _thread(session, clock, item_id)
    _deal(session, clock, item_id, thread, pickup_in=timedelta(hours=2))
    _ran(session, item_id, clock.now())

    clock.advance(hours=3, minutes=settings.no_show_grace_minutes - 1)
    assert _kinds(session, clock, settings) == {}

    clock.advance(minutes=1)
    [trigger] = compute_triggers(session, clock, settings)[item_id]
    assert (trigger.kind, trigger.thread_id) == ("pickup_elapsed", thread.id)

    _ran(session, item_id, clock.now())
    clock.advance(hours=5)
    assert _kinds(session, clock, settings) == {}


def test_pickup_elapsed_rearms_while_the_deal_stays_open(session, clock, settings):
    """Waiting once at pickup_elapsed must not leave the item promised for good on a quiet
    listing: it fires again when the owner's time to answer is up, then daily."""
    item_id = _item(session, clock)
    thread = _thread(session, clock, item_id)
    _deal(session, clock, item_id, thread, pickup_in=timedelta(hours=2))  # ends at +3h
    clock.advance(hours=3, minutes=30)
    assert _kinds(session, clock, settings) == {item_id: ["pickup_elapsed"]}
    _ran(session, item_id, clock.now())  # the agent decided to wait

    fired = []
    for _ in range(24 * 5):
        clock.advance(hours=1)
        found = compute_triggers(session, clock, settings).get(item_id, [])
        if found:
            fired.append((clock.now(), found[0].kind, found[0].detail))
            _ran(session, item_id, clock.now())

    assert [kind for _, kind, _ in fired] == ["pickup_elapsed"] * 5
    assert fired[0][2] == "the owner has had until now to say whether the buyer came"
    assert "the deal is still open" in fired[1][2]
    gaps = {later - earlier for (earlier, _, _), (later, _, _) in zip(fired, fired[1:])}
    assert gaps == {timedelta(hours=24)}


def test_a_scheduled_wake_fires_once_its_time_comes(session, clock, settings):
    item_id = _item(session, clock)
    _ran(session, item_id, clock.now())
    wakes.schedule(
        session, item_id, clock.now() + timedelta(hours=24), "follow up with Maya", clock.now()
    )
    session.commit()

    clock.advance(hours=23, minutes=59)
    assert _kinds(session, clock, settings) == {}
    clock.advance(minutes=1)
    [trigger] = compute_triggers(session, clock, settings)[item_id]
    assert (trigger.kind, trigger.detail) == ("scheduled", "follow up with Maya")


# ---------------------------------------------------------------- deadline


def test_deadline_fires_until_a_run_after_it(session, clock, settings):
    item_id = _item(session, clock)
    _ran(session, item_id, clock.now())
    clock.advance(days=14)
    assert _kinds(session, clock, settings) == {item_id: ["deadline", "idle"]}

    _ran(session, item_id, clock.now())
    clock.advance(days=1)
    assert _kinds(session, clock, settings) == {}


def test_deadline_ignores_committed_items(session, clock, settings):
    item_id = _item(session, clock)
    thread = _thread(session, clock, item_id)
    _deal(session, clock, item_id, thread, pickup_in=timedelta(days=20))
    _ran(session, item_id, clock.now())
    clock.advance(days=15)

    assert _kinds(session, clock, settings) == {}


# ---------------------------------------------------------------- owner


def test_unconsumed_wake_requests_fire_with_their_reason(session, clock, settings):
    item_id = _item(session, clock)
    _ran(session, item_id, clock.now())
    session.add(
        WakeRequest(
            item_id=item_id, reason="The owner set the floor to $90.00.", created_at=clock.now()
        )
    )
    session.add(
        WakeRequest(item_id=item_id, reason="done", created_at=clock.now(), consumed_at=clock.now())
    )
    session.commit()

    [trigger] = compute_triggers(session, clock, settings)[item_id]

    assert (trigger.kind, trigger.detail) == (
        "human_override",
        "The owner set the floor to $90.00.",
    )


def test_approval_decided_after_the_last_run(session, clock, settings):
    item_id = _item(session, clock)
    _ran(session, item_id, clock.now())

    def approval(status, decided_at):
        session.add(
            PendingApproval(
                item_id=item_id,
                tool="publish",
                args_json="{}",
                status=status,
                created_at=clock.now() - timedelta(hours=1),
                decided_at=decided_at,
            )
        )

    approval(ApprovalStatus.PENDING, None)
    approval(ApprovalStatus.REJECTED, clock.now() - timedelta(minutes=1))
    approval(ApprovalStatus.APPROVED, clock.now() + timedelta(minutes=1))
    session.commit()
    clock.advance(minutes=2)

    [trigger] = compute_triggers(session, clock, settings)[item_id]

    assert (trigger.kind, trigger.detail) == (
        "approval_decided",
        "approval #3 for publish was approved",
    )
