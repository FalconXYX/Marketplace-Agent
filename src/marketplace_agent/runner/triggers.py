"""Why an item needs the agent now. Pure reads: the runner records what it acted on.

Each kind fires once per cause, because a successful wake moves item.last_agent_run_at (and the
runner moves item.last_wake_views, marks messages handled, and consumes wake requests):

| Kind | Fires when | Stops after a wake because |
|---|---|---|
| new_message | unhandled inbound in a thread that is not frozen | the messages are marked handled |
| stats_moved | total views - last_wake_views >= stats_wake_view_delta | last_wake_views is reset |
| idle | now - max(last run, first publish) >= idle_wake_days | the last run moves |
| pickup_elapsed | pickup end + no-show grace passed after the last run | the last run moves |
| deadline | goal deadline passed after the last run, item LISTED | the last run moves |
| human_override | an unconsumed WakeRequest from the owner | the request is consumed |
| scheduled | a wake the agent scheduled (schedule_wake) has come due | the request is consumed |
| approval_decided | an approval decided after the last run | the last run moves |

idle also covers a COMMITTED item with no booked pickup, so a buyer who goes quiet after a deal
cannot hold the item forever. pickup_elapsed re-arms while a deal stays open after its pickup, so
waiting at one wake never leaves the item promised for good:

| pickup_elapsed fires at | Why |
|---|---|
| pickup end + no-show grace | The buyer is late; the owner has just been asked whether they came |
| handover.no_show_from (pickup end + 6h) | The owner had time to answer; a no-show is allowed |
| every PICKUP_OVERDUE_EVERY after that | The deal is still open |
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from sqlmodel import Session, col, func, select

from marketplace_agent.agent.context import Trigger
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.items import get_goal
from marketplace_agent.core.wakes import is_scheduled, scheduled_reason
from marketplace_agent.core.models import (
    ApprovalStatus,
    Buyer,
    Direction,
    Item,
    ItemStatus,
    Listing,
    ListingStatsSnapshot,
    Message,
    PendingApproval,
    Thread,
    ThreadState,
    WakeRequest,
)
from marketplace_agent.guard.handover import no_show_from
from marketplace_agent.guard.ledger import active_commitment

PRIORITY_KINDS = frozenset(
    {"new_message", "pickup_elapsed", "human_override", "approval_decided", "deadline"}
)
WAKEABLE = (ItemStatus.LISTED, ItemStatus.COMMITTED)
PICKUP_OVERDUE_EVERY = timedelta(hours=24)

_Finder = Callable[[Session, Settings, datetime, Item], list[Trigger]]


def compute_triggers(
    session: Session, clock: Clock, settings: Settings
) -> dict[int, list[Trigger]]:
    """Triggers per item, for LISTED or COMMITTED items that are not paused."""
    now = clock.now()
    query = (
        select(Item)
        .where(col(Item.status).in_(WAKEABLE), col(Item.paused).is_(False))
        .order_by(col(Item.id))
    )
    found: dict[int, list[Trigger]] = {}
    for item in session.exec(query).all():
        assert item.id is not None
        triggers = [t for finder in _FINDERS for t in finder(session, settings, now, item)]
        if triggers:
            found[item.id] = triggers
    return found


def total_views(session: Session, item_id: int) -> int:
    """Sum over the item's listings of each one's latest snapshot views."""
    latest = (
        select(func.max(ListingStatsSnapshot.id))
        .join(Listing, col(Listing.id) == ListingStatsSnapshot.listing_id)
        .where(Listing.item_id == item_id)
        .group_by(col(ListingStatsSnapshot.listing_id))
    )
    query = select(func.coalesce(func.sum(ListingStatsSnapshot.views), 0)).where(
        col(ListingStatsSnapshot.id).in_(latest)
    )
    return int(session.exec(query).one())


def _new_messages(session: Session, settings: Settings, now: datetime, item: Item) -> list[Trigger]:
    query = (
        select(Message, Thread, Buyer)
        .join(Thread, col(Thread.id) == Message.thread_id)
        .join(Buyer, col(Buyer.id) == Thread.buyer_id)
        .where(
            Thread.item_id == item.id,
            Thread.state != ThreadState.FROZEN,
            Message.direction == Direction.INBOUND,
            col(Message.handled).is_(False),
        )
        .order_by(col(Thread.id), col(Message.sent_at), col(Message.id))
    )
    waiting: dict[int, tuple[str, list[int]]] = {}
    for message, thread, buyer in session.exec(query).all():
        assert thread.id is not None and message.id is not None
        name = buyer.display_name or buyer.handle
        waiting.setdefault(thread.id, (name, []))[1].append(message.id)
    return [
        Trigger(
            kind="new_message",
            detail=f"{len(ids)} new message(s) from {name}",
            thread_id=thread_id,
            message_ids=tuple(ids),
        )
        for thread_id, (name, ids) in waiting.items()
    ]


def _stats_moved(session: Session, settings: Settings, now: datetime, item: Item) -> list[Trigger]:
    assert item.id is not None
    views = total_views(session, item.id)
    moved = views - item.last_wake_views
    if moved < max(1, settings.stats_wake_view_delta):
        return []
    return [Trigger(kind="stats_moved", detail=f"views up {moved} to {views} since your last run")]


def _idle(session: Session, settings: Settings, now: datetime, item: Item) -> list[Trigger]:
    if settings.idle_wake_days <= 0 or not _may_idle(session, item):
        return []
    first_published = session.exec(
        select(func.min(Listing.published_at)).where(Listing.item_id == item.id)
    ).one()
    stamps = [at for at in (item.last_agent_run_at, first_published) if at is not None]
    if not stamps:
        return []
    quiet = now - max(stamps)
    if quiet < timedelta(days=settings.idle_wake_days):
        return []
    return [Trigger(kind="idle", detail=f"nothing has happened for {quiet.days} day(s)")]


def _may_idle(session: Session, item: Item) -> bool:
    if item.status == ItemStatus.LISTED:
        return True
    assert item.id is not None
    active = active_commitment(session, item.id)
    return active is not None and active.pickup is None


def _pickup_elapsed(
    session: Session, settings: Settings, now: datetime, item: Item
) -> list[Trigger]:
    assert item.id is not None
    active = active_commitment(session, item.id)
    if active is None or active.pickup is None or active.pickup.pickup_end is None:
        return []
    end = active.pickup.pickup_end
    late = end + timedelta(minutes=settings.no_show_grace_minutes)
    answered_by = no_show_from(settings, end)
    if now < late:
        return []
    if now < answered_by:
        due, detail = late, "the booked pickup and its no-show grace period are over"
    else:
        laps = (now - answered_by) // PICKUP_OVERDUE_EVERY
        due = answered_by + laps * PICKUP_OVERDUE_EVERY
        detail = (
            "the owner has had until now to say whether the buyer came"
            if laps == 0
            else f"the pickup ended {_span(now - end)} ago and the deal is still open"
        )
    if not _before(item.last_agent_run_at, due):
        return []
    return [Trigger(kind="pickup_elapsed", detail=detail, thread_id=active.accepted.thread_id)]


def _span(span: timedelta) -> str:
    days = span.total_seconds() / 86_400
    return f"{days:.1f} days" if days >= 1 else f"{span.total_seconds() / 3600:.0f}h"


def _deadline(session: Session, settings: Settings, now: datetime, item: Item) -> list[Trigger]:
    assert item.id is not None
    goal = get_goal(session, item.id)
    if item.status != ItemStatus.LISTED or goal is None or goal.deadline > now:
        return []
    if not _before(item.last_agent_run_at, goal.deadline):
        return []
    return [Trigger(kind="deadline", detail="the goal's deadline has passed")]


def _wake_requests(
    session: Session, settings: Settings, now: datetime, item: Item
) -> list[Trigger]:
    """The owner's wake requests, and the agent's own scheduled wakes once they are due."""
    query = (
        select(WakeRequest)
        .where(
            WakeRequest.item_id == item.id,
            col(WakeRequest.consumed_at).is_(None),
            col(WakeRequest.created_at) <= now,
        )
        .order_by(col(WakeRequest.created_at), col(WakeRequest.id))
    )
    return [
        Trigger(kind="scheduled", detail=scheduled_reason(r))
        if is_scheduled(r)
        else Trigger(kind="human_override", detail=r.reason)
        for r in session.exec(query).all()
    ]


def _approval_decided(
    session: Session, settings: Settings, now: datetime, item: Item
) -> list[Trigger]:
    query = (
        select(PendingApproval)
        .where(
            PendingApproval.item_id == item.id,
            PendingApproval.status != ApprovalStatus.PENDING,
            col(PendingApproval.decided_at).is_not(None),
        )
        .order_by(col(PendingApproval.decided_at), col(PendingApproval.id))
    )
    return [
        Trigger(
            kind="approval_decided",
            detail=f"approval #{a.id} for {a.tool} was {a.status.value}",
        )
        for a in session.exec(query).all()
        if a.decided_at is not None and _before(item.last_agent_run_at, a.decided_at)
    ]


def _before(last_run: datetime | None, moment: datetime) -> bool:
    """The agent has not run since moment. A run at the very moment counts as after it, so a
    frozen test clock cannot refire the trigger."""
    return last_run is None or last_run < moment


_FINDERS: tuple[_Finder, ...] = (
    _pickup_elapsed,
    _wake_requests,
    _approval_decided,
    _new_messages,
    _deadline,
    _stats_moved,
    _idle,
)
