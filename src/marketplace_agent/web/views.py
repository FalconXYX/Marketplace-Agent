"""Read models for the web pages. Queries only: every change goes through ControlService."""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlmodel import Session, col, func, select

from marketplace_agent.config import Settings
from marketplace_agent.control.describe import describe_call
from marketplace_agent.core.items import (
    get_attributes,
    get_goal,
    get_item,
    item_photos,
    list_items,
)
from marketplace_agent.core.models import (
    AgentRun,
    ApprovalStatus,
    Escalation,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    PendingApproval,
    Photo,
    Thread,
)
from marketplace_agent.guard import handover, ledger
from marketplace_agent.tools.impl.common import buyer_name

MESSAGES_PER_THREAD = 6
RECENT_RUNS = 5
SHOWN_LISTINGS = (ListingStatus.LIVE, ListingStatus.AWAITING_HUMAN)
STATUS_ORDER = (
    ItemStatus.COMMITTED,
    ItemStatus.LISTED,
    ItemStatus.DRAFT,
    ItemStatus.SOLD,
    ItemStatus.WITHDRAWN,
)
ON_SALE = (ItemStatus.LISTED, ItemStatus.COMMITTED)
CLOSED = (ItemStatus.SOLD, ItemStatus.WITHDRAWN)


@dataclass(frozen=True)
class ItemCard:
    item: Item
    price_cents: int | None
    listings: list[Listing]
    approvals: int
    escalations: int
    deal_cents: int | None
    photo_id: int | None


@dataclass(frozen=True)
class BundledItem:
    id: int
    title: str
    price_cents: int  # its share of the bundle total


@dataclass(frozen=True)
class DealView:
    id: int
    price_cents: int  # this item's price; its share when bundled
    buyer: str
    undo_until: datetime | None  # None once the undo window has closed
    pickup_start: datetime | None
    pickup_end: datetime | None
    pickup_check: Escalation | None  # open while the owner is asked whether the buyer came
    total_cents: int = 0  # the whole deal: price_cents, or the bundle's total
    partners: tuple[BundledItem, ...] = ()  # the bundle's other items; empty for one item

    @property
    def bundle(self) -> bool:
        return bool(self.partners)


@dataclass(frozen=True)
class ThreadView:
    thread: Thread
    buyer: str
    messages: list[Message]  # the latest few, oldest first
    earlier: int  # how many older messages are not shown


@dataclass(frozen=True)
class RunView:
    started_at: datetime
    triggers: list[str]
    rationale: str
    outcomes: list[str]
    error: str | None


@dataclass(frozen=True)
class ApprovalView:
    approval: PendingApproval
    details: list[str]


@dataclass(frozen=True)
class ItemView:
    item: Item
    goal: Goal | None
    attributes: dict[str, str]
    photos: list[Photo]
    listings: list[Listing]
    deal: DealView | None
    approvals: list[ApprovalView]
    escalations: list[Escalation]  # open, except the pickup check, which the deal shows
    threads: list[ThreadView]
    runs: list[RunView]

    @property
    def on_sale(self) -> bool:
        return self.item.status in ON_SALE

    @property
    def closed(self) -> bool:
        """Sold or withdrawn: nothing more is sent to buyers from here."""
        return self.item.status in CLOSED

    @property
    def thread_ids(self) -> set[int]:
        return {view.thread.id for view in self.threads if view.thread.id is not None}

    @property
    def anchors(self) -> set[str]:
        """Where the page can show a flash message: its sections and cards."""
        anchors = {"top"}
        if self.on_sale:
            anchors.add("controls")
        if self.deal is not None:
            anchors.add("deal")
        anchors.update(f"approval-{entry.approval.id}" for entry in self.approvals)
        anchors.update(f"escalation-{escalation.id}" for escalation in self.escalations)
        anchors.update(f"thread-{thread_id}" for thread_id in self.thread_ids)
        return anchors


def item_cards(session: Session) -> list[tuple[ItemStatus, list[ItemCard]]]:
    """Every item, grouped by status: deals first, then for sale, drafts, sold, withdrawn."""
    goals = {goal.item_id: goal for goal in session.exec(select(Goal)).all()}
    listings: dict[int, list[Listing]] = defaultdict(list)
    for listing in session.exec(
        select(Listing).where(col(Listing.status).in_(SHOWN_LISTINGS)).order_by(col(Listing.id))
    ).all():
        listings[listing.item_id].append(listing)
    approvals = _counts(session, PendingApproval, PendingApproval.status == ApprovalStatus.PENDING)
    escalations = _counts(session, Escalation, col(Escalation.resolved_at).is_(None))
    thumbnails = _thumbnails(session)
    groups: dict[ItemStatus, list[ItemCard]] = defaultdict(list)
    for item in list_items(session):
        assert item.id is not None
        goal = goals.get(item.id)
        groups[item.status].append(
            ItemCard(
                item=item,
                price_cents=goal.list_price_cents if goal else None,
                listings=listings[item.id],
                approvals=approvals.get(item.id, 0),
                escalations=escalations.get(item.id, 0),
                deal_cents=_deal_cents(session, item),
                photo_id=thumbnails.get(item.id),
            )
        )
    return [(status, groups[status]) for status in STATUS_ORDER if groups[status]]


def loose_escalations(session: Session) -> list[Escalation]:
    """Open escalations about no item, such as a scam freeze in a conversation with no listing."""
    query = (
        select(Escalation)
        .where(col(Escalation.item_id).is_(None), col(Escalation.resolved_at).is_(None))
        .order_by(col(Escalation.id))
    )
    return list(session.exec(query).all())


def item_view(session: Session, settings: Settings, now: datetime, item_id: int) -> ItemView:
    """Raises ItemNotFound."""
    item = get_item(session, item_id)
    deal = _deal(session, now, item_id)
    check_id = deal.pickup_check.id if deal and deal.pickup_check else None
    return ItemView(
        item=item,
        goal=get_goal(session, item_id),
        attributes=get_attributes(item),
        photos=item_photos(session, item_id),
        listings=_listings(session, item_id),
        deal=deal,
        approvals=[_approval(session, settings, a) for a in _pending(session, item_id)],
        escalations=[e for e in _open_escalations(session, item_id) if e.id != check_id],
        threads=_threads(session, item_id),
        runs=_runs(session, item_id),
    )


def _counts(session: Session, model: Any, condition: Any) -> dict[int, int]:
    query = select(model.item_id, func.count()).where(condition).group_by(model.item_id)
    return {item_id: count for item_id, count in session.exec(query).all() if item_id is not None}


def _thumbnails(session: Session) -> dict[int, int]:
    first: dict[int, int] = {}
    for photo in session.exec(select(Photo).order_by(col(Photo.position), col(Photo.id))).all():
        assert photo.id is not None
        first.setdefault(photo.item_id, photo.id)
    return first


def _deal_cents(session: Session, item: Item) -> int | None:
    if item.status != ItemStatus.COMMITTED or item.id is None:
        return None
    active = ledger.active_commitment(session, item.id)
    return active.accepted.price_cents if active else None


def _deal(session: Session, now: datetime, item_id: int) -> DealView | None:
    active = ledger.active_commitment(session, item_id)
    if active is None:
        return None
    accepted, pickup = active.accepted, active.pickup
    assert accepted.id is not None
    deadline = accepted.undo_deadline
    rows = ledger.deal_rows(session, accepted)
    partners = []
    for row in rows:
        other = session.get(Item, row.item_id)
        if row.item_id != item_id and other is not None:
            title = other.title or other.hint or f"Item {row.item_id}"
            partners.append(BundledItem(row.item_id, title, row.price_cents or 0))
    return DealView(
        id=accepted.id,
        price_cents=accepted.price_cents or 0,
        buyer=buyer_name(session, accepted.buyer_id),
        undo_until=deadline if deadline is not None and now < deadline else None,
        pickup_start=pickup.pickup_start if pickup else None,
        pickup_end=pickup.pickup_end if pickup else None,
        pickup_check=handover.open_check(session, accepted.id),
        total_cents=sum(row.price_cents or 0 for row in rows),
        partners=tuple(partners),
    )


def _listings(session: Session, item_id: int) -> list[Listing]:
    query = select(Listing).where(Listing.item_id == item_id).order_by(col(Listing.id).desc())
    return list(session.exec(query).all())


def _pending(session: Session, item_id: int) -> list[PendingApproval]:
    query = (
        select(PendingApproval)
        .where(
            PendingApproval.item_id == item_id,
            PendingApproval.status == ApprovalStatus.PENDING,
        )
        .order_by(col(PendingApproval.id))
    )
    return list(session.exec(query).all())


def _approval(session: Session, settings: Settings, approval: PendingApproval) -> ApprovalView:
    """What the parked call would do, in the words the Discord card uses: the buyer by name,
    times in local time."""
    try:
        args: Any = json.loads(approval.args_json or "{}")
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        return ApprovalView(approval=approval, details=[approval.args_json])
    said = describe_call(session, settings, approval.item_id, approval.tool, args)
    return ApprovalView(approval=approval, details=said.splitlines())


def _open_escalations(session: Session, item_id: int) -> list[Escalation]:
    query = (
        select(Escalation)
        .where(Escalation.item_id == item_id, col(Escalation.resolved_at).is_(None))
        .order_by(col(Escalation.id))
    )
    return list(session.exec(query).all())


def _threads(session: Session, item_id: int) -> list[ThreadView]:
    threads = session.exec(select(Thread).where(Thread.item_id == item_id)).all()
    views = [_thread(session, thread) for thread in threads]
    return sorted(views, key=_last_activity, reverse=True)


def _thread(session: Session, thread: Thread) -> ThreadView:
    query = (
        select(Message)
        .where(Message.thread_id == thread.id)
        .order_by(col(Message.sent_at).desc(), col(Message.id).desc())
    )
    messages = list(session.exec(query).all())
    shown = messages[:MESSAGES_PER_THREAD]
    return ThreadView(
        thread=thread,
        buyer=buyer_name(session, thread.buyer_id),
        messages=list(reversed(shown)),
        earlier=len(messages) - len(shown),
    )


def _last_activity(view: ThreadView) -> datetime:
    return view.messages[-1].sent_at if view.messages else view.thread.created_at


def _runs(session: Session, item_id: int) -> list[RunView]:
    query = (
        select(AgentRun)
        .where(AgentRun.item_id == item_id)
        .order_by(col(AgentRun.started_at).desc(), col(AgentRun.id).desc())
        .limit(RECENT_RUNS)
    )
    return [_run(run) for run in session.exec(query).all()]


def _run(run: AgentRun) -> RunView:
    triggers = [
        f"{t.get('kind')} ({t['detail']})" if t.get("detail") else str(t.get("kind"))
        for t in _json_list(run.triggers_json)
    ]
    outcomes = [
        f"{o.get('tool')} {o.get('status')}: {o.get('detail')}"
        for o in _json_list(run.results_json)
    ]
    return RunView(
        started_at=run.started_at,
        triggers=triggers,
        rationale=run.rationale,
        outcomes=outcomes,
        error=run.error,
    )


def _json_list(text: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(text or "[]")
    except json.JSONDecodeError:
        return []
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []
