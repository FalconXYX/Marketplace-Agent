"""Append-only commitment ledger: the only code that promises an item to a buyer.

Every write runs under LEDGER_LOCK and first claims SQLite's write lock (a touch of the item
row), so its re-validation reads every commitment committed by other sessions. A session that
has written but not yet committed keeps the database write lock, so a second writer waits for
it instead of validating against stale state. Functions flush; the caller commits.

| Event | Item | Thread |
|---|---|---|
| accept | COMMITTED | COMMITTED |
| book_pickup | unchanged | unchanged |
| release, undo | LISTED, or unchanged when SOLD or WITHDRAWN | NEGOTIATING |
| complete | SOLD | CLOSED |
| close_item SOLD | SOLD, completing an active deal | CLOSED |
| close_item WITHDRAWN | WITHDRAWN, releasing an active deal | NEGOTIATING |

A bundle is one deal over 2 to 5 items: an ACCEPTED row per item, all sharing a bundle_key,
whose prices add up to the agreed total. Every event on any of its rows (book_pickup, release,
undo, complete, close_item) is applied to all of them at once, with one timestamp, and the
event rows carry the bundle_key too. Rows of one bundle share one pickup slot and never
conflict with each other.

A FROZEN thread keeps its state through every event. Writes outside the ledger that must agree
with it (the owner's floor change) take the same write lock with claim_item.
"""
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import exists, func, update
from sqlalchemy.orm import aliased
from sqlmodel import Session, col, select

from marketplace_agent.core.clock import Clock
from marketplace_agent.core.items import get_goal
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Thread,
    ThreadState,
)

LEDGER_LOCK = threading.RLock()
BUNDLE_MIN_ITEMS = 2
BUNDLE_MAX_ITEMS = 5

_CLOSING = (CommitmentKind.RELEASED, CommitmentKind.UNDONE, CommitmentKind.COMPLETED)
WITHDRAWAL = "The owner withdrew the item."  # the reason on a deal that a withdrawal released
_TERMINAL_ITEM = (ItemStatus.SOLD, ItemStatus.WITHDRAWN)


class LedgerError(Exception):
    pass


class LedgerConflict(LedgerError):
    pass


class UndoExpired(LedgerError):
    pass


@dataclass(frozen=True)
class ActiveCommitment:
    accepted: Commitment
    pickup: Commitment | None


def is_active(session: Session, accepted: Commitment) -> bool:
    if accepted.kind != CommitmentKind.ACCEPTED:
        return False
    closing = select(Commitment.id).where(
        Commitment.ref_id == accepted.id, col(Commitment.kind).in_(_CLOSING)
    )
    return session.exec(closing).first() is None


def active_commitment(session: Session, item_id: int) -> ActiveCommitment | None:
    rows = _active_accepted(session, Commitment.item_id == item_id)
    return _with_pickup(session, rows[0]) if rows else None


def active_commitment_for_thread(session: Session, thread_id: int) -> ActiveCommitment | None:
    """The active deal in a thread. A bundle can put several rows on one thread; the row for
    the thread's own item comes first."""
    rows = _active_accepted(session, Commitment.thread_id == thread_id)
    if not rows:
        return None
    thread = session.get(Thread, thread_id)
    own = [row for row in rows if thread is not None and row.item_id == thread.item_id]
    return _with_pickup(session, (own or rows)[0])


def active_pickups(session: Session) -> list[ActiveCommitment]:
    """One entry per active ACCEPTED row with a booking, so a bundle appears once per item."""
    active = (_with_pickup(session, row) for row in _active_accepted(session))
    return [entry for entry in active if entry.pickup is not None]


def item_history(session: Session, item_id: int) -> list[Commitment]:
    query = (
        select(Commitment)
        .where(Commitment.item_id == item_id)
        .order_by(col(Commitment.created_at), col(Commitment.id))
    )
    return list(session.exec(query).all())


def bundle_rows(session: Session, bundle_key: str) -> list[Commitment]:
    """The bundle's ACCEPTED rows, one per item, in the order they were written."""
    query = (
        select(Commitment)
        .where(Commitment.bundle_key == bundle_key, Commitment.kind == CommitmentKind.ACCEPTED)
        .order_by(col(Commitment.id))
    )
    return list(session.exec(query).all())


def deal_rows(session: Session, accepted: Commitment) -> list[Commitment]:
    """The ACCEPTED rows that every event on accepted applies to: its bundle, or itself."""
    if accepted.bundle_key is None:
        return [accepted]
    return bundle_rows(session, accepted.bundle_key) or [accepted]


def allocate(total_cents: int, weights: Sequence[int]) -> list[int]:
    """Split total_cents in proportion to weights, summing exactly. Each share is rounded down,
    then the cents left over go one each to the largest remainders (ties: the earlier weight)."""
    if total_cents < 0:
        raise ValueError("total_cents cannot be negative.")
    whole = sum(weights)
    if not weights or any(weight < 0 for weight in weights) or whole <= 0:
        raise ValueError("weights must be non-negative with a positive sum.")
    shares = [total_cents * weight // whole for weight in weights]
    by_remainder = sorted(
        range(len(weights)), key=lambda i: (-(total_cents * weights[i] % whole), i)
    )
    for index in by_remainder[: total_cents - sum(shares)]:
        shares[index] += 1
    return shares


def split_bundle(
    total_cents: int, list_prices: Sequence[int], minimums: Sequence[int]
) -> list[int]:
    """Item prices for a bundle: total_cents in proportion to list prices, except that an item
    whose share would fall below its minimum gets exactly its minimum and the rest is split the
    same way among the others. Sums exactly to total_cents. LedgerError when the total is below
    the sum of the minimums."""
    if len(list_prices) != len(minimums):
        raise ValueError("list_prices and minimums must have the same length.")
    need = sum(minimums)
    if total_cents < need:
        raise LedgerError(
            f"{total_cents} cents is below the {need} cents these items need together."
        )
    prices: list[int | None] = [None] * len(list_prices)
    while True:
        free = [i for i, price in enumerate(prices) if price is None]
        left = total_cents - sum(price for price in prices if price is not None)
        shares = allocate(left, [list_prices[i] for i in free])
        short = [i for i, share in zip(free, shares, strict=True) if share < minimums[i]]
        if not short:
            for i, share in zip(free, shares, strict=True):
                prices[i] = share
            return [price for price in prices if price is not None]
        for i in short:  # raising them only lowers the others' shares, so they stay short
            prices[i] = minimums[i]


def minimum_cents(goal: Goal) -> int:
    """The least the item may be promised for: its list price when firm, else its floor."""
    if goal.firmness == Firmness.FIRM:
        return max(goal.list_price_cents, goal.floor_price_cents)
    return goal.floor_price_cents


def accept(
    session: Session,
    clock: Clock,
    *,
    item_id: int,
    thread_id: int,
    buyer_id: int,
    price_cents: int,
    undo_minutes: int,
    actor: str = "agent",
    reason: str = "",
) -> Commitment:
    if undo_minutes < 0:
        raise LedgerError("undo_minutes cannot be negative.")
    with _ledger_write(session, clock, item_id) as item:
        _check_available(session, item)
        thread = _thread_for(session, thread_id, item_id=item_id, buyer_id=buyer_id)
        _check_price(session, item_id, price_cents)
        now = clock.now()
        row = Commitment(
            item_id=item_id,
            thread_id=thread_id,
            buyer_id=buyer_id,
            kind=CommitmentKind.ACCEPTED,
            price_cents=price_cents,
            undo_deadline=now + timedelta(minutes=undo_minutes),
            reason=reason,
            actor=actor,
            created_at=now,
        )
        session.add(row)
        item.status = ItemStatus.COMMITTED
        thread.state = ThreadState.COMMITTED
        return row


def accept_bundle(
    session: Session,
    clock: Clock,
    *,
    item_ids: Sequence[int],
    thread_id: int,
    buyer_id: int,
    total_cents: int,
    undo_minutes: int,
    actor: str = "agent",
) -> list[Commitment]:
    """Promise 2 to 5 items to one buyer for one total: an ACCEPTED row per item, in item_ids
    order, sharing a new bundle_key and priced by split_bundle over list prices and minimums.
    thread_id is where the deal was made: the buyer's thread about one of the items. Each row
    goes on the buyer's latest thread about its own item, else on thread_id. The agreed total
    is kept in every row's reason, "... for <total> cents in total."."""
    ids = list(item_ids)
    if not BUNDLE_MIN_ITEMS <= len(ids) <= BUNDLE_MAX_ITEMS:
        raise LedgerError(
            f"A bundle has {BUNDLE_MIN_ITEMS} to {BUNDLE_MAX_ITEMS} items, not {len(ids)}."
        )
    if len(set(ids)) != len(ids):
        raise LedgerError("An item can be in a bundle only once.")
    if undo_minutes < 0:
        raise LedgerError("undo_minutes cannot be negative.")
    with _ledger_write_all(session, clock, ids) as items:
        for item_id in ids:
            _check_available(session, items[item_id])
        deal = _bundle_thread(session, thread_id, item_ids=ids, buyer_id=buyer_id)
        threads = [bundle_item_thread(session, deal, item_id) for item_id in ids]
        for thread in threads:
            if thread.state == ThreadState.FROZEN:
                raise LedgerConflict(f"Thread {thread.id} is frozen; do not make deals in it.")
        prices = _bundle_prices(session, ids, total_cents)
        now = clock.now()
        key = _new_bundle_key(session)
        listed = ", ".join(str(item_id) for item_id in ids)
        rows = [
            Commitment(
                item_id=item_id,
                thread_id=thread.id,
                buyer_id=buyer_id,
                kind=CommitmentKind.ACCEPTED,
                price_cents=price,
                undo_deadline=now + timedelta(minutes=undo_minutes),
                reason=f"Bundle of items {listed} for {total_cents} cents in total.",
                actor=actor,
                bundle_key=key,
                created_at=now,
            )
            for item_id, thread, price in zip(ids, threads, prices, strict=True)
        ]
        session.add_all(rows)
        for item in items.values():
            item.status = ItemStatus.COMMITTED
        for thread in threads:
            thread.state = ThreadState.COMMITTED
        return rows


def bundle_item_thread(session: Session, deal_thread: Thread, item_id: int) -> Thread:
    """The thread a bundle row for item_id goes on: deal_thread when it is about that item, else
    the buyer's latest thread about it, else deal_thread."""
    if deal_thread.item_id == item_id:
        return deal_thread
    query = (
        select(Thread)
        .where(Thread.item_id == item_id, Thread.buyer_id == deal_thread.buyer_id)
        .order_by(col(Thread.id).desc())
        .execution_options(populate_existing=True)
    )
    return session.exec(query).first() or deal_thread


def book_pickup(
    session: Session,
    clock: Clock,
    *,
    accepted_id: int,
    start: datetime,
    end: datetime,
    actor: str = "agent",
) -> Commitment:
    """Book (or rebook) the deal's pickup. A bundle gets one booking per row, all at this slot;
    the one for accepted_id is returned."""
    start, end = _utc(start, "start"), _utc(end, "end")
    if start >= end:
        raise LedgerError("Pickup start must be before its end.")
    accepted = _accepted_row(session, accepted_id)
    rows = deal_rows(session, accepted)
    with _ledger_write_all(session, clock, [row.item_id for row in rows]):
        for row in rows:
            _require_active(session, row)
        clash = _overlapping_pickup(session, rows, start, end)
        if clash is not None:
            raise LedgerConflict(
                f"Pickup overlaps another booked pickup ({clash.pickup_start:%Y-%m-%d %H:%M} to "
                f"{clash.pickup_end:%H:%M} UTC, item {clash.item_id}). "
                "Choose a time that does not overlap it."
            )
        now = clock.now()
        booked = [
            Commitment(
                item_id=row.item_id,
                thread_id=row.thread_id,
                buyer_id=row.buyer_id,
                kind=CommitmentKind.PICKUP_BOOKED,
                ref_id=row.id,
                price_cents=row.price_cents,
                pickup_start=start,
                pickup_end=end,
                actor=actor,
                bundle_key=row.bundle_key,
                created_at=now,
            )
            for row in rows
        ]
        session.add_all(booked)
        return booked[_position(rows, accepted)]


def release(
    session: Session, clock: Clock, *, accepted_id: int, reason: str, actor: str = "agent"
) -> Commitment:
    return _close(session, clock, accepted_id, CommitmentKind.RELEASED, actor=actor, reason=reason)


def undo(session: Session, clock: Clock, *, accepted_id: int, actor: str = "human") -> Commitment:
    return _close(session, clock, accepted_id, CommitmentKind.UNDONE, actor=actor)


def complete(
    session: Session, clock: Clock, *, accepted_id: int, actor: str = "agent"
) -> Commitment:
    return _close(session, clock, accepted_id, CommitmentKind.COMPLETED, actor=actor)


def close_item(
    session: Session,
    clock: Clock,
    *,
    item_id: int,
    status: ItemStatus,
    actor: str = "human",
    reason: str = "",
) -> ActiveCommitment | None:
    """Take the item off the market for good: SOLD completes its active deal, WITHDRAWN releases
    it. A bundle deal is completed or released whole; its other items become SOLD or LISTED with
    it. A released partner's row gets its own reason: only this item's row says WITHDRAWAL, the
    mark of a deal that ended with its item off the market. The deal is read under the write
    lock, so an accept still in flight in another session is waited for and closed too, never
    left promising a sold item. Returns the deal it closed."""
    if status not in _TERMINAL_ITEM:
        raise LedgerError(f"close_item takes SOLD or WITHDRAWN, not {status}.")
    with _ledger_write(session, clock, item_id) as item:
        if item.status in (ItemStatus.SOLD, status):
            raise LedgerConflict(f"Item {item_id} is already {item.status.value}.")
        active = active_commitment(session, item_id)
        if active is not None:
            rows = deal_rows(session, active.accepted)
            items = {row.item_id: claim_item(session, clock, row.item_id) for row in rows}
            sold = status == ItemStatus.SOLD
            kind = CommitmentKind.COMPLETED if sold else CommitmentKind.RELEASED
            partner = reason if sold else f"The owner withdrew item {item_id} from the bundle."
            reasons = {row.item_id: reason if row.item_id == item_id else partner for row in rows}
            _close_rows(session, clock.now(), rows, items, kind, actor, reasons)
        item.status = status
        return active


def claim_item(session: Session, clock: Clock, item_id: int) -> Item:
    """Take SQLite's write lock by touching the item row, then read the item fresh. Until the
    session ends, no other session can commit, and every read sees what others committed: a
    check made now cannot be overtaken by a concurrent write."""
    session.exec(update(Item).where(col(Item.id) == item_id).values(updated_at=clock.now()))
    item = session.get(Item, item_id, populate_existing=True)
    if item is None:
        raise LedgerError(f"Item {item_id} does not exist.")
    return item


@contextmanager
def _ledger_write(session: Session, clock: Clock, item_id: int) -> Iterator[Item]:
    with LEDGER_LOCK:
        yield claim_item(session, clock, item_id)
        session.flush()


@contextmanager
def _ledger_write_all(
    session: Session, clock: Clock, item_ids: Sequence[int]
) -> Iterator[dict[int, Item]]:
    with LEDGER_LOCK:
        yield {item_id: claim_item(session, clock, item_id) for item_id in item_ids}
        session.flush()


def _close(
    session: Session,
    clock: Clock,
    accepted_id: int,
    kind: CommitmentKind,
    *,
    actor: str,
    reason: str = "",
) -> Commitment:
    accepted = _accepted_row(session, accepted_id)
    rows = deal_rows(session, accepted)
    with _ledger_write_all(session, clock, [row.item_id for row in rows]) as items:
        for row in rows:
            _require_active(session, row)
        now = clock.now()
        if kind == CommitmentKind.UNDONE and _undo_expired(accepted, now):
            raise UndoExpired(
                f"The undo window for commitment {accepted.id} closed at "
                f"{accepted.undo_deadline:%Y-%m-%d %H:%M} UTC. Release the commitment instead."
            )
        closing = _close_rows(session, now, rows, items, kind, actor, reason)
        return closing[_position(rows, accepted)]


def _close_rows(
    session: Session,
    now: datetime,
    rows: Sequence[Commitment],
    items: dict[int, Item],
    kind: CommitmentKind,
    actor: str,
    reason: str | Mapping[int, str],
) -> list[Commitment]:
    """Close every row of one deal, with one reason or one per item id. A completed bundle is
    one completed deal for the buyer."""
    reasons = reason if isinstance(reason, Mapping) else {row.item_id: reason for row in rows}
    closing = [
        _append_closing(session, now, items[row.item_id], row, kind, actor, reasons[row.item_id])
        for row in rows
    ]
    if kind == CommitmentKind.COMPLETED:
        _count_completion(session, rows[0].buyer_id)
    return closing


def _append_closing(
    session: Session,
    now: datetime,
    item: Item,
    accepted: Commitment,
    kind: CommitmentKind,
    actor: str,
    reason: str,
) -> Commitment:
    """The closing row, and the item and thread states it implies. Releasing or undoing a deal
    on an item already SOLD or WITHDRAWN leaves the item as it is: it is not for sale again."""
    row = Commitment(
        item_id=accepted.item_id,
        thread_id=accepted.thread_id,
        buyer_id=accepted.buyer_id,
        kind=kind,
        ref_id=accepted.id,
        price_cents=accepted.price_cents,
        reason=reason,
        actor=actor,
        bundle_key=accepted.bundle_key,
        created_at=now,
    )
    session.add(row)
    if kind == CommitmentKind.COMPLETED:
        item.status = ItemStatus.SOLD
        _set_thread_state(session, accepted.thread_id, ThreadState.CLOSED)
    else:
        if item.status not in _TERMINAL_ITEM:
            item.status = ItemStatus.LISTED
        _set_thread_state(session, accepted.thread_id, ThreadState.NEGOTIATING)
    return row


def _active_accepted(session: Session, *conditions: object) -> list[Commitment]:
    closing = aliased(Commitment)
    closed = exists().where(closing.ref_id == Commitment.id, col(closing.kind).in_(_CLOSING))
    query = (
        select(Commitment)
        .where(Commitment.kind == CommitmentKind.ACCEPTED, ~closed, *conditions)
        .order_by(col(Commitment.id))
    )
    return list(session.exec(query).all())


def _with_pickup(session: Session, accepted: Commitment) -> ActiveCommitment:
    query = (
        select(Commitment)
        .where(Commitment.ref_id == accepted.id, Commitment.kind == CommitmentKind.PICKUP_BOOKED)
        .order_by(col(Commitment.id).desc())
    )
    return ActiveCommitment(accepted=accepted, pickup=session.exec(query).first())


def _accepted_row(session: Session, accepted_id: int) -> Commitment:
    row = session.get(Commitment, accepted_id)
    if row is None or row.kind != CommitmentKind.ACCEPTED:
        raise LedgerConflict(f"Commitment {accepted_id} is not an accepted offer.")
    return row


def _position(rows: Sequence[Commitment], accepted: Commitment) -> int:
    return next(i for i, row in enumerate(rows) if row.id == accepted.id)


def _require_active(session: Session, accepted: Commitment) -> None:
    if not is_active(session, accepted):
        raise LedgerConflict(
            f"Commitment {accepted.id} was already released, undone, or completed; "
            "it cannot change any more."
        )


def _check_available(session: Session, item: Item) -> None:
    if item.status in _TERMINAL_ITEM:
        raise LedgerConflict(f"Item {item.id} is {item.status}; it cannot be promised again.")
    current = active_commitment(session, item.id) if item.id is not None else None
    if current is not None:
        raise LedgerConflict(
            f"Item {item.id} is already promised to buyer {current.accepted.buyer_id} "
            f"(commitment {current.accepted.id}). Release that commitment before accepting "
            "another offer."
        )


def _thread_for(session: Session, thread_id: int, *, item_id: int, buyer_id: int) -> Thread:
    thread = session.get(Thread, thread_id, populate_existing=True)
    if thread is None or thread.item_id != item_id or thread.buyer_id != buyer_id:
        raise LedgerError(
            f"Thread {thread_id} is not buyer {buyer_id}'s thread for item {item_id}."
        )
    return _dealable(session, thread)


def _bundle_thread(
    session: Session, thread_id: int, *, item_ids: Sequence[int], buyer_id: int
) -> Thread:
    thread = session.get(Thread, thread_id, populate_existing=True)
    if thread is None or thread.item_id not in item_ids or thread.buyer_id != buyer_id:
        raise LedgerError(
            f"Thread {thread_id} is not buyer {buyer_id}'s thread about one of the bundle's "
            "items."
        )
    return _dealable(session, thread)


def _dealable(session: Session, thread: Thread) -> Thread:
    if thread.state == ThreadState.FROZEN:
        raise LedgerConflict(f"Thread {thread.id} is frozen; do not make deals in it.")
    buyer = session.get(Buyer, thread.buyer_id, populate_existing=True)
    if buyer is not None and buyer.flagged_scam:
        raise LedgerConflict(
            f"Buyer {thread.buyer_id} is flagged as a scammer; do not make deals."
        )
    return thread


def _check_price(session: Session, item_id: int, price_cents: int) -> None:
    goal = get_goal(session, item_id)
    if goal is None:
        return
    session.refresh(goal)
    if price_cents < goal.floor_price_cents:
        raise LedgerError(f"{price_cents} cents is below the floor of {goal.floor_price_cents}.")
    if goal.firmness == Firmness.FIRM and price_cents < goal.list_price_cents:
        raise LedgerError(
            f"The price is firm at {goal.list_price_cents} cents; {price_cents} is below it."
        )


def _bundle_prices(session: Session, item_ids: Sequence[int], total_cents: int) -> list[int]:
    goals = []
    for item_id in item_ids:
        goal = get_goal(session, item_id)
        if goal is None:
            raise LedgerError(f"Item {item_id} has no price goal, so it cannot be in a bundle.")
        session.refresh(goal)
        goals.append(goal)
    prices = split_bundle(
        total_cents,
        [goal.list_price_cents for goal in goals],
        [minimum_cents(goal) for goal in goals],
    )
    for item_id, price in zip(item_ids, prices, strict=True):
        _check_price(session, item_id, price)
    return prices


def _new_bundle_key(session: Session) -> str:
    """Unique because every later row has a higher id than any existing one (rows are never
    deleted); read under the write lock, so no other writer can take the same number."""
    last = session.exec(select(func.max(Commitment.id))).one()
    return f"bundle-{(last or 0) + 1}"


def _overlapping_pickup(
    session: Session, rows: Sequence[Commitment], start: datetime, end: datetime
) -> Commitment | None:
    """Another deal's active pickup sharing any instant with [start, end). The deal's own
    bookings (every row of its bundle) are ignored, so rescheduling never clashes with itself."""
    own = {row.id for row in rows}
    for other in active_pickups(session):
        booked = other.pickup
        if other.accepted.id in own or booked is None:
            continue
        if booked.pickup_start is None or booked.pickup_end is None:
            continue
        if start < booked.pickup_end and booked.pickup_start < end:
            return booked
    return None


def _undo_expired(accepted: Commitment, now: datetime) -> bool:
    return accepted.undo_deadline is None or now > accepted.undo_deadline


def _set_thread_state(session: Session, thread_id: int | None, state: ThreadState) -> None:
    thread = session.get(Thread, thread_id, populate_existing=True) if thread_id else None
    if thread is not None and thread.state != ThreadState.FROZEN:
        thread.state = state


def _count_completion(session: Session, buyer_id: int | None) -> None:
    buyer = session.get(Buyer, buyer_id, populate_existing=True) if buyer_id else None
    if buyer is not None:
        buyer.completed_count += 1


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise LedgerError(f"Pickup {name} must be timezone-aware.")
    return value.astimezone(UTC)
