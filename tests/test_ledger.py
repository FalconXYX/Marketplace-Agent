import re
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from itertools import pairwise

import pytest
from sqlalchemy import event, text
from sqlmodel import Session, select

from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.db import init_db, make_engine
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Firmness,
    Item,
    ItemStatus,
    Thread,
    ThreadState,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.ledger import LedgerConflict, LedgerError, UndoExpired

FLOOR = 8_000
LIST = 10_000


def _item(session: Session, clock: FakeClock, firmness: Firmness = Firmness.SLIGHT) -> Item:
    item = Item(
        title="Desk", status=ItemStatus.LISTED, created_at=clock.now(), updated_at=clock.now()
    )
    session.add(item)
    session.flush()
    set_goal(
        session, clock, item.id, list_price_cents=LIST, floor_price_cents=FLOOR, firmness=firmness
    )
    return item


def _thread(session: Session, clock: FakeClock, item: Item, handle: str) -> Thread:
    buyer = Buyer(platform="fake", handle=handle)
    session.add(buyer)
    session.flush()
    thread = Thread(
        platform="fake",
        remote_id=f"t-{item.id}-{handle}",
        item_id=item.id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    return thread


def _accept(
    session, clock, thread: Thread, price: int = 9_000, undo_minutes: int = 10
) -> Commitment:
    return ledger.accept(
        session,
        clock,
        item_id=thread.item_id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=price,
        undo_minutes=undo_minutes,
    )


def _book(
    session, clock, accepted: Commitment, hours_ahead: float, minutes: int = 60
) -> Commitment:
    start = clock.now() + timedelta(hours=hours_ahead)
    return ledger.book_pickup(
        session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(minutes=minutes)
    )


def _count(session: Session, kind: CommitmentKind) -> int:
    return len(session.exec(select(Commitment).where(Commitment.kind == kind)).all())


# ---------------------------------------------------------------- accept


def test_accept_promises_the_item_and_opens_an_undo_window(session, clock):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "alice")

    accepted = _accept(session, clock, thread, price=9_000, undo_minutes=10)

    assert accepted.kind == CommitmentKind.ACCEPTED
    assert accepted.price_cents == 9_000
    assert accepted.undo_deadline == clock.now() + timedelta(minutes=10)
    assert item.status == ItemStatus.COMMITTED
    assert thread.state == ThreadState.COMMITTED
    active = ledger.active_commitment(session, item.id)
    assert active is not None and active.accepted.id == accepted.id and active.pickup is None
    assert ledger.active_commitment_for_thread(session, thread.id).accepted.id == accepted.id


def test_second_accept_for_the_same_item_conflicts_and_appends_nothing(session, clock):
    item = _item(session, clock)
    alice = _thread(session, clock, item, "alice")
    bob = _thread(session, clock, item, "bob")
    _accept(session, clock, alice)

    with pytest.raises(LedgerConflict, match="already promised"):
        _accept(session, clock, bob, price=LIST)
    with pytest.raises(LedgerConflict):
        _accept(session, clock, alice, price=LIST)

    assert _count(session, CommitmentKind.ACCEPTED) == 1
    assert ledger.active_commitment_for_thread(session, bob.id) is None


def test_item_can_be_promised_again_after_release(session, clock):
    item = _item(session, clock)
    alice = _thread(session, clock, item, "alice")
    bob = _thread(session, clock, item, "bob")
    first = _accept(session, clock, alice)
    ledger.release(session, clock, accepted_id=first.id, reason="backed out")

    second = _accept(session, clock, bob)

    assert ledger.active_commitment(session, item.id).accepted.id == second.id
    assert not ledger.is_active(session, first)


@pytest.mark.parametrize(
    ("firmness", "price"),
    [(Firmness.SLIGHT, FLOOR - 1), (Firmness.FLEXIBLE, 1), (Firmness.FIRM, LIST - 1)],
)
def test_accept_rechecks_floor_and_firm_price(session, clock, firmness, price):
    item = _item(session, clock, firmness)
    thread = _thread(session, clock, item, "alice")

    with pytest.raises(LedgerError):
        _accept(session, clock, thread, price=price)

    assert _count(session, CommitmentKind.ACCEPTED) == 0
    assert item.status == ItemStatus.LISTED


def test_accept_at_exactly_floor_and_firm_list_price_is_allowed(session, clock):
    slight = _item(session, clock)
    firm = _item(session, clock, Firmness.FIRM)
    _accept(session, clock, _thread(session, clock, slight, "a"), price=FLOOR)
    _accept(session, clock, _thread(session, clock, firm, "b"), price=LIST)
    assert _count(session, CommitmentKind.ACCEPTED) == 2


@pytest.mark.parametrize("status", [ItemStatus.SOLD, ItemStatus.WITHDRAWN])
def test_accept_refuses_sold_or_withdrawn_items(session, clock, status):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "alice")
    item.status = status
    session.flush()

    with pytest.raises(LedgerConflict):
        _accept(session, clock, thread)


def test_accept_refuses_frozen_threads_and_flagged_buyers(session, clock):
    item = _item(session, clock)
    frozen = _thread(session, clock, item, "frozen")
    frozen.state = ThreadState.FROZEN
    flagged = _thread(session, clock, item, "flagged")
    session.get(Buyer, flagged.buyer_id).flagged_scam = True
    session.flush()

    with pytest.raises(LedgerConflict, match="frozen"):
        _accept(session, clock, frozen)
    with pytest.raises(LedgerConflict, match="scam"):
        _accept(session, clock, flagged)


def test_accept_refuses_a_thread_from_another_item_or_buyer(session, clock):
    item = _item(session, clock)
    other = _item(session, clock)
    stranger = _thread(session, clock, other, "stranger")
    mine = _thread(session, clock, item, "mine")

    with pytest.raises(LedgerError):
        ledger.accept(
            session,
            clock,
            item_id=item.id,
            thread_id=stranger.id,
            buyer_id=stranger.buyer_id,
            price_cents=9_000,
            undo_minutes=10,
        )
    with pytest.raises(LedgerError):
        ledger.accept(
            session,
            clock,
            item_id=item.id,
            thread_id=mine.id,
            buyer_id=stranger.buyer_id,
            price_cents=9_000,
            undo_minutes=10,
        )


def test_accept_reads_the_current_item_not_a_stale_copy(engine, clock):
    """Another session sold the item after this one loaded it."""
    with Session(engine, expire_on_commit=False) as setup:
        item = _item(setup, clock)
        thread = _thread(setup, clock, item, "alice")
        setup.commit()
    with Session(engine, expire_on_commit=False) as stale:
        stale.get(Item, item.id)
        with Session(engine) as other:
            other.get(Item, item.id).status = ItemStatus.SOLD
            other.commit()
        with pytest.raises(LedgerConflict, match="sold"):
            _accept(stale, clock, thread)


def test_ledger_writes_keep_the_callers_unflushed_edits(session, clock):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "alice")
    session.commit()
    thread.last_outbound_at = clock.now()
    item.title = "Oak desk"

    _accept(session, clock, thread)

    assert (thread.last_outbound_at, item.title) == (clock.now(), "Oak desk")
    assert (thread.state, item.status) == (ThreadState.COMMITTED, ItemStatus.COMMITTED)


# ---------------------------------------------------------------- concurrency


def _race(
    engine, attempts: list[Callable[[Session], object]], hold_seconds: float = 0.3
) -> list[str]:
    """Run attempts in parallel threads, each in its own session, released together by a barrier.
    A successful attempt holds its uncommitted write for hold_seconds before committing, so the
    other attempt validates while the winner's row is not yet visible."""
    barrier = threading.Barrier(len(attempts))
    outcomes: list[str] = ["unfinished"] * len(attempts)

    def run(index: int) -> None:
        with Session(engine, expire_on_commit=False) as session:
            session.exec(select(Item)).all()  # load a stale view before the race
            barrier.wait()
            try:
                attempts[index](session)
                time.sleep(hold_seconds)
                session.commit()
                outcomes[index] = "won"
            except LedgerConflict:
                session.rollback()
                outcomes[index] = "conflict"
            except Exception as exc:  # any other failure must fail the test
                session.rollback()
                outcomes[index] = repr(exc)

    workers = [threading.Thread(target=run, args=(i,)) for i in range(len(attempts))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    return outcomes


@pytest.fixture
def file_engine(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'race.db'}")
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.mark.parametrize("round_", range(3))
def test_two_threads_racing_to_accept_one_item_admit_exactly_one(file_engine, clock, round_):
    with Session(file_engine, expire_on_commit=False) as setup:
        item = _item(setup, clock)
        threads = [_thread(setup, clock, item, name) for name in ("alice", "bob")]
        setup.commit()

    outcomes = _race(file_engine, [lambda s, t=t: _accept(s, clock, t) for t in threads])

    assert sorted(outcomes) == ["conflict", "won"]
    with Session(file_engine) as check:
        accepted = check.exec(
            select(Commitment).where(Commitment.kind == CommitmentKind.ACCEPTED)
        ).all()
        assert len(accepted) == 1
        winner = threads[outcomes.index("won")]
        assert accepted[0].thread_id == winner.id
        assert check.get(Item, item.id).status == ItemStatus.COMMITTED
        loser = threads[outcomes.index("conflict")]
        assert check.get(Thread, loser.id).state == ThreadState.NEGOTIATING


def test_two_threads_racing_to_book_overlapping_pickups_admit_exactly_one(file_engine, clock):
    with Session(file_engine, expire_on_commit=False) as setup:
        accepted = []
        for name in ("alice", "bob"):
            item = _item(setup, clock)
            accepted.append(_accept(setup, clock, _thread(setup, clock, item, name)))
        setup.commit()

    outcomes = _race(
        file_engine, [lambda s, a=a: _book(s, clock, a, hours_ahead=24) for a in accepted]
    )

    assert sorted(outcomes) == ["conflict", "won"]
    with Session(file_engine) as check:
        assert len(ledger.active_pickups(check)) == 1


def test_undo_racing_complete_lets_only_one_close_the_commitment(file_engine, clock):
    with Session(file_engine, expire_on_commit=False) as setup:
        item = _item(setup, clock)
        accepted = _accept(setup, clock, _thread(setup, clock, item, "alice"))
        setup.commit()

    outcomes = _race(
        file_engine,
        [
            lambda s: ledger.undo(s, clock, accepted_id=accepted.id),
            lambda s: ledger.complete(s, clock, accepted_id=accepted.id),
        ],
    )

    assert sorted(outcomes) == ["conflict", "won"]
    with Session(file_engine) as check:
        closing = check.exec(select(Commitment).where(Commitment.ref_id == accepted.id)).all()
        assert len(closing) == 1


# ---------------------------------------------------------------- pickups


def test_overlapping_pickups_across_items_conflict(session, clock):
    first = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    second = _accept(session, clock, _thread(session, clock, _item(session, clock), "b"))
    _book(session, clock, first, hours_ahead=24, minutes=60)

    for offset in (0, 0.5, -0.5, 0.25):
        with pytest.raises(LedgerConflict, match="overlaps"):
            _book(session, clock, second, hours_ahead=24 + offset, minutes=60)
    with pytest.raises(LedgerConflict):  # a long booking that swallows the first one
        _book(session, clock, second, hours_ahead=23, minutes=180)
    assert ledger.active_commitment(session, second.item_id).pickup is None


def test_back_to_back_pickups_do_not_overlap(session, clock):
    first = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    second = _accept(session, clock, _thread(session, clock, _item(session, clock), "b"))
    third = _accept(session, clock, _thread(session, clock, _item(session, clock), "c"))
    _book(session, clock, first, hours_ahead=24, minutes=60)

    _book(session, clock, second, hours_ahead=25, minutes=60)
    _book(session, clock, third, hours_ahead=23, minutes=60)

    assert len(ledger.active_pickups(session)) == 3


def test_rescheduling_your_own_pickup_does_not_conflict_with_itself(session, clock):
    accepted = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    _book(session, clock, accepted, hours_ahead=24, minutes=60)

    moved = _book(session, clock, accepted, hours_ahead=24.5, minutes=60)

    active = ledger.active_commitment(session, accepted.item_id)
    assert active.pickup.id == moved.id
    assert _count(session, CommitmentKind.PICKUP_BOOKED) == 2
    assert [p.pickup.id for p in ledger.active_pickups(session)] == [moved.id]


def test_only_the_latest_booking_blocks_the_calendar(session, clock):
    first = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    second = _accept(session, clock, _thread(session, clock, _item(session, clock), "b"))
    _book(session, clock, first, hours_ahead=24)
    _book(session, clock, first, hours_ahead=48)

    _book(session, clock, second, hours_ahead=24)  # the slot first moved away from

    with pytest.raises(LedgerConflict):
        _book(session, clock, second, hours_ahead=48)


def test_a_released_commitment_frees_its_pickup_slot(session, clock):
    first = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    second = _accept(session, clock, _thread(session, clock, _item(session, clock), "b"))
    _book(session, clock, first, hours_ahead=24)
    ledger.release(session, clock, accepted_id=first.id, reason="no show")

    _book(session, clock, second, hours_ahead=24)

    assert [p.accepted.id for p in ledger.active_pickups(session)] == [second.id]


def test_pickup_needs_a_real_aware_interval(session, clock):
    accepted = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    start = clock.now() + timedelta(days=1)

    with pytest.raises(LedgerError):
        ledger.book_pickup(session, clock, accepted_id=accepted.id, start=start, end=start)
    with pytest.raises(LedgerError):
        ledger.book_pickup(
            session, clock, accepted_id=accepted.id, start=start, end=start - timedelta(minutes=1)
        )
    with pytest.raises(LedgerError, match="timezone"):
        naive = start.replace(tzinfo=None)
        ledger.book_pickup(
            session, clock, accepted_id=accepted.id, start=naive, end=naive + timedelta(hours=1)
        )


def test_pickup_requires_an_active_accepted_row(session, clock):
    accepted = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    booked = _book(session, clock, accepted, hours_ahead=24)
    ledger.release(session, clock, accepted_id=accepted.id, reason="changed mind")

    with pytest.raises(LedgerConflict):
        _book(session, clock, accepted, hours_ahead=30)
    with pytest.raises(LedgerConflict, match="not an accepted offer"):
        ledger.book_pickup(
            session,
            clock,
            accepted_id=booked.id,
            start=clock.now() + timedelta(days=3),
            end=clock.now() + timedelta(days=3, hours=1),
        )


# ---------------------------------------------------------------- closing events


def test_undo_inside_the_window_relists(session, clock):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "alice")
    accepted = _accept(session, clock, thread, undo_minutes=10)
    clock.advance(minutes=10)  # exactly at the deadline still counts

    undone = ledger.undo(session, clock, accepted_id=accepted.id)

    assert undone.kind == CommitmentKind.UNDONE and undone.ref_id == accepted.id
    assert undone.actor == "human"
    assert item.status == ItemStatus.LISTED
    assert thread.state == ThreadState.NEGOTIATING
    assert ledger.active_commitment(session, item.id) is None


def test_undo_after_the_deadline_raises_and_changes_nothing(session, clock):
    item = _item(session, clock)
    accepted = _accept(session, clock, _thread(session, clock, item, "alice"), undo_minutes=10)
    clock.advance(minutes=10, seconds=1)

    with pytest.raises(UndoExpired):
        ledger.undo(session, clock, accepted_id=accepted.id)

    assert item.status == ItemStatus.COMMITTED
    assert ledger.is_active(session, accepted)
    assert _count(session, CommitmentKind.UNDONE) == 0


def test_release_relists_and_keeps_the_reason(session, clock):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "alice")
    accepted = _accept(session, clock, thread)

    released = ledger.release(
        session, clock, accepted_id=accepted.id, reason="no show", actor="agent"
    )

    assert (released.kind, released.reason, released.ref_id) == (
        CommitmentKind.RELEASED,
        "no show",
        accepted.id,
    )
    assert item.status == ItemStatus.LISTED
    assert thread.state == ThreadState.NEGOTIATING


def test_complete_sells_closes_and_credits_the_buyer(session, clock):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "alice")
    accepted = _accept(session, clock, thread, price=9_500)
    _book(session, clock, accepted, hours_ahead=2)

    done = ledger.complete(session, clock, accepted_id=accepted.id)

    assert done.kind == CommitmentKind.COMPLETED and done.price_cents == 9_500
    assert item.status == ItemStatus.SOLD
    assert thread.state == ThreadState.CLOSED
    assert session.get(Buyer, thread.buyer_id).completed_count == 1
    assert ledger.active_pickups(session) == []


@pytest.mark.parametrize(
    "close",
    [
        lambda s, c, a: ledger.release(s, c, accepted_id=a, reason="again"),
        lambda s, c, a: ledger.undo(s, c, accepted_id=a),
        lambda s, c, a: ledger.complete(s, c, accepted_id=a),
    ],
    ids=["release", "undo", "complete"],
)
@pytest.mark.parametrize(
    "first",
    [
        lambda s, c, a: ledger.release(s, c, accepted_id=a, reason="first"),
        lambda s, c, a: ledger.undo(s, c, accepted_id=a),
        lambda s, c, a: ledger.complete(s, c, accepted_id=a),
    ],
    ids=["released", "undone", "completed"],
)
def test_a_closed_commitment_accepts_no_further_events(session, clock, first, close):
    item = _item(session, clock)
    accepted = _accept(session, clock, _thread(session, clock, item, "alice"))
    first(session, clock, accepted.id)
    status_after_first = item.status

    with pytest.raises(LedgerConflict):
        close(session, clock, accepted.id)

    assert item.status == status_after_first
    assert len(ledger.item_history(session, item.id)) == 2


@pytest.mark.parametrize("status", [ItemStatus.SOLD, ItemStatus.WITHDRAWN])
@pytest.mark.parametrize("close", [ledger.undo, ledger.release])
def test_undoing_a_deal_never_puts_a_sold_or_withdrawn_item_back_on_sale(
    session, clock, status, close
):
    """A deal can outlive its item's sale (an accept that raced the owner's `agent sold` before
    close_item); undoing it later must not relist the item."""
    item = _item(session, clock)
    accepted = _accept(session, clock, _thread(session, clock, item, "maya"))
    item.status = status
    session.flush()

    extra = {"reason": "the owner sold it"} if close is ledger.release else {}
    close(session, clock, accepted_id=accepted.id, **extra)

    assert item.status == status
    assert ledger.active_commitment(session, item.id) is None


@pytest.mark.parametrize(
    ("status", "closing"),
    [(ItemStatus.SOLD, CommitmentKind.COMPLETED), (ItemStatus.WITHDRAWN, CommitmentKind.RELEASED)],
)
def test_close_item_closes_the_active_deal_with_it(session, clock, status, closing):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "maya")
    accepted = _accept(session, clock, thread)

    closed = ledger.close_item(session, clock, item_id=item.id, status=status, reason="owner")

    assert closed is not None and closed.accepted.id == accepted.id
    assert item.status == status and _count(session, closing) == 1
    assert ledger.active_commitment(session, item.id) is None
    with pytest.raises(LedgerConflict, match="already"):
        ledger.close_item(session, clock, item_id=item.id, status=status)


def test_close_item_without_a_deal_just_closes_the_item(session, clock):
    item = _item(session, clock)
    assert ledger.close_item(session, clock, item_id=item.id, status=ItemStatus.SOLD) is None
    assert item.status == ItemStatus.SOLD and session.exec(select(Commitment)).all() == []
    with pytest.raises(LedgerError):
        ledger.close_item(session, clock, item_id=item.id, status=ItemStatus.LISTED)


def test_a_frozen_thread_stays_frozen_through_release(session, clock):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "alice")
    accepted = _accept(session, clock, thread)
    thread.state = ThreadState.FROZEN  # buyer turned out to be a scammer after the deal
    session.flush()

    ledger.release(session, clock, accepted_id=accepted.id, reason="scam")

    assert thread.state == ThreadState.FROZEN
    assert item.status == ItemStatus.LISTED


def test_closing_events_reference_only_accepted_rows(session, clock):
    accepted = _accept(session, clock, _thread(session, clock, _item(session, clock), "a"))
    booked = _book(session, clock, accepted, hours_ahead=24)

    assert not ledger.is_active(session, booked)
    for close in (ledger.complete, ledger.undo):
        with pytest.raises(LedgerConflict):
            close(session, clock, accepted_id=booked.id)
    with pytest.raises(LedgerConflict):
        ledger.release(session, clock, accepted_id=999_999, reason="missing")


def test_item_history_is_chronological(session, clock):
    item = _item(session, clock)
    alice = _thread(session, clock, item, "alice")
    first = _accept(session, clock, alice)
    clock.advance(minutes=1)
    _book(session, clock, first, hours_ahead=24)
    clock.advance(minutes=1)
    ledger.release(session, clock, accepted_id=first.id, reason="x")
    clock.advance(minutes=1)
    _accept(session, clock, _thread(session, clock, item, "bob"))

    kinds = [row.kind for row in ledger.item_history(session, item.id)]

    assert kinds == [
        CommitmentKind.ACCEPTED,
        CommitmentKind.PICKUP_BOOKED,
        CommitmentKind.RELEASED,
        CommitmentKind.ACCEPTED,
    ]


# ---------------------------------------------------------------- append-only

_COMMITMENT_REWRITE = re.compile(
    r"^\s*(?:UPDATE\s+(?:OR\s+\w+\s+)?\"?commitment\"?\b|DELETE\s+FROM\s+\"?commitment\"?\b"
    r"|(?:INSERT\s+OR\s+REPLACE|REPLACE)\s+INTO\s+\"?commitment\"?\b)",
    re.IGNORECASE,
)
_COMMITMENT_INSERT = re.compile(r"^\s*INSERT\s+INTO\s+\"?commitment\"?\b", re.IGNORECASE)


def test_ledger_never_updates_or_deletes_a_commitment_row(engine, session, clock):
    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    def rows() -> list[tuple]:
        return [
            tuple(r)
            for r in session.connection().execute(text("SELECT * FROM commitment ORDER BY id"))
        ]

    item = _item(session, clock)
    alice = _thread(session, clock, item, "alice")
    bob = _thread(session, clock, item, "bob")
    other = _accept(session, clock, _thread(session, clock, _item(session, clock), "carol"))
    _book(session, clock, other, hours_ahead=24)
    session.commit()

    event.listen(engine, "before_cursor_execute", capture)
    history: list[list[tuple]] = [rows()]
    try:
        steps = [
            lambda: _accept(session, clock, alice),
            lambda: _book(session, clock, ledger.active_commitment(session, item.id).accepted, 30),
            lambda: _book(session, clock, ledger.active_commitment(session, item.id).accepted, 31),
            lambda: ledger.release(
                session,
                clock,
                accepted_id=ledger.active_commitment(session, item.id).accepted.id,
                reason="no show",
            ),
            lambda: _accept(session, clock, bob),
            lambda: ledger.undo(
                session, clock, accepted_id=ledger.active_commitment(session, item.id).accepted.id
            ),
            lambda: _accept(session, clock, alice),
            lambda: ledger.complete(
                session, clock, accepted_id=ledger.active_commitment(session, item.id).accepted.id
            ),
            lambda: ledger.complete(session, clock, accepted_id=other.id),
        ]
        for step in steps:
            step()
            session.commit()
            history.append(rows())
        with pytest.raises(LedgerConflict):
            _accept(session, clock, bob)
        session.rollback()
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert [s for s in statements if _COMMITMENT_REWRITE.match(s)] == []
    assert len([s for s in statements if _COMMITMENT_INSERT.match(s)]) == len(steps)
    # The listener does see updates, just none to the commitment table.
    assert any(s.lstrip().upper().startswith("UPDATE ITEM") for s in statements)
    for before, after in pairwise(history):
        assert after[: len(before)] == before
        assert len(after) == len(before) + 1
