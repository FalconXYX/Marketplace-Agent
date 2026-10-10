"""Bundles: allocation, the bundle ledger, accept_bundle and the whole-bundle deal tools.

FakeClock starts Thu 2026-10-01 16:00 UTC, which is 12:00 in Toronto. Thursday pickups run
17:00-20:00 local (21:00-24:00 UTC).

| Item | List | Floor | Firmness |
|---|---|---|---|
| desk | $120 | $95 | slight |
| chair | $50 | $30 | slight |
| lamp | $20 | $15 | firm |

Ann has a thread on the desk (where deals are made) and on the chair, none on the lamp.
"""
import itertools
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

import pytest
from hypothesis import example, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from pydantic import ValidationError
from sqlalchemy import Engine, event
from sqlmodel import Session, col, select

from marketplace_agent.control.service import ControlService
from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.db import init_db, make_engine
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Firmness,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Memory,
    Message,
    Thread,
    ThreadState,
    TrustRecord,
    WakeRequest,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.ledger import LedgerConflict, LedgerError, UndoExpired
from marketplace_agent.guard.validate import validate
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import ListingStats, PlatformError, PlatformRef
from marketplace_agent.sim.audit import audit
from marketplace_agent.sim.harness import sim_settings
from marketplace_agent.tools.base import Allow, Reject, ToolContext
from marketplace_agent.tools.impl.deals import (
    AcceptBundleArgs,
    AcceptBundleTool,
    AcceptOfferArgs,
    AcceptOfferTool,
    BookPickupArgs,
    BookPickupTool,
    CompleteSaleTool,
    ReleaseCommitmentArgs,
    ReleaseCommitmentTool,
    ThreadArgs,
)
from marketplace_agent.tools.registry import default_registry

BUNDLE, ACCEPT, BOOK, RELEASE, COMPLETE = (
    AcceptBundleTool(),
    AcceptOfferTool(),
    BookPickupTool(),
    ReleaseCommitmentTool(),
    CompleteSaleTool(),
)
PICKUP_START = datetime(2026, 10, 1, 22, tzinfo=UTC)  # Thu 18:00 local
WAIT = 0.3  # long enough for the other side to reach its blocking point


# ---------------------------------------------------------------- world


def _item(
    session: Session,
    clock: FakeClock,
    title: str,
    list_cents: int,
    floor_cents: int,
    firmness: Firmness = Firmness.SLIGHT,
) -> Item:
    now = clock.now()
    item = Item(title=title, status=ItemStatus.LISTED, created_at=now, updated_at=now)
    session.add(item)
    session.flush()
    set_goal(
        session,
        clock,
        item.id,
        list_price_cents=list_cents,
        floor_price_cents=floor_cents,
        firmness=firmness,
    )
    return item


def _buyer(session: Session, handle: str) -> Buyer:
    buyer = Buyer(platform="fake", handle=handle, display_name=handle.title())
    session.add(buyer)
    session.flush()
    return buyer


def _thread(
    session: Session, clock: FakeClock, item: Item, buyer: Buyer, *, wrote: bool = True
) -> Thread:
    thread = Thread(
        platform="fake",
        remote_id=f"rt-{item.id}-{buyer.handle}",
        item_id=item.id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    if wrote:
        session.add(
            Message(
                thread_id=thread.id,
                direction=Direction.INBOUND,
                body="Would you do $150 for the desk and the chair?",
                sent_at=clock.now() - timedelta(minutes=5),
                via="buyer",
            )
        )
        session.flush()
    return thread


@dataclass
class Shop:
    desk: Item
    chair: Item
    lamp: Item
    ann: Buyer
    desk_thread: Thread
    chair_thread: Thread

    @property
    def ids(self) -> list[int]:
        return [self.desk.id, self.chair.id]


def _shop(session: Session, clock: FakeClock) -> Shop:
    desk = _item(session, clock, "IKEA desk", 12_000, 9_500)
    chair = _item(session, clock, "Office chair", 5_000, 3_000)
    lamp = _item(session, clock, "Brass lamp", 2_000, 1_500, Firmness.FIRM)
    ann = _buyer(session, "ann")
    shop = Shop(
        desk,
        chair,
        lamp,
        ann,
        _thread(session, clock, desk, ann),
        _thread(session, clock, chair, ann, wrote=False),
    )
    session.commit()
    return shop


@pytest.fixture
def shop(session, clock) -> Shop:
    return _shop(session, clock)


def _bundle(
    session: Session,
    clock: FakeClock,
    shop: Shop,
    item_ids: list[int] | None = None,
    total: int = 15_000,
    undo_minutes: int = 10,
) -> list[Commitment]:
    rows = ledger.accept_bundle(
        session,
        clock,
        item_ids=item_ids or shop.ids,
        thread_id=shop.desk_thread.id,
        buyer_id=shop.ann.id,
        total_cents=total,
        undo_minutes=undo_minutes,
    )
    session.commit()
    return rows


def _book(
    session: Session, clock: FakeClock, accepted: Commitment, start: datetime = PICKUP_START
) -> Commitment:
    booked = ledger.book_pickup(
        session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(hours=1)
    )
    session.commit()
    return booked


def _events(session: Session, accepted: Commitment) -> list[Commitment]:
    query = select(Commitment).where(Commitment.ref_id == accepted.id).order_by(Commitment.id)
    return list(session.exec(query).all())


def _all_rows(session: Session) -> list[Commitment]:
    return list(session.exec(select(Commitment).order_by(Commitment.id)).all())


# ---------------------------------------------------------------- allocate


def test_allocate_examples():
    assert ledger.allocate(100, [1, 1, 1]) == [34, 33, 33]
    assert ledger.allocate(15_000, [12_000, 5_000]) == [10_588, 4_412]
    assert ledger.allocate(1, [1, 1]) == [1, 0]  # a tie goes to the earlier weight
    assert ledger.allocate(0, [3, 4]) == [0, 0]
    assert ledger.allocate(999, [0, 5]) == [0, 999]


@pytest.mark.parametrize(
    ("total", "weights"), [(-1, [1]), (10, []), (10, [1, -1]), (10, [0, 0])]
)
def test_allocate_refuses_nonsense(total, weights):
    with pytest.raises(ValueError):
        ledger.allocate(total, weights)


@given(
    total=st.integers(0, 10**9),
    weights=st.lists(st.integers(0, 10**7), min_size=1, max_size=6).filter(lambda w: sum(w) > 0),
)
def test_allocate_sums_exactly_and_rounds_each_share_by_under_a_cent(total, weights):
    shares = ledger.allocate(total, weights)
    whole = sum(weights)
    assert sum(shares) == total
    for share, weight in zip(shares, weights, strict=True):
        exact_floor = total * weight // whole
        assert share in (exact_floor, exact_floor + 1)
        assert abs(share * whole - total * weight) < whole  # within one cent of exact
    assert all(share == 0 for share, weight in zip(shares, weights, strict=True) if weight == 0)


@given(
    total=st.integers(0, 10**8),
    weights=st.lists(st.integers(1, 10**6), min_size=2, max_size=5),
)
def test_allocate_follows_the_weights(total, weights):
    shares = ledger.allocate(total, weights)
    for (s1, w1), (s2, w2) in itertools.combinations(zip(shares, weights, strict=True), 2):
        if w1 > w2:
            assert s1 >= s2
        if w1 == w2:
            assert abs(s1 - s2) <= 1


@st.composite
def _bundle_terms(draw) -> tuple[int, list[int], list[int]]:
    lists = draw(st.lists(st.integers(1, 10**6), min_size=2, max_size=5))
    minimums = [draw(st.integers(1, price)) for price in lists]
    total = draw(st.integers(sum(minimums), sum(lists) * 2))
    return total, lists, minimums


@given(terms=_bundle_terms())
def test_split_bundle_sums_exactly_and_respects_every_minimum(terms):
    total, lists, minimums = terms
    prices = ledger.split_bundle(total, lists, minimums)
    assert sum(prices) == total
    assert all(price >= low for price, low in zip(prices, minimums, strict=True))
    proportional = ledger.allocate(total, lists)
    if all(share >= low for share, low in zip(proportional, minimums, strict=True)):
        assert prices == proportional
    # Above its minimum, an item never gets more per list cent than any other item, beyond
    # a cent of rounding: raising one to its minimum only takes from the rest.
    for (p1, l1, m1), (p2, l2, _) in itertools.permutations(zip(prices, lists, minimums), 2):
        if p1 > m1:
            assert (p1 - 1) * l2 <= (p2 + 1) * l1


def test_split_bundle_raises_a_short_share_to_its_minimum():
    # $130 by list price would give the desk $91.76, under its $95 floor.
    assert ledger.split_bundle(13_000, [12_000, 5_000], [9_500, 3_000]) == [9_500, 3_500]
    assert ledger.split_bundle(12_500, [12_000, 5_000], [9_500, 3_000]) == [9_500, 3_000]
    with pytest.raises(LedgerError, match="below the 12500 cents"):
        ledger.split_bundle(12_499, [12_000, 5_000], [9_500, 3_000])


# ---------------------------------------------------------------- ledger: accept


def test_accept_bundle_promises_every_item_under_one_key(session, clock, shop):
    rows = _bundle(session, clock, shop)

    assert [row.item_id for row in rows] == shop.ids
    assert [row.price_cents for row in rows] == [10_588, 4_412]
    assert {row.kind for row in rows} == {CommitmentKind.ACCEPTED}
    key = rows[0].bundle_key
    assert key is not None and key.startswith("bundle-") and rows[1].bundle_key == key
    assert {row.buyer_id for row in rows} == {shop.ann.id}
    assert {row.undo_deadline for row in rows} == {clock.now() + timedelta(minutes=10)}
    assert all("for 15000 cents in total" in row.reason for row in rows)
    # Each row sits on Ann's own thread about its item.
    assert [row.thread_id for row in rows] == [shop.desk_thread.id, shop.chair_thread.id]
    assert shop.desk.status == shop.chair.status == ItemStatus.COMMITTED
    assert shop.desk_thread.state == shop.chair_thread.state == ThreadState.COMMITTED
    assert shop.lamp.status == ItemStatus.LISTED
    assert ledger.active_commitment(session, shop.desk.id).accepted.id == rows[0].id
    assert ledger.active_commitment(session, shop.chair.id).accepted.id == rows[1].id
    assert ledger.bundle_rows(session, key) == rows
    assert ledger.deal_rows(session, rows[1]) == rows


def test_a_row_without_its_own_thread_goes_on_the_deal_thread(session, clock, shop):
    rows = _bundle(session, clock, shop, [shop.lamp.id, shop.desk.id], total=13_000)

    assert [row.thread_id for row in rows] == [shop.desk_thread.id, shop.desk_thread.id]
    assert [row.price_cents for row in rows] == [2_000, 11_000]  # lamp firm at $20
    # Both rows are on the desk thread; the desk's own row answers for it.
    in_thread = ledger.active_commitment_for_thread(session, shop.desk_thread.id)
    assert in_thread.accepted.item_id == shop.desk.id
    assert ledger.active_commitment_for_thread(session, shop.chair_thread.id) is None


def test_bundle_keys_are_unique_even_at_one_instant(session, clock, shop):
    first = _bundle(session, clock, shop)
    ledger.undo(session, clock, accepted_id=first[0].id)
    second = _bundle(session, clock, shop)

    assert first[0].bundle_key != second[0].bundle_key
    assert ledger.bundle_rows(session, second[0].bundle_key) == second


def _promise_chair_to_bob(session: Session, clock: FakeClock, shop: Shop) -> Commitment:
    bob = _buyer(session, "bob")
    return ledger.accept(
        session,
        clock,
        item_id=shop.chair.id,
        thread_id=_thread(session, clock, shop.chair, bob).id,
        buyer_id=bob.id,
        price_cents=5_000,
        undo_minutes=10,
    )


def _rug(session: Session, clock: FakeClock) -> Item:
    """A listed item with no goal."""
    now = clock.now()
    rug = Item(title="Rug", status=ItemStatus.LISTED, created_at=now, updated_at=now)
    session.add(rug)
    session.flush()
    return rug


Terms = dict[str, Any]
REFUSALS: dict[str, tuple[Callable[[Session, FakeClock, Shop, Terms], None], type, str]] = {
    "one item": (lambda s, c, shop, t: t.update(item_ids=[shop.desk.id]), LedgerError, "not 1"),
    "six items": (lambda s, c, shop, t: t.update(item_ids=[1, 2, 3, 4, 5, 6]), LedgerError, "6"),
    "an item twice": (
        lambda s, c, shop, t: t.update(item_ids=[shop.desk.id, shop.desk.id]),
        LedgerError,
        "only once",
    ),
    "negative undo": (lambda s, c, shop, t: t.update(undo_minutes=-1), LedgerError, "negative"),
    "a promised item": (
        lambda s, c, shop, t: _promise_chair_to_bob(s, c, shop),
        LedgerConflict,
        "already promised",
    ),
    "a sold item": (
        lambda s, c, shop, t: setattr(shop.chair, "status", ItemStatus.SOLD),
        LedgerConflict,
        "sold",
    ),
    "a thread about another item": (
        lambda s, c, shop, t: t.update(thread_id=_thread(s, c, shop.lamp, shop.ann).id),
        LedgerError,
        "not buyer",
    ),
    "another buyer": (
        lambda s, c, shop, t: t.update(buyer_id=_buyer(s, "bob").id),
        LedgerError,
        "not buyer",
    ),
    "a frozen deal thread": (
        lambda s, c, shop, t: setattr(shop.desk_thread, "state", ThreadState.FROZEN),
        LedgerConflict,
        "frozen",
    ),
    "a frozen row thread": (
        lambda s, c, shop, t: setattr(shop.chair_thread, "state", ThreadState.FROZEN),
        LedgerConflict,
        "frozen",
    ),
    "a flagged buyer": (
        lambda s, c, shop, t: setattr(shop.ann, "flagged_scam", True),
        LedgerConflict,
        "flagged",
    ),
    "a total under the floors": (
        lambda s, c, shop, t: t.update(total_cents=12_499),
        LedgerError,
        "below",
    ),
    "a total under a firm price": (
        lambda s, c, shop, t: t.update(item_ids=[shop.desk.id, shop.lamp.id], total_cents=11_499),
        LedgerError,
        "below",
    ),
    "an item without a goal": (
        lambda s, c, shop, t: t.update(item_ids=[shop.desk.id, _rug(s, c).id]),
        LedgerError,
        "no price goal",
    ),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_accept_bundle_refusals_append_nothing(session, clock, shop, case):
    setup, error, match = REFUSALS[case]
    terms: Terms = {
        "item_ids": shop.ids,
        "thread_id": shop.desk_thread.id,
        "buyer_id": shop.ann.id,
        "total_cents": 15_000,
        "undo_minutes": 10,
    }
    setup(session, clock, shop, terms)
    session.commit()
    before = _all_rows(session)

    with pytest.raises(error, match=match):
        ledger.accept_bundle(session, clock, **terms)
    session.rollback()

    assert _all_rows(session) == before
    assert session.get(Item, shop.desk.id).status == ItemStatus.LISTED


def test_a_firm_item_is_held_to_its_list_price_inside_a_bundle(session, clock, shop):
    # $115 is the desk's floor plus the lamp's firm price; less is refused.
    rows = _bundle(session, clock, shop, [shop.desk.id, shop.lamp.id], total=11_500)
    assert [row.price_cents for row in rows] == [9_500, 2_000]


# ---------------------------------------------------------------- ledger: events


def test_book_pickup_books_every_row_at_one_slot(session, clock, shop):
    desk_row, chair_row = _bundle(session, clock, shop)

    booked = _book(session, clock, chair_row)

    assert booked.ref_id == chair_row.id and booked.bundle_key == chair_row.bundle_key
    for row in (desk_row, chair_row):
        pickup = ledger.active_commitment(session, row.item_id).pickup
        assert pickup is not None and pickup.ref_id == row.id
        assert (pickup.pickup_start, pickup.pickup_end) == (
            PICKUP_START,
            PICKUP_START + timedelta(hours=1),
        )
    assert len(ledger.active_pickups(session)) == 2

    later = PICKUP_START + timedelta(minutes=30)  # overlaps its own slot: a reschedule
    _book(session, clock, desk_row, later)
    for row in (desk_row, chair_row):
        assert ledger.active_commitment(session, row.item_id).pickup.pickup_start == later


def test_a_bundle_pickup_cannot_overlap_another_deal(session, clock, shop):
    bob = _buyer(session, "bob")
    lamp_deal = ledger.accept(
        session, clock, item_id=shop.lamp.id, thread_id=_thread(session, clock, shop.lamp, bob).id,
        buyer_id=bob.id, price_cents=2_000, undo_minutes=10,
    )
    _book(session, clock, lamp_deal)
    desk_row, _ = _bundle(session, clock, shop)
    before = _all_rows(session)

    with pytest.raises(LedgerConflict, match="overlaps another booked pickup"):
        _book(session, clock, desk_row, PICKUP_START + timedelta(minutes=30))
    session.rollback()

    assert _all_rows(session) == before


@pytest.mark.parametrize(
    ("close", "kind", "status", "state"),
    [
        ("release", CommitmentKind.RELEASED, ItemStatus.LISTED, ThreadState.NEGOTIATING),
        ("undo", CommitmentKind.UNDONE, ItemStatus.LISTED, ThreadState.NEGOTIATING),
        ("complete", CommitmentKind.COMPLETED, ItemStatus.SOLD, ThreadState.CLOSED),
    ],
)
def test_closing_any_row_closes_the_whole_bundle(session, clock, shop, close, kind, status, state):
    desk_row, chair_row = _bundle(session, clock, shop)
    _book(session, clock, desk_row)
    clock.advance(minutes=5)
    extra = {"reason": "changed plans"} if close == "release" else {}

    closing = getattr(ledger, close)(session, clock, accepted_id=chair_row.id, **extra)
    session.commit()

    assert closing.ref_id == chair_row.id and closing.kind == kind
    for row in (desk_row, chair_row):
        events = _events(session, row)
        assert [event.kind for event in events] == [CommitmentKind.PICKUP_BOOKED, kind]
        assert events[-1].created_at == clock.now()
        assert events[-1].bundle_key == row.bundle_key
        assert ledger.active_commitment(session, row.item_id) is None
    assert shop.desk.status == shop.chair.status == status
    assert shop.desk_thread.state == shop.chair_thread.state == state
    assert ledger.active_pickups(session) == []
    assert shop.ann.completed_count == (1 if close == "complete" else 0)  # one deal, not two


def test_a_bundle_cannot_be_undone_after_its_window(session, clock, shop):
    desk_row, _ = _bundle(session, clock, shop)
    clock.advance(minutes=11)
    before = _all_rows(session)

    with pytest.raises(UndoExpired):
        ledger.undo(session, clock, accepted_id=desk_row.id)
    session.rollback()

    assert _all_rows(session) == before


def test_a_closed_bundle_takes_no_more_events_and_frees_its_items(session, clock, shop):
    desk_row, chair_row = _bundle(session, clock, shop)
    ledger.release(session, clock, accepted_id=desk_row.id, reason="backed out")
    session.commit()

    with pytest.raises(LedgerConflict):
        ledger.book_pickup(
            session, clock, accepted_id=chair_row.id, start=PICKUP_START,
            end=PICKUP_START + timedelta(hours=1),
        )
    with pytest.raises(LedgerConflict):
        ledger.complete(session, clock, accepted_id=chair_row.id)
    session.rollback()

    single = _promise_chair_to_bob(session, clock, shop)
    assert ledger.active_commitment(session, shop.chair.id).accepted.id == single.id


@pytest.mark.parametrize(
    ("status", "kind", "other"),
    [
        (ItemStatus.SOLD, CommitmentKind.COMPLETED, ItemStatus.SOLD),
        (ItemStatus.WITHDRAWN, CommitmentKind.RELEASED, ItemStatus.LISTED),
    ],
)
def test_close_item_closes_the_whole_bundle(session, clock, shop, status, kind, other):
    desk_row, chair_row = _bundle(session, clock, shop)

    closed = ledger.close_item(session, clock, item_id=shop.desk.id, status=status)
    session.commit()

    assert closed is not None and closed.accepted.id == desk_row.id
    assert [e.kind for e in _events(session, desk_row)] == [kind]
    assert [e.kind for e in _events(session, chair_row)] == [kind]
    assert (shop.desk.status, shop.chair.status) == (status, other)


_COMMITMENT_REWRITE = re.compile(
    r"^\s*(?:UPDATE\s+\"?commitment\"?\b|DELETE\s+FROM\s+\"?commitment\"?\b|REPLACE\b)",
    re.IGNORECASE,
)


def test_bundle_events_only_append(engine, session, clock, shop):
    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        desk_row, _ = _bundle(session, clock, shop)
        _book(session, clock, desk_row)
        _book(session, clock, desk_row, PICKUP_START + timedelta(hours=1))
        ledger.release(session, clock, accepted_id=desk_row.id, reason="no-show")
        session.commit()
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert [s for s in statements if _COMMITMENT_REWRITE.match(s)] == []
    assert len(_all_rows(session)) == 2 + 2 + 2 + 2


# ---------------------------------------------------------------- ledger: races


@pytest.fixture
def file_engine(tmp_path) -> Iterator[Engine]:
    engine = make_engine(f"sqlite:///{tmp_path / 'bundle.db'}")
    init_db(engine)
    yield engine
    engine.dispose()


class Background:
    """Runs a call on another thread and keeps its result or error."""

    def __init__(self, call: Callable[[], Any]):
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()
        threading.Thread(target=self._run, args=(call,), daemon=True).start()

    def _run(self, call: Callable[[], Any]) -> None:
        try:
            self.result = call()
        except BaseException as exc:  # noqa: BLE001  reported to the test
            self.error = exc
        finally:
            self.done.set()

    def join(self) -> None:
        assert self.done.wait(10), "the background call never finished"


def _race_shop(engine: Engine, clock: FakeClock) -> tuple[Shop, Thread]:
    """The shop, and Bob's thread on the chair."""
    with Session(engine, expire_on_commit=False) as setup:
        shop = _shop(setup, clock)
        bob_thread = _thread(setup, clock, shop.chair, _buyer(setup, "bob"))
        setup.commit()
    return shop, bob_thread


def _bob_takes_the_chair(session: Session, clock: FakeClock, bob_thread: Thread) -> Commitment:
    return ledger.accept(
        session, clock, item_id=bob_thread.item_id, thread_id=bob_thread.id,
        buyer_id=bob_thread.buyer_id, price_cents=5_000, undo_minutes=10,
    )


def _ann_takes_both(session: Session, clock: FakeClock, shop: Shop) -> list[Commitment]:
    return ledger.accept_bundle(
        session, clock, item_ids=shop.ids, thread_id=shop.desk_thread.id, buyer_id=shop.ann.id,
        total_cents=15_000, undo_minutes=10,
    )


def _active_by_item(engine: Engine) -> dict[int, list[Commitment]]:
    with Session(engine) as check:
        rows = check.exec(select(Commitment).where(Commitment.kind == CommitmentKind.ACCEPTED))
        found: dict[int, list[Commitment]] = {}
        for row in rows.all():
            if ledger.is_active(check, row):
                found.setdefault(row.item_id, []).append(row)
        return found


@pytest.mark.parametrize("head_start", ["bundle", "single", None])
def test_bundle_and_single_accept_racing_admit_exactly_one(file_engine, clock, head_start):
    """Released together; with a head start, the other side validates while the leader's
    rows are written but not committed."""
    shop, bob_thread = _race_shop(file_engine, clock)
    barrier = threading.Barrier(2)
    outcomes = ["unfinished", "unfinished"]
    lagging = {"bundle": 1, "single": 0}.get(head_start or "")

    def run(index: int, attempt: Callable[[Session], object]) -> None:
        with Session(file_engine, expire_on_commit=False) as session:
            session.exec(select(Item)).all()  # a stale view before the race
            barrier.wait()
            if index == lagging:
                time.sleep(WAIT / 3)
            try:
                attempt(session)
                time.sleep(WAIT)  # the winner's rows stay uncommitted meanwhile
                session.commit()
                outcomes[index] = "won"
            except LedgerConflict:
                session.rollback()
                outcomes[index] = "conflict"
            except Exception as exc:  # any other failure must fail the test
                session.rollback()
                outcomes[index] = repr(exc)

    attempts = [
        partial(_ann_takes_both, clock=clock, shop=shop),
        partial(_bob_takes_the_chair, clock=clock, bob_thread=bob_thread),
    ]
    workers = [
        threading.Thread(target=run, args=(i, attempt)) for i, attempt in enumerate(attempts)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)

    assert sorted(outcomes) == ["conflict", "won"]
    if head_start is not None:
        assert outcomes[1 - lagging] == "won"
    active = _active_by_item(file_engine)
    assert all(len(rows) == 1 for rows in active.values())
    if outcomes[0] == "won":
        assert {item: rows[0].buyer_id for item, rows in active.items()} == {
            shop.desk.id: shop.ann.id,
            shop.chair.id: shop.ann.id,
        }
    else:
        assert {item: rows[0].buyer_id for item, rows in active.items()} == {
            shop.chair.id: bob_thread.buyer_id
        }
        with Session(file_engine) as check:
            assert check.get(Item, shop.desk.id).status == ItemStatus.LISTED
            bundled = select(Commitment).where(col(Commitment.bundle_key).is_not(None))
            assert check.exec(bundled).all() == []


def test_a_single_accept_waits_for_an_open_bundle_then_conflicts(file_engine, clock):
    shop, bob_thread = _race_shop(file_engine, clock)
    with Session(file_engine, expire_on_commit=False) as agent:
        _ann_takes_both(agent, clock, shop)

        def other() -> Commitment:
            with Session(file_engine, expire_on_commit=False) as session:
                return _bob_takes_the_chair(session, clock, bob_thread)

        single = Background(other)
        assert not single.done.wait(WAIT)  # waits for the bundle's write instead of reading past it
        agent.commit()
    single.join()

    assert isinstance(single.error, LedgerConflict)
    assert set(_active_by_item(file_engine)) == {shop.desk.id, shop.chair.id}


def test_a_bundle_waits_for_an_open_single_accept_then_conflicts(file_engine, clock):
    shop, bob_thread = _race_shop(file_engine, clock)
    with Session(file_engine, expire_on_commit=False) as agent:
        _bob_takes_the_chair(agent, clock, bob_thread)

        def other() -> list[Commitment]:
            with Session(file_engine, expire_on_commit=False) as session:
                return _ann_takes_both(session, clock, shop)

        bundle = Background(other)
        assert not bundle.done.wait(WAIT)
        agent.commit()
    bundle.join()

    assert isinstance(bundle.error, LedgerConflict)
    active = _active_by_item(file_engine)
    assert set(active) == {shop.chair.id}
    assert active[shop.chair.id][0].buyer_id == bob_thread.buyer_id


# ---------------------------------------------------------------- tools


class StubAdapter:
    """Records delist calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.delisted: list[str] = []

    def publish(self, listing, *, assist=False):
        return PlatformRef(self.name, f"{self.name}-1")

    def update(self, ref, listing):
        pass

    def delist(self, ref):
        self.delisted.append(ref.remote_id)

    def fetch_threads(self, since):
        return []

    def send_message(self, thread_remote_id, body):
        return None

    def stats(self, ref):
        return ListingStats()

    def search(self, query, limit=20):
        return []

    def healthcheck(self):
        return []


@pytest.fixture
def adapters() -> dict[str, StubAdapter]:
    return {name: StubAdapter(name) for name in ("fake", "kijiji")}


def _ctx(session, clock, settings, notifier, adapters, item: Item, actor="agent") -> ToolContext:
    return ToolContext(
        session=session,
        clock=clock,
        settings=settings,
        platforms=adapters,
        notifier=notifier,
        item_id=item.id,
        actor=actor,
    )


@pytest.fixture
def ctx(session, clock, settings, notifier, adapters, shop) -> ToolContext:
    return _ctx(session, clock, settings, notifier, adapters, shop.desk)


def _args(shop: Shop, item_ids: list[int] | None = None, total: int = 15_000) -> AcceptBundleArgs:
    return AcceptBundleArgs(
        thread_id=shop.desk_thread.id, item_ids=item_ids or shop.ids, total_cents=total
    )


def _reject(tool, ctx: ToolContext, args) -> str:
    verdict = tool.check(ctx, args)
    assert isinstance(verdict, Reject), verdict
    return verdict.reason


def _run(tool, ctx: ToolContext, args):
    assert tool.check(ctx, args) == Allow()
    result = tool.execute(ctx, args)
    ctx.session.commit()
    return result


def _listing(session: Session, item: Item, platform: str) -> Listing:
    listing = Listing(
        item_id=item.id,
        platform=platform,
        remote_id=f"{platform}-r{item.id}",
        url=f"https://{platform}.test/{item.id}",
        status=ListingStatus.LIVE,
        price_cents=0,
        title=item.title,
        description="",
        published_at=item.created_at,
    )
    session.add(listing)
    session.commit()
    return listing


def test_accept_bundle_is_registered_after_accept_offer_with_its_trust_key():
    names = default_registry().names()
    assert names.index("accept_bundle") == names.index("accept_offer") + 1
    assert BUNDLE.trust_action == "accept_offer"
    spec = BUNDLE.spec()
    assert spec["args_schema"]["additionalProperties"] is False
    assert 40 < len(spec["description"]) < 400 and "—" not in spec["description"]


@pytest.mark.parametrize(
    "raw",
    [
        {"thread_id": 1, "item_ids": [1], "total_cents": 100},
        {"thread_id": 1, "item_ids": [1, 2, 3, 4, 5, 6], "total_cents": 100},
        {"thread_id": 1, "item_ids": [1, 2], "total_cents": 0},
        {"thread_id": 1, "item_ids": [1, 2], "total_cents": 100, "price_cents": 100},
    ],
)
def test_accept_bundle_args_are_bounded(raw):
    with pytest.raises(ValidationError):
        BUNDLE.parse_args(raw)


def test_accept_bundle_promises_both_and_sends_one_commitment_notice(
    ctx, session, clock, notifier, shop
):
    args = _args(shop)
    assert validate(ctx, BUNDLE, args) == Allow()

    result = _run(BUNDLE, ctx, args)

    rows = [session.get(Commitment, row_id) for row_id in result.data["commitment_ids"]]
    assert [row.item_id for row in rows] == shop.ids
    assert result.data["commitment_id"] == rows[0].id
    assert result.data["bundle_key"] == rows[0].bundle_key
    assert result.data["prices_cents"] == {str(shop.desk.id): 10_588, str(shop.chair.id): 4_412}
    assert result.summary == (
        f"Accepted $150.00 from Ann for 2 items: IKEA desk (item {shop.desk.id}, $105.88), "
        f"Office chair (item {shop.chair.id}, $44.12). They are all promised to Ann; the owner "
        "can undo until Thu Oct 1 12:10. Book one pickup for all of them."
    )
    [notice] = notifier.notices
    assert notice.kind == NoticeKind.COMMITMENT
    assert notice.title == "Bundle deal: IKEA desk, Office chair"
    assert notice.ref_id == rows[0].id and notice.item_id == shop.desk.id
    assert notice.expires_at == clock.now() + timedelta(minutes=10)
    assert "to Ann for $150.00 in total" in notice.body
    assert "Undo before Thu Oct 1 12:10 to cancel the whole bundle" in notice.body


def test_the_notice_undo_undoes_the_whole_bundle(
    engine, ctx, session, clock, settings, notifier, adapters, shop
):
    _run(BUNDLE, ctx, _args(shop))
    [notice] = notifier.notices
    control = ControlService(engine, clock, settings, adapters, notifier, default_registry())

    said = control.undo_commitment(notice.ref_id)

    assert said == (
        f"Undid the $150.00 bundle deal with Ann (items {shop.desk.id}, {shop.chair.id}). "
        f"Item {shop.desk.id} is listed again; the agent will re-plan. "
        f"Item {shop.chair.id} is listed again; the agent will re-plan."
    )
    session.expire_all()
    assert [r.kind for r in _all_rows(session)].count(CommitmentKind.UNDONE) == 2
    assert session.get(Item, shop.desk.id).status == ItemStatus.LISTED
    assert session.get(Item, shop.chair.id).status == ItemStatus.LISTED
    woken = session.exec(select(WakeRequest.item_id)).all()
    assert sorted(woken) == sorted(shop.ids)
    lessons = session.exec(select(Memory).where(Memory.source == "undo")).all()
    assert sorted(m.item_id for m in lessons) == sorted(shop.ids)
    assert all("bundle deal with Ann" in m.lesson for m in lessons)
    [trust] = session.exec(select(TrustRecord).where(TrustRecord.action == "accept_offer")).all()
    assert trust.corrected == 1


def test_accept_bundle_check_writes_nothing(ctx, session, shop):
    flushes: list[int] = []
    event.listen(session, "after_flush", lambda *_: flushes.append(1))
    assert BUNDLE.check(ctx, _args(shop)) == Allow()
    assert flushes == [] and not session.new and not session.dirty


def test_accept_bundle_rejects_a_thread_of_another_item(ctx, session, clock, shop):
    args = AcceptBundleArgs(thread_id=shop.chair_thread.id, item_ids=shop.ids, total_cents=15_000)
    assert "not a conversation about this item" in _reject(BUNDLE, ctx, args)


@pytest.mark.parametrize(("change", "match"), [("frozen", "is frozen"), ("flagged", "scammer")])
def test_accept_bundle_rejects_an_unsafe_buyer(ctx, session, shop, change, match):
    if change == "frozen":
        shop.desk_thread.state = ThreadState.FROZEN
    else:
        shop.ann.flagged_scam = True
    assert match in _reject(BUNDLE, ctx, _args(shop))


def test_accept_bundle_needs_a_buyer_message(ctx, session, clock, shop):
    bob = _buyer(session, "bob")
    silent = _thread(session, clock, shop.desk, bob, wrote=False)
    args = AcceptBundleArgs(thread_id=silent.id, item_ids=shop.ids, total_cents=15_000)
    assert "has not written anything yet" in _reject(BUNDLE, ctx, args)


def test_accept_bundle_must_include_the_woken_item(ctx, shop):
    reason = _reject(BUNDLE, ctx, _args(shop, [shop.chair.id, shop.lamp.id], 7_000))
    assert f"must include this item ({shop.desk.id})" in reason


def test_accept_bundle_rejects_a_repeated_item(ctx, shop):
    assert "each item once" in _reject(BUNDLE, ctx, _args(shop, [shop.desk.id, shop.desk.id]))


def test_accept_bundle_rejects_an_unknown_item(ctx, shop):
    assert "There is no item 999" in _reject(BUNDLE, ctx, _args(shop, [shop.desk.id, 999]))


def test_accept_bundle_rejects_a_paused_item_unless_the_owner_asks(
    session, clock, settings, notifier, adapters, ctx, shop
):
    shop.chair.paused = True
    reason = _reject(BUNDLE, ctx, _args(shop))
    assert reason == (
        f"Item {shop.chair.id} (Office chair) is paused by the owner. Leave it out of the bundle."
    )
    owner = _ctx(session, clock, settings, notifier, adapters, shop.desk, actor="human")
    assert BUNDLE.check(owner, _args(shop)) == Allow()


def test_accept_bundle_rejects_an_item_promised_to_someone_else(ctx, session, clock, shop):
    _promise_chair_to_bob(session, clock, shop)
    reason = _reject(BUNDLE, ctx, _args(shop))
    assert f"Item {shop.chair.id} (Office chair) is already promised to Bob" in reason
    assert "at $50.00. A bundle can replace that deal only once it is released." in reason


def test_accept_bundle_on_an_item_already_promised_to_the_same_buyer(ctx, session, clock, shop):
    ledger.accept(
        session, clock, item_id=shop.desk.id, thread_id=shop.desk_thread.id,
        buyer_id=shop.ann.id, price_cents=11_000, undo_minutes=10,
    )
    reason = _reject(BUNDLE, ctx, _args(shop))
    assert reason == (
        f"This item is already promised to Ann (thread {shop.desk_thread.id}) at $110.00. A "
        "bundle can replace that deal only once it is released."
    )


@pytest.mark.parametrize(
    ("status", "match"),
    [
        (ItemStatus.DRAFT, "is a draft"),
        (ItemStatus.SOLD, "is sold"),
        (ItemStatus.WITHDRAWN, "is withdrawn"),
    ],
)
def test_accept_bundle_needs_listed_items(ctx, shop, status, match):
    shop.chair.status = status
    assert match in _reject(BUNDLE, ctx, _args(shop))


def test_accept_bundle_needs_this_item_listed(ctx, shop):
    shop.desk.status = ItemStatus.DRAFT
    reason = _reject(BUNDLE, ctx, _args(shop))
    assert "You can make a bundle deal only once it is listed" in reason


def test_accept_bundle_needs_a_goal_on_every_item(ctx, session, clock, shop):
    rug = _rug(session, clock)
    assert "has no price goal" in _reject(BUNDLE, ctx, _args(shop, [shop.desk.id, rug.id]))


def test_accept_bundle_rejects_a_frozen_thread_about_another_item(ctx, shop):
    shop.chair_thread.state = ThreadState.FROZEN
    reason = _reject(BUNDLE, ctx, _args(shop))
    assert f"thread {shop.chair_thread.id}" in reason and "is frozen" in reason


def test_accept_bundle_total_must_cover_every_minimum(ctx, shop):
    ids = [shop.desk.id, shop.chair.id, shop.lamp.id]
    reason = _reject(BUNDLE, ctx, _args(shop, ids, 14_499))
    assert reason == (
        "$144.99 is below the $145.00 these items need together (IKEA desk floor $95.00; "
        "Office chair floor $30.00; Brass lamp firm at $20.00). Counter at $145.00 or more, "
        "or decline."
    )
    assert BUNDLE.check(ctx, _args(shop, ids, 14_500)) == Allow()


def test_a_single_offer_on_a_bundled_item_is_refused(
    session, clock, settings, notifier, adapters, ctx, shop
):
    _run(BUNDLE, ctx, _args(shop))
    bob = _buyer(session, "bob")
    bob_thread = _thread(session, clock, shop.chair, bob)
    chair_ctx = _ctx(session, clock, settings, notifier, adapters, shop.chair)
    reason = _reject(ACCEPT, chair_ctx, AcceptOfferArgs(thread_id=bob_thread.id, price_cents=5_000))
    assert "already promised to Ann" in reason


def test_book_pickup_books_the_whole_bundle_from_any_item(
    session, clock, settings, notifier, adapters, ctx, shop
):
    _run(BUNDLE, ctx, _args(shop, [shop.desk.id, shop.lamp.id], 13_000))
    # The lamp's row is on the desk thread (Ann never wrote about the lamp), so the lamp's
    # context may run the deal through that thread.
    lamp_ctx = _ctx(session, clock, settings, notifier, adapters, shop.lamp)
    args = BookPickupArgs(thread_id=shop.desk_thread.id, start="2026-10-01T18:00")

    result = _run(BOOK, lamp_ctx, args)

    assert result.summary == (
        "Pickup booked with Ann for IKEA desk, Brass lamp: Thu Oct 1 18:00-19:00."
    )
    for item in (shop.desk, shop.lamp):
        assert ledger.active_commitment(session, item.id).pickup.pickup_start == PICKUP_START
    # Rescheduling into its own slot is no clash.
    _run(BOOK, ctx, BookPickupArgs(thread_id=shop.desk_thread.id, start="2026-10-01T18:30"))
    assert len(ledger.active_pickups(session)) == 2


def test_a_bundle_thread_still_belongs_to_its_items_only(
    session, clock, settings, notifier, adapters, ctx, shop
):
    _run(BUNDLE, ctx, _args(shop))
    lamp_ctx = _ctx(session, clock, settings, notifier, adapters, shop.lamp)
    args = BookPickupArgs(thread_id=shop.desk_thread.id, start="2026-10-01T18:00")
    assert "not a conversation about this item" in _reject(BOOK, lamp_ctx, args)


def test_a_paused_bundle_item_blocks_changes_to_the_bundle(ctx, session, clock, shop):
    _run(BUNDLE, ctx, _args(shop))
    _book(session, clock, ledger.active_commitment(session, shop.desk.id).accepted)
    shop.chair.paused = True
    thread_id = shop.desk_thread.id
    clock.set(PICKUP_START)
    for tool, args in (
        (BOOK, BookPickupArgs(thread_id=thread_id, start="2026-10-01T19:00")),
        (RELEASE, ReleaseCommitmentArgs(thread_id=thread_id, reason="changed plans")),
        (COMPLETE, ThreadArgs(thread_id=thread_id)),
    ):
        reason = _reject(tool, ctx, args)
        assert f"bundle with item {shop.chair.id} (Office chair), which the owner paused" in reason


def test_release_commitment_releases_the_whole_bundle(ctx, session, shop):
    _run(BUNDLE, ctx, _args(shop))

    args = ReleaseCommitmentArgs(thread_id=shop.desk_thread.id, reason="backed out")
    result = _run(RELEASE, ctx, args)

    assert result.summary == (
        "Released the bundle deal with Ann (IKEA desk, Office chair); the items are available "
        "again."
    )
    assert len(result.data["commitment_ids"]) == 2
    assert shop.desk.status == shop.chair.status == ItemStatus.LISTED
    assert ledger.active_pickups(session) == [] and shop.ann.no_show_count == 0


def test_complete_sale_sells_and_delists_every_item_and_records_each_outcome(
    ctx, session, clock, notifier, adapters, shop
):
    listings = [
        _listing(session, shop.desk, "fake"),
        _listing(session, shop.desk, "kijiji"),
        _listing(session, shop.chair, "fake"),
    ]
    _run(BUNDLE, ctx, _args(shop))
    _book(session, clock, ledger.active_commitment(session, shop.chair.id).accepted)
    clock.set(PICKUP_START)

    result = _run(COMPLETE, ctx, ThreadArgs(thread_id=shop.desk_thread.id))

    assert result.summary == "Sold IKEA desk, Office chair to Ann for $150.00 in total."
    assert result.data["errors"] == [] and len(result.data["commitment_ids"]) == 2
    assert shop.desk.status == shop.chair.status == ItemStatus.SOLD
    assert shop.desk_thread.state == shop.chair_thread.state == ThreadState.CLOSED
    assert shop.ann.completed_count == 1
    assert all(listing.status == ListingStatus.DELISTED for listing in listings)
    assert sorted(adapters["fake"].delisted) == sorted(
        [listings[0].remote_id, listings[2].remote_id]
    )
    assert adapters["kijiji"].delisted == [listings[1].remote_id]
    [sold] = notifier.of_kind(NoticeKind.SOLD)
    assert sold.title == "Sold: IKEA desk, Office chair" and sold.item_id == shop.desk.id
    outcomes = session.exec(select(Memory).where(Memory.source == "outcome")).all()
    assert sorted(m.item_id for m in outcomes) == sorted(shop.ids)


# ---------------------------------------------------------------- owner controls on a bundle


@pytest.fixture
def control(engine, clock, settings, notifier, adapters) -> ControlService:
    return ControlService(engine, clock, settings, adapters, notifier, default_registry())


def _booked_bundle(session: Session, clock: FakeClock, shop: Shop) -> list[Commitment]:
    rows = _bundle(session, clock, shop)
    _book(session, clock, rows[0])
    return rows


def test_mark_sold_on_any_bundled_item_sells_and_delists_every_item(
    session, clock, control, adapters, shop
):
    listings = [_listing(session, shop.desk, "fake"), _listing(session, shop.chair, "kijiji")]
    _booked_bundle(session, clock, shop)
    clock.set(PICKUP_START)

    control.mark_sold(shop.chair.id)

    session.expire_all()
    assert session.get(Item, shop.desk.id).status == ItemStatus.SOLD
    assert session.get(Item, shop.chair.id).status == ItemStatus.SOLD
    assert adapters["fake"].delisted == [listings[0].remote_id]
    assert adapters["kijiji"].delisted == [listings[1].remote_id]
    assert all(session.get(Listing, x.id).status == ListingStatus.DELISTED for x in listings)


def test_mark_no_show_on_a_bundle_wakes_every_released_item(session, clock, control, shop):
    _booked_bundle(session, clock, shop)
    clock.set(PICKUP_START + timedelta(hours=2))

    said = control.mark_no_show(shop.desk.id)

    assert said == (
        f"Released the deal with Ann as a no-show. The agent will re-plan items "
        f"{shop.desk.id}, {shop.chair.id}."
    )
    session.expire_all()
    assert session.get(Buyer, shop.ann.id).no_show_count == 1
    assert {session.get(Item, i).status for i in shop.ids} == {ItemStatus.LISTED}
    woken = session.exec(select(WakeRequest)).all()
    assert sorted(w.item_id for w in woken) == sorted(shop.ids)
    assert all("did not come to the pickup" in w.reason for w in woken)


def test_withdrawing_a_bundled_item_wakes_the_rest_of_the_bundle(
    session, clock, control, notifier, shop
):
    _bundle(session, clock, shop)

    control.withdraw(shop.desk.id)

    session.expire_all()
    assert session.get(Item, shop.desk.id).status == ItemStatus.WITHDRAWN
    assert session.get(Item, shop.chair.id).status == ItemStatus.LISTED
    [wake] = session.exec(select(WakeRequest)).all()
    assert wake.item_id == shop.chair.id
    assert f"withdrew item {shop.desk.id}" in wake.reason and "bundle deal with Ann" in wake.reason
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "Tell Ann the deal is off"
    assert f"with items {shop.chair.id} as a bundle" in alert.body


def test_a_bundle_gets_one_pickup_check_that_any_item_answers(
    session, clock, settings, control, notifier, shop
):
    from marketplace_agent.core.models import Escalation
    from marketplace_agent.guard import handover

    rows = _booked_bundle(session, clock, shop)
    clock.set(PICKUP_START + timedelta(hours=1, minutes=5))

    assert handover.ask_owner(session, clock, settings, notifier) == 1
    session.commit()
    assert handover.ask_owner(session, clock, settings, notifier) == 0

    [notice] = notifier.of_kind(NoticeKind.ESCALATION)
    assert notice.title == "Did Ann pick up IKEA desk and Office chair?"
    assert "If Ann took them" in notice.body
    [check] = session.exec(select(Escalation)).all()
    assert all(handover.open_check(session, row.id) is check for row in rows)

    control.mark_sold(shop.chair.id)

    session.expire_all()
    assert session.get(Escalation, check.id).resolved_at is not None


@pytest.mark.parametrize("came", [True, False])
def test_a_bundle_pickup_check_is_answered_with_its_id_from_any_item(
    session, clock, settings, control, notifier, shop, came
):
    """The web and Discord pass the check's id with the answer; the check sits on the bundle's
    first item, and the owner may answer it from the page of any item in the bundle."""
    from marketplace_agent.control.service import ControlError
    from marketplace_agent.core.models import Escalation
    from marketplace_agent.guard import handover

    _booked_bundle(session, clock, shop)
    clock.set(PICKUP_START + timedelta(hours=1, minutes=5))
    handover.ask_owner(session, clock, settings, notifier)
    session.commit()
    [check] = session.exec(select(Escalation)).all()
    assert check.item_id == shop.desk.id
    with pytest.raises(ControlError, match=f"no pickup check with id {check.id} on item"):
        control.mark_sold(shop.lamp.id, check_id=check.id)

    if came:
        control.mark_sold(shop.chair.id, check_id=check.id)
    else:
        control.mark_no_show(shop.chair.id, check_id=check.id)

    session.expire_all()
    want = ItemStatus.SOLD if came else ItemStatus.LISTED
    assert {session.get(Item, i).status for i in shop.ids} == {want}
    assert session.get(Item, shop.lamp.id).status == ItemStatus.LISTED
    assert session.get(Escalation, check.id).resolved_at is not None


def test_mark_sold_on_a_bundle_returns_every_listing_still_up(
    session, clock, control, adapters, notifier, shop
):
    from marketplace_agent.runner.takedown import takedown_state

    _listing(session, shop.desk, "fake")
    stuck = _listing(session, shop.chair, "kijiji")
    _bundle(session, clock, shop)

    def down(ref: PlatformRef) -> None:
        raise PlatformError("timeout")

    adapters["kijiji"].delist = down

    still_up = control.mark_sold(shop.desk.id)

    assert still_up == [f"kijiji ({stuck.url or 'no link'})"]
    session.expire_all()
    listing = session.get(Listing, stuck.id)
    assert listing.status == ListingStatus.LIVE
    assert takedown_state(listing).alerted and takedown_state(listing).error == "timeout"
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.item_id == shop.chair.id and "Office chair" in alert.body


def test_complete_sale_of_one_item_still_records_its_outcome(ctx, session, clock, shop):
    accepted = ledger.accept(
        session, clock, item_id=shop.desk.id, thread_id=shop.desk_thread.id,
        buyer_id=shop.ann.id, price_cents=11_000, undo_minutes=10,
    )
    _book(session, clock, accepted)
    clock.set(PICKUP_START)

    result = _run(COMPLETE, ctx, ThreadArgs(thread_id=shop.desk_thread.id))

    assert result.summary == "Sold to Ann for $110.00."
    [outcome] = session.exec(select(Memory).where(Memory.source == "outcome")).all()
    assert outcome.item_id == shop.desk.id


# ---------------------------------------------------------------- property: ledger with bundles

P_ITEMS = 3  # 0 desk $120/$95, 1 chair $50/$30, 2 lamp $20 firm
P_LISTS = (12_000, 5_000, 2_000)
P_MINIMUMS = (9_500, 3_000, 2_000)
P_THREADS = ((0, 0), (1, 0), (0, 1), (2, 1))  # (item, buyer): Ann: desk, chair. Bob: desk, lamp
P_UNDO = 10
LEDGER_INVARIANTS = {
    "double_commitment",
    "accept_below_minimum",
    "pickup_overlap",
    "pickup_invalid",
    "ledger_ref_inactive",
    "late_undo",
    "bundle_shape",
    "bundle_total",
    "bundle_split",
}


@dataclass
class PDeal:
    ids: dict[int, int]  # item index -> ACCEPTED id
    prices: dict[int, int]
    undo_deadline: datetime
    pickup: tuple[datetime, datetime] | None = None


class PModel:
    """The bundle ledger written the obvious way: a deal is a set of items that move together."""

    def __init__(self) -> None:
        self.active: dict[int, PDeal] = {}  # item index -> its deal (shared by a bundle)
        self.sold: set[int] = set()

    def can_promise(self, items: list[int]) -> bool:
        return all(i not in self.sold and i not in self.active for i in items)


p_bundle = st.tuples(
    st.just("bundle"),
    st.integers(0, len(P_THREADS) - 1),
    st.lists(st.integers(0, P_ITEMS - 1), max_size=2, unique=True),  # beside the thread's item
    st.sampled_from((-1, 0, 1, 3_333, 10_000)),  # total minus the items' minimums
)
p_ops = st.lists(
    st.one_of(
        st.tuples(
            st.just("accept"),
            st.integers(0, len(P_THREADS) - 1),
            st.sampled_from((1, 2_000, 3_000, 9_500, 12_000)),
        ),
        p_bundle,
        p_bundle,
        st.tuples(
            st.just("book"), st.integers(0, P_ITEMS - 1), st.integers(1, 48), st.integers(-30, 120)
        ),
        st.tuples(st.sampled_from(("release", "undo", "complete")), st.integers(0, P_ITEMS - 1)),
        st.tuples(st.just("advance"), st.integers(1, 600)),
    ),
    max_size=24,
)


class PWorld:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.engine = make_engine("sqlite://")
        init_db(self.engine)
        self.session = Session(self.engine, expire_on_commit=False)
        firm = (Firmness.SLIGHT, Firmness.SLIGHT, Firmness.FIRM)
        floors = (9_500, 3_000, 1_500)
        self.items = [
            _item(self.session, self.clock, f"item {n}", P_LISTS[n], floors[n], firm[n]).id
            for n in range(P_ITEMS)
        ]
        buyers = [_buyer(self.session, name) for name in ("ann", "bob")]
        self.threads: list[tuple[int, int]] = []
        for item, buyer in P_THREADS:
            on = self.session.get(Item, self.items[item])
            thread = _thread(self.session, self.clock, on, buyers[buyer])
            self.threads.append((thread.id, buyers[buyer].id))
        self.session.commit()

    def close(self) -> None:
        self.session.close()
        self.engine.dispose()


def _p_apply(world: PWorld, model: PModel, op: tuple[Any, ...]) -> None:
    kind = op[0]
    if kind == "advance":
        world.clock.advance(minutes=op[1])
        return
    now = world.clock.now()
    if kind in ("accept", "bundle"):
        thread_item, buyer = P_THREADS[op[1]]
        thread_id, buyer_id = world.threads[op[1]]
        if kind == "accept":
            items, total = [thread_item], op[2]
            expected = model.can_promise(items) and total >= P_MINIMUMS[thread_item]
            call = partial(
                ledger.accept, item_id=world.items[thread_item], thread_id=thread_id,
                buyer_id=buyer_id, price_cents=total, undo_minutes=P_UNDO,
            )
        else:
            items = [thread_item, *op[2]]
            total = sum(P_MINIMUMS[i] for i in items) + op[3]
            expected = (
                2 <= len(items) <= 5
                and len(set(items)) == len(items)
                and thread_item in items
                and model.can_promise(items)
                and total >= sum(P_MINIMUMS[i] for i in items)
            )
            call = partial(
                ledger.accept_bundle, item_ids=[world.items[i] for i in items],
                thread_id=thread_id, buyer_id=buyer_id, total_cents=total, undo_minutes=P_UNDO,
            )
    else:
        deal = model.active.get(op[1])
        if deal is None:
            return
        accepted_id = deal.ids[op[1]]
        if kind == "book":
            start = now + timedelta(hours=op[2])
            end = start + timedelta(minutes=op[3])
            others = [d for i, d in model.active.items() if d is not deal]
            expected = start < end and not any(
                d.pickup and start < d.pickup[1] and d.pickup[0] < end for d in others
            )
            call = partial(ledger.book_pickup, accepted_id=accepted_id, start=start, end=end)
        else:
            expected = kind != "undo" or now <= deal.undo_deadline
            write = {"release": ledger.release, "undo": ledger.undo, "complete": ledger.complete}
            extra = {"reason": "test"} if kind == "release" else {}
            call = partial(write[kind], accepted_id=accepted_id, **extra)
    try:
        result = call(world.session, world.clock)
        world.session.commit()
    except ledger.LedgerError:
        world.session.rollback()
        result = None
    assert (result is not None) == expected, (op, model.active)
    if result is None:
        return
    if kind in ("accept", "bundle"):
        rows = result if isinstance(result, list) else [result]
        deal = PDeal(
            ids={i: row.id for i, row in zip(items, rows, strict=True)},
            prices={i: row.price_cents for i, row in zip(items, rows, strict=True)},
            undo_deadline=now + timedelta(minutes=P_UNDO),
        )
        assert sum(deal.prices.values()) == total
        assert all(deal.prices[i] >= P_MINIMUMS[i] for i in items)
        for i in items:
            model.active[i] = deal
    elif kind == "book":
        model.active[op[1]].pickup = (start, end)
    else:
        for i in list(model.active[op[1]].ids):
            del model.active[i]
            if kind == "complete":
                model.sold.add(i)


def _p_agree(world: PWorld, model: PModel) -> None:
    for n, item_id in enumerate(world.items):
        got = ledger.active_commitment(world.session, item_id)
        want = model.active.get(n)
        status = world.session.get(Item, item_id, populate_existing=True).status
        if want is None:
            assert got is None
            assert status == (ItemStatus.SOLD if n in model.sold else ItemStatus.LISTED)
            continue
        assert got is not None and got.accepted.id == want.ids[n]
        assert got.accepted.price_cents == want.prices[n]
        booked = (got.pickup.pickup_start, got.pickup.pickup_end) if got.pickup else None
        assert booked == want.pickup
        assert status == ItemStatus.COMMITTED


@hypothesis_settings(max_examples=150, deadline=None)
@given(ops=p_ops)
@example(
    ops=[
        ("bundle", 0, [1], 0),  # Ann: desk and chair at exactly their floors
        ("book", 1, 5, 60),  # booked through the chair's row
        ("accept", 2, 12_000),  # Bob on the desk: promised
        ("bundle", 3, [0], 1),  # Bob: lamp and desk: the desk is promised
        ("accept", 3, 2_000),  # Bob: the lamp alone
        ("book", 2, 5, 30),  # overlaps the bundle's pickup
        ("book", 2, 7, 30),
        ("advance", 5),
        ("undo", 0),  # undoes desk and chair
        ("bundle", 0, [1, 2], 0),  # the lamp is promised
        ("bundle", 0, [1], 3_333),
        ("book", 0, 7, 30),  # overlaps the lamp's pickup
        ("book", 0, 9, 60),
        ("advance", 600),
        ("undo", 1),  # too late
        ("complete", 1),  # sells desk and chair
        ("complete", 2),
    ]
)
def test_ledger_with_bundles_matches_a_naive_model_and_the_audit(ops):
    world = PWorld()
    model = PModel()
    try:
        for op in ops:
            _p_apply(world, model, op)
            _p_agree(world, model)
        world.session.expire_all()
        found = audit(world.session, sim_settings())
        assert [v for v in found if v.split(" ")[1].rstrip(":") in LEDGER_INVARIANTS] == []
    finally:
        world.close()
