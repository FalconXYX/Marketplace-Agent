"""Two writers on a file-backed database, as `agent run` and an owner's CLI command are.

The shared in-memory engine the other tests use has one connection and no locking, so it cannot
show these. Each test holds one session's write open on purpose and checks what the other side
does meanwhile and after.
"""

import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from marketplace_agent.control.service import ControlError, ControlService
from marketplace_agent.core.db import init_db, make_engine
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    PatternEntry,
    Thread,
    ThreadState,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.ledger import LedgerConflict
from marketplace_agent.guard.validate import validate
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import ListingPayload
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.runner.takedown import owed_takedowns, retry_takedowns
from marketplace_agent.tools.base import Allow, ToolContext
from marketplace_agent.tools.impl.listing import RelistTool
from marketplace_agent.tools.impl.messaging import SendMessageArgs, SendMessageTool
from marketplace_agent.tools.impl.pricing import UpdatePriceArgs, UpdatePriceTool
from marketplace_agent.tools.registry import default_registry
from test_runner import AVAILABLE, World

WAIT = 0.3  # long enough for the other side to reach its blocking point
UNBLOCKED = 3.0  # an owner write done within this did not wait out busy_timeout (5 s)


@pytest.fixture
def file_engine(tmp_path) -> Iterator[Engine]:
    engine = make_engine(f"sqlite:///{tmp_path / 'agent.db'}")
    init_db(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def market(clock) -> FakeMarketplace:
    return FakeMarketplace(clock)


@pytest.fixture
def control(file_engine, clock, settings, notifier, market) -> ControlService:
    return ControlService(
        file_engine, clock, settings, {"fake": market}, notifier, default_registry()
    )


@pytest.fixture
def desk(file_engine, clock, market) -> tuple[int, int, int]:
    """A listed desk ($120, floor $95) with a live listing and Maya's thread: (item, thread,
    buyer) ids."""
    now = clock.now()
    with Session(file_engine) as session:
        item = Item(
            title="IKEA desk", status=ItemStatus.LISTED, created_at=now, updated_at=now
        )
        buyer = Buyer(platform="fake", handle="maya", display_name="Maya")
        session.add_all([item, buyer])
        session.flush()
        set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
        ref = market.publish(ListingPayload("IKEA desk", "Solid.", 12_000, [], "Bloor"))
        remote = market.add_buyer("scripted", ref.remote_id)
        thread = Thread(
            platform="fake",
            remote_id=remote,
            item_id=item.id,
            buyer_id=buyer.id,
            state=ThreadState.NEGOTIATING,
            created_at=now,
        )
        session.add_all(
            [
                thread,
                Listing(
                    item_id=item.id,
                    platform="fake",
                    remote_id=ref.remote_id,
                    status=ListingStatus.LIVE,
                    price_cents=12_000,
                    title="IKEA desk",
                    description="Solid.",
                    published_at=now,
                ),
            ]
        )
        session.flush()
        session.add(
            Message(
                thread_id=thread.id,
                direction=Direction.INBOUND,
                body="Would you take $110?",
                sent_at=now,
                handled=True,
                via="buyer",
            )
        )
        session.commit()
        return item.id, thread.id, buyer.id


class Background:
    """Runs a call on another thread and keeps its result or error."""

    def __init__(self, call: Callable[[], Any]):
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(call,), daemon=True)
        self._thread.start()

    def _run(self, call: Callable[[], Any]) -> None:
        try:
            self.result = call()
        except BaseException as exc:  # noqa: BLE001  reported to the test
            self.error = exc
        finally:
            self.done.set()

    def join(self) -> None:
        assert self.done.wait(10), "the background call never finished"


class Release(threading.Event):
    """Set to let a slowed platform call finish; calling is set once the call has started."""

    def __init__(self) -> None:
        super().__init__()
        self.calling = threading.Event()


def _slow(market: FakeMarketplace, method: str, *, after: int = 0) -> Release:
    """Make market.<method> hang, from call number after + 1 on, until the Release is set, as a
    browser edit or a paced HTTP call does."""
    release, real, calls = Release(), getattr(market, method), [0]

    def slow(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] > after:
            release.calling.set()
            release.wait(10)
        return real(*args, **kwargs)

    setattr(market, method, slow)
    return release


def _owner_write_while(control: ControlService, item_id: int, release: Release) -> Background:
    """The owner pauses the item while a slowed platform call is under way; returns it once it
    is answered or UNBLOCKED has passed."""
    assert release.calling.wait(10)
    owner = Background(lambda: control.pause(item_id))
    owner.done.wait(UNBLOCKED)
    return owner


def _accept_open(session: Session, clock, desk: tuple[int, int, int]) -> Commitment:
    """The agent's accept_offer, flushed but not committed: it holds the write lock."""
    item_id, thread_id, buyer_id = desk
    return ledger.accept(
        session,
        clock,
        item_id=item_id,
        thread_id=thread_id,
        buyer_id=buyer_id,
        price_cents=11_000,
        undo_minutes=10,
    )


def _ledger_kinds(engine: Engine) -> list[CommitmentKind]:
    with Session(engine) as session:
        rows = session.exec(select(Commitment).order_by(Commitment.id)).all()
        return [row.kind for row in rows]


# ---------------------------------------------------------------- mark_sold, withdraw


def test_mark_sold_completes_a_deal_the_agent_is_accepting_right_now(
    file_engine, clock, control, desk
):
    item_id = desk[0]
    with Session(file_engine, expire_on_commit=False) as agent:
        _accept_open(agent, clock, desk)
        owner = Background(lambda: control.mark_sold(item_id, 11_000))
        assert not owner.done.wait(WAIT)  # waits for the agent's write instead of reading past it
        agent.commit()
    owner.join()

    assert owner.error is None
    assert _ledger_kinds(file_engine) == [CommitmentKind.ACCEPTED, CommitmentKind.COMPLETED]
    with Session(file_engine) as session:
        assert session.get(Item, item_id).status == ItemStatus.SOLD
        assert ledger.active_commitment(session, item_id) is None


def test_withdraw_releases_a_deal_the_agent_is_accepting_right_now(
    file_engine, clock, control, desk, notifier
):
    item_id = desk[0]
    with Session(file_engine, expire_on_commit=False) as agent:
        _accept_open(agent, clock, desk)
        owner = Background(lambda: control.withdraw(item_id))
        assert not owner.done.wait(WAIT)
        agent.commit()
    owner.join()

    assert owner.error is None
    assert _ledger_kinds(file_engine) == [CommitmentKind.ACCEPTED, CommitmentKind.RELEASED]
    with Session(file_engine) as session:
        assert session.get(Item, item_id).status == ItemStatus.WITHDRAWN
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "Tell Maya the deal is off"


# ---------------------------------------------------------------- price and floor


def test_set_floor_during_an_agent_price_push_checks_the_new_price_at_once(
    file_engine, clock, settings, notifier, control, desk, market
):
    """The agent lowers to $100 and pushes it; the owner asks for a $110 floor meanwhile. The
    price is committed before the push, so the owner is answered at once, against it."""
    item_id = desk[0]
    release = _slow(market, "update")

    def agent_reprice() -> None:
        with Session(file_engine, expire_on_commit=False) as session:
            ctx = ToolContext(session, clock, settings, {"fake": market}, notifier, item_id)
            result = UpdatePriceTool().execute(ctx, UpdatePriceArgs(price_cents=10_000))
            assert result.ok
            session.commit()

    agent = Background(agent_reprice)
    assert release.calling.wait(10)
    owner = Background(lambda: control.set_floor(item_id, 11_000))
    answered = owner.done.wait(UNBLOCKED)  # while the push is still running
    release.set()
    agent.join()
    owner.join()
    assert answered

    assert agent.error is None
    assert isinstance(owner.error, ControlError)
    assert "cannot be above the list price ($100.00)" in str(owner.error)
    with Session(file_engine) as session:
        goal = session.exec(select(Goal)).one()
        assert (goal.floor_price_cents, goal.list_price_cents) == (9_500, 10_000)


def test_update_price_rechecks_a_floor_raised_after_its_check(
    file_engine, clock, settings, notifier, control, desk, market
):
    item_id = desk[0]
    tool, args = UpdatePriceTool(), UpdatePriceArgs(price_cents=10_000)
    with Session(file_engine, expire_on_commit=False) as session:
        ctx = ToolContext(session, clock, settings, {"fake": market}, notifier, item_id)
        assert tool.check(ctx, args) == Allow()
        control.set_floor(item_id, 11_000)  # lands between the check and the write

        with pytest.raises(LedgerConflict, match=r"floor to \$110.00"):
            tool.execute(ctx, args)
        session.rollback()

    with Session(file_engine) as session:
        goal = session.exec(select(Goal)).one()
        assert (goal.floor_price_cents, goal.list_price_cents) == (11_000, 12_000)
    [listing] = market.listings.values()
    assert listing.payload.price_cents == 12_000  # nothing was pushed


# ---------------------------------------------------------------- platform calls


def test_a_slow_send_does_not_lock_out_the_owners_undo(
    file_engine, clock, settings, notifier, control, desk, market
):
    """Browser automation can take seconds per message; the owner's undo must not wait on it."""
    item_id, thread_id, _ = desk
    with Session(file_engine, expire_on_commit=False) as session:
        accepted = _accept_open(session, clock, desk)
        session.commit()
    sending, release = threading.Event(), threading.Event()
    real_send = market.send_message

    def slow_send(thread_remote_id, body):
        sending.set()
        release.wait(10)
        return real_send(thread_remote_id, body)

    market.send_message = slow_send  # type: ignore[method-assign]

    def agent_send() -> None:
        with Session(file_engine, expire_on_commit=False) as session:
            ctx = ToolContext(session, clock, settings, {"fake": market}, notifier, item_id)
            args = SendMessageArgs(thread_id=thread_id, body="Deal at $110! When suits you?")
            assert SendMessageTool().execute(ctx, args).ok
            session.commit()

    agent = Background(agent_send)
    assert sending.wait(10)
    try:
        assert "Undid the $110.00 deal with Maya" in control.undo_commitment(accepted.id)
    finally:
        release.set()
        agent.join()
    assert agent.error is None
    assert _ledger_kinds(file_engine) == [CommitmentKind.ACCEPTED, CommitmentKind.UNDONE]


# ---------------------------------------------------------------- no lock over a platform call


def _as_agent(file_engine, clock, settings, notifier, market, item_id, call) -> Background:
    """Run call(ctx) in a fresh session on another thread, and commit, as the agent loop does."""

    def run() -> None:
        with Session(file_engine, expire_on_commit=False) as session:
            ctx = ToolContext(session, clock, settings, {"fake": market}, notifier, item_id)
            result = call(ctx)
            assert result.ok, result.summary
            session.commit()

    return Background(run)


def test_a_slow_cache_reply_does_not_lock_out_the_owner(file_engine, clock, settings, notifier):
    """The pattern-cache hit is counted after the send, not flushed before it."""
    world = World(file_engine, clock, settings, notifier)
    item_id, listing = world.item()
    world.buyer(listing, AVAILABLE)
    control = ControlService(
        file_engine, clock, settings, {"fake": world.market}, notifier, default_registry()
    )
    release = _slow(world.market, "send_message")
    tick = Background(world.tick)
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        tick.join()
    assert tick.error is None and tick.result.cache_replies == 1
    with Session(file_engine) as session:
        assert sum(entry.hits for entry in session.exec(select(PatternEntry))) == 1
        assert session.get(Item, item_id).paused


def test_an_agents_first_send_does_not_lock_out_the_owner(
    file_engine, clock, settings, notifier, control, desk, market
):
    """The guard's trust check reads; the record is written after the send."""
    item_id, thread_id, _ = desk
    release = _slow(market, "send_message")
    tool = SendMessageTool()
    args = SendMessageArgs(thread_id=thread_id, body="Still here, when suits you?")

    def guarded_send(ctx: ToolContext):
        assert validate(ctx, tool, args) == Allow()
        return tool.execute(ctx, args)

    agent = _as_agent(file_engine, clock, settings, notifier, market, item_id, guarded_send)
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        agent.join()
    assert agent.error is None
    with Session(file_engine) as session:
        assert session.get(Item, item_id).paused


def test_a_slow_price_push_does_not_lock_out_the_owner(
    file_engine, clock, settings, notifier, control, desk, market
):
    """A Facebook edit takes a minute; the owner's pause must not wait for it, or fail."""
    item_id = desk[0]
    release = _slow(market, "update")
    agent = _as_agent(
        file_engine, clock, settings, notifier, market, item_id,
        lambda ctx: UpdatePriceTool().execute(ctx, UpdatePriceArgs(price_cents=10_000)),
    )  # fmt: skip
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        agent.join()
    assert agent.error is None
    with Session(file_engine) as session:
        goal = session.exec(select(Goal)).one()
        assert goal.list_price_cents == 10_000 and session.get(Item, item_id).paused
    [listing] = market.listings.values()
    assert listing.payload.price_cents == 10_000


def test_a_slow_rewrite_does_not_lock_out_the_owner(
    file_engine, clock, settings, notifier, control, desk, market
):
    item_id = desk[0]
    release = _slow(market, "update")
    rewrite = default_registry().get("rewrite_listing")
    assert rewrite is not None
    agent = _as_agent(
        file_engine, clock, settings, notifier, market, item_id,
        lambda ctx: rewrite.execute(ctx, rewrite.parse_args({"title": "White IKEA desk"})),
    )  # fmt: skip
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        agent.join()
    assert agent.error is None
    with Session(file_engine) as session:
        assert session.get(Item, item_id).title == "White IKEA desk"
        [row] = session.exec(select(Listing)).all()
        assert row.title == "White IKEA desk"


def test_taking_listings_down_after_a_sale_does_not_lock_out_the_owner(
    file_engine, clock, settings, notifier, control, desk, market
):
    item_id, thread_id, _ = desk
    with Session(file_engine, expire_on_commit=False) as session:
        _accept_open(session, clock, desk)
        session.commit()
    release = _slow(market, "delist")
    complete = default_registry().get("complete_sale")
    assert complete is not None
    agent = _as_agent(
        file_engine, clock, settings, notifier, market, item_id,
        lambda ctx: complete.execute(ctx, complete.parse_args({"thread_id": thread_id})),
    )  # fmt: skip
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        agent.join()
    assert agent.error is None
    assert _ledger_kinds(file_engine) == [CommitmentKind.ACCEPTED, CommitmentKind.COMPLETED]
    with Session(file_engine) as session:
        assert session.exec(select(Listing)).one().status == ListingStatus.DELISTED


def test_an_owner_goal_change_does_not_hold_the_ledger_during_its_push(
    file_engine, clock, control, desk, market
):
    """set_goal pushes the new price; another ledger write (here a floor) goes ahead meanwhile
    instead of waiting for the platforms, and is checked against the new price."""
    item_id = desk[0]
    release = _slow(market, "update")
    owner = Background(
        lambda: control.set_goal(item_id, list_price_cents=11_000, floor_price_cents=9_500)
    )
    try:
        assert release.calling.wait(10)
        other = Background(lambda: control.set_floor(item_id, 10_000))
        assert other.done.wait(UNBLOCKED) and other.error is None
    finally:
        release.set()
        owner.join()
    assert owner.error is None
    with Session(file_engine) as session:
        goal = session.exec(select(Goal)).one()
        assert (goal.list_price_cents, goal.floor_price_cents) == (11_000, 10_000)


def _chair(file_engine, clock, market) -> int:
    """A second listed item ($50, floor $30) with a live listing, to bundle with the desk."""
    now = clock.now()
    with Session(file_engine) as session:
        item = Item(
            title="Office chair", status=ItemStatus.LISTED, created_at=now, updated_at=now
        )
        session.add(item)
        session.flush()
        set_goal(session, clock, item.id, list_price_cents=5_000, floor_price_cents=3_000)
        ref = market.publish(ListingPayload("Office chair", "Comfy.", 5_000, [], "Bloor"))
        session.add(
            Listing(
                item_id=item.id,
                platform="fake",
                remote_id=ref.remote_id,
                status=ListingStatus.LIVE,
                price_cents=5_000,
                title="Office chair",
                description="Comfy.",
                published_at=now,
            )
        )
        session.commit()
        return item.id


def _bundle(file_engine, clock, desk: tuple[int, int, int], chair_id: int) -> None:
    """Maya takes the desk and the chair together for $150, agreed in the desk's thread."""
    item_id, thread_id, buyer_id = desk
    with Session(file_engine, expire_on_commit=False) as session:
        ledger.accept_bundle(
            session,
            clock,
            item_ids=[item_id, chair_id],
            thread_id=thread_id,
            buyer_id=buyer_id,
            total_cents=15_000,
            undo_minutes=10,
        )
        session.commit()


def _all_down(file_engine, item_ids: list[int]) -> None:
    with Session(file_engine) as session:
        assert {session.get(Item, i).status for i in item_ids} == {ItemStatus.SOLD}
        statuses = {listing.status for listing in session.exec(select(Listing)).all()}
        assert statuses == {ListingStatus.DELISTED}


def test_taking_a_bundles_listings_down_does_not_lock_out_the_owner(
    file_engine, clock, settings, notifier, control, desk, market
):
    """complete_sale takes down every item's listings; the owner's write goes ahead while the
    second item's takedown is still under way."""
    item_id, thread_id, _ = desk
    chair_id = _chair(file_engine, clock, market)
    _bundle(file_engine, clock, desk, chair_id)
    release = _slow(market, "delist", after=1)  # the chair's listing hangs
    complete = default_registry().get("complete_sale")
    assert complete is not None
    agent = _as_agent(
        file_engine, clock, settings, notifier, market, item_id,
        lambda ctx: complete.execute(ctx, complete.parse_args({"thread_id": thread_id})),
    )  # fmt: skip
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        agent.join()
    assert agent.error is None
    _all_down(file_engine, [item_id, chair_id])


def test_marking_a_bundle_sold_does_not_lock_out_the_owner_during_its_takedowns(
    file_engine, clock, control, desk, market
):
    item_id = desk[0]
    chair_id = _chair(file_engine, clock, market)
    _bundle(file_engine, clock, desk, chair_id)
    release = _slow(market, "delist", after=1)
    seller = Background(lambda: control.mark_sold(item_id))
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        seller.join()
    assert seller.error is None and seller.result == []
    _all_down(file_engine, [item_id, chair_id])


def test_reading_stats_does_not_lock_out_the_owner(
    file_engine, clock, settings, notifier, control, desk, market
):
    """Each Kijiji stats call is paced 1 to 3 s; with many listings the pass must write only
    once every call is done."""
    from marketplace_agent.runner.ingest import snapshot_stats

    item_id = desk[0]
    with Session(file_engine) as session:
        for title in ("Desk lamp", "Desk chair"):
            ref = market.publish(ListingPayload(title, "Solid.", 5_000, [], "Bloor"))
            session.add(
                Listing(
                    item_id=item_id,
                    platform="fake",
                    remote_id=ref.remote_id,
                    status=ListingStatus.LIVE,
                    price_cents=5_000,
                    title=title,
                    description="Solid.",
                    published_at=clock.now(),
                )
            )
        session.commit()
    release = _slow(market, "stats", after=2)  # the third call hangs

    def runner() -> int:
        with Session(file_engine) as session:
            stored = snapshot_stats(session, clock, market, notifier)
            session.commit()
            return stored

    stats = Background(runner)
    try:
        owner = _owner_write_while(control, item_id, release)
        assert owner.done.is_set() and owner.error is None
    finally:
        release.set()
        stats.join()
    assert stats.error is None and stats.result == 3


@pytest.mark.parametrize("during", ["delist", "publish"])
def test_a_sale_during_a_relist_leaves_the_fresh_post_owed_a_takedown(
    file_engine, clock, settings, notifier, control, desk, market, during
):
    """The relist holds no write lock during its platform calls, so the owner's mark_sold can
    commit meanwhile and mark the old post DELISTED. The fresh post must still be stored LIVE,
    so the takedown pass owes it and takes it down."""
    item_id = desk[0]
    clock.advance(hours=25)
    real = getattr(market, during)

    def owner_sells_meanwhile(*args, **kwargs):
        result = real(*args, **kwargs)
        owner = Background(lambda: control.mark_sold(item_id))
        owner.join()
        assert owner.error is None
        return result

    setattr(market, during, owner_sells_meanwhile)
    with Session(file_engine, expire_on_commit=False) as session:
        ctx = ToolContext(session, clock, settings, {"fake": market}, notifier, item_id)
        tool = RelistTool()
        args = tool.parse_args({"platform": "fake"})
        assert tool.check(ctx, args) == Allow()
        assert tool.execute(ctx, args).ok
        session.commit()
    setattr(market, during, real)

    fresh = [ref for ref, posted in market.listings.items() if posted.live]
    assert len(fresh) == 1  # the relisted copy, posted after the owner's takedown
    with Session(file_engine) as session:
        assert session.get(Item, item_id).status == ItemStatus.SOLD
        [owed] = owed_takedowns(session)
        assert (owed.status, owed.remote_id) == (ListingStatus.LIVE, fresh[0])
        assert retry_takedowns(session, clock, settings, {"fake": market}, notifier).taken_down
    assert not any(posted.live for posted in market.listings.values())
