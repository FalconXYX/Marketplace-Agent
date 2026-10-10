"""accept_offer, book_pickup, release_commitment, complete_sale.

FakeClock starts Thu 2026-10-01 16:00 UTC, which is 12:00 in Toronto. Thursday pickups run
17:00-20:00 local (21:00-24:00 UTC).
"""
import itertools
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import event
from sqlmodel import select

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
    Message,
    Thread,
    ThreadState,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.ledger import LedgerConflict
from marketplace_agent.guard.validate import after_execute, validate
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import ListingStats, PlatformError, PlatformRef
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.runner.takedown import TakedownState, takedown_state
from marketplace_agent.tools.base import Allow, Reject, ToolContext
from marketplace_agent.tools.impl.deals import (
    AcceptOfferArgs,
    AcceptOfferTool,
    BookPickupArgs,
    BookPickupTool,
    CompleteSaleTool,
    ReleaseCommitmentArgs,
    ReleaseCommitmentTool,
    ThreadArgs,
)

LIST, FLOOR = 12_000, 9_500
ACCEPT, BOOK, RELEASE, COMPLETE = (
    AcceptOfferTool(),
    BookPickupTool(),
    ReleaseCommitmentTool(),
    CompleteSaleTool(),
)
PICKUP_START = datetime(2026, 10, 1, 22, tzinfo=UTC)  # Thu 18:00 local
AVAILABILITY = "Mon-Fri 17:00-20:00, Sat-Sun 10:00-18:00"


class StubAdapter:
    """Records calls; fail[method] raises once."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.calls: list[tuple[str, object]] = []
        self.fail: dict[str, Exception] = {}
        self._ids = itertools.count(1)

    def _call(self, method: str, arg: object) -> None:
        self.calls.append((method, arg))
        if method in self.fail:
            raise self.fail.pop(method)

    def called(self, method: str) -> list:
        return [arg for name, arg in self.calls if name == method]

    def publish(self, listing, *, assist=False):
        self._call("publish", listing)
        return PlatformRef(self.name, f"{self.name}-{next(self._ids)}")

    def update(self, ref, listing):
        self._call("update", (ref, listing))

    def delist(self, ref):
        self._call("delist", ref)

    def fetch_threads(self, since):
        return []

    def send_message(self, thread_remote_id, body):
        self._call("send_message", (thread_remote_id, body))
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


@pytest.fixture
def item(session, clock) -> Item:
    return _item(session, clock)


@pytest.fixture
def thread(session, clock, item) -> Thread:
    return _thread(session, clock, item, "ann")


@pytest.fixture
def ctx(session, clock, settings, notifier, item, adapters) -> ToolContext:
    return _ctx(session, clock, settings, notifier, item, adapters)


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


def _item(
    session,
    clock,
    *,
    firmness: Firmness = Firmness.SLIGHT,
    status: ItemStatus = ItemStatus.LISTED,
    goal: bool = True,
) -> Item:
    item = Item(title="IKEA desk", status=status, created_at=clock.now(), updated_at=clock.now())
    session.add(item)
    session.flush()
    if goal:
        set_goal(
            session,
            clock,
            item.id,
            list_price_cents=LIST,
            floor_price_cents=FLOOR,
            firmness=firmness,
        )
    session.commit()
    return item


def _thread(session, clock, item: Item, handle: str, *, wrote: bool = True) -> Thread:
    buyer = Buyer(platform="fake", handle=handle, display_name=handle.title())
    session.add(buyer)
    session.flush()
    thread = Thread(
        platform="fake",
        remote_id=f"rt-{item.id}-{handle}",
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
                body="Would you take $110?",
                sent_at=clock.now() - timedelta(minutes=5),
            )
        )
    session.commit()
    return thread


def _listing(session, item: Item, platform: str) -> Listing:
    listing = Listing(
        item_id=item.id,
        platform=platform,
        remote_id=f"{platform}-r{item.id}",
        url=f"https://{platform}.test/{item.id}",
        status=ListingStatus.LIVE,
        price_cents=LIST,
        title=item.title,
        description="",
    )
    session.add(listing)
    session.commit()
    return listing


def _accept(session, clock, thread: Thread, price: int = 11_000) -> Commitment:
    accepted = ledger.accept(
        session,
        clock,
        item_id=thread.item_id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=price,
        undo_minutes=10,
    )
    session.commit()
    return accepted


def _book(session, clock, accepted: Commitment, start: datetime = PICKUP_START) -> Commitment:
    booked = ledger.book_pickup(
        session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(hours=1)
    )
    session.commit()
    return booked


def _reject(tool, ctx: ToolContext, args) -> str:
    verdict = tool.check(ctx, args)
    assert isinstance(verdict, Reject), verdict
    return verdict.reason


def _run(tool, ctx: ToolContext, args):
    assert tool.check(ctx, args) == Allow()
    result = tool.execute(ctx, args)
    ctx.session.commit()
    return result


def _kinds(session, item: Item) -> list[CommitmentKind]:
    return [c.kind for c in ledger.item_history(session, item.id)]


def test_trust_keys():
    keys = [tool.trust_action for tool in (ACCEPT, BOOK, RELEASE, COMPLETE)]
    assert keys == ["accept_offer", "book_pickup", "release_commitment", "complete_sale"]


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (ACCEPT, lambda t: AcceptOfferArgs(thread_id=t.id, price_cents=11_000)),
        (BOOK, lambda t: BookPickupArgs(thread_id=t.id, start="2026-10-01T18:30")),
        (RELEASE, lambda t: ReleaseCommitmentArgs(thread_id=t.id, reason="x", no_show=True)),
        (COMPLETE, lambda t: ThreadArgs(thread_id=t.id)),
    ],
)
def test_check_writes_nothing(ctx, session, clock, item, tool, args):
    other = _item(session, clock)
    _book(session, clock, _accept(session, clock, _thread(session, clock, other, "zed")))
    thread = _thread(session, clock, item, "ann")
    if tool is not ACCEPT:
        _accept(session, clock, thread)
    flushes = []
    event.listen(session, "after_flush", lambda *_: flushes.append(1))
    tool.check(ctx, args(thread))
    assert flushes == [] and not session.new and not session.dirty


# ---------------------------------------------------------------- accept_offer


def test_accept_offer_commits_and_sends_one_commitment_notice(
    ctx, session, clock, notifier, item, thread
):
    args = AcceptOfferArgs(thread_id=thread.id, price_cents=11_000)
    assert validate(ctx, ACCEPT, args) == Allow()

    result = ACCEPT.execute(ctx, args)
    after_execute(ctx, ACCEPT, args, result)
    session.commit()

    accepted = session.get(Commitment, result.data["commitment_id"])
    assert accepted.kind == CommitmentKind.ACCEPTED and accepted.price_cents == 11_000
    assert accepted.buyer_id == thread.buyer_id and accepted.actor == "agent"
    assert item.status == ItemStatus.COMMITTED and thread.state == ThreadState.COMMITTED
    assert result.ok and "Accepted $110.00 from Ann" in result.summary
    assert "undo until Thu Oct 1 12:10" in result.summary
    [notice] = notifier.notices
    assert notice.kind == NoticeKind.COMMITMENT
    assert notice.ref_id == accepted.id and notice.item_id == item.id
    assert notice.expires_at == clock.now() + timedelta(minutes=10)
    assert "Ann for $110.00" in notice.body and "Undo before Thu Oct 1 12:10" in notice.body


def test_accept_offer_rejects_a_thread_of_another_item(ctx, session, clock):
    other = _thread(session, clock, _item(session, clock), "zed")
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=other.id, price_cents=11_000))
    assert "not a conversation about this item" in reason


def test_accept_offer_rejects_a_frozen_thread(ctx, session, thread):
    thread.state = ThreadState.FROZEN
    thread.frozen_reason = "sends a payment link"
    session.commit()
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=11_000))
    assert "frozen (sends a payment link). Never make a deal with this buyer" in reason


def test_accept_offer_rejects_a_flagged_buyer(ctx, session, thread):
    session.get(Buyer, thread.buyer_id).flagged_scam = True
    session.commit()
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=11_000))
    assert "flagged as a scammer. Never make a deal with this buyer" in reason


def test_accept_offer_needs_a_buyer_message(ctx, session, clock, item):
    silent = _thread(session, clock, item, "quiet", wrote=False)
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=silent.id, price_cents=11_000))
    assert "has not written anything yet. Accept only an offer a buyer has actually made" in reason


def test_accept_offer_rejects_a_second_buyer(ctx, session, clock, item, thread):
    _accept(session, clock, thread)
    second = _thread(session, clock, item, "bo")
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=second.id, price_cents=12_000))
    assert f"already promised to Ann (thread {thread.id}) at $110.00" in reason
    assert "Tell other buyers it is pending" in reason


def test_accept_offer_needs_a_listed_item(session, clock, settings, notifier, adapters):
    draft = _item(session, clock, status=ItemStatus.DRAFT)
    thread = _thread(session, clock, draft, "ann")
    ctx = _ctx(session, clock, settings, notifier, draft, adapters)
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=11_000))
    assert "still a draft" in reason


def test_accept_offer_needs_a_goal(session, clock, settings, notifier, adapters):
    bare = _item(session, clock, goal=False)
    thread = _thread(session, clock, bare, "ann")
    ctx = _ctx(session, clock, settings, notifier, bare, adapters)
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=11_000))
    assert "no price goal (list price and floor) yet" in reason and "Escalate." in reason


def test_accept_offer_never_below_the_floor(ctx, thread):
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=9_499))
    assert reason == "$94.99 is below the floor of $95.00. Counter at $95.00 or more, or decline."
    assert ACCEPT.check(ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=FLOOR)) == Allow()


def test_accept_offer_firm_needs_the_list_price(session, clock, settings, notifier, adapters):
    firm = _item(session, clock, firmness=Firmness.FIRM)
    thread = _thread(session, clock, firm, "ann")
    ctx = _ctx(session, clock, settings, notifier, firm, adapters)
    reason = _reject(ACCEPT, ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=11_999))
    assert reason == "The price is firm at $120.00. Accept $120.00 or more, or decline."
    assert ACCEPT.check(ctx, AcceptOfferArgs(thread_id=thread.id, price_cents=LIST)) == Allow()


def test_accept_offer_race_is_caught_by_the_ledger(ctx, session, clock, item, thread):
    args = AcceptOfferArgs(thread_id=thread.id, price_cents=11_000)
    assert ACCEPT.check(ctx, args) == Allow()
    _accept(session, clock, _thread(session, clock, item, "bo"))  # wins between check and execute

    with pytest.raises(LedgerConflict):
        ACCEPT.execute(ctx, args)


# ---------------------------------------------------------------- book_pickup


def test_book_pickup_books_local_time(ctx, session, clock, item, thread):
    accepted = _accept(session, clock, thread)
    args = BookPickupArgs(thread_id=thread.id, start="2026-10-01T18:00", duration_minutes=45)

    result = _run(BOOK, ctx, args)

    active = ledger.active_commitment(session, item.id)
    assert active.pickup.pickup_start == PICKUP_START
    assert active.pickup.pickup_end == PICKUP_START + timedelta(minutes=45)
    assert result.summary == "Pickup booked with Ann: Thu Oct 1 18:00-18:45."
    assert result.data["commitment_id"] == accepted.id
    assert result.data["pickup_id"] == active.pickup.id
    assert result.data["start"] == PICKUP_START.isoformat()


def test_book_pickup_again_reschedules(ctx, session, clock, item, thread):
    _book(session, clock, _accept(session, clock, thread))
    assert _run(BOOK, ctx, BookPickupArgs(thread_id=thread.id, start="2026-10-01T18:30")).ok
    pickup = ledger.active_commitment(session, item.id).pickup
    assert pickup.pickup_start == PICKUP_START + timedelta(minutes=30)


def test_book_pickup_needs_an_accepted_deal(ctx, thread):
    reason = _reject(BOOK, ctx, BookPickupArgs(thread_id=thread.id, start="2026-10-01T18:00"))
    assert "no accepted deal in this thread. Accept an offer with accept_offer" in reason


def test_book_pickup_rejects_a_thread_of_another_item(ctx, session, clock):
    other = _thread(session, clock, _item(session, clock), "zed")
    reason = _reject(BOOK, ctx, BookPickupArgs(thread_id=other.id, start="2026-10-01T18:00"))
    assert "not a conversation about this item" in reason


def test_book_pickup_rejects_a_frozen_thread(ctx, session, clock, thread):
    _accept(session, clock, thread)
    thread.state = ThreadState.FROZEN
    session.commit()
    reason = _reject(BOOK, ctx, BookPickupArgs(thread_id=thread.id, start="2026-10-01T18:00"))
    assert "Never book a pickup with this buyer (release the deal)" in reason


def test_book_pickup_rejects_junk_times(ctx, session, clock, thread):
    _accept(session, clock, thread)
    reason = _reject(BOOK, ctx, BookPickupArgs(thread_id=thread.id, start="tomorrow at 6"))
    assert "is not an ISO 8601 date and time" in reason
    assert "Write start as a local date and time: YYYY-MM-DDTHH:MM" in reason


def test_book_pickup_rejects_the_past(ctx, session, clock, thread):
    _accept(session, clock, thread)
    reason = _reject(BOOK, ctx, BookPickupArgs(thread_id=thread.id, start="2026-09-30T18:00"))
    assert "Wed Sep 30 18:00 is in the past; it is now Thu Oct 1 12:00" in reason
    assert f"Pick a future time within: {AVAILABILITY}" in reason


@pytest.mark.parametrize(
    ("start", "minutes"),
    [("2026-10-01T15:00", 60), ("2026-10-01T19:30", 60), ("2026-10-01T20:00", 15)],
)
def test_book_pickup_must_fit_availability(ctx, session, clock, thread, start, minutes):
    _accept(session, clock, thread)
    args = BookPickupArgs(thread_id=thread.id, start=start, duration_minutes=minutes)
    reason = _reject(BOOK, ctx, args)
    assert "is outside pickup availability" in reason
    assert f"The whole pickup must fit inside one window: {AVAILABILITY}." in reason


def test_book_pickup_overlap_names_the_conflicting_time(ctx, session, clock, item, thread):
    other = _thread(session, clock, _item(session, clock), "zed")
    _book(session, clock, _accept(session, clock, other))
    _accept(session, clock, thread)

    reason = _reject(BOOK, ctx, BookPickupArgs(thread_id=thread.id, start="2026-10-01T18:30"))

    assert "overlaps another pickup booked Thu Oct 1 18:00-19:00" in reason
    assert "Choose a time that does not overlap it" in reason
    back_to_back = BookPickupArgs(thread_id=thread.id, start="2026-10-01T19:00")
    assert BOOK.check(ctx, back_to_back) == Allow()


@pytest.mark.parametrize("minutes", [14, 241])
def test_book_pickup_duration_bounds(minutes):
    with pytest.raises(ValidationError):
        BOOK.parse_args({"thread_id": 1, "start": "2026-10-01T18:00", "duration_minutes": minutes})


# ---------------------------------------------------------------- release_commitment


def test_release_frees_the_item(ctx, session, clock, item, thread):
    accepted = _accept(session, clock, thread)
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="Buyer backed out")

    result = _run(RELEASE, ctx, args)

    assert result.summary == "Released the deal with Ann; the item is available again."
    assert result.data == {"commitment_id": accepted.id}
    assert _kinds(session, item) == [CommitmentKind.ACCEPTED, CommitmentKind.RELEASED]
    assert ledger.item_history(session, item.id)[-1].reason == "Buyer backed out"
    assert item.status == ItemStatus.LISTED and thread.state == ThreadState.NEGOTIATING
    assert session.get(Buyer, thread.buyer_id).no_show_count == 0


def test_release_no_show_waits_until_the_owner_could_answer(ctx, session, clock, thread):
    """A buyer who came and paid in person may never write again: silence after the pickup is
    no evidence, so the owner (asked at pickup end) gets until pickup end + 6h to answer."""
    _book(session, clock, _accept(session, clock, thread))  # Thu 18:00-19:00 local
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="Did not come", no_show=True)

    clock.set(PICKUP_START + timedelta(minutes=30))
    reason = _reject(RELEASE, ctx, args)
    assert "runs until Thu Oct 1 19:00, and Ann may still come" in reason
    assert "It is a no-show only from Fri Oct 2 01:00" in reason

    clock.set(PICKUP_START + timedelta(hours=1, minutes=30))  # the old 30 min grace is over
    reason = _reject(RELEASE, ctx, args)
    assert "silence does not mean Ann did not come" in reason
    assert "agent sold or agent noshow" in reason and "release only from Fri Oct 2 01:00." in reason
    assert "Meanwhile you may message the buyer to check." in reason

    clock.set(PICKUP_START + timedelta(hours=7))
    result = _run(RELEASE, ctx, args)
    assert result.summary.startswith("Marked a no-show and released the deal with Ann")
    assert session.get(Buyer, thread.buyer_id).no_show_count == 1


def test_release_after_a_silent_pickup_needs_the_same_evidence(ctx, session, clock, thread):
    """no_show=false must not be a way around the wait: the item may be sold already."""
    _book(session, clock, _accept(session, clock, thread))
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="Never came", no_show=False)

    clock.set(PICKUP_START + timedelta(hours=2))
    reason = _reject(RELEASE, ctx, args)
    assert "or once Ann writes that they are not coming" in reason

    session.add(
        Message(
            thread_id=thread.id,
            direction=Direction.INBOUND,
            body="So sorry, can't make it after all",
            sent_at=clock.now(),
        )
    )
    session.commit()
    assert RELEASE.check(ctx, args) == Allow()


def test_release_before_the_pickup_starts_needs_no_wait(ctx, session, clock, thread):
    _book(session, clock, _accept(session, clock, thread))
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="Backed out")
    assert RELEASE.check(ctx, args) == Allow()


def test_the_owner_knows_whether_the_buyer_came_so_only_the_grace_applies(
    ctx, session, clock, settings, notifier, item, thread, adapters
):
    _book(session, clock, _accept(session, clock, thread))
    owner = _ctx(session, clock, settings, notifier, item, adapters, actor="human")
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="Did not come", no_show=True)

    clock.set(PICKUP_START + timedelta(hours=1, minutes=29))
    assert _reject(RELEASE, owner, args) == "It is a no-show only from Thu Oct 1 19:30."
    clock.advance(minutes=1)
    assert RELEASE.check(owner, args) == Allow()
    assert isinstance(RELEASE.check(ctx, args), Reject)  # the agent still waits for the owner


def test_no_show_reason_never_suggests_a_message_the_follow_up_rule_blocks(
    ctx, session, clock, thread
):
    """The agent sent the address that morning: asking where they are would be rejected."""
    _book(session, clock, _accept(session, clock, thread))
    clock.set(PICKUP_START - timedelta(hours=6))
    session.add(
        Message(
            thread_id=thread.id,
            direction=Direction.OUTBOUND,
            body="See you at 18:00! The address is [PICKUP_ADDRESS].",
            sent_at=clock.now(),
            via="agent",
        )
    )
    session.commit()
    clock.set(PICKUP_START + timedelta(minutes=40))

    reason = _reject(
        RELEASE, ctx, ReleaseCommitmentArgs(thread_id=thread.id, reason="Late", no_show=True)
    )

    assert "ask the buyer" not in reason
    assert "Your last message to the buyer has no reply yet, so wait." in reason


def test_release_no_show_needs_a_booked_pickup(ctx, session, clock, thread):
    _accept(session, clock, thread)
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="Gone", no_show=True)
    assert "No pickup was booked, so this cannot be a no-show" in _reject(RELEASE, ctx, args)


def test_release_needs_an_active_deal(ctx, thread):
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="x")
    assert f"no active deal in thread {thread.id} to release" in _reject(RELEASE, ctx, args)


def test_release_needs_a_reason(ctx, session, clock, thread):
    _accept(session, clock, thread)
    args = ReleaseCommitmentArgs(thread_id=thread.id, reason="  ")
    assert _reject(RELEASE, ctx, args) == "Say in reason why the deal is being released."


def test_release_rejects_a_thread_of_another_item(ctx, session, clock):
    other = _thread(session, clock, _item(session, clock), "zed")
    args = ReleaseCommitmentArgs(thread_id=other.id, reason="x")
    assert "not a conversation about this item" in _reject(RELEASE, ctx, args)


# ---------------------------------------------------------------- complete_sale


def test_complete_sale_marks_sold_and_delists_everywhere(
    ctx, session, clock, notifier, item, thread, adapters
):
    fake, kijiji = _listing(session, item, "fake"), _listing(session, item, "kijiji")
    accepted = _accept(session, clock, thread)
    _book(session, clock, accepted)
    clock.set(PICKUP_START - timedelta(hours=1))

    result = _run(COMPLETE, ctx, ThreadArgs(thread_id=thread.id))

    assert result.ok and result.summary == "Sold to Ann for $110.00."
    assert result.data == {"commitment_id": accepted.id, "errors": []}
    assert item.status == ItemStatus.SOLD and thread.state == ThreadState.CLOSED
    assert session.get(Buyer, thread.buyer_id).completed_count == 1
    for listing, adapter in ((fake, adapters["fake"]), (kijiji, adapters["kijiji"])):
        assert [ref.remote_id for ref in adapter.called("delist")] == [listing.remote_id]
        assert listing.status == ListingStatus.DELISTED and listing.delisted_at == clock.now()
    [sold] = notifier.of_kind(NoticeKind.SOLD)
    assert sold.ref_id == accepted.id and sold.title == "Sold: IKEA desk"


def test_complete_sale_survives_a_platform_failing(
    ctx, session, clock, notifier, item, thread, adapters
):
    fake, kijiji = _listing(session, item, "fake"), _listing(session, item, "kijiji")
    _book(session, clock, _accept(session, clock, thread))
    clock.set(PICKUP_START)
    adapters["kijiji"].fail["delist"] = PlatformError("kijiji is down")

    result = _run(COMPLETE, ctx, ThreadArgs(thread_id=thread.id))

    assert result.ok and item.status == ItemStatus.SOLD
    assert _kinds(session, item)[-1] == CommitmentKind.COMPLETED
    assert fake.status == ListingStatus.DELISTED
    assert kijiji.status == ListingStatus.LIVE
    assert takedown_state(kijiji) == TakedownState(0, clock.now(), "kijiji is down", alerted=True)
    assert result.data["errors"] == [f"kijiji ({kijiji.url}): kijiji is down"]
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert "Remove it by hand" in alert.body and alert.item_id == item.id
    [sold] = notifier.of_kind(NoticeKind.SOLD)
    assert "Still up, remove by hand" in sold.body


def test_complete_sale_counts_a_post_already_gone_as_down(
    ctx, session, clock, notifier, item, thread, adapters
):
    kijiji = _listing(session, item, "kijiji")
    _book(session, clock, _accept(session, clock, thread))
    clock.set(PICKUP_START)
    adapters["kijiji"].fail["delist"] = ListingGone("listing not found")

    result = _run(COMPLETE, ctx, ThreadArgs(thread_id=thread.id))

    assert result.ok and result.data["errors"] == []
    assert kijiji.status == ListingStatus.DELISTED
    assert notifier.of_kind(NoticeKind.ALERT) == []  # nothing to remove by hand


def test_complete_sale_with_a_disconnected_platform(
    session, clock, settings, notifier, item, thread
):
    listing = _listing(session, item, "kijiji")
    _book(session, clock, _accept(session, clock, thread))
    clock.set(PICKUP_START)
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": StubAdapter()})

    assert _run(COMPLETE, ctx, ThreadArgs(thread_id=thread.id)).ok
    assert takedown_state(listing).error == "kijiji is not connected"


def test_complete_sale_needs_an_active_deal(ctx, thread):
    reason = _reject(COMPLETE, ctx, ThreadArgs(thread_id=thread.id))
    assert f"no active deal in thread {thread.id} to complete" in reason


def test_complete_sale_needs_a_booked_pickup(ctx, session, clock, thread):
    _accept(session, clock, thread)
    reason = _reject(COMPLETE, ctx, ThreadArgs(thread_id=thread.id))
    assert "No pickup is booked for this deal. Book one with book_pickup" in reason


def test_complete_sale_not_before_the_pickup_hour(ctx, session, clock, thread):
    _book(session, clock, _accept(session, clock, thread))
    clock.set(PICKUP_START - timedelta(hours=1, seconds=1))
    reason = _reject(COMPLETE, ctx, ThreadArgs(thread_id=thread.id))
    assert "The pickup is Thu Oct 1 18:00" in reason
    assert "no earlier than Thu Oct 1 17:00" in reason


def test_complete_sale_rejects_a_thread_of_another_item(ctx, session, clock):
    other = _thread(session, clock, _item(session, clock), "zed")
    assert "not a conversation about this item" in _reject(
        COMPLETE, ctx, ThreadArgs(thread_id=other.id)
    )


def test_complete_sale_counts_only_live_listings(ctx, session, clock, item, thread, adapters):
    gone = _listing(session, item, "fake")
    gone.status = ListingStatus.DELISTED
    session.commit()
    _book(session, clock, _accept(session, clock, thread))
    clock.set(PICKUP_START)
    _run(COMPLETE, ctx, ThreadArgs(thread_id=thread.id))
    assert adapters["fake"].called("delist") == []
    query = select(Listing).where(Listing.item_id == item.id)
    assert [lst.status for lst in session.exec(query)] == [ListingStatus.DELISTED]
