"""Courtesy reply: a buyer writing about a SOLD or WITHDRAWN item hears so, once per thread."""

import itertools
from datetime import timedelta

import pytest
from sqlmodel import Session, select

from marketplace_agent.agent.loop import Agent
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.control.service import ControlService
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Item,
    ItemStatus,
    Message,
    Thread,
    ThreadState,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.invariants import outbound_count_today
from marketplace_agent.guard.validate import validate, validate_courtesy
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.runner.takedown import (
    COURTESY_WINDOW,
    courtesy_candidates,
    send_courtesy_replies,
)
from marketplace_agent.runner.tick import Runner
from marketplace_agent.tools.base import Allow, Reject, ToolContext
from marketplace_agent.tools.impl.messaging import (
    COURTESY_REPLIES,
    COURTESY_VIA,
    SendMessageTool,
)
from marketplace_agent.tools.registry import default_registry

SOLD_REPLY = "Sorry, it's sold now. Thanks for asking!"
TOOL = SendMessageTool()


class Sender:
    """A platform that records what it delivers; fail raises once."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.sent: list[tuple[str, str]] = []
        self.fail: Exception | None = None
        self._ids = itertools.count(1)

    def send_message(self, thread_remote_id: str, body: str) -> str:
        if self.fail is not None:
            failure, self.fail = self.fail, None
            raise failure
        self.sent.append((thread_remote_id, body))
        return f"m{next(self._ids)}"


@pytest.fixture
def sender() -> Sender:
    return Sender()


def _item(session, clock, status=ItemStatus.SOLD, *, paused=False) -> Item:
    now = clock.now()
    item = Item(title="IKEA desk", status=status, paused=paused, created_at=now, updated_at=now)
    session.add(item)
    session.flush()
    set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
    session.commit()
    return item


def _thread(
    session,
    clock,
    item: Item,
    handle: str,
    *,
    state=ThreadState.NEGOTIATING,
    flagged=False,
    remote_id: str | None = None,
) -> Thread:
    buyer = session.exec(select(Buyer).where(Buyer.handle == handle)).first()
    if buyer is None:
        buyer = Buyer(platform="fake", handle=handle, display_name=handle.title())
        buyer.flagged_scam = flagged
        session.add(buyer)
        session.flush()
    thread = Thread(
        platform="fake",
        remote_id=remote_id or f"rt-{item.id}-{handle}",
        item_id=item.id,
        buyer_id=buyer.id,
        state=state,
        created_at=clock.now(),
    )
    session.add(thread)
    session.commit()
    return thread


def _say(session, clock, thread: Thread, body="Is this still available?", *, ago=timedelta(0)):
    at = clock.now() - ago
    inbound = Direction.INBOUND
    session.add(Message(thread_id=thread.id, direction=inbound, body=body, sent_at=at, via="buyer"))
    thread.last_inbound_at = at
    session.add(thread)
    session.commit()


def _send(session, clock, settings, notifier, platforms):
    return send_courtesy_replies(session, clock, settings, platforms, notifier)


def _outbound(session, thread: Thread) -> list[Message]:
    query = select(Message).where(
        Message.thread_id == thread.id, Message.direction == Direction.OUTBOUND
    )
    return list(session.exec(query).all())


def _ctx(session, clock, settings, notifier, item, platforms, actor="system") -> ToolContext:
    return ToolContext(
        session=session,
        clock=clock,
        settings=settings,
        platforms=platforms,
        notifier=notifier,
        item_id=item.id,
        actor=actor,
    )


# ---------------------------------------------------------------- once per thread


def test_a_buyer_writing_on_a_sold_item_gets_one_courtesy_reply(
    session, clock, settings, notifier, sender
):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)

    sent, errors = _send(session, clock, settings, notifier, {"fake": sender})

    assert (sent, errors) == (1, [])
    assert sender.sent == [(thread.remote_id, SOLD_REPLY)]
    [reply] = _outbound(session, thread)
    assert (reply.body, reply.via, reply.handled) == (SOLD_REPLY, COURTESY_VIA, True)
    inbound = session.exec(select(Message).where(Message.direction == Direction.INBOUND)).one()
    assert inbound.handled
    assert session.get(Thread, thread.id).state == ThreadState.CLOSED
    assert notifier.notices == []

    clock.advance(hours=2)
    _say(session, clock, thread, "Are you sure? I can pay more.")
    clock.advance(days=2)
    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])
    assert len(sender.sent) == 1


def test_withdrawn_items_get_their_own_wording(session, clock, settings, notifier, sender):
    item = _item(session, clock, ItemStatus.WITHDRAWN)
    _say(session, clock, _thread(session, clock, item, "ann"))

    _send(session, clock, settings, notifier, {"fake": sender})

    assert [body for _, body in sender.sent] == [COURTESY_REPLIES[ItemStatus.WITHDRAWN]]
    assert "sold" not in COURTESY_REPLIES[ItemStatus.WITHDRAWN]


def test_each_thread_gets_its_own_reply(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    threads = [_thread(session, clock, item, handle) for handle in ("ann", "bo", "cy")]
    for thread in threads:
        _say(session, clock, thread)
        _say(session, clock, thread, "Hello?")  # two messages, still one reply

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (3, [])
    assert sorted(to for to, _ in sender.sent) == sorted(t.remote_id for t in threads)


def test_items_still_for_sale_are_left_to_the_agent(session, clock, settings, notifier, sender):
    for status in (ItemStatus.DRAFT, ItemStatus.LISTED, ItemStatus.COMMITTED):
        item = _item(session, clock, status)
        _say(session, clock, _thread(session, clock, item, f"b{status.value}"))

    assert courtesy_candidates(session, clock, {"fake": sender}) == []


# ---------------------------------------------------------------- never


def test_never_to_a_frozen_thread(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "scammer", state=ThreadState.FROZEN)
    _say(session, clock, thread, "Send me the code")

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])
    assert sender.sent == []

    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})
    verdict = validate_courtesy(ctx, TOOL, args)
    assert isinstance(verdict, Reject) and "frozen" in verdict.reason


def test_never_to_a_flagged_buyer(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "scammer", flagged=True)
    _say(session, clock, thread)

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})
    verdict = validate_courtesy(ctx, TOOL, args)
    assert isinstance(verdict, Reject) and "flagged" in verdict.reason


@pytest.mark.parametrize("same_thread", [True, False])
def test_never_to_the_buyer_who_had_the_deal(
    session, clock, settings, notifier, sender, same_thread
):
    """Not in the deal's own thread, nor in another thread of theirs (a bundle agreed elsewhere)."""
    item = _item(session, clock)
    deal_thread = _thread(session, clock, item, "ann", state=ThreadState.CLOSED)
    session.add(
        Commitment(
            item_id=item.id,
            thread_id=deal_thread.id,
            buyer_id=deal_thread.buyer_id,
            kind=CommitmentKind.ACCEPTED,
            price_cents=11_000,
            created_at=clock.now(),
        )
    )
    session.commit()
    if same_thread:
        thread = deal_thread
    else:
        thread = _thread(session, clock, item, "ann", state=ThreadState.OPEN, remote_id="rt-2")
    _say(session, clock, thread, "Thanks again, the desk is great!")

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})
    verdict = validate_courtesy(ctx, TOOL, args)
    assert isinstance(verdict, Reject) and "had a deal" in verdict.reason


def _ended_deal(session, clock, item: Item, thread: Thread, kind: CommitmentKind, reason=""):
    accepted = Commitment(
        item_id=item.id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        kind=CommitmentKind.ACCEPTED,
        price_cents=11_000,
        created_at=clock.now(),
    )
    session.add(accepted)
    session.flush()
    session.add(
        Commitment(
            item_id=item.id,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            kind=kind,
            ref_id=accepted.id,
            price_cents=11_000,
            reason=reason,
            created_at=clock.now(),
        )
    )
    session.commit()


@pytest.mark.parametrize("kind", [CommitmentKind.RELEASED, CommitmentKind.UNDONE])
def test_a_buyer_whose_deal_fell_through_earlier_gets_the_reply(
    session, clock, settings, notifier, sender, kind
):
    """Their deal ended while the item stayed for sale; then it sold to someone else. Nobody
    else will answer them: the owner was never told, and a SOLD item never wakes the agent."""
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _ended_deal(session, clock, item, thread, kind, reason="buyer backed out")
    _say(session, clock, thread, "Sorry about earlier, can I still get the desk?")

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (1, [])
    assert sender.sent == [(thread.remote_id, SOLD_REPLY)]


def test_never_to_the_buyer_who_bought_it(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann", state=ThreadState.CLOSED)
    _ended_deal(session, clock, item, thread, CommitmentKind.COMPLETED)
    _say(session, clock, thread, "Thanks again, the desk is great!")

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])


def test_never_to_the_buyer_whose_deal_the_withdrawal_cancelled(
    engine, session, clock, settings, notifier, sender
):
    """The owner was told to tell them the deal is off, and that the agent would not."""
    item = _item(session, clock, ItemStatus.LISTED)
    thread = _thread(session, clock, item, "ann")
    ledger.accept(
        session,
        clock,
        item_id=item.id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=11_000,
        undo_minutes=0,
    )
    session.commit()
    registry = default_registry()
    control = ControlService(engine, clock, settings, {"fake": sender}, notifier, registry)
    control.withdraw(item.id)
    session.expire_all()  # the owner's withdrawal committed in another session
    _say(session, clock, thread, "Still on for tomorrow?")

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])
    assert [n.title for n in notifier.notices] == ["Tell Ann the deal is off"]


def _bundle_less_the_chair(engine, session, clock, settings, notifier, sender, *, chair_thread):
    """Mei agreed to desk + chair in her desk thread; withdrawing the chair cancelled the deal
    and put the desk back on sale. Returns the desk, Mei's desk and chair threads, and the
    control."""
    desk = _item(session, clock, ItemStatus.LISTED)
    chair = _item(session, clock, ItemStatus.LISTED)
    thread = _thread(session, clock, desk, "mei")
    on_chair = _thread(session, clock, chair, "mei") if chair_thread else None
    ledger.accept_bundle(
        session,
        clock,
        item_ids=[desk.id, chair.id],
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        total_cents=20_000,
        undo_minutes=0,
    )
    session.commit()
    registry = default_registry()
    control = ControlService(engine, clock, settings, {"fake": sender}, notifier, registry)
    control.withdraw(chair.id)
    session.expire_all()
    assert session.get(Item, desk.id).status == ItemStatus.LISTED
    return desk, thread, on_chair, control


def test_a_bundle_cancelled_by_a_partners_withdrawal_leaves_an_ordinary_buyer(
    engine, session, clock, settings, notifier, sender
):
    """The desk stayed for sale after the chair's withdrawal, so when it is withdrawn too, Mei
    asks about it like anyone else."""
    desk, thread, on_chair, control = _bundle_less_the_chair(
        engine, session, clock, settings, notifier, sender, chair_thread=True
    )
    control.withdraw(desk.id)
    session.expire_all()
    _say(session, clock, thread, "Is the desk still available?")
    _say(session, clock, on_chair, "And the chair?")  # its withdrawal cancelled her deal

    assert _send(session, clock, settings, notifier, {"fake": sender})[0] == 1
    assert sender.sent == [(thread.remote_id, COURTESY_REPLIES[ItemStatus.WITHDRAWN])]


def test_a_partners_withdrawal_in_the_same_thread_leaves_an_ordinary_buyer(
    engine, session, clock, settings, notifier, sender
):
    """The chair's row sits on Mei's desk thread; it says nothing about the desk."""
    desk, thread, _, control = _bundle_less_the_chair(
        engine, session, clock, settings, notifier, sender, chair_thread=False
    )
    control.mark_sold(desk.id, 11_000)
    session.expire_all()
    _say(session, clock, thread, "Is the desk still available?")

    assert _send(session, clock, settings, notifier, {"fake": sender})[0] == 1
    assert sender.sent == [(thread.remote_id, SOLD_REPLY)]


def test_never_on_a_paused_item(session, clock, settings, notifier, sender):
    item = _item(session, clock, paused=True)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})
    assert isinstance(validate_courtesy(ctx, TOOL, args), Reject)


def test_old_threads_are_not_revived(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    _say(session, clock, _thread(session, clock, item, "ann"), ago=COURTESY_WINDOW + timedelta(1))

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])


def test_a_disconnected_platform_is_skipped(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    _say(session, clock, _thread(session, clock, item, "ann"))

    assert _send(session, clock, settings, notifier, {"kijiji": Sender("kijiji")}) == (0, [])


# ---------------------------------------------------------------- outbound cap


def test_the_reply_counts_against_the_outbound_cap(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    first, second = _thread(session, clock, item, "ann"), _thread(session, clock, item, "bo")
    _say(session, clock, first)
    _say(session, clock, second)
    settings.daily_outbound_cap["fake"] = 1

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (1, [])
    assert outbound_count_today(session, clock, "fake") == 1
    assert len(sender.sent) == 1  # the cap stopped the second thread's reply

    midnight = clock.now().replace(hour=0, minute=0, second=0, microsecond=0)
    clock.set(midnight + timedelta(days=1, minutes=1))
    assert _send(session, clock, settings, notifier, {"fake": sender}) == (1, [])
    assert {to for to, _ in sender.sent} == {first.remote_id, second.remote_id}


def test_a_full_cap_holds_the_reply_back(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    settings.daily_outbound_cap["fake"] = 0

    assert _send(session, clock, settings, notifier, {"fake": sender}) == (0, [])
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})
    verdict = validate_courtesy(ctx, TOOL, args)
    assert isinstance(verdict, Reject) and "daily limit" in verdict.reason


# ---------------------------------------------------------------- the guard


def test_the_sold_block_is_unchanged_for_the_agent_and_the_system(
    session, clock, settings, notifier, sender
):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})

    for actor in ("agent", "system"):
        ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender}, actor=actor)
        verdict = validate(ctx, TOOL, args)
        assert isinstance(verdict, Reject) and "sold" in verdict.reason


@pytest.mark.parametrize("actor", ["agent", "human"])
def test_only_the_system_may_use_the_courtesy_path(
    session, clock, settings, notifier, sender, actor
):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender}, actor=actor)

    assert isinstance(validate_courtesy(ctx, TOOL, args), Reject)


@pytest.mark.parametrize(
    "body",
    [
        "Sorry, it's sold now. Thanks for asking! I have a lamp for $5 too.",
        COURTESY_REPLIES[ItemStatus.WITHDRAWN],  # the wording for the other status
        "Sold, but text me at 416-555-0199",
    ],
)
def test_only_the_exact_reply_passes(session, clock, settings, notifier, sender, body):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    args = TOOL.parse_args({"thread_id": thread.id, "body": body})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})

    verdict = validate_courtesy(ctx, TOOL, args)

    assert isinstance(verdict, Reject) and "only" in verdict.reason


def test_the_courtesy_path_needs_a_closed_item(session, clock, settings, notifier, sender):
    item = _item(session, clock, ItemStatus.LISTED)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})

    assert isinstance(validate_courtesy(ctx, TOOL, args), Reject)


def test_the_guard_refuses_a_second_reply(session, clock, settings, notifier, sender):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    _send(session, clock, settings, notifier, {"fake": sender})
    clock.advance(hours=1)
    _say(session, clock, thread, "Hello?")
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})

    verdict = validate_courtesy(ctx, TOOL, args)

    assert isinstance(verdict, Reject) and "already" in verdict.reason


def test_an_allowed_courtesy_reply_still_passes_the_text_rules(
    session, clock, settings, notifier, sender
):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    args = TOOL.parse_args({"thread_id": thread.id, "body": SOLD_REPLY})
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": sender})

    assert validate_courtesy(ctx, TOOL, args) == Allow()


# ---------------------------------------------------------------- failures and the runner


def test_a_failed_send_leaves_the_thread_for_the_next_try(
    session, clock, settings, notifier, sender
):
    item = _item(session, clock)
    thread = _thread(session, clock, item, "ann")
    _say(session, clock, thread)
    sender.fail = PlatformError("fake: 503")

    sent, errors = _send(session, clock, settings, notifier, {"fake": sender})

    assert sent == 0 and "503" in errors[0]
    assert _outbound(session, thread) == []
    assert _send(session, clock, settings, notifier, {"fake": sender}) == (1, [])


def test_the_runner_replies_without_a_model_call(engine, clock, settings, notifier, sender):
    provider = ScriptedProvider(lambda request: pytest.fail("no model call for a courtesy reply"))
    router = ProviderRouter([provider], QuotaTracker(engine, clock, {}))
    registry = default_registry()
    platforms = {"fake": _NoIngest(sender)}
    agent = Agent(router, registry, settings, clock, notifier, platforms)
    runner = Runner(engine, clock, settings, platforms, agent, router, notifier, registry)
    with Session(engine, expire_on_commit=False) as session:
        item = _item(session, clock)
        _say(session, clock, _thread(session, clock, item, "ann"))

    first, second = runner.tick(), runner.tick()

    assert (first.courtesy_replies, second.courtesy_replies) == (1, 0)
    assert [body for _, body in sender.sent] == [SOLD_REPLY]
    assert provider.requests == []


class _NoIngest:
    """Sender plus the read side of an adapter that has nothing to report."""

    def __init__(self, sender: Sender) -> None:
        self.sender, self.name = sender, sender.name

    def send_message(self, thread_remote_id: str, body: str) -> str:
        return self.sender.send_message(thread_remote_id, body)

    def fetch_threads(self, since):
        return []

    def delist(self, ref) -> None:
        pass

    def search(self, query, limit=20):
        return []
