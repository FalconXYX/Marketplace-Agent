import itertools
from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import event
from sqlmodel import select

from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    Buyer,
    Direction,
    Firmness,
    Item,
    ItemStatus,
    Message,
    Thread,
    ThreadState,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.invariants import ADDRESS_PLACEHOLDER
from marketplace_agent.guard.validate import validate
from marketplace_agent.platforms.base import (
    ListingStats,
    PlatformAuthError,
    PlatformError,
    PlatformRef,
)
from marketplace_agent.tools.base import Allow, Reject, ToolContext
from marketplace_agent.tools.impl.messaging import SendMessageArgs, SendMessageTool

LIST, FLOOR = 12_000, 9_500
ADDRESS = "123 Example St"  # conftest settings.pickup_address
TOOL = SendMessageTool()


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

    def called(self, method: str) -> list[object]:
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
        return f"m{next(self._ids)}"

    def stats(self, ref):
        return ListingStats()

    def search(self, query, limit=20):
        return []

    def healthcheck(self):
        return []


@pytest.fixture
def adapter() -> StubAdapter:
    return StubAdapter()


@pytest.fixture
def item(session, clock) -> Item:
    return _item(session, clock)


@pytest.fixture
def thread(session, clock, item) -> Thread:
    return _thread(session, clock, item, "ann")


def _item(session, clock, firmness: Firmness = Firmness.SLIGHT) -> Item:
    item = Item(
        title="IKEA desk", status=ItemStatus.LISTED, created_at=clock.now(), updated_at=clock.now()
    )
    session.add(item)
    session.flush()
    set_goal(
        session, clock, item.id, list_price_cents=LIST, floor_price_cents=FLOOR, firmness=firmness
    )
    session.commit()
    return item


def _thread(session, clock, item: Item, handle: str, platform: str = "fake") -> Thread:
    buyer = Buyer(platform=platform, handle=handle, display_name=handle.title())
    session.add(buyer)
    session.flush()
    thread = Thread(
        platform=platform,
        remote_id=f"rt-{item.id}-{handle}",
        item_id=item.id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    _say(session, clock, thread, "Hi, is this still available?", minutes_ago=60)
    return thread


def _say(session, clock, thread: Thread, body: str, minutes_ago: int = 0) -> None:
    at = clock.now() - timedelta(minutes=minutes_ago)
    session.add(Message(thread_id=thread.id, direction=Direction.INBOUND, body=body, sent_at=at))
    session.commit()


def _outbound(session, clock, thread: Thread, via: str = "agent") -> None:
    session.add(
        Message(
            thread_id=thread.id,
            direction=Direction.OUTBOUND,
            body="hello",
            sent_at=clock.now(),
            via=via,
        )
    )
    session.commit()


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


@pytest.fixture
def ctx(session, clock, settings, notifier, item, adapter) -> ToolContext:
    return _ctx(session, clock, settings, notifier, item, {"fake": adapter})


def _args(thread: Thread, body: str = "Yes, still available! Pickup near Bloor and Ossington."):
    return SendMessageArgs(thread_id=thread.id, body=body)


def _reject(ctx: ToolContext, thread: Thread, body: str | None = None) -> str:
    verdict = TOOL.check(ctx, _args(thread) if body is None else _args(thread, body))
    assert isinstance(verdict, Reject), verdict
    return verdict.reason


def _send(ctx: ToolContext, thread: Thread, body: str | None = None):
    args = _args(thread) if body is None else _args(thread, body)
    assert TOOL.check(ctx, args) == Allow()
    result = TOOL.execute(ctx, args)
    ctx.session.commit()
    return result


def _outbound_rows(session, thread: Thread) -> list[Message]:
    query = select(Message).where(
        Message.thread_id == thread.id, Message.direction == Direction.OUTBOUND
    )
    return list(session.exec(query).all())


def _book_past_undo(session, clock, thread: Thread) -> None:
    accepted = ledger.accept(
        session,
        clock,
        item_id=thread.item_id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=11_000,
        undo_minutes=10,
    )
    start = clock.now() + timedelta(hours=6)
    ledger.book_pickup(
        session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(hours=1)
    )
    session.commit()
    clock.advance(minutes=11)


# ---------------------------------------------------------------- happy path


def test_send_delivers_and_stores_the_message(ctx, session, clock, thread, adapter):
    result = _send(ctx, thread)

    assert result.ok
    assert adapter.called("send_message") == [(thread.remote_id, _args(thread).body)]
    [row] = _outbound_rows(session, thread)
    assert row.body == _args(thread).body
    assert row.via == "agent" and row.remote_id == "m1" and row.handled
    assert thread.last_outbound_at == clock.now()
    assert result.data == {"message_id": row.id, "thread_id": thread.id}
    assert "Ann" in result.summary and _args(thread).body in result.summary


def test_trust_platform_is_the_threads_platform(ctx, session, clock, item):
    kijiji_thread = _thread(session, clock, item, "bo", platform="kijiji")
    assert TOOL.trust_platform(ctx, _args(kijiji_thread)) == "kijiji"
    assert TOOL.trust_action == "send_message"


def test_validate_allows_an_agent_message(ctx, thread):
    assert validate(ctx, TOOL, _args(thread)) == Allow()


def test_check_writes_nothing(ctx, session, thread):
    flushes = []
    event.listen(session, "after_flush", lambda *_: flushes.append(1))
    TOOL.check(ctx, _args(thread))
    TOOL.check(ctx, _args(thread, "call me at 416-555-0199"))
    assert flushes == [] and not session.new and not session.dirty


def test_unknown_keys_are_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        TOOL.parse_args({"thread_id": 1, "body": "hi", "urgent": True})


# ---------------------------------------------------------------- thread and platform rejects


def test_rejects_a_thread_of_another_item(ctx, session, clock):
    other = _thread(session, clock, _item(session, clock), "zed")
    reason = _reject(ctx, other)
    assert "not a conversation about this item" in reason
    assert "Use a thread id from the Threads section" in reason


def test_rejects_an_unknown_thread(ctx):
    verdict = TOOL.check(ctx, SendMessageArgs(thread_id=999, body="hi"))
    assert isinstance(verdict, Reject) and "Use a thread id" in verdict.reason


def test_rejects_a_frozen_thread(ctx, session, thread):
    thread.state = ThreadState.FROZEN
    thread.frozen_reason = "asks for a verification code"
    session.commit()
    reason = _reject(ctx, thread)
    assert "frozen (asks for a verification code)" in reason
    assert "Never message this buyer" in reason and "escalate" in reason


def test_rejects_a_flagged_buyer(ctx, session, thread):
    session.get(Buyer, thread.buyer_id).flagged_scam = True
    session.commit()
    reason = _reject(ctx, thread)
    assert "flagged as a scammer" in reason and "Never message this buyer" in reason


def test_rejects_when_the_platform_is_not_connected(
    session, clock, settings, notifier, item, thread
):
    ctx = _ctx(session, clock, settings, notifier, item, {})
    reason = _reject(ctx, thread)
    assert "fake is not connected" in reason and "Try again on a later wake" in reason


# ---------------------------------------------------------------- outbound cap


def test_outbound_cap_counts_automated_messages_only(session, clock, settings, notifier, item):
    capped = settings.model_copy(update={"daily_outbound_cap": {"fake": 2}})
    ctx = _ctx(session, clock, capped, notifier, item, {"fake": StubAdapter()})
    target = _thread(session, clock, item, "ann")
    _outbound(session, clock, _thread(session, clock, item, "bo"), via="human")
    _outbound(session, clock, _thread(session, clock, item, "cy"), via="system")
    assert TOOL.check(ctx, _args(target)) == Allow()

    _outbound(session, clock, _thread(session, clock, item, "di"), via="agent")
    reason = _reject(ctx, target)

    assert "daily limit of 2 messages on fake is used up (2 sent today)" in reason
    assert "Send nothing more on this platform until Thu Oct 1 20:00" in reason


def test_outbound_cap_resets_at_utc_midnight(session, clock, settings, notifier, item):
    capped = settings.model_copy(update={"daily_outbound_cap": {"fake": 1}})
    ctx = _ctx(session, clock, capped, notifier, item, {"fake": StubAdapter()})
    target = _thread(session, clock, item, "ann")
    _outbound(session, clock, _thread(session, clock, item, "bo"))
    assert isinstance(TOOL.check(ctx, _args(target)), Reject)

    clock.advance(hours=8)  # 00:00 UTC
    assert TOOL.check(ctx, _args(target)) == Allow()


def test_a_platform_missing_from_the_cap_table_allows_nothing(
    session, clock, settings, notifier, item
):
    capped = settings.model_copy(update={"daily_outbound_cap": {}})
    ctx = _ctx(session, clock, capped, notifier, item, {"fake": StubAdapter()})
    reason = _reject(ctx, _thread(session, clock, item, "ann"))
    assert "daily limit of 0 messages on fake" in reason


# ---------------------------------------------------------------- follow-up streak


def test_follow_up_streak_rule(ctx, session, clock, thread):
    assert _send(ctx, thread).ok  # reply to the buyer

    reason = _reject(ctx, thread)
    assert "Your last message to this buyer (Thu Oct 1 12:00) has no reply yet" in reason
    assert "one follow-up is allowed from Fri Oct 2 12:00" in reason

    clock.advance(hours=23, minutes=59)
    assert isinstance(TOOL.check(ctx, _args(thread)), Reject)
    clock.advance(minutes=1)
    assert _send(ctx, thread, "Just checking in, still interested?").ok  # the one follow-up

    clock.advance(days=3)
    reason = _reject(ctx, thread)
    assert "2 messages in a row without a reply" in reason
    assert "Do not message them again until they write back" in reason

    _say(session, clock, thread, "Sorry, was away. Still have it?")
    assert _send(ctx, thread).ok


def test_human_messages_count_toward_the_streak(ctx, session, clock, thread):
    _outbound(session, clock, thread, via="human")
    assert "has no reply yet" in _reject(ctx, thread)


# ---------------------------------------------------------------- text rules


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Text me at 416-555-0199", "remove it"),
        ("Email me at seller@example.com", "Keep the conversation on the platform"),
        ("I could do $80 for it.", "Do not restate numbers under the floor"),
        ("Send me the 6 digit code Google texted you", "just say it is cash at pickup only"),
        ("   ", "Write what you want to say"),
        ("x" * 1001, "Say it in fewer words"),
        (f"Come to {ADDRESS}", f"write {ADDRESS_PLACEHOLDER}"),
        (f"I'm at {ADDRESS_PLACEHOLDER}", "accept an offer first"),
    ],
)
def test_text_rules(ctx, thread, body, expected):
    assert expected in _reject(ctx, thread, body)


def test_firm_price_rejects_any_lower_number(session, clock, settings, notifier, adapter):
    item = _item(session, clock, Firmness.FIRM)
    thread = _thread(session, clock, item, "ann")
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": adapter})

    reason = _reject(ctx, thread, "Best I can do is $110.")
    assert "below the firm price of $120.00" in reason
    assert "decline politely without naming any lower number" in reason
    assert TOOL.check(ctx, _args(thread, "Sorry, it's firm at $120.")) == Allow()


def test_placeholder_waits_for_the_undo_window(ctx, session, clock, thread):
    accepted = ledger.accept(
        session,
        clock,
        item_id=thread.item_id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=11_000,
        undo_minutes=10,
    )
    session.commit()
    assert "book one with book_pickup first" in _reject(ctx, thread, ADDRESS_PLACEHOLDER)

    start = clock.now() + timedelta(hours=6)
    ledger.book_pickup(
        session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(hours=1)
    )
    session.commit()
    assert "the owner can still undo the deal until 12:10" in _reject(
        ctx, thread, ADDRESS_PLACEHOLDER
    )


# ---------------------------------------------------------------- address substitution


def test_placeholder_is_filled_in_only_for_the_platform(
    ctx, session, clock, notifier, thread, adapter
):
    _book_past_undo(session, clock, thread)
    body = f"Great, see you then! The address is {ADDRESS_PLACEHOLDER}."

    result = _send(ctx, thread, body)

    assert adapter.called("send_message") == [
        (thread.remote_id, f"Great, see you then! The address is {ADDRESS}.")
    ]
    [row] = _outbound_rows(session, thread)
    assert row.body == body
    assert ADDRESS_PLACEHOLDER in result.summary and ADDRESS not in result.summary
    assert ADDRESS not in repr(result.data)
    assert all(ADDRESS not in f"{n.title} {n.body}" for n in notifier.notices)


def test_a_failed_send_never_echoes_the_address(ctx, session, clock, thread, adapter):
    _book_past_undo(session, clock, thread)
    adapter.fail["send_message"] = PlatformAuthError(f"rejected body: Meet at {ADDRESS}")

    with pytest.raises(PlatformAuthError) as raised:
        TOOL.execute(ctx, _args(thread, f"Meet at {ADDRESS_PLACEHOLDER}"))

    assert ADDRESS not in str(raised.value)
    assert ADDRESS_PLACEHOLDER in str(raised.value)
    assert raised.value.__cause__ is None


def test_placeholder_needs_a_configured_address(session, clock, settings, notifier, item, thread):
    _book_past_undo(session, clock, thread)
    blank = settings.model_copy(update={"pickup_address": ""})
    ctx = _ctx(session, clock, blank, notifier, item, {"fake": StubAdapter()})
    reason = _reject(ctx, thread, f"Meet at {ADDRESS_PLACEHOLDER}")
    assert "No pickup address is configured" in reason and "escalate" in reason


# ---------------------------------------------------------------- platform failure


def test_platform_failure_raises_and_rolls_back(ctx, session, thread, adapter):
    adapter.fail["send_message"] = PlatformError("network down")

    with pytest.raises(PlatformError, match="network down"):
        TOOL.execute(ctx, _args(thread))
    session.rollback()

    assert _outbound_rows(session, thread) == []
    assert session.get(Thread, thread.id).last_outbound_at is None


# ---------------------------------------------------------------- human and system actors


def test_the_owner_skips_the_automation_rules(session, clock, settings, notifier, item, thread):
    capped = settings.model_copy(update={"daily_outbound_cap": {"fake": 0}})
    adapter = StubAdapter()
    human = _ctx(session, clock, capped, notifier, item, {"fake": adapter}, actor="human")
    thread.state = ThreadState.FROZEN
    session.get(Buyer, thread.buyer_id).flagged_scam = True
    _outbound(session, clock, thread)
    _outbound(session, clock, thread)

    result = _send(human, thread, f"Come by {ADDRESS} after six.")

    assert result.ok
    assert adapter.called("send_message") == [(thread.remote_id, f"Come by {ADDRESS} after six.")]
    stored = _outbound_rows(session, thread)[-1]
    assert stored.body == f"Come by {ADDRESS_PLACEHOLDER} after six." and stored.via == "human"
    assert ADDRESS not in result.summary


def test_the_owner_may_use_the_placeholder_anytime(
    session, clock, settings, notifier, item, thread
):
    adapter = StubAdapter()
    human = _ctx(session, clock, settings, notifier, item, {"fake": adapter}, actor="human")
    _send(human, thread, f"It's at {ADDRESS_PLACEHOLDER}")
    assert adapter.called("send_message") == [(thread.remote_id, f"It's at {ADDRESS}")]


def test_the_owner_still_gets_the_hard_text_rules(session, clock, settings, notifier, item, thread):
    human = _ctx(session, clock, settings, notifier, item, {"fake": StubAdapter()}, actor="human")
    assert "remove it" in _reject(human, thread, "Call 416-555-0199")


def test_system_replies_follow_the_automation_rules(
    session, clock, settings, notifier, item, thread
):
    system = _ctx(session, clock, settings, notifier, item, {"fake": StubAdapter()}, actor="system")
    assert _send(system, thread).ok
    assert _outbound_rows(session, thread)[0].via == "system"
    assert "has no reply yet" in _reject(system, thread)
