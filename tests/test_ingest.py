"""runner.ingest: platform threads into the database, idempotently and safely."""

from datetime import datetime, timedelta

import pytest
from sqlmodel import Session, func, select

from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    Buyer,
    Direction,
    Escalation,
    Item,
    ItemStatus,
    Listing,
    ListingStatsSnapshot,
    ListingStatus,
    Message,
    Thread,
    ThreadState,
    WakeRequest,
)
from marketplace_agent.guard.invariants import ADDRESS_PLACEHOLDER, leaks_address, outbound_streak
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import (
    ListingPayload,
    ListingStats,
    PlatformAuthError,
    PlatformError,
    RemoteMessage,
    RemoteThread,
)
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.runner.ingest import ECHO_WINDOW, ingest, snapshot_stats
from marketplace_agent.tools.base import Allow, ToolContext
from marketplace_agent.tools.impl.messaging import SendMessageArgs, SendMessageTool

SCAM = "I will text you a 6 digit code, please send it back to me"


class StubAdapter:
    """Returns fixed threads and stats; fail[method] raises once."""

    name = "fake"

    def __init__(self, threads: list[RemoteThread] | None = None) -> None:
        self.threads = threads or []
        self.views: dict[str, ListingStats] = {}
        self.fail: dict[str, Exception] = {}

    def fetch_threads(self, since: datetime | None) -> list[RemoteThread]:
        if "fetch_threads" in self.fail:
            raise self.fail.pop("fetch_threads")
        return self.threads

    def stats(self, ref):
        if ref.remote_id in self.fail:
            raise self.fail.pop(ref.remote_id)
        return self.views.get(ref.remote_id, ListingStats())


@pytest.fixture
def market(clock) -> FakeMarketplace:
    return FakeMarketplace(clock)


def _listed(session: Session, clock, market: FakeMarketplace) -> tuple[int, str]:
    """A LISTED item with a goal and a live listing on the fake market."""
    now = clock.now()
    item = Item(
        title="IKEA desk",
        description="Solid.",
        status=ItemStatus.LISTED,
        created_at=now,
        updated_at=now,
    )
    session.add(item)
    session.flush()
    set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
    payload = ListingPayload("IKEA desk", "Solid.", 12_000, [], "Bloor & Ossington")
    ref = market.publish(payload)
    session.add(
        Listing(
            item_id=item.id,
            platform="fake",
            remote_id=ref.remote_id,
            url=ref.url,
            status=ListingStatus.LIVE,
            price_cents=12_000,
            title="IKEA desk",
            description="Solid.",
            published_at=now,
        )
    )
    session.commit()
    return item.id, ref.remote_id


def _run(session, clock, settings, notifier, adapter, since=None):
    result = ingest(session, clock, settings, notifier, adapter, since)
    session.commit()
    return result


def _count(session: Session, model) -> int:
    return session.exec(select(func.count()).select_from(model)).one()


def _messages(session: Session, thread_id: int | None = None) -> list[Message]:
    query = select(Message).order_by(Message.id)
    if thread_id is not None:
        query = query.where(Message.thread_id == thread_id)
    return list(session.exec(query).all())


def _thread(session: Session) -> Thread:
    return session.exec(select(Thread)).one()


def _remote(*messages: RemoteMessage, listing: str | None = "fake-l1", rid="t1") -> RemoteThread:
    return RemoteThread(rid, listing, "maya7", "Maya K.", list(messages))


# ---------------------------------------------------------------- threads and buyers


def test_new_thread_maps_to_item_and_creates_buyer(session, clock, settings, notifier, market):
    item_id, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, "Hi there, does it wobble at all?")

    result = _run(session, clock, settings, notifier, market)

    thread = _thread(session)
    buyer = session.get(Buyer, thread.buyer_id)
    [message] = _messages(session)
    assert (result.new_threads, result.new_inbound, result.errors) == (1, [message.id], [])
    assert (thread.item_id, thread.platform, thread.remote_id) == (item_id, "fake", remote)
    assert thread.state == ThreadState.OPEN
    assert thread.last_inbound_at == clock.now()
    assert buyer.platform == "fake" and buyer.display_name
    assert message.direction == Direction.INBOUND
    assert (message.handled, message.via) == (False, "buyer")
    assert message.remote_id is not None


def test_thread_for_unknown_listing_has_no_item(session, clock, settings, notifier):
    message = RemoteMessage("m1", "Still available?", clock.now(), from_buyer=True)
    adapter = StubAdapter([_remote(message, listing="gone-listing")])

    _run(session, clock, settings, notifier, adapter)

    assert _thread(session).item_id is None


def test_thread_without_item_is_linked_once_its_listing_is_known(
    session, clock, settings, notifier, market
):
    message = RemoteMessage("m1", "Still available?", clock.now(), from_buyer=True)
    adapter = StubAdapter([_remote(message, listing="fake-l1")])
    _run(session, clock, settings, notifier, adapter)
    assert _thread(session).item_id is None

    item_id, listing = _listed(session, clock, market)
    assert listing == "fake-l1"
    _run(session, clock, settings, notifier, adapter)

    assert _thread(session).item_id == item_id


def test_buyer_is_upserted_and_renamed(session, clock, settings, notifier):
    first = RemoteMessage("m1", "Hi", clock.now(), from_buyer=True)
    adapter = StubAdapter([_remote(first)])
    _run(session, clock, settings, notifier, adapter)
    adapter.threads = [
        RemoteThread(
            "t2",
            None,
            "maya7",
            "Maya Kim",
            [RemoteMessage("m2", "Hello again", clock.now(), from_buyer=True)],
        )
    ]

    _run(session, clock, settings, notifier, adapter)

    [buyer] = session.exec(select(Buyer)).all()
    assert buyer.display_name == "Maya Kim"
    assert _count(session, Thread) == 2


# ---------------------------------------------------------------- idempotency


def test_same_platform_state_twice_creates_nothing_new(session, clock, settings, notifier, market):
    _, listing = _listed(session, clock, market)
    for text in ("Is this still available?", "Would you take $100?"):
        thread = market.add_buyer("scripted", listing)
        market.buyer_say(thread, text)
    first = _run(session, clock, settings, notifier, market)
    counts = [_count(session, model) for model in (Thread, Buyer, Message, Escalation)]

    again = _run(session, clock, settings, notifier, market)
    since = _run(session, clock, settings, notifier, market, since=clock.now())

    assert (first.new_threads, len(first.new_inbound)) == (2, 2)
    for repeat in (again, since):
        assert (repeat.new_threads, repeat.new_inbound, repeat.frozen_threads) == (0, [], [])
    assert [_count(session, model) for model in (Thread, Buyer, Message, Escalation)] == counts


def test_since_brings_only_newer_messages(session, clock, settings, notifier, market):
    _, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, "Hi, does it come apart?")
    _run(session, clock, settings, notifier, market)
    checkpoint = clock.advance(minutes=30)
    market.buyer_say(remote, "Also, how tall is it?")

    result = _run(session, clock, settings, notifier, market, since=checkpoint)

    assert len(result.new_inbound) == 1
    assert [m.body for m in _messages(session)] == [
        "Hi, does it come apart?",
        "Also, how tall is it?",
    ]


# ---------------------------------------------------------------- inbound


def test_address_is_redacted_from_stored_inbound(session, clock, settings, notifier, market):
    _, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, "Is pickup at 123 Example St? I know the area.")

    _run(session, clock, settings, notifier, market)

    [message] = _messages(session)
    assert "123 Example" not in message.body
    assert ADDRESS_PLACEHOLDER in message.body


def test_offer_moves_open_thread_to_negotiating(session, clock, settings, notifier, market):
    _, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, "Would you take $100?")

    _run(session, clock, settings, notifier, market)

    thread = _thread(session)
    assert (thread.state, thread.last_offer_cents) == (ThreadState.NEGOTIATING, 10_000)


def test_question_keeps_thread_open(session, clock, settings, notifier, market):
    _, listing = _listed(session, clock, market)
    market.buyer_say(market.add_buyer("scripted", listing), "What are the dimensions?")

    _run(session, clock, settings, notifier, market)

    thread = _thread(session)
    assert (thread.state, thread.last_offer_cents) == (ThreadState.OPEN, None)


def test_closed_thread_reopens_when_buyer_writes(session, clock, settings, notifier, market):
    _, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, "What are the dimensions?")
    _run(session, clock, settings, notifier, market)
    thread = _thread(session)
    thread.state = ThreadState.CLOSED
    session.commit()
    clock.advance(days=4)
    market.buyer_say(remote, "Sorry, been busy. Still have it?")

    _run(session, clock, settings, notifier, market)

    session.refresh(thread)
    assert thread.state == ThreadState.NEGOTIATING
    assert thread.last_inbound_at == clock.now()


# ---------------------------------------------------------------- scams


def test_scam_message_freezes_thread_flags_buyer_and_escalates(
    session, clock, settings, notifier, market
):
    item_id, listing = _listed(session, clock, market)
    market.buyer_say(market.add_buyer("scripted", listing), SCAM)

    result = _run(session, clock, settings, notifier, market)

    thread = _thread(session)
    buyer = session.get(Buyer, thread.buyer_id)
    [message] = _messages(session)
    [escalation] = session.exec(select(Escalation)).all()
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert result.frozen_threads == [thread.id]
    assert thread.state == ThreadState.FROZEN
    assert "verification code" in thread.frozen_reason
    assert buyer.flagged_scam
    assert (message.handled, message.intent) == (True, "scam")
    assert (escalation.item_id, escalation.thread_id) == (item_id, thread.id)
    assert alert.item_id == item_id and alert.ref_id == escalation.id


def test_frozen_thread_stores_later_messages_handled_without_new_alerts(
    session, clock, settings, notifier, market
):
    _, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, SCAM)
    _run(session, clock, settings, notifier, market)
    clock.advance(minutes=5)
    market.buyer_say(remote, "Hello?? Would you take $100?")

    _run(session, clock, settings, notifier, market)
    _run(session, clock, settings, notifier, market)

    later = _messages(session)[-1]
    assert later.handled and later.intent is None
    assert _thread(session).last_offer_cents is None
    assert _count(session, Escalation) == 1
    assert len(notifier.of_kind(NoticeKind.ALERT)) == 1


def test_flagged_buyer_new_thread_is_frozen(session, clock, settings, notifier):
    scam = RemoteMessage("m1", SCAM, clock.now(), from_buyer=True)
    adapter = StubAdapter([_remote(scam, rid="t1")])
    _run(session, clock, settings, notifier, adapter)
    hello = RemoteMessage("m2", "Hi, is the lamp available?", clock.now(), from_buyer=True)
    adapter.threads = [_remote(hello, rid="t2")]

    result = _run(session, clock, settings, notifier, adapter)

    second = session.exec(select(Thread).where(Thread.remote_id == "t2")).one()
    assert second.state == ThreadState.FROZEN
    assert result.frozen_threads == [second.id]
    assert all(m.handled for m in _messages(session, second.id))


def test_warn_level_signal_does_not_freeze(session, clock, settings, notifier, market):
    _, listing = _listed(session, clock, market)
    market.buyer_say(market.add_buyer("scripted", listing), "can you ship it to me?")

    _run(session, clock, settings, notifier, market)

    assert _thread(session).state == ThreadState.OPEN
    assert not _messages(session)[0].handled


# ---------------------------------------------------------------- outbound echoes


def _send(session, clock, settings, notifier, market, item_id, thread_id, body) -> Message:
    ctx = ToolContext(session, clock, settings, {"fake": market}, notifier, item_id)
    result = SendMessageTool().execute(ctx, SendMessageArgs(thread_id=thread_id, body=body))
    session.commit()
    return session.get(Message, result.data["message_id"])


def test_echo_of_our_own_message_is_not_stored_twice(session, clock, settings, notifier, market):
    item_id, listing = _listed(session, clock, market)
    market.buyer_say(market.add_buyer("scripted", listing), "What are the dimensions?")
    _run(session, clock, settings, notifier, market)
    sent = _send(
        session, clock, settings, notifier, market, item_id, _thread(session).id, "120 x 60 cm."
    )

    _run(session, clock, settings, notifier, market)

    outbound = [m for m in _messages(session) if m.direction == Direction.OUTBOUND]
    assert [m.id for m in outbound] == [sent.id]
    assert outbound[0].via == "agent"


def test_a_reply_stamped_before_our_message_is_still_the_buyers_turn(
    session, clock, settings, notifier, market
):
    """Platforms stamp buyer messages on their own clock, often to the minute, so a reply 40s
    after our counter can carry an earlier time than the counter."""
    item_id, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, "Would you take $90?")
    _run(session, clock, settings, notifier, market)
    thread_id = _thread(session).id
    clock.advance(seconds=30)
    counter = _send(
        session, clock, settings, notifier, market, item_id, thread_id, "Could do $110."
    )
    clock.advance(seconds=40)
    minute = counter.sent_at.replace(second=0, microsecond=0)
    market.buyer_say(remote, "Ok $110 works, deal!", at=minute)

    _run(session, clock, settings, notifier, market)

    assert outbound_streak(session, thread_id) == (0, None)
    thread = session.get(Thread, thread_id)
    assert thread.last_inbound_at > thread.last_outbound_at  # not a ghost
    ctx = ToolContext(session, clock, settings, {"fake": market}, notifier, item_id)
    reply = SendMessageArgs(thread_id=thread_id, body="Great, it's yours for $110.")
    assert SendMessageTool().check(ctx, reply) == Allow()


def test_echo_without_remote_id_matches_by_body_and_takes_its_id(
    session, clock, settings, notifier
):
    hello = RemoteMessage("m1", "Hi", clock.now(), from_buyer=True)
    adapter = StubAdapter([_remote(hello)])
    _run(session, clock, settings, notifier, adapter)
    thread = _thread(session)
    stored = Message(
        thread_id=thread.id,
        direction=Direction.OUTBOUND,
        body=f"See you at {ADDRESS_PLACEHOLDER}",
        sent_at=clock.now(),
        handled=True,
        via="agent",
    )
    session.add(stored)
    session.commit()
    echo = RemoteMessage(
        "m2", "See you at  123 Example St", clock.now() + timedelta(minutes=2), from_buyer=False
    )
    adapter.threads = [_remote(hello, echo)]

    _run(session, clock, settings, notifier, adapter)
    _run(session, clock, settings, notifier, adapter)

    outbound = [m for m in _messages(session) if m.direction == Direction.OUTBOUND]
    assert [(m.id, m.remote_id) for m in outbound] == [(stored.id, "m2")]


def test_seller_message_typed_on_the_platform_is_stored_as_human(
    session, clock, settings, notifier
):
    hello = RemoteMessage("m1", "Hi", clock.now(), from_buyer=True)
    adapter = StubAdapter([_remote(hello)])
    _run(session, clock, settings, notifier, adapter)
    thread = _thread(session)
    session.add(
        Message(
            thread_id=thread.id,
            direction=Direction.OUTBOUND,
            body="Yes, it's available.",
            sent_at=clock.now(),
            via="agent",
        )
    )
    session.commit()
    late = clock.now() + ECHO_WINDOW + timedelta(minutes=1)
    typed = RemoteMessage("m3", "Yes, it's available.", late, from_buyer=False)
    adapter.threads = [_remote(hello, typed)]

    _run(session, clock, settings, notifier, adapter)
    _run(session, clock, settings, notifier, adapter)

    human = [m for m in _messages(session) if m.via == "human"]
    assert len(human) == 1
    assert (human[0].direction, human[0].handled, human[0].remote_id) == (
        Direction.OUTBOUND,
        True,
        "m3",
    )
    session.refresh(thread)
    assert thread.last_outbound_at == late


@pytest.mark.parametrize(
    "typed",
    [
        "Pickup is at 123 Example St, ring the bell.",
        "come to 123 example street after 6",
        "It's 123  EXAMPLE ST. Side door.",
    ],
)
def test_seller_message_typed_on_the_platform_is_stored_with_the_address_redacted(
    session, clock, settings, notifier, market, typed
):
    """The owner typing the address on the platform is the likeliest way it reaches storage, and
    from there briefings and plans."""
    _, listing = _listed(session, clock, market)
    remote = market.add_buyer("scripted", listing)
    market.buyer_say(remote, "Where do I pick it up?")
    market.send_message(remote, typed)  # typed by the owner: nothing stored matches it

    _run(session, clock, settings, notifier, market)
    _run(session, clock, settings, notifier, market)

    [human] = [m for m in _messages(session) if m.via == "human"]
    assert human.direction == Direction.OUTBOUND and human.handled
    assert ADDRESS_PLACEHOLDER in human.body
    assert "123" not in human.body and "example" not in human.body.lower()
    assert not any(leaks_address(m.body, settings) for m in _messages(session))


def test_naive_timestamp_is_skipped_and_reported(session, clock, settings, notifier):
    naive = RemoteMessage("m1", "Hi", datetime(2026, 10, 1, 12, 0), from_buyer=True)
    good = RemoteMessage("m2", "Hello?", clock.now(), from_buyer=True)
    adapter = StubAdapter([_remote(naive, good)])

    result = _run(session, clock, settings, notifier, adapter)

    assert [m.remote_id for m in _messages(session)] == ["m2"]
    assert len(result.errors) == 1 and "m1" in result.errors[0]


def test_fetch_failure_propagates_and_stores_nothing(session, clock, settings, notifier):
    adapter = StubAdapter()
    adapter.fail["fetch_threads"] = PlatformError("timeout")

    with pytest.raises(PlatformError):
        ingest(session, clock, settings, notifier, adapter, None)


# ---------------------------------------------------------------- stats snapshots


def _snapshots(session: Session) -> list[ListingStatsSnapshot]:
    return list(session.exec(select(ListingStatsSnapshot).order_by(ListingStatsSnapshot.id)))


def test_snapshot_only_when_changed_or_stale(session, clock, settings, market):
    _, listing = _listed(session, clock, market)
    adapter = StubAdapter()
    adapter.views[listing] = ListingStats(views=10, saves=1)

    stored = [snapshot_stats(session, clock, adapter)]
    clock.advance(hours=5)
    stored.append(snapshot_stats(session, clock, adapter))
    adapter.views[listing] = ListingStats(views=12, saves=1)
    stored.append(snapshot_stats(session, clock, adapter))
    clock.advance(hours=6)
    stored.append(snapshot_stats(session, clock, adapter))

    assert stored == [1, 0, 1, 1]
    assert [s.views for s in _snapshots(session)] == [10, 12, 12]


def test_snapshot_skips_listings_that_are_not_live(session, clock, market):
    item_id, _ = _listed(session, clock, market)
    session.add(
        Listing(
            item_id=item_id,
            platform="fake",
            remote_id="fake-old",
            status=ListingStatus.DELISTED,
            price_cents=12_000,
            title="IKEA desk",
            description="Solid.",
        )
    )
    session.commit()

    assert snapshot_stats(session, clock, StubAdapter()) == 1


def test_snapshot_failure_skips_one_listing_but_auth_error_stops(session, clock, market):
    item_id, first = _listed(session, clock, market)
    session.add(
        Listing(
            item_id=item_id,
            platform="fake",
            remote_id="fake-l9",
            status=ListingStatus.LIVE,
            price_cents=12_000,
            title="IKEA desk",
            description="Solid.",
        )
    )
    session.commit()
    adapter = StubAdapter()
    adapter.fail[first] = PlatformError("not found")

    assert snapshot_stats(session, clock, adapter) == 1

    adapter.fail[first] = PlatformAuthError("signed out")
    clock.advance(hours=7)
    with pytest.raises(PlatformAuthError):
        snapshot_stats(session, clock, adapter)


def test_snapshot_marks_a_post_gone_from_the_platform_and_wakes_the_agent(
    session, clock, settings, notifier, market
):
    item_id, remote_id = _listed(session, clock, market)
    market.remove_listing(remote_id)

    assert snapshot_stats(session, clock, market, notifier) == 0
    session.commit()

    listing = session.exec(select(Listing)).one()
    assert listing.status == ListingStatus.DELISTED and listing.delisted_at == clock.now()
    assert listing.last_error == f"gone from fake: fake: listing {remote_id} not found"
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "The fake post is gone" and alert.item_id == item_id
    [wake] = session.exec(select(WakeRequest)).all()
    assert wake.item_id == item_id and "Relist or publish on fake" in wake.reason
    assert snapshot_stats(session, clock, market, notifier) == 0  # not LIVE any more
    assert len(notifier.of_kind(NoticeKind.ALERT)) == 1


@pytest.mark.parametrize("status", [ItemStatus.SOLD, ItemStatus.WITHDRAWN])
def test_a_gone_post_on_a_closed_item_is_quietly_done(session, clock, notifier, market, status):
    item_id, remote_id = _listed(session, clock, market)
    session.get(Item, item_id).status = status
    session.commit()
    market.remove_listing(remote_id)

    snapshot_stats(session, clock, market, notifier)
    session.commit()

    assert session.exec(select(Listing)).one().status == ListingStatus.DELISTED
    assert notifier.notices == []  # no "the agent can relist it" for a post meant to go
    assert session.exec(select(WakeRequest)).all() == []


def test_a_gone_post_on_a_promised_item_alerts_without_a_wake(session, clock, notifier, market):
    item_id, remote_id = _listed(session, clock, market)
    session.get(Item, item_id).status = ItemStatus.COMMITTED
    session.commit()
    market.remove_listing(remote_id)

    snapshot_stats(session, clock, market, notifier)
    session.commit()

    assert len(notifier.of_kind(NoticeKind.ALERT)) == 1
    assert session.exec(select(WakeRequest)).all() == []
