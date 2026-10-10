"""runner.tick: ingest, pattern cache, wakes, and housekeeping across many ticks.

Every trigger must fire once per cause: a trigger that fires on every tick spends the free-tier
quota in an hour, so these tests run multi-tick sequences on a FakeClock and count wakes exactly.
"""

import json
import threading
from datetime import timedelta

import pytest
from sqlmodel import Session, select

from marketplace_agent.agent.loop import Agent
from marketplace_agent.ai import pattern_cache
from marketplace_agent.ai.base import (
    Capability,
    LLMRequest,
    ProviderError,
    QuotaState,
    RateLimited,
)
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import ProviderLimit
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    AgentRun,
    ApprovalStatus,
    Buyer,
    Direction,
    Escalation,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    PendingApproval,
    Thread,
    ThreadState,
    WakeRequest,
)
from marketplace_agent.guard import ledger
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import ListingPayload, PlatformAuthError, PlatformError
from marketplace_agent.platforms.facebook.driver import PROFILE_IN_USE, ProfileInUse
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.runner.ingest import ingest
from marketplace_agent.runner.tick import (
    FAILED_WAKE_BACKOFF,
    NO_MODEL_ESCALATION,
    QUOTA_ESCALATION,
    Runner,
    TickReport,
)
from marketplace_agent.tools.base import ToolContext
from marketplace_agent.tools.impl.listing import PlatformArgs, RelistTool
from marketplace_agent.tools.registry import default_registry

AVAILABLE = "Is this still available?"
QUIET = {"rationale": "Nothing to do yet.", "actions": []}


class FixedStateRouter(ProviderRouter):
    """A real router whose quota state per capability is set by the test."""

    def __init__(self, providers, quota, states: dict[Capability, QuotaState]):
        super().__init__(providers, quota)
        self.states = states

    def state(self, capability: Capability = Capability.REASONING) -> QuotaState:
        return self.states.get(capability, QuotaState.HEALTHY)


class World:
    """A runner over a FakeMarketplace with a scripted model and the real tools."""

    def __init__(self, engine, clock, settings, notifier, script=None, states=None):
        self.engine, self.clock, self.settings, self.notifier = engine, clock, settings, notifier
        self.market = FakeMarketplace(clock)
        self.provider = ScriptedProvider(script or (lambda request: QUIET))
        quota = QuotaTracker(engine, clock, {})
        self.router = FixedStateRouter([self.provider], quota, states or {})
        registry = default_registry()
        platforms = {"fake": self.market}
        agent = Agent(self.router, registry, settings, clock, notifier, platforms)
        self.runner = Runner(
            engine, clock, settings, platforms, agent, self.router, notifier, registry
        )
        with Session(engine) as session:
            pattern_cache.seed(session, clock)
            session.commit()

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def item(self, title: str = "IKEA desk", *, last_run=None) -> tuple[int, str]:
        """A LISTED item with a goal and a live listing published now."""
        now = self.clock.now()
        with self.session() as session:
            item = Item(
                title=title,
                description="Solid.",
                status=ItemStatus.LISTED,
                created_at=now,
                updated_at=now,
                last_agent_run_at=last_run,
            )
            session.add(item)
            session.flush()
            set_goal(session, self.clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
            ref = self.market.publish(ListingPayload(title, "Solid.", 12_000, [], "Bloor"))
            session.add(
                Listing(
                    item_id=item.id,
                    platform="fake",
                    remote_id=ref.remote_id,
                    url=ref.url,
                    status=ListingStatus.LIVE,
                    price_cents=12_000,
                    title=title,
                    description="Solid.",
                    published_at=now,
                )
            )
            session.commit()
            return item.id, ref.remote_id

    def buyer(self, listing: str, *texts: str) -> str:
        thread = self.market.add_buyer("scripted", listing)
        for text in texts:
            self.market.buyer_say(thread, text)
        return thread

    def tick(self) -> TickReport:
        return self.runner.tick()

    def run(self, hours: float, every: timedelta) -> list[TickReport]:
        """Tick, then advance, until hours have passed."""
        reports = []
        end = self.clock.now() + timedelta(hours=hours)
        while self.clock.now() < end:
            reports.append(self.tick())
            self.clock.advance(every)
        return reports

    def wakes(self, reports: list[TickReport]) -> int:
        return sum(report.wakes for report in reports)

    def thread_of(self, remote_id: str) -> Thread:
        with self.session() as session:
            return session.exec(select(Thread).where(Thread.remote_id == remote_id)).one()

    def messages(self, remote_id: str) -> list[Message]:
        thread = self.thread_of(remote_id)
        with self.session() as session:
            query = select(Message).where(Message.thread_id == thread.id).order_by(Message.id)
            return list(session.exec(query).all())

    def update(self, model, row_id: int, **fields) -> None:
        with self.session() as session:
            row = session.get(model, row_id)
            for name, value in fields.items():
                setattr(row, name, value)
            session.commit()


@pytest.fixture
def world(engine, clock, settings, notifier) -> World:
    return World(engine, clock, settings, notifier)


def _world(engine, clock, settings, notifier, **kwargs) -> World:
    return World(engine, clock, settings, notifier, **kwargs)


def _item_ids_in_order(requests: list[LLMRequest]) -> list[int]:
    return [int(r.prompt.splitlines()[0].removeprefix("ITEM_ID: ")) for r in requests]


# ---------------------------------------------------------------- pattern cache


def test_first_contact_availability_is_answered_without_the_model(world):
    _, listing = world.item()
    remote = world.buyer(listing, AVAILABLE)

    first, second = world.tick(), world.tick()

    assert (first.ingested, first.cache_replies, first.wakes) == (1, 1, 0)
    assert (second.ingested, second.cache_replies, second.wakes) == (0, 0, 0)
    assert world.provider.requests == []
    [(to, body, _)] = world.market.sent
    assert to == remote
    assert body == "Hi! Yes, it's still available. Pickup is near Bloor St W & Ossington Ave."
    inbound, reply = world.messages(remote)
    assert (inbound.handled, inbound.intent) == (True, "availability")
    assert (reply.direction, reply.via, reply.body) == (Direction.OUTBOUND, "pattern_cache", body)


def test_availability_after_a_reply_goes_to_the_agent(world):
    _, listing = world.item()
    remote = world.buyer(listing, AVAILABLE)
    world.tick()
    world.clock.advance(minutes=10)
    world.market.buyer_say(remote, AVAILABLE)

    report = world.tick()

    assert (report.cache_replies, report.wakes) == (0, 1)
    assert len(world.market.sent) == 1


def test_two_unhandled_messages_go_to_the_agent(world):
    _, listing = world.item()
    world.buyer(listing, "Hi", AVAILABLE)

    report = world.tick()

    assert (report.cache_replies, report.wakes) == (0, 1)
    assert world.market.sent == []


def test_cache_never_answers_a_frozen_thread(world):
    _, listing = world.item()
    remote = world.buyer(listing, AVAILABLE)
    with world.session() as session:
        ingest(session, world.clock, world.settings, world.notifier, world.market, None)
        session.commit()
    world.update(Thread, world.thread_of(remote).id, state=ThreadState.FROZEN)

    report = world.tick()

    assert (report.cache_replies, report.wakes) == (0, 0)
    assert world.market.sent == []


def test_cache_says_pending_when_another_buyer_has_the_item(world):
    item_id, listing = world.item()
    holder = world.buyer(listing, "Deal at $110, I'll take it")
    world.tick()
    with world.session() as session:
        thread = session.exec(select(Thread).where(Thread.remote_id == holder)).one()
        ledger.accept(
            session,
            world.clock,
            item_id=item_id,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            price_cents=11_000,
            undo_minutes=10,
        )
        session.commit()
    world.buyer(listing, AVAILABLE)

    report = world.tick()

    assert report.cache_replies == 1
    assert world.market.sent[-1][1].startswith("Hi! It's pending with someone else")


def _bundle_desk_and_chair(world, desk_remote: str, desk_id: int, chair_id: int) -> int:
    """The desk buyer takes the desk and the chair as a bundle; both rows go on the desk thread.
    Returns the desk deal's ACCEPTED id."""
    thread = world.thread_of(desk_remote)
    with world.session() as session:
        rows = ledger.accept_bundle(
            session,
            world.clock,
            item_ids=[desk_id, chair_id],
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            total_cents=20_000,
            undo_minutes=0,
        )
        session.commit()
        return rows[0].id


def test_cache_never_tells_the_bundle_buyer_their_own_item_is_pending(world):
    """The chair is promised to the desk buyer in the desk thread. When that buyer then asks
    about the chair in a new thread, "pending with someone else" would be false: the agent
    answers instead."""
    desk_id, desk_listing = world.item()
    chair_id, chair_listing = world.item("Office chair")
    desk_remote = world.buyer(desk_listing, "Would you take $200 for the desk and the chair?")
    world.tick()
    _bundle_desk_and_chair(world, desk_remote, desk_id, chair_id)
    world.provider.requests.clear()
    chair_remote = world.market.open_other_thread(desk_remote, chair_listing, AVAILABLE)

    report = world.tick()

    assert report.cache_replies == 0
    assert not any("pending" in body for _, body, _ in world.market.sent)
    assert chair_id in _item_ids_in_order(world.provider.requests)
    assert world.thread_of(chair_remote).buyer_id == world.thread_of(desk_remote).buyer_id


def test_cache_skips_paused_items(world):
    item_id, listing = world.item()
    world.update(Item, item_id, paused=True)
    world.buyer(listing, AVAILABLE)

    report = world.tick()

    assert (report.ingested, report.cache_replies, report.wakes) == (1, 0, 0)


def test_cache_reply_still_passes_the_guard(engine, clock, settings, notifier):
    settings.daily_outbound_cap = {"fake": 0}
    world = _world(engine, clock, settings, notifier)
    _, listing = world.item()
    remote = world.buyer(listing, AVAILABLE)

    report = world.tick()

    assert report.cache_replies == 0
    assert world.market.sent == []
    assert report.wakes == 1  # the agent sees the waiting message (and the cap) instead
    assert world.messages(remote)[0].handled  # handled by the agent's wake


# ---------------------------------------------------------------- wakes fire once


def test_new_message_wakes_once(world):
    _, listing = world.item()
    world.buyer(listing, "What are the dimensions?")

    reports = world.run(hours=2, every=timedelta(minutes=2))

    assert world.wakes(reports) == 1
    assert len(world.provider.requests) == 1


def test_idle_fires_once_per_idle_wake_days(world):
    world.item()

    reports = world.run(hours=24 * 7, every=timedelta(hours=1))

    assert world.wakes(reports) == 2  # day 3 and day 6
    idle_wakes = [i for i, r in enumerate(reports) if r.wakes]
    assert idle_wakes == [72, 144]


def test_stats_moved_fires_once_per_view_delta(world):
    _, listing = world.item()
    fake = world.market.listings[listing]
    delta = world.settings.stats_wake_view_delta
    wakes = []
    for views in (delta + 5, delta + 5, delta + 15, 2 * delta + 5, 2 * delta + 5):
        fake.views = views
        world.clock.advance(minutes=10)
        wakes.append(world.tick().wakes)

    assert wakes == [1, 0, 0, 1, 0]
    with world.session() as session:
        item = session.exec(select(Item)).one()
    assert item.last_wake_views == 2 * delta + 5


def test_stats_baseline_drops_after_a_relist(world):
    _, listing = world.item()
    delta = world.settings.stats_wake_view_delta
    world.market.listings[listing].views = 4 * delta
    world.tick()
    new_ref = world.market.publish(ListingPayload("IKEA desk", "Solid.", 12_000, [], "Bloor"))
    with world.session() as session:
        row = session.exec(select(Listing)).one()
        row.remote_id = new_ref.remote_id
        session.commit()
    world.clock.advance(hours=1)
    assert world.tick().wakes == 0

    world.market.listings[new_ref.remote_id].views = delta
    world.clock.advance(hours=1)

    assert world.tick().wakes == 1


def test_pickup_elapsed_fires_once(world):
    item_id, listing = world.item()
    remote = world.buyer(listing, "Deal at $110")
    world.tick()
    thread = world.thread_of(remote)
    with world.session() as session:
        accepted = ledger.accept(
            session,
            world.clock,
            item_id=item_id,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            price_cents=11_000,
            undo_minutes=10,
        )
        start = world.clock.now() + timedelta(hours=2)
        ledger.book_pickup(
            session,
            world.clock,
            accepted_id=accepted.id,
            start=start,
            end=start + timedelta(hours=1),
        )
        session.commit()

    reports = world.run(hours=8, every=timedelta(minutes=10))

    assert world.wakes(reports) == 1
    assert "pickup_elapsed" in world.provider.requests[-1].prompt


def test_deadline_fires_once(world):
    item_id, _ = world.item(last_run=None)
    world.update(Item, item_id, last_agent_run_at=world.clock.now())
    world.clock.advance(days=13, hours=23)
    world.update(Item, item_id, last_agent_run_at=world.clock.now())

    reports = world.run(hours=12, every=timedelta(minutes=30))

    assert world.wakes(reports) == 1
    assert "deadline" in world.provider.requests[0].prompt


def test_wake_request_is_consumed(world):
    item_id, _ = world.item(last_run=None)
    with world.session() as session:
        session.add(
            WakeRequest(
                item_id=item_id, reason="The owner set the floor.", created_at=world.clock.now()
            )
        )
        session.commit()

    reports = world.run(hours=1, every=timedelta(minutes=5))

    assert world.wakes(reports) == 1
    assert "The owner set the floor." in world.provider.requests[0].prompt
    with world.session() as session:
        [request] = session.exec(select(WakeRequest)).all()
    assert request.consumed_at is not None


def test_expired_approval_wakes_the_agent_once(engine, clock, settings, notifier):
    world = _world(engine, clock, settings, notifier)
    item_id, _ = world.item(last_run=clock.now())
    with world.session() as session:
        session.add(
            PendingApproval(item_id=item_id, tool="publish", args_json="{}", created_at=clock.now())
        )
        session.commit()

    reports = world.run(hours=30, every=timedelta(hours=1))

    with world.session() as session:
        approval = session.exec(select(PendingApproval)).one()
    assert approval.status == ApprovalStatus.EXPIRED
    assert approval.decided_at == clock.now() - timedelta(hours=6)
    assert world.wakes(reports) == 1
    assert "approval_decided" in world.provider.requests[0].prompt


# ---------------------------------------------------------------- failures and quota


def test_failing_wakes_back_off_instead_of_retrying_every_tick(engine, clock, settings, notifier):
    world = _world(
        engine, clock, settings, notifier, script=lambda r: ProviderError("model is down")
    )
    _, listing = world.item()
    world.buyer(listing, "What are the dimensions?")

    reports = world.run(hours=1, every=timedelta(minutes=2))

    assert world.wakes(reports) == 4  # minutes 0, 6, 16, 36: waits of 5, 10, 20, then 40
    assert len(world.provider.requests) == 4
    assert all("model is down" in e for r in reports if r.wakes for e in r.errors)


def test_backoff_resets_after_a_successful_wake(engine, clock, settings, notifier):
    replies = iter([ProviderError("down"), QUIET, QUIET])
    world = _world(engine, clock, settings, notifier, script=lambda r: next(replies))
    _, listing = world.item()
    remote = world.buyer(listing, "What are the dimensions?")
    assert world.tick().wakes == 1
    clock.advance(FAILED_WAKE_BACKOFF)
    assert world.tick().wakes == 1

    world.market.buyer_say(remote, "And the height?")

    assert world.tick().wakes == 1


def test_exhausted_quota_escalates_once_per_item_and_never_wakes(engine, clock, settings, notifier):
    spent = {c: QuotaState.EXHAUSTED for c in Capability}
    world = _world(engine, clock, settings, notifier, states=spent)
    item_id, listing = world.item()
    world.item("Lamp")
    world.buyer(listing, "What are the dimensions?")
    world.clock.advance(days=4)  # both items idle too

    reports = world.run(hours=1, every=timedelta(minutes=2))

    assert world.wakes(reports) == 0 and world.provider.requests == []
    assert [r.escalated for r in reports[:2]] == [1, 0]
    assert {r.quota_state for r in reports} == {"exhausted"}
    with world.session() as session:
        [escalation] = session.exec(select(Escalation)).all()
        assert (escalation.item_id, escalation.reason) == (item_id, QUOTA_ESCALATION)
        escalation.resolved_at = clock.now()
        session.commit()
    assert len(notifier.of_kind(NoticeKind.ESCALATION)) == 1

    assert world.tick().escalated == 1


def _limited(world: World, limits: dict[str, ProviderLimit], providers=None) -> World:
    """Swap the world's fixed-state router for a real one over the given limits."""
    quota = QuotaTracker(world.engine, world.clock, limits)
    router = ProviderRouter([world.provider] if providers is None else providers, quota)
    world.router = world.runner.router = world.runner.agent.router = router
    return world


def _escalations(world: World) -> list[Escalation]:
    with world.session() as session:
        return list(session.exec(select(Escalation).order_by(Escalation.id)).all())


def test_a_full_per_minute_window_waits_a_tick_instead_of_escalating(world, notifier):
    _limited(world, {"scripted": ProviderLimit(rpm=2, rpd=250)})
    for title in ("Desk", "Lamp", "Chair"):
        _, listing = world.item(title)
        world.buyer(listing, "Would you take $100?")

    first = world.tick()

    assert (first.wakes, first.escalated, first.quota_state) == (2, 0, "healthy")
    assert _escalations(world) == [] and notifier.of_kind(NoticeKind.ESCALATION) == []
    world.clock.advance(minutes=2)
    assert (world.tick().wakes, _escalations(world)) == (1, [])


def test_a_short_429_cooldown_waits_a_tick_instead_of_escalating(world, notifier):
    limited = RateLimited("429")
    limited.retry_after, limited.daily = 30, False  # type: ignore[attr-defined]
    world.provider._responder = None
    world.provider._queue.extend([limited, QUIET, QUIET])
    _limited(world, {"scripted": ProviderLimit(rpm=10, rpd=250)})
    for title in ("Desk", "Lamp"):
        _, listing = world.item(title)
        world.buyer(listing, "Would you take $100?")

    report = world.tick()

    assert (report.escalated, report.quota_state) == (0, "healthy")
    assert _escalations(world) == []
    world.clock.advance(minutes=10)  # cooldown and the failed wake's backoff are over
    assert world.tick().wakes == 2
    assert _escalations(world) == []


def test_quota_escalation_resolves_once_the_agent_answers(engine, clock, settings, notifier):
    states = {c: QuotaState.EXHAUSTED for c in Capability}
    world = _world(engine, clock, settings, notifier, states=states)
    item_id, listing = world.item()
    remote = world.buyer(listing, "What are the dimensions?")
    assert world.tick().escalated == 1

    states.clear()  # quota is back
    assert world.tick().wakes == 1
    [escalation] = _escalations(world)
    assert escalation.resolved_at == clock.now() and escalation.resolution

    states.update({c: QuotaState.EXHAUSTED for c in Capability})  # spent again, a new message
    world.market.buyer_say(remote, "And the height?")
    assert world.tick().escalated == 1
    assert len(notifier.of_kind(NoticeKind.ESCALATION)) == 2


def test_no_model_configured_says_so_instead_of_quota_used_up(world, notifier):
    _limited(world, {}, providers=[])
    _, listing = world.item()
    world.buyer(listing, "Would you take $100?")

    assert world.tick().escalated == 1

    [escalation] = _escalations(world)
    assert escalation.reason == NO_MODEL_ESCALATION
    [notice] = notifier.of_kind(NoticeKind.ESCALATION)
    assert "GEMINI_API_KEY" in notice.body and "quota resets" not in notice.body


def test_constrained_quota_wakes_only_priority_triggers(engine, clock, settings, notifier):
    world = _world(
        engine, clock, settings, notifier, states={Capability.REASONING: QuotaState.CONSTRAINED}
    )
    world.item("Lamp")
    clock.advance(days=4)
    chatty, listing = world.item()
    world.buyer(listing, "What are the dimensions?")

    report = world.tick()

    assert (report.wakes, report.quota_state) == (1, "constrained")
    assert _item_ids_in_order(world.provider.requests) == [chatty]


def test_cheap_model_fallback_counts_as_constrained(engine, clock, settings, notifier):
    states = {Capability.REASONING: QuotaState.EXHAUSTED, Capability.CHEAP: QuotaState.HEALTHY}
    world = _world(engine, clock, settings, notifier, states=states)
    _, listing = world.item()
    world.buyer(listing, "What are the dimensions?")

    report = world.tick()

    assert (report.wakes, report.escalated, report.quota_state) == (1, 0, "constrained")
    assert world.provider.requests[0].capability == Capability.CHEAP


def test_items_with_a_deal_wake_first(world):
    plain, plain_listing = world.item("Lamp")
    world.buyer(plain_listing, "What are the dimensions?")
    committed, listing = world.item()
    remote = world.buyer(listing, "Deal at $110")
    world.tick()
    thread = world.thread_of(remote)
    with world.session() as session:
        ledger.accept(
            session,
            world.clock,
            item_id=committed,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            price_cents=11_000,
            undo_minutes=10,
        )
        session.commit()
    world.provider.requests.clear()
    world.market.buyer_say(remote, "When can I come?")
    world.buyer(plain_listing, "Hello?")

    world.tick()

    assert _item_ids_in_order(world.provider.requests) == [committed, plain]


def test_a_bundle_partner_sold_by_an_earlier_wake_is_not_woken(engine, clock, settings, notifier):
    """Both items carry triggers when the tick starts; the desk's wake completes the bundle, so
    the chair is SOLD before its turn. Waking it would spend a request on an item the agent can
    no longer act on and swallow the other buyer's message, which is owed a courtesy reply."""
    desk_thread: list[int] = []

    def complete_the_desk(request: LLMRequest):
        prompt = request.prompt
        if not prompt.startswith(f"ITEM_ID: {desk_id}\n") or "RESULTS" in prompt:
            return QUIET
        args = json.dumps({"thread_id": desk_thread[0]})
        return {
            "rationale": "They picked both up.",
            "actions": [{"tool": "complete_sale", "args_json": args}],
        }

    world = _world(engine, clock, settings, notifier, script=complete_the_desk)
    desk_id, desk_listing = world.item()
    chair_id, chair_listing = world.item("Office chair")
    desk_remote = world.buyer(desk_listing, "Would you take $200 for the desk and the chair?")
    world.tick()
    accepted_id = _bundle_desk_and_chair(world, desk_remote, desk_id, chair_id)
    desk_thread.append(world.thread_of(desk_remote).id)
    with world.session() as session:
        start = clock.now() + timedelta(hours=1)
        ledger.book_pickup(
            session, clock, accepted_id=accepted_id, start=start, end=start + timedelta(hours=1)
        )
        session.commit()
    clock.advance(hours=9)  # past the pickup's end and the owner's time to answer
    world.market.buyer_say(desk_remote, "Picked both up, thanks!")
    other = world.buyer(chair_listing, "Could you deliver the chair to Etobicoke?")
    world.provider.requests.clear()

    report = world.tick()

    assert _item_ids_in_order(world.provider.requests) == [desk_id]
    assert report.wakes == 1 and report.courtesy_replies == 1
    with world.session() as session:
        assert {session.get(Item, i).status for i in (desk_id, chair_id)} == {ItemStatus.SOLD}
    assert world.market.sent[-1][:2] == (other, "Sorry, it's sold now. Thanks for asking!")


def test_one_broken_wake_does_not_stop_the_others(world, monkeypatch):
    first, listing_a = world.item("Lamp")
    _, listing_b = world.item()
    world.buyer(listing_a, "Hi?")
    world.buyer(listing_b, "Hello?")
    wake = world.runner.agent.wake

    def flaky(session, item_id, triggers):
        if item_id == first:
            raise RuntimeError("bug")
        return wake(session, item_id, triggers)

    monkeypatch.setattr(world.runner.agent, "wake", flaky)

    report = world.tick()

    assert report.wakes == 1
    assert report.errors == [f"item {first}: wake failed: RuntimeError: bug"]


# ---------------------------------------------------------------- platforms


def test_platform_failure_is_isolated(world):
    _, listing = world.item()
    world.buyer(listing, "What are the dimensions?")
    world.market.fail_next["fetch_threads"] = PlatformError("timeout")

    first = world.tick()
    second = world.tick()

    assert first.errors == ["fake: timeout"] and first.ingested == 0
    assert (second.errors, second.ingested, second.wakes) == ([], 1, 1)


def test_auth_failure_alerts_once_per_day(world):
    for _ in range(3):
        world.market.fail_next["fetch_threads"] = PlatformAuthError("session expired")
        world.tick()
    world.clock.advance(days=1)
    world.market.fail_next["fetch_threads"] = PlatformAuthError("session expired")
    world.tick()

    alerts = [n for n in world.notifier.of_kind(NoticeKind.ALERT) if "needs you" in n.title]
    assert len(alerts) == 2


def test_the_auth_alert_gives_the_platforms_own_remedy(world):
    """Kijiji has no sign-in command: "sign in again" would point nowhere."""
    remedy = "kijiji: Kijiji asked for an extra sign-in step. It signs in again by itself."
    world.market.fail_next["fetch_threads"] = PlatformAuthError(remedy)

    report = world.tick()

    [alert] = world.notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "fake needs you" and alert.body.startswith(remedy)
    assert "sign in again" not in alert.body.lower()
    assert report.errors == [f"fake: needs the owner: {remedy}"]


def test_a_browser_profile_in_use_alerts_the_owner_once_a_day(world, tmp_path):
    """`agent web` next to `agent run` with Facebook: one of them cannot open the browser."""
    in_use = ProfileInUse(PROFILE_IN_USE.format(path=tmp_path))
    for _ in range(2):
        world.market.fail_next["fetch_threads"] = in_use
        world.tick()

    [alert] = world.notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "fake needs you" and "run `agent serve` on its own" in alert.body


def test_cache_send_failure_leaves_the_message_for_the_agent(world):
    _, listing = world.item()
    world.buyer(listing, AVAILABLE)
    world.market.fail_next["send_message"] = PlatformError("down")

    report = world.tick()

    assert (report.cache_replies, report.wakes) == (0, 1)
    assert report.errors == ["cache reply to message 1: down"]


def test_unexpected_ingest_error_does_not_stop_the_tick(world, monkeypatch):
    item_id, _ = world.item()
    with world.session() as session:
        session.add(WakeRequest(item_id=item_id, reason="Owner.", created_at=world.clock.now()))
        session.commit()

    def broken(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("marketplace_agent.runner.tick.ingest", broken)

    report = world.tick()

    assert report.errors == ["fake: ingest failed: RuntimeError: bug"]
    assert report.wakes == 1


def test_a_buyer_on_the_post_a_relist_replaced_still_reaches_the_agent(world):
    item_id, listing = world.item()
    remote = world.buyer(listing, "Would you take $110?")  # written, not fetched yet
    world.clock.advance(hours=25)
    with world.session() as session:
        ctx = ToolContext(
            session, world.clock, world.settings, {"fake": world.market}, world.notifier, item_id
        )
        assert RelistTool().execute(ctx, PlatformArgs(platform="fake")).ok
        session.commit()

    report = world.tick()

    assert world.thread_of(remote).item_id == item_id
    assert report.wakes == 1 and len(world.provider.requests) == 1


def _deal_with_pickup(world: World, remote: str, *, hours: float = 2) -> tuple[int, int]:
    """accept_offer and a booked one-hour pickup `hours` from now, as the agent would."""
    thread = world.thread_of(remote)
    with world.session() as session:
        accepted = ledger.accept(
            session,
            world.clock,
            item_id=thread.item_id,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            price_cents=11_000,
            undo_minutes=10,
        )
        start = world.clock.now() + timedelta(hours=hours)
        ledger.book_pickup(
            session, world.clock, accepted_id=accepted.id, start=start,
            end=start + timedelta(hours=1),
        )
        session.commit()
        return thread.id, accepted.id


def test_the_owner_is_asked_once_whether_the_buyer_came(world, notifier):
    item_id, listing = world.item()
    remote = world.buyer(listing, "Deal at $110, can I come by?")
    world.tick()
    _deal_with_pickup(world, remote)

    world.run(hours=2.5, every=timedelta(minutes=30))
    assert notifier.of_kind(NoticeKind.ESCALATION) == []  # the pickup is still on
    world.run(hours=3, every=timedelta(minutes=30))

    [asked] = notifier.of_kind(NoticeKind.ESCALATION)
    buyer = world.thread_of(remote).buyer_id
    with world.session() as session:
        name = session.get(Buyer, buyer).display_name
    assert asked.title == f"Did {name} pick up IKEA desk?"
    assert f"agent sold {item_id}" in asked.body and f"agent noshow {item_id}" in asked.body
    assert "no-show from Thu Oct 1 21:00" in asked.body  # pickup end + 6h
    [check] = _escalations(world)
    assert check.reason.startswith("Pickup check") and check.resolved_at is None


def test_the_agent_cannot_call_a_silent_buyer_a_no_show_before_the_owner_could_answer(
    engine, clock, settings, notifier
):
    """Example 3 used to teach releasing on silence; a buyer who came and paid in person often
    never writes again, so the item may already be gone."""
    attempts: list[str] = []

    def release_when_late(request: LLMRequest):
        prompt = request.prompt
        if "pickup_elapsed" not in prompt.split("## ITEM")[0] or "RESULTS" in prompt:
            return QUIET
        thread_id = int(prompt.split("[thread ")[1].split(",")[0])
        attempts.append(prompt)
        return {
            "rationale": "No word from the buyer, so it is a no-show.",
            "actions": [
                {
                    "tool": "release_commitment",
                    "args_json": json.dumps(
                        {"thread_id": thread_id, "reason": "no-show, no word", "no_show": True}
                    ),
                }
            ],
        }

    world = _world(engine, clock, settings, notifier, script=release_when_late)
    _, listing = world.item()
    remote = world.buyer(listing, "Deal at $110, can I come by?")
    world.tick()
    thread_id, accepted_id = _deal_with_pickup(world, remote)

    world.run(hours=4, every=timedelta(minutes=30))  # pickup ends at +3h, grace at +3.5h

    assert len(attempts) == 1
    with world.session() as session:
        assert ledger.active_commitment_for_thread(session, thread_id) is not None
        run = session.exec(select(AgentRun).order_by(AgentRun.id.desc())).first()
        [outcome] = [o for o in json.loads(run.results_json) if o["step"] == 1]
        assert outcome["status"] == "rejected"
        assert "silence does not mean" in outcome["detail"]

    world.run(hours=6, every=timedelta(minutes=30))  # the owner never answered

    assert len(attempts) == 2  # pickup_elapsed fired again once the owner's time was up
    with world.session() as session:
        assert ledger.active_commitment_for_thread(session, thread_id) is None
        assert session.get(Buyer, world.thread_of(remote).buyer_id).no_show_count == 1
    [check] = [e for e in _escalations(world) if e.reason.startswith("Pickup check")]
    assert check.resolved_at is not None  # the deal closed, so the question is moot


# ---------------------------------------------------------------- housekeeping


def test_ghosted_thread_closes_once(world):
    _, listing = world.item(last_run=None)
    remote = world.buyer(listing, "Would you take $100?")
    world.tick()
    thread = world.thread_of(remote)
    world.update(Thread, thread.id, last_outbound_at=world.clock.now() + timedelta(minutes=1))

    world.clock.advance(hours=71)
    world.tick()
    assert world.thread_of(remote).state == ThreadState.NEGOTIATING
    world.clock.advance(hours=2)
    world.tick()
    world.tick()

    assert world.thread_of(remote).state == ThreadState.CLOSED
    with world.session() as session:
        assert session.get(Buyer, thread.buyer_id).ghost_count == 1


def test_outbound_cap_alerts_once_per_day(engine, clock, settings, notifier):
    settings.daily_outbound_cap = {"fake": 0}
    world = _world(engine, clock, settings, notifier)

    world.run(hours=3, every=timedelta(hours=1))
    clock.advance(days=1)
    world.tick()

    caps = [n for n in notifier.of_kind(NoticeKind.ALERT) if "cap" in n.title]
    assert len(caps) == 2


# ---------------------------------------------------------------- end to end


def test_agent_reply_reaches_the_buyer_and_its_echo_is_not_duplicated(
    engine, clock, settings, notifier
):
    def reply(request: LLMRequest):
        thread_id = int(request.prompt.split("[thread ")[1].split(",")[0])
        body = "Thanks! I can do $110 if you can pick it up this week."
        args = json.dumps({"thread_id": thread_id, "body": body})
        return {"rationale": "Counter.", "actions": [{"tool": "send_message", "args_json": args}]}

    world = _world(engine, clock, settings, notifier, script=reply)
    _, listing = world.item()
    remote = world.buyer(listing, "Would you take $100?")

    world.tick()
    clock.advance(minutes=2)
    world.tick()

    [(to, body, _)] = world.market.sent
    assert to == remote and "$110" in body
    inbound, outbound = world.messages(remote)
    assert inbound.handled and outbound.via == "agent"
    with world.session() as session:
        [run] = session.exec(select(AgentRun)).all()
    assert run.error is None


# ---------------------------------------------------------------- run_forever


def test_run_forever_sleeps_with_jitter_and_survives_a_failing_tick(world, monkeypatch):
    calls = {"ticks": 0}
    stop = threading.Event()
    sleeps: list[float] = []

    def tick():
        calls["ticks"] += 1
        if calls["ticks"] == 1:
            raise RuntimeError("boom")
        return TickReport(started_at=world.clock.now())

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            stop.set()

    monkeypatch.setattr(world.runner, "tick", tick)

    world.runner.run_forever(stop=stop, sleep=sleep)

    assert calls["ticks"] == 3
    base = world.settings.tick_seconds
    assert all(0.8 * base <= s <= 1.2 * base for s in sleeps)


def test_run_forever_returns_when_stopped(world, monkeypatch):
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(world.runner, "tick", lambda: pytest.fail("ticked after stop"))

    world.runner.run_forever(stop=stop, sleep=lambda s: pytest.fail("slept after stop"))


# ---------------------------------------------------------------- comps refresh hook


COMPS_MODULE = "marketplace_agent.core.comps"


def _fake_comps_module(refresh_due):
    import types

    module = types.ModuleType(COMPS_MODULE)
    module.refresh_due = refresh_due
    return module


def test_housekeeping_refreshes_stale_comps(world, monkeypatch):
    import sys

    calls = []

    def refresh_due(session, clock, platforms, **kwargs):
        calls.append((clock, dict(platforms), kwargs))
        return 0

    monkeypatch.setitem(sys.modules, COMPS_MODULE, _fake_comps_module(refresh_due))

    report = world.tick()

    assert calls == [(world.clock, {"fake": world.market}, {})]
    assert report.errors == []


def test_housekeeping_runs_without_the_comps_module(world, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, COMPS_MODULE, None)  # import fails

    assert world.tick().errors == []


def test_a_failing_comps_refresh_is_reported_not_raised(world, monkeypatch):
    import sys

    def refresh_due(session, clock, platforms, **kwargs):
        raise PlatformError("search is down")

    monkeypatch.setitem(sys.modules, COMPS_MODULE, _fake_comps_module(refresh_due))

    report = world.tick()

    assert any("comps refresh" in error and "search is down" in error for error in report.errors)
