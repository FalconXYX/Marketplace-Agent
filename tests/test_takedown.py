"""Convergent takedown: every listing of a SOLD or WITHDRAWN item comes down eventually."""

from datetime import timedelta

import pytest
from sqlmodel import Session, select

from marketplace_agent.agent.loop import Agent
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.control.service import ControlService
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import Item, ItemStatus, Listing, ListingStatus
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import (
    ListingGone,
    ListingPayload,
    PlatformAuthError,
    PlatformError,
)
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.runner.takedown import (
    ALERT_AFTER,
    MAX_BACKOFF,
    TakedownState,
    backoff,
    encode_state,
    owed_takedowns,
    retry_takedowns,
    takedown_state,
)
from marketplace_agent.runner.tick import Runner
from marketplace_agent.tools.registry import default_registry

TICK = timedelta(minutes=2)
URL = "https://kijiji.invalid/v/desk-1"


class DelistStub:
    """delist plays outcomes in order (None = success); every call is recorded."""

    def __init__(self, name: str = "kijiji", outcomes: list[Exception | None] | None = None):
        self.name = name
        self.outcomes = list(outcomes or [])
        self.delisted: list[str] = []

    def delist(self, ref) -> None:
        self.delisted.append(ref.remote_id)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome


def _item(session, clock, status=ItemStatus.SOLD, title="IKEA desk") -> Item:
    now = clock.now()
    item = Item(title=title, status=status, created_at=now, updated_at=now)
    session.add(item)
    session.flush()
    set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
    session.commit()
    return item


def _listing(
    session,
    clock,
    item: Item,
    *,
    platform="kijiji",
    status=ListingStatus.LIVE,
    remote_id: str | None = "k-1",
    url: str | None = URL,
    last_error: str | None = None,
) -> Listing:
    listing = Listing(
        item_id=item.id,
        platform=platform,
        remote_id=remote_id,
        url=url,
        status=status,
        price_cents=12_000,
        title=item.title,
        description="Solid.",
        published_at=clock.now() if remote_id else None,
        last_error=last_error,
    )
    session.add(listing)
    session.commit()
    return listing


def _retry(session, clock, settings, platforms, notifier):
    return retry_takedowns(session, clock, settings, platforms, notifier, tick=TICK)


def _reload(session, listing: Listing) -> Listing:
    return session.get(Listing, listing.id, populate_existing=True)


# ---------------------------------------------------------------- the retry loop


def test_failing_twice_then_succeeding_converges_with_backoff(session, clock, settings, notifier):
    listing = _listing(session, clock, _item(session, clock))
    stub = DelistStub(outcomes=[PlatformError("503"), PlatformError("timeout"), None])
    platforms = {"kijiji": stub}

    first = _retry(session, clock, settings, platforms, notifier)
    state = takedown_state(_reload(session, listing))
    assert (first.taken_down, first.failed) == (0, 1)
    assert (state.failures, state.next_at, state.error) == (1, clock.now() + TICK, "503")

    _retry(session, clock, settings, platforms, notifier)  # not due yet
    assert len(stub.delisted) == 1

    clock.advance(TICK)
    _retry(session, clock, settings, platforms, notifier)
    state = takedown_state(_reload(session, listing))
    assert (state.failures, state.next_at) == (2, clock.now() + 2 * TICK)

    clock.advance(TICK)
    _retry(session, clock, settings, platforms, notifier)  # one tick in: still waiting
    assert len(stub.delisted) == 2

    clock.advance(TICK)
    last = _retry(session, clock, settings, platforms, notifier)
    listing = _reload(session, listing)
    assert (last.taken_down, last.failed, last.errors) == (1, 0, [])
    assert (listing.status, listing.delisted_at, listing.last_error) == (
        ListingStatus.DELISTED,
        clock.now(),
        None,
    )
    assert stub.delisted == ["k-1"] * 3
    assert notifier.notices == []  # two failures never reach the alert
    assert owed_takedowns(session) == []


def test_listing_gone_counts_as_taken_down(session, clock, settings, notifier):
    listing = _listing(session, clock, _item(session, clock))
    stub = DelistStub(outcomes=[ListingGone("not found")])

    report = _retry(session, clock, settings, {"kijiji": stub}, notifier)

    listing = _reload(session, listing)
    assert (report.taken_down, report.failed, report.errors) == (1, 0, [])
    assert listing.status == ListingStatus.DELISTED and listing.delisted_at == clock.now()
    assert takedown_state(listing).failures == 0
    assert notifier.notices == []
    clock.advance(hours=7)
    _retry(session, clock, settings, {"kijiji": stub}, notifier)
    assert stub.delisted == ["k-1"]


def test_alerts_once_after_three_failures_and_keeps_retrying(session, clock, settings, notifier):
    item = _item(session, clock)
    listing = _listing(session, clock, item)
    stub = DelistStub(outcomes=[PlatformError("503")] * 6)
    platforms = {"kijiji": stub}

    for _ in range(6):
        _retry(session, clock, settings, platforms, notifier)
        clock.advance(MAX_BACKOFF)

    assert len(stub.delisted) == 6
    assert takedown_state(_reload(session, listing)).failures == 6
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.item_id == item.id
    assert "kijiji" in alert.title and URL in alert.body and "by hand" in alert.body
    assert "503" in alert.body and "sold" in alert.body

    _retry(session, clock, settings, platforms, notifier)  # finally works
    assert _reload(session, listing).status == ListingStatus.DELISTED
    assert len(notifier.of_kind(NoticeKind.ALERT)) == 1
    [info] = notifier.of_kind(NoticeKind.INFO)
    assert "kijiji" in info.title and "No need to remove it by hand" in info.body


def test_alert_comes_at_the_third_failure_not_before(session, clock, settings, notifier):
    _listing(session, clock, _item(session, clock))
    stub = DelistStub(outcomes=[PlatformAuthError("signed out")] * 3)
    counts = []
    for _ in range(ALERT_AFTER):
        _retry(session, clock, settings, {"kijiji": stub}, notifier)
        counts.append(len(notifier.of_kind(NoticeKind.ALERT)))
        clock.advance(MAX_BACKOFF)
    assert counts == [0, 0, 1]


def test_a_platform_that_is_not_connected_is_a_failure(session, clock, settings, notifier):
    listing = _listing(session, clock, _item(session, clock), platform="facebook", url=None)

    for _ in range(ALERT_AFTER):
        report = _retry(session, clock, settings, {}, notifier)
        clock.advance(MAX_BACKOFF)

    assert report.failed == 1 and "not connected" in report.errors[0]
    assert _reload(session, listing).status == ListingStatus.LIVE
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert "facebook post k-1" in alert.body  # no url: the remote id names it


def test_an_adapter_bug_is_a_failure_not_a_crash(session, clock, settings, notifier):
    item = _item(session, clock)
    broken = _listing(session, clock, item, platform="kijiji")
    fine = _listing(session, clock, item, platform="fake", remote_id="f-1")
    platforms = {"kijiji": DelistStub(outcomes=[KeyError("x")]), "fake": DelistStub("fake")}

    report = _retry(session, clock, settings, platforms, notifier)

    assert (report.taken_down, report.failed) == (1, 1)
    assert "KeyError" in takedown_state(_reload(session, broken)).error
    assert _reload(session, fine).status == ListingStatus.DELISTED


def test_only_live_pending_or_awaiting_listings_of_closed_items_are_owed(
    session, clock, settings, notifier
):
    listed = _item(session, clock, ItemStatus.LISTED, "Lamp")
    committed = _item(session, clock, ItemStatus.COMMITTED, "Chair")
    sold = _item(session, clock, ItemStatus.SOLD, "Desk")
    withdrawn = _item(session, clock, ItemStatus.WITHDRAWN, "Rug")
    _listing(session, clock, listed, remote_id="a")
    _listing(session, clock, committed, remote_id="b")
    _listing(session, clock, sold, remote_id="c", status=ListingStatus.DELISTED)
    _listing(session, clock, sold, remote_id="d", status=ListingStatus.FAILED)
    owed = [
        _listing(session, clock, sold, remote_id="e"),
        _listing(session, clock, withdrawn, remote_id="f"),
        _listing(session, clock, sold, remote_id="g", status=ListingStatus.PENDING),
    ]
    stub = DelistStub()

    _retry(session, clock, settings, {"kijiji": stub}, notifier)

    assert stub.delisted == ["e", "f", "g"]
    assert all(_reload(session, x).status == ListingStatus.DELISTED for x in owed)


def test_an_unposted_assist_form_is_dropped_and_the_owner_told_not_to_post(
    session, clock, settings, notifier
):
    item = _item(session, clock)
    waiting = _listing(
        session, clock, item, platform="facebook", status=ListingStatus.AWAITING_HUMAN,
        remote_id=None, url=None,
    )
    pending = _listing(session, clock, item, status=ListingStatus.PENDING, remote_id=None)
    stub = DelistStub("facebook")

    report = _retry(session, clock, settings, {"facebook": stub, "kijiji": stub}, notifier)
    _retry(session, clock, settings, {"facebook": stub, "kijiji": stub}, notifier)

    assert report.taken_down == 2 and stub.delisted == []
    assert _reload(session, waiting).status == ListingStatus.DELISTED
    assert _reload(session, pending).status == ListingStatus.DELISTED
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert "facebook" in alert.title and "without clicking Post" in alert.body


def test_a_listing_taken_down_elsewhere_meanwhile_is_skipped(
    engine, session, clock, settings, notifier
):
    listing = _listing(session, clock, _item(session, clock))
    stub = DelistStub()
    with Session(engine) as other:  # the owner's mark_sold got there first
        row = other.get(Listing, listing.id)
        row.status = ListingStatus.DELISTED
        other.add(row)
        other.commit()

    retry_takedowns(session, clock, settings, {"kijiji": stub}, notifier, tick=TICK)

    assert stub.delisted == []


# ---------------------------------------------------------------- stored state


def test_state_round_trips_through_last_error(clock):
    state = TakedownState(4, clock.now() + timedelta(minutes=16), "kijiji: 503: try: later")
    listing = Listing(item_id=1, platform="kijiji", price_cents=1, title="t", description="")
    listing.last_error = encode_state(state)
    assert takedown_state(listing) == state
    assert listing.last_error.startswith("takedown 4 next 2026-")


@pytest.mark.parametrize("error", ["kijiji: 503", "alerted: 503", " alerted: x"])
def test_the_alerted_flag_round_trips_through_last_error(clock, error):
    listing = Listing(item_id=1, platform="kijiji", price_cents=1, title="t", description="")
    for alerted in (True, False):
        state = TakedownState(0, clock.now(), error, alerted=alerted)
        listing.last_error = encode_state(state)
        assert takedown_state(listing) == state


@pytest.mark.parametrize(
    "text", [None, "", "kijiji: 503", "takedown x next y: z", "takedown 2 next tomorrow: z"]
)
def test_other_last_errors_mean_no_failures_yet(text):
    listing = Listing(
        item_id=1, platform="kijiji", price_cents=1, title="t", description="", last_error=text
    )
    assert takedown_state(listing) == TakedownState()


def test_the_first_retry_after_the_sale_is_due_at_once(session, clock, settings, notifier):
    """complete_sale stored a plain error: no retry state yet, so the next pass tries."""
    listing = _listing(session, clock, _item(session, clock), last_error="kijiji: 503")
    stub = DelistStub()

    _retry(session, clock, settings, {"kijiji": stub}, notifier)

    assert stub.delisted == ["k-1"] and _reload(session, listing).last_error is None


def test_retry_state_survives_a_restart(engine, clock, settings, notifier):
    failing = {"kijiji": DelistStub(outcomes=[PlatformError("x")] * 3)}
    with Session(engine, expire_on_commit=False) as session:
        listing = _listing(session, clock, _item(session, clock))
        for _ in range(2):
            _retry(session, clock, settings, failing, notifier)
            clock.advance(MAX_BACKOFF)
    with Session(engine, expire_on_commit=False) as fresh:  # a new process
        _retry(fresh, clock, settings, failing, notifier)
        assert takedown_state(fresh.get(Listing, listing.id)).failures == 3
    assert len(notifier.of_kind(NoticeKind.ALERT)) == 1


def test_backoff_doubles_from_one_tick_and_caps_at_six_hours():
    assert [backoff(n, TICK) for n in (1, 2, 3, 4)] == [TICK, 2 * TICK, 4 * TICK, 8 * TICK]
    assert backoff(0, TICK) == timedelta(0)
    assert backoff(9, TICK) == MAX_BACKOFF == timedelta(hours=6)
    assert backoff(10_000, TICK) == MAX_BACKOFF  # no overflow after months of failures


def test_the_default_tick_never_skips_a_runner_tick(session, clock, settings, notifier):
    """run_forever ticks 20% early at most; one failure must not make the next tick skip."""
    listing = _listing(session, clock, _item(session, clock))
    stub = DelistStub(outcomes=[PlatformError("x"), None])

    retry_takedowns(session, clock, settings, {"kijiji": stub}, notifier)
    clock.advance(seconds=settings.tick_seconds * 0.8)
    retry_takedowns(session, clock, settings, {"kijiji": stub}, notifier)

    assert _reload(session, listing).status == ListingStatus.DELISTED


def test_the_pickup_address_never_reaches_the_stored_error(session, clock, settings, notifier):
    listing = _listing(session, clock, _item(session, clock))
    stub = DelistStub(outcomes=[PlatformError(f"rejected post at {settings.pickup_address}")])

    report = _retry(session, clock, settings, {"kijiji": stub}, notifier)

    assert settings.pickup_address not in _reload(session, listing).last_error
    assert settings.pickup_address not in report.errors[0]


# ---------------------------------------------------------------- through the runner


def _runner(engine, clock, settings, notifier, market) -> tuple[Runner, ScriptedProvider]:
    provider = ScriptedProvider(lambda request: {"rationale": "Quiet.", "actions": []})
    router = ProviderRouter([provider], QuotaTracker(engine, clock, {}))
    registry = default_registry()
    platforms = {"fake": market}
    agent = Agent(router, registry, settings, clock, notifier, platforms)
    runner = Runner(engine, clock, settings, platforms, agent, router, notifier, registry)
    return runner, provider


def test_a_sale_whose_takedown_failed_converges_on_later_ticks(engine, clock, settings, notifier):
    market = FakeMarketplace(clock)
    runner, provider = _runner(engine, clock, settings, notifier, market)
    registry = default_registry()
    control = ControlService(engine, clock, settings, {"fake": market}, notifier, registry)
    ref = market.publish(ListingPayload("IKEA desk", "Solid.", 12_000, [], "Bloor"))
    with Session(engine, expire_on_commit=False) as session:
        item = _item(session, clock, ItemStatus.LISTED)
        listing = _listing(session, clock, item, platform="fake", remote_id=ref.remote_id)
    market.fail_next["delist"] = PlatformError("fake: 503")

    control.mark_sold(item.id)
    with Session(engine) as session:
        assert session.get(Listing, listing.id).status == ListingStatus.LIVE
    assert market.listings[ref.remote_id].live

    market.fail_next["delist"] = PlatformError("fake: still 503")
    first = runner.tick()
    clock.advance(seconds=settings.tick_seconds)
    second = runner.tick()

    assert (first.takedowns, second.takedowns) == (0, 1)
    assert any("still 503" in error for error in first.errors)
    assert not market.listings[ref.remote_id].live
    with Session(engine) as session:
        stored = session.get(Listing, listing.id)
        assert stored.status == ListingStatus.DELISTED and stored.last_error is None
        assert session.exec(select(Item).where(Item.status == ItemStatus.SOLD)).one()
    assert provider.requests == []


def test_an_owner_told_at_the_sale_to_remove_it_hears_once_it_is_down(
    engine, clock, settings, notifier
):
    """mark_sold's takedown failed, so the owner was sent to remove the post by hand. The next
    tick takes it down: without a word, they would go looking for a post that is gone."""
    market = FakeMarketplace(clock)
    runner, _ = _runner(engine, clock, settings, notifier, market)
    registry = default_registry()
    control = ControlService(engine, clock, settings, {"fake": market}, notifier, registry)
    ref = market.publish(ListingPayload("IKEA desk", "Solid.", 12_000, [], "Bloor"))
    with Session(engine, expire_on_commit=False) as session:
        item = _item(session, clock, ItemStatus.LISTED)
        _listing(session, clock, item, platform="fake", remote_id=ref.remote_id)
    market.fail_next["delist"] = PlatformError("fake: 503")

    control.mark_sold(item.id)
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert "Remove it by hand" in alert.body

    assert runner.tick().takedowns == 1

    assert not market.listings[ref.remote_id].live
    [info] = [n for n in notifier.of_kind(NoticeKind.INFO) if "listing is down" in n.title]
    assert info.item_id == item.id and "No need to remove it by hand" in info.body
    assert notifier.of_kind(NoticeKind.ALERT) == [alert]


def test_a_quiet_retry_success_says_nothing(session, clock, settings, notifier):
    """Nobody was told to remove it by hand, so there is nothing to call off."""
    _listing(session, clock, _item(session, clock))
    stub = DelistStub(outcomes=[PlatformError("x"), None])

    _retry(session, clock, settings, {"kijiji": stub}, notifier)
    clock.advance(MAX_BACKOFF)
    assert _retry(session, clock, settings, {"kijiji": stub}, notifier).taken_down == 1

    assert notifier.notices == []


def test_a_broken_takedown_pass_does_not_stop_housekeeping(
    engine, clock, settings, notifier, monkeypatch
):
    from marketplace_agent.runner import takedown

    market = FakeMarketplace(clock)
    runner, _ = _runner(engine, clock, settings, notifier, market)
    called = []
    monkeypatch.setattr(takedown, "retry_takedowns", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(runner, "_alert_caps", lambda session: called.append(True))

    report = runner.tick()

    assert called == [True]
    assert any("ZeroDivisionError" in error for error in report.errors)
