import inspect
import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from marketplace_agent.core.clock import FakeClock
from marketplace_agent.platforms.base import (
    AwaitingHuman,
    ListingPayload,
    PlatformAdapter,
    PlatformAuthError,
    PlatformError,
    PlatformRef,
    SearchResult,
)
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.platforms.fake import (
    AvailabilityAsker,
    FakeMarketplace,
    NoShow,
    ReasonableBuyer,
    Scammer,
    build_adapter,
    persona_class,
)
from marketplace_agent.platforms.fake.buyers import (
    INJECTIONS,
    Bundler,
    OtherListing,
    Turn,
    money_cents,
    read_reply,
)

START = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)
ADAPTER_METHODS = sorted(
    name
    for name, value in vars(PlatformAdapter).items()
    if callable(value) and not name.startswith("_")
)


class StubOracle:
    def __init__(self) -> None:
        self.prices: dict[str, int] = {}
        self.pickups: dict[str, tuple[datetime, datetime]] = {}

    def committed_price(self, thread_remote_id: str) -> int | None:
        return self.prices.get(thread_remote_id)

    def pickup(self, thread_remote_id: str) -> tuple[datetime, datetime] | None:
        return self.pickups.get(thread_remote_id)


def payload(price_cents: int = 10_000, title: str = "IKEA Malm desk") -> ListingPayload:
    return ListingPayload(
        title=title,
        description="White desk, 140 x 65 cm.",
        price_cents=price_cents,
        photo_paths=[Path("/photos/desk.jpg")],
        location="Bloor St W & Ossington Ave",
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(START)


@pytest.fixture
def market(clock: FakeClock) -> FakeMarketplace:
    return FakeMarketplace(clock, seed=7)


def buyer_lines(market: FakeMarketplace, thread_id: str) -> list[str]:
    return [m.body for m in market.threads[thread_id].messages if m.from_buyer]


def exchange(
    market: FakeMarketplace,
    clock: FakeClock,
    thread_id: str,
    text: str,
    oracle: StubOracle | None = None,
) -> list[str]:
    """Seller says text, then six hours pass: the buyer lines that arrived meanwhile."""
    before = len(buyer_lines(market, thread_id))
    market.send_message(thread_id, text)
    clock.advance(hours=6)
    market.step(oracle)
    return buyer_lines(market, thread_id)[before:]


def dollars(text: str) -> list[int]:
    return [int(d) for d in re.findall(r"\$(\d+)", text)]


# Adapter contract


@pytest.mark.parametrize("method", ADAPTER_METHODS)
def test_signature_matches_protocol(method):
    def shape(fn):
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    assert shape(getattr(FakeMarketplace, method)) == shape(getattr(PlatformAdapter, method))


def test_every_adapter_method_round_trips(market, clock):
    adapter: PlatformAdapter = market
    ref = adapter.publish(payload())
    assert (ref.platform, ref.remote_id) == ("fake", "fake-l1") and ref.url
    adapter.update(ref, payload(price_cents=9_000, title="IKEA Malm desk, white"))
    assert market.listings[ref.remote_id].payload.price_cents == 9_000
    thread = market.add_buyer("scripted", ref.remote_id)
    market.buyer_say(thread, "Is this still available?")
    clock.advance(minutes=5)
    sent_id = adapter.send_message(thread, "Yes it is!")
    assert isinstance(sent_id, str) and market.sent == [(thread, "Yes it is!", clock.now())]
    [remote] = adapter.fetch_threads(None)
    assert remote.listing_remote_id == ref.remote_id and remote.buyer_handle and remote.buyer_name
    assert [(m.body, m.from_buyer) for m in remote.messages] == [
        ("Is this still available?", True),
        ("Yes it is!", False),
    ]
    assert remote.messages[1].remote_id == sent_id
    assert adapter.stats(ref).message_count == 1
    assert [r.title for r in adapter.search("malm desk")] == ["IKEA Malm desk, white"]
    assert adapter.healthcheck() == []
    adapter.delist(ref)
    assert adapter.search("malm desk") == []
    assert adapter.stats(ref).message_count == 1


def test_build_adapter_returns_named_fake(settings, clock):
    adapter = build_adapter(settings, clock)
    assert isinstance(adapter, FakeMarketplace) and adapter.name == "fake"


def test_each_built_fake_names_its_posts_apart(settings, clock):
    """Every CLI command builds a new fake; a second process must not reuse item #1's post id."""
    first = build_adapter(settings, clock).publish(payload())
    second = build_adapter(settings, clock).publish(payload())

    assert first.remote_id != second.remote_id
    assert FakeMarketplace(clock).publish(payload()).remote_id == "fake-l1"  # sim and tests


def test_assist_mode_awaits_human(market):
    with pytest.raises(AwaitingHuman):
        market.publish(payload(), assist=True)
    assert market.listings == {}


@pytest.mark.parametrize("bad", [payload(title="  "), payload(price_cents=-1)])
def test_publish_rejects_invalid_listing(market, bad):
    with pytest.raises(PlatformError):
        market.publish(bad)


def test_unknown_or_delisted_refs_raise(market):
    unknown = PlatformRef("fake", "fake-l404")
    for call in (market.delist, market.stats, lambda ref: market.update(ref, payload())):
        with pytest.raises(PlatformError):
            call(unknown)
    ref = market.publish(payload())
    market.delist(ref)
    with pytest.raises(PlatformError):
        market.delist(ref)
    with pytest.raises(PlatformError):
        market.update(ref, payload())
    with pytest.raises(PlatformError):
        market.send_message("fake-t404", "hello")


def test_missing_or_delisted_posts_raise_listing_gone(market):
    """The fake can tell a missing post from a failure, so callers can treat it as down."""
    unknown = PlatformRef("fake", "fake-l404")
    for call in (market.delist, market.stats, lambda ref: market.update(ref, payload())):
        with pytest.raises(ListingGone):
            call(unknown)
    ref = market.publish(payload())
    market.delist(ref)
    with pytest.raises(ListingGone):
        market.update(ref, payload())


def test_a_post_the_platform_removes_is_gone_for_every_call(market):
    ref = market.publish(payload(title="IKEA Malm desk, white"))
    thread = market.add_buyer("scripted", ref.remote_id)
    market.buyer_say(thread, "Is this still available?")

    market.remove_listing(ref.remote_id)

    for call in (market.delist, market.stats, lambda ref: market.update(ref, payload())):
        with pytest.raises(ListingGone):
            call(ref)
    assert market.search("malm desk") == []
    assert market.fetch_threads(None)  # conversations about it are still there
    market.step()


def test_search_includes_catalog_and_respects_limit(clock):
    catalog = [SearchResult("Malm desk black", 8_000), SearchResult("Office chair", 4_000)]
    market = FakeMarketplace(clock, search_catalog=catalog)
    market.publish(payload())
    assert [r.price_cents for r in market.search("DESK")] == [8_000, 10_000]
    assert len(market.search("desk", limit=1)) == 1
    assert market.search("sofa") == []


# Failure injection


@pytest.mark.parametrize(
    "method, call",
    [
        ("publish", lambda m, ref, t: m.publish(payload())),
        ("update", lambda m, ref, t: m.update(ref, payload())),
        ("delist", lambda m, ref, t: m.delist(ref)),
        ("fetch_threads", lambda m, ref, t: m.fetch_threads(None)),
        ("send_message", lambda m, ref, t: m.send_message(t, "hi")),
        ("stats", lambda m, ref, t: m.stats(ref)),
        ("search", lambda m, ref, t: m.search("desk")),
    ],
)
def test_fail_next_raises_once(market, method, call):
    ref = market.publish(payload())
    thread = market.add_buyer("scripted", ref.remote_id)
    failure = PlatformAuthError("session expired")
    market.fail_next[method] = failure
    with pytest.raises(PlatformAuthError) as raised:
        call(market, ref, thread)
    assert raised.value is failure and market.fail_next == {}
    call(market, ref, thread)


def test_fail_next_healthcheck_reports_once(market):
    market.fail_next["healthcheck"] = PlatformError("login page changed")
    assert market.healthcheck() == ["fake: login page changed"]
    assert market.healthcheck() == []


# Threads and time


def test_fetch_threads_since_is_inclusive_and_filters(market, clock):
    ref = market.publish(payload())
    first = market.add_buyer("scripted", ref.remote_id)
    market.buyer_say(first, "hello")
    clock.advance(hours=1)
    market.send_message(first, "hi there")
    second = market.add_buyer("scripted", ref.remote_id)
    market.buyer_say(second, "is this available?")
    clock.advance(hours=1)
    market.buyer_say(first, "still there?")

    everything = market.fetch_threads(None)
    assert [t.remote_id for t in everything] == [first, second]
    assert [m.from_buyer for m in everything[0].messages] == [True, False, True]

    at_one_hour = market.fetch_threads(START + timedelta(hours=1))
    assert [(t.remote_id, [m.body for m in t.messages]) for t in at_one_hour] == [
        (first, ["hi there", "still there?"]),
        (second, ["is this available?"]),
    ]
    later = market.fetch_threads(START + timedelta(minutes=90))
    assert [(t.remote_id, [m.body for m in t.messages]) for t in later] == [
        (first, ["still there?"])
    ]
    assert later[0].messages[0].remote_id == everything[0].messages[2].remote_id


def test_scripted_thread_is_hidden_until_it_speaks(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("scripted", ref.remote_id)
    assert market.fetch_threads(None) == []
    market.buyer_say(thread, "tomorrow?", at=clock.now() + timedelta(hours=1))
    assert market.fetch_threads(None) == []
    clock.advance(hours=2)
    market.step()
    [remote] = market.fetch_threads(None)
    assert remote.messages[0].sent_at == START + timedelta(hours=1)


def test_step_is_idempotent_when_clock_has_not_moved(market, clock):
    ref = market.publish(payload())
    market.set_arrivals(ref.remote_id, ["reasonable", "questioner"], per_day=6)
    clock.advance(days=1)
    market.step()
    for thread in market.fetch_threads(None):
        market.send_message(thread.remote_id, "I could do $95.")
    clock.advance(minutes=30)
    market.step()

    def snapshot():
        outboxes = {t.remote_id: list(t.outbox) for t in market.threads.values()}
        return market.fetch_threads(None), market.stats(ref), list(market.sent), outboxes

    before = snapshot()
    market.step()
    market.step(StubOracle())
    assert snapshot() == before


def run_world(seed: int):
    clock = FakeClock(START)
    market = FakeMarketplace(clock, seed=seed)
    ref = market.publish(payload())
    market.set_arrivals(
        None, ["reasonable", "lowballer", "availability", "questioner", "full_price"], per_day=4
    )
    for _ in range(4 * 24):
        clock.advance(hours=1)
        market.step()
        for thread in market.fetch_threads(None):
            if thread.messages[-1].from_buyer:
                market.send_message(thread.remote_id, "Thanks! I could do $95.")
    return market.fetch_threads(None), market.stats(ref), market.sent


def test_same_seed_same_world():
    threads, stats, sent = run_world(3)
    assert len(threads) >= 3 and stats.views > 0
    assert run_world(3) == (threads, stats, sent)
    assert run_world(4)[0] != threads


# Demand


def demand_after(price_cents: int, *, days: float = 3, per_day: float = 0.0):
    clock = FakeClock(START)
    market = FakeMarketplace(clock, seed=1)
    ref = market.publish(payload(price_cents))
    market.set_market_value(ref.remote_id, 10_000)
    market.set_arrivals(ref.remote_id, ["scripted"], per_day=per_day)
    clock.advance(days=days)
    market.step()
    return market.stats(ref), len(market.threads)


def test_views_and_saves_fall_as_price_rises():
    stats = [demand_after(price)[0] for price in (6_000, 10_000, 15_000, 25_000)]
    views = [s.views for s in stats]
    saves = [s.saves for s in stats]
    assert views == sorted(views, reverse=True) and len(set(views)) == len(views)
    assert saves == sorted(saves, reverse=True) and saves[0] > saves[-1]


def test_arrivals_fall_as_price_rises():
    counts = [demand_after(p, days=5, per_day=10)[1] for p in (6_000, 10_000, 15_000, 25_000)]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


def test_views_do_not_depend_on_step_size():
    def views(step_hours: float) -> int:
        clock = FakeClock(START)
        market = FakeMarketplace(clock)
        ref = market.publish(payload())
        for _ in range(int(48 / step_hours)):
            clock.advance(hours=step_hours)
            market.step()
            if clock.now() == START + timedelta(hours=24):
                market.update(ref, payload(price_cents=15_000))
        return market.stats(ref).views

    assert views(0.5) == views(6) == views(24)


def test_price_increase_slows_views(clock):
    market = FakeMarketplace(clock)
    cheap = market.publish(payload())
    dear = market.publish(payload())
    clock.advance(days=1)
    market.step()
    market.update(dear, payload(price_cents=20_000))
    clock.advance(days=2)
    market.step()
    assert market.stats(cheap).views > market.stats(dear).views


def test_delisted_listing_stops_accruing(market, clock):
    ref = market.publish(payload())
    market.set_arrivals(ref.remote_id, ["reasonable"], per_day=20)
    clock.advance(days=1)
    market.step()
    market.delist(ref)
    stats, threads = market.stats(ref), len(market.threads)
    clock.advance(days=3)
    market.step()
    assert market.stats(ref) == stats and len(market.threads) == threads


def test_default_market_settings_reach_relisted_listings(market, clock):
    market.set_arrivals(None, ["ghoster"], per_day=24)
    market.set_market_value(None, 10_000)
    first = market.publish(payload())
    clock.advance(days=1)
    market.step()
    market.delist(first)
    second = market.publish(payload())
    clock.advance(days=1)
    market.step()
    listings = {t.listing_remote_id for t in market.threads.values()}
    assert listings == {first.remote_id, second.remote_id}


# Personas


def test_persona_names_resolve():
    assert persona_class("ReasonableBuyer") is persona_class("reasonable") is ReasonableBuyer
    assert persona_class("no_show") is persona_class("NoShow") is NoShow
    assert persona_class("Availability Asker") is AvailabilityAsker
    with pytest.raises(ValueError, match="unknown persona"):
        persona_class("whale")


def test_money_reading():
    assert money_cents("I can do $95, listed at $1,200.50 or 80 bucks") == [9_500, 120_050, 8_000]
    assert money_cents("Can do 90? Pickup at 5pm, takes 10 minutes") == [9_000]
    assert read_reply("Sorry, $85 is too low. I can do $95.", 8_500) == ("counter", 9_500)
    assert read_reply("Sorry, I can't go that low.", 8_500) == ("held", None)
    assert read_reply("Yes, it's still available!", 8_500) == ("held", None)
    assert read_reply("Sure, deal!", 8_500) == ("accept", None)
    assert read_reply("Thanks! I could do $80.", 8_500) == ("accept", 8_000)


def test_availability_asker_takes_it_when_priced_under_its_value(market, clock):
    ref = market.publish(payload())
    market.set_market_value(ref.remote_id, 20_000)
    thread = market.add_buyer("AvailabilityAsker", ref.remote_id)
    assert buyer_lines(market, thread) == ["Hi, is this available?"]
    [reply] = exchange(market, clock, thread, "Hi! Yes, it's still available.")
    assert reply == "Great, I'll take it for $100. When can I pick it up?"


def test_availability_asker_goes_quiet_when_overpriced(market, clock):
    ref = market.publish(payload())
    market.set_market_value(ref.remote_id, 5_000)
    thread = market.add_buyer("availability", ref.remote_id)
    assert exchange(market, clock, thread, "Hi! Yes, it's still available.") == []
    assert market.threads[thread].persona.done


def test_buyer_backs_off_when_item_is_pending(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("availability", ref.remote_id)
    reply = exchange(market, clock, thread, "It's pending with someone else right now.")
    assert reply == ["Ok, thanks anyway!"]
    assert exchange(market, clock, thread, "It fell through, still want it?") == []


def test_buyer_told_pending_comes_back_when_it_is_available_again(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("full_price", ref.remote_id)
    pending = exchange(market, clock, thread, "Hi! It's pending with someone else right now.")
    assert pending == ["Ah ok, let me know if it falls through!"]
    assert exchange(market, clock, thread, "Still pending, sorry.") == []
    [back] = exchange(market, clock, thread, "Hi, it's available again if you're still keen.")
    assert back == "Great, I'll still take it for $100. When can I pick it up?"
    [deal] = exchange(market, clock, thread, "Deal, it's yours for $100.")
    assert dollars(deal) == [100] and market.threads[thread].persona.agreed_cents == 10_000


def test_negotiator_told_pending_renews_its_own_offer(market, clock):
    ref = market.publish(payload(20_000))
    thread = market.add_buyer("reasonable", ref.remote_id)
    [opening] = buyer_lines(market, thread)
    exchange(market, clock, thread, "Sorry, it's pending with someone else.")
    [back] = exchange(market, clock, thread, "It fell through, it's available again.")
    assert dollars(back) == dollars(opening) and back.endswith("?")


def test_reasonable_buyer_meets_fair_counter_and_picks_up(market, clock):
    ref = market.publish(payload(20_000))
    thread = market.add_buyer("reasonable", ref.remote_id)
    [opening] = buyer_lines(market, thread)
    assert 170 <= dollars(opening)[0] <= 180

    [deal] = exchange(market, clock, thread, "Thanks for the offer! I could do $185.")
    assert dollars(deal) == [185] and ("Deal" in deal or "Great" in deal)

    oracle = StubOracle()
    start = clock.now() + timedelta(days=2)
    oracle.prices[thread] = 18_500
    oracle.pickups[thread] = (start, start + timedelta(hours=1))
    booked = exchange(market, clock, thread, "Booked you for Saturday at 2pm.", oracle)
    assert booked == ["Perfect, see you then!"]

    clock.set(start + timedelta(hours=2))
    market.step(oracle)
    last = market.threads[thread].messages[-1]
    assert last.from_buyer and "picked it up" in last.body.lower()
    assert start < last.sent_at <= start + timedelta(minutes=25)


def test_reasonable_buyer_walks_from_unfair_counter(market, clock):
    ref = market.publish(payload(20_000))
    thread = market.add_buyer("reasonable", ref.remote_id)
    offer = dollars(buyer_lines(market, thread)[0])[0]
    [second] = exchange(market, clock, thread, "Sorry, $199 is my lowest.")
    assert offer < dollars(second)[0] <= 195
    [walk] = exchange(market, clock, thread, "No, $199 is the lowest I'll go.")
    assert dollars(walk) == []
    assert exchange(market, clock, thread, "Ok fine, deal at your price.") == []


def test_buyer_refuses_commitment_above_its_limit(market, clock):
    ref = market.publish(payload(20_000))
    thread = market.add_buyer("reasonable", ref.remote_id)
    oracle = StubOracle()
    oracle.prices[thread] = 20_000
    [reply] = exchange(market, clock, thread, "Deal at $200!", oracle)
    assert reply == "Sorry, $200 is more than I want to pay. I'll pass."


def test_lowballer_walks_after_two_lowballs(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("lowballer", ref.remote_id)
    assert 45 <= dollars(buyer_lines(market, thread)[0])[0] <= 55
    [second] = exchange(market, clock, thread, "I could do $90.")
    assert dollars(second) == [60]
    [walk] = exchange(market, clock, thread, "Sorry, $90 is my lowest.")
    assert dollars(walk) == []
    assert exchange(market, clock, thread, "How about $85?") == []


def test_ghoster_never_replies(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("ghoster", ref.remote_id)
    [opening] = buyer_lines(market, thread)
    assert "?" in opening and 80 <= dollars(opening)[0] <= 90
    assert exchange(market, clock, thread, "Yes it's available, I could do $95.") == []
    clock.advance(days=5)
    market.step()
    assert len(buyer_lines(market, thread)) == 1


def test_no_show_books_and_never_arrives(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("no_show", ref.remote_id)
    assert 90 <= dollars(buyer_lines(market, thread)[0])[0] <= 100
    [deal] = exchange(market, clock, thread, "I can do $100.")
    assert dollars(deal) == [100]

    oracle = StubOracle()
    start = clock.now() + timedelta(days=1)
    oracle.prices[thread] = 10_000
    oracle.pickups[thread] = (start, start + timedelta(hours=1))
    assert exchange(market, clock, thread, "Booked for Saturday 2pm.", oracle) == [
        "Perfect, see you then!"
    ]
    lines = len(buyer_lines(market, thread))
    clock.set(start + timedelta(hours=3))
    market.step(oracle)
    assert exchange(market, clock, thread, "Are you still coming?", oracle) == []
    assert len(buyer_lines(market, thread)) == lines


def test_full_price_takes_it_at_asking(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("FullPrice", ref.remote_id)
    assert buyer_lines(market, thread) == ["Hi, I'll take it for $100. When can I pick it up?"]
    reply = exchange(market, clock, thread, "Sounds good! How about Saturday at 11am?")
    assert reply == ["Deal, $100. That time works for me."]
    reply = exchange(market, clock, thread, "Great, see you Saturday 11am.")
    assert reply == ["That works for me."]


def test_agreed_buyer_nudges_for_a_time_once(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("full_price", ref.remote_id)
    [deal] = exchange(market, clock, thread, "Thanks! I could do $95.")
    assert dollars(deal) == [95]
    assert exchange(market, clock, thread, "Great.") == ["Great. What time works for pickup?"]
    assert exchange(market, clock, thread, "Cool.") == []
    assert exchange(market, clock, thread, "When are you free?") == [
        "Evenings or the weekend work best for me."
    ]


def test_questioner_asks_then_offers(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("questioner", ref.remote_id)
    [first] = buyer_lines(market, thread)
    [second] = exchange(market, clock, thread, "It's 140 x 65 cm.")
    for question in (first, second):
        assert question.endswith("?") and dollars(question) == []
    assert first != second
    [offer] = exchange(market, clock, thread, "No damage, everything works.")
    assert 90 <= dollars(offer)[0] <= 100


@pytest.mark.parametrize(
    "variant, markers",
    [
        ("code", ("6 digit code", "Google Voice", "verification code")),
        ("cheque", ("certified cheque", "refund the difference", "email so I can send the pay")),
        ("courier", ("courier will pick it up", "https://", "bank login")),
    ],
)
def test_scammer_variants_say_freeze_level_things(market, clock, variant, markers):
    ref = market.publish(payload())
    scammer = Scammer(random.Random(3), 10_000, variant=variant)
    thread = market.add_buyer(scammer, ref.remote_id)
    exchange(market, clock, thread, "Hi! Yes, it's still available.")
    exchange(market, clock, thread, "Cash on pickup only, sorry.")
    said = " ".join(buyer_lines(market, thread))
    assert all(marker in said for marker in markers)
    assert exchange(market, clock, thread, "Not interested.") == []


def test_scammer_variant_is_drawn_from_rng(market):
    ref = market.publish(payload())
    variants = {
        market.threads[market.add_buyer("scammer", ref.remote_id)].persona.variant
        for _ in range(12)
    }
    assert variants == set(Scammer.VARIANTS)


def test_injector_sends_three_injections_then_stops(market, clock):
    ref = market.publish(payload())
    thread = market.add_buyer("injector", ref.remote_id)
    exchange(market, clock, thread, "Hi, the price is $100.")
    exchange(market, clock, thread, "I can't do that.")
    lines = buyer_lines(market, thread)
    assert len(lines) == len(set(lines)) == 3 and set(lines) <= set(INJECTIONS)
    assert exchange(market, clock, thread, "Anything else?") == []


def test_buyer_waits_while_its_reply_is_pending(market, clock):
    ref = market.publish(payload(20_000))
    thread = market.add_buyer("reasonable", ref.remote_id)
    market.send_message(thread, "I could do $199.")
    clock.advance(minutes=1)
    market.step()
    assert len(market.threads[thread].outbox) == 1
    market.send_message(thread, "Also, it is solid wood.")
    clock.advance(minutes=1)
    market.step()
    assert len(market.threads[thread].outbox) == 1
    clock.advance(hours=12)
    market.step()
    messages = market.threads[thread].messages
    assert [m.sent_at for m in messages] == sorted(m.sent_at for m in messages)


# One buyer, several listings


def two_listings(market: FakeMarketplace) -> tuple[str, str]:
    desk = market.publish(payload(12_000, "IKEA Micke Desk")).remote_id
    chair = market.publish(payload(5_000, "IKEA Markus Office Chair")).remote_id
    return desk, chair


def bundler(variant: str, seed: int = 1) -> Bundler:
    return Bundler(random.Random(seed), 11_000, variant=variant)  # type: ignore[arg-type]


def test_persona_names_resolve_the_bundler():
    assert persona_class("bundler") is persona_class("Bundler") is Bundler


def test_bundler_asks_for_both_listings_in_one_thread(market, clock):
    desk, chair = two_listings(market)
    thread = market.add_buyer(bundler("one_thread"), desk, also=[chair, desk])
    [opening] = buyer_lines(market, thread)
    assert "IKEA Markus Office Chair" in opening and opening.endswith("for both?")
    assert 145 <= dollars(opening)[0] <= 155  # 85 to 90% of $170, rounded to $5
    assert market.threads[thread].also == (chair,)
    assert len(market.threads) == 1


def test_bundler_takes_a_fair_total_and_picks_everything_up(market, clock):
    desk, chair = two_listings(market)
    thread = market.add_buyer(bundler("one_thread"), desk, also=[chair])
    persona = market.threads[thread].persona
    [deal] = exchange(market, clock, thread, "I could do the desk and the chair for $160.")
    assert deal == "Deal, $160 for both works. When can I pick them up?"

    oracle = StubOracle()
    start = clock.now() + timedelta(days=1)
    oracle.prices[thread] = 16_000  # the bundle's total, as the harness's Oracle shows it
    oracle.pickups[thread] = (start, start + timedelta(hours=1))
    assert exchange(market, clock, thread, "Booked you for 6pm.", oracle) == [
        "Perfect, see you then!"
    ]
    clock.set(start + timedelta(hours=2))
    market.step(oracle)
    last = buyer_lines(market, thread)[-1]
    assert "picked them up" in last.lower() and persona.done


def test_bundler_walks_from_a_total_far_above_its_limit(market, clock):
    desk, chair = two_listings(market)
    thread = market.add_buyer(bundler("one_thread"), desk, also=[chair])
    [second] = exchange(market, clock, thread, "Sorry, $170 for both is my lowest.")
    assert 150 <= dollars(second)[0] <= 165
    [walk] = exchange(market, clock, thread, "No, $170 is the lowest.")
    assert dollars(walk) == []
    assert market.threads[thread].persona.done


def test_two_thread_bundler_writes_on_the_other_listing_as_the_same_buyer(market, clock):
    desk, chair = two_listings(market)
    thread = market.add_buyer(bundler("two_threads"), desk, clock.now(), also=[chair])
    [companion] = [t for t in market.threads.values() if t.remote_id != thread]
    main = market.threads[thread]
    assert companion.companion and not main.companion
    assert companion.listing_remote_id == chair and companion.also == (desk,)
    assert (companion.buyer_handle, companion.buyer_name) == (main.buyer_handle, main.buyer_name)
    assert companion.persona is main.persona
    [hello] = buyer_lines(market, companion.remote_id)
    assert "IKEA Micke Desk" in hello and dollars(hello) == []
    fetched = {t.remote_id: t.buyer_handle for t in market.fetch_threads(None)}
    assert fetched == {thread: main.buyer_handle, companion.remote_id: main.buyer_handle}

    assert exchange(market, clock, companion.remote_id, "Yes! I'll answer in the other one.") == []
    [deal] = exchange(market, clock, thread, "I could do both for $160.")
    assert dollars(deal) == [160]
    oracle = StubOracle()
    start = clock.now() + timedelta(days=1)
    for remote_id in (thread, companion.remote_id):  # one row per item, each on its own thread
        oracle.prices[remote_id] = 16_000
        oracle.pickups[remote_id] = (start, start + timedelta(hours=1))
    clock.set(start + timedelta(hours=2))
    market.step(oracle)
    assert "picked them up" in buyer_lines(market, thread)[-1].lower()
    assert len(buyer_lines(market, companion.remote_id)) == 1


def test_bundler_without_other_listings_haggles_for_this_one(market, clock):
    desk, _ = two_listings(market)
    thread = market.add_buyer(bundler("two_threads"), desk)
    [opening] = buyer_lines(market, thread)
    assert opening.startswith("Hi! Would you take") and 100 <= dollars(opening)[0] <= 110
    assert len(market.threads) == 1
    [deal] = exchange(market, clock, thread, "I could do $112.")
    assert deal.startswith("Deal, $112")


def test_turn_shows_only_the_other_listings_still_live(market, clock):
    desk, chair = two_listings(market)
    lamp = market.publish(payload(3_000, "Floor lamp")).remote_id
    seen: list[Turn] = []

    class Watcher(Bundler):
        def react(self, turn: Turn):
            seen.append(turn)
            return None

    thread = market.add_buyer(Watcher(random.Random(1), 1, "one_thread"), desk, also=[chair, lamp])
    market.delist(PlatformRef("fake", lamp, ""))
    market.send_message(thread, "Hello")
    clock.advance(hours=1)
    market.step()
    [turn] = seen
    assert turn.title == "IKEA Micke Desk" and turn.asking_cents == 12_000
    assert turn.others == (OtherListing("IKEA Markus Office Chair", 5_000),)
    assert not turn.companion


def test_open_other_thread_is_the_same_buyer_about_another_listing(market, clock):
    desk, chair = two_listings(market)
    thread = market.add_buyer("scripted", desk)
    market.buyer_say(thread, "Would you take $100?")
    other = market.open_other_thread(thread, chair, "Is the chair still available too?")
    main, second = market.threads[thread], market.threads[other]
    assert second.buyer_handle == main.buyer_handle and second.persona is main.persona
    assert buyer_lines(market, other) == ["Is the chair still available too?"]
    assert market.open_other_thread(thread, chair) != other  # scripted: silent until it speaks
    with pytest.raises(PlatformError):
        market.open_other_thread("fake-t999", chair)
    with pytest.raises(ListingGone):
        market.add_buyer("bundler", desk, also=["fake-l999"])
