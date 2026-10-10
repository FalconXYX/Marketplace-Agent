"""In-memory marketplace for tests and simulation, deterministic for a given seed and call sequence.

Adapter calls change state at clock.now() and never advance the world. Only step() does: it
accrues views and saves, spawns arriving buyers, lets buyers react to seller messages and to the
Oracle, and delivers buyer messages whose time has come. Demand is constant between price changes
and whole days of listing age, so totals do not depend on how often step() runs.

| Demand | Rate |
|---|---|
| Views | BASE_VIEWS_PER_DAY x freshness x exp(-VIEW_ELASTICITY x (price/value - 1)) |
| Saves | BASE_SAVES_PER_DAY x freshness x exp(-SAVE_ELASTICITY x (price/value - 1)) |
| Arrivals | per_day x freshness x exp(-ARRIVAL_ELASTICITY x (price/value - 1)), Poisson |

price/value is clamped to [MIN_RATIO, MAX_RATIO]. Freshness halves every FRESH_HALF_LIFE_DAYS of
listing age down to FRESH_FLOOR, so a relist (a new publish) restores it. Value is the listing's
market value, else the market default, else its first price.

One buyer can write about several listings: add_buyer(also=) shows a persona the seller's other
listings it wants (Turn.others), and open_other_thread starts a thread by the same buyer (handle,
name, persona) on another listing, as a persona that writes_about_others does by itself.
"""
import itertools
import math
import random
import re
from bisect import insort
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Protocol

from marketplace_agent.core.clock import Clock
from marketplace_agent.platforms.base import (
    AwaitingHuman,
    ListingPayload,
    ListingStats,
    PlatformError,
    PlatformRef,
    RemoteMessage,
    RemoteThread,
    SearchResult,
)
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.platforms.fake.buyers import OtherListing, Persona, Turn, persona_class

DAY = timedelta(days=1)
BASE_VIEWS_PER_DAY = 40.0
BASE_SAVES_PER_DAY = 2.4
VIEW_ELASTICITY = 1.5
SAVE_ELASTICITY = 2.5
ARRIVAL_ELASTICITY = 3.0
MIN_RATIO, MAX_RATIO = 0.5, 3.0
FRESH_HALF_LIFE_DAYS = 5.0
FRESH_FLOOR = 0.3

_FIRST_NAMES = (
    "Alex", "Priya", "Jordan", "Sam", "Mei", "Omar", "Chris", "Fatima",
    "Liam", "Sofia", "Noah", "Aisha", "Ethan", "Hana", "Lucas", "Maya",
)


class Oracle(Protocol):
    """Ledger truth the simulation harness shows buyers, keyed by thread remote id."""

    def committed_price(self, thread_remote_id: str) -> int | None: ...

    def pickup(self, thread_remote_id: str) -> tuple[datetime, datetime] | None: ...


@dataclass(frozen=True)
class Arrivals:
    personas: tuple[type[Persona], ...]
    per_day: float


@dataclass
class FakeListing:
    ref: PlatformRef
    payload: ListingPayload
    published_at: datetime
    rng: random.Random
    threshold: float                     # integrated arrival demand that fires the next buyer
    prices: list[tuple[datetime, int]]   # price history, oldest first
    accrued_to: datetime
    delisted_at: datetime | None = None
    removed: bool = False                # gone from the platform: no adapter call finds it
    market_value_cents: int | None = None
    arrivals: Arrivals | None = None
    views: float = 0.0
    saves: float = 0.0
    hazard: float = 0.0                  # arrival demand integrated since the last buyer

    @property
    def live(self) -> bool:
        return self.delisted_at is None

    def price_at(self, when: datetime) -> int:
        price = self.prices[0][1]
        for changed_at, cents in self.prices:
            if changed_at > when:
                break
            price = cents
        return price


@dataclass
class FakeThread:
    remote_id: str
    listing_remote_id: str
    buyer_handle: str
    buyer_name: str
    persona: Persona
    messages: list[RemoteMessage] = field(default_factory=list)
    outbox: list[tuple[datetime, str, str]] = field(default_factory=list)  # (due, remote id, body)
    unread: list[str] = field(default_factory=list)  # seller messages the buyer has not reacted to
    also: tuple[str, ...] = ()  # other listings this buyer wants, by remote id
    companion: bool = False  # opened beside the buyer's main thread, about another listing


class FakeMarketplace:
    """PlatformAdapter over in-memory state, plus simulation hooks.

    sent holds (thread_remote_id, body, at) for every seller message as delivered. fail_next maps
    a method name to an exception raised once by its next call (healthcheck reports it instead).
    """

    def __init__(
        self,
        clock: Clock,
        *,
        seed: int = 0,
        name: str = "fake",
        search_catalog: Sequence[SearchResult] | None = None,
        id_prefix: str = "",
    ):
        """id_prefix starts every remote id this instance makes, so instances in different
        processes over one database never hand out the same post id."""
        self.clock = clock
        self.id_prefix = id_prefix
        self.seed = seed
        self.name = name
        self.search_catalog = list(search_catalog or ())
        self.listings: dict[str, FakeListing] = {}
        self.threads: dict[str, FakeThread] = {}
        self.sent: list[tuple[str, str, datetime]] = []
        self.fail_next: dict[str, Exception] = {}
        self._default_value_cents: int | None = None
        self._default_arrivals: Arrivals | None = None
        self._stepped_at: datetime | None = None
        self._ids = itertools.count(1)

    # PlatformAdapter

    def publish(self, listing: ListingPayload, *, assist: bool = False) -> PlatformRef:
        self._maybe_fail("publish")
        _validate(listing)
        if assist:
            raise AwaitingHuman(f"{self.name}: listing prepared; a human must submit it")
        now = self.clock.now()
        remote_id = self._new_id("l")
        ref = PlatformRef(self.name, remote_id, f"https://{self.name}.invalid/listings/{remote_id}")
        rng = random.Random(f"{self.seed}/{remote_id}")
        self.listings[remote_id] = FakeListing(
            ref=ref,
            payload=_copy(listing),
            published_at=now,
            rng=rng,
            threshold=rng.expovariate(1.0),
            prices=[(now, listing.price_cents)],
            accrued_to=now,
        )
        return ref

    def update(self, ref: PlatformRef, listing: ListingPayload) -> None:
        self._maybe_fail("update")
        found = self._live(ref.remote_id)
        _validate(listing)
        if listing.price_cents != found.payload.price_cents:
            found.prices.append((self.clock.now(), listing.price_cents))
        found.payload = _copy(listing)

    def delist(self, ref: PlatformRef) -> None:
        self._maybe_fail("delist")
        self._live(ref.remote_id).delisted_at = self.clock.now()

    def fetch_threads(self, since: datetime | None) -> list[RemoteThread]:
        """Threads with a message at or after since (all when None), with only those messages."""
        self._maybe_fail("fetch_threads")
        found = []
        for thread in self.threads.values():
            messages = [m for m in thread.messages if since is None or m.sent_at >= since]
            if messages:
                found.append(
                    RemoteThread(
                        remote_id=thread.remote_id,
                        listing_remote_id=thread.listing_remote_id,
                        buyer_handle=thread.buyer_handle,
                        buyer_name=thread.buyer_name,
                        messages=messages,
                    )
                )
        return found

    def send_message(self, thread_remote_id: str, body: str) -> str:
        self._maybe_fail("send_message")
        thread = self._thread(thread_remote_id)
        if not body.strip():
            raise PlatformError("message body is empty")
        now = self.clock.now()
        remote_id = self._new_id("m")
        insort(thread.messages, RemoteMessage(remote_id, body, now, from_buyer=False), key=_sent_at)
        thread.unread.append(body)
        self.sent.append((thread_remote_id, body, now))
        return remote_id

    def stats(self, ref: PlatformRef) -> ListingStats:
        """message_count is the number of buyers who have written about the listing."""
        self._maybe_fail("stats")
        listing = self._listing(ref.remote_id)
        conversations = sum(
            1
            for thread in self.threads.values()
            if thread.listing_remote_id == ref.remote_id
            and any(m.from_buyer for m in thread.messages)
        )
        return ListingStats(
            views=int(listing.views), saves=int(listing.saves), message_count=conversations
        )

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Catalog entries, then live listings here, whose titles contain every query word."""
        self._maybe_fail("search")
        words = re.findall(r"\w+", query.lower())
        own = [
            SearchResult(lst.payload.title, lst.payload.price_cents, lst.ref.url, lst.published_at)
            for lst in self.listings.values()
            if lst.live
        ]
        hits = [r for r in (*self.search_catalog, *own) if all(w in r.title.lower() for w in words)]
        return hits[: max(0, limit)]

    def healthcheck(self) -> list[str]:
        failure = self.fail_next.pop("healthcheck", None)
        return [] if failure is None else [f"{self.name}: {failure}"]

    # Simulation hooks

    def add_buyer(
        self,
        persona: str | Persona,
        listing_remote_id: str,
        at: datetime | None = None,
        *,
        also: Sequence[str] = (),
    ) -> str:
        """Open a thread; the buyer's first message, if any, arrives at `at` (default now).

        persona is a persona name (see buyers.persona_class) or a ready instance. also names
        other listings this buyer wants too: the persona sees the live ones in Turn.others, and
        one that writes_about_others opens a companion thread on each, all at `at`.
        """
        listing = self._listing(listing_remote_id)
        wanted = tuple(dict.fromkeys(r for r in also if r != listing_remote_id))
        for remote_id in wanted:
            self._listing(remote_id)
        when = at or self.clock.now()
        thread = self._open_thread(listing, persona, when, also=wanted)
        if thread.persona.writes_about_others:
            for remote_id in wanted:
                self._open_thread(
                    self._listing(remote_id),
                    thread.persona,
                    when,
                    also=(listing_remote_id,),
                    buyer=thread,
                )
        return thread.remote_id

    def open_other_thread(
        self,
        thread_remote_id: str,
        listing_remote_id: str,
        text: str | None = None,
        at: datetime | None = None,
    ) -> str:
        """The buyer of thread_remote_id writes about another listing: a new thread with the same
        handle, name, and persona. Its first message is text if given, else the persona's
        companion opening (none for most personas), arriving at `at` (default now)."""
        main = self._thread(thread_remote_id)
        listing = self._listing(listing_remote_id)
        when = at or self.clock.now()
        thread = self._open_thread(
            listing,
            main.persona,
            when,
            also=(main.listing_remote_id,),
            buyer=main,
            opening=text,
        )
        return thread.remote_id

    def buyer_say(self, thread_remote_id: str, text: str, at: datetime | None = None) -> str:
        """Scripted buyer message, delivered at `at` (default now; a future time waits for step)."""
        thread = self._thread(thread_remote_id)
        remote_id = self._schedule(thread, text, at or self.clock.now())
        self._deliver(thread, self.clock.now())
        return remote_id

    def remove_listing(self, listing_remote_id: str) -> None:
        """The platform takes the post down on its own (expiry, moderation, or the owner deleting
        it by hand): it is gone, so update, delist, and stats raise ListingGone."""
        listing = self._listing(listing_remote_id)
        listing.delisted_at = listing.delisted_at or self.clock.now()
        listing.removed = True

    def set_market_value(self, listing_remote_id: str | None, cents: int) -> None:
        """Value demand is measured against. None sets the default for listings without their own,
        which survives a relist. Applies from the last step on."""
        if cents <= 0:
            raise ValueError("market value must be positive cents")
        if listing_remote_id is None:
            self._default_value_cents = cents
        else:
            self._listing(listing_remote_id).market_value_cents = cents

    def set_arrivals(
        self, listing_remote_id: str | None, personas: Sequence[str], per_day: float
    ) -> None:
        """Buyers arrive at per_day (when priced at market value and fresh), each persona drawn
        uniformly from personas; repeat a name to weight it. None sets the default, as above."""
        if per_day < 0:
            raise ValueError("per_day cannot be negative")
        arrivals = Arrivals(tuple(persona_class(p) for p in personas), per_day)
        if listing_remote_id is None:
            self._default_arrivals = arrivals
        else:
            self._listing(listing_remote_id).arrivals = arrivals

    def step(self, oracle: Oracle | None = None) -> None:
        """Advance the world to clock.now(). A no-op unless the clock moved since the last step."""
        now = self.clock.now()
        if self._stepped_at is not None and now <= self._stepped_at:
            return
        self._stepped_at = now
        for listing in list(self.listings.values()):
            self._accrue(listing, now)
        for thread in list(self.threads.values()):
            self._deliver(thread, now)
            self._consult(thread, now, oracle)
            self._deliver(thread, now)

    # Internals

    def _maybe_fail(self, method: str) -> None:
        failure = self.fail_next.pop(method, None)
        if failure is not None:
            raise failure

    def _listing(self, remote_id: str) -> FakeListing:
        listing = self.listings.get(remote_id)
        if listing is None or listing.removed:
            raise ListingGone(f"{self.name}: listing {remote_id} not found")
        return listing

    def _live(self, remote_id: str) -> FakeListing:
        listing = self._listing(remote_id)
        if not listing.live:
            raise ListingGone(f"{self.name}: listing {remote_id} was delisted")
        return listing

    def _thread(self, remote_id: str) -> FakeThread:
        thread = self.threads.get(remote_id)
        if thread is None:
            raise PlatformError(f"{self.name}: thread {remote_id} not found")
        return thread

    def _market_value(self, listing: FakeListing) -> int:
        value = listing.market_value_cents or self._default_value_cents or listing.prices[0][1]
        return max(1, value)

    def _open_thread(
        self,
        listing: FakeListing,
        persona: str | type[Persona] | Persona,
        at: datetime,
        *,
        also: tuple[str, ...] = (),
        buyer: FakeThread | None = None,
        opening: str | None = None,
    ) -> FakeThread:
        """A new thread on listing. With buyer, the same person as in that thread writes again,
        about another listing: a companion thread. opening overrides the persona's first line."""
        number = next(self._ids)
        remote_id = f"{self.name}-{self.id_prefix}t{number}"
        rng = random.Random(f"{self.seed}/{remote_id}")
        first = rng.choice(_FIRST_NAMES)
        if not isinstance(persona, Persona):
            cls = persona_class(persona) if isinstance(persona, str) else persona
            persona = cls(rng, self._market_value(listing))
        handle, name = f"{first.lower()}{number}", f"{first} {rng.choice('ABCDEFGHJKLMNPRSTW')}."
        thread = FakeThread(
            remote_id=remote_id,
            listing_remote_id=listing.ref.remote_id,
            buyer_handle=buyer.buyer_handle if buyer else handle,
            buyer_name=buyer.buyer_name if buyer else name,
            persona=persona,
            also=also,
            companion=buyer is not None,
        )
        self.threads[remote_id] = thread
        if opening is None:
            opening = persona.first_message(self._turn(thread, at, (), at, None))
        if opening is not None:
            self._schedule(thread, opening, at)
            self._deliver(thread, self.clock.now())
        return thread

    def _turn(
        self,
        thread: FakeThread,
        now: datetime,
        seller_said: tuple[str, ...],
        last_message_at: datetime,
        oracle: Oracle | None,
    ) -> Turn:
        listing = self.listings[thread.listing_remote_id]
        others = tuple(
            OtherListing(other.payload.title, other.price_at(now))
            for other in (self.listings.get(r) for r in thread.also)
            if other is not None and other.live and not other.removed
        )
        return Turn(
            now=now,
            asking_cents=listing.price_at(now),
            seller_said=seller_said,
            last_message_at=last_message_at,
            committed_cents=oracle.committed_price(thread.remote_id) if oracle else None,
            pickup=oracle.pickup(thread.remote_id) if oracle else None,
            title=listing.payload.title,
            others=others,
            companion=thread.companion,
        )

    def _new_id(self, kind: str) -> str:
        return f"{self.name}-{self.id_prefix}{kind}{next(self._ids)}"

    def _schedule(self, thread: FakeThread, body: str, at: datetime) -> str:
        remote_id = self._new_id("m")
        insort(thread.outbox, (at, remote_id, body), key=lambda entry: entry[0])
        return remote_id

    def _deliver(self, thread: FakeThread, now: datetime) -> None:
        while thread.outbox and thread.outbox[0][0] <= now:
            at, remote_id, body = thread.outbox.pop(0)
            message = RemoteMessage(remote_id, body, at, from_buyer=True)
            insort(thread.messages, message, key=_sent_at)

    def _consult(self, thread: FakeThread, now: datetime, oracle: Oracle | None) -> None:
        """Give the buyer its move; it waits while one of its own messages is still pending."""
        if thread.persona.done or thread.outbox or not thread.messages:
            return
        turn = self._turn(
            thread, now, tuple(thread.unread), thread.messages[-1].sent_at, oracle
        )
        thread.unread.clear()
        reply = thread.persona.react(turn)
        if reply is not None:
            self._schedule(thread, reply.text, reply.at)

    def _accrue(self, listing: FakeListing, now: datetime) -> None:
        end = min(now, listing.delisted_at or now)
        if end <= listing.accrued_to:
            return
        value = self._market_value(listing)
        arrivals = listing.arrivals or self._default_arrivals
        for start, stop in _pieces(listing, listing.accrued_to, end):
            ratio = min(max(listing.price_at(start) / value, MIN_RATIO), MAX_RATIO)
            fresh = _freshness(listing, start)
            days = (stop - start) / DAY
            listing.views += BASE_VIEWS_PER_DAY * fresh * _demand(ratio, VIEW_ELASTICITY) * days
            listing.saves += BASE_SAVES_PER_DAY * fresh * _demand(ratio, SAVE_ELASTICITY) * days
            if arrivals and arrivals.personas and arrivals.per_day > 0:
                rate = arrivals.per_day * fresh * _demand(ratio, ARRIVAL_ELASTICITY)
                self._arrive(listing, arrivals.personas, start, stop, rate)
        listing.accrued_to = end

    def _arrive(
        self,
        listing: FakeListing,
        personas: tuple[type[Persona], ...],
        start: datetime,
        stop: datetime,
        rate: float,
    ) -> None:
        """Poisson arrivals: a buyer arrives each time integrated demand crosses an Exp(1) draw."""
        at = start
        while True:
            wait_days = max(0.0, listing.threshold - listing.hazard) / rate
            if wait_days > (stop - at) / DAY:
                listing.hazard += rate * ((stop - at) / DAY)
                return
            at = min(stop, at + timedelta(days=wait_days))
            listing.hazard = 0.0
            listing.threshold = listing.rng.expovariate(1.0)
            self._open_thread(listing, listing.rng.choice(personas), at)


def _validate(listing: ListingPayload) -> None:
    if not listing.title.strip():
        raise PlatformError("listing title is required")
    if listing.price_cents < 0:
        raise PlatformError("listing price cannot be negative")


def _copy(listing: ListingPayload) -> ListingPayload:
    return replace(
        listing, photo_paths=list(listing.photo_paths), attributes=dict(listing.attributes)
    )


def _sent_at(message: RemoteMessage) -> datetime:
    return message.sent_at


def _demand(ratio: float, elasticity: float) -> float:
    return math.exp(-elasticity * (ratio - 1.0))


def _freshness(listing: FakeListing, at: datetime) -> float:
    age_days = (at - listing.published_at) // DAY
    return max(FRESH_FLOOR, 0.5 ** (age_days / FRESH_HALF_LIFE_DAYS))


def _pieces(
    listing: FakeListing, start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """Split [start, end) at price changes and whole days of listing age."""
    cuts = {start, end}
    cuts.update(changed_at for changed_at, _ in listing.prices if start < changed_at < end)
    mark = listing.published_at + DAY * ((start - listing.published_at) // DAY + 1)
    while mark < end:
        cuts.add(mark)
        mark += DAY
    return list(itertools.pairwise(sorted(cuts)))
