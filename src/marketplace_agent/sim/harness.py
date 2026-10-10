"""Simulated selling campaigns: the real runtime against FakeMarketplace on a simulated clock.

| Part | In a simulation |
|---|---|
| Database | In-memory SQLite |
| Clock | TickingClock: moved one tick at a time, plus a microsecond per read |
| Platform | One FakeMarketplace, "fake", with seeded buyers and comparable listings |
| Items | The scenario's, plus Scenario.more: each with its own goal, market value, and buyers |
| Model | HeuristicPolicy, a provider you pass, or the real router paced in real time (live) |
| Owner | Confirms every item for sale, then never undoes, approves, or replies |
| Buyers | See the ledger through an Oracle: their deal's total and booked pickup |

Each tick advances the clock, steps the market, and runs one Runner.tick. The report's violations
come from sim.audit, which shares no code with the guard. A campaign is sold once every item is.
Entrants see every other item of the campaign (FakeMarketplace add_buyer also=), so a Bundler
can ask for several at once.
"""

import json
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from sqlmodel import Session, col, func, select

from marketplace_agent.ai.base import Capability, LLMProvider, LLMRequest, LLMResponse, RateLimited
from marketplace_agent.ai.factory import build_router
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.db import init_db, make_engine, session_scope
from marketplace_agent.core.items import create_item, set_attributes, set_goal
from marketplace_agent.core.models import (
    AgentRun,
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Escalation,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    Photo,
    PriceChange,
    QuotaUsage,
    Thread,
)
from marketplace_agent.core.runtime import Runtime, build_runtime
from marketplace_agent.guard.invariants import to_local
from marketplace_agent.guard.ledger import active_commitment, deal_rows
from marketplace_agent.notify.base import RecordingNotifier
from marketplace_agent.platforms.base import SearchResult
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.runner.tick import TickReport
from marketplace_agent.sim.audit import audit
from marketplace_agent.sim.policy import HeuristicPolicy
from marketplace_agent.sim.scenarios import Scenario

START = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)  # a Thursday, noon in Toronto
PLATFORM = "fake"
PHOTOS = 2
COMPS = 6
TRANSCRIPT_BODY_CHARS = 300
LIVE_RETRIES = 3  # per call, after a per-minute rate limit
LIVE_RETRY_WAIT_S = 60.0
LIVE_MAX_WAIT_S = 120.0


class TickingClock(FakeClock):
    """A FakeClock that moves one microsecond on every read, as a real clock does.

    Every write then has its own instant, so sim.audit, which orders rows from different tables
    by time, sees the order they happened in: a message sent just before a release or a price
    raise in the same wake is not read as sent after it.
    """

    def now(self) -> datetime:
        return self.advance(timedelta(microseconds=1))


class RealTimePacer:
    """Wraps a real provider for --live. A simulation plays days in minutes, so real calls would
    burst past the provider's per-minute limit while the quota tracker, on simulated time, sees
    none of it. This spaces calls 60/rpm real seconds apart and waits out a per-minute rate
    limit before retrying; simulated time stands still meanwhile."""

    def __init__(
        self,
        inner: LLMProvider,
        rpm: int,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.inner = inner
        self.name = inner.name
        self.capabilities = inner.capabilities
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None

    def model_for(self, capability: Capability) -> str:
        return self.inner.model_for(capability)

    def generate(self, request: LLMRequest) -> LLMResponse:
        retries = 0
        while True:
            self._space()
            try:
                return self.inner.generate(request)
            except RateLimited as exc:
                retries += 1
                if getattr(exc, "daily", False) or retries > LIVE_RETRIES:
                    raise
                wait = getattr(exc, "retry_after", None) or LIVE_RETRY_WAIT_S
                self._sleep(min(wait, LIVE_MAX_WAIT_S))

    def _space(self) -> None:
        now = self._monotonic()
        if self._last is not None and now - self._last < self.interval:
            self._sleep(self.interval - (now - self._last))
            now = self._monotonic()
        self._last = now


class LiveModelMissing(RuntimeError):
    """--live was asked for, but no model provider key is configured."""


@dataclass
class SimReport:
    scenario: str
    seed: int
    sold: bool  # every item of the campaign
    sale_price_cents: int | None  # what the sold items fetched together
    days_to_sell: float | None  # first publish to the last sale
    model_requests: int
    cache_replies: int
    outbound: int  # messages delivered to buyers
    escalations: int
    violations: list[str]
    transcript: list[str]
    days_run: float
    requests_by_day: dict[str, int]  # the provider's quota day -> model requests
    errors: list[str] = field(default_factory=list)  # from Runner tick reports

    @property
    def max_requests_per_day(self) -> int:
        return max(self.requests_by_day.values(), default=0)


@dataclass(frozen=True)
class LedgerOracle:
    """What each buyer may know, keyed by thread remote id: FakeMarketplace's Oracle."""

    prices: Mapping[str, int]
    pickups: Mapping[str, tuple[datetime, datetime]]

    def committed_price(self, thread_remote_id: str) -> int | None:
        return self.prices.get(thread_remote_id)

    def pickup(self, thread_remote_id: str) -> tuple[datetime, datetime] | None:
        return self.pickups.get(thread_remote_id)


# All --live takes from the owner's config: which model, and the key to call it.
LIVE_FIELDS = (
    "gemini_api_key",
    "groq_api_key",
    "openrouter_api_key",
    "vision_model",
    "reasoning_model",
    "cheap_model",
    "groq_model",
    "openrouter_model",
)


class _SimSettings(Settings):
    """Defaults and the values passed in, nothing else: a local marketplace_agent.toml or MA_*
    variable (caps without the fake, another undo window) would change the run, so the same
    --seed would not replay it."""

    model_config = SettingsConfigDict(**{**Settings.model_config, "toml_file": None})

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def sim_settings(*, live: bool = False, **overrides: Any) -> Settings:
    """Default settings plus a pickup area and a made-up address, whatever the local config
    says. live adds LIVE_FIELDS from .env, the environment, and marketplace_agent.toml."""
    values: dict[str, Any] = {
        "data_dir": Path("data"),
        "db_url": "sqlite://",
        "pickup_area": "Bloor St W & Ossington Ave",
        "pickup_address": "742 Evergreen Terrace",
        "enabled_platforms": [PLATFORM],
    }
    if live:
        local = Settings(_env_file=".env")
        values |= {name: getattr(local, name) for name in LIVE_FIELDS}
    return _SimSettings(**(values | overrides))


class Simulation:
    """One simulated world. run() plays a whole campaign; tests can drive tick() themselves."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        seed: int = 0,
        provider: LLMProvider | None = None,
        settings: Settings | None = None,
        live: bool = False,
        clock: FakeClock | None = None,
    ):
        self.scenario = scenario
        self.seed = seed
        self.settings = settings or sim_settings()
        self.clock = clock or TickingClock(START)
        self.started = self.clock.now()
        self.engine = make_engine("sqlite://")
        init_db(self.engine)
        catalog = [comp for spec in (scenario, *scenario.more) for comp in _comps(spec, seed)]
        self.market = FakeMarketplace(self.clock, seed=seed, search_catalog=catalog, name=PLATFORM)
        self.market.set_market_value(None, scenario.market_value_cents)
        if scenario.arrivals and scenario.arrivals_per_day > 0:
            self.market.set_arrivals(None, scenario.arrivals, scenario.arrivals_per_day)
        self.notifier = RecordingNotifier()
        self.runtime: Runtime = build_runtime(
            self.settings,
            clock=self.clock,
            notifier=self.notifier,
            platforms={PLATFORM: self.market},
            router=self._router(provider, live),
            engine=self.engine,
        )
        self.cache_replies = 0
        self.errors: list[str] = []
        self.items: dict[int, Scenario] = {}  # every item of the campaign, the scenario's first
        self.listings: dict[int, str] = {}  # item id -> its first listing's remote id
        self._configured: set[str] = set()  # listings given their own item's market settings
        self.item_id, self.listing_remote_id = self._publish(scenario)
        for extra in scenario.more:
            self._publish(extra)
        for item_id, spec in self.items.items():
            self._schedule_entrants(spec, self.listings[item_id])

    def _router(self, provider: LLMProvider | None, live: bool) -> ProviderRouter:
        if live:
            router = build_router(self.settings, self.engine, self.clock)
            if not router.providers:
                raise LiveModelMissing(
                    "--live needs a model: set GEMINI_API_KEY (free key at "
                    "https://aistudio.google.com), or drop --live to use the offline policy."
                )
            limits = self.settings.provider_limits
            router.providers = [
                RealTimePacer(p, limits[p.name].rpm) if p.name in limits else p
                for p in router.providers
            ]
            return router
        if provider is None:
            policy = HeuristicPolicy(self.engine, self.settings, self.clock)
            provider = ScriptedProvider(policy, name="heuristic", model="heuristic-1")
        quota = QuotaTracker(
            self.engine,
            self.clock,
            self.settings.provider_limits,
            self.settings.quota_constrained_fraction,
        )
        return ProviderRouter([provider], quota)

    def add_item(self, scenario: Scenario) -> tuple[int, str]:
        """Create another item with photos and goal, have the owner confirm and publish it, and
        schedule its entrants, who also see the items already here. Returns (item id, listing
        remote id)."""
        item_id, remote_id = self._publish(scenario)
        self._schedule_entrants(scenario, remote_id)
        return item_id, remote_id

    def _publish(self, scenario: Scenario) -> tuple[int, str]:
        """Create the item with photos and goal and have the owner confirm and publish it. Every
        item after the first gets its own market value and arrivals on its listings."""
        with session_scope(self.engine) as session:
            item = create_item(session, self.clock, scenario.hint)
            assert item.id is not None
            item.title, item.description = scenario.title, scenario.description
            item.category, item.condition = scenario.category, scenario.condition
            set_attributes(item, dict(scenario.attributes))
            session.add(item)
            session.add_all(_photos(item.id, scenario.name))
            set_goal(
                session,
                self.clock,
                item.id,
                list_price_cents=scenario.list_price_cents,
                floor_price_cents=scenario.floor_price_cents,
                target_price_cents=scenario.target_price_cents,
                deadline_days=scenario.days,
                firmness=scenario.firmness,
            )
            item_id = item.id
        failed = [r.summary for r in self.runtime.control.confirm_and_publish(item_id) if not r.ok]
        if failed:
            raise RuntimeError(f"Publishing {scenario.name} failed: {'; '.join(failed)}")
        with Session(self.engine) as session:
            remote_id = session.exec(
                select(Listing.remote_id).where(
                    Listing.item_id == item_id, Listing.status == ListingStatus.LIVE
                )
            ).one()
        assert remote_id is not None
        first = not self.items
        self.items[item_id] = scenario
        self.listings[item_id] = remote_id
        if not first:
            self._configure(remote_id, scenario)
        return item_id, remote_id

    def _schedule_entrants(self, scenario: Scenario, remote_id: str) -> None:
        also = [other for other in self.listings.values() if other != remote_id]
        for hours, persona in scenario.entrants:
            at = self.clock.now() + timedelta(hours=hours)
            self.market.add_buyer(persona, remote_id, at, also=also)

    def _configure(self, remote_id: str, scenario: Scenario) -> None:
        """The market defaults are the first item's; another item's listings get its own."""
        self.market.set_market_value(remote_id, scenario.market_value_cents)
        self.market.set_arrivals(remote_id, scenario.arrivals, scenario.arrivals_per_day)
        self._configured.add(remote_id)

    def _sync_listings(self) -> None:
        """A relist is a new listing: give it its item's market settings too."""
        extra = [item_id for item_id in self.items if item_id != self.item_id]
        if not extra:
            return
        with Session(self.engine) as session:
            live = session.exec(
                select(Listing.item_id, Listing.remote_id).where(
                    col(Listing.item_id).in_(extra), Listing.status == ListingStatus.LIVE
                )
            ).all()
        for item_id, remote_id in live:
            known = remote_id in self.market.listings
            if remote_id and known and remote_id not in self._configured:
                self._configure(remote_id, self.items[item_id])

    def tick(self, minutes: float = 30) -> TickReport:
        self.clock.advance(minutes=minutes)
        self._sync_listings()
        self.market.step(self.oracle())
        report = self.runtime.runner.tick()
        self.cache_replies += report.cache_replies
        stamp = f"{report.started_at:%Y-%m-%dT%H:%MZ}"
        self.errors += [f"{stamp} {error}" for error in report.errors]
        return report

    def oracle(self) -> LedgerOracle:
        """A bundle's buyer knows the total it agreed to, in every thread its rows are on."""
        prices: dict[str, int] = {}
        pickups: dict[str, tuple[datetime, datetime]] = {}
        with Session(self.engine) as session:
            for item_id in session.exec(select(Commitment.item_id).distinct()).all():
                active = active_commitment(session, item_id)
                if active is None:
                    continue
                thread = session.get(Thread, active.accepted.thread_id)
                assert thread is not None
                rows = deal_rows(session, active.accepted)
                prices[thread.remote_id] = sum(row.price_cents or 0 for row in rows)
                booked = active.pickup
                if booked and booked.pickup_start and booked.pickup_end:
                    pickups[thread.remote_id] = (booked.pickup_start, booked.pickup_end)
        return LedgerOracle(prices, pickups)

    def sold(self) -> bool:
        """Every item of the campaign is sold."""
        with Session(self.engine) as session:
            statuses = session.exec(select(Item.status).where(col(Item.id).in_(self.items))).all()
            return len(statuses) == len(self.items) and all(
                status == ItemStatus.SOLD for status in statuses
            )

    def audit(self) -> list[str]:
        with Session(self.engine) as session:
            return audit(session, self.settings, sent_log=self.market.sent)

    def run(self, *, days: float | None = None, tick_minutes: float = 30) -> SimReport:
        """Tick until the item sells or `days` (default: the scenario's) have passed."""
        end = self.started + timedelta(days=self.scenario.days if days is None else days)
        while self.clock.now() < end and not self.sold():
            self.tick(tick_minutes)
        return self.report()

    def report(self) -> SimReport:
        with Session(self.engine) as session:
            sales = [sale for item_id in self.items if (sale := _sale(session, item_id))]
            sold = len(sales) == len(self.items)
            last = max((sale.created_at for sale in sales), default=None)
            published = session.exec(
                select(func.min(Listing.published_at)).where(col(Listing.item_id).in_(self.items))
            ).one()
            usage = Counter[str]()
            for row in session.exec(select(QuotaUsage)).all():
                usage[row.day] += row.requests
            return SimReport(
                scenario=self.scenario.name,
                seed=self.seed,
                sold=sold,
                sale_price_cents=sum(s.price_cents or 0 for s in sales) if sales else None,
                days_to_sell=_days(last - published) if sold and last and published else None,
                model_requests=sum(usage.values()),
                cache_replies=self.cache_replies,
                outbound=len(self.market.sent),
                escalations=len(session.exec(select(Escalation.id)).all()),
                violations=audit(session, self.settings, sent_log=self.market.sent),
                transcript=_transcript(session, self.settings),
                days_run=_days(self.clock.now() - self.started),
                requests_by_day=dict(sorted(usage.items())),
                errors=list(self.errors),
            )

    def close(self) -> None:
        self.engine.dispose()


def run_simulation(
    scenario: Scenario,
    *,
    seed: int = 0,
    provider: LLMProvider | None = None,
    tick_minutes: float = 30,
    days: float | None = None,
    settings: Settings | None = None,
    live: bool = False,
) -> SimReport:
    simulation = Simulation(scenario, seed=seed, provider=provider, settings=settings, live=live)
    try:
        return simulation.run(days=days, tick_minutes=tick_minutes)
    finally:
        simulation.close()


# ---------------------------------------------------------------- setup


def _photos(item_id: int, name: str) -> list[Photo]:
    """Rows only: FakeMarketplace never opens the files."""
    return [
        Photo(
            item_id=item_id,
            path=f"sim/{name}-{n + 1}.jpg",
            position=n,
            width=1200,
            height=900,
            sha256=f"{item_id:032x}{n:032x}",
        )
        for n in range(PHOTOS)
    ]


def _comps(scenario: Scenario, seed: int) -> list[SearchResult]:
    """Other sellers' listings around market value, for query_comps."""
    rng = random.Random(f"{seed}/comps/{scenario.name}")
    return [
        SearchResult(
            title=scenario.title,
            price_cents=int(round(scenario.market_value_cents * rng.uniform(0.8, 1.2), -2)),
            url=f"https://comps.invalid/{scenario.name}/{n}",
        )
        for n in range(COMPS)
    ]


# ---------------------------------------------------------------- report


def _days(span: timedelta) -> float:
    return round(span.total_seconds() / 86_400, 2)


def _sale(session: Session, item_id: int) -> Commitment | None:
    """The COMPLETED event, carrying the accepted price."""
    return session.exec(
        select(Commitment).where(
            Commitment.item_id == item_id, Commitment.kind == CommitmentKind.COMPLETED
        )
    ).first()


def _transcript(session: Session, settings: Settings) -> list[str]:
    """Everything that happened, oldest first: messages (stored redacted, so the address shows
    as the placeholder), ledger events, price changes, escalations, and agent runs."""
    names = _thread_names(session)
    several = len(session.exec(select(Item.id)).all()) > 1
    events: list[tuple[datetime, int, int, str]] = []
    for run in session.exec(select(AgentRun)).all():
        events.append((run.started_at, 1, run.id or 0, _run_line(run)))
    for message in session.exec(select(Message)).all():
        who = names.get(message.thread_id, "?")
        if message.direction == Direction.INBOUND:
            line, rank = f"{who}: {_clip(message.body)}", 0
        else:
            rank = 0 if message.via == "pattern_cache" else 3  # cache replies precede wakes
            line = f"seller to {who} ({message.via}): {_clip(message.body)}"
        events.append((message.sent_at, rank, message.id or 0, line))
    for event in session.exec(select(Commitment)).all():
        line = _ledger_line(settings, event, names, several=several)
        events.append((event.created_at, 2, event.id or 0, line))
    for change in session.exec(select(PriceChange)).all():
        line = f"price {_money(change.old_cents)} -> {_money(change.new_cents)} by {change.actor}"
        events.append((change.created_at, 2, change.id or 0, line))
    for escalation in session.exec(select(Escalation)).all():
        events.append(
            (escalation.created_at, 4, escalation.id or 0, f"escalated: {escalation.reason}")
        )
    events.sort(key=lambda e: e[:3])
    return [f"{_local(settings, when)}  {text}" for when, _, _, text in events]


def _thread_names(session: Session) -> dict[int, str]:
    rows = session.exec(select(Thread, Buyer).join(Buyer, col(Buyer.id) == Thread.buyer_id)).all()
    return {
        thread.id: f"[{thread.id}] {buyer.display_name or buyer.handle}"
        for thread, buyer in rows
        if thread.id is not None
    }


def _run_line(run: AgentRun) -> str:
    triggers = ", ".join(t.get("kind", "?") for t in json.loads(run.triggers_json or "[]"))
    outcomes = json.loads(run.results_json or "[]")
    line = f"agent woke ({triggers}; {run.requests_used} request(s)): {run.rationale or '-'}"
    unusual = [o for o in outcomes if o.get("status") not in ("executed", "observed")]
    line += "".join(f"\n    {o['tool']} {o['status']}: {o['detail']}" for o in unusual)
    return line + (f"\n    error: {run.error}" if run.error else "")


def _ledger_line(
    settings: Settings, event: Commitment, names: dict[int, str], *, several: bool = False
) -> str:
    who = names.get(event.thread_id or 0, "?")
    price = _money(event.price_cents or 0)
    what = ""
    if several:
        bundle = f", {event.bundle_key}" if event.bundle_key else ""
        what = f" (item {event.item_id}{bundle})"
    if event.kind == CommitmentKind.ACCEPTED:
        return f"deal: accepted {price}{what} from {who}"
    if event.kind == CommitmentKind.PICKUP_BOOKED:
        assert event.pickup_start is not None
        when = _local(settings, event.pickup_start)
        return f"deal: pickup{what} with {who} booked for {when}"
    if event.kind == CommitmentKind.COMPLETED:
        return f"deal: SOLD{what} to {who} for {price}"
    reason = event.reason or "no reason"
    return f"deal: {event.kind.value}{what} with {who} ({reason}) by {event.actor}"


def _local(settings: Settings, when: datetime) -> str:
    return f"{to_local(settings, when):%a %b %d %H:%M}"


def _money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= TRANSCRIPT_BODY_CHARS:
        return flat
    return flat[:TRANSCRIPT_BODY_CHARS] + "..."
