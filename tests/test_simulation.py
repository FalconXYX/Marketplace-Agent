"""Whole campaigns: the real runtime, HeuristicPolicy as the model, scripted buyers, sim.audit.

Each scenario is simulated once at seed 0 and shared by the tests in this module.

Measured at seed 0 (model requests on the busiest simulated day): desk 8, lowballers 7,
scam_wave 5, no_show_recovery 7, injection 9, firm 8, bundle 7 (two items). Gemini's free tier
allows 250 a day.
"""

import json
import random
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlmodel import Session, col, select
from typer.testing import CliRunner

from marketplace_agent.ai.base import (
    Capability,
    LLMRequest,
    LLMResponse,
    QuotaState,
    RateLimited,
)
from marketplace_agent.config import ProviderLimit, Settings
from marketplace_agent.cli import app
from marketplace_agent.core.models import (
    AgentRun,
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Escalation,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    PriceChange,
    Thread,
    ThreadState,
)
from marketplace_agent.guard.invariants import detect_scam
from marketplace_agent.platforms.fake import Injector, NoShow
from marketplace_agent.platforms.fake.buyers import Bundler
from marketplace_agent.sim.harness import (
    RealTimePacer,
    SimReport,
    Simulation,
    run_simulation,
    sim_settings,
)
from marketplace_agent.sim.scenarios import SCENARIOS, Scenario
from marketplace_agent.tools.registry import default_registry

DESK_REQUESTS_PER_DAY = 25  # free tier: Gemini 250/day, shared by every listing the owner has
ANY_REQUESTS_PER_DAY = 25
STALE = Scenario(
    name="stale",
    hint="Patio set $150",
    title="Patio Table and 4 Chairs",
    description="Glass-top patio table and four stacking chairs.",
    list_price_cents=15_000,
    floor_price_cents=11_000,
    target_price_cents=13_000,
    market_value_cents=10_000,
    arrivals=("lowballer",),
    arrivals_per_day=1.0,
    days=10,
)


@pytest.fixture(scope="module")
def sims() -> Iterator[dict[str, Simulation]]:
    runs = {name: Simulation(scenario, seed=0) for name, scenario in SCENARIOS.items()}
    for sim in runs.values():
        sim.run()
    yield runs
    for sim in runs.values():
        sim.close()


@pytest.fixture(scope="module")
def reports(sims: dict[str, Simulation]) -> dict[str, SimReport]:
    return {name: sim.report() for name, sim in sims.items()}


@pytest.fixture(scope="module")
def stale() -> Iterator[Simulation]:
    """Only lowballers, so the listing goes stale and the seller has to tune it."""
    sim = Simulation(STALE, seed=0)
    sim.run()
    yield sim
    sim.close()


def _ledger(sim: Simulation, kind: CommitmentKind) -> list[Commitment]:
    with Session(sim.engine) as session:
        query = select(Commitment).where(Commitment.kind == kind).order_by(col(Commitment.id))
        return list(session.exec(query).all())


def _outcomes(sim: Simulation) -> list[dict]:
    with Session(sim.engine) as session:
        runs = session.exec(select(AgentRun).order_by(col(AgentRun.id))).all()
        return [o for run in runs for o in json.loads(run.results_json or "[]")]


def _goal(sim: Simulation) -> Goal:
    with Session(sim.engine) as session:
        goal = session.exec(select(Goal).where(Goal.item_id == sim.item_id)).one()
        session.expunge(goal)
        return goal


def _persona_threads(sim: Simulation, persona: type) -> list[str]:
    return [
        remote_id
        for remote_id, thread in sim.market.threads.items()
        if isinstance(thread.persona, persona)
    ]


# ---------------------------------------------------------------- every scenario


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_runs_clean(name: str, reports: dict[str, SimReport]):
    report = reports[name]
    assert report.violations == []
    assert report.errors == []
    assert report.sold or report.days_run >= SCENARIOS[name].days
    assert report.model_requests > 0 and report.outbound > 0


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_policy_never_trips_the_guard(name: str, sims: dict[str, Simulation]):
    """The stand-in seller plays by the rules, so every rejection in a run is a policy bug."""
    statuses = {(o["tool"], o["status"]) for o in _outcomes(sims[name])}
    assert {status for _, status in statuses} <= {"executed", "observed"}, statuses


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_requests_per_day_fit_the_free_tier(name: str, reports: dict[str, SimReport]):
    ceiling = DESK_REQUESTS_PER_DAY if name == "desk" else ANY_REQUESTS_PER_DAY
    assert 0 < reports[name].max_requests_per_day <= ceiling, reports[name].requests_by_day


def test_every_tool_is_exercised(sims: dict[str, Simulation], stale: Simulation):
    used = {
        o["tool"]
        for sim in [*sims.values(), stale]
        for o in _outcomes(sim)
        if o["status"] in ("executed", "observed")
    }
    # publish runs once per campaign, as the owner confirming the item; the fake platform is the
    # only one enabled, so the agent never has another platform to publish on.
    assert used == set(default_registry().names()) - {"publish"}
    for sim in sims.values():
        with Session(sim.engine) as session:
            assert session.exec(select(Listing).where(Listing.item_id == sim.item_id)).first()


# ---------------------------------------------------------------- per scenario


def test_desk_sells_at_or_above_floor(sims, reports):
    report = reports["desk"]
    goal = _goal(sims["desk"])
    assert report.sold
    assert report.sale_price_cents is not None
    assert report.sale_price_cents >= goal.floor_price_cents
    assert report.days_to_sell is not None and report.days_to_sell <= SCENARIOS["desk"].days


def test_no_show_recovery_releases_then_sells_to_another_buyer(sims):
    sim = sims["no_show_recovery"]
    released = _ledger(sim, CommitmentKind.RELEASED)
    [sale] = _ledger(sim, CommitmentKind.COMPLETED)
    no_shows = set(_persona_threads(sim, NoShow))
    with Session(sim.engine) as session:
        released_threads = [session.get(Thread, r.thread_id) for r in released]
        sold_thread = session.get(Thread, sale.thread_id)
        assert any(t.remote_id in no_shows for t in released_threads)
        assert sold_thread.remote_id not in no_shows
        assert sold_thread.buyer_id not in {t.buyer_id for t in released_threads}
        no_show_buyer = session.get(Buyer, released_threads[0].buyer_id)
        assert no_show_buyer.no_show_count == 1
    assert released[0].created_at < sale.created_at
    assert released[0].reason.startswith("no-show")


def test_scam_wave_freezes_every_scam_thread_and_never_writes_to_it_again(sims):
    sim = sims["scam_wave"]
    with Session(sim.engine) as session:
        threads = session.exec(select(Thread)).all()
        scams = {}
        for thread in threads:
            inbound = session.exec(
                select(Message)
                .where(Message.thread_id == thread.id, Message.direction == Direction.INBOUND)
                .order_by(col(Message.sent_at))
            ).all()
            first_scam = next(
                (m for m in inbound if (s := detect_scam(m.body)) and s.severity == "freeze"),
                None,
            )
            if first_scam is not None:
                scams[thread.remote_id] = first_scam.sent_at
                assert thread.state == ThreadState.FROZEN
                assert session.get(Buyer, thread.buyer_id).flagged_scam
                assert session.exec(
                    select(Escalation).where(Escalation.thread_id == thread.id)
                ).first()
    assert len(scams) >= 3
    late = [(r, at) for r, _, at in sim.market.sent if r in scams and at >= scams[r]]
    assert late == []
    assert sims["scam_wave"].sold()


def test_injection_never_moves_a_deal_below_floor(sims):
    sim = sims["injection"]
    goal = _goal(sim)
    accepted = _ledger(sim, CommitmentKind.ACCEPTED)
    assert accepted and all(a.price_cents >= goal.floor_price_cents for a in accepted)
    injectors = set(_persona_threads(sim, Injector))
    assert injectors
    with Session(sim.engine) as session:
        for remote_id in injectors:
            thread = session.exec(select(Thread).where(Thread.remote_id == remote_id)).first()
            if thread is None:  # arrived after the sale, never fetched
                continue
            assert all(a.thread_id != thread.id for a in accepted)
            assert session.exec(select(Escalation).where(Escalation.thread_id == thread.id)).first()
    address = sim.settings.pickup_address
    assert all(address not in body for r, body, _ in sim.market.sent if r in injectors)


def test_firm_never_accepts_below_list(sims):
    sim = sims["firm"]
    goal = _goal(sim)
    accepted = _ledger(sim, CommitmentKind.ACCEPTED)
    assert accepted and all(a.price_cents >= goal.list_price_cents for a in accepted)
    with Session(sim.engine) as session:
        assert session.exec(select(PriceChange)).all() == []


def test_stale_listing_is_tuned_but_never_below_floor(stale: Simulation):
    report = stale.report()
    assert report.violations == []
    assert not report.sold
    with Session(stale.engine) as session:
        changes = session.exec(select(PriceChange).order_by(col(PriceChange.id))).all()
    assert len(changes) >= 2
    assert all(STALE.floor_price_cents <= c.new_cents < c.old_cents for c in changes)
    tuned = {o["tool"] for o in _outcomes(stale) if o["status"] == "executed"}
    assert {"rewrite_listing", "update_price", "relist"} <= tuned


def _accepted_bundles(sim: Simulation) -> list[dict]:
    with Session(sim.engine) as session:
        runs = session.exec(select(AgentRun).order_by(col(AgentRun.id))).all()
        return [
            action
            for run in runs
            for plan in json.loads(run.plan_json or "[]")
            for action in plan.get("actions", [])
            if action.get("tool") == "accept_bundle"
        ]


def test_bundle_sells_both_items_as_one_deal(sims, reports):
    sim, report = sims["bundle"], reports["bundle"]
    assert report.sold and len(sim.items) == 2
    [bundle] = [o for o in _outcomes(sim) if o["tool"] == "accept_bundle"]
    assert bundle["status"] == "executed"
    accepted = _ledger(sim, CommitmentKind.ACCEPTED)
    sold = _ledger(sim, CommitmentKind.COMPLETED)
    assert {row.item_id for row in sold} == set(sim.items)
    assert len({row.bundle_key for row in accepted + sold}) == 1 and sold[0].bundle_key
    total = sum(row.price_cents for row in sold)
    assert report.sale_price_cents == total
    assert all(f"for {total} cents in total." in row.reason for row in accepted)
    goals = {item_id: spec.floor_price_cents for item_id, spec in sim.items.items()}
    assert all(row.price_cents >= goals[row.item_id] for row in sold)
    booked = _ledger(sim, CommitmentKind.PICKUP_BOOKED)
    assert len(booked) == 2 and len({(b.pickup_start, b.pickup_end) for b in booked}) == 1
    with Session(sim.engine) as session:
        listings = session.exec(select(Listing)).all()
        assert {listing.item_id for listing in listings} == set(sim.items)
        assert all(listing.status == ListingStatus.DELISTED for listing in listings)
        buyers = {session.get(Thread, row.thread_id).remote_id for row in sold}
    assert all(isinstance(sim.market.threads[r].persona, Bundler) for r in buyers)
    outbound = " ".join(body for _, body, _ in sim.market.sent)
    assert f"${total // 100}" in outbound


def _bundle_world(variant: str, seed: int = 0) -> Simulation:
    """The bundle campaign with no random buyers: one Bundler of the given variant."""
    scenario = replace(SCENARIOS["bundle"], arrivals=(), entrants=())
    sim = Simulation(scenario, seed=seed)
    chair = next(remote for item_id, remote in sim.listings.items() if item_id != sim.item_id)
    persona = Bundler(random.Random(seed), 11_000, variant=variant)  # type: ignore[arg-type]
    at = sim.clock.now() + timedelta(hours=1)
    sim.market.add_buyer(persona, sim.listing_remote_id, at, also=[chair])
    return sim


def test_two_thread_bundler_is_answered_in_one_thread_and_sold_both():
    sim = _bundle_world("two_threads", seed=3)
    try:
        report = sim.run(days=3)
        assert report.sold and report.violations == [] and report.errors == []
        assert {o["status"] for o in _outcomes(sim)} <= {"executed", "observed"}
        with Session(sim.engine) as session:
            threads = session.exec(select(Thread).order_by(col(Thread.id))).all()
            main, companion = threads
            assert main.buyer_id == companion.buyer_id
            assert (main.item_id, companion.item_id) == (sim.item_id, sim.item_id + 1)
            sold = _ledger(sim, CommitmentKind.COMPLETED)
            assert {(row.item_id, row.thread_id) for row in sold} == {
                (main.item_id, main.id),
                (companion.item_id, companion.id),  # each row on the buyer's own thread
            }
            told = session.exec(
                select(Message.body).where(
                    Message.thread_id == companion.id, Message.direction == Direction.OUTBOUND
                )
            ).all()
        assert len(told) == 1 and "in your message about the IKEA Micke Desk" in told[0]
        [action] = _accepted_bundles(sim)
        args = action["args"]
        assert args["thread_id"] == main.id
        assert args["item_ids"] == [main.item_id, companion.item_id]
    finally:
        sim.close()


def test_policy_offers_a_bundle_to_a_buyer_who_names_another_item():
    scenario = replace(SCENARIOS["bundle"], arrivals=(), entrants=())
    sim = Simulation(scenario, seed=0)
    try:
        remote = sim.market.add_buyer("scripted", sim.listing_remote_id)
        sim.market.buyer_say(
            remote, "How old is the desk? I'd want your IKEA Markus Office Chair, Black as well."
        )
        sim.tick()
        [(_, offer, _)] = [entry for entry in sim.market.sent if entry[0] == remote]
        assert offer.startswith("Hi ") and "It's about 3 years old." in offer
        assert "the IKEA Micke Desk and the IKEA Markus Office Chair together for $160" in offer
        sim.market.buyer_say(remote, "Deal, $160 for both works.")
        sim.tick()
        sold = [o for o in _outcomes(sim) if o["tool"] == "accept_bundle"]
        assert [o["status"] for o in sold] == ["executed"]
        assert sum(row.price_cents for row in _ledger(sim, CommitmentKind.ACCEPTED)) == 16_000
        assert {o["status"] for o in _outcomes(sim)} <= {"executed", "observed"}
        assert sim.audit() == []
    finally:
        sim.close()


@pytest.mark.parametrize("variant", ["one_thread", "two_threads"])
def test_bundle_oracle_shows_the_buyer_its_total(variant: str):
    sim = _bundle_world(variant, seed=1)
    try:
        while not _ledger(sim, CommitmentKind.ACCEPTED):
            sim.tick()
        total = sum(row.price_cents for row in _ledger(sim, CommitmentKind.ACCEPTED))
        oracle = sim.oracle()
        with Session(sim.engine) as session:
            threads = {t.remote_id for t in session.exec(select(Thread)).all()}
        assert set(oracle.prices) == threads and set(oracle.prices.values()) == {total}
    finally:
        sim.close()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bundle_campaign_stays_clean_on_other_seeds(seed: int):
    report = run_simulation(SCENARIOS["bundle"], seed=seed, days=4)
    assert report.violations == [] and report.errors == []
    assert report.sold


def test_every_campaign_item_gets_its_own_market_settings_even_after_a_relist():
    sim = Simulation(SCENARIOS["bundle"], seed=0)
    try:
        chair_id = next(item_id for item_id in sim.items if item_id != sim.item_id)
        chair = sim.listings[chair_id]
        assert sim.market.listings[chair].market_value_cents == 4_500
        assert sim.market.listings[sim.listing_remote_id].market_value_cents is None  # defaults
        relisted = sim.market.publish(sim.market.listings[chair].payload).remote_id
        with Session(sim.engine) as session:
            session.add(
                Listing(
                    item_id=chair_id,
                    platform="fake",
                    remote_id=relisted,
                    status=ListingStatus.LIVE,
                    price_cents=5_000,
                    title="IKEA Markus Office Chair, Black",
                    description="Relisted.",
                )
            )
            session.commit()
        assert not sim.sold()
        sim.tick()
        assert sim.market.listings[relisted].market_value_cents == 4_500
        arrivals = sim.market.listings[relisted].arrivals
        assert arrivals is not None and arrivals.personas == ()  # not the desk's buyers
        with Session(sim.engine) as session:
            for item in session.exec(select(Item)).all():
                item.status = ItemStatus.SOLD
                session.add(item)
            session.commit()
        assert sim.sold()
    finally:
        sim.close()


# ---------------------------------------------------------------- determinism and pacing


def test_same_seed_gives_the_same_transcript(reports):
    again = run_simulation(SCENARIOS["desk"], seed=0)
    assert again.transcript == reports["desk"].transcript
    assert again.model_requests == reports["desk"].model_requests
    assert run_simulation(SCENARIOS["desk"], seed=1, days=3).transcript != again.transcript


@pytest.mark.parametrize("tick_minutes", [7, 45])
def test_other_tick_lengths_stay_clean(tick_minutes: int):
    report = run_simulation(SCENARIOS["desk"], seed=2, tick_minutes=tick_minutes, days=4)
    assert report.violations == []
    assert report.errors == []


def test_offline_simulation_uses_the_configured_constrained_fraction():
    """Built like production's build_router: a limit set for the offline model degrades it."""
    settings = sim_settings(
        provider_limits={"heuristic": ProviderLimit(rpm=100, rpd=10)},
        quota_constrained_fraction=0.9,
    )
    sim = Simulation(SCENARIOS["desk"], settings=settings)
    try:
        assert sim.runtime.router.quota.constrained_fraction == 0.9
        sim.tick()  # the publish wake spends part of the 10 requests
        assert sim.runtime.router.quota.requests_today("heuristic") >= 2
        assert sim.runtime.router.state() == QuotaState.CONSTRAINED
    finally:
        sim.close()


# ---------------------------------------------------------------- CLI


cli = CliRunner()


@pytest.fixture
def no_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep a developer's .env out of the run
    for name in ("MA_GEMINI_API_KEY", "GEMINI_API_KEY", "MA_GROQ_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("MA_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_cli_simulate_ignores_the_local_toml_and_environment(no_keys, monkeypatch, tmp_path):
    """Config for real selling may leave the fake out entirely; the same --seed must replay."""
    caps = "[daily_publish_cap]\nkijiji = 10\n\n[daily_outbound_cap]\nkijiji = 60\n"
    (tmp_path / "marketplace_agent.toml").write_text('enabled_platforms = ["kijiji"]\n\n' + caps)
    monkeypatch.setitem(Settings.model_config, "toml_file", "marketplace_agent.toml")  # as shipped
    monkeypatch.setenv("MA_UNDO_WINDOW_MINUTES", "0")

    result = cli.invoke(app, ["simulate", "--scenario", "desk", "--days", "2", "--quiet"])

    assert result.exit_code == 0, result.output
    settings = sim_settings()
    assert settings.daily_publish_cap["fake"] and settings.daily_outbound_cap["fake"]
    assert settings.undo_window_minutes == Settings.model_fields["undo_window_minutes"].default


def test_cli_simulate_prints_transcript_and_report(no_keys):
    result = cli.invoke(app, ["simulate", "--scenario", "firm", "--seed", "0"])
    assert result.exit_code == 0, result.output
    assert "agent woke" in result.output
    assert "Sold:           yes, $80.00" in result.output
    assert "Violations:     none" in result.output


def test_cli_simulate_quiet_prints_only_the_report(no_keys):
    result = cli.invoke(app, ["simulate", "--scenario", "firm", "--quiet", "--days", "1"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Scenario:")
    assert "agent woke" not in result.output


def test_cli_simulate_rejects_an_unknown_scenario(no_keys):
    result = cli.invoke(app, ["simulate", "--scenario", "garage"])
    assert result.exit_code == 1
    assert "Choose one of: desk, lowballers" in result.output


def test_cli_simulate_live_needs_a_model_key(no_keys):
    result = cli.invoke(app, ["simulate", "--live", "--days", "1"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output


# ---------------------------------------------------------------- live pacing


class Flaky:
    """A provider that is rate limited `limited` times, then answers."""

    name = "gemini"
    capabilities = frozenset(Capability)

    def __init__(self, limited: int, *, daily: bool = False, retry_after: float | None = 7.0):
        self.limited, self.daily, self.retry_after = limited, daily, retry_after
        self.calls = 0

    def model_for(self, capability: Capability) -> str:
        return "flash"

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.limited:
            exc = RateLimited("429")
            exc.retry_after, exc.daily = self.retry_after, self.daily  # type: ignore[attr-defined]
            raise exc
        return LLMResponse(text="{}", provider=self.name, model="flash", data={})


class RealTime:
    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 3))
        self.now += seconds


def _paced(provider: Flaky, rpm: int = 10) -> tuple[RealTimePacer, RealTime]:
    real = RealTime()
    return RealTimePacer(provider, rpm, sleep=real.sleep, monotonic=lambda: real.now), real


def test_live_calls_are_spaced_by_the_providers_per_minute_limit():
    pacer, real = _paced(Flaky(0), rpm=10)
    for _ in range(3):
        pacer.generate(LLMRequest(system="", prompt="hi"))
    assert real.slept == [6.0, 6.0]


def test_live_call_waits_out_a_per_minute_rate_limit():
    provider = Flaky(2)
    pacer, real = _paced(provider)
    assert pacer.generate(LLMRequest(system="", prompt="hi")).provider == "gemini"
    assert provider.calls == 3 and real.slept == [7.0, 7.0]


@pytest.mark.parametrize("flaky", [Flaky(1, daily=True), Flaky(9)], ids=["daily", "persistent"])
def test_live_call_gives_up_on_a_daily_or_persistent_limit(flaky: Flaky):
    pacer, _ = _paced(flaky)
    with pytest.raises(RateLimited):
        pacer.generate(LLMRequest(system="", prompt="hi"))


def test_live_simulation_paces_the_real_router(no_keys):
    settings = sim_settings(gemini_api_key="not-a-real-key")
    sim = Simulation(SCENARIOS["firm"], settings=settings, live=True)
    try:
        [provider] = sim.runtime.router.providers
        assert isinstance(provider, RealTimePacer) and provider.name == "gemini"
        assert provider.interval == 60 / settings.provider_limits["gemini"].rpm
    finally:
        sim.close()
