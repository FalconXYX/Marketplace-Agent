"""One pass of the runner: ingest, cheap replies, wakes, housekeeping.

| Step | What | Model requests |
|---|---|---|
| 1 Ingest | Threads and stats per platform; a failing platform is skipped | 0 |
| 2 Pattern cache | First-contact "is this available?" answered from a template | 0 |
| 3 Pickup checks | Ask the owner whether the buyer came, once a pickup ends | 0 |
| 4 Wakes | Items with triggers, commitments first, filtered by quota | 1 to 3 each |
| 5 Courtesy replies | One fixed reply per thread about a sold or withdrawn item | 0 |
| 6 Housekeeping | Owed takedowns, expire approvals, ghosts, cap alerts, view baselines, comps | 0 |

Courtesy replies come after the wakes, so live deals get the outbound cap first. Takedowns and
courtesy replies: runner/takedown.py.

| Quota | Wakes |
|---|---|
| healthy | every item with a trigger |
| constrained, or only the cheap model has quota left | items with a PRIORITY_KINDS trigger |
| throttled: a per-minute limit or a short 429 cooldown | none this tick; the next tick retries |
| exhausted | none; one open escalation per item with buyers waiting |

A quota escalation resolves itself once a wake of that item gets a plan.

A wake that gets no plan (provider error) leaves its triggers armed, so the item backs off before
the next try: FAILED_WAKE_BACKOFF doubling per failure, up to MAX_FAILED_WAKE_BACKOFF.
"""

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import Engine
from sqlmodel import Session, col, func, select

from marketplace_agent.agent.context import Trigger
from marketplace_agent.agent.loop import Agent
from marketplace_agent.ai import pattern_cache
from marketplace_agent.ai.base import Capability, QuotaState
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.ai.templates import availability_reply
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.db import session_scope
from marketplace_agent.core.models import (
    AgentRun,
    ApprovalStatus,
    Buyer,
    Direction,
    Escalation,
    Item,
    Message,
    PendingApproval,
    Thread,
    ThreadState,
    WakeRequest,
)
from marketplace_agent.guard import handover
from marketplace_agent.guard.invariants import outbound_count_today
from marketplace_agent.guard.ledger import active_commitment, active_commitment_for_thread
from marketplace_agent.guard.validate import validate
from marketplace_agent.notify.base import Notice, NoticeKind, Notifier
from marketplace_agent.platforms.base import PlatformAdapter, PlatformAuthError, PlatformError
from marketplace_agent.runner import takedown
from marketplace_agent.runner.ingest import ingest, snapshot_stats
from marketplace_agent.runner.triggers import (
    PRIORITY_KINDS,
    WAKEABLE,
    compute_triggers,
    total_views,
)
from marketplace_agent.tools.base import Allow, ToolContext
from marketplace_agent.tools.registry import ToolRegistry

FETCH_OVERLAP = timedelta(hours=1)  # refetch margin for late or clock-skewed messages
FAILED_WAKE_BACKOFF = timedelta(minutes=5)
MAX_FAILED_WAKE_BACKOFF = timedelta(hours=6)
GHOST_AFTER = timedelta(hours=72)
JITTER = 0.2
QUOTA_ESCALATION = (
    "Model quota is used up, so buyers are waiting with no reply. Answer them yourself, or the "
    "agent will catch up when the quota resets."
)
NO_MODEL_ESCALATION = (
    "No model is configured (set GEMINI_API_KEY), so buyers are waiting with no reply. Answer "
    "them yourself; the agent answers once a model key is set."
)
_WAITING_REASONS = (QUOTA_ESCALATION, NO_MODEL_ESCALATION)
THROTTLED = "throttled"  # TickReport.quota_state when wakes wait for a per-minute limit
_KIND_ORDER = (
    "pickup_elapsed",
    "human_override",
    "approval_decided",
    "new_message",
    "scheduled",
    "deadline",
    "stats_moved",
    "idle",
)

logger = logging.getLogger("marketplace_agent.runner")


@dataclass
class TickReport:
    started_at: datetime
    ingested: int = 0
    cache_replies: int = 0
    wakes: int = 0
    courtesy_replies: int = 0
    takedowns: int = 0  # owed listings taken down this tick
    escalated: int = 0
    quota_state: str = QuotaState.HEALTHY.value
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.ingested} new message(s), {self.cache_replies} cache reply(ies), "
            f"{self.wakes} wake(s), {self.courtesy_replies} courtesy reply(ies), "
            f"{self.takedowns} takedown(s), {self.escalated} escalated, quota {self.quota_state}, "
            f"{len(self.errors)} error(s)"
        )


class Runner:
    def __init__(
        self,
        engine: Engine,
        clock: Clock,
        settings: Settings,
        platforms: Mapping[str, PlatformAdapter],
        agent: Agent,
        router: ProviderRouter,
        notifier: Notifier,
        registry: ToolRegistry,
    ):
        self.engine = engine
        self.clock = clock
        self.settings = settings
        self.platforms = platforms
        self.agent = agent
        self.router = router
        self.notifier = notifier
        self.registry = registry
        self._since: dict[str, datetime] = {}
        self._alerted: dict[tuple[str, str], date] = {}
        self._rng = random.Random()

    def tick(self) -> TickReport:
        report = TickReport(started_at=self.clock.now())
        self._ingest_all(report)
        self._answer_from_cache(report)
        self._ask_about_pickups(report)
        self._wake_all(report)
        self._courtesy_replies(report)
        self._housekeeping(report)
        return report

    def run_forever(
        self,
        stop: threading.Event | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Tick until stop is set, sleeping tick_seconds +-20% between ticks. A failing tick is
        logged and the loop carries on."""
        stop = stop or threading.Event()
        while not stop.is_set():
            try:
                report = self.tick()
            except Exception:
                logger.exception("tick failed")
            else:
                logger.info("tick: %s", report.summary())
                for error in report.errors:
                    logger.warning("tick error: %s", error)
            if stop.is_set():
                return
            sleep(self.settings.tick_seconds * self._rng.uniform(1 - JITTER, 1 + JITTER))

    # ------------------------------------------------------------ 1 ingest

    def _ingest_all(self, report: TickReport) -> None:
        for name, adapter in self.platforms.items():
            try:
                with session_scope(self.engine) as session:
                    result = ingest(
                        session,
                        self.clock,
                        self.settings,
                        self.notifier,
                        adapter,
                        self._since.get(name),
                    )
                self._since[name] = report.started_at - FETCH_OVERLAP
                report.ingested += len(result.new_inbound)
                report.escalated += len(result.frozen_threads)
                report.errors.extend(result.errors)
                with session_scope(self.engine) as session:
                    snapshot_stats(session, self.clock, adapter, self.notifier)
            except PlatformAuthError as exc:  # signed out, or a profile in use: the text says
                report.errors.append(f"{name}: needs the owner: {exc}")
                self._alert_once(
                    "auth",
                    name,
                    f"{name} needs you",
                    f"{exc} Until then nothing is read or sent on {name}; the agent tries "
                    "again every tick.",
                )
            except PlatformError as exc:
                report.errors.append(f"{name}: {exc}")
            except Exception as exc:  # a bug on one platform must not stop the tick
                logger.exception("ingest from %s failed", name)
                report.errors.append(f"{name}: ingest failed: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ 2 pattern cache

    def _answer_from_cache(self, report: TickReport) -> None:
        with Session(self.engine, expire_on_commit=False) as session:
            candidates = _cache_candidates(session)
            for message_id in candidates:
                try:
                    if self._cache_reply(session, message_id):
                        report.cache_replies += 1
                    session.commit()
                except PlatformError as exc:  # left unhandled, so the agent answers it instead
                    session.rollback()
                    report.errors.append(f"cache reply to message {message_id}: {exc}")
                except Exception as exc:  # a bug in one reply must not stop the tick
                    session.rollback()
                    logger.exception("cache reply to message %s failed", message_id)
                    report.errors.append(f"cache reply to message {message_id}: {exc!r}")

    def _cache_reply(self, session: Session, message_id: int) -> bool:
        """Answer one first-contact availability question through the guard, as "system". The
        pattern's hit is counted after the send: a write before it would hold SQLite's write
        lock, and so block the owner's undo, for as long as the platform call takes."""
        message = session.get(Message, message_id)
        assert message is not None
        thread = session.get(Thread, message.thread_id)
        assert thread is not None and thread.item_id is not None
        if active_commitment_for_thread(session, message.thread_id) is not None:
            return False
        entry = pattern_cache.find(session, message.body, first_contact=True)
        if entry is None:
            return False
        replied = entry.intent == "availability" and self._availability_reply(
            session, message, thread
        )
        pattern_cache.count_hit(session, entry)
        return replied

    def _availability_reply(self, session: Session, message: Message, thread: Thread) -> bool:
        assert thread.item_id is not None
        active = active_commitment(session, thread.item_id)
        if active is not None and active.accepted.buyer_id == thread.buyer_id:
            return False  # promised to this buyer in another thread (a bundle): the agent answers
        body = availability_reply(
            committed=active is not None, pickup_area=self.settings.pickup_area
        )
        tool = self.registry.get("send_message")
        assert tool is not None
        args = tool.parse_args({"thread_id": thread.id, "body": body})
        ctx = ToolContext(
            session=session,
            clock=self.clock,
            settings=self.settings,
            platforms=self.platforms,
            notifier=self.notifier,
            item_id=thread.item_id,
            actor="system",
        )
        verdict = validate(ctx, tool, args)
        if not isinstance(verdict, Allow):
            logger.info("cache reply to message %s not sent: %s", message.id, verdict)
            return False
        result = tool.execute(ctx, args)
        sent = session.get(Message, result.data["message_id"])
        assert sent is not None
        sent.via = "pattern_cache"
        message.handled = True
        message.intent = "availability"
        session.add_all([sent, message])
        return True

    # ------------------------------------------------------------ 3 pickup checks

    def _ask_about_pickups(self, report: TickReport) -> None:
        """Before the wakes, so a pickup_elapsed briefing already shows the owner was asked."""
        with session_scope(self.engine) as session:
            report.escalated += handover.ask_owner(
                session, self.clock, self.settings, self.notifier
            )

    # ------------------------------------------------------------ 4 wakes

    def _wake_all(self, report: TickReport) -> None:
        state = self._quota_state()
        report.quota_state = state.value
        with Session(self.engine, expire_on_commit=False) as session:
            found = compute_triggers(session, self.clock, self.settings)
            if state == QuotaState.EXHAUSTED:
                if self._throttled():
                    report.quota_state = THROTTLED
                else:
                    report.escalated += self._escalate_waiting(session, found)
                return
            if state == QuotaState.CONSTRAINED:
                found = {
                    i: ts for i, ts in found.items() if any(t.kind in PRIORITY_KINDS for t in ts)
                }
            order = _wake_order(session, found)
        for item_id in order:
            with Session(self.engine, expire_on_commit=False) as session:
                if not _still_wakeable(session, item_id) or self._backing_off(session, item_id):
                    continue
                if self._wake(session, item_id, found[item_id], report):
                    report.wakes += 1

    def _throttled(self) -> bool:
        """Held back for a minute or two, not out of quota: the next tick retries."""
        return self.router.throttled(Capability.REASONING) or self.router.throttled(
            Capability.CHEAP
        )

    def _quota_state(self) -> QuotaState:
        """The agent falls back to the cheap model when reasoning is spent, so only both spent
        means no wakes; reasoning spent alone wakes like constrained."""
        state = self.router.state(Capability.REASONING)
        if state == QuotaState.EXHAUSTED and self.router.state(Capability.CHEAP) != state:
            return QuotaState.CONSTRAINED
        return state

    def _wake(
        self, session: Session, item_id: int, triggers: list[Trigger], report: TickReport
    ) -> bool:
        """Wake one item and record what it acted on. False when no run was started."""
        started = self.clock.now()
        try:
            result = self.agent.wake(session, item_id, triggers)
        except Exception as exc:  # one broken item must not stop the others
            session.rollback()
            logger.exception("wake of item %s failed", item_id)
            report.errors.append(f"item {item_id}: wake failed: {type(exc).__name__}: {exc}")
            return False
        if result.run_id is None:
            if not self._throttled():
                report.escalated += self._escalate_waiting(session, {item_id: triggers})
            return False
        if result.error:
            report.errors.append(f"item {item_id}: {result.error}")
        item = session.get(Item, item_id, populate_existing=True)
        assert item is not None
        if item.last_agent_run_at is not None and item.last_agent_run_at >= started:
            self._mark_seen(session, item, started)
            self._resolve_waiting(session, item_id)
        session.commit()
        return True

    def _mark_seen(self, session: Session, item: Item, started: datetime) -> None:
        """After a wake that got a plan: the agent has seen these views and wake requests."""
        assert item.id is not None
        item.last_wake_views = total_views(session, item.id)
        session.add(item)
        query = select(WakeRequest).where(
            WakeRequest.item_id == item.id,
            col(WakeRequest.consumed_at).is_(None),
            col(WakeRequest.created_at) <= started,
        )
        for request in session.exec(query).all():
            request.consumed_at = self.clock.now()
            session.add(request)

    def _backing_off(self, session: Session, item_id: int) -> bool:
        item = session.get(Item, item_id)
        assert item is not None
        query = select(AgentRun.started_at).where(AgentRun.item_id == item_id)
        if item.last_agent_run_at is not None:
            query = query.where(col(AgentRun.started_at) > item.last_agent_run_at)
        failed = session.exec(query.order_by(col(AgentRun.started_at).desc()).limit(10)).all()
        if not failed:
            return False
        wait = min(FAILED_WAKE_BACKOFF * 2 ** (len(failed) - 1), MAX_FAILED_WAKE_BACKOFF)
        return self.clock.now() < failed[0] + wait

    def _escalate_waiting(self, session: Session, found: Mapping[int, list[Trigger]]) -> int:
        """One open quota escalation per item with buyers waiting. Returns how many were new."""
        reason = QUOTA_ESCALATION if self.router.providers else NO_MODEL_ESCALATION
        raised = 0
        for item_id, triggers in found.items():
            waiting = sum(len(t.message_ids) for t in triggers if t.kind == "new_message")
            if not waiting or _open_waiting_escalations(session, item_id):
                continue
            escalation = Escalation(item_id=item_id, reason=reason, created_at=self.clock.now())
            session.add(escalation)
            session.flush()
            self.notifier.notify(
                Notice(
                    kind=NoticeKind.ESCALATION,
                    title="Buyers are waiting",
                    body=f"{waiting} message(s) on item {item_id}. {reason}",
                    item_id=item_id,
                    ref_id=escalation.id,
                )
            )
            raised += 1
        session.commit()
        return raised

    def _resolve_waiting(self, session: Session, item_id: int) -> None:
        """The agent has answered: an open quota escalation for the item no longer holds."""
        for escalation in _open_waiting_escalations(session, item_id):
            escalation.resolved_at = self.clock.now()
            escalation.resolution = "The agent answered once a model was available again."
            session.add(escalation)

    # ------------------------------------------------------------ 5 courtesy replies

    def _courtesy_replies(self, report: TickReport) -> None:
        with Session(self.engine, expire_on_commit=False) as session:
            sent, errors = takedown.send_courtesy_replies(
                session,
                self.clock,
                self.settings,
                self.platforms,
                self.notifier,
                self.registry.get("send_message"),
            )
        report.courtesy_replies += sent
        report.errors.extend(errors)

    # ------------------------------------------------------------ 6 housekeeping

    def _housekeeping(self, report: TickReport) -> None:
        self._retry_takedowns(report)
        with session_scope(self.engine) as session:
            self._expire_approvals(session)
            self._close_ghosted(session)
            self._rebase_views(session)
            self._alert_caps(session)
        self._refresh_comps(report)

    def _retry_takedowns(self, report: TickReport) -> None:
        """Isolated: a failing takedown pass must not stop the rest of housekeeping."""
        try:
            with Session(self.engine, expire_on_commit=False) as session:
                done = takedown.retry_takedowns(
                    session,
                    self.clock,
                    self.settings,
                    self.platforms,
                    self.notifier,
                    tick=timedelta(seconds=self.settings.tick_seconds * (1 - JITTER)),
                )
        except Exception as exc:
            logger.exception("takedown pass failed")
            report.errors.append(f"takedowns: {type(exc).__name__}: {exc}")
            return
        report.takedowns += done.taken_down
        report.errors.extend(done.errors)

    def _refresh_comps(self, report: TickReport) -> None:
        """Stale comps for listed items, at most daily per query (core.comps owns the cache).
        Last, since it calls every platform's search. Skipped when core.comps is absent."""
        try:
            from marketplace_agent.core.comps import refresh_due
        except ImportError:
            return
        try:
            with session_scope(self.engine) as session:
                refresh_due(session, self.clock, self.platforms)
        except Exception as exc:
            logger.exception("comps refresh failed")
            report.errors.append(f"comps refresh: {type(exc).__name__}: {exc}")

    def _expire_approvals(self, session: Session) -> None:
        now = self.clock.now()
        cutoff = now - timedelta(hours=self.settings.approval_ttl_hours)
        query = select(PendingApproval).where(
            PendingApproval.status == ApprovalStatus.PENDING,
            col(PendingApproval.created_at) <= cutoff,
        )
        for approval in session.exec(query).all():
            approval.status = ApprovalStatus.EXPIRED
            approval.decided_at = now
            session.add(approval)

    def _close_ghosted(self, session: Session) -> None:
        cutoff = self.clock.now() - GHOST_AFTER
        query = select(Thread).where(
            Thread.state == ThreadState.NEGOTIATING,
            col(Thread.last_outbound_at) <= cutoff,
        )
        for thread in session.exec(query).all():
            assert thread.last_outbound_at is not None
            if (
                thread.last_inbound_at is not None
                and thread.last_inbound_at >= thread.last_outbound_at
            ):
                continue
            buyer = session.get(Buyer, thread.buyer_id)
            assert buyer is not None
            buyer.ghost_count += 1
            thread.state = ThreadState.CLOSED
            session.add_all([buyer, thread])

    def _rebase_views(self, session: Session) -> None:
        """A relist restarts a listing's view count; lower the baseline so stats_moved can fire
        again on the new count."""
        query = select(Item).where(col(Item.status).in_(WAKEABLE), Item.last_wake_views > 0)
        for item in session.exec(query).all():
            assert item.id is not None
            views = total_views(session, item.id)
            if views < item.last_wake_views:
                item.last_wake_views = views
                session.add(item)

    def _alert_caps(self, session: Session) -> None:
        for name in self.platforms:
            cap = self.settings.daily_outbound_cap.get(name, 0)
            sent = outbound_count_today(session, self.clock, name)
            if sent >= cap:
                self._alert_once(
                    "outbound_cap",
                    name,
                    f"Daily message cap reached on {name}",
                    f"{sent} of {cap} messages sent today. The agent sends nothing more on {name} "
                    "until midnight UTC; buyers who write will wait.",
                )

    def _alert_once(self, kind: str, platform: str, title: str, body: str) -> None:
        """At most one alert per kind and platform per UTC day."""
        today = self.clock.now().astimezone(UTC).date()
        if self._alerted.get((kind, platform)) == today:
            return
        self._alerted[kind, platform] = today
        self.notifier.notify(Notice(kind=NoticeKind.ALERT, title=title, body=body))


def _cache_candidates(session: Session) -> list[int]:
    """Unhandled inbound messages that are alone in their thread, on a first contact, in a thread
    that is not frozen, for a wakeable item that is not paused."""
    unhandled = (
        select(Message.thread_id)
        .where(Message.direction == Direction.INBOUND, col(Message.handled).is_(False))
        .group_by(col(Message.thread_id))
        .having(func.count(col(Message.id)) == 1)
    )
    replied = select(Message.thread_id).where(Message.direction == Direction.OUTBOUND)
    query = (
        select(Message.id)
        .join(Thread, col(Thread.id) == Message.thread_id)
        .join(Item, col(Item.id) == Thread.item_id)
        .where(
            Message.direction == Direction.INBOUND,
            col(Message.handled).is_(False),
            col(Message.thread_id).in_(unhandled),
            col(Message.thread_id).not_in(replied),
            Thread.state != ThreadState.FROZEN,
            col(Item.status).in_(WAKEABLE),
            col(Item.paused).is_(False),
        )
        .order_by(col(Message.sent_at), col(Message.id))
    )
    return [message_id for message_id in session.exec(query).all() if message_id is not None]


def _still_wakeable(session: Session, item_id: int) -> bool:
    """found was computed before any wake ran: an earlier wake may have closed this item since
    (a bundle partner's complete_sale), or the owner paused or sold it. Its buyers then get the
    courtesy reply, not a wake that cannot write to them."""
    item = session.get(Item, item_id, populate_existing=True)
    return item is not None and item.status in WAKEABLE and not item.paused


def _wake_order(session: Session, found: Mapping[int, list[Trigger]]) -> list[int]:
    """Items with an active commitment first, then by their most urgent trigger kind."""

    def key(item_id: int) -> tuple[bool, int, int]:
        urgency = min(_kind_rank(t.kind) for t in found[item_id])
        return active_commitment(session, item_id) is None, urgency, item_id

    return sorted(found, key=key)


def _kind_rank(kind: str) -> int:
    return _KIND_ORDER.index(kind) if kind in _KIND_ORDER else len(_KIND_ORDER)


def _open_waiting_escalations(session: Session, item_id: int) -> list[Escalation]:
    query = select(Escalation).where(
        Escalation.item_id == item_id,
        col(Escalation.reason).in_(_WAITING_REASONS),
        col(Escalation.resolved_at).is_(None),
    )
    return list(session.exec(query).all())
