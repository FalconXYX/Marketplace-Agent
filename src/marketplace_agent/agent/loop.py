"""One wake: build the context, ask the model for a plan, run each action through the guard.

| Quota (REASONING) | Wake runs on |
|---|---|
| HEALTHY | REASONING, max_steps |
| CONSTRAINED | REASONING, 2 steps |
| EXHAUSTED, CHEAP not | CHEAP, 1 step |
| Both EXHAUSTED | No call; error "quota exhausted" |

SQLite has one writer and QuotaTracker.record commits on its own connection, so the session is
committed before every model call. Messages are marked handled only after a call succeeds, and
committed at once so a later tool failure cannot roll them back.

A message often announces what the actions before it did ("Deal, it's yours, see you Saturday
at 5"). So once an action that changes the deal, price, or listing is rejected, fails, or waits
for approval, the send_message actions after it in that plan are skipped, not sent; the next
step shows the model why, and it rewrites them.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError
from sqlmodel import Session, select

from marketplace_agent.agent.context import AgentContext, Trigger, build_context
from marketplace_agent.agent.plan import INTENTS, PLAN_WIRE_SCHEMA, Plan, PlannedAction
from marketplace_agent.ai import pattern_cache
from marketplace_agent.ai.base import Capability, LLMRequest, ProviderError, QuotaState
from marketplace_agent.ai.prompting import load_prompt
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.items import get_item
from marketplace_agent.core.models import AgentRun, ApprovalStatus, Message, PendingApproval
from marketplace_agent.guard.invariants import redact_address
from marketplace_agent.guard.validate import after_execute, validate
from marketplace_agent.notify.base import Notice, NoticeKind, Notifier
from marketplace_agent.platforms.base import PlatformAdapter
from marketplace_agent.tools.base import (
    NeedsApproval,
    Reject,
    Tool,
    ToolContext,
    ToolResult,
)

MAX_OUTPUT_TOKENS = 8192  # Gemini counts thinking tokens as output; 2048 truncates plans
CONSTRAINED_STEPS = 2
DETAIL_CHARS = 2000
FOLLOW_UP_STATUSES = frozenset({"rejected", "failed", "observed", "skipped"})
NOT_DONE = frozenset({"rejected", "failed", "needs_approval"})
DEPENDENT_TOOLS = frozenset({"send_message"})
INDEPENDENT_TOOLS = frozenset({"send_message", "remember", "escalate", "schedule_wake"})

logger = logging.getLogger("marketplace_agent.agent")


class ToolLookup(Protocol):
    """What the loop needs from tools.registry.ToolRegistry."""

    def get(self, name: str) -> Tool | None: ...

    def names(self) -> list[str]: ...

    def specs(self) -> list[dict[str, Any]]: ...


@dataclass
class ActionOutcome:
    step: int
    tool: str
    args: dict[str, Any]
    status: str  # executed | rejected | needs_approval | failed | observed | skipped
    detail: str


@dataclass
class WakeResult:
    run_id: int | None
    rationale: str
    outcomes: list[ActionOutcome]
    requests_used: int
    quota_state: QuotaState
    error: str | None = None


class Agent:
    def __init__(
        self,
        router: ProviderRouter,
        registry: ToolLookup,
        settings: Settings,
        clock: Clock,
        notifier: Notifier,
        platforms: Mapping[str, PlatformAdapter],
        *,
        max_steps: int = 3,
    ):
        self.router = router
        self.registry = registry
        self.settings = settings
        self.clock = clock
        self.notifier = notifier
        self.platforms = platforms
        self.max_steps = max_steps

    def wake(self, session: Session, item_id: int, triggers: Sequence[Trigger]) -> WakeResult:
        get_item(session, item_id)
        quota_state = self.router.state(Capability.REASONING)
        budget = self._budget(quota_state)
        if budget is None:
            return WakeResult(None, "", [], 0, quota_state, error="quota exhausted")
        capability, steps = budget
        run = AgentRun(
            item_id=item_id,
            started_at=self.clock.now(),
            triggers_json=json.dumps([asdict(t) for t in triggers]),
            quota_state=quota_state.value,
        )
        session.add(run)
        session.flush()
        wake = _Wake(self, session, run, capability, quota_state)
        wake.run_steps(triggers, steps)
        return wake.finish()

    def _budget(self, state: QuotaState) -> tuple[Capability, int] | None:
        if state == QuotaState.HEALTHY:
            return Capability.REASONING, self.max_steps
        if state == QuotaState.CONSTRAINED:
            return Capability.REASONING, min(CONSTRAINED_STEPS, self.max_steps)
        if self.router.state(Capability.CHEAP) != QuotaState.EXHAUSTED:
            return Capability.CHEAP, 1
        return None


class _Wake:
    """State of one wake across its steps."""

    def __init__(
        self,
        agent: Agent,
        session: Session,
        run: AgentRun,
        capability: Capability,
        quota_state: QuotaState,
    ):
        self.agent = agent
        self.session = session
        self.run = run
        self.run_id = run.id
        self.item_id = run.item_id
        self.capability = capability
        self.quota_state = quota_state
        self.plans: list[dict[str, Any]] = []
        self.outcomes: list[ActionOutcome] = []
        self.rationale = ""
        self.provider: str | None = None
        self.requests = 0
        self.error: str | None = None

    def run_steps(self, triggers: Sequence[Trigger], steps: int) -> None:
        for step in range(1, steps + 1):
            context = build_context(
                self.session,
                self.agent.settings,
                self.agent.clock,
                self.item_id,
                triggers,
                quota_state=self.quota_state,
                tool_specs=self.agent.registry.specs(),
            )
            plan = self._ask(context, step, steps)
            if plan is None:
                return
            if step == 1:
                self._absorb_messages(context, plan)
            done = self._act_all(step, plan)
            if not self._wants_follow_up(done, step, steps):
                return

    def finish(self) -> WakeResult:
        settings = self.agent.settings
        run = self.run
        run.rationale = self.rationale
        run.plan_json = redact_address(json.dumps(self.plans), settings)
        run.results_json = redact_address(json.dumps([asdict(o) for o in self.outcomes]), settings)
        run.provider = self.provider
        run.requests_used = self.requests
        run.error = self.error
        run.finished_at = self.agent.clock.now()
        self.session.add(run)
        if self.plans:  # a wake with no plan leaves time-based triggers armed to fire again
            item = get_item(self.session, self.item_id)
            item.last_agent_run_at = run.finished_at
            self.session.add(item)
        self.session.commit()
        return WakeResult(
            run_id=self.run_id,
            rationale=self.rationale,
            outcomes=self.outcomes,
            requests_used=self.requests,
            quota_state=self.quota_state,
            error=self.error,
        )

    def _ask(self, context: AgentContext, step: int, steps: int) -> Plan | None:
        prompt = context.text
        if step > 1:
            prompt += "\n\n" + self._results_section(step, steps)
        request = LLMRequest(
            system=load_prompt("agent_system"),
            prompt=prompt,
            capability=self.capability,
            json_schema=PLAN_WIRE_SCHEMA,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            purpose="wake",
        )
        self.session.commit()
        try:
            response = self.agent.router.generate(request)
        except ProviderError as exc:
            self.error = f"step {step}: {type(exc).__name__}: {exc}"
            return None
        self.requests += 1
        self.provider = response.provider
        try:
            plan = Plan.from_wire(response.data or {})
        except (AttributeError, TypeError, ValueError) as exc:
            self.error = f"step {step}: unusable plan from {response.provider}: {exc}"
            return None
        self.rationale = plan.rationale
        self.plans.append({"step": step, **plan.model_dump()})
        return plan

    def _absorb_messages(self, context: AgentContext, plan: Plan) -> None:
        """Mark the step's NEW messages handled, store their labels, and teach the pattern cache
        availability phrasings. Committed at once: a later tool rollback must not undo it."""
        labels = {
            label.message_id: label.intent
            for label in plan.message_labels
            if label.intent in INTENTS
        }
        for message_id in context.unhandled_message_ids:
            message = self.session.get(Message, message_id)
            if message is None:
                continue
            message.handled = True
            intent = labels.get(message_id)
            if intent:
                message.intent = intent
                if intent == "availability":
                    pattern_cache.learn(self.session, self.agent.clock, message.body, intent)
            self.session.add(message)
        self.session.commit()

    def _act_all(self, step: int, plan: Plan) -> list[ActionOutcome]:
        """Run the plan in order; messages after an action that did not happen are skipped."""
        done: list[ActionOutcome] = []
        blocker: ActionOutcome | None = None
        for action in plan.actions:
            if blocker is not None and action.tool in DEPENDENT_TOOLS:
                done.append(self._skip(step, action, blocker))
                continue
            outcome = self._act(step, action, plan.rationale)
            done.append(outcome)
            if outcome.status in NOT_DONE and action.tool not in INDEPENDENT_TOOLS:
                blocker = blocker or outcome
        return done

    def _skip(self, step: int, action: PlannedAction, blocker: ActionOutcome) -> ActionOutcome:
        outcome = ActionOutcome(
            step,
            action.tool,
            action.args,
            "skipped",
            f"Not sent: it came after {blocker.tool}, which was {blocker.status}, and may say "
            "that it happened. Fix or drop that first, then send what is still true.",
        )
        self.outcomes.append(outcome)
        return outcome

    def _act(self, step: int, action: PlannedAction, rationale: str) -> ActionOutcome:
        status, detail = self._outcome(action, rationale)
        detail = redact_address(detail, self.agent.settings)
        outcome = ActionOutcome(step, action.tool, action.args, status, detail)
        self.outcomes.append(outcome)
        return outcome

    def _outcome(self, action: PlannedAction, rationale: str) -> tuple[str, str]:
        tool = self.agent.registry.get(action.tool)
        if tool is None:
            known = ", ".join(self.agent.registry.names())
            return "rejected", f"There is no tool named {action.tool!r}. Use one of: {known}."
        if action.parse_error:
            return "rejected", (
                f"{action.parse_error}. Send args_json as a JSON object string, for example "
                '"{\\"thread_id\\": 4}", matching the tool\'s schema.'
            )
        try:
            args = tool.parse_args(action.args)
        except ValidationError as exc:
            return "rejected", _invalid_args(tool, exc)
        try:
            result = self._guarded(tool, args, action.why or rationale)
        except Exception as exc:  # a tool bug must not end the wake
            self.session.rollback()
            logger.warning("tool %s raised %s", tool.name, type(exc).__name__)
            return "failed", f"{tool.name} raised {type(exc).__name__}: {exc}"
        self.session.commit()
        return result

    def _guarded(self, tool: Tool, args: BaseModel, why: str) -> tuple[str, str]:
        ctx = self._tool_context()
        verdict = validate(ctx, tool, args)
        if isinstance(verdict, Reject):
            return "rejected", verdict.reason
        if isinstance(verdict, NeedsApproval):
            return "needs_approval", self._park(ctx, tool, args, why, verdict.reason)
        result = tool.execute(ctx, args)
        after_execute(ctx, tool, args, result)
        if not result.ok:
            return "failed", result.summary
        if tool.read_only:
            return "observed", _observation(result)
        return "executed", result.summary

    def _park(self, ctx: ToolContext, tool: Tool, args: BaseModel, why: str, reason: str) -> str:
        """A PendingApproval for the owner, unless the same call is already waiting."""
        args_json = json.dumps(args.model_dump(mode="json"), sort_keys=True)
        waiting = self.session.exec(
            select(PendingApproval).where(
                PendingApproval.item_id == self.item_id,
                PendingApproval.tool == tool.name,
                PendingApproval.args_json == args_json,
                PendingApproval.status == ApprovalStatus.PENDING,
            )
        ).first()
        if waiting is not None:
            return (
                f"The same {tool.name} call is already waiting for the owner (approval "
                f"#{waiting.id}). Do not ask again; carry on with other work."
            )
        approval = PendingApproval(
            item_id=self.item_id,
            platform=tool.trust_platform(ctx, args),
            tool=tool.name,
            args_json=args_json,
            rationale=why,
            agent_run_id=self.run_id,
            created_at=self.agent.clock.now(),
        )
        self.session.add(approval)
        self.session.flush()
        self.agent.notifier.notify(
            Notice(
                kind=NoticeKind.APPROVAL,
                title=f"Approve {tool.name}?",
                body=f"{args_json}\n{why}".strip(),
                item_id=self.item_id,
                ref_id=approval.id,
            )
        )
        return reason

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            session=self.session,
            clock=self.agent.clock,
            settings=self.agent.settings,
            platforms=self.agent.platforms,
            notifier=self.agent.notifier,
            item_id=self.item_id,
            agent_run_id=self.run_id,
            actor="agent",
        )

    def _wants_follow_up(self, done: list[ActionOutcome], step: int, steps: int) -> bool:
        if step >= steps or not any(o.status in FOLLOW_UP_STATUSES for o in done):
            return False
        return self.agent.router.state(self.capability) != QuotaState.EXHAUSTED

    def _results_section(self, step: int, steps: int) -> str:
        lines = [
            "## RESULTS OF YOUR LAST PLAN",
            f"This is step {step} of at most {steps} in this wake. Your rationale was: "
            f"{self.rationale or '(none)'}",
        ]
        for o in self.outcomes:
            args = json.dumps(o.args, separators=(",", ":"))
            lines.append(f"- step {o.step} {o.tool} {args} -> {o.status}: {o.detail}")
        lines.append(
            "Executed actions are done; do not repeat them. Fix or drop what was rejected or "
            "failed, use what you observed, and return a new plan. Return no actions if nothing "
            "more is needed. Labels for messages are no longer needed."
        )
        return redact_address("\n".join(lines), self.agent.settings)


def _invalid_args(tool: Tool, exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'args'}: {error['msg']}"
        for error in exc.errors()
    )
    return f"Invalid arguments for {tool.name}: {problems}. Match the args schema in TOOLS."


def _observation(result: ToolResult) -> str:
    data = json.dumps(result.data, separators=(",", ":"), default=str)
    text = f"{result.summary} {data}" if result.data else result.summary
    return text if len(text) <= DETAIL_CHARS else f"{text[:DETAIL_CHARS]} [cut]"
