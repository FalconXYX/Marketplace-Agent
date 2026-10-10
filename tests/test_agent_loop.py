"""Agent.wake with stub tools, a real ProviderRouter and QuotaTracker, and ScriptedProvider."""

import json
import sqlite3
from datetime import timedelta
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from marketplace_agent.agent.context import Trigger
from marketplace_agent.agent.loop import MAX_OUTPUT_TOKENS, Agent, WakeResult
from marketplace_agent.agent.plan import PLAN_WIRE_SCHEMA
from marketplace_agent.ai.base import Capability, LLMRequest, ProviderError, QuotaState
from marketplace_agent.ai.prompting import load_prompt
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import ProviderLimit, Settings
from marketplace_agent.core.db import init_db, make_engine
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    AgentRun,
    Buyer,
    Direction,
    Item,
    ItemStatus,
    Memory,
    MemoryScope,
    Message,
    PatternEntry,
    PendingApproval,
    QuotaUsage,
    Thread,
    ThreadState,
)
from marketplace_agent.guard.invariants import ADDRESS_PLACEHOLDER
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext, ToolResult, Verdict

# ---------------------------------------------------------------- stub tools


class SendArgs(BaseModel):
    thread_id: int
    body: str = Field(min_length=1)


class SendStub(Tool):
    """Delivers with the placeholder filled in, like the real send_message, and echoes it."""

    name: ClassVar[str] = "send_message"
    description: ClassVar[str] = "Message a buyer."
    Args: ClassVar[type[BaseModel]] = SendArgs
    trust_action: ClassVar[str | None] = "send_message"

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, SendArgs)
        thread = ctx.session.get(Thread, args.thread_id)
        if thread is None or thread.item_id != ctx.item_id:
            return Reject(f"Thread {args.thread_id} is not about this item. Use a listed thread.")
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, SendArgs)
        delivered = args.body.replace(ADDRESS_PLACEHOLDER, ctx.settings.pickup_address)
        ctx.session.add(
            Message(
                thread_id=args.thread_id,
                direction=Direction.OUTBOUND,
                body=args.body,
                sent_at=ctx.clock.now(),
                via=ctx.actor,
            )
        )
        return ToolResult(ok=True, summary=f"Delivered: {delivered}")


class PriceArgs(BaseModel):
    price_cents: int


class FloorGuardStub(Tool):
    name: ClassVar[str] = "update_price"
    description: ClassVar[str] = "Change the asking price."
    Args: ClassVar[type[BaseModel]] = PriceArgs

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        return Reject("$50.00 is below the floor of $95.00. Choose $95.00 or more.")

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        raise AssertionError("never runs")


class PlatformArgs(BaseModel):
    platform: str


class PublishStub(Tool):
    """facebook publish starts at trust tier ASK, so the real guard asks the owner."""

    name: ClassVar[str] = "publish"
    description: ClassVar[str] = "Publish on a platform."
    Args: ClassVar[type[BaseModel]] = PlatformArgs
    trust_action: ClassVar[str | None] = "publish"

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        assert isinstance(args, PlatformArgs)
        return args.platform

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        raise AssertionError("needs approval first")


class NoArgs(BaseModel):
    pass


class ExplodeStub(Tool):
    name: ClassVar[str] = "explode"
    description: ClassVar[str] = "Writes, then crashes."
    Args: ClassVar[type[BaseModel]] = NoArgs

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        item = ctx.session.get(Item, ctx.item_id)
        assert item is not None
        item.title = "half-written"
        ctx.session.flush()
        raise RuntimeError("platform timed out")


class RelistStub(Tool):
    name: ClassVar[str] = "relist"
    description: ClassVar[str] = "Relist."
    Args: ClassVar[type[BaseModel]] = NoArgs

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        return ToolResult(ok=False, summary="The platform refused. Try again tomorrow.")


class StatsStub(Tool):
    name: ClassVar[str] = "get_stats"
    description: ClassVar[str] = "Listing stats."
    Args: ClassVar[type[BaseModel]] = NoArgs
    read_only: ClassVar[bool] = True

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        return ToolResult(ok=True, summary="Stats for 1 listing.", data={"views": 40})


class Registry:
    def __init__(self, *tools: Tool):
        self.tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]


def _registry() -> Registry:
    return Registry(
        SendStub(), FloorGuardStub(), PublishStub(), ExplodeStub(), RelistStub(), StatsStub()
    )


# ---------------------------------------------------------------- plans and setup


def act(tool: str, **args: Any) -> dict[str, Any]:
    return {"tool": tool, "args_json": json.dumps(args), "why": f"try {tool}"}


def plan(*actions: dict[str, Any], rationale: str = "r", labels=()) -> dict[str, Any]:
    return {
        "rationale": rationale,
        "message_labels": [{"message_id": m, "intent": i} for m, i in labels],
        "actions": list(actions),
    }


def _seed(session: Session, clock, *bodies: str) -> tuple[int, int, list[int]]:
    """A listed item with a goal, one buyer thread, and unhandled inbound messages."""
    item = Item(
        title="IKEA desk", status=ItemStatus.LISTED, created_at=clock.now(), updated_at=clock.now()
    )
    session.add(item)
    session.flush()
    set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
    buyer = Buyer(platform="fake", handle="maya", display_name="Maya")
    session.add(buyer)
    session.flush()
    thread = Thread(
        platform="fake",
        remote_id="t1",
        item_id=item.id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    ids = []
    for n, body in enumerate(bodies):
        message = Message(
            thread_id=thread.id,
            direction=Direction.INBOUND,
            body=body,
            sent_at=clock.now() - timedelta(minutes=len(bodies) - n),
            via="buyer",
        )
        session.add(message)
        session.flush()
        ids.append(message.id)
    session.commit()
    return item.id, thread.id, ids


def _router(engine, clock, *providers, limits=None) -> ProviderRouter:
    return ProviderRouter(list(providers), QuotaTracker(engine, clock, limits or {}))


def _agent(router, settings, clock, notifier, registry=None, max_steps=3) -> Agent:
    return Agent(
        router, registry or _registry(), settings, clock, notifier, {}, max_steps=max_steps
    )


def _wake(session, settings, clock, notifier, script, *, max_steps=3, registry=None):
    provider = ScriptedProvider(script)
    agent = _agent(
        _router(session.get_bind(), clock, provider), settings, clock, notifier, registry, max_steps
    )
    return provider, agent


def _statuses(result: WakeResult) -> list[tuple[int, str, str]]:
    return [(o.step, o.tool, o.status) for o in result.outcomes]


# ---------------------------------------------------------------- the request


def test_request_uses_agent_prompt_schema_and_room_for_thinking(session, settings, clock, notifier):
    item_id, _, _ = _seed(session, clock, "Hi")
    provider, agent = _wake(session, settings, clock, notifier, [plan()])

    agent.wake(session, item_id, [Trigger("new_message")])

    [request] = provider.requests
    assert request.system == load_prompt("agent_system")
    assert request.prompt.split("\n", 1)[0] == f"ITEM_ID: {item_id}"
    assert request.capability == Capability.REASONING
    assert request.json_schema == PLAN_WIRE_SCHEMA
    assert request.max_output_tokens == MAX_OUTPUT_TOKENS == 8192
    assert request.purpose == "wake"


# ---------------------------------------------------------------- outcomes


def test_each_outcome_kind(session, settings, clock, notifier):
    item_id, thread_id, _ = _seed(session, clock, "Would you take 50?")
    script = [
        plan(
            act("send_message", thread_id=thread_id, body="I can do $110."),
            act("update_price", price_cents=5000),
            act("publish", platform="facebook"),
            act("relist"),
            act("get_stats"),
        ),
        plan(),
    ]
    _, agent = _wake(session, settings, clock, notifier, script)

    result = agent.wake(session, item_id, [Trigger("new_message")])

    assert _statuses(result) == [
        (1, "send_message", "executed"),
        (1, "update_price", "rejected"),
        (1, "publish", "needs_approval"),
        (1, "relist", "failed"),
        (1, "get_stats", "observed"),
    ]
    by_tool = {o.tool: o for o in result.outcomes}
    assert by_tool["update_price"].detail.startswith("$50.00 is below the floor")
    assert by_tool["relist"].detail == "The platform refused. Try again tomorrow."
    assert by_tool["get_stats"].detail == 'Stats for 1 listing. {"views":40}'
    assert "needs the owner's approval" in by_tool["publish"].detail
    assert by_tool["send_message"].args == {"thread_id": thread_id, "body": "I can do $110."}
    sent = session.exec(select(Message).where(Message.direction == Direction.OUTBOUND)).all()
    assert [m.body for m in sent] == ["I can do $110."]
    assert result.error is None and result.requests_used == 2


def test_needs_approval_parks_the_call_once_and_tells_the_owner(session, settings, clock, notifier):
    item_id, _, _ = _seed(session, clock)
    script = [plan(act("publish", platform="facebook"))] * 2
    _, agent = _wake(session, settings, clock, notifier, script)

    first = agent.wake(session, item_id, [Trigger("idle")])
    second = agent.wake(session, item_id, [Trigger("idle")])

    [approval] = session.exec(select(PendingApproval)).all()
    assert (approval.item_id, approval.tool, approval.platform) == (item_id, "publish", "facebook")
    assert json.loads(approval.args_json) == {"platform": "facebook"}
    assert approval.agent_run_id == first.run_id
    assert approval.rationale == "try publish"
    [notice] = notifier.of_kind(NoticeKind.APPROVAL)
    assert notice.ref_id == approval.id and notice.item_id == item_id
    assert first.requests_used == 1  # needs_approval alone asks for no follow-up step
    assert "already waiting for the owner" in second.outcomes[0].detail
    assert second.outcomes[0].status == "needs_approval"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ({"tool": "teleport", "args_json": "{}"}, "There is no tool named 'teleport'. Use one of:"),
        ({"tool": "update_price", "args_json": "{price: 5}"}, "args_json is not valid JSON"),
        ({"tool": "update_price", "args_json": "[5]"}, "args_json must encode a JSON object"),
        (
            {"tool": "update_price", "args_json": '{"price_cents": "lots"}'},
            "Invalid arguments for update_price: price_cents: Input should be a valid integer",
        ),
        (
            {"tool": "send_message", "args_json": '{"thread_id": 1}'},
            "Invalid arguments for send_message: body: Field required",
        ),
    ],
)
def test_malformed_actions_are_rejected_with_a_precise_reason(
    session, settings, clock, notifier, raw, reason
):
    item_id, _, _ = _seed(session, clock)
    _, agent = _wake(session, settings, clock, notifier, [plan(raw), plan()])

    result = agent.wake(session, item_id, [])

    assert result.outcomes[0].status == "rejected"
    assert reason in result.outcomes[0].detail


def test_tool_exception_rolls_back_and_later_actions_still_run(session, settings, clock, notifier):
    item_id, thread_id, [message_id] = _seed(session, clock, "Still there?")
    script = [
        plan(
            act("explode"), act("get_stats"), act("send_message", thread_id=thread_id, body="Yes")
        ),
        plan(),
    ]
    _, agent = _wake(session, settings, clock, notifier, script)

    result = agent.wake(session, item_id, [Trigger("new_message")])

    assert _statuses(result)[:3] == [
        (1, "explode", "failed"),
        (1, "get_stats", "observed"),
        (1, "send_message", "skipped"),  # it may announce what explode was doing
    ]
    assert result.outcomes[0].detail == "explode raised RuntimeError: platform timed out"
    session.expire_all()
    assert session.get(Item, item_id).title == "IKEA desk"
    assert session.get(Message, message_id).handled  # committed before the tools ran
    assert session.exec(select(Message).where(Message.direction == Direction.OUTBOUND)).all() == []


def test_messages_after_a_rejected_or_parked_action_wait_for_the_next_step(
    session, settings, clock, notifier
):
    item_id, thread_id, _ = _seed(session, clock, "Lower?")
    script = [
        plan(
            act("send_message", thread_id=thread_id, body="Let me check."),
            act("update_price", price_cents=5_000),
            act("send_message", thread_id=thread_id, body="Dropped it to $50!"),
        ),
        plan(
            act("publish", platform="facebook"),
            act("send_message", thread_id=thread_id, body="Posted on Facebook too!"),
        ),
        plan(),
    ]
    _, agent = _wake(session, settings, clock, notifier, script)

    result = agent.wake(session, item_id, [Trigger("new_message")])

    assert _statuses(result) == [
        (1, "send_message", "executed"),  # before the rejection: unaffected
        (1, "update_price", "rejected"),
        (1, "send_message", "skipped"),
        (2, "publish", "needs_approval"),
        (2, "send_message", "skipped"),
    ]
    assert result.outcomes[2].detail.startswith(
        "Not sent: it came after update_price, which was rejected"
    )


def test_address_in_a_tool_result_is_redacted_everywhere(session, settings, clock, notifier):
    item_id, thread_id, _ = _seed(session, clock)
    body = f"See you at {ADDRESS_PLACEHOLDER}"
    _, agent = _wake(
        session,
        settings,
        clock,
        notifier,
        [plan(act("send_message", thread_id=thread_id, body=body))],
    )

    result = agent.wake(session, item_id, [])

    assert result.outcomes[0].detail == f"Delivered: See you at {ADDRESS_PLACEHOLDER}"
    run = session.get(AgentRun, result.run_id)
    assert "123 Example" not in run.results_json and "123 Example" not in run.plan_json


# ---------------------------------------------------------------- steps


def test_follow_up_step_sees_reject_reasons_and_fresh_context(session, settings, clock, notifier):
    item_id, thread_id, _ = _seed(session, clock, "Lower?")
    script = [
        plan(
            act("send_message", thread_id=thread_id, body="Best I can do is $110."),
            act("update_price", price_cents=5000),
            rationale="Counter and cut the price.",
        ),
        plan(rationale="Holding the price."),
    ]
    provider, agent = _wake(session, settings, clock, notifier, script)

    result = agent.wake(session, item_id, [Trigger("new_message")])

    assert len(provider.requests) == 2 and result.requests_used == 2
    follow_up = provider.requests[1].prompt
    assert follow_up.startswith(f"ITEM_ID: {item_id}")
    assert "## RESULTS OF YOUR LAST PLAN" in follow_up
    assert "This is step 2 of at most 3" in follow_up
    assert "Your rationale was: Counter and cut the price." in follow_up
    assert (
        '- step 1 update_price {"price_cents":5000} -> rejected: $50.00 is below the floor'
        in follow_up
    )
    assert "-> executed: Delivered: Best I can do is $110." in follow_up
    assert (
        "] You · " in follow_up
        and "Best I can do is $110." in follow_up.split("## CONVERSATIONS")[1]
    )
    assert "· NEW" not in follow_up  # step 1 marked them handled
    assert result.rationale == "Holding the price."


def test_observations_feed_a_follow_up_step(session, settings, clock, notifier):
    item_id, _, _ = _seed(session, clock)
    provider, agent = _wake(session, settings, clock, notifier, [plan(act("get_stats")), plan()])

    agent.wake(session, item_id, [Trigger("idle")])

    assert len(provider.requests) == 2
    assert '-> observed: Stats for 1 listing. {"views":40}' in provider.requests[1].prompt


def test_no_follow_up_when_everything_executed(session, settings, clock, notifier):
    item_id, thread_id, _ = _seed(session, clock, "Hi")
    provider, agent = _wake(
        session,
        settings,
        clock,
        notifier,
        [plan(act("send_message", thread_id=thread_id, body="Hi!"))],
    )
    agent.wake(session, item_id, [])
    assert len(provider.requests) == 1


@pytest.mark.parametrize("max_steps", [1, 2, 3])
def test_max_steps_is_honoured(session, settings, clock, notifier, max_steps):
    item_id, _, _ = _seed(session, clock)
    provider, agent = _wake(
        session,
        settings,
        clock,
        notifier,
        lambda _r: plan(act("update_price", price_cents=1)),
        max_steps=max_steps,
    )

    result = agent.wake(session, item_id, [])

    assert len(provider.requests) == result.requests_used == max_steps
    assert [o.step for o in result.outcomes] == list(range(1, max_steps + 1))


# ---------------------------------------------------------------- messages


def test_labels_are_applied_and_availability_phrasings_learned(session, settings, clock, notifier):
    item_id, _, [ask, offer] = _seed(session, clock, "Still have this?", "Would you take 90?")
    script = [plan(labels=[(ask, "availability"), (offer, "offer"), (9999, "spam")])]
    _, agent = _wake(session, settings, clock, notifier, script)

    agent.wake(session, item_id, [Trigger("new_message")])

    session.expire_all()
    assert (session.get(Message, ask).handled, session.get(Message, ask).intent) == (
        True,
        "availability",
    )
    assert (session.get(Message, offer).handled, session.get(Message, offer).intent) == (
        True,
        "offer",
    )
    learned = session.exec(select(PatternEntry)).all()
    assert [(p.normalized, p.intent, p.source) for p in learned] == [
        ("still have this", "availability", "learned")
    ]


def test_messages_stay_unhandled_when_the_provider_fails(session, settings, clock, notifier):
    item_id, _, [message_id] = _seed(session, clock, "Hi, is this available?")
    _, agent = _wake(session, settings, clock, notifier, [ProviderError("upstream 500")])

    result = agent.wake(session, item_id, [Trigger("new_message")])

    assert result.error == "step 1: ProviderError: upstream 500"
    assert result.outcomes == [] and result.requests_used == 0
    session.expire_all()
    assert session.get(Message, message_id).handled is False
    run = session.get(AgentRun, result.run_id)
    assert run.error == result.error and run.finished_at is not None
    assert session.get(Item, item_id).last_agent_run_at is None  # time triggers stay armed


def test_unusable_plan_is_an_error_and_leaves_messages_unhandled(
    session, settings, clock, notifier
):
    item_id, _, [message_id] = _seed(session, clock, "Hi")
    _, agent = _wake(session, settings, clock, notifier, [{"rationale": "x", "actions": "oops"}])

    result = agent.wake(session, item_id, [])

    assert result.error.startswith("step 1: unusable plan from scripted")
    session.expire_all()
    assert session.get(Message, message_id).handled is False


def test_failure_on_a_later_step_keeps_the_first_steps_work(session, settings, clock, notifier):
    item_id, _, [message_id] = _seed(session, clock, "Hi")
    script = [plan(act("update_price", price_cents=1)), ProviderError("rate limited")]
    _, agent = _wake(session, settings, clock, notifier, script)

    result = agent.wake(session, item_id, [])

    assert result.error == "step 2: ProviderError: rate limited"
    assert _statuses(result) == [(1, "update_price", "rejected")]
    session.expire_all()
    assert session.get(Message, message_id).handled is True
    stamped = session.get(Item, item_id).last_agent_run_at
    assert stamped.replace(tzinfo=None) == clock.now().replace(tzinfo=None)


# ---------------------------------------------------------------- the run record


def test_agent_run_is_persisted_with_rationale_plan_and_outcomes(
    session, settings, clock, notifier
):
    item_id, thread_id, _ = _seed(session, clock, "Hi")
    script = [
        plan(act("update_price", price_cents=5000), rationale="Cut it."),
        plan(act("send_message", thread_id=thread_id, body="Hi Maya!"), rationale="Just reply."),
    ]
    _, agent = _wake(session, settings, clock, notifier, script)
    trigger = Trigger("new_message", "1 new", thread_id=thread_id, message_ids=(1,))

    result = agent.wake(session, item_id, [trigger])

    session.expire_all()
    run = session.get(AgentRun, result.run_id)
    assert run.rationale == result.rationale == "Just reply."
    assert run.provider == "scripted" and run.requests_used == 2
    assert run.quota_state == "healthy" and run.error is None
    assert run.finished_at is not None
    assert json.loads(run.triggers_json) == [
        {"kind": "new_message", "detail": "1 new", "thread_id": thread_id, "message_ids": [1]}
    ]
    plans = json.loads(run.plan_json)
    assert [(p["step"], p["rationale"]) for p in plans] == [(1, "Cut it."), (2, "Just reply.")]
    assert plans[0]["actions"][0]["args"] == {"price_cents": 5000}
    results = json.loads(run.results_json)
    assert [(r["step"], r["tool"], r["status"]) for r in results] == [
        (1, "update_price", "rejected"),
        (2, "send_message", "executed"),
    ]
    assert session.get(Item, item_id).last_agent_run_at is not None


def test_recent_run_appears_in_the_next_wakes_context(session, settings, clock, notifier):
    item_id, _, _ = _seed(session, clock)
    script = [
        plan(act("update_price", price_cents=1), rationale="Tried a cut."),
        plan(rationale="Holding the price."),
        plan(),
    ]
    provider, agent = _wake(session, settings, clock, notifier, script)

    agent.wake(session, item_id, [Trigger("idle")])
    clock.advance(hours=1)
    agent.wake(session, item_id, [Trigger("idle")])

    later = provider.requests[2].prompt.split("## YOUR RECENT RUNS")[1]
    assert "woke for idle: Holding the price." in later
    assert "update_price rejected ($50.00 is below the floor" in later


# ---------------------------------------------------------------- quota ladder


def _use(quota: QuotaTracker, provider: str, model: str, n: int) -> None:
    for _ in range(n):
        quota.record(provider, model)


def _tight(rpd: int) -> ProviderLimit:
    return ProviderLimit(rpm=1000, rpd=rpd)


def _rejecting(_request: LLMRequest) -> dict[str, Any]:
    return plan(act("update_price", price_cents=1))


def test_healthy_quota_runs_all_steps(session, settings, clock, notifier, engine):
    item_id, _, _ = _seed(session, clock)
    main = ScriptedProvider(_rejecting, name="main", capabilities=frozenset({Capability.REASONING}))
    router = _router(engine, clock, main, limits={"main": _tight(10)})

    result = _agent(router, settings, clock, notifier).wake(session, item_id, [])

    assert result.quota_state == QuotaState.HEALTHY
    assert len(main.requests) == 3


def test_constrained_quota_caps_the_wake_at_two_steps(session, settings, clock, notifier, engine):
    item_id, _, _ = _seed(session, clock)
    main = ScriptedProvider(_rejecting, name="main", capabilities=frozenset({Capability.REASONING}))
    router = _router(engine, clock, main, limits={"main": _tight(10)})
    _use(router.quota, "main", "scripted-1", 8)

    result = _agent(router, settings, clock, notifier).wake(session, item_id, [])

    assert result.quota_state == QuotaState.CONSTRAINED
    assert len(main.requests) == 2
    assert "at most 2 steps" in main.requests[0].prompt
    assert session.get(AgentRun, result.run_id).quota_state == "constrained"


def test_exhausted_reasoning_falls_back_to_cheap_for_one_step(
    session, settings, clock, notifier, engine
):
    item_id, _, _ = _seed(session, clock)
    main = ScriptedProvider(_rejecting, name="main", capabilities=frozenset({Capability.REASONING}))
    backup = ScriptedProvider(_rejecting, name="backup", capabilities=frozenset({Capability.CHEAP}))
    router = _router(engine, clock, main, backup, limits={"main": _tight(1), "backup": _tight(10)})
    _use(router.quota, "main", "scripted-1", 1)

    result = _agent(router, settings, clock, notifier).wake(session, item_id, [])

    assert result.quota_state == QuotaState.EXHAUSTED
    assert main.requests == []
    [request] = backup.requests
    assert request.capability == Capability.CHEAP
    assert "exactly one step" in request.prompt
    assert _statuses(result) == [(1, "update_price", "rejected")]
    assert result.error is None and result.run_id is not None


def test_fully_exhausted_quota_makes_no_call(session, settings, clock, notifier, engine):
    item_id, _, [message_id] = _seed(session, clock, "Hi")
    main = ScriptedProvider(_rejecting, name="main")
    router = _router(engine, clock, main, limits={"main": _tight(1)})
    _use(router.quota, "main", "scripted-1", 1)

    result = _agent(router, settings, clock, notifier).wake(session, item_id, [])

    assert result == WakeResult(None, "", [], 0, QuotaState.EXHAUSTED, error="quota exhausted")
    assert main.requests == []
    assert session.exec(select(AgentRun)).all() == []
    assert session.get(Message, message_id).handled is False


def test_follow_up_stops_when_quota_runs_out_mid_wake(session, settings, clock, notifier, engine):
    item_id, _, _ = _seed(session, clock)
    main = ScriptedProvider(_rejecting, name="main")
    router = _router(engine, clock, main, limits={"main": _tight(19)})
    _use(router.quota, "main", "scripted-1", 18)  # constrained, one request left

    result = _agent(router, settings, clock, notifier).wake(session, item_id, [])

    assert len(main.requests) == 1 and result.requests_used == 1


# ---------------------------------------------------------------- file database


def test_loop_commits_before_every_model_call(tmp_path, clock, notifier):
    """With a file database QuotaTracker.record writes on its own connection, as in production.
    An uncommitted write held across router.generate would make it wait 5s and then fail."""
    db = tmp_path / "agent.db"
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        db_url=f"sqlite:///{db}",
        pickup_area="Bloor St W & Ossington Ave",
        pickup_address="123 Example St",
    )
    engine = make_engine(settings.database_url)
    init_db(engine)
    locks_free: list[bool] = []

    def responder(request: LLMRequest) -> dict[str, Any]:
        probe = sqlite3.connect(db, timeout=0.05)
        try:
            probe.execute("BEGIN EXCLUSIVE")
            probe.rollback()
            locks_free.append(True)
        except sqlite3.OperationalError:
            locks_free.append(False)
        finally:
            probe.close()
        if "RESULTS OF YOUR LAST PLAN" in request.prompt:
            return plan()
        return plan(
            act("send_message", thread_id=thread_id, body="Hi!"), act("update_price", price_cents=1)
        )

    with Session(engine, expire_on_commit=False) as session:
        item_id, thread_id, _ = _seed(session, clock, "Hi")
        session.add(
            Memory(scope=MemoryScope.GLOBAL, lesson="IKEA desk lesson", created_at=clock.now())
        )
        session.commit()  # recall() will now write a use count while building the context
        main = ScriptedProvider(responder, name="main")
        router = _router(engine, clock, main, limits={"main": _tight(100)})

        result = _agent(router, settings, clock, notifier).wake(session, item_id, [])

    assert locks_free == [True, True]
    assert result.error is None and result.requests_used == 2
    assert _statuses(result)[0] == (1, "send_message", "executed")
    with Session(engine) as check:
        assert check.exec(select(QuotaUsage.requests)).one() == 2
        assert check.exec(select(Memory.uses)).one() == 2
        assert check.get(AgentRun, result.run_id).finished_at is not None
    engine.dispose()


# ---------------------------------------------------------------- real tools, when present


def test_wake_with_the_default_registry(session, settings, clock, notifier):
    registry_module = pytest.importorskip("marketplace_agent.tools.registry")
    registry = registry_module.default_registry()
    item_id, _, _ = _seed(session, clock)
    script = [
        plan(
            act("remember", scope="item", lesson="Asked $120 on day one."),
            act("get_stats"),
            act("no_such_tool"),
        ),
        plan(),
    ]
    provider, agent = _wake(session, settings, clock, notifier, script, registry=registry)

    result = agent.wake(session, item_id, [Trigger("idle")])

    assert _statuses(result) == [
        (1, "remember", "executed"),
        (1, "get_stats", "observed"),
        (1, "no_such_tool", "rejected"),
    ]
    assert '"name":"send_message"' in provider.requests[0].prompt
    lessons = session.exec(select(Memory).where(Memory.item_id == item_id)).all()
    assert [m.lesson for m in lessons] == ["Asked $120 on day one."]
