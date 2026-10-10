from typing import ClassVar

import pytest
from pydantic import BaseModel
from sqlmodel import select

from marketplace_agent.core.models import Item, ItemStatus, TrustRecord, TrustTier
from marketplace_agent.guard.trust import record_correction, tier
from marketplace_agent.guard.validate import after_execute, validate
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.tools.base import (
    Allow,
    NeedsApproval,
    Reject,
    Tool,
    ToolContext,
    ToolResult,
    Verdict,
)


class ProbeArgs(BaseModel):
    platform: str = "*"


class Probe(Tool):
    """Records whether check ran; returns a preset verdict."""

    name: ClassVar[str] = "update_price"
    description: ClassVar[str] = "test tool"
    Args: ClassVar[type[BaseModel]] = ProbeArgs
    trust_action: ClassVar[str | None] = "update_price"

    def __init__(self, verdict: Verdict | None = None):
        self.verdict = verdict or Allow()
        self.checks = 0

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        assert isinstance(args, ProbeArgs)
        return args.platform

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        self.checks += 1
        return self.verdict

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        return ToolResult(ok=True, summary="did it")


class ReadOnlyProbe(Probe):
    name: ClassVar[str] = "get_stats"
    trust_action: ClassVar[str | None] = None
    read_only: ClassVar[bool] = True


class RememberProbe(Probe):
    name: ClassVar[str] = "remember"
    trust_action: ClassVar[str | None] = None


class EscalateProbe(Probe):
    name: ClassVar[str] = "escalate"
    trust_action: ClassVar[str | None] = None


class AcceptProbe(Probe):
    name: ClassVar[str] = "accept_offer"
    trust_action: ClassVar[str | None] = "accept_offer"


class UngatedProbe(Probe):
    name: ClassVar[str] = "ungated"
    trust_action: ClassVar[str | None] = None


@pytest.fixture
def item(session, clock) -> Item:
    item = Item(
        title="Desk", status=ItemStatus.LISTED, created_at=clock.now(), updated_at=clock.now()
    )
    session.add(item)
    session.flush()
    return item


def _ctx(session, clock, settings, notifier, item_id: int, actor: str = "agent") -> ToolContext:
    return ToolContext(
        session=session,
        clock=clock,
        settings=settings,
        platforms={},
        notifier=notifier,
        item_id=item_id,
        actor=actor,
    )


def _trust_rows(session) -> int:
    return len(session.exec(select(TrustRecord)).all())


# ---------------------------------------------------------------- validate


def test_missing_item_is_rejected_before_anything_else(session, clock, settings, notifier):
    probe = Probe()
    verdict = validate(_ctx(session, clock, settings, notifier, 999), probe, ProbeArgs())
    assert isinstance(verdict, Reject) and "999" in verdict.reason
    assert probe.checks == 0


@pytest.mark.parametrize("actor", ["agent", "system"])
def test_paused_item_rejects_automated_actors(session, clock, settings, notifier, item, actor):
    item.paused = True
    probe = Probe()

    verdict = validate(_ctx(session, clock, settings, notifier, item.id, actor), probe, ProbeArgs())

    assert isinstance(verdict, Reject) and "paused by the human" in verdict.reason
    assert probe.checks == 0
    assert _trust_rows(session) == 0


def test_paused_item_still_lets_the_human_act(session, clock, settings, notifier, item):
    item.paused = True
    probe = Probe()
    verdict = validate(
        _ctx(session, clock, settings, notifier, item.id, "human"), probe, ProbeArgs()
    )
    assert verdict == Allow() and probe.checks == 1


@pytest.mark.parametrize("status", [ItemStatus.SOLD, ItemStatus.WITHDRAWN])
def test_closed_item_rejects_writes_but_allows_reads_remember_and_escalate(
    session, clock, settings, notifier, item, status
):
    item.status = status
    ctx = _ctx(session, clock, settings, notifier, item.id)

    write = Probe()
    verdict = validate(ctx, write, ProbeArgs())
    assert isinstance(verdict, Reject) and status.value in verdict.reason
    assert "remember" in verdict.reason and write.checks == 0

    for allowed in (ReadOnlyProbe(), RememberProbe(), EscalateProbe()):
        assert validate(ctx, allowed, ProbeArgs()) == Allow()
        assert allowed.checks == 1


def test_closed_item_check_also_applies_to_the_human(session, clock, settings, notifier, item):
    item.status = ItemStatus.SOLD
    verdict = validate(
        _ctx(session, clock, settings, notifier, item.id, "human"), Probe(), ProbeArgs()
    )
    assert isinstance(verdict, Reject)


@pytest.mark.parametrize(
    "verdict", [Reject("price is below the floor"), NeedsApproval("tool wants it")]
)
def test_tool_check_verdict_is_returned_as_is_and_skips_trust(
    session, clock, settings, notifier, item, verdict
):
    probe = Probe(verdict)
    assert (
        validate(_ctx(session, clock, settings, notifier, item.id), probe, ProbeArgs()) is verdict
    )
    assert _trust_rows(session) == 0


def test_ask_tier_needs_approval_only_for_the_agent(session, clock, settings, notifier, item):
    args = ProbeArgs(platform="facebook")  # ("facebook", "update_price") defaults to ASK

    verdict = validate(_ctx(session, clock, settings, notifier, item.id), Probe(), args)
    assert isinstance(verdict, NeedsApproval)
    assert "update_price" in verdict.reason and "facebook" in verdict.reason

    for actor in ("human", "system"):
        assert (
            validate(_ctx(session, clock, settings, notifier, item.id, actor), Probe(), args)
            == Allow()
        )


def test_trust_is_consulted_for_the_tools_platform(session, clock, settings, notifier, item):
    ctx = _ctx(session, clock, settings, notifier, item.id)
    assert validate(ctx, Probe(), ProbeArgs(platform="kijiji")) == Allow()
    record_correction(session, clock, "kijiji", "update_price")
    assert isinstance(validate(ctx, Probe(), ProbeArgs(platform="kijiji")), NeedsApproval)


def test_ungated_tools_skip_the_trust_ladder(session, clock, settings, notifier, item):
    assert (
        validate(_ctx(session, clock, settings, notifier, item.id), UngatedProbe(), ProbeArgs())
        == Allow()
    )
    assert _trust_rows(session) == 0


# ---------------------------------------------------------------- after_execute


def test_act_and_notify_success_notifies_and_counts(session, clock, settings, notifier, item):
    ctx = _ctx(session, clock, settings, notifier, item.id)

    after_execute(ctx, Probe(), ProbeArgs(), ToolResult(ok=True, summary="Price now $105.00"))

    [notice] = notifier.notices
    assert (notice.kind, notice.body, notice.item_id) == (
        NoticeKind.INFO,
        "Price now $105.00",
        item.id,
    )
    record = session.exec(select(TrustRecord)).one()
    assert (record.platform, record.action, record.executed, record.consecutive_clean) == (
        "*",
        "update_price",
        1,
        1,
    )


def test_accept_offer_never_sends_the_info_notice(session, clock, settings, notifier, item):
    after_execute(
        _ctx(session, clock, settings, notifier, item.id),
        AcceptProbe(),
        ProbeArgs(),
        ToolResult(True, "ok"),
    )
    assert notifier.notices == []
    assert session.exec(select(TrustRecord)).one().executed == 1


def test_complete_sale_never_sends_the_info_notice(session, clock, settings, notifier, item):
    class CompleteProbe(Probe):
        name: ClassVar[str] = "complete_sale"
        trust_action: ClassVar[str | None] = "complete_sale"

    after_execute(
        _ctx(session, clock, settings, notifier, item.id),
        CompleteProbe(),
        ProbeArgs(),
        ToolResult(True, "Sold to Maya for $110.00."),
    )
    assert notifier.notices == []  # the tool's own SOLD notice says the same
    assert session.exec(select(TrustRecord)).one().executed == 1


def test_accept_bundle_never_sends_the_info_notice(session, clock, settings, notifier, item):
    class BundleProbe(Probe):
        name: ClassVar[str] = "accept_bundle"
        trust_action: ClassVar[str | None] = "accept_offer"

    after_execute(
        _ctx(session, clock, settings, notifier, item.id),
        BundleProbe(),
        ProbeArgs(),
        ToolResult(True, "Accepted $150.00 from Ann for 2 items."),
    )
    assert notifier.notices == []  # its own COMMITMENT notice says the same
    assert session.exec(select(TrustRecord)).one().executed == 1


def test_info_notice_title_names_the_action_and_item(session, clock, settings, notifier, item):
    ctx = _ctx(session, clock, settings, notifier, item.id)
    after_execute(ctx, Probe(), ProbeArgs(), ToolResult(True, "ok"))
    [notice] = notifier.notices
    assert notice.title == "Price changed: Desk"


def test_silent_tier_counts_without_notifying(session, clock, settings, notifier, item):
    class Messenger(Probe):
        name: ClassVar[str] = "send_message"
        trust_action: ClassVar[str | None] = "send_message"

    after_execute(
        _ctx(session, clock, settings, notifier, item.id),
        Messenger(),
        ProbeArgs(),
        ToolResult(True, "sent"),
    )
    assert notifier.notices == []
    assert session.exec(select(TrustRecord)).one().executed == 1


def test_notice_uses_the_tier_before_this_execution(session, clock, settings, notifier, item):
    settings = settings.model_copy(update={"trust_promote_after": 1})
    ctx = _ctx(session, clock, settings, notifier, item.id)

    after_execute(ctx, Probe(), ProbeArgs(), ToolResult(True, "first"))  # promotes to SILENT
    after_execute(ctx, Probe(), ProbeArgs(), ToolResult(True, "second"))

    assert [n.body for n in notifier.notices] == ["first"]
    assert tier(session, clock, "*", "update_price") == TrustTier.SILENT


@pytest.mark.parametrize(
    ("actor", "result", "tool"),
    [
        ("human", ToolResult(True, "ok"), Probe()),
        ("system", ToolResult(True, "ok"), Probe()),
        ("agent", ToolResult(False, "platform error"), Probe()),
        ("agent", ToolResult(True, "ok"), UngatedProbe()),
    ],
    ids=["human", "system", "failed", "ungated"],
)
def test_after_execute_ignores_non_agent_failed_and_ungated(
    session, clock, settings, notifier, item, actor, result, tool
):
    after_execute(
        _ctx(session, clock, settings, notifier, item.id, actor), tool, ProbeArgs(), result
    )
    assert notifier.notices == []
    assert _trust_rows(session) == 0
