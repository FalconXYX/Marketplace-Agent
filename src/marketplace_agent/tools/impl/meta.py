"""The agent's own bookkeeping: escalate to the owner, remember lessons, schedule a wake."""
from datetime import timedelta
from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from sqlmodel import col, select

from marketplace_agent.core import wakes
from marketplace_agent.core.models import Escalation, MemoryScope, Thread
from marketplace_agent.guard.invariants import (
    contains_contact_info,
    leaks_address,
    parse_local_datetime,
    redact_address,
)
from marketplace_agent.memory import store
from marketplace_agent.notify.base import Notice, NoticeKind
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext, ToolResult, Verdict
from marketplace_agent.tools.impl.common import ToolArgs, fmt_local, item_of, thread_for_item


MAX_WAKE_AHEAD = timedelta(days=7)
MAX_WAKE_REASON = 200


class EscalateArgs(ToolArgs):
    reason: str
    thread_id: int | None = None


class RememberArgs(ToolArgs):
    scope: Literal["item", "buyer", "global"]
    lesson: str
    buyer_id: int | None = None


class ScheduleWakeArgs(ToolArgs):
    at: str
    reason: str = Field(max_length=MAX_WAKE_REASON)


class EscalateTool(Tool):
    name: ClassVar[str] = "escalate"
    description: ClassVar[str] = (
        "Hand something to the owner when you are genuinely out of your depth: a safety worry, "
        "a question only they can answer, or something no tool can do. Say exactly what you "
        "need from them; pass thread_id when it is about one conversation."
    )
    Args: ClassVar[type[BaseModel]] = EscalateArgs

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, EscalateArgs)
        if not args.reason.strip():
            return Reject("Say in reason what you need from the owner.")
        if args.thread_id is not None:
            thread = thread_for_item(ctx, args.thread_id)
            if isinstance(thread, Reject):
                return thread
        duplicate = _open_duplicate(ctx, args)
        if duplicate is None:
            return Allow()
        return Reject(
            f"You already escalated this (escalation {duplicate.id}, "
            f"{fmt_local(ctx.settings, duplicate.created_at)}) and the owner has not resolved it. "
            "Do not escalate it again; carry on with what you can do."
        )

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, EscalateArgs)
        reason = redact_address(args.reason.strip(), ctx.settings)
        escalation = Escalation(
            item_id=ctx.item_id,
            thread_id=args.thread_id,
            reason=reason,
            created_at=ctx.clock.now(),
        )
        ctx.session.add(escalation)
        ctx.session.flush()
        ctx.notifier.notify(
            Notice(
                kind=NoticeKind.ESCALATION,
                title=f"Needs you: {item_of(ctx).title}",
                body=reason,
                item_id=ctx.item_id,
                ref_id=escalation.id,
            )
        )
        return ToolResult(
            ok=True,
            summary=f"Escalated to the owner: {reason}",
            data={"escalation_id": escalation.id},
        )


class RememberTool(Tool):
    name: ClassVar[str] = "remember"
    description: ClassVar[str] = (
        f"Save a lesson of at most {store.MAX_LESSON_CHARS} characters for later wakes. Scope "
        "'item' is about this item, 'buyer' about one buyer (give buyer_id), 'global' about "
        "selling anything. Record what worked or failed and why, never contact details."
    )
    Args: ClassVar[type[BaseModel]] = RememberArgs

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, RememberArgs)
        lesson = " ".join(args.lesson.split())
        if not lesson:
            return Reject("The lesson is empty. Write what you learned.")
        if len(lesson) > store.MAX_LESSON_CHARS:
            return Reject(
                f"The lesson is {len(lesson)} characters; the limit is {store.MAX_LESSON_CHARS}. "
                "Say it shorter."
            )
        if contains_contact_info(lesson) or leaks_address(lesson, ctx.settings):
            return Reject(
                "Lessons must not hold phone numbers, emails, links, or the pickup address. "
                "Describe the pattern instead."
            )
        if args.scope == "buyer":
            return _buyer_reason(ctx, args.buyer_id) or Allow()
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, RememberArgs)
        scope = MemoryScope(args.scope)
        memory = store.remember(
            ctx.session,
            ctx.clock,
            scope,
            args.lesson,
            item_id=ctx.item_id,
            buyer_id=args.buyer_id if scope == MemoryScope.BUYER else None,
            source=ctx.actor,
        )
        if memory is None:
            return ToolResult(
                ok=True, summary="Already known; nothing new saved.", data={"memory_id": None}
            )
        return ToolResult(
            ok=True,
            summary=f"Remembered ({scope.value}): {memory.lesson}",
            data={"memory_id": memory.id},
        )


class ScheduleWakeTool(Tool):
    name: ClassVar[str] = "schedule_wake"
    description: ClassVar[str] = (
        "Be woken at a local time, e.g. 2026-10-10T18:00, to come back to something: a "
        "follow-up once it is allowed, a late buyer, a decision you are putting off. Nothing "
        "else wakes you for these. One per item, up to 7 days ahead; scheduling again moves it."
    )
    Args: ClassVar[type[BaseModel]] = ScheduleWakeArgs

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, ScheduleWakeArgs)
        if not args.reason.strip():
            return Reject("Say in reason what you want to come back to.")
        try:
            at = parse_local_datetime(ctx.settings, args.at)
        except ValueError as exc:
            return Reject(f"{exc}. Write at as a local date and time: YYYY-MM-DDTHH:MM.")
        now = ctx.clock.now()
        if at <= now:
            return Reject(
                f"{fmt_local(ctx.settings, at)} is not in the future; it is now "
                f"{fmt_local(ctx.settings, now)}. Act now instead, or pick a later time."
            )
        if at > now + MAX_WAKE_AHEAD:
            return Reject(
                f"That is more than 7 days ahead. Pick a time before "
                f"{fmt_local(ctx.settings, now + MAX_WAKE_AHEAD)}."
            )
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, ScheduleWakeArgs)
        at = parse_local_datetime(ctx.settings, args.at)
        reason = redact_address(" ".join(args.reason.split()), ctx.settings)
        wakes.schedule(ctx.session, ctx.item_id, at, reason, ctx.clock.now())
        return ToolResult(
            ok=True,
            summary=f"You will be woken {fmt_local(ctx.settings, at)}: {reason}",
            data={"at": at.isoformat()},
        )


def _normalized(reason: str) -> str:
    return " ".join(reason.split()).casefold()


def _open_duplicate(ctx: ToolContext, args: EscalateArgs) -> Escalation | None:
    same_thread = (
        col(Escalation.thread_id).is_(None)
        if args.thread_id is None
        else col(Escalation.thread_id) == args.thread_id
    )
    query = select(Escalation).where(
        Escalation.item_id == ctx.item_id, same_thread, col(Escalation.resolved_at).is_(None)
    )
    wanted = _normalized(redact_address(args.reason, ctx.settings))
    return next((e for e in ctx.session.exec(query).all() if _normalized(e.reason) == wanted), None)


def _buyer_reason(ctx: ToolContext, buyer_id: int | None) -> Reject | None:
    query = select(Thread.buyer_id).where(Thread.item_id == ctx.item_id).distinct()
    buyers = sorted(ctx.session.exec(query).all())
    if buyer_id in buyers:
        return None
    given = "No buyer_id was given" if buyer_id is None else f"Buyer {buyer_id} has no thread here"
    return Reject(
        f"{given}. A buyer lesson needs the buyer_id of a buyer in this item's threads: "
        f"{', '.join(map(str, buyers)) or 'there are none yet'}."
    )
