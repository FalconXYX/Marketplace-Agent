"""send_message: the only way a message reaches a buyer.

| Rule | Agent, system | Human |
|---|---|---|
| Thread is this item's, platform connected | Yes | Yes |
| Thread not frozen, buyer not flagged | Yes | No |
| Daily outbound cap, follow-up streak | Yes | No |
| Text rules (contact info, money, scam, length) | Yes | Yes |
| Address rule ([PICKUP_ADDRESS] only after a booked pickup past the undo window) | Yes | No |

The model writes [PICKUP_ADDRESS]; only the platform receives the real address. Stored bodies,
summaries, and errors carry the placeholder.

On a SOLD or WITHDRAWN item the guard blocks every message but one: the system's courtesy reply
(COURTESY_REPLIES), at most once per thread, never to a buyer who holds or bought the item or
whose deal the owner's withdrawal cancelled (had_deal). It passes guard.validate.validate_courtesy
and is stored with via COURTESY_VIA. No model call.
"""
from datetime import timedelta
from typing import ClassVar

from pydantic import BaseModel
from sqlmodel import Session, col, or_, select

from marketplace_agent.core.items import get_goal
from marketplace_agent.core.models import (
    Commitment,
    CommitmentKind,
    Direction,
    Item,
    ItemStatus,
    Message,
    Thread,
)
from marketplace_agent.guard.ledger import WITHDRAWAL
from marketplace_agent.guard.invariants import (
    ADDRESS_PLACEHOLDER,
    check_outbound_text,
    outbound_count_today,
    outbound_streak,
    redact_address,
)
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext, ToolResult, Verdict
from marketplace_agent.tools.impl.common import (
    ToolArgs,
    buyer_name,
    caps_reset_at,
    fmt_local,
    item_of,
    thread_for_item,
    unsafe_thread,
)

FOLLOW_UP_AFTER = timedelta(hours=24)
_DEAL_ENDS = (CommitmentKind.RELEASED, CommitmentKind.UNDONE, CommitmentKind.COMPLETED)
COURTESY_VIA = "courtesy"  # Message.via of the courtesy reply; it counts toward the outbound cap
COURTESY_REPLIES = {
    ItemStatus.SOLD: "Sorry, it's sold now. Thanks for asking!",
    ItemStatus.WITHDRAWN: "Sorry, it's no longer available. Thanks for asking!",
}


class SendMessageArgs(ToolArgs):
    thread_id: int
    body: str


class SendMessageTool(Tool):
    name: ClassVar[str] = "send_message"
    description: ClassVar[str] = (
        "Send one message to the buyer in a thread. Put everything you want to tell that buyer "
        "in one message: after it you may send at most one follow-up, and only 24h later if they "
        f"have not replied. Never type the pickup address; write {ADDRESS_PLACEHOLDER}, which "
        "is filled in only after a pickup is booked and the owner's undo window has passed."
    )
    Args: ClassVar[type[BaseModel]] = SendMessageArgs
    trust_action: ClassVar[str | None] = "send_message"

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        assert isinstance(args, SendMessageArgs)
        thread = ctx.session.get(Thread, args.thread_id)
        return thread.platform if thread is not None else "*"

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, SendMessageArgs)
        thread = thread_for_item(ctx, args.thread_id)
        if isinstance(thread, Reject):
            return thread
        if thread.platform not in ctx.platforms:
            return Reject(
                f"{thread.platform} is not connected right now, so this buyer cannot be "
                "messaged. Try again on a later wake, or escalate if it is urgent."
            )
        if ctx.actor != "human":
            rejected = (
                unsafe_thread(ctx, thread, "message this buyer")
                or _cap_reached(ctx, thread.platform)
                or _streak_limit(ctx, thread)
            )
            if rejected:
                return rejected
        reasons = _text_reasons(ctx, thread, args.body)
        return Reject(" ".join(reasons)) if reasons else Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        """Sends first, then stores: a platform call can be slow (browser automation), and a
        write before it would hold SQLite's write lock, and so block the owner's undo, for as
        long as it takes."""
        assert isinstance(args, SendMessageArgs)
        thread = ctx.session.get(Thread, args.thread_id)
        assert thread is not None
        now = ctx.clock.now()
        stored = redact_address(args.body, ctx.settings)
        delivered = args.body.replace(ADDRESS_PLACEHOLDER, ctx.settings.pickup_address)
        try:
            remote_id = ctx.platforms[thread.platform].send_message(thread.remote_id, delivered)
        except PlatformError as exc:
            # Adapters may echo the body they failed to send; that text may hold the address.
            raise type(exc)(redact_address(str(exc), ctx.settings)) from None
        message = Message(
            thread_id=thread.id,
            direction=Direction.OUTBOUND,
            body=stored,
            sent_at=now,
            remote_id=remote_id,
            handled=True,
            via=ctx.actor,
        )
        ctx.session.add(message)
        thread.last_outbound_at = now
        ctx.session.flush()
        name = buyer_name(ctx.session, thread.buyer_id)
        return ToolResult(
            ok=True,
            summary=f"Sent to {name} in thread {thread.id}: {stored}",
            data={"message_id": message.id, "thread_id": thread.id},
        )


def _cap_reached(ctx: ToolContext, platform: str) -> Reject | None:
    cap = ctx.settings.daily_outbound_cap.get(platform, 0)
    sent = outbound_count_today(ctx.session, ctx.clock, platform)
    if sent < cap:
        return None
    return Reject(
        f"The daily limit of {cap} messages on {platform} is used up ({sent} sent today). "
        f"Send nothing more on this platform until {caps_reset_at(ctx)}; escalate anything "
        "urgent."
    )


def _streak_limit(ctx: ToolContext, thread: Thread) -> Reject | None:
    count, last = outbound_streak(ctx.session, thread.id)
    if count >= 2:
        return Reject(
            f"You have sent {count} messages in a row without a reply from this buyer. Do not "
            "message them again until they write back."
        )
    if count == 1 and last is not None and ctx.clock.now() - last < FOLLOW_UP_AFTER:
        return Reject(
            f"Your last message to this buyer ({fmt_local(ctx.settings, last)}) has no reply "
            "yet. Wait for them; one follow-up is allowed from "
            f"{fmt_local(ctx.settings, last + FOLLOW_UP_AFTER)} if they stay quiet."
        )
    return None


def _text_reasons(ctx: ToolContext, thread: Thread, body: str) -> list[str]:
    reasons = check_outbound_text(
        ctx.session,
        ctx.settings,
        ctx.clock,
        item=item_of(ctx),
        goal=get_goal(ctx.session, ctx.item_id),
        thread=thread,
        body=body,
    )
    if ctx.actor == "human":
        reasons = [r for r in reasons if ADDRESS_PLACEHOLDER not in r]
    if ADDRESS_PLACEHOLDER in body and not ctx.settings.pickup_address.strip():
        reasons.append(
            f"No pickup address is configured, so {ADDRESS_PLACEHOLDER} cannot be filled in. "
            "Share the pickup area instead and escalate so the owner can send the address."
        )
    return reasons


def courtesy_reply(status: ItemStatus) -> str | None:
    """The one message a buyer may get about a SOLD or WITHDRAWN item; None for other states."""
    return COURTESY_REPLIES.get(status)


def had_courtesy_reply(session: Session, thread_id: int) -> bool:
    query = select(Message.id).where(
        Message.thread_id == thread_id,
        Message.direction == Direction.OUTBOUND,
        Message.via == COURTESY_VIA,
    )
    return session.exec(query).first() is not None


def had_deal(session: Session, thread: Thread) -> bool:
    """The buyer holds or held this item, so the owner talks to them, not a canned reply: a deal
    on this item, in this thread or in another one (a bundle agreed elsewhere), that is still
    active, was completed, or was cancelled by the owner withdrawing the item (the owner was told
    to tell them). A deal released or undone while the item stayed for sale leaves an ordinary
    buyer, and so does a deal on another item that happens to sit in this thread."""
    query = select(Commitment).where(
        Commitment.kind == CommitmentKind.ACCEPTED,
        Commitment.item_id == thread.item_id,
        or_(Commitment.thread_id == thread.id, Commitment.buyer_id == thread.buyer_id),
    )
    return any(_still_theirs(session, accepted) for accepted in session.exec(query).all())


def _still_theirs(session: Session, accepted: Commitment) -> bool:
    closing = session.exec(
        select(Commitment).where(
            Commitment.ref_id == accepted.id, col(Commitment.kind).in_(_DEAL_ENDS)
        )
    ).first()
    if closing is None or closing.kind == CommitmentKind.COMPLETED:
        return True
    if closing.kind != CommitmentKind.RELEASED or closing.reason != WITHDRAWAL:
        return False
    item = session.get(Item, accepted.item_id)
    return item is not None and item.status == ItemStatus.WITHDRAWN
