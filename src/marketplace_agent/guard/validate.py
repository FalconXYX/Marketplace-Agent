"""The gate between a planned tool call and the world. The first non-Allow check wins.

| # | Check | Verdict |
|---|---|---|
| 1 | Item exists | Reject |
| 2 | Item not paused (agent and system; the human may act) | Reject |
| 3 | Item not SOLD/WITHDRAWN, unless the tool is read-only, remember, or escalate | Reject |
| 4 | tool.check | As returned |
| 5 | Trust tier ASK for an agent call | NeedsApproval |

Check 3 has one exemption, outside validate: validate_courtesy lets the system send the fixed
courtesy reply to a buyer who writes about a closed item. validate itself never loosens it.
"""
from pydantic import BaseModel

from marketplace_agent.core.models import Item, ItemStatus, Thread, TrustTier
from marketplace_agent.guard.trust import current_tier, record_executed, tier
from marketplace_agent.notify.base import Notice, NoticeKind
from marketplace_agent.tools.base import (
    Allow,
    NeedsApproval,
    Reject,
    Tool,
    ToolContext,
    ToolResult,
    Verdict,
)
from marketplace_agent.tools.impl.messaging import (
    SendMessageArgs,
    SendMessageTool,
    courtesy_reply,
    had_courtesy_reply,
    had_deal,
)

_CLOSED_ITEM = (ItemStatus.SOLD, ItemStatus.WITHDRAWN)
_ALLOWED_WHEN_CLOSED = ("remember", "escalate")
_OWN_NOTICE = ("accept_offer", "accept_bundle", "complete_sale")  # send COMMITMENT / SOLD themselves
_NOTICE_TITLES = {
    "update_price": "Price changed",
    "rewrite_listing": "Listing rewritten",
    "publish": "Posted",
    "relist": "Relisted",
    "book_pickup": "Pickup booked",
    "release_commitment": "Deal released",
}


def validate(ctx: ToolContext, tool: Tool, args: BaseModel) -> Verdict:
    item = ctx.session.get(Item, ctx.item_id)
    if item is None:
        return Reject(f"Item {ctx.item_id} does not exist. Act only on the item in your context.")
    if item.paused and ctx.actor != "human":
        return Reject(
            "This item is paused by the human. Take no actions on it until they resume it."
        )
    closed = item.status in _CLOSED_ITEM and tool.name not in _ALLOWED_WHEN_CLOSED
    if closed and not tool.read_only:
        return Reject(
            f"The item is {item.status.value}, so {tool.name} is no longer allowed. "
            "You may only remember lessons or escalate to the human."
        )
    verdict = tool.check(ctx, args)
    if not isinstance(verdict, Allow):
        return verdict
    if tool.trust_action and ctx.actor == "agent":
        platform = tool.trust_platform(ctx, args)
        if current_tier(ctx.session, platform, tool.trust_action) == TrustTier.ASK:
            return NeedsApproval(
                f"{tool.trust_action} on {platform} needs the owner's approval at the current "
                "trust level. It has been sent to them; do not retry it."
            )
    return Allow()


def validate_courtesy(ctx: ToolContext, tool: Tool, args: BaseModel) -> Verdict:
    """The system's one courtesy reply to a buyer writing about a SOLD or WITHDRAWN item.

    | # | Check | Verdict |
    |---|---|---|
    | 1 | Actor "system", tool send_message | Reject |
    | 2 | Item exists, not paused, SOLD or WITHDRAWN | Reject |
    | 3 | Body is exactly courtesy_reply(item.status) | Reject |
    | 4 | Thread had no courtesy reply yet; not had_deal (the owner answers those buyers) | Reject |
    | 5 | send_message check: frozen, flagged buyer, outbound cap, streak, text | As returned |

    No trust tier: the actor is never the agent.
    """
    if ctx.actor != "system" or not isinstance(tool, SendMessageTool):
        return Reject("Only the system's courtesy reply may use this path.")
    assert isinstance(args, SendMessageArgs)
    item = ctx.session.get(Item, ctx.item_id)
    if item is None:
        return Reject(f"Item {ctx.item_id} does not exist.")
    if item.paused:
        return Reject("The item is paused by the human; nothing is sent about it.")
    expected = courtesy_reply(item.status)
    if expected is None:
        return Reject(f"The item is {item.status.value}, not sold or withdrawn.")
    if args.body != expected:
        return Reject(f"On a {item.status.value} item only {expected!r} may be sent.")
    thread = ctx.session.get(Thread, args.thread_id)
    if thread is not None and thread.item_id == item.id:
        if had_courtesy_reply(ctx.session, args.thread_id):
            return Reject(f"Thread {thread.id} already had its courtesy reply.")
        if had_deal(ctx.session, thread):
            return Reject(
                f"The buyer in thread {thread.id} had a deal on this item; the owner answers them."
            )
    return tool.check(ctx, args)


def after_execute(ctx: ToolContext, tool: Tool, args: BaseModel, result: ToolResult) -> None:
    """Count a successful agent action toward trust promotion. Tell the human about it when its
    tier before this execution was ACT_AND_NOTIFY; accept_offer and complete_sale send their own
    notice."""
    if ctx.actor != "agent" or not result.ok or not tool.trust_action:
        return
    platform = tool.trust_platform(ctx, args)
    before = tier(ctx.session, ctx.clock, platform, tool.trust_action)
    record_executed(ctx.session, ctx.clock, ctx.settings, platform, tool.trust_action)
    if before == TrustTier.ACT_AND_NOTIFY and tool.name not in _OWN_NOTICE:
        ctx.notifier.notify(
            Notice(
                kind=NoticeKind.INFO,
                title=_notice_title(ctx, tool),
                body=result.summary,
                item_id=ctx.item_id,
            )
        )


def _notice_title(ctx: ToolContext, tool: Tool) -> str:
    """Reads like the Deal and Sold notices, e.g. "Pickup booked: IKEA desk"."""
    item = ctx.session.get(Item, ctx.item_id)
    label = _NOTICE_TITLES.get(tool.name, tool.name.replace("_", " ").capitalize())
    name = (item.title or item.hint) if item is not None else ""
    return f"{label}: {name}" if name else label
