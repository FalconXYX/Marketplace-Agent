"""update_price: move the asking price on the goal and on every live listing.

The firm-price and once-per-24h rules bind the agent only; the owner may reprice at will. A
price change re-posts the listing's title and description, so while a listing is live that text
must pass check_listing_text at the new price, for the owner too.

| Step | Write lock |
|---|---|
| Record the change and check the goal again | held, then committed |
| Push to every live listing (a browser edit takes a minute) | not held |
| Mirror each listing's result | held briefly; the caller commits |
| Every listing refused: put the old price back | held briefly, committed |
"""
from datetime import timedelta
from typing import ClassVar

from pydantic import BaseModel, Field
from sqlmodel import col, select

from marketplace_agent.core.items import get_goal, touch
from marketplace_agent.core.models import Firmness, Goal, PriceChange
from marketplace_agent.guard.ledger import LedgerConflict, claim_item
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext, ToolResult, Verdict
from marketplace_agent.tools.impl.common import (
    NO_GOAL,
    ToolArgs,
    fmt_local,
    fmt_money,
    goal_at_price,
    item_of,
    listing_payload,
    listing_text_problems,
    listing_trust_platform,
    live_listings,
    not_listed,
    push_to_live_listings,
)

PRICE_CHANGE_INTERVAL = timedelta(hours=24)


class UpdatePriceArgs(ToolArgs):
    price_cents: int = Field(gt=0)


class UpdatePriceTool(Tool):
    name: ClassVar[str] = "update_price"
    description: ClassVar[str] = (
        "Change the asking price on every live listing, in cents. Use it when traffic or "
        "interest says the price is wrong. Never below the floor, never down when the price is "
        "firm, and at most once per 24 hours."
    )
    Args: ClassVar[type[BaseModel]] = UpdatePriceArgs
    trust_action: ClassVar[str | None] = "update_price"

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        return listing_trust_platform(ctx.session, ctx.item_id)

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, UpdatePriceArgs)
        goal = get_goal(ctx.session, ctx.item_id)
        if goal is None:
            return NO_GOAL
        rejected = not_listed(item_of(ctx), "change the price")
        if rejected:
            return rejected
        price = args.price_cents
        if price < goal.floor_price_cents:
            return Reject(
                f"{fmt_money(price)} is below the floor of {fmt_money(goal.floor_price_cents)}. "
                f"Choose {fmt_money(goal.floor_price_cents)} or more."
            )
        if price == goal.list_price_cents:
            return Reject(f"The price is already {fmt_money(price)}. Choose a different price.")
        text = _text_problems(ctx, goal, price)
        if text:
            fix = (
                "Fix the text with rewrite_listing first, leaving prices out of it (the price "
                "shows on its own), then change the price."
                if ctx.actor == "agent"
                else "Fix the title or description first, then change the price."
            )
            return Reject(
                f"Changing the price re-posts the listing text, which breaks a rule. {text} {fix}"
            )
        if ctx.actor == "human":
            return Allow()
        if goal.firmness == Firmness.FIRM and price < goal.list_price_cents:
            return Reject(
                f"The price is firm at {fmt_money(goal.list_price_cents)}: you may raise it but "
                "never lower it. Keep it, or escalate if you think the owner should lower it."
            )
        return _too_soon(ctx) or Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        """Commits the new price before the platform calls, together with anything else the
        session holds (set_goal's other fields): a floor set meanwhile is checked against it.
        When every listing refuses it, the old price is put back and PlatformError raised."""
        assert isinstance(args, UpdatePriceArgs)
        item = item_of(ctx)
        goal = get_goal(ctx.session, ctx.item_id)
        assert goal is not None
        old = goal.list_price_cents
        change = _record_change(ctx, goal, args.price_cents)  # its flush takes the write lock
        _recheck(ctx, goal, args.price_cents, old)
        touch(item, ctx.clock)
        payload = listing_payload(ctx.session, ctx.settings, item, goal)
        change_id = change.id
        assert change_id is not None
        ctx.session.commit()  # no write lock held during the platform calls
        try:
            pushed = push_to_live_listings(ctx, payload)
        except PlatformError:
            _take_back(ctx, change_id, old, args.price_cents)
            raise
        summary = f"Price changed from {fmt_money(old)} to {fmt_money(args.price_cents)}."
        summary += pushed.note("the old price")
        return ToolResult(
            ok=not pushed.errors,
            summary=summary,
            data={"errors": pushed.errors, "gone": pushed.gone},
        )


def _recheck(ctx: ToolContext, goal: Goal, price: int, old: int) -> None:
    """check() ran before the write lock: the owner may have raised the floor or made the price
    firm since. Read the goal again under the lock, before any listing changes; a conflict rolls
    the change back."""
    ctx.session.refresh(goal)
    if price < goal.floor_price_cents:
        raise LedgerConflict(
            f"The owner just set the floor to {fmt_money(goal.floor_price_cents)}, so "
            f"{fmt_money(price)} is below it now. Choose {fmt_money(goal.floor_price_cents)} or "
            "more."
        )
    if ctx.actor != "human" and goal.firmness == Firmness.FIRM and price < old:
        raise LedgerConflict(
            f"The owner just made the price firm at {fmt_money(old)}: you may raise it but never "
            "lower it."
        )
    text = _text_problems(ctx, goal, price)
    if text:
        raise LedgerConflict(
            f"The owner just changed the goal, and the listing text now breaks a rule. {text} "
            "Fix it with rewrite_listing before changing the price."
        )


def _text_problems(ctx: ToolContext, goal: Goal, price: int) -> str:
    """Rules the item's text breaks at price, while a live listing would re-post it."""
    if not live_listings(ctx.session, ctx.item_id):
        return ""
    item = item_of(ctx)
    return " ".join(
        listing_text_problems(
            ctx.settings,
            goal_at_price(goal, price),
            title=item.title,
            description=item.description,
        )
    )


def _too_soon(ctx: ToolContext) -> Reject | None:
    since = ctx.clock.now() - PRICE_CHANGE_INTERVAL
    query = (
        select(PriceChange)
        .where(PriceChange.item_id == ctx.item_id, col(PriceChange.created_at) > since)
        .order_by(col(PriceChange.created_at).desc())
    )
    last = ctx.session.exec(query).first()
    if last is None:
        return None
    until = fmt_local(ctx.settings, last.created_at + PRICE_CHANGE_INTERVAL)
    return Reject(
        f"The price last changed {fmt_local(ctx.settings, last.created_at)}; it may change once "
        f"per 24 hours. Leave it until {until}."
    )


def _record_change(ctx: ToolContext, goal: Goal, new_cents: int) -> PriceChange:
    change = PriceChange(
        item_id=ctx.item_id,
        old_cents=goal.list_price_cents,
        new_cents=new_cents,
        actor=ctx.actor,
        created_at=ctx.clock.now(),
    )
    ctx.session.add(change)
    goal.list_price_cents = new_cents
    ctx.session.add(goal)
    ctx.session.flush()
    return change


def _take_back(ctx: ToolContext, change_id: int, old: int, new: int) -> None:
    """No listing took the new price: restore the old one and drop the change, so nothing
    changed anywhere. A goal changed meanwhile (another price, or a floor above the old price)
    is left as it is."""
    ctx.session.rollback()  # the per-listing errors noted during the push
    claim_item(ctx.session, ctx.clock, ctx.item_id)
    goal = get_goal(ctx.session, ctx.item_id)
    assert goal is not None
    ctx.session.refresh(goal)
    if goal.list_price_cents == new and goal.floor_price_cents <= old:
        goal.list_price_cents = old
        ctx.session.add(goal)
        change = ctx.session.get(PriceChange, change_id)
        if change is not None:
            ctx.session.delete(change)
    ctx.session.commit()
