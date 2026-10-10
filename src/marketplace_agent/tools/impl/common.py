"""Helpers shared by the tool implementations."""
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict
from sqlalchemy import ColumnElement, and_
from sqlmodel import Session, col, select

from marketplace_agent.config import Settings
from marketplace_agent.core.items import get_attributes, get_item, item_photos
from marketplace_agent.core.models import (
    Buyer,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Photo,
    Thread,
    ThreadState,
)
from marketplace_agent.core.clock import Clock
from marketplace_agent.guard.invariants import check_listing_text, to_local
from marketplace_agent.intake.photos import photo_file
from marketplace_agent.notify.base import Notice, NoticeKind
from marketplace_agent.platforms.base import ListingPayload, PlatformError, PlatformRef
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.tools.base import Reject, ToolContext

NEEDS_CONDITION = frozenset({"facebook"})  # its form has no "not sure"; the adapter refuses

NO_GOAL = Reject(
    "This item has no price goal (list price and floor) yet, and only the owner can set one. "
    "Escalate."
)


class ToolArgs(BaseModel):
    """Unknown keys are an error, so a misspelled optional argument is reported, not ignored."""

    model_config = ConfigDict(extra="forbid")


def item_of(ctx: ToolContext) -> Item:
    return get_item(ctx.session, ctx.item_id)


def thread_for_item(ctx: ToolContext, thread_id: int) -> Thread | Reject:
    thread = ctx.session.get(Thread, thread_id)
    if thread is None or thread.item_id != ctx.item_id:
        return Reject(
            f"Thread {thread_id} is not a conversation about this item. Use a thread id from "
            "the Threads section of your context."
        )
    return thread


def unsafe_thread(ctx: ToolContext, thread: Thread, action: str) -> Reject | None:
    """Reject a frozen thread or a flagged buyer. action completes "Never <action>"."""
    if thread.state == ThreadState.FROZEN:
        why = f" ({thread.frozen_reason})" if thread.frozen_reason else ""
        problem = f"Thread {thread.id} is frozen{why}"
    elif _flagged(ctx.session, thread.buyer_id):
        problem = f"The buyer in thread {thread.id} is flagged as a scammer"
    else:
        return None
    return Reject(f"{problem}. Never {action}; escalate if the owner should look at it.")


def _flagged(session: Session, buyer_id: int) -> bool:
    buyer = session.get(Buyer, buyer_id)
    return buyer is not None and buyer.flagged_scam


def buyer_name(session: Session, buyer_id: int | None) -> str:
    buyer = session.get(Buyer, buyer_id) if buyer_id is not None else None
    if buyer is None:
        return "the buyer"
    return buyer.display_name or buyer.handle


def not_listed(item: Item, action: str) -> Reject | None:
    """Reject unless the item is LISTED. action completes "You can <action> only ..."."""
    if item.status == ItemStatus.LISTED:
        return None
    if item.status == ItemStatus.DRAFT:
        return Reject(
            f"The item is still a draft that the owner has not confirmed for sale. You can "
            f"{action} only once it is listed; escalate if it should be."
        )
    if item.status == ItemStatus.COMMITTED:
        return Reject(
            f"The item is promised to a buyer. You can {action} only if that deal is released."
        )
    return Reject(f"The item is {item.status.value}; there is nothing more to {action}.")


def missing_condition(item: Item, platform: str) -> str | None:
    """Why the item cannot be posted on platform as it is, for want of a condition."""
    if item.condition or platform not in NEEDS_CONDITION:
        return None
    return f"{platform.capitalize()} needs the item's condition, and none is set."


def listing_payload(
    session: Session,
    settings: Settings,
    item: Item,
    goal: Goal,
    *,
    photos: Sequence[Photo] | None = None,
) -> ListingPayload:
    """What a listing of the item shows; photos, when given, in place of the stored order."""
    shown = item_photos(session, item.id) if photos is None else photos
    return ListingPayload(
        title=item.title,
        description=item.description,
        price_cents=goal.list_price_cents,
        photo_paths=[photo_file(settings, p).resolve() for p in shown],
        location=settings.pickup_area,
        category=item.category,
        condition=item.condition,
        attributes=get_attributes(item),
    )


def live_listings(session: Session, item_id: int) -> list[Listing]:
    query = (
        select(Listing)
        .where(Listing.item_id == item_id, Listing.status == ListingStatus.LIVE)
        .order_by(col(Listing.id))
    )
    return list(session.exec(query).all())


def listing_trust_platform(session: Session, item_id: int) -> str:
    """Trust key for a change pushed to every live listing: "facebook" while one is live there,
    since Facebook edits start gated, else "*"."""
    live = live_listings(session, item_id)
    return "facebook" if any(listing.platform == "facebook" for listing in live) else "*"


def latest_listing(session: Session, item_id: int, platform: str) -> Listing | None:
    """Publish adds a row per attempt, so the newest row is the platform's current state.
    History rows, which only remember a post a relist replaced, do not count."""
    query = (
        select(Listing)
        .where(Listing.item_id == item_id, Listing.platform == platform, ~replaced_post())
        .order_by(col(Listing.id).desc())
    )
    return session.exec(query).first()


def replaced_post() -> ColumnElement[bool]:
    """A history row a relist leaves for the post it took down: DELISTED and never published.
    Every row that went up has published_at, so only these match."""
    return and_(Listing.status == ListingStatus.DELISTED, col(Listing.published_at).is_(None))


def ref_for(listing: Listing) -> PlatformRef:
    if listing.remote_id is None:
        raise ValueError(f"Listing {listing.id} has no remote id; it was never published.")
    return PlatformRef(listing.platform, listing.remote_id, listing.url)


@dataclass
class Pushed:
    errors: list[str] = field(default_factory=list)  # listings still showing the old version
    gone: list[str] = field(default_factory=list)  # platforms whose post had vanished

    def note(self, what: str) -> str:
        """Sentences to append to a tool summary; what is e.g. "the old price"."""
        parts = []
        if self.errors:
            parts.append(f" Some listings still show {what}: " + "; ".join(self.errors))
        for platform in self.gone:
            parts.append(
                f" The {platform} post is gone from {platform} (removed or expired), so the item "
                f"is not on {platform} now. Relist or publish on {platform} to put it back up."
            )
        return "".join(parts)


def push_to_live_listings(ctx: ToolContext, payload: ListingPayload) -> Pushed:
    """Send payload to every live listing of the item and mirror it on the rows that took it.

    Per-listing errors (also stored in listing.last_error) come back when only some fail. A post
    that is gone from its platform is marked DELISTED, reported in gone, and alerted; it is not
    a failure. Raises PlatformError when every listing that still exists fails, so the caller
    can undo its change and nothing has changed anywhere. Writes nothing until every platform
    call is done: call it with no write pending, and no write lock is held meanwhile.
    """
    listings = live_listings(ctx.session, ctx.item_id)
    pushed = Pushed()
    vanished: list[tuple[Listing, ListingGone]] = []
    with ctx.session.no_autoflush:
        for listing in listings:
            try:
                _update_remote(ctx, listing, payload)
            except ListingGone as exc:
                mark_gone(ctx.clock, listing, exc)
                vanished.append((listing, exc))
                continue
            except PlatformError as exc:
                listing.last_error = str(exc)
                pushed.errors.append(f"{listing.platform}: {exc}")
                continue
            listing.title = payload.title
            listing.description = payload.description
            listing.price_cents = payload.price_cents
            listing.last_error = None
    tried = len(listings) - len(vanished)
    if tried and len(pushed.errors) == tried:
        raise PlatformError("No listing could be updated. " + "; ".join(pushed.errors))
    item = get_item(ctx.session, ctx.item_id)
    for listing, exc in vanished:
        pushed.gone.append(listing.platform)
        ctx.notifier.notify(gone_notice(item, listing, exc))
    ctx.session.flush()
    return pushed


def mark_gone(clock: Clock, listing: Listing, why: object) -> None:
    """The post vanished from its platform: the row is DELISTED, which relist and publish accept."""
    listing.status = ListingStatus.DELISTED
    listing.delisted_at = clock.now()
    listing.last_error = f"gone from {listing.platform}: {why}"


def gone_notice(item: Item, listing: Listing, why: object) -> Notice:
    return Notice(
        kind=NoticeKind.ALERT,
        title=f"The {listing.platform} post is gone",
        body=f"{item.title or item.hint}: the {listing.platform} listing "
        f"({listing.url or listing.remote_id}) is no longer on {listing.platform} ({why}). It "
        "was removed, expired, or deleted by hand. The agent can relist it.",
        item_id=item.id,
    )


def listing_text_problems(
    settings: Settings,
    goal: Goal,
    *,
    title: str,
    description: str,
    unchanged: Collection[str] = (),
) -> list[str]:
    """check_listing_text on the text a listing would carry, by field. unchanged names fields
    kept from before, so the reason says to rewrite those too."""
    reasons = []
    for name, text in (("title", title), ("description", description)):
        where = f"the current {name} (rewrite it too)" if name in unchanged else f"the {name}"
        reasons += [f"In {where}: {r}" for r in check_listing_text(settings, goal=goal, text=text)]
    return reasons


def goal_at_price(goal: Goal, price_cents: int) -> Goal:
    """A detached copy of goal with another list price, to check text against."""
    return Goal(
        item_id=goal.item_id,
        list_price_cents=price_cents,
        target_price_cents=goal.target_price_cents,
        floor_price_cents=goal.floor_price_cents,
        deadline=goal.deadline,
        firmness=goal.firmness,
    )


def _update_remote(ctx: ToolContext, listing: Listing, payload: ListingPayload) -> None:
    adapter = ctx.platforms.get(listing.platform)
    if adapter is None:
        raise PlatformError(f"{listing.platform} is not connected")
    adapter.update(ref_for(listing), payload)


def caps_reset_at(ctx: ToolContext) -> str:
    """When the daily outbound and publish counts restart (midnight UTC), in local time."""
    now = ctx.clock.now().astimezone(UTC)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return fmt_local(ctx.settings, midnight)


def fmt_money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def fmt_local(settings: Settings, dt: datetime) -> str:
    """Seller-local time for reasons and summaries, e.g. "Thu Oct 1 18:00"."""
    local = to_local(settings, dt)
    return f"{local:%a %b} {local.day} {local:%H:%M}"
