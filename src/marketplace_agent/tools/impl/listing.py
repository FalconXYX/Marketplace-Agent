"""Listing tools: rewrite_listing, publish, relist.

Publish adds a Listing row per attempt, so a FAILED or DELISTED row stays as history and the
newest row on a platform is its current state. Relist reuses the newest row and keeps the
replaced post's remote id on a history row (DELISTED, never published), so a buyer who wrote on
the old post just before the relist still maps to the item at ingest.

| Relist step fails | Result |
|---|---|
| Taking the old post down (auth) | Raised: nothing changes |
| Taking the old post down (ListingGone) | Already down: marked DELISTED, the fresh copy is posted |
| Taking the old post down (other) | Stops: the old post stays LIVE, nothing is reposted |
| Posting the fresh copy | The row is DELISTED: the item is not on that platform |

publish, relist, and rewrite_listing post the item's whole title and description, so each checks
all of it with check_listing_text, not only what changed: a floor raised since the text was
written can make old text illegal.
"""
from dataclasses import replace
from datetime import timedelta
from typing import ClassVar

from pydantic import BaseModel, Field
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import col, select

from marketplace_agent.core.items import get_goal, item_photos, touch
from marketplace_agent.core.models import Goal, Item, Listing, ListingStatus, Photo
from marketplace_agent.guard.invariants import MAX_LISTING_CHARS, publish_count_today
from marketplace_agent.notify.base import Notice, NoticeKind
from marketplace_agent.platforms.base import (
    AwaitingHuman,
    ListingPayload,
    PlatformAuthError,
    PlatformError,
)
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext, ToolResult, Verdict
from marketplace_agent.tools.impl.common import (
    NO_GOAL,
    ToolArgs,
    caps_reset_at,
    fmt_local,
    fmt_money,
    item_of,
    latest_listing,
    listing_payload,
    listing_text_problems,
    listing_trust_platform,
    mark_gone,
    missing_condition,
    not_listed,
    push_to_live_listings,
    ref_for,
)

MAX_TITLE_CHARS = 100
RELIST_INTERVAL = timedelta(hours=24)
_IN_FLIGHT = (ListingStatus.PENDING, ListingStatus.LIVE, ListingStatus.AWAITING_HUMAN)


class RewriteListingArgs(ToolArgs):
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=MAX_LISTING_CHARS)
    photo_order: list[int] | None = None


class PlatformArgs(ToolArgs):
    platform: str


class RewriteListingTool(Tool):
    name: ClassVar[str] = "rewrite_listing"
    description: ClassVar[str] = (
        "Change the title, description, or photo order on every live listing. Give only what "
        "changes; photo_order lists every photo id, best first. No contact details, no address, "
        "and no price under the floor in the text."
    )
    Args: ClassVar[type[BaseModel]] = RewriteListingArgs
    trust_action: ClassVar[str | None] = "rewrite_listing"

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        return listing_trust_platform(ctx.session, ctx.item_id)

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, RewriteListingArgs)
        item = item_of(ctx)
        goal = get_goal(ctx.session, ctx.item_id)
        rejected = not_listed(item, "rewrite the listing") or (None if goal else NO_GOAL)
        if rejected:
            return rejected
        photo_ids = [photo.id for photo in item_photos(ctx.session, item.id)]
        if args.photo_order is not None and sorted(args.photo_order) != sorted(photo_ids):
            return Reject(
                f"photo_order must list each of this item's photo ids exactly once, in any "
                f"order: {photo_ids}. You gave {args.photo_order}."
            )
        for field, value in (("title", args.title), ("description", args.description)):
            if value is not None and not value.strip():
                return Reject(f"The {field} cannot be blank. Leave it out to keep the current one.")
        changes = _text_changes(item, args)
        if not changes and args.photo_order in (None, photo_ids):
            return Reject(
                "That is the listing exactly as it is. Give a different title, description, or "
                "photo_order, or skip this action."
            )
        reasons = listing_text_problems(
            ctx.settings,
            goal,
            title=changes.get("title", item.title),
            description=changes.get("description", item.description),
            unchanged=[name for name in ("title", "description") if name not in changes],
        )
        return Reject(" ".join(reasons)) if reasons else Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        """Pushes first, then stores, as send_message does: a write before the platform calls
        would hold the write lock for as long as a browser edit takes."""
        assert isinstance(args, RewriteListingArgs)
        item = item_of(ctx)
        goal = get_goal(ctx.session, ctx.item_id)
        assert goal is not None
        changes = _text_changes(item, args)
        payload = replace(
            listing_payload(
                ctx.session, ctx.settings, item, goal, photos=_in_order(ctx, args.photo_order)
            ),
            **changes,
        )
        pushed = push_to_live_listings(ctx, payload)  # every listing failed: nothing written
        for field, text in changes.items():
            setattr(item, field, text)
        changed = list(changes)
        if _reorder_photos(ctx, item, args.photo_order):
            changed.append("photo order")
        touch(item, ctx.clock)
        ctx.session.add(item)
        ctx.session.flush()
        summary = f"Rewrote the {', '.join(changed)}." + pushed.note("the old version")
        return ToolResult(
            ok=not pushed.errors,
            summary=summary,
            data={"errors": pushed.errors, "gone": pushed.gone},
        )


class PublishTool(Tool):
    name: ClassVar[str] = "publish"
    description: ClassVar[str] = (
        "Post the item on a platform where it is not listed yet, at the current asking price. "
        "Each platform has a daily posting limit; use relist, not publish, to refresh a listing."
    )
    Args: ClassVar[type[BaseModel]] = PlatformArgs
    trust_action: ClassVar[str | None] = "publish"

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        assert isinstance(args, PlatformArgs)
        return args.platform

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, PlatformArgs)
        item = item_of(ctx)
        rejected = (
            not_listed(item, "publish")
            or (None if get_goal(ctx.session, item.id) else NO_GOAL)
            or _unusable_platform(ctx, args.platform)
            or _no_condition(ctx, item, args.platform)
            or _already_listed(ctx, args.platform)
            or _publish_cap(ctx, args.platform)
        )
        if rejected:
            return rejected
        if not item_photos(ctx.session, item.id):
            return Reject("The item has no photos and a listing needs one. Escalate to the owner.")
        if not item.title.strip() or not item.description.strip():
            return Reject(
                "The item needs a title and a description before it can be posted. Write them "
                "with rewrite_listing first."
            )
        return _unpostable_text(ctx, item) or Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, PlatformArgs)
        item, goal = _item_and_goal(ctx)
        payload = listing_payload(ctx.session, ctx.settings, item, goal)
        listing = Listing(
            item_id=item.id,
            platform=args.platform,
            status=ListingStatus.PENDING,
            price_cents=goal.list_price_cents,
            title=item.title,
            description=item.description,
        )
        error = _post(ctx, item, listing, payload)  # before any write: no lock held meanwhile
        ctx.session.add(listing)
        if error is not None:
            listing.status = ListingStatus.FAILED
            ctx.session.flush()
            return ToolResult(
                ok=False,
                summary=f"Publishing on {args.platform} failed: {error}",
                data={"listing_id": listing.id, "error": str(error)},
            )
        if listing.status == ListingStatus.LIVE:
            listing.published_at = ctx.clock.now()
        ctx.session.flush()
        return ToolResult(
            ok=True, summary=_posted_summary(listing, "Published"), data=_listing_data(listing)
        )


class RelistTool(Tool):
    name: ClassVar[str] = "relist"
    description: ClassVar[str] = (
        "Take a listing down and post it again fresh, which puts it back at the top of search; "
        "also puts back a post that is gone from the platform. Allowed once 24 hours have passed "
        "since it was last posted; counts toward the daily posting limit."
    )
    Args: ClassVar[type[BaseModel]] = PlatformArgs
    trust_action: ClassVar[str | None] = "relist"

    def trust_platform(self, ctx: ToolContext, args: BaseModel) -> str:
        assert isinstance(args, PlatformArgs)
        return args.platform

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, PlatformArgs)
        item = item_of(ctx)
        rejected = (
            not_listed(item, "relist")
            or (None if get_goal(ctx.session, item.id) else NO_GOAL)
            or _unusable_platform(ctx, args.platform)
            or _no_condition(ctx, item, args.platform)
        )
        if rejected:
            return rejected
        listing = latest_listing(ctx.session, item.id, args.platform)
        if listing is None or listing.status not in (ListingStatus.LIVE, ListingStatus.DELISTED):
            state = f" (the latest one is {listing.status.value})" if listing else ""
            return Reject(
                f"There is no live or delisted listing on {args.platform} to relist{state}. "
                "Use publish to post it there."
            )
        return (
            _too_soon(ctx, listing)
            or _publish_cap(ctx, args.platform)
            or _unpostable_text(ctx, item)
            or Allow()
        )

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, PlatformArgs)
        item, goal = _item_and_goal(ctx)
        listing = latest_listing(ctx.session, item.id, args.platform)
        assert listing is not None
        payload = listing_payload(ctx.session, ctx.settings, item, goal)
        stuck = _take_down(ctx, listing)
        if stuck is not None:
            listing.last_error = f"could not take the old post down: {stuck}"
            ctx.session.flush()
            return ToolResult(
                ok=False,
                summary=(
                    f"Relisting on {args.platform} stopped: the old post {listing.remote_id} "
                    f"could not be taken down ({stuck}), so nothing was reposted; two copies "
                    "would split the buyers. It is still live. Try again later, or escalate if "
                    "it keeps failing."
                ),
                data={"listing_id": listing.id, "error": str(stuck)},
            )
        replaced = (listing.remote_id, listing.url, listing.price_cents, listing.title)
        error = _post(ctx, item, listing, payload)
        if error is not None:
            ctx.session.flush()
            summary = f"Relisting on {args.platform} failed: {error}."
            if listing.status == ListingStatus.DELISTED:
                summary += f" The old listing is down, so the item is not on {args.platform} now."
            return ToolResult(
                ok=False, summary=summary, data={"listing_id": listing.id, "error": str(error)}
            )
        _keep_replaced(ctx, listing, *replaced)
        if listing.status == ListingStatus.LIVE:
            listing.last_relisted_at = ctx.clock.now()
            listing.delisted_at = None
        _write_whole_state(listing)
        ctx.session.flush()
        return ToolResult(
            ok=True, summary=_posted_summary(listing, "Relisted"), data=_listing_data(listing)
        )


def _item_and_goal(ctx: ToolContext) -> tuple[Item, Goal]:
    goal = get_goal(ctx.session, ctx.item_id)
    assert goal is not None
    return item_of(ctx), goal


def _text_changes(item: Item, args: RewriteListingArgs) -> dict[str, str]:
    proposed = {"title": args.title, "description": args.description}
    return {
        field: text.strip()
        for field, text in proposed.items()
        if text is not None and text.strip() != getattr(item, field)
    }


def _in_order(ctx: ToolContext, order: list[int] | None) -> list[Photo] | None:
    """The item's photos in the given order (check() made sure it lists each once)."""
    if order is None:
        return None
    by_id = {photo.id: photo for photo in item_photos(ctx.session, ctx.item_id)}
    return [by_id[photo_id] for photo_id in order]


def _reorder_photos(ctx: ToolContext, item: Item, order: list[int] | None) -> bool:
    """Apply order as photo positions; False when it is missing or already the order."""
    photos = item_photos(ctx.session, item.id)
    if order is None or order == [photo.id for photo in photos]:
        return False
    position = {photo_id: index for index, photo_id in enumerate(order)}
    for photo in photos:
        photo.position = position[photo.id]
        ctx.session.add(photo)
    return True


def _unpostable_text(ctx: ToolContext, item: Item) -> Reject | None:
    """The stored title and description break a listing text rule, e.g. written before the
    owner raised the floor, or drafted with a phone number in it."""
    goal = get_goal(ctx.session, item.id)
    assert goal is not None
    reasons = listing_text_problems(
        ctx.settings, goal, title=item.title, description=item.description
    )
    if not reasons:
        return None
    fix = (
        "Fix it with rewrite_listing first."
        if ctx.actor == "agent"
        else "Fix the title or description first."
    )
    return Reject(f"The listing text cannot be posted as it is. {' '.join(reasons)} {fix}")


def _unusable_platform(ctx: ToolContext, platform: str) -> Reject | None:
    usable = [name for name in ctx.settings.enabled_platforms if name in ctx.platforms]
    if platform in usable:
        return None
    return Reject(
        f"{platform!r} is not an enabled, connected platform. Choose one of: "
        f"{', '.join(usable) or 'none (escalate)'}."
    )


def _no_condition(ctx: ToolContext, item: Item, platform: str) -> Reject | None:
    missing = missing_condition(item, platform)
    if missing is None:
        return None
    fix = (
        "Escalate so the owner sets it (Condition, under Controls on the item page)."
        if ctx.actor == "agent"
        else "Set it under Controls on the item page first."
    )
    return Reject(f"{missing} {fix}")


def _already_listed(ctx: ToolContext, platform: str) -> Reject | None:
    query = (
        select(Listing)
        .where(
            Listing.item_id == ctx.item_id,
            Listing.platform == platform,
            col(Listing.status).in_(_IN_FLIGHT),
        )
        .order_by(col(Listing.id).desc())
    )
    listing = ctx.session.exec(query).first()
    if listing is None:
        return None
    if listing.status == ListingStatus.LIVE:
        return Reject(
            f"The item is already live on {platform}. Use relist to refresh it, or update_price "
            "or rewrite_listing to change it."
        )
    if listing.status == ListingStatus.AWAITING_HUMAN:
        return Reject(
            f"A {platform} post is waiting for the owner to finish it. Do not publish again."
        )
    return Reject(f"A {platform} post is already in progress. Do not publish again.")


def _publish_cap(ctx: ToolContext, platform: str) -> Reject | None:
    cap = ctx.settings.daily_publish_cap.get(platform, 0)
    posted = publish_count_today(ctx.session, ctx.clock, platform)
    if posted < cap:
        return None
    return Reject(
        f"The daily limit of {cap} posts on {platform} is used up ({posted} today). Post nothing "
        f"more there until {caps_reset_at(ctx)}."
    )


def _too_soon(ctx: ToolContext, listing: Listing) -> Reject | None:
    posted = [at for at in (listing.published_at, listing.last_relisted_at) if at is not None]
    if not posted:
        return None
    last = max(posted)
    if ctx.clock.now() - last >= RELIST_INTERVAL:
        return None
    return Reject(
        f"The {listing.platform} listing was last posted {fmt_local(ctx.settings, last)}, and a "
        "repost within 24 hours counts as a duplicate. Relist no earlier than "
        f"{fmt_local(ctx.settings, last + RELIST_INTERVAL)}."
    )


def _take_down(ctx: ToolContext, listing: Listing) -> PlatformError | None:
    """Delist the old post before reposting; the error when it may still be up. ListingGone
    means it is already down. Any other failure might leave the old post live, and a repost
    would then put two copies up, with buyers on the old one mapped to nothing."""
    if listing.status != ListingStatus.LIVE:
        return None
    try:
        ctx.platforms[listing.platform].delist(ref_for(listing))
    except ListingGone as exc:
        mark_gone(ctx.clock, listing, exc)
        return None
    except PlatformAuthError:
        raise
    except PlatformError as exc:
        return exc
    listing.status = ListingStatus.DELISTED
    listing.delisted_at = ctx.clock.now()
    return None


def _write_whole_state(listing: Listing) -> None:
    """Write the row's state even where it ends as loaded: LIVE, then DELISTED, then LIVE again
    is no change to SQLAlchemy. Another session (the owner's mark_sold) may have marked the old
    post DELISTED meanwhile; left out of the UPDATE, that status would stick to the fresh post,
    and a live post on a sold item would never be owed a takedown."""
    for column in ("status", "remote_id", "url", "delisted_at", "last_error"):
        flag_modified(listing, column)


def _keep_replaced(
    ctx: ToolContext,
    listing: Listing,
    remote_id: str | None,
    url: str | None,
    price_cents: int,
    title: str,
) -> None:
    """A history row for the post a relist replaced, so ingest still maps its threads."""
    if remote_id is None or remote_id == listing.remote_id:
        return
    ctx.session.add(
        Listing(
            item_id=listing.item_id,
            platform=listing.platform,
            remote_id=remote_id,
            url=url,
            status=ListingStatus.DELISTED,
            price_cents=price_cents,
            title=title,
            description="",
            delisted_at=ctx.clock.now(),
            last_error=f"replaced by a relist of listing {listing.id}",
        )
    )


def _post(
    ctx: ToolContext, item: Item, listing: Listing, payload: ListingPayload
) -> PlatformError | None:
    """Publish payload into listing: LIVE, or AWAITING_HUMAN in assist mode. On failure, records
    the error and returns it; the caller decides the status."""
    try:
        ref = ctx.platforms[listing.platform].publish(payload)
    except AwaitingHuman:
        listing.status = ListingStatus.AWAITING_HUMAN
        listing.remote_id = listing.url = None
        ctx.notifier.notify(
            Notice(
                kind=NoticeKind.ALERT,
                title=f"Finish posting on {listing.platform}",
                body=f"{item.title} is filled in on {listing.platform}. Click Post to publish it.",
                item_id=item.id,
            )
        )
    except PlatformError as exc:
        listing.last_error = str(exc)
        return exc
    else:
        taken = _known_post(ctx, listing.platform, ref.remote_id)
        if taken is not None:
            listing.last_error = str(taken)
            return taken
        listing.status = ListingStatus.LIVE
        listing.remote_id, listing.url = ref.remote_id, ref.url
        listing.last_error = None
    listing.price_cents = payload.price_cents
    listing.title, listing.description = payload.title, payload.description
    return None


def _known_post(ctx: ToolContext, platform: str, remote_id: str) -> PlatformError | None:
    """The platform answered with the id of a post already on record (one of the owner's other
    listings, or the post a relist just took down): the new post was not confirmed, and taking
    that id would mix up two items' buyers and take the wrong post down later."""
    query = select(Listing).where(Listing.platform == platform, Listing.remote_id == remote_id)
    with ctx.session.no_autoflush:
        known = ctx.session.exec(query).first()
    if known is None:
        return None
    return PlatformError(
        f"{platform} answered with post {remote_id}, which is already item #{known.item_id}'s "
        f"listing, so the new post was not confirmed. Check {platform} before posting again."
    )


def _posted_summary(listing: Listing, verb: str) -> str:
    if listing.status == ListingStatus.AWAITING_HUMAN:
        return f"Prepared on {listing.platform}; the owner must click Post to finish it."
    where = f": {listing.url}" if listing.url else ""
    return f"{verb} on {listing.platform} at {fmt_money(listing.price_cents)}{where}"


def _listing_data(listing: Listing) -> dict[str, object]:
    return {"listing_id": listing.id, "status": listing.status.value, "url": listing.url}
