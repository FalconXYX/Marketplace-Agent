"""Deal tools: accept_offer, accept_bundle, book_pickup, release_commitment, complete_sale.

Each check mirrors the ledger's rules so the model gets a reason; the ledger re-checks under
LEDGER_LOCK, so a race raises LedgerConflict and the agent loop rolls the action back.

A bundle (accept_bundle) is one deal over several items. book_pickup, release_commitment and
complete_sale on any of its threads act on the whole bundle, from the context of any of its
items, and refuse while another of its items is paused (except for the owner).

Once a booked pickup has started, the item may already be with the buyer, paid for in person,
and a buyer who came often never writes again. So the agent's release then needs evidence
(guard/handover.py): the buyer wrote since the pickup started (not for a no-show), or the owner,
asked when the pickup ended, has had until no_show_from to say otherwise.
"""
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from sqlmodel import col, select

from marketplace_agent.core.items import get_goal, get_item
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    Direction,
    Firmness,
    Item,
    ItemStatus,
    ListingStatus,
    Message,
    Thread,
    ThreadState,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.handover import no_show_from
from marketplace_agent.guard.invariants import (
    availability_text,
    outbound_streak,
    parse_local_datetime,
    redact_address,
    to_local,
    within_availability,
)
from marketplace_agent.guard.ledger import BUNDLE_MAX_ITEMS, BUNDLE_MIN_ITEMS, ActiveCommitment
from marketplace_agent.notify.base import Notice, NoticeKind
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext, ToolResult, Verdict
from marketplace_agent.tools.impl.common import (
    NO_GOAL,
    ToolArgs,
    buyer_name,
    fmt_local,
    fmt_money,
    item_of,
    live_listings,
    mark_gone,
    not_listed,
    ref_for,
    thread_for_item,
    unsafe_thread,
)
from marketplace_agent.tools.impl.messaging import FOLLOW_UP_AFTER

COMPLETE_EARLIEST = timedelta(hours=1)  # before pickup start


class AcceptOfferArgs(ToolArgs):
    thread_id: int
    price_cents: int = Field(gt=0)


class AcceptBundleArgs(ToolArgs):
    thread_id: int
    item_ids: list[int] = Field(min_length=BUNDLE_MIN_ITEMS, max_length=BUNDLE_MAX_ITEMS)
    total_cents: int = Field(gt=0)


class BookPickupArgs(ToolArgs):
    thread_id: int
    start: str
    duration_minutes: int = Field(default=60, ge=15, le=240)


class ReleaseCommitmentArgs(ToolArgs):
    thread_id: int
    reason: str
    no_show: bool = False


class ThreadArgs(ToolArgs):
    thread_id: int


class AcceptOfferTool(Tool):
    name: ClassVar[str] = "accept_offer"
    description: ClassVar[str] = (
        "Promise the item to the buyer in a thread at an agreed price, in cents. This locks the "
        "item to them, so do it only when they have clearly agreed to a price at or above the "
        "floor (at or above the list price when firm). Then book a pickup."
    )
    Args: ClassVar[type[BaseModel]] = AcceptOfferArgs
    trust_action: ClassVar[str | None] = "accept_offer"

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, AcceptOfferArgs)
        thread = thread_for_item(ctx, args.thread_id)
        if isinstance(thread, Reject):
            return thread
        rejected = unsafe_thread(ctx, thread, "make a deal with this buyer")
        if rejected:
            return rejected
        if not _buyer_has_written(ctx, thread):
            return Reject(
                "This buyer has not written anything yet. Accept only an offer a buyer has "
                "actually made."
            )
        current = ledger.active_commitment(ctx.session, ctx.item_id)
        if current is not None:
            accepted = current.accepted
            return Reject(
                f"The item is already promised to {buyer_name(ctx.session, accepted.buyer_id)} "
                f"(thread {accepted.thread_id}) at {fmt_money(accepted.price_cents or 0)}. Tell "
                "other buyers it is pending; release that deal first only if it fell through."
            )
        rejected = not_listed(item_of(ctx), "accept an offer")
        return rejected or _price_reason(ctx, args.price_cents) or Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, AcceptOfferArgs)
        thread = _thread(ctx, args.thread_id)
        accepted = ledger.accept(
            ctx.session,
            ctx.clock,
            item_id=ctx.item_id,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            price_cents=args.price_cents,
            undo_minutes=ctx.settings.undo_window_minutes,
            actor=ctx.actor,
        )
        assert accepted.undo_deadline is not None
        name = buyer_name(ctx.session, thread.buyer_id)
        price = fmt_money(args.price_cents)
        until = fmt_local(ctx.settings, accepted.undo_deadline)
        ctx.notifier.notify(
            Notice(
                kind=NoticeKind.COMMITMENT,
                title=f"Deal: {item_of(ctx).title}",
                body=f"Promised to {name} for {price}. Undo before {until} to cancel it.",
                item_id=ctx.item_id,
                ref_id=accepted.id,
                expires_at=accepted.undo_deadline,
            )
        )
        summary = (
            f"Accepted {price} from {name}; the item is promised to them. The owner can undo "
            f"until {until}."
        )
        return ToolResult(ok=True, summary=summary, data={"commitment_id": accepted.id})


class AcceptBundleTool(Tool):
    name: ClassVar[str] = "accept_bundle"
    description: ClassVar[str] = (
        "Promise 2 to 5 of the owner's items, this one included, to one buyer for one agreed "
        "total in cents. The total is split by list price, never below an item's floor (its "
        "list price when firm). Only when the buyer clearly agreed to that total for exactly "
        "these items. Then book one pickup for all of them."
    )
    Args: ClassVar[type[BaseModel]] = AcceptBundleArgs
    trust_action: ClassVar[str | None] = "accept_offer"

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, AcceptBundleArgs)
        thread = thread_for_item(ctx, args.thread_id)
        if isinstance(thread, Reject):
            return thread
        rejected = unsafe_thread(ctx, thread, "make a deal with this buyer")
        if rejected:
            return rejected
        if not _buyer_has_written(ctx, thread):
            return Reject(
                "This buyer has not written anything yet. Accept only an offer a buyer has "
                "actually made."
            )
        ids = args.item_ids
        if ctx.item_id not in ids:
            return Reject(
                f"item_ids must include this item ({ctx.item_id}): a bundle is agreed in the "
                "thread of one of its items."
            )
        if len(set(ids)) != len(ids):
            return Reject("List each item once in item_ids.")
        for item_id in ids:
            rejected = _bundle_item_reason(ctx, thread, item_id)
            if rejected:
                return rejected
        return _bundle_total_reason(ctx, ids, args.total_cents) or Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, AcceptBundleArgs)
        thread = _thread(ctx, args.thread_id)
        rows = ledger.accept_bundle(
            ctx.session,
            ctx.clock,
            item_ids=args.item_ids,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            total_cents=args.total_cents,
            undo_minutes=ctx.settings.undo_window_minutes,
            actor=ctx.actor,
        )
        own = next(row for row in rows if row.item_id == ctx.item_id)
        assert own.undo_deadline is not None
        name = buyer_name(ctx.session, thread.buyer_id)
        total = fmt_money(args.total_cents)
        parts = _bundle_parts(ctx, rows)
        until = fmt_local(ctx.settings, own.undo_deadline)
        ctx.notifier.notify(
            Notice(
                kind=NoticeKind.COMMITMENT,
                title="Bundle deal: " + ", ".join(_title(ctx, row.item_id) for row in rows),
                body=f"Promised {parts} to {name} for {total} in total. Undo before {until} to "
                "cancel the whole bundle.",
                item_id=ctx.item_id,
                ref_id=own.id,
                expires_at=own.undo_deadline,
            )
        )
        summary = (
            f"Accepted {total} from {name} for {len(rows)} items: {parts}. They are all "
            f"promised to {name}; the owner can undo until {until}. Book one pickup for all "
            "of them."
        )
        return ToolResult(
            ok=True,
            summary=summary,
            data={
                "commitment_id": own.id,
                "commitment_ids": [row.id for row in rows],
                "bundle_key": own.bundle_key,
                "prices_cents": {str(row.item_id): row.price_cents for row in rows},
            },
        )


class BookPickupTool(Tool):
    name: ClassVar[str] = "book_pickup"
    description: ClassVar[str] = (
        "Book the pickup time agreed with the buyer of an accepted deal. start is local time, "
        "e.g. 2026-10-10T18:00. The whole pickup must fit inside the seller's availability and "
        "not overlap another booked pickup. Booking again reschedules."
    )
    Args: ClassVar[type[BaseModel]] = BookPickupArgs
    trust_action: ClassVar[str | None] = "book_pickup"

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, BookPickupArgs)
        thread = _deal_thread(ctx, args.thread_id)
        if isinstance(thread, Reject):
            return thread
        rejected = unsafe_thread(ctx, thread, "book a pickup with this buyer (release the deal)")
        if rejected:
            return rejected
        active = ledger.active_commitment_for_thread(ctx.session, thread.id)
        if active is None:
            return Reject(
                "There is no accepted deal in this thread. Accept an offer with accept_offer "
                "before booking a pickup."
            )
        rejected = _bundle_paused_reason(ctx, active, "book its pickup")
        if rejected:
            return rejected
        try:
            start, end = _window(ctx, args)
        except ValueError as exc:
            return Reject(f"{exc}. Write start as a local date and time: YYYY-MM-DDTHH:MM.")
        return _timing_reason(ctx, active, start, end) or Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, BookPickupArgs)
        active = _active_for(ctx, args.thread_id)
        start, end = _window(ctx, args)
        booked = ledger.book_pickup(
            ctx.session,
            ctx.clock,
            accepted_id=active.accepted.id,
            start=start,
            end=end,
            actor=ctx.actor,
        )
        name = buyer_name(ctx.session, active.accepted.buyer_id)
        rows = ledger.deal_rows(ctx.session, active.accepted)
        what = f" for {_titles(ctx, rows)}" if len(rows) > 1 else ""
        return ToolResult(
            ok=True,
            summary=f"Pickup booked with {name}{what}: {_span(ctx, start, end)}.",
            data={
                "commitment_id": active.accepted.id,
                "pickup_id": booked.id,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )


class ReleaseCommitmentTool(Tool):
    name: ClassVar[str] = "release_commitment"
    description: ClassVar[str] = (
        "Cancel the accepted deal in a thread so the item is available again, e.g. the buyer "
        "backed out. Set no_show true only when the buyer missed a booked pickup; it goes on "
        "their record, and after a pickup it needs the owner's answer or the time COMMITMENT gives."
    )
    Args: ClassVar[type[BaseModel]] = ReleaseCommitmentArgs
    trust_action: ClassVar[str | None] = "release_commitment"

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, ReleaseCommitmentArgs)
        thread = _deal_thread(ctx, args.thread_id)
        if isinstance(thread, Reject):
            return thread
        active = ledger.active_commitment_for_thread(ctx.session, thread.id)
        if active is None:
            return Reject(f"There is no active deal in thread {thread.id} to release.")
        if not args.reason.strip():
            return Reject("Say in reason why the deal is being released.")
        return (
            _bundle_paused_reason(ctx, active, "release it")
            or _release_reason(ctx, active, args.no_show)
            or Allow()
        )

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, ReleaseCommitmentArgs)
        active = _active_for(ctx, args.thread_id)
        rows = ledger.deal_rows(ctx.session, active.accepted)
        ledger.release(
            ctx.session,
            ctx.clock,
            accepted_id=active.accepted.id,
            reason=args.reason.strip(),
            actor=ctx.actor,
        )
        buyer = ctx.session.get(Buyer, active.accepted.buyer_id)
        if args.no_show and buyer is not None:
            buyer.no_show_count += 1
            ctx.session.add(buyer)
            ctx.session.flush()
        name = buyer_name(ctx.session, active.accepted.buyer_id)
        what = "Marked a no-show and released" if args.no_show else "Released"
        if len(rows) > 1:
            summary = (
                f"{what} the bundle deal with {name} ({_titles(ctx, rows)}); the items are "
                "available again."
            )
        else:
            summary = f"{what} the deal with {name}; the item is available again."
        data: dict[str, Any] = {"commitment_id": active.accepted.id}
        if len(rows) > 1:
            data["commitment_ids"] = [row.id for row in rows]
        return ToolResult(ok=True, summary=summary, data=data)


class CompleteSaleTool(Tool):
    name: ClassVar[str] = "complete_sale"
    description: ClassVar[str] = (
        "Record that the buyer collected the item and paid, which marks it (every item of a "
        "bundle) sold and takes every listing down. Only for a deal with a booked pickup, once "
        "that pickup is under way and the buyer or owner confirms the handover."
    )
    Args: ClassVar[type[BaseModel]] = ThreadArgs
    trust_action: ClassVar[str | None] = "complete_sale"

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, ThreadArgs)
        thread = _deal_thread(ctx, args.thread_id)
        if isinstance(thread, Reject):
            return thread
        active = ledger.active_commitment_for_thread(ctx.session, thread.id)
        if active is None:
            return Reject(f"There is no active deal in thread {thread.id} to complete.")
        rejected = _bundle_paused_reason(ctx, active, "complete it")
        if rejected:
            return rejected
        pickup = active.pickup
        if pickup is None or pickup.pickup_start is None:
            return Reject(
                "No pickup is booked for this deal. Book one with book_pickup; complete the sale "
                "once the buyer has collected the item."
            )
        earliest = pickup.pickup_start - COMPLETE_EARLIEST
        if ctx.clock.now() < earliest:
            return Reject(
                f"The pickup is {fmt_local(ctx.settings, pickup.pickup_start)}. Complete the "
                f"sale only once the buyer has the item, no earlier than "
                f"{fmt_local(ctx.settings, earliest)}."
            )
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, ThreadArgs)
        active = _active_for(ctx, args.thread_id)
        accepted = active.accepted
        rows = ledger.deal_rows(ctx.session, accepted)
        ledger.complete(ctx.session, ctx.clock, accepted_id=accepted.id, actor=ctx.actor)
        _record_outcomes(ctx, [row.item_id for row in rows])
        items = [get_item(ctx.session, row.item_id) for row in rows]
        ctx.session.commit()  # the sale stands whatever the platforms say; no lock meanwhile
        errors = _delist_everywhere(ctx, items)
        name = buyer_name(ctx.session, accepted.buyer_id)
        total = fmt_money(sum(row.price_cents or 0 for row in rows))
        if len(rows) > 1:
            sold = f"Sold {_titles(ctx, rows)} to {name} for {total} in total."
            title = f"Sold: {_titles(ctx, rows)}"
        else:
            sold = f"Sold to {name} for {total}."
            title = f"Sold: {item_of(ctx).title}"
        if errors:
            sold += " Still up, remove by hand: " + "; ".join(errors)
        ctx.notifier.notify(
            Notice(
                kind=NoticeKind.SOLD,
                title=title,
                body=sold,
                item_id=ctx.item_id,
                ref_id=accepted.id,
            )
        )
        data: dict[str, Any] = {"commitment_id": accepted.id, "errors": errors}
        if len(rows) > 1:
            data["commitment_ids"] = [row.id for row in rows]
        return ToolResult(ok=True, summary=sold, data=data)


def _deal_thread(ctx: ToolContext, thread_id: int) -> Thread | Reject:
    """thread_for_item, or the thread this item's bundle row is on (a bundle agreed in the
    thread of another of its items), so the deal can be run from any of its items."""
    active = ledger.active_commitment(ctx.session, ctx.item_id)
    if active is not None and active.accepted.thread_id == thread_id:
        thread = ctx.session.get(Thread, thread_id)
        if thread is not None:
            return thread
    return thread_for_item(ctx, thread_id)


def _thread(ctx: ToolContext, thread_id: int) -> Thread:
    thread = ctx.session.get(Thread, thread_id)
    assert thread is not None
    return thread


def _active_for(ctx: ToolContext, thread_id: int) -> ActiveCommitment:
    active = ledger.active_commitment_for_thread(ctx.session, thread_id)
    if active is None:
        raise ledger.LedgerConflict(f"Thread {thread_id} has no active deal any more.")
    return active


def _buyer_has_written(ctx: ToolContext, thread: Thread) -> bool:
    query = select(Message.id).where(
        Message.thread_id == thread.id, Message.direction == Direction.INBOUND
    )
    return ctx.session.exec(query).first() is not None


def _price_reason(ctx: ToolContext, price_cents: int) -> Reject | None:
    goal = get_goal(ctx.session, ctx.item_id)
    if goal is None:
        return NO_GOAL
    floor = fmt_money(goal.floor_price_cents)
    if price_cents < goal.floor_price_cents:
        return Reject(
            f"{fmt_money(price_cents)} is below the floor of {floor}. Counter at {floor} or "
            "more, or decline."
        )
    if goal.firmness == Firmness.FIRM and price_cents < goal.list_price_cents:
        firm = fmt_money(goal.list_price_cents)
        return Reject(f"The price is firm at {firm}. Accept {firm} or more, or decline.")
    return None


def _bundle_item_reason(ctx: ToolContext, thread: Thread, item_id: int) -> Reject | None:
    """Why item_id cannot go into a bundle with the buyer in thread, or None."""
    item = ctx.session.get(Item, item_id)
    if item is None:
        return Reject(
            f"There is no item {item_id}. Use item ids from the Also interested in section."
        )
    this = item_id == ctx.item_id
    label = "This item" if this else f"Item {item_id} ({item.title or item.hint})"
    leave_out = "" if this else " Leave it out of the bundle."
    if item.paused and ctx.actor != "human":
        return Reject(f"{label} is paused by the owner.{leave_out}")
    current = ledger.active_commitment(ctx.session, item_id)
    if current is not None:
        accepted = current.accepted
        promised = (
            f"{label} is already promised to {buyer_name(ctx.session, accepted.buyer_id)} "
            f"(thread {accepted.thread_id}) at {fmt_money(accepted.price_cents or 0)}."
        )
        return Reject(promised + " A bundle can replace that deal only once it is released.")
    if this:
        rejected = not_listed(item, "make a bundle deal")
        if rejected:
            return rejected
    elif item.status == ItemStatus.DRAFT:
        return Reject(f"{label} is a draft the owner has not confirmed for sale.{leave_out}")
    elif item.status != ItemStatus.LISTED:
        return Reject(f"{label} is {item.status.value}.{leave_out}")
    if get_goal(ctx.session, item_id) is None:
        return NO_GOAL if this else Reject(f"{label} has no price goal.{leave_out}")
    row_thread = ledger.bundle_item_thread(ctx.session, thread, item_id)
    if row_thread.state == ThreadState.FROZEN:
        return Reject(
            f"The buyer's thread {row_thread.id} about {label.lower() if this else label} is "
            "frozen. Never make a deal with this buyer; escalate if the owner should look at it."
        )
    return None


def _bundle_total_reason(
    ctx: ToolContext, item_ids: Sequence[int], total_cents: int
) -> Reject | None:
    goals = [get_goal(ctx.session, item_id) for item_id in item_ids]
    parts, need = [], 0
    for item_id, goal in zip(item_ids, goals, strict=True):
        assert goal is not None
        minimum = ledger.minimum_cents(goal)
        need += minimum
        rule = "firm at" if goal.firmness == Firmness.FIRM else "floor"
        parts.append(f"{_title(ctx, item_id)} {rule} {fmt_money(minimum)}")
    if total_cents >= need:
        return None
    return Reject(
        f"{fmt_money(total_cents)} is below the {fmt_money(need)} these items need together "
        f"({'; '.join(parts)}). Counter at {fmt_money(need)} or more, or decline."
    )


def _bundle_paused_reason(
    ctx: ToolContext, active: ActiveCommitment, action: str
) -> Reject | None:
    """A bundle changes all of its items, so another item the owner paused blocks it."""
    if ctx.actor == "human":
        return None
    for row in ledger.deal_rows(ctx.session, active.accepted):
        item = ctx.session.get(Item, row.item_id)
        if row.item_id != ctx.item_id and item is not None and item.paused:
            return Reject(
                f"This deal is a bundle with item {item.id} ({item.title or item.hint}), which "
                f"the owner paused. You cannot {action} until they resume it; escalate if it "
                "is urgent."
            )
    return None


def _title(ctx: ToolContext, item_id: int) -> str:
    item = ctx.session.get(Item, item_id)
    return (item.title or item.hint) if item is not None else f"item {item_id}"


def _titles(ctx: ToolContext, rows: Sequence[Commitment]) -> str:
    return ", ".join(_title(ctx, row.item_id) for row in rows)


def _bundle_parts(ctx: ToolContext, rows: Sequence[Commitment]) -> str:
    return ", ".join(
        f"{_title(ctx, row.item_id)} (item {row.item_id}, {fmt_money(row.price_cents or 0)})"
        for row in rows
    )


def _record_outcomes(ctx: ToolContext, item_ids: Sequence[int]) -> None:
    from marketplace_agent.core.outcomes import record_outcome

    for item_id in item_ids:
        record_outcome(ctx.session, ctx.clock, item_id)


def _window(ctx: ToolContext, args: BookPickupArgs) -> tuple[datetime, datetime]:
    start = parse_local_datetime(ctx.settings, args.start)
    return start, start + timedelta(minutes=args.duration_minutes)


def _timing_reason(
    ctx: ToolContext, active: ActiveCommitment, start: datetime, end: datetime
) -> Reject | None:
    hours = availability_text(ctx.settings)
    if start <= ctx.clock.now():
        return Reject(
            f"{fmt_local(ctx.settings, start)} is in the past; it is now "
            f"{fmt_local(ctx.settings, ctx.clock.now())}. Pick a future time within: {hours}."
        )
    if not within_availability(ctx.settings, start, end):
        return Reject(
            f"{_span(ctx, start, end)} is outside pickup availability. The whole pickup must fit "
            f"inside one window: {hours}."
        )
    clash = _clash(ctx, active, start, end)
    if clash is not None:
        return Reject(
            f"That overlaps another pickup booked {_span(ctx, *clash)}. Choose a time that "
            "does not overlap it."
        )
    return None


def _clash(
    ctx: ToolContext, active: ActiveCommitment, start: datetime, end: datetime
) -> tuple[datetime, datetime] | None:
    """Another deal's active pickup sharing any instant with [start, end); this deal's own
    bookings (every item of its bundle) do not count, so rescheduling never clashes with itself."""
    own = {row.id for row in ledger.deal_rows(ctx.session, active.accepted)}
    for other in ledger.active_pickups(ctx.session):
        booked = other.pickup
        if other.accepted.id in own or booked is None:
            continue
        if booked.pickup_start is None or booked.pickup_end is None:
            continue
        if start < booked.pickup_end and booked.pickup_start < end:
            return booked.pickup_start, booked.pickup_end
    return None


def _release_reason(ctx: ToolContext, active: ActiveCommitment, no_show: bool) -> Reject | None:
    """See the module docstring."""
    pickup = active.pickup
    if pickup is None or pickup.pickup_start is None or pickup.pickup_end is None:
        if no_show:
            return Reject(
                "No pickup was booked, so this cannot be a no-show. Release it with no_show false."
            )
        return None
    now = ctx.clock.now()
    if ctx.actor == "human":  # the owner was at the pickup: only the no-show grace applies
        allowed = pickup.pickup_end + timedelta(minutes=ctx.settings.no_show_grace_minutes)
        if no_show and now < allowed:
            return Reject(f"It is a no-show only from {fmt_local(ctx.settings, allowed)}.")
        return None
    if now < pickup.pickup_start and not no_show:
        return None
    allowed = no_show_from(ctx.settings, pickup.pickup_end)
    if now >= allowed:
        return None
    if not no_show and _buyer_wrote_since(ctx, active, pickup.pickup_start):
        return None
    name = buyer_name(ctx.session, active.accepted.buyer_id)
    until = fmt_local(ctx.settings, allowed)
    if now < pickup.pickup_end:
        why = (
            f"The pickup runs until {fmt_local(ctx.settings, pickup.pickup_end)}, and {name} may "
            f"still come. It is a no-show only from {until}."
        )
    else:
        why = (
            f"The pickup is over, but silence does not mean {name} did not come: a buyer who "
            "paid in person often never writes again. The owner has been asked (they answer "
            f"with agent sold or agent noshow). Without an answer, release only from {until}"
            + ("." if no_show else f", or once {name} writes that they are not coming.")
        )
    return Reject(f"{why} {_meanwhile(ctx, active)}")


def _meanwhile(ctx: ToolContext, active: ActiveCommitment) -> str:
    """What the model can do now, given the follow-up rule for this thread."""
    assert active.accepted.thread_id is not None
    count, last = outbound_streak(ctx.session, active.accepted.thread_id)
    if count == 0:
        return "Meanwhile you may message the buyer to check."
    if count == 1 and last is not None and ctx.clock.now() - last >= FOLLOW_UP_AFTER:
        return "Meanwhile one follow-up to the buyer is allowed."
    return "Your last message to the buyer has no reply yet, so wait."


def _buyer_wrote_since(ctx: ToolContext, active: ActiveCommitment, since: datetime) -> bool:
    query = select(Message.id).where(
        Message.thread_id == active.accepted.thread_id,
        Message.direction == Direction.INBOUND,
        col(Message.sent_at) >= since,
    )
    return ctx.session.exec(query).first() is not None


def _delist_everywhere(ctx: ToolContext, items: Sequence[Item]) -> list[str]:
    """Take down every live listing of items. A failure is sent as an ALERT and stored as retry
    state (the runner keeps trying), and never undoes the sale. A post already gone from its
    platform counts as down. Call it with no write pending: it writes only after every platform
    call, so no write lock is held meanwhile."""
    from marketplace_agent.runner.takedown import sale_takedown_failed  # it imports the tools

    errors = []
    with ctx.session.no_autoflush:
        for item in items:
            for listing in live_listings(ctx.session, item.id):
                adapter = ctx.platforms.get(listing.platform)
                try:
                    if adapter is None:
                        raise PlatformError(f"{listing.platform} is not connected")
                    adapter.delist(ref_for(listing))
                except ListingGone as exc:
                    mark_gone(ctx.clock, listing, exc)
                    continue
                except PlatformError as exc:
                    error = redact_address(str(exc), ctx.settings)
                    sale_takedown_failed(ctx.clock, listing, error)
                    where = listing.url or listing.remote_id
                    errors.append(f"{listing.platform} ({where}): {error}")
                    ctx.notifier.notify(
                        Notice(
                            kind=NoticeKind.ALERT,
                            title=f"Take down the {listing.platform} listing",
                            body=f"{item.title} sold, but its {listing.platform} listing could "
                            f"not be taken down: {error}. Remove it by hand.",
                            item_id=item.id,
                        )
                    )
                    continue
                listing.status = ListingStatus.DELISTED
                listing.delisted_at = ctx.clock.now()
    ctx.session.flush()
    return errors


def _span(ctx: ToolContext, start: datetime, end: datetime) -> str:
    return f"{fmt_local(ctx.settings, start)}-{to_local(ctx.settings, end):%H:%M}"
