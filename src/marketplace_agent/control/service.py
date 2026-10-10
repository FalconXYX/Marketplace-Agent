"""Human overrides shared by the CLI, web, and Discord.

Tool-backed actions run the real tool as actor "human" through guard.validate, so the hard rules
still apply. Every change that the agent should react to writes a WakeRequest, and every
correction of the agent writes an item Memory so it learns.

| Method | Acts through | Trust | Memory | Wake |
|---|---|---|---|---|
| confirm_and_publish | publish tool per platform | | | yes |
| undo_commitment | ledger.undo | correction (accept_offer) | undo | yes |
| approve | the parked tool | executed, when it ran | | yes |
| reject | | correction | correction, with a note | yes |
| set_price | update_price tool | correction after an agent reprice < 48h | correction | yes |
| set_goal, on sale | the row, then update_price if repriced | as set_price | as set_price | yes |
| set_goal, not on sale | core.items.set_goal | | | no |
| reply_as_human | send_message tool | | | yes |
| set_floor, set_firmness, set_condition, resume, resolve_escalation | the row | | | yes |
| mark_listing_gone | the listing row: DELISTED, so the agent may relist | | | yes |
| pause | the row | | | no: paused items never wake |
| mark_sold, withdraw | ledger, delist everywhere | | outcome line, global | no: final |
| mark_no_show | ledger.release, buyer's no-show count | | | yes |

mark_sold and mark_no_show answer the pickup check the runner opens when a pickup ends
(guard/handover.py). A button or form keyed on one check passes its id (answer_pickup_check, or
check_id), so an old card cannot act on a deal agreed after it. A goal change that makes the
stored listing text break a rule (an amount now under the floor) says so in its wake, so the
agent rewrites it.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, get_args

from pydantic import ValidationError
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from marketplace_agent.ai.drafting import Condition
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.db import session_scope
from marketplace_agent.core.items import (
    ItemNotFound,
    check_deadline,
    check_goal_prices,
    get_goal,
    get_item,
    touch,
)
from marketplace_agent.core.items import set_goal as write_goal  # the method is set_goal too
from marketplace_agent.core.models import (
    ApprovalStatus,
    Buyer,
    Commitment,
    CommitmentKind,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    MemoryScope,
    PendingApproval,
    PriceChange,
    Thread,
    WakeRequest,
)
from marketplace_agent.core.outcomes import record_outcomes
from marketplace_agent.guard import handover, ledger
from marketplace_agent.guard.invariants import redact_address
from marketplace_agent.guard.trust import record_correction, record_executed
from marketplace_agent.guard.validate import validate
from marketplace_agent.memory.store import remember
from marketplace_agent.notify.base import Notice, NoticeKind, Notifier
from marketplace_agent.platforms.base import PlatformAdapter, PlatformError, PlatformRef
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.runner.takedown import sale_takedown_failed
from marketplace_agent.tools.base import Allow, Tool, ToolContext, ToolResult
from marketplace_agent.tools.impl.common import (
    fmt_local,
    fmt_money,
    listing_text_problems,
    listing_trust_platform,
    live_listings,
    mark_gone,
)
from marketplace_agent.tools.registry import ToolRegistry

CORRECTION_WINDOW = timedelta(hours=48)  # a human reprice this soon after the agent's corrects it
ARGS_IN_LESSON = 120
GoalTerms = tuple[int, int, datetime, Firmness]  # floor, target, deadline, firmness


class ControlError(Exception):
    """The request cannot be carried out; the message says why, for the human."""


class StaleCheck(ControlError):
    """An answer to a pickup check that is already answered or no longer about the active deal."""


class _Refused(ControlError):
    """A tool call the guard rejected or whose arguments are invalid."""


class ControlService:
    def __init__(
        self,
        engine: Engine,
        clock: Clock,
        settings: Settings,
        platforms: Mapping[str, PlatformAdapter],
        notifier: Notifier,
        registry: ToolRegistry,
    ):
        self.engine = engine
        self.clock = clock
        self.settings = settings
        self.platforms = platforms
        self.notifier = notifier
        self.registry = registry

    # ------------------------------------------------------------ listing

    def confirm_and_publish(
        self, item_id: int, platforms: Sequence[str] | None = None
    ) -> list[ToolResult]:
        """Confirm a draft for sale (DRAFT -> LISTED) and publish it on each platform."""
        names = list(platforms or self.settings.enabled_platforms)
        with session_scope(self.engine) as session:
            item = _item(session, item_id)
            if item.status not in (ItemStatus.DRAFT, ItemStatus.LISTED):
                raise ControlError(
                    f"Item {item_id} is {item.status.value}; only a draft or listed item can be "
                    "published."
                )
            if get_goal(session, item_id) is None:
                raise ControlError(f"Item {item_id} has no goal. Set one with `agent goal` first.")
            if not item.title.strip():
                raise ControlError(
                    f"Item {item_id} has no title. Draft it with `agent draft` first."
                )
            if item.status == ItemStatus.DRAFT:
                item.status = ItemStatus.LISTED
                touch(item, self.clock)
                session.add(item)
                session.commit()
            results = [self._run(session, item_id, "publish", {"platform": p}) for p in names]
            done = [n for n, r in zip(names, results, strict=True) if r.ok]
            where = ", ".join(done) or "no platform yet"
            self._wake(
                session, item_id, f"The owner confirmed the item for sale; posted on {where}."
            )
        return results

    def set_price(self, item_id: int, price_cents: int) -> ToolResult:
        with session_scope(self.engine) as session:
            _item(session, item_id)
            corrected = _recent_agent_price_change(session, item_id, self.clock.now())
            result = self._run(session, item_id, "update_price", {"price_cents": price_cents})
            if not result.ok:
                return result
            self._overrode(session, item_id, corrected, price_cents)
            price = fmt_money(price_cents)
            self._wake(session, item_id, f"The owner set the asking price to {price}.")
            return result

    def set_goal(
        self,
        item_id: int,
        *,
        list_price_cents: int,
        floor_price_cents: int,
        target_price_cents: int | None = None,
        deadline_days: int = 14,
        firmness: Firmness = Firmness.SLIGHT,
    ) -> Goal:
        """Replace the goal, keeping its notes. On an item for sale (LISTED or COMMITTED) the
        rest is written under the write lock first, then a new list price goes through the
        update_price tool (live listings, and the PriceChange row that the 24h limit and
        correction detection read), which commits it all before its platform calls: a refusal
        changes nothing, and when no listing takes the price the whole goal is put back.
        Elsewhere it is core.items.set_goal."""
        with session_scope(self.engine) as session:
            item = _item(session, item_id)
            goal = get_goal(session, item_id)
            if goal is None or item.status not in (ItemStatus.LISTED, ItemStatus.COMMITTED):
                try:
                    saved = write_goal(
                        session,
                        self.clock,
                        item_id,
                        list_price_cents=list_price_cents,
                        floor_price_cents=floor_price_cents,
                        target_price_cents=target_price_cents,
                        deadline_days=deadline_days,
                        firmness=firmness,
                        notes=goal.notes if goal else "",
                    )
                except ValueError as exc:
                    raise ControlError(str(exc)) from exc
                if item.status in (ItemStatus.LISTED, ItemStatus.COMMITTED):
                    self._wake(session, item_id, "The owner set the goal: " + _terms(saved))
                return saved
            with ledger.LEDGER_LOCK:
                ledger.claim_item(session, self.clock, item_id)
                session.refresh(goal)
                target = list_price_cents if target_price_cents is None else target_price_cents
                now = self.clock.now()
                try:
                    check_goal_prices(list_price_cents, floor_price_cents, target)
                    due = check_deadline(now + timedelta(days=deadline_days), now)
                except ValueError as exc:
                    raise ControlError(str(exc)) from exc
                before = _goal_terms(goal)
                goal.floor_price_cents, goal.target_price_cents = floor_price_cents, target
                goal.deadline, goal.firmness = due, firmness
                session.add(goal)
                session.flush()
                reprice = list_price_cents != goal.list_price_cents
                corrected = _recent_agent_price_change(session, item_id, now) if reprice else None
            # The lock is left before the price push: it commits, then takes as long as the
            # platforms do, and the agent's deals must not wait for that.
            if reprice:
                written = _goal_terms(goal)
                try:
                    self._reprice(session, item_id, list_price_cents)
                except ControlError:
                    self._restore_goal(session, item_id, written, before)
                    raise
                self._overrode(session, item_id, corrected, list_price_cents)
            self._wake(
                session,
                item_id,
                "The owner set the goal: " + _terms(goal) + self._text_warning(session, item),
            )
            return goal

    def _restore_goal(
        self, session: Session, item_id: int, written: GoalTerms, before: GoalTerms
    ) -> None:
        """Put back the goal set_goal wrote, when its price change failed after committing.
        Uncommitted, the write is simply rolled back; changed by someone since, it stays."""
        session.rollback()
        with ledger.LEDGER_LOCK:
            ledger.claim_item(session, self.clock, item_id)
            goal = _goal(session, item_id)
            session.refresh(goal)
            if _goal_terms(goal) == written:
                goal.floor_price_cents, goal.target_price_cents = before[0], before[1]
                goal.deadline, goal.firmness = before[2], before[3]
                session.add(goal)
            session.commit()

    def _reprice(self, session: Session, item_id: int, price_cents: int) -> None:
        """update_price as the owner, which commits before its platform calls and takes the
        price back when no listing takes it; any refusal or failure raises ControlError."""
        tool = self._tool("update_price")
        try:
            result = self._execute(session, item_id, tool, {"price_cents": price_cents})
        except (PlatformError, ledger.LedgerError) as exc:
            raise ControlError(f"The price could not be changed, so nothing was: {exc}") from exc
        if not result.ok:
            raise ControlError(f"The price could not be changed, so nothing was: {result.summary}")

    def _overrode(
        self, session: Session, item_id: int, corrected: PriceChange | None, price_cents: int
    ) -> None:
        """A human reprice soon after the agent's corrects it: trust and memory."""
        if corrected is None:
            return
        platform = listing_trust_platform(session, item_id)
        record_correction(session, self.clock, platform, "update_price")
        self._learn(
            session,
            item_id,
            f"The owner overrode your price change to {fmt_money(corrected.new_cents)} "
            f"({self._when(corrected.created_at)}) and set {fmt_money(price_cents)}.",
            source="correction",
        )

    def set_floor(self, item_id: int, floor_cents: int) -> None:
        """Checked under the write lock against the latest list price, including one an agent
        price change committed and is still pushing to the platforms, so the new floor is never
        above a price going up meanwhile."""
        if floor_cents <= 0:
            raise ControlError("The floor must be more than $0.")
        with session_scope(self.engine) as session, ledger.LEDGER_LOCK:
            _goal(session, item_id)
            ledger.claim_item(session, self.clock, item_id)
            goal = _goal(session, item_id)
            session.refresh(goal)
            ceiling = min(goal.list_price_cents, goal.target_price_cents)
            if floor_cents > ceiling:
                raise ControlError(
                    "The floor cannot be above the list price "
                    f"({fmt_money(goal.list_price_cents)}) or the target "
                    f"({fmt_money(goal.target_price_cents)})."
                )
            goal.floor_price_cents = floor_cents
            session.add(goal)
            self._wake(
                session,
                item_id,
                f"The owner set the floor to {fmt_money(floor_cents)}."
                + self._text_warning(session, _item(session, item_id)),
            )

    def set_firmness(self, item_id: int, firmness: Firmness) -> None:
        with session_scope(self.engine) as session:
            goal = _goal(session, item_id)
            goal.firmness = firmness
            session.add(goal)
            self._wake(
                session,
                item_id,
                f"The owner set the price firmness to {firmness.value}."
                + self._text_warning(session, _item(session, item_id)),
            )

    def set_condition(self, item_id: int, condition: str) -> None:
        """The item's condition, for posts from now on (Facebook requires one); live posts keep
        theirs. condition is one of ai.drafting.Condition."""
        if condition not in get_args(Condition):
            raise ControlError(f"Condition is one of: {', '.join(get_args(Condition))}.")
        with session_scope(self.engine) as session:
            item = _item(session, item_id)
            if item.status in (ItemStatus.SOLD, ItemStatus.WITHDRAWN):
                raise ControlError(f"Item {item_id} is already {item.status.value}.")
            item.condition = condition
            touch(item, self.clock)
            session.add(item)
            if item.status in (ItemStatus.LISTED, ItemStatus.COMMITTED):
                self._wake(
                    session,
                    item_id,
                    f"The owner set the item's condition to {condition.replace('_', ' ')}; new "
                    "posts and relists carry it.",
                )

    def mark_listing_gone(self, item_id: int, platform: str | None = None) -> str:
        """The owner says a post is no longer up (removed, expired, or deleted by hand). Marks it
        DELISTED, so the agent may relist or publish there again, and wakes the agent."""
        with session_scope(self.engine) as session:
            item = _item(session, item_id)
            live = [
                listing
                for listing in live_listings(session, item_id)
                if platform in (None, listing.platform)
            ]
            if not live:
                where = f" on {platform}" if platform else ""
                raise ControlError(f"Item {item_id} has no live listing{where}.")
            if len(live) > 1:
                names = ", ".join(listing.platform for listing in live)
                raise ControlError(
                    f"Item {item_id} is live on {names}. Say which post is gone with --platform."
                )
            [listing] = live
            mark_gone(self.clock, listing, "the owner reported it gone")
            session.add(listing)
            if item.status == ItemStatus.LISTED:
                self._wake(
                    session,
                    item_id,
                    f"The owner says the {listing.platform} post is gone from "
                    f"{listing.platform}, so the item is not on {listing.platform} now. Relist "
                    f"or publish on {listing.platform} to put it back up.",
                )
                return (
                    f"Marked the {listing.platform} listing of item {item_id} as gone. The agent "
                    "will decide whether to post it again."
                )
        return f"Marked the {listing.platform} listing of item {item_id} as gone."

    # ------------------------------------------------------------ deals

    def undo_commitment(self, commitment_id: int) -> str:
        """Undo the deal, or the whole bundle it belongs to: every item learns of it, and every
        item back on sale wakes."""
        with session_scope(self.engine) as session:
            accepted = session.get(Commitment, commitment_id)
            if accepted is None or accepted.kind != CommitmentKind.ACCEPTED:
                raise ControlError(f"There is no deal with id {commitment_id}.")
            rows = ledger.deal_rows(session, accepted)
            try:
                ledger.undo(session, self.clock, accepted_id=commitment_id, actor="human")
            except ledger.UndoExpired as exc:
                raise ControlError(
                    f"Too late to undo deal {commitment_id}: the undo window closed "
                    f"{self._when(accepted.undo_deadline)}."
                ) from exc
            except ledger.LedgerConflict as exc:
                raise ControlError(
                    f"Deal {commitment_id} was already released, undone, or completed."
                ) from exc
            total = fmt_money(sum(row.price_cents or 0 for row in rows))
            deal = f"{total} deal with {_buyer(session, accepted)}"
            if len(rows) > 1:
                ids = ", ".join(str(row.item_id) for row in rows)
                deal = f"{total} bundle deal with {_buyer(session, accepted)} (items {ids})"
            for row in rows:
                self._learn(
                    session,
                    row.item_id,
                    f"The owner undid your {deal} ({self._when(accepted.created_at)}). Check "
                    "with the owner (escalate) before agreeing to a similar deal.",
                    source="undo",
                )
            record_correction(session, self.clock, "*", "accept_offer")
            items = [_item(session, row.item_id) for row in rows]
            again = "the items are" if len(rows) > 1 else "the item is"
            for item in items:
                if item.status == ItemStatus.LISTED:
                    self._wake(
                        session, item.id, f"The owner undid your {deal}; {again} available again."
                    )
        said = [f"Undid the {deal}".rstrip(".") + "."]  # a name like "Aisha P." ends it
        for item in items:  # an item sold or withdrawn meanwhile stays off sale
            if item.status == ItemStatus.LISTED:
                said.append(f"Item {item.id} is listed again; the agent will re-plan.")
            else:
                said.append(f"Item {item.id} stays {item.status.value}.")
        return " ".join(said)

    def approve(self, approval_id: int) -> ToolResult:
        """Run the parked call as the owner. A call that ran counts as a clean execution for
        the trust ladder."""
        with session_scope(self.engine) as session:
            approval = _pending(session, approval_id)
            tool = self._tool(approval.tool)
            result = self._run(session, approval.item_id, tool.name, json.loads(approval.args_json))
            approval.status = ApprovalStatus.APPROVED
            approval.decided_at = self.clock.now()
            session.add(approval)
            if result.ok and tool.trust_action:
                record_executed(
                    session, self.clock, self.settings, approval.platform, tool.trust_action
                )
            outcome = result.summary if result.ok else f"it could not run: {result.summary}"
            self._wake(session, approval.item_id, f"The owner approved {tool.name}; {outcome}")
            return result

    def reject(self, approval_id: int, note: str = "") -> None:
        with session_scope(self.engine) as session:
            approval = _pending(session, approval_id)
            approval.status = ApprovalStatus.REJECTED
            approval.decided_at = self.clock.now()
            session.add(approval)
            tool = self.registry.get(approval.tool)
            action = tool.trust_action if tool and tool.trust_action else approval.tool
            record_correction(session, self.clock, approval.platform, action)
            call = f"{approval.tool} {_clip(approval.args_json, ARGS_IN_LESSON)}"
            note = redact_address(" ".join(note.split()), self.settings)
            if note:
                self._learn(
                    session,
                    approval.item_id,
                    f"The owner rejected {call}: {note}",
                    source="correction",
                )
            said = f" Their note: {note}" if note else ""
            self._wake(session, approval.item_id, f"The owner rejected {call}.{said}")

    def mark_sold(
        self, item_id: int, price_cents: int | None = None, *, check_id: int | None = None
    ) -> list[str]:
        """Complete the active deal if there is one, else mark the item sold; then take every
        live listing down. A bundle sells whole, its other items too. The outcome line keeps
        the owner's price, else the deal's; for a bundle, price_cents is what the whole bundle
        sold for. The deal is read under the ledger's write lock (close_item), so one the agent is
        accepting right now is completed, not left open on a sold item. With check_id, the answer
        to that pickup check: refused (StaleCheck) unless it still asks about the active deal.
        Returns the listings still up, as "<platform> (<url>)"; each is also sent as an ALERT to
        remove by hand, and the runner keeps trying to take it down."""
        with session_scope(self.engine) as session:
            with ledger.LEDGER_LOCK:
                item = _item(session, item_id)
                if check_id is not None:
                    self._require_check(session, item_id, check_id)
                if item.status == ItemStatus.SOLD:
                    raise ControlError(f"Item {item_id} is already sold.")
                active = self._close_item(session, item_id, ItemStatus.SOLD, "")
                sold = [item_id, *_partners(session, active, item_id)]
                for sold_id in sold:
                    handover.resolve_checks(
                        session, self.clock, sold_id, "The owner marked it sold."
                    )
                record_outcomes(session, self.clock, item_id, final_price_cents=price_cents)
                items = [_item(session, sold_id) for sold_id in sold]
                session.commit()  # no write lock held during the platform calls
            return self._delist_all(session, items)

    def mark_no_show(self, item_id: int, *, check_id: int | None = None) -> str:
        """The owner says the buyer did not come to the booked pickup: release the deal as a
        no-show, which goes on the buyer's record, and wake the agent to go back to others.
        With check_id, the answer to that pickup check: refused (StaleCheck) unless it still asks
        about the active deal, so an old card cannot release a deal agreed since."""
        with session_scope(self.engine) as session, ledger.LEDGER_LOCK:
            _item(session, item_id)
            if check_id is not None:
                self._require_check(session, item_id, check_id)
            active = ledger.active_commitment(session, item_id)
            if active is None:
                raise ControlError(f"Item {item_id} has no active deal.")
            pickup = active.pickup
            if pickup is None or pickup.pickup_start is None:
                raise ControlError(
                    f"The deal on item {item_id} has no booked pickup, so nobody missed one. Use "
                    "`agent undo` within the undo window, or let the agent release it."
                )
            if self.clock.now() < pickup.pickup_start:
                raise ControlError(
                    f"The pickup has not started yet ({self._when(pickup.pickup_start)})."
                )
            name = _buyer(session, active.accepted)
            try:
                ledger.release(
                    session,
                    self.clock,
                    accepted_id=active.accepted.id,
                    reason="no-show, confirmed by the owner",
                    actor="human",
                )
            except ledger.LedgerConflict as exc:
                raise ControlError(str(exc)) from exc
            buyer = session.get(Buyer, active.accepted.buyer_id)
            if buyer is not None:
                buyer.no_show_count += 1
                session.add(buyer)
            released = [item_id, *_partners(session, active, item_id)]
            for released_id in released:
                handover.resolve_checks(
                    session, self.clock, released_id, "The owner confirmed a no-show."
                )
                self._wake(
                    session,
                    released_id,
                    f"The owner says {name} did not come to the pickup. The deal is released as "
                    "a no-show and the item is available again.",
                )
        ids = ", ".join(str(released_id) for released_id in released)
        what = f"item {ids}" if len(released) == 1 else f"items {ids}"
        return f"Released the deal with {name} as a no-show. The agent will re-plan {what}."

    def answer_pickup_check(self, check_id: int, *, came: bool) -> str:
        """A one-tap answer to the pickup check with this escalation id: came is mark_sold, else
        mark_no_show, both for the deal that check names only (StaleCheck otherwise)."""
        with session_scope(self.engine) as session:
            escalation = session.get(Escalation, check_id)
            item_id = escalation.item_id if escalation is not None else None
            if escalation is None or item_id is None or handover.check_ids(escalation) is None:
                raise ControlError(f"There is no pickup check with id {check_id}.")
        if not came:
            return self.mark_no_show(item_id, check_id=check_id)
        still_up = self.mark_sold(item_id, check_id=check_id)
        if still_up:
            return f"Marked item #{item_id} sold.{takedown_note(still_up)}"
        return f"Marked item #{item_id} sold and took its listings down."

    def _require_check(self, session: Session, item_id: int, check_id: int) -> None:
        """Take the write lock, then refuse unless pickup check check_id on this item is open and
        its deal and booking are still current. A bundle's one check, on its first item, is
        answered from any of its items."""
        ledger.claim_item(session, self.clock, item_id)
        escalation = session.get(Escalation, check_id, populate_existing=True)
        if escalation is None or item_id not in _check_items(session, escalation):
            raise ControlError(f"There is no pickup check with id {check_id} on item {item_id}.")
        if escalation.resolved_at is not None:
            answer = f" {escalation.resolution}" if escalation.resolution else ""
            raise StaleCheck(
                f"This pickup check is already answered.{answer} Nothing was changed."
            )
        if not handover.still_asked(session, escalation):
            raise StaleCheck(
                "This pickup check no longer applies: the deal it asked about is closed, or its "
                "pickup was rebooked. Nothing was changed."
            )

    def withdraw(self, item_id: int) -> list[str]:
        """Take the item off the market. An active deal is released, and the owner is told to
        let that buyer know. Returns the listings still up, as mark_sold does."""
        with session_scope(self.engine) as session:
            item = _item(session, item_id)
            if item.status in (ItemStatus.SOLD, ItemStatus.WITHDRAWN):
                raise ControlError(f"Item {item_id} is already {item.status.value}.")
            active = self._close_item(session, item_id, ItemStatus.WITHDRAWN, ledger.WITHDRAWAL)
            if active is not None:
                name = _buyer(session, active.accepted)
                partners = _partners(session, active, item_id)
                what = item.title or item.hint
                if partners:
                    what = f"{what} was promised to them with items " + ", ".join(
                        str(partner) for partner in partners
                    )
                    what += " as a bundle"
                else:
                    what = f"{what} was promised to them"
                thread = session.get(Thread, active.accepted.thread_id)
                where = f" on {thread.platform}" if thread is not None else ""
                self.notifier.notify(
                    Notice(
                        kind=NoticeKind.ALERT,
                        title=f"Tell {name} the deal is off",
                        body=f"{what}; withdrawing it cancelled that deal, and the agent will "
                        f"not message them about it. Write to them yourself{where}.",
                        item_id=item_id,
                    )
                )
                for partner in partners:
                    self._wake(
                        session,
                        partner,
                        f"The owner withdrew item {item_id} ({item.title or item.hint}), which "
                        f"cancelled the bundle deal with {name}; this item is available again. "
                        f"The owner is telling {name} the deal is off.",
                    )
            record_outcomes(session, self.clock, item_id)
            session.commit()  # no write lock held during the platform calls
            return self._delist_all(session, [item])

    def _close_item(
        self, session: Session, item_id: int, status: ItemStatus, reason: str
    ) -> ledger.ActiveCommitment | None:
        try:
            return ledger.close_item(
                session, self.clock, item_id=item_id, status=status, actor="human", reason=reason
            )
        except ledger.LedgerConflict as exc:
            raise ControlError(str(exc)) from exc

    # ------------------------------------------------------------ agent control

    def pause(self, item_id: int) -> None:
        with session_scope(self.engine) as session:
            item = _item(session, item_id)
            item.paused = True
            touch(item, self.clock)
            session.add(item)

    def resume(self, item_id: int) -> None:
        with session_scope(self.engine) as session:
            item = _item(session, item_id)
            if not item.paused:
                return
            item.paused = False
            touch(item, self.clock)
            session.add(item)
            self._wake(session, item_id, "The owner resumed the agent on this item after a pause.")

    def resolve_escalation(self, escalation_id: int, resolution: str) -> None:
        with session_scope(self.engine) as session:
            escalation = session.get(Escalation, escalation_id)
            if escalation is None:
                raise ControlError(f"There is no escalation with id {escalation_id}.")
            if escalation.resolved_at is not None:
                raise ControlError(f"Escalation {escalation_id} is already resolved.")
            answer = redact_address(" ".join(resolution.split()), self.settings)
            escalation.resolved_at = self.clock.now()
            escalation.resolution = answer
            session.add(escalation)
            if escalation.item_id is not None:
                self._wake(
                    session,
                    escalation.item_id,
                    f"The owner answered escalation #{escalation_id}: {answer or '(no note)'}",
                )

    def reply_as_human(self, thread_id: int, body: str) -> ToolResult:
        """Send as the owner: the text rules apply, the follow-up, cap, and address rules do not."""
        with session_scope(self.engine) as session:
            thread = session.get(Thread, thread_id)
            if thread is None:
                raise ControlError(f"There is no conversation with id {thread_id}.")
            if thread.item_id is None:
                raise ControlError(
                    f"Conversation {thread_id} is not linked to any of your items, so it "
                    "cannot be answered from here. Reply on the platform."
                )
            item_id = thread.item_id
            item = _item(session, item_id)
            if item.status in (ItemStatus.SOLD, ItemStatus.WITHDRAWN):
                raise ControlError(
                    f"Item {item_id} is {item.status.value}, so nothing more is sent about it "
                    f"from here. Write to {_name(session, thread.buyer_id)} in the "
                    f"{thread.platform} app or site instead."
                )
            result = self._run(
                session, item_id, "send_message", {"thread_id": thread_id, "body": body}
            )
            if result.ok:
                self._wake(session, item_id, f"The owner replied in thread {thread_id} themselves.")
            return result

    # ------------------------------------------------------------ helpers

    def _tool(self, name: str) -> Tool:
        tool = self.registry.get(name)
        if tool is None:
            raise ControlError(f"There is no tool named {name!r}.")
        return tool

    def _run(self, session: Session, item_id: int, name: str, raw: dict[str, Any]) -> ToolResult:
        """Run a tool as the owner through the guard. A rejection or a platform or ledger failure
        comes back as ok=False with the reason; a failed execution is rolled back."""
        tool = self._tool(name)
        try:
            result = self._execute(session, item_id, tool, raw)
        except _Refused as exc:
            return ToolResult(ok=False, summary=str(exc))
        except (PlatformError, ledger.LedgerError) as exc:
            session.rollback()
            return ToolResult(ok=False, summary=f"{name} failed: {exc}")
        session.commit()
        return result

    def _execute(
        self, session: Session, item_id: int, tool: Tool, raw: dict[str, Any]
    ) -> ToolResult:
        """Validate and execute as the owner, without committing. _Refused when the arguments
        are invalid or the guard says no; platform and ledger errors propagate."""
        try:
            args = tool.parse_args(raw)
        except ValidationError as exc:
            raise _Refused(f"Invalid {tool.name} request: {exc.errors()[0]['msg']}") from exc
        ctx = ToolContext(
            session=session,
            clock=self.clock,
            settings=self.settings,
            platforms=self.platforms,
            notifier=self.notifier,
            item_id=item_id,
            actor="human",
        )
        verdict = validate(ctx, tool, args)
        if not isinstance(verdict, Allow):
            raise _Refused(verdict.reason)
        return tool.execute(ctx, args)

    def _text_warning(self, session: Session, item: Item) -> str:
        """Rules the stored listing text breaks under the current goal, for a wake reason."""
        goal = get_goal(session, item.id)
        if goal is None or not item.title.strip():
            return ""
        reasons = listing_text_problems(
            self.settings, goal, title=item.title, description=item.description
        )
        if not reasons:
            return ""
        return (
            " The listing text now breaks a rule, so it cannot be posted or reposted until you "
            "rewrite it with rewrite_listing: " + " ".join(reasons)
        )

    def _wake(self, session: Session, item_id: int, reason: str) -> None:
        session.add(WakeRequest(item_id=item_id, reason=reason, created_at=self.clock.now()))

    def _learn(self, session: Session, item_id: int, lesson: str, *, source: str) -> None:
        remember(session, self.clock, MemoryScope.ITEM, lesson, item_id=item_id, source=source)

    def _delist_all(self, session: Session, items: Sequence[Item]) -> list[str]:
        """Take down every live listing of items. A failure is sent as an ALERT and kept on the
        listing as retry state, so the runner keeps trying and says once it is down. Returns the
        ones still up, as "<platform> (<url>)". Call it with no write pending: it writes only
        after every platform call, so no write lock is held meanwhile."""
        by_id = {item.id: item for item in items}
        query = select(Listing).where(
            col(Listing.item_id).in_(list(by_id)), Listing.status == ListingStatus.LIVE
        )
        still_up: list[str] = []
        with session.no_autoflush:
            for listing in session.exec(query.order_by(col(Listing.id))).all():
                item = by_id[listing.item_id]
                try:
                    self._delist(listing)
                except ListingGone as exc:
                    mark_gone(self.clock, listing, exc)
                except PlatformError as exc:
                    error = redact_address(str(exc), self.settings)
                    sale_takedown_failed(self.clock, listing, error)
                    self.notifier.notify(
                        Notice(
                            kind=NoticeKind.ALERT,
                            title=f"Take down the {listing.platform} listing",
                            body=f"{item.title or item.hint}: the {listing.platform} listing "
                            f"could not be taken down ({error}). Remove it by hand: "
                            f"{listing.url or '-'}",
                            item_id=item.id,
                        )
                    )
                    still_up.append(f"{listing.platform} ({listing.url or 'no link'})")
                else:
                    listing.status = ListingStatus.DELISTED
                    listing.delisted_at = self.clock.now()
                session.add(listing)
        return still_up

    def _delist(self, listing: Listing) -> None:
        adapter = self.platforms.get(listing.platform)
        if adapter is None:
            raise PlatformError(f"{listing.platform} is not connected")
        if listing.remote_id is None:
            raise PlatformError("the listing has no remote id")
        adapter.delist(PlatformRef(listing.platform, listing.remote_id, listing.url))

    def _when(self, when: datetime | None) -> str:
        return fmt_local(self.settings, when) if when is not None else "earlier"


def _partners(
    session: Session, active: ledger.ActiveCommitment | None, item_id: int
) -> list[int]:
    """The other items of the bundle deal that was active on item_id."""
    if active is None:
        return []
    rows = ledger.deal_rows(session, active.accepted)
    return [row.item_id for row in rows if row.item_id != item_id]


def _check_items(session: Session, escalation: Escalation) -> set[int]:
    """The items a pickup check asks about: its own, and the rest of its deal's bundle. Empty
    for any other escalation."""
    ids = handover.check_ids(escalation)
    if ids is None or escalation.item_id is None:
        return set()
    accepted = session.get(Commitment, ids[0])
    rows = ledger.deal_rows(session, accepted) if accepted is not None else []
    return {escalation.item_id, *(row.item_id for row in rows)}


def _item(session: Session, item_id: int) -> Item:
    try:
        return get_item(session, item_id)
    except ItemNotFound as exc:
        raise ControlError(f"There is no item with id {item_id}.") from exc


def _goal(session: Session, item_id: int) -> Goal:
    _item(session, item_id)
    goal = get_goal(session, item_id)
    if goal is None:
        raise ControlError(f"Item {item_id} has no goal. Set one with `agent goal` first.")
    return goal


def _pending(session: Session, approval_id: int) -> PendingApproval:
    approval = session.get(PendingApproval, approval_id)
    if approval is None:
        raise ControlError(f"There is no approval with id {approval_id}.")
    if approval.status != ApprovalStatus.PENDING:
        raise ControlError(f"Approval {approval_id} is already {approval.status.value}.")
    return approval


def _recent_agent_price_change(session: Session, item_id: int, now: datetime) -> PriceChange | None:
    query = (
        select(PriceChange)
        .where(
            PriceChange.item_id == item_id,
            PriceChange.actor == "agent",
            col(PriceChange.created_at) >= now - CORRECTION_WINDOW,
        )
        .order_by(col(PriceChange.created_at).desc(), col(PriceChange.id).desc())
    )
    return session.exec(query).first()


def _buyer(session: Session, accepted: Commitment) -> str:
    return _name(session, accepted.buyer_id)


def _name(session: Session, buyer_id: int | None) -> str:
    buyer = session.get(Buyer, buyer_id) if buyer_id is not None else None
    if buyer is None:
        return "the buyer"
    return buyer.display_name or buyer.handle


def _goal_terms(goal: Goal) -> GoalTerms:
    return goal.floor_price_cents, goal.target_price_cents, goal.deadline, goal.firmness


def _terms(goal: Goal) -> str:
    return (
        f"list {fmt_money(goal.list_price_cents)}, target {fmt_money(goal.target_price_cents)}, "
        f"floor {fmt_money(goal.floor_price_cents)}, {goal.firmness.value}."
    )


def takedown_note(still_up: Sequence[str]) -> str:
    """For the reply to mark_sold or withdraw: the posts still up, or ""."""
    if not still_up:
        return ""
    return f" Could not take it down on {', '.join(still_up)}: remove it there by hand."


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
