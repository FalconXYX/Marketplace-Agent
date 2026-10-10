"""What the Discord bot shows and does, without Discord: notices become cards, buttons and slash
commands become ControlService calls, and the daily digest is plain text. notify/discord_bot.py
turns cards into embeds and buttons, and forms into modals.

| Notice | Buttons |
|---|---|
| COMMITMENT | Undo, until the undo deadline |
| ESCALATION, pickup check | Picked up (mark_sold), No-show (mark_no_show), for that check's deal |
| ESCALATION, any other | Resolve, Reply as me (when it has a thread), Pause |
| APPROVAL | Approve, Reject |
| INFO | Pause |
| ALERT, frozen scam conversation | Resolve |
| ALERT, any other; SOLD, DIGEST | none |

A button's custom_id is "ma:<action>:<id>". The id names the record the action works on:

| Action | Id | ControlService | Form |
|---|---|---|---|
| undo | commitment | undo_commitment | |
| approve | approval | approve | |
| reject | approval | reject | note (modal) |
| resolve | escalation | resolve_escalation | answer (modal) |
| reply | thread | reply_as_human | text (modal) |
| pause, resume | item | pause, resume | |
| sold | item | mark_sold | price (slash command only) |
| pickedup, noshow | pickup check (escalation) | answer_pickup_check | |
| price, floor | item | set_price, set_floor | amount |
| firm | item | set_firmness | level |

Every card text passes through redact_address, so the exact pickup address never reaches Discord.
"""

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, SQLModel, col, select

from marketplace_agent.ai.base import Capability
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.control.describe import describe_call, quote
from marketplace_agent.control.service import (
    ControlError,
    ControlService,
    StaleCheck,
    takedown_note,
)
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.items import MAX_PRICE_CENTS, get_goal, get_item
from marketplace_agent.core.models import (
    AgentRun,
    ApprovalStatus,
    Commitment,
    CommitmentKind,
    Direction,
    Escalation,
    Firmness,
    Item,
    ItemStatus,
    Listing,
    Message,
    PendingApproval,
    Thread,
    ThreadState,
    TrustTier,
)
from marketplace_agent.guard.handover import no_show_from
from marketplace_agent.guard.invariants import (
    ADDRESS_PLACEHOLDER,
    MAX_MESSAGE_CHARS,
    redact_address,
    to_local,
)
from marketplace_agent.guard.ledger import ActiveCommitment, active_commitment, deal_rows
from marketplace_agent.guard.trust import summary as trust_summary
from marketplace_agent.notify.base import Notice, NoticeKind
from marketplace_agent.runner.triggers import total_views
from marketplace_agent.tools.impl.common import buyer_name, fmt_local, fmt_money, live_listings

ButtonStyle = Literal["primary", "secondary", "success", "danger"]
Form = Mapping[str, str]  # what the owner typed into a modal or slash command

# Discord limits
EMBED_MAX = 6000
TITLE_MAX = 256
DESCRIPTION_MAX = 4096
FIELD_NAME_MAX = 256
FIELD_VALUE_MAX = 1024
LABEL_MAX = 80
MAX_FIELDS = 25

COLORS: dict[NoticeKind, int] = {
    NoticeKind.COMMITMENT: 0x2ECC71,
    NoticeKind.ESCALATION: 0xE67E22,
    NoticeKind.APPROVAL: 0x3498DB,
    NoticeKind.INFO: 0x95A5A6,
    NoticeKind.ALERT: 0xE74C3C,
    NoticeKind.SOLD: 0xF1C40F,
    NoticeKind.DIGEST: 0x9B59B6,
}
ITEM_COLOR = 0x5865F2

CUSTOM_ID_TEMPLATE = r"ma:(?P<action>[a-z]+):(?P<ref>[1-9][0-9]{0,18})"
_CUSTOM_ID = re.compile(CUSTOM_ID_TEMPLATE)
PICKUP_CHECK = re.compile(r"^Pickup check \(deal #(\d+), pickup #\d+\)")  # guard/handover.py

NO_OWNER = (
    "Nobody can act from Discord yet: set discord_owner_id (MA_DISCORD_OWNER_ID) to your Discord "
    "user id."
)
NOT_OWNER = "Only the owner can do that."
UNKNOWN_BUTTON = "That button is not one the agent knows."
DATABASE_BUSY = "The database is busy: the agent is in the middle of an action. Try again."
LIVE_THREAD_WINDOW = timedelta(hours=72)
_OPEN_ITEMS = (ItemStatus.DRAFT, ItemStatus.LISTED, ItemStatus.COMMITTED)
_DOLLARS = re.compile(r"\d+(?:\.\d{1,2})?")


# ---------------------------------------------------------------- cards


@dataclass(frozen=True)
class Button:
    label: str
    custom_id: str
    style: ButtonStyle = "secondary"
    disabled: bool = False


@dataclass(frozen=True)
class CardField:
    name: str
    value: str
    inline: bool = False


@dataclass(frozen=True)
class Card:
    title: str
    description: str
    color: int
    fields: tuple[CardField, ...] = ()
    buttons: tuple[Button, ...] = ()


def notice_ready(session: Session, notice: Notice) -> bool:
    """Whether the record a notice points at is visible yet. Notices are sent inside the
    sender's transaction, so the row may commit a moment after the notice arrives."""
    model = _REF_MODELS.get(notice.kind)
    if model is None or notice.ref_id is None:
        return True
    return session.get(model, notice.ref_id) is not None


def render_notice(session: Session, clock: Clock, settings: Settings, notice: Notice) -> Card:
    """The card for a notice; falls back to the notice's own text when its record is missing."""
    if notice.kind == NoticeKind.COMMITMENT:
        return _commitment_card(session, clock, settings, notice)
    if notice.kind == NoticeKind.ESCALATION:
        return _escalation_card(session, settings, notice)
    if notice.kind == NoticeKind.APPROVAL:
        return _approval_card(session, settings, notice)
    if notice.kind == NoticeKind.ALERT and notice.ref_id is not None:
        return _frozen_card(session, settings, notice)
    buttons = _pause_buttons(notice.item_id) if notice.kind == NoticeKind.INFO else ()
    return _notice_card(session, settings, notice, buttons=buttons)


def item_url(settings: Settings, item_id: int) -> str | None:
    """The item's page on the web intake app, when it has a reachable address."""
    if not settings.web_host or settings.web_host in ("0.0.0.0", "::"):
        return None
    return f"http://{settings.web_host}:{settings.web_port}/items/{item_id}"


def _commitment_card(session: Session, clock: Clock, settings: Settings, notice: Notice) -> Card:
    accepted = _row(session, Commitment, notice.ref_id)
    if accepted is None or accepted.kind != CommitmentKind.ACCEPTED:
        return _notice_card(session, settings, notice, buttons=_undo_buttons(notice, open=True))
    active = active_commitment(session, accepted.item_id)
    live = active is not None and active.accepted.id == accepted.id
    deadline = accepted.undo_deadline or notice.expires_at
    can_undo = live and deadline is not None and clock.now() < deadline
    name = buyer_name(session, accepted.buyer_id)
    rows = deal_rows(session, accepted)
    if len(rows) > 1:
        lines = [
            f"Deal #{accepted.id}: a bundle promised to {name} for {_total(rows)} in total: "
            f"{_bundle_parts(session, rows)}. Undo cancels all of it."
        ]
    else:
        lines = [f"Deal #{accepted.id}: promised to {name} for {_price(accepted)}."]
    if deadline is not None:
        verb = "Undo until" if can_undo else "Undo closed at"
        lines.append(f"{verb} {fmt_local(settings, deadline)}")
    if not live:
        lines.append("This deal is closed.")
    pickup = active.pickup if active is not None and live else None
    return _card(
        settings,
        notice.title,
        "\n".join(lines),
        COLORS[notice.kind],
        fields=[
            CardField("Pickup", _pickup_text(settings, pickup)),
            *_item_fields(session, settings, notice.item_id),
        ],
        buttons=_undo_buttons(notice, open=can_undo),
    )


def _undo_buttons(notice: Notice, *, open: bool) -> tuple[Button, ...]:
    if notice.ref_id is None:
        return ()
    return (Button("Undo", custom_id("undo", notice.ref_id), "danger", disabled=not open),)


def _escalation_card(session: Session, settings: Settings, notice: Notice) -> Card:
    escalation = _row(session, Escalation, notice.ref_id)
    check = PICKUP_CHECK.match(escalation.reason) if escalation is not None else None
    if escalation is not None and check is not None:
        return _pickup_check_card(session, settings, notice, escalation, int(check[1]))
    done = escalation is not None and escalation.resolved_at is not None
    buttons: list[Button] = []
    if notice.ref_id is not None:
        buttons.append(Button("Resolve", custom_id("resolve", notice.ref_id), "primary", done))
    thread_id = escalation.thread_id if escalation is not None else None
    if thread_id is not None:
        buttons.append(Button("Reply as me", custom_id("reply", thread_id), disabled=done))
    buttons.extend(_pause_buttons(notice.item_id))
    fields = _last_buyer_message(session, thread_id)
    return _card(
        settings,
        notice.title,
        notice.body,
        COLORS[notice.kind],
        fields=[*fields, *_item_fields(session, settings, notice.item_id)],
        buttons=buttons,
    )


def _pickup_check_card(
    session: Session, settings: Settings, notice: Notice, escalation: Escalation, deal_id: int
) -> Card:
    accepted = session.get(Commitment, deal_id)
    item_id = escalation.item_id
    name = buyer_name(session, accepted.buyer_id) if accepted is not None else "the buyer"
    active = active_commitment(session, item_id) if item_id is not None else None
    pickup = active.pickup if active is not None and active.accepted.id == deal_id else None
    rows = deal_rows(session, accepted) if accepted is not None else []
    it = "them" if len(rows) > 1 else "it"
    lines = [f"Tap Picked up if {name} took {it}, or No-show if they did not come."]
    if pickup is not None and pickup.pickup_end is not None:
        until = fmt_local(settings, no_show_from(settings, pickup.pickup_end))
        lines.append(f"Without an answer, the agent may treat them as a no-show from {until}.")
    fields = [CardField("Pickup", _pickup_text(settings, pickup))]
    if len(rows) > 1:
        deal = f"{_total(rows)} with {name} for {_bundle_parts(session, rows)}"
        fields.insert(0, CardField("Deal", deal))
    elif accepted is not None:
        fields.insert(0, CardField("Deal", f"{_price(accepted)} with {name}", inline=True))
    buttons: tuple[Button, ...] = ()
    if item_id is not None and escalation.id is not None:
        done = escalation.resolved_at is not None
        buttons = (
            Button("Picked up", custom_id("pickedup", escalation.id), "success", done),
            Button("No-show", custom_id("noshow", escalation.id), "danger", done),
        )
    return _card(
        settings,
        notice.title,
        "\n".join(lines),
        COLORS[notice.kind],
        fields=[*fields, *_item_fields(session, settings, item_id)],
        buttons=buttons,
    )


def _frozen_card(session: Session, settings: Settings, notice: Notice) -> Card:
    """A frozen scam conversation (runner/ingest.py) opens an escalation and refers to it: the
    owner resolves it here, or it waits in every digest."""
    escalation = _row(session, Escalation, notice.ref_id)
    buttons: tuple[Button, ...] = ()
    if escalation is not None and notice.ref_id is not None:
        done = escalation.resolved_at is not None
        buttons = (Button("Resolve", custom_id("resolve", notice.ref_id), "primary", done),)
    return _notice_card(session, settings, notice, buttons=buttons)


def _approval_card(session: Session, settings: Settings, notice: Notice) -> Card:
    approval = _row(session, PendingApproval, notice.ref_id)
    if approval is None:
        return _notice_card(session, settings, notice, buttons=_approval_buttons(notice.ref_id))
    args = json.loads(approval.args_json)
    lines = [
        "The agent wants to:",
        describe_call(session, settings, approval.item_id, approval.tool, args),
    ]
    if approval.rationale.strip():
        lines += ["", f"Why: {approval.rationale.strip()}"]
    if approval.status != ApprovalStatus.PENDING:
        lines += ["", f"Already {approval.status.value}."]
    expires = approval.created_at + timedelta(hours=settings.approval_ttl_hours)
    return _card(
        settings,
        f"Approve #{approval.id}? {_item_label(session, approval.item_id)}",
        "\n".join(lines),
        COLORS[notice.kind],
        fields=[
            CardField("Expires", fmt_local(settings, expires), inline=True),
            *_item_fields(session, settings, approval.item_id),
        ],
        buttons=_approval_buttons(approval.id, disabled=approval.status != ApprovalStatus.PENDING),
    )


def _approval_buttons(approval_id: int | None, *, disabled: bool = False) -> tuple[Button, ...]:
    if approval_id is None:
        return ()
    return (
        Button("Approve", custom_id("approve", approval_id), "success", disabled),
        Button("Reject", custom_id("reject", approval_id), "danger", disabled),
    )


def _pause_buttons(item_id: int | None) -> tuple[Button, ...]:
    return () if item_id is None else (Button("Pause item", custom_id("pause", item_id)),)


def _notice_card(
    session: Session, settings: Settings, notice: Notice, *, buttons: Sequence[Button] = ()
) -> Card:
    return _card(
        settings,
        notice.title,
        notice.body,
        COLORS[notice.kind],
        fields=_item_fields(session, settings, notice.item_id),
        buttons=buttons,
    )


def _item_fields(session: Session, settings: Settings, item_id: int | None) -> list[CardField]:
    """The link back to the item: its web page, and each live listing."""
    item = _row(session, Item, item_id)
    if item is None or item_id is None:
        return []
    label = f"#{item_id} {_label(item)}".replace("[", "(").replace("]", ")")
    url = item_url(settings, item_id)
    lines = [f"[{label}]({url})" if url else label]
    lines += [f"{listing.platform}: {listing.url}" for listing in _linked(session, item_id)]
    return [CardField("Item", "\n".join(lines))]


def _linked(session: Session, item_id: int) -> list[Listing]:
    return [listing for listing in live_listings(session, item_id) if listing.url]


def _last_buyer_message(session: Session, thread_id: int | None) -> list[CardField]:
    thread = _row(session, Thread, thread_id)
    if thread is None:
        return []
    query = (
        select(Message)
        .where(Message.thread_id == thread.id, Message.direction == Direction.INBOUND)
        .order_by(col(Message.sent_at).desc(), col(Message.id).desc())
    )
    message = session.exec(query).first()
    if message is None:
        return []
    name = buyer_name(session, thread.buyer_id)
    return [CardField(f"Last from {name}", quote(message.body))]


def _card(
    settings: Settings,
    title: str,
    description: str,
    color: int,
    *,
    fields: Sequence[CardField] = (),
    buttons: Sequence[Button] = (),
) -> Card:
    """A card inside Discord's limits, with the pickup address redacted from every text."""

    def safe(text: str, limit: int) -> str:
        return _clip(redact_address(text, settings), limit)

    title = safe(title, TITLE_MAX) or "Notice"
    kept = tuple(
        CardField(safe(f.name, FIELD_NAME_MAX), safe(f.value, FIELD_VALUE_MAX) or "-", f.inline)
        for f in fields[:MAX_FIELDS]
    )
    room = EMBED_MAX - len(title) - sum(len(f.name) + len(f.value) for f in kept)
    return Card(
        title=title,
        description=safe(description, min(DESCRIPTION_MAX, room)),
        color=color,
        fields=kept,
        buttons=tuple(
            Button(_clip(b.label, LABEL_MAX), b.custom_id, b.style, b.disabled) for b in buttons
        ),
    )


# ---------------------------------------------------------------- actions


@dataclass(frozen=True)
class ActionResponse:
    text: str
    ephemeral: bool = True
    disable_buttons: bool = False


@dataclass(frozen=True)
class FormField:
    key: str
    label: str  # Discord allows 45 characters
    required: bool
    placeholder: str = ""
    max_length: int = 1000


@dataclass(frozen=True)
class ModalSpec:
    title: str
    fields: tuple[FormField, ...]


class InvalidInput(ValueError):
    """A form or command value the owner typed cannot be used; the message says why."""


def custom_id(action: str, ref: int) -> str:
    if action not in _HANDLERS or ref < 1:
        raise ValueError(f"No button for {action!r} on {ref}")
    return f"ma:{action}:{ref}"


def parse_custom_id(value: str) -> tuple[str, int]:
    """(action, id) from "ma:<action>:<id>"; ValueError for anything else."""
    match = _CUSTOM_ID.fullmatch(value)
    if match is None or match["action"] not in _HANDLERS:
        raise ValueError(f"Not an agent button: {value!r}")
    return match["action"], int(match["ref"])


def check_owner(user_id: int, owner_id: int | None) -> ActionResponse | None:
    """The refusal for anyone but the owner, or None for the owner."""
    if owner_id is None:
        return ActionResponse(NO_OWNER)
    return None if user_id == owner_id else ActionResponse(NOT_OWNER)


def modal_for(value: str) -> ModalSpec | None:
    """The form to ask for before running a button's action, if it needs one."""
    try:
        action, ref = parse_custom_id(value)
    except ValueError:
        return None
    spec = MODALS.get(action)
    return None if spec is None else ModalSpec(f"{spec.title} #{ref}", spec.fields)


def handle_action(
    control: ControlService,
    custom_id: str,
    user_id: int,
    owner_id: int | None,
    form: Form | None = None,
) -> ActionResponse:
    """Check the owner, then run the action through ControlService. A refusal comes back as an
    ephemeral text; it never raises for something the owner can fix."""
    refused = check_owner(user_id, owner_id)
    if refused is not None:
        return refused
    try:
        action, ref = parse_custom_id(custom_id)
    except ValueError:
        return ActionResponse(UNKNOWN_BUTTON)
    try:
        return _HANDLERS[action](control, ref, form or {})
    except StaleCheck as exc:
        return ActionResponse(str(exc), disable_buttons=True)
    except (ControlError, InvalidInput) as exc:
        return ActionResponse(str(exc))
    except OperationalError as exc:
        if "locked" in str(exc.orig) or "busy" in str(exc.orig):
            return ActionResponse(DATABASE_BUSY)
        raise


def parse_dollars(text: str) -> int:
    """Cents from "120", "120.50", "$1,200"."""
    raw = text.strip().removeprefix("$").replace(",", "")
    if not _DOLLARS.fullmatch(raw):
        raise InvalidInput(f"Expected dollars like 120 or 120.50, got {text!r}.")
    cents = int(Decimal(raw) * 100)
    if not 0 < cents <= MAX_PRICE_CENTS:
        raise InvalidInput(f"Expected an amount above $0 and up to {fmt_money(MAX_PRICE_CENTS)}.")
    return cents


def _done(text: str) -> ActionResponse:
    """A decision that is now made: everyone sees it, and the card's buttons switch off."""
    return ActionResponse(text, ephemeral=False, disable_buttons=True)


def _said(text: str) -> ActionResponse:
    return ActionResponse(text, ephemeral=False)


def _undo(control: ControlService, deal_id: int, _form: Form) -> ActionResponse:
    return _done(control.undo_commitment(deal_id))


def _approve(control: ControlService, approval_id: int, _form: Form) -> ActionResponse:
    result = control.approve(approval_id)
    if result.ok:
        return _done(f"Approved #{approval_id}: {result.summary}")
    return _done(f"Approved #{approval_id}, but it could not run: {result.summary}")


def _reject(control: ControlService, approval_id: int, form: Form) -> ActionResponse:
    note = form.get("note", "").strip()
    control.reject(approval_id, note)
    return _done(f"Rejected #{approval_id}." + (f" Note: {note}" if note else ""))


def _resolve(control: ControlService, escalation_id: int, form: Form) -> ActionResponse:
    answer = form.get("answer", "").strip()
    control.resolve_escalation(escalation_id, answer)
    said = f" Answer: {answer}" if answer else ""
    return _done(f"Resolved escalation #{escalation_id}; the agent will re-plan.{said}")


def _reply(control: ControlService, thread_id: int, form: Form) -> ActionResponse:
    text = form.get("text", "").strip()
    if not text:
        return ActionResponse("Write the message to send first.")
    result = control.reply_as_human(thread_id, text)
    return ActionResponse(result.summary, ephemeral=not result.ok)


def _pause(control: ControlService, item_id: int, _form: Form) -> ActionResponse:
    control.pause(item_id)
    return _said(f"Paused item #{item_id}. The agent takes no action on it until you resume it.")


def _resume(control: ControlService, item_id: int, _form: Form) -> ActionResponse:
    control.resume(item_id)
    return _said(f"Resumed item #{item_id}.")


def _sold(control: ControlService, item_id: int, form: Form) -> ActionResponse:
    price = form.get("price", "").strip()
    still_up = control.mark_sold(item_id, parse_dollars(price) if price else None)
    if still_up:
        return _done(f"Marked item #{item_id} sold.{takedown_note(still_up)}")
    return _done(f"Marked item #{item_id} sold and took its listings down.")


def _picked_up(control: ControlService, check_id: int, _form: Form) -> ActionResponse:
    return _done(control.answer_pickup_check(check_id, came=True))


def _noshow(control: ControlService, check_id: int, _form: Form) -> ActionResponse:
    return _done(control.answer_pickup_check(check_id, came=False))


def _set_price(control: ControlService, item_id: int, form: Form) -> ActionResponse:
    result = control.set_price(item_id, parse_dollars(form.get("amount", "")))
    return ActionResponse(result.summary, ephemeral=not result.ok)


def _set_floor(control: ControlService, item_id: int, form: Form) -> ActionResponse:
    cents = parse_dollars(form.get("amount", ""))
    control.set_floor(item_id, cents)
    return _said(f"Set the floor of item #{item_id} to {fmt_money(cents)}. The agent will re-plan.")


def _set_firmness(control: ControlService, item_id: int, form: Form) -> ActionResponse:
    level = form.get("level", "").strip().lower()
    if level not in set(Firmness):
        raise InvalidInput("Firmness is firm, slight, or flexible.")
    control.set_firmness(item_id, Firmness(level))
    return _said(f"Set item #{item_id} to {level}. The agent will re-plan.")


_HANDLERS: dict[str, Callable[[ControlService, int, Form], ActionResponse]] = {
    "undo": _undo,
    "approve": _approve,
    "reject": _reject,
    "resolve": _resolve,
    "reply": _reply,
    "pause": _pause,
    "resume": _resume,
    "sold": _sold,
    "pickedup": _picked_up,
    "noshow": _noshow,
    "price": _set_price,
    "floor": _set_floor,
    "firm": _set_firmness,
}
ACTIONS = frozenset(_HANDLERS)

MODALS: dict[str, ModalSpec] = {
    "reject": ModalSpec(
        "Reject", (FormField("note", "Why? The agent learns from it", required=False),)
    ),
    "resolve": ModalSpec(
        "Resolve escalation",
        (FormField("answer", "What should the agent know or do?", required=False),),
    ),
    "reply": ModalSpec(
        "Reply as you",
        (
            FormField(
                "text",
                "Message to the buyer",
                required=True,
                placeholder=f"Write {ADDRESS_PLACEHOLDER} to share the address",
                max_length=MAX_MESSAGE_CHARS,
            ),
        ),
    ),
}


# ---------------------------------------------------------------- slash command views


def items_overview(session: Session, settings: Settings) -> str:
    """One line per item still in play, then a count of the finished ones."""
    query = select(Item).where(col(Item.status).in_(_OPEN_ITEMS)).order_by(col(Item.id))
    lines = [_item_line(session, item) for item in session.exec(query).all()]
    finished = Counter(
        item.status.value
        for item in session.exec(select(Item).where(col(Item.status).not_in(_OPEN_ITEMS))).all()
    )
    if finished:
        lines.append(", ".join(f"{count} {status}" for status, count in sorted(finished.items())))
    text = "\n".join(lines) or "No items yet. Add one on the web page or with `agent add-item`."
    return _clip(redact_address(text, settings), DESCRIPTION_MAX)


def _item_line(session: Session, item: Item) -> str:
    assert item.id is not None
    goal = get_goal(session, item.id)
    parts = [item.status.value + (", paused" if item.paused else "")]
    parts.append(fmt_money(goal.list_price_cents) if goal else "no price yet")
    platforms = sorted({listing.platform for listing in live_listings(session, item.id)})
    if platforms:
        parts.append("on " + ", ".join(platforms))
    parts += _waiting_counts(session, item.id)
    return f"#{item.id} {_label(item)}: " + ", ".join(parts)


def _waiting_counts(session: Session, item_id: int) -> list[str]:
    approvals = len(_pending_approvals(session, item_id))
    escalations = len(_open_escalations(session, item_id))
    counts = [f"{approvals} approval(s) waiting"] if approvals else []
    return counts + ([f"{escalations} escalation(s) open"] if escalations else [])


def item_card(session: Session, clock: Clock, settings: Settings, item_id: int) -> Card:
    """The item at a glance with Pause or Resume; ItemNotFound when there is no such item."""
    item = get_item(session, item_id)
    goal = get_goal(session, item_id)
    status = item.status.value + (", paused" if item.paused else "")
    lines = [f"Status: {status}"]
    if goal is not None:
        lines.append(
            f"Price: {fmt_money(goal.list_price_cents)}, target "
            f"{fmt_money(goal.target_price_cents)}, floor {fmt_money(goal.floor_price_cents)}, "
            f"{goal.firmness.value}"
        )
        lines.append(f"Deadline: {fmt_local(settings, goal.deadline)}")
    fields = [
        *_listing_fields(session, item_id),
        *_deal_fields(session, clock, settings, active_commitment(session, item_id)),
        *_waiting_fields(session, settings, item_id),
        *_last_run_fields(session, settings, item_id),
        *_item_fields(session, settings, item_id),
    ]
    action = "resume" if item.paused else "pause"
    return _card(
        settings,
        f"#{item_id} {_label(item)}",
        "\n".join(lines),
        ITEM_COLOR,
        fields=fields,
        buttons=[Button(action.capitalize(), custom_id(action, item_id))],
    )


def _listing_fields(session: Session, item_id: int) -> list[CardField]:
    query = select(Listing).where(Listing.item_id == item_id).order_by(col(Listing.id))
    latest = {listing.platform: listing for listing in session.exec(query).all()}
    lines = [
        f"{platform}: {listing.status.value}" + (f", {listing.url}" if listing.url else "")
        for platform, listing in sorted(latest.items())
    ]
    return [CardField("Listings", "\n".join(lines))] if lines else []


def _deal_fields(
    session: Session, clock: Clock, settings: Settings, active: ActiveCommitment | None
) -> list[CardField]:
    if active is None:
        return []
    accepted = active.accepted
    lines = [
        f"#{accepted.id}: {_price(accepted)} with {buyer_name(session, accepted.buyer_id)}",
        f"Pickup: {_pickup_text(settings, active.pickup)}",
    ]
    rows = deal_rows(session, accepted)
    if len(rows) > 1:
        lines.insert(1, f"Bundle, {_total(rows)} in total: {_bundle_parts(session, rows)}")
    if accepted.undo_deadline is not None and clock.now() < accepted.undo_deadline:
        lines.append(f"Undo until {fmt_local(settings, accepted.undo_deadline)}")
    return [CardField("Deal", "\n".join(lines))]


def _waiting_fields(session: Session, settings: Settings, item_id: int) -> list[CardField]:
    lines = [
        f"Approval #{approval.id}: {_wants(session, settings, approval)}"
        for approval in _pending_approvals(session, item_id)
    ]
    lines += [f"Escalation #{e.id}: {e.reason}" for e in _open_escalations(session, item_id)]
    return [CardField("Waiting for you", "\n".join(lines))] if lines else []


def _wants(session: Session, settings: Settings, approval: PendingApproval) -> str:
    """The first line of what a parked call would do."""
    args = json.loads(approval.args_json)
    return describe_call(session, settings, approval.item_id, approval.tool, args).splitlines()[0]


def _last_run_fields(session: Session, settings: Settings, item_id: int) -> list[CardField]:
    query = (
        select(AgentRun)
        .where(AgentRun.item_id == item_id)
        .order_by(col(AgentRun.started_at).desc(), col(AgentRun.id).desc())
    )
    run = session.exec(query).first()
    if run is None:
        return []
    what = run.rationale or run.error or "no plan"
    return [CardField("Last agent run", f"{fmt_local(settings, run.started_at)}: {what}")]


def status_text(session: Session, settings: Settings, router: ProviderRouter) -> str:
    """Quota state and use today, and the trust tier of every action tried so far."""
    if router.providers:
        lines = [f"Quota: {router.state(Capability.REASONING).value}"]
    else:
        lines = ["Quota: no model configured (set GEMINI_API_KEY)"]
    trust = [
        f"{record.platform} {record.action}: {_tier_name(record.tier)} (executed "
        f"{record.executed}, corrected {record.corrected})"
        for record in trust_summary(session)
    ]
    lines += _quota_lines(router.quota, settings)
    lines += ["", "Trust:", *(trust or ["No trust records yet; defaults apply."])]
    return "\n".join(lines)


def _tier_name(tier: int) -> str:
    return TrustTier(tier).name.lower().replace("_", " ")


def _quota_lines(tracker: QuotaTracker, settings: Settings) -> list[str]:
    used: Counter[str] = Counter()
    for usage in tracker.usage_today():
        used[usage.provider] += usage.requests
    lines = []
    for provider, requests in sorted(used.items()):
        limit = settings.provider_limits.get(provider)
        of = f" of {limit.rpd}" if limit is not None else ""
        lines.append(f"{provider}: {requests}{of} requests today")
    return lines or ["No model requests today."]


# ---------------------------------------------------------------- digest


def next_digest_at(settings: Settings, now: datetime) -> datetime:
    """The next discord_digest_hour in the seller's time zone after now, as UTC."""
    zone = ZoneInfo(settings.timezone)
    local = now.astimezone(zone)
    at = time(settings.discord_digest_hour)
    due = datetime.combine(local.date(), at, tzinfo=zone)
    if due <= local:
        due = datetime.combine(local.date() + timedelta(days=1), at, tzinfo=zone)
    return due.astimezone(UTC)


def build_digest(session: Session, clock: Clock, settings: Settings) -> str:
    """Today so far, in the seller's time zone: sales, live negotiations, what the agent did,
    what waits for the owner, items going stale, and model requests used."""
    now = clock.now()
    since = _local_midnight(settings, now)
    local = to_local(settings, now)
    sections = [
        ("Sold today", _sold_today(session, since)),
        ("Live negotiations", _negotiations(session, settings, now)),
        ("Agent actions today", _agent_actions(session, since)),
        ("Waiting for you", _waiting(session, settings)),
        ("Going stale", _stale(session, settings, now)),
        ("Model requests today", _quota_lines(_tracker(session, clock, settings), settings)),
    ]
    blocks = ["\n".join([heading, *(lines or ["none"])]) for heading, lines in sections]
    text = "\n\n".join([f"Digest for {local:%a %b} {local.day}", *blocks])
    return redact_address(text, settings)


def _sold_today(session: Session, since: datetime) -> list[str]:
    query = select(Item).where(Item.status == ItemStatus.SOLD, col(Item.updated_at) >= since)
    lines = []
    for item in session.exec(query.order_by(col(Item.id))).all():
        deal = _latest(session, item.id, CommitmentKind.COMPLETED)
        sale = f": {_price(deal)} to {buyer_name(session, deal.buyer_id)}" if deal else ""
        lines.append(f"#{item.id} {_label(item)}{sale}")
    return lines


def _negotiations(session: Session, settings: Settings, now: datetime) -> list[str]:
    lines = []
    query = select(Item).where(col(Item.status).in_((ItemStatus.LISTED, ItemStatus.COMMITTED)))
    for item in session.exec(query.order_by(col(Item.id))).all():
        assert item.id is not None
        active = active_commitment(session, item.id)
        if active is not None:
            accepted = active.accepted
            name = buyer_name(session, accepted.buyer_id)
            pickup = _pickup_text(settings, active.pickup)
            deal = f"deal with {name} at {_price(accepted)}, pickup {pickup}"
            lines.append(f"#{item.id} {_label(item)}: {deal}")
        talking = _live_threads(session, item.id, now)
        if talking:
            names = ", ".join(_thread_line(session, thread) for thread in talking)
            lines.append(f"#{item.id} {_label(item)}: talking with {names}")
    return lines


def _live_threads(session: Session, item_id: int, now: datetime) -> list[Thread]:
    query = (
        select(Thread)
        .where(
            Thread.item_id == item_id,
            col(Thread.state).in_((ThreadState.OPEN, ThreadState.NEGOTIATING)),
            col(Thread.last_inbound_at) >= now - LIVE_THREAD_WINDOW,
        )
        .order_by(col(Thread.id))
    )
    return list(session.exec(query).all())


def _thread_line(session: Session, thread: Thread) -> str:
    name = buyer_name(session, thread.buyer_id)
    offer = thread.last_offer_cents
    return f"{name} (offered {fmt_money(offer)})" if offer else name


def _agent_actions(session: Session, since: datetime) -> list[str]:
    runs = session.exec(select(AgentRun).where(col(AgentRun.started_at) >= since)).all()
    if not runs:
        return []
    done: Counter[str] = Counter()
    for run in runs:
        for outcome in _outcomes(run):
            if outcome.get("status") == "executed":
                done[str(outcome.get("tool", "?")).replace("_", " ")] += 1
    lines = [f"{len(runs)} wake(s)"]
    lines += [f"{tool}: {count}" for tool, count in sorted(done.items())]
    return lines


def _outcomes(run: AgentRun) -> list[dict[str, Any]]:
    try:
        found = json.loads(run.results_json or "[]")
    except json.JSONDecodeError:
        return []
    return [entry for entry in found if isinstance(entry, dict)] if isinstance(found, list) else []


def _waiting(session: Session, settings: Settings) -> list[str]:
    approvals = session.exec(
        select(PendingApproval)
        .where(PendingApproval.status == ApprovalStatus.PENDING)
        .order_by(col(PendingApproval.id))
    ).all()
    lines = [
        f"Approval #{approval.id} on #{approval.item_id}: {_wants(session, settings, approval)}"
        for approval in approvals
    ]
    escalations = session.exec(
        select(Escalation).where(col(Escalation.resolved_at).is_(None)).order_by(col(Escalation.id))
    ).all()
    for escalation in escalations:
        on = f" on #{escalation.item_id}" if escalation.item_id else ""
        lines.append(f"Escalation #{escalation.id}{on}: {_clip(escalation.reason, 160)}")
    return lines


def _stale(session: Session, settings: Settings, now: datetime) -> list[str]:
    """Listed for idle_wake_days or more with no buyer message in that time."""
    quiet = timedelta(days=settings.idle_wake_days)
    query = select(Item).where(Item.status == ItemStatus.LISTED, col(Item.paused).is_(False))
    lines = []
    for item in session.exec(query.order_by(col(Item.id))).all():
        assert item.id is not None
        since = _listed_since(session, item.id)
        if since is None or now - since < quiet:
            continue
        last = _last_inbound(session, item.id)
        if last is not None and now - last < quiet:
            continue
        heard = f"last buyer message {(now - last).days} days ago" if last else "no buyer messages"
        listed = f"listed {(now - since).days} days, {total_views(session, item.id)} views"
        lines.append(f"#{item.id} {_label(item)}: {listed}, {heard}")
    return lines


def _listed_since(session: Session, item_id: int) -> datetime | None:
    published = [listing.published_at for listing in live_listings(session, item_id)]
    return min((when for when in published if when is not None), default=None)


def _last_inbound(session: Session, item_id: int) -> datetime | None:
    query = (
        select(Message.sent_at)
        .join(Thread, col(Thread.id) == Message.thread_id)
        .where(Thread.item_id == item_id, Message.direction == Direction.INBOUND)
        .order_by(col(Message.sent_at).desc())
    )
    return session.exec(query).first()


def _tracker(session: Session, clock: Clock, settings: Settings) -> QuotaTracker:
    engine = session.get_bind()
    assert isinstance(engine, Engine)
    return QuotaTracker(engine, clock, settings.provider_limits)


# ---------------------------------------------------------------- helpers


_REF_MODELS: dict[NoticeKind, type[SQLModel]] = {
    NoticeKind.COMMITMENT: Commitment,
    NoticeKind.ESCALATION: Escalation,
    NoticeKind.APPROVAL: PendingApproval,
    NoticeKind.SOLD: Commitment,
    NoticeKind.ALERT: Escalation,  # only a frozen scam conversation's alert has a ref_id
}


def _row[T](session: Session, model: type[T], row_id: int | None) -> T | None:
    return session.get(model, row_id) if row_id is not None else None


def _pending_approvals(session: Session, item_id: int) -> list[PendingApproval]:
    query = select(PendingApproval).where(
        PendingApproval.item_id == item_id, PendingApproval.status == ApprovalStatus.PENDING
    )
    return list(session.exec(query.order_by(col(PendingApproval.id))).all())


def _open_escalations(session: Session, item_id: int) -> list[Escalation]:
    query = select(Escalation).where(
        Escalation.item_id == item_id, col(Escalation.resolved_at).is_(None)
    )
    return list(session.exec(query.order_by(col(Escalation.id))).all())


def _latest(session: Session, item_id: int | None, kind: CommitmentKind) -> Commitment | None:
    query = select(Commitment).where(Commitment.item_id == item_id, Commitment.kind == kind)
    return session.exec(query.order_by(col(Commitment.id).desc())).first()


def _item_label(session: Session, item_id: int) -> str:
    item = session.get(Item, item_id)
    return _label(item) if item is not None else f"item #{item_id}"


def _label(item: Item) -> str:
    return item.title or item.hint or f"item #{item.id}"


def _price(commitment: Commitment) -> str:
    return fmt_money(commitment.price_cents or 0)


def _total(rows: Sequence[Commitment]) -> str:
    return fmt_money(sum(row.price_cents or 0 for row in rows))


def _bundle_parts(session: Session, rows: Sequence[Commitment]) -> str:
    """Each item of a bundle deal with its share, e.g. "IKEA desk (#1) $105.88"."""
    parts = []
    for row in rows:
        item = session.get(Item, row.item_id)
        title = (item.title or item.hint) if item is not None else "item"
        parts.append(f"{title} (#{row.item_id}) {_price(row)}")
    return ", ".join(parts)


def _pickup_text(settings: Settings, pickup: Commitment | None) -> str:
    if pickup is None or pickup.pickup_start is None:
        return "Not booked yet"
    start = fmt_local(settings, pickup.pickup_start)
    if pickup.pickup_end is None:
        return start
    return f"{start} to {to_local(settings, pickup.pickup_end):%H:%M}"


def _local_midnight(settings: Settings, now: datetime) -> datetime:
    local = to_local(settings, now)
    return datetime.combine(local.date(), time(), tzinfo=local.tzinfo).astimezone(UTC)




def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
