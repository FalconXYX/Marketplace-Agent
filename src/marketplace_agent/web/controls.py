"""The owner's controls: each route is one ControlService call, answered with the panel it came
from, the outcome shown inline."""

from typing import Annotated

from fastapi import APIRouter, Form, Request, Response
from sqlmodel import Session

from marketplace_agent.control.service import ControlService, takedown_note
from marketplace_agent.core.models import Commitment, Escalation, PendingApproval, Thread
from marketplace_agent.tools.impl.common import fmt_money
from marketplace_agent.web.forms import (
    optional_dollars,
    parse_condition,
    parse_dollars,
    parse_firmness,
)
from marketplace_agent.web.pages import RecordId
from marketplace_agent.web.ui import CONDITION_LABELS, FIRMNESS_LABELS, Action, get_ui

router = APIRouter()

Text = Annotated[str, Form()]
CheckId = Annotated[int | None, Form(ge=1, le=2**63 - 1)]  # the pickup check a deal's answer is for
Record = Commitment | PendingApproval | Escalation | Thread
RECORDS: dict[type[Record], tuple[str, str]] = {  # its name for the owner, where outcomes show
    Commitment: ("deal", "deal"),
    PendingApproval: ("approval", "approval-{}"),
    Escalation: ("escalation", "escalation-{}"),
    Thread: ("conversation", "thread-{}"),
}


# ---------------------------------------------------------------- deals, approvals, escalations


@router.post("/commitments/{deal_id}/undo")
def undo(request: Request, deal_id: RecordId) -> Response:
    return _on_record(request, Commitment, deal_id, lambda c: c.undo_commitment(deal_id))


@router.post("/approvals/{approval_id}/approve")
def approve(request: Request, approval_id: RecordId) -> Response:
    return _on_record(request, PendingApproval, approval_id, lambda c: c.approve(approval_id))


@router.post("/approvals/{approval_id}/reject")
def reject(request: Request, approval_id: RecordId, note: Text = "") -> Response:
    def run(control: ControlService) -> str:
        control.reject(approval_id, note)
        return "Rejected. The agent will re-plan" + (" with your note." if note.strip() else ".")

    return _on_record(request, PendingApproval, approval_id, run)


@router.post("/escalations/{escalation_id}/resolve")
def resolve(request: Request, escalation_id: RecordId, resolution: Text = "") -> Response:
    def run(control: ControlService) -> str:
        control.resolve_escalation(escalation_id, resolution)
        return "Resolved."

    return _on_record(request, Escalation, escalation_id, run)


@router.post("/threads/{thread_id}/reply")
def reply(request: Request, thread_id: RecordId, body: Text = "") -> Response:
    return _on_record(request, Thread, thread_id, lambda c: c.reply_as_human(thread_id, body))


# ---------------------------------------------------------------- item controls


@router.post("/items/{item_id}/price")
def price(request: Request, item_id: RecordId, price: Text = "") -> Response:
    return _on_item(request, item_id, lambda c: c.set_price(item_id, parse_dollars(price, "Price")))


@router.post("/items/{item_id}/floor")
def floor(request: Request, item_id: RecordId, floor: Text = "") -> Response:
    def run(control: ControlService) -> str:
        cents = parse_dollars(floor, "Floor")
        control.set_floor(item_id, cents)
        return f"Floor set to {fmt_money(cents)}. The agent will re-plan."

    return _on_item(request, item_id, run)


@router.post("/items/{item_id}/firmness")
def firmness(request: Request, item_id: RecordId, firmness: Text = "") -> Response:
    def run(control: ControlService) -> str:
        level = parse_firmness(firmness)
        control.set_firmness(item_id, level)
        return f"Firmness set to {FIRMNESS_LABELS[level].lower()}. The agent will re-plan."

    return _on_item(request, item_id, run)


@router.post("/items/{item_id}/condition")
def condition(request: Request, item_id: RecordId, condition: Text = "") -> Response:
    def run(control: ControlService) -> str:
        control.set_condition(item_id, parse_condition(condition))
        return f"Condition set to {CONDITION_LABELS[condition].lower()}. New posts carry it."

    return _on_item(request, item_id, run)


@router.post("/items/{item_id}/pause")
def pause(request: Request, item_id: RecordId) -> Response:
    def run(control: ControlService) -> str:
        control.pause(item_id)
        return "Paused. The agent does nothing on this item until you resume it."

    return _on_item(request, item_id, run)


@router.post("/items/{item_id}/resume")
def resume(request: Request, item_id: RecordId) -> Response:
    def run(control: ControlService) -> str:
        control.resume(item_id)
        return "Resumed. The agent is back on it."

    return _on_item(request, item_id, run)


@router.post("/items/{item_id}/sold")
def sold(
    request: Request,
    item_id: RecordId,
    price: Text = "",
    anchor: Text = "controls",
    check: CheckId = None,
) -> Response:
    """Also the pickup check's "Picked up" answer, sent from the deal with anchor=deal and the
    check's id, so a page left open cannot answer for a deal agreed since."""

    def run(control: ControlService) -> str:
        still_up = control.mark_sold(item_id, optional_dollars(price, "Sale price"), check_id=check)
        return "Marked sold." + takedown_note(still_up)

    return _on_item(request, item_id, run, anchor)


@router.post("/items/{item_id}/noshow")
def no_show(request: Request, item_id: RecordId, check: CheckId = None) -> Response:
    """The pickup check's "No-show" answer, with the check's id as for sold."""
    return _on_item(request, item_id, lambda c: c.mark_no_show(item_id, check_id=check), "deal")


@router.post("/items/{item_id}/withdraw")
def withdraw(request: Request, item_id: RecordId) -> Response:
    def run(control: ControlService) -> str:
        return "Withdrawn." + takedown_note(control.withdraw(item_id))

    return _on_item(request, item_id, run)


# ---------------------------------------------------------------- helpers


def _on_item(request: Request, item_id: int, action: Action, anchor: str = "controls") -> Response:
    return get_ui(request).act(request, item_id, anchor, action)


def _on_record(request: Request, model: type[Record], record_id: int, action: Action) -> Response:
    """Act on a deal, approval, escalation, or conversation, answering with its item's panel, or
    the home page's when it concerns no item."""
    ui = get_ui(request)
    label, anchor = RECORDS[model]
    with Session(ui.runtime.engine) as session:
        record = session.get(model, record_id)
    if record is None:
        return ui.message(request, 404, f"There is no {label} with id {record_id}.")
    return ui.act(request, record.item_id, anchor.format(record_id), action)
