"""Wakes the agent schedules for itself, stored as WakeRequest rows dated in the future.

| WakeRequest | created_at | Fires (runner/triggers.py) |
|---|---|---|
| From the owner (control) | When made | At once, kind human_override |
| From the agent (schedule_wake) | The time it asked for | Once that time comes, kind scheduled |

Either is consumed by the first wake after it is due.
"""
from datetime import datetime

from sqlmodel import Session, col, select

from marketplace_agent.core.models import WakeRequest

SCHEDULED_PREFIX = "[scheduled] "


def is_scheduled(request: WakeRequest) -> bool:
    return request.reason.startswith(SCHEDULED_PREFIX)


def scheduled_reason(request: WakeRequest) -> str:
    return request.reason.removeprefix(SCHEDULED_PREFIX)


def pending_scheduled(session: Session, item_id: int) -> list[WakeRequest]:
    """The item's scheduled wakes not yet consumed, soonest first."""
    query = (
        select(WakeRequest)
        .where(
            WakeRequest.item_id == item_id,
            col(WakeRequest.consumed_at).is_(None),
            col(WakeRequest.reason).startswith(SCHEDULED_PREFIX),
        )
        .order_by(col(WakeRequest.created_at), col(WakeRequest.id))
    )
    return list(session.exec(query).all())


def schedule(session: Session, item_id: int, at: datetime, reason: str, now: datetime) -> None:
    """One scheduled wake per item: a new one replaces any still pending."""
    for request in pending_scheduled(session, item_id):
        request.consumed_at = now
        session.add(request)
    session.add(WakeRequest(item_id=item_id, reason=SCHEDULED_PREFIX + reason, created_at=at))
    session.flush()
