"""Did the buyer come? After a booked pickup, only the owner knows.

A buyer who came and paid in person often never writes again, so silence after a pickup is no
evidence of a no-show. The runner asks the owner when the pickup ends, and the agent may treat
the buyer as a no-show only once the owner has had OWNER_ANSWER_WINDOW to answer.

| When | What happens |
|---|---|
| The booked pickup ends | A pickup check opens: an Escalation and a notice asking the owner |
| The owner runs `agent sold` | The deal completes; the check resolves |
| The owner runs `agent noshow` | The deal is released as a no-show; the check resolves |
| No answer by pickup end + OWNER_ANSWER_WINDOW | The agent may release it as a no-show |
| The deal closes another way, or the pickup is rebooked | The check resolves |

A check's reason starts "Pickup check (deal #<accepted id>, pickup #<booking id>)", which is how
it is found again; Escalation has no other field for it. A bundle gets one check, on its first
row: its rows share one booking slot, and `agent sold` or `agent noshow` on any item closes all.
"""

import re
from datetime import datetime, timedelta

from sqlmodel import Session, col, select

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import Buyer, Commitment, CommitmentKind, Escalation, Item
from marketplace_agent.guard.invariants import to_local
from marketplace_agent.guard.ledger import ActiveCommitment, active_pickups, deal_rows, is_active
from marketplace_agent.notify.base import Notice, NoticeKind, Notifier

OWNER_ANSWER_WINDOW = timedelta(hours=6)  # from pickup end; the owner was there for it
_PREFIX = "Pickup check"
_CHECK = re.compile(r"^Pickup check \(deal #(\d+), pickup #(\d+)\)")


def no_show_from(settings: Settings, pickup_end: datetime) -> datetime:
    """When the agent may call a silent buyer a no-show, if the owner has not answered."""
    grace = timedelta(minutes=settings.no_show_grace_minutes)
    return pickup_end + max(grace, OWNER_ANSWER_WINDOW)


def ask_owner(session: Session, clock: Clock, settings: Settings, notifier: Notifier) -> int:
    """Open a pickup check for every active deal whose booked pickup has ended and has none, and
    resolve checks that no longer apply. Flushes; the caller commits. Returns how many opened."""
    now = clock.now()
    asked = set()
    for escalation in _checks(session):
        deal_id, pickup_id = _ids(escalation)
        asked.add((deal_id, pickup_id))  # answered or not, never asked twice
        if escalation.resolved_at is None and not _current(session, deal_id, pickup_id):
            _resolve(escalation, now, "The deal closed or the pickup was rebooked.")
            session.add(escalation)
    opened = 0
    for active in active_pickups(session):
        pickup = active.pickup
        assert pickup is not None
        if pickup.pickup_end is None or pickup.pickup_end > now:
            continue
        if (active.accepted.id, pickup.id) in asked:
            continue
        if deal_rows(session, active.accepted)[0].id != active.accepted.id:
            continue  # a bundle is asked about once, on its first row
        _open(session, settings, notifier, active, now)
        opened += 1
    session.flush()
    return opened


def resolve_checks(session: Session, clock: Clock, item_id: int, answer: str) -> None:
    """The owner answered for this item's deal."""
    for escalation in _open_checks(session, item_id):
        _resolve(escalation, clock.now(), answer)
        session.add(escalation)


def open_check(session: Session, accepted_id: int) -> Escalation | None:
    """The open pickup check for a deal (any row of a bundle), if the owner is being asked."""
    accepted = session.get(Commitment, accepted_id)
    ids = {row.id for row in deal_rows(session, accepted)} if accepted else {accepted_id}
    for escalation in _open_checks(session):
        if _ids(escalation)[0] in ids:
            return escalation
    return None


def check_ids(escalation: Escalation) -> tuple[int, int] | None:
    """(deal id, pickup id) a pickup check asks about; None for any other escalation."""
    return _ids(escalation) if _CHECK.match(escalation.reason) else None


def still_asked(session: Session, escalation: Escalation) -> bool:
    """The check is open, and its deal and booking are still the ones in play: an answer to it
    concerns the deal it named, not one agreed since."""
    ids = check_ids(escalation)
    return ids is not None and escalation.resolved_at is None and _current(session, *ids)


def _open(
    session: Session,
    settings: Settings,
    notifier: Notifier,
    active: ActiveCommitment,
    now: datetime,
) -> None:
    accepted, pickup = active.accepted, active.pickup
    assert pickup is not None and pickup.pickup_start is not None and pickup.pickup_end is not None
    buyer = session.get(Buyer, accepted.buyer_id) if accepted.buyer_id is not None else None
    name = (buyer.display_name or buyer.handle) if buyer is not None else "the buyer"
    names = []
    for row in deal_rows(session, accepted):
        item = session.get(Item, row.item_id)
        names.append((item.title or item.hint) if item is not None else f"item {row.item_id}")
    what = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    goods, it = ("the item", "it") if len(names) == 1 else ("the bundle", "them")
    start = to_local(settings, pickup.pickup_start)
    when = f"{start:%a %b} {start.day} {start:%H:%M}"
    deadline = to_local(settings, no_show_from(settings, pickup.pickup_end))
    until = f"{deadline:%a %b} {deadline.day} {deadline:%H:%M}"
    escalation = Escalation(
        item_id=accepted.item_id,
        thread_id=accepted.thread_id,
        reason=(
            f"{_PREFIX} (deal #{accepted.id}, pickup #{pickup.id}): the owner is asked whether "
            f"{name} collected {goods} at the {when} pickup. Without an answer, a no-show "
            f"release is allowed from {until}."
        ),
        created_at=now,
    )
    session.add(escalation)
    session.flush()
    notifier.notify(
        Notice(
            kind=NoticeKind.ESCALATION,
            title=f"Did {name} pick up {what}?",
            body=(
                f"The {when} pickup is over. If {name} took {it}, run `agent sold "
                f"{accepted.item_id}`. If they did not come, run `agent noshow "
                f"{accepted.item_id}`. Without an answer, the agent may treat them as a no-show "
                f"from {until}."
            ),
            item_id=accepted.item_id,
            ref_id=escalation.id,
        )
    )


def _open_checks(session: Session, item_id: int | None = None) -> list[Escalation]:
    return [e for e in _checks(session, item_id) if e.resolved_at is None]


def _checks(session: Session, item_id: int | None = None) -> list[Escalation]:
    query = select(Escalation).where(col(Escalation.reason).startswith(_PREFIX))
    if item_id is not None:
        query = query.where(Escalation.item_id == item_id)
    found = session.exec(query.order_by(col(Escalation.id))).all()
    return [escalation for escalation in found if _CHECK.match(escalation.reason)]


def _ids(escalation: Escalation) -> tuple[int, int]:
    found = _CHECK.match(escalation.reason)
    assert found is not None
    return int(found[1]), int(found[2])


def _current(session: Session, deal_id: int, pickup_id: int) -> bool:
    """The deal is still active and the asked-about booking is still its latest."""
    accepted = session.get(Commitment, deal_id)
    if accepted is None or not is_active(session, accepted):
        return False
    query = select(Commitment.id).where(
        Commitment.ref_id == deal_id, Commitment.kind == CommitmentKind.PICKUP_BOOKED
    )
    latest = max((booking for booking in session.exec(query).all() if booking), default=0)
    return latest == pickup_id


def _resolve(escalation: Escalation, now: datetime, answer: str) -> None:
    escalation.resolved_at = now
    escalation.resolution = answer
