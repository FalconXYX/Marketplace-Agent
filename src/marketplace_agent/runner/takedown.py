"""Sold means gone everywhere: convergent takedown and one courtesy reply per thread. No model.

A Listing still LIVE, PENDING, or AWAITING_HUMAN on a SOLD or WITHDRAWN item is owed a takedown.
complete_sale, mark_sold, and withdraw try once; retry_takedowns keeps trying every tick until
it holds.

| Outcome | Listing | Owner |
|---|---|---|
| Delisted | DELISTED, last_error cleared | INFO if told to remove it by hand earlier |
| ListingGone | DELISTED (mark_gone): already down is done | |
| No remote id: never posted | DELISTED locally | |
| No remote id: AWAITING_HUMAN assist form | DELISTED locally | ALERT: do not click Post |
| Any other failure | Stays up, retried with backoff | ALERT at failure ALERT_AFTER, once |

Retry state lives in Listing.last_error so it survives a restart (models are frozen), as
"takedown <failures> next <ISO time>[ alerted]: <error>"; alerted means the owner was told to
remove it by hand. A failed takedown at the moment of sale or withdrawal is stored with
sale_takedown_failed: no retry failures yet, due at once, alerted (that path sent the ALERT).
Any other last_error reads as no failures yet. After n failures the next try waits backoff(n):
a tick, 2, 4 ... capped at MAX_BACKOFF.

Courtesy replies: a buyer who writes about a SOLD or WITHDRAWN item gets COURTESY_REPLIES' fixed
text once per thread, sent as "system" through guard.validate_courtesy (frozen thread, flagged
buyer, outbound cap, streak, and text rules all apply). A buyer who holds or bought the item,
or whose deal the owner's withdrawal cancelled, is left to the owner (messaging.had_deal); one
whose deal was released or undone earlier gets the reply like anyone else. Only messages from
the last COURTESY_WINDOW count, so an old thread is never revived.
"""

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlmodel import Session, col, select

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import (
    Buyer,
    Direction,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    Thread,
    ThreadState,
)
from marketplace_agent.guard.invariants import redact_address
from marketplace_agent.guard.validate import validate_courtesy
from marketplace_agent.notify.base import Notice, NoticeKind, Notifier
from marketplace_agent.platforms.base import (
    ListingGone,
    PlatformAdapter,
    PlatformError,
    PlatformRef,
)
from marketplace_agent.tools.base import Allow, Tool, ToolContext
from marketplace_agent.tools.impl.common import mark_gone
from marketplace_agent.tools.impl.messaging import (
    COURTESY_VIA,
    SendMessageTool,
    courtesy_reply,
    had_deal,
)

CLOSED_ITEM = (ItemStatus.SOLD, ItemStatus.WITHDRAWN)
OWED = (ListingStatus.LIVE, ListingStatus.PENDING, ListingStatus.AWAITING_HUMAN)
ALERT_AFTER = 3
MAX_BACKOFF = timedelta(hours=6)
SHORTEST_TICK = 0.8  # of tick_seconds: run_forever's jitter is +-20%, so one tick never waits less
COURTESY_WINDOW = timedelta(days=7)

_STATE = re.compile(r"takedown (\d+) next (\S+?)( alerted)?: (.*)", re.DOTALL)

logger = logging.getLogger("marketplace_agent.runner")


@dataclass(frozen=True)
class TakedownState:
    failures: int = 0
    next_at: datetime | None = None
    error: str = ""
    alerted: bool = False  # the owner was told to remove it by hand


@dataclass
class TakedownReport:
    taken_down: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def takedown_state(listing: Listing) -> TakedownState:
    """The retry state stored in listing.last_error; none yet for any other text."""
    match = _STATE.fullmatch(listing.last_error or "")
    if match is None:
        return TakedownState()
    try:
        next_at = datetime.fromisoformat(match[2])
    except ValueError:
        return TakedownState()
    return TakedownState(int(match[1]), next_at, match[4], match[3] is not None)


def encode_state(state: TakedownState) -> str:
    assert state.next_at is not None
    alerted = " alerted" if state.alerted else ""
    return f"takedown {state.failures} next {state.next_at.isoformat()}{alerted}: {state.error}"


def sale_takedown_failed(clock: Clock, listing: Listing, error: str) -> None:
    """complete_sale, mark_sold, or withdraw could not take listing down and sent the owner an
    ALERT to remove it by hand: retry at once, and tell the owner once it is down."""
    listing.last_error = encode_state(TakedownState(0, clock.now(), error, alerted=True))


def listing_error(listing: Listing) -> str | None:
    """listing.last_error for the owner: retry state in words, any other text as stored."""
    match = _STATE.fullmatch(listing.last_error or "")
    if match is None:
        return listing.last_error
    failures = int(match[1])
    tries = f" ({failures} more tries failed)" if failures > 1 else ""
    tries = " (1 more try failed)" if failures == 1 else tries
    return f"Could not take it down: {match[4]}{tries}. The agent keeps trying."


def backoff(failures: int, tick: timedelta) -> timedelta:
    """Wait after the n-th consecutive failure: tick * 2^(n-1), at most MAX_BACKOFF."""
    if failures < 1:
        return timedelta(0)
    return min(tick * 2 ** min(failures - 1, 20), MAX_BACKOFF)


def owed_takedowns(session: Session) -> list[Listing]:
    query = (
        select(Listing)
        .join(Item, col(Item.id) == Listing.item_id)
        .where(col(Listing.status).in_(OWED), col(Item.status).in_(CLOSED_ITEM))
        .order_by(col(Listing.id))
    )
    return list(session.exec(query).all())


def _owed(session: Session, listing_id: int) -> Listing | None:
    """The listing as stored now (another session may have taken it down meanwhile)."""
    listing = session.get(Listing, listing_id, populate_existing=True)
    return listing if listing is not None and listing.status in OWED else None


def retry_takedowns(
    session: Session,
    clock: Clock,
    settings: Settings,
    platforms: Mapping[str, PlatformAdapter],
    notifier: Notifier,
    *,
    tick: timedelta | None = None,
) -> TakedownReport:
    """Try every owed takedown that is due. Commits after each listing, so no write lock is held
    during a platform call. tick defaults to the shortest gap between two runner ticks."""
    tick = tick if tick is not None else timedelta(seconds=settings.tick_seconds * SHORTEST_TICK)
    report = TakedownReport()
    now = clock.now()
    for listing_id in [listing.id for listing in owed_takedowns(session)]:
        assert listing_id is not None
        listing = _owed(session, listing_id)
        if listing is None:
            continue
        state = takedown_state(listing)
        if state.next_at is not None and now < state.next_at:
            continue
        item = session.get(Item, listing.item_id)
        assert item is not None
        if listing.remote_id is None:
            _drop_unposted(clock, notifier, item, listing)
            report.taken_down += 1
        else:
            error = _delist(clock, platforms, listing)
            if error is None:
                _done(clock, notifier, item, listing, state)
                report.taken_down += 1
            else:
                error = redact_address(error, settings)
                _failed(clock, notifier, item, listing, state, error, tick)
                report.failed += 1
                report.errors.append(f"take down {listing.platform} listing {listing.id}: {error}")
        session.add(listing)
        session.commit()
    return report


def _delist(clock: Clock, platforms: Mapping[str, PlatformAdapter], listing: Listing) -> str | None:
    """None when the post is down (or already gone), else why it may still be up."""
    assert listing.remote_id is not None
    adapter = platforms.get(listing.platform)
    if adapter is None:
        return f"{listing.platform} is not connected"
    try:
        adapter.delist(PlatformRef(listing.platform, listing.remote_id, listing.url))
    except ListingGone as exc:
        mark_gone(clock, listing, exc)
        return None
    except PlatformError as exc:
        return str(exc) or type(exc).__name__
    except Exception as exc:  # a bug in one adapter must not stop the others
        logger.exception("takedown of listing %s failed", listing.id)
        return f"{type(exc).__name__}: {exc}"
    listing.status = ListingStatus.DELISTED
    listing.delisted_at = clock.now()
    listing.last_error = None
    return None


def _done(
    clock: Clock, notifier: Notifier, item: Item, listing: Listing, before: TakedownState
) -> None:
    if listing.status != ListingStatus.DELISTED:  # mark_gone sets it; be explicit otherwise
        listing.status = ListingStatus.DELISTED
        listing.delisted_at = clock.now()
    if before.alerted or before.failures >= ALERT_AFTER:
        notifier.notify(
            Notice(
                kind=NoticeKind.INFO,
                title=f"The {listing.platform} listing is down",
                body=f"{_name(item)}: the agent took the {listing.platform} listing down after "
                f"all ({_where(listing)}). No need to remove it by hand.",
                item_id=item.id,
            )
        )


def _failed(
    clock: Clock,
    notifier: Notifier,
    item: Item,
    listing: Listing,
    before: TakedownState,
    error: str,
    tick: timedelta,
) -> None:
    failures = before.failures + 1
    now = clock.now()
    alerted = before.alerted or failures >= ALERT_AFTER
    state = TakedownState(failures, now + backoff(failures, tick), error, alerted)
    listing.last_error = encode_state(state)
    if failures == ALERT_AFTER:
        notifier.notify(
            Notice(
                kind=NoticeKind.ALERT,
                title=f"Remove the {listing.platform} listing by hand",
                body=f"{_name(item)} is {item.status.value}, but its {listing.platform} listing "
                f"is still up: {failures} tries to take it down failed ({error}). Remove it by "
                f"hand: {_where(listing)}. The agent keeps trying and will say once it is down.",
                item_id=item.id,
            )
        )


def _drop_unposted(clock: Clock, notifier: Notifier, item: Item, listing: Listing) -> None:
    """Nothing to take down by API: the post never went up, or it waits for the owner's click."""
    waiting = listing.status == ListingStatus.AWAITING_HUMAN
    listing.status = ListingStatus.DELISTED
    listing.delisted_at = clock.now()
    listing.last_error = f"never posted; the item is {item.status.value}"
    if waiting:
        notifier.notify(
            Notice(
                kind=NoticeKind.ALERT,
                title=f"Do not post {_name(item)} on {listing.platform}",
                body=f"{_name(item)} is {item.status.value}. Close the prepared "
                f"{listing.platform} form without clicking Post; if you already posted it, "
                f"delete it on {listing.platform}.",
                item_id=item.id,
            )
        )


def _where(listing: Listing) -> str:
    return listing.url or f"{listing.platform} post {listing.remote_id}"


def _name(item: Item) -> str:
    return item.title or item.hint or f"Item {item.id}"


# ---------------------------------------------------------------- courtesy replies


def courtesy_candidates(
    session: Session, clock: Clock, platforms: Mapping[str, PlatformAdapter]
) -> list[int]:
    """Thread ids owed a courtesy reply: a closed, unpaused item; a connected platform; a thread
    not frozen, from an unflagged buyer, with an unhandled message from the last COURTESY_WINDOW,
    no courtesy reply yet, and no deal with this buyer on the item."""
    recent = select(Message.thread_id).where(
        Message.direction == Direction.INBOUND,
        col(Message.handled).is_(False),
        col(Message.sent_at) >= clock.now() - COURTESY_WINDOW,
    )
    replied = select(Message.thread_id).where(
        Message.direction == Direction.OUTBOUND, Message.via == COURTESY_VIA
    )
    query = (
        select(Thread)
        .join(Item, col(Item.id) == Thread.item_id)
        .join(Buyer, col(Buyer.id) == Thread.buyer_id)
        .where(
            col(Item.status).in_(CLOSED_ITEM),
            col(Item.paused).is_(False),
            Thread.state != ThreadState.FROZEN,
            col(Buyer.flagged_scam).is_(False),
            col(Thread.platform).in_(list(platforms)),
            col(Thread.id).in_(recent),
            col(Thread.id).not_in(replied),
        )
        .order_by(col(Thread.id))
    )
    threads = session.exec(query).all()
    return [t.id for t in threads if t.id is not None and not had_deal(session, t)]


def send_courtesy_replies(
    session: Session,
    clock: Clock,
    settings: Settings,
    platforms: Mapping[str, PlatformAdapter],
    notifier: Notifier,
    tool: Tool | None = None,
) -> tuple[int, list[str]]:
    """Send every owed courtesy reply. Returns (sent, errors). Commits after each thread; a
    failed send rolls back and leaves the thread for the next tick."""
    tool = tool or SendMessageTool()
    sent, errors = 0, []
    for thread_id in courtesy_candidates(session, clock, platforms):
        try:
            if _courtesy(session, clock, settings, platforms, notifier, tool, thread_id):
                sent += 1
            session.commit()
        except PlatformError as exc:
            session.rollback()
            errors.append(f"courtesy reply in thread {thread_id}: {exc}")
        except Exception as exc:  # one broken thread must not stop the others
            session.rollback()
            logger.exception("courtesy reply in thread %s failed", thread_id)
            errors.append(f"courtesy reply in thread {thread_id}: {exc!r}")
    return sent, errors


def _courtesy(
    session: Session,
    clock: Clock,
    settings: Settings,
    platforms: Mapping[str, PlatformAdapter],
    notifier: Notifier,
    tool: Tool,
    thread_id: int,
) -> bool:
    thread = session.get(Thread, thread_id)
    assert thread is not None and thread.item_id is not None
    item = session.get(Item, thread.item_id)
    assert item is not None
    body = courtesy_reply(item.status)
    assert body is not None
    args = tool.parse_args({"thread_id": thread_id, "body": body})
    ctx = ToolContext(
        session=session,
        clock=clock,
        settings=settings,
        platforms=platforms,
        notifier=notifier,
        item_id=item.id,
        actor="system",
    )
    verdict = validate_courtesy(ctx, tool, args)
    if not isinstance(verdict, Allow):
        logger.info("courtesy reply in thread %s not sent: %s", thread_id, verdict)
        return False
    result = tool.execute(ctx, args)
    message = session.get(Message, result.data["message_id"])
    assert message is not None
    message.via = COURTESY_VIA
    if thread.state not in (ThreadState.FROZEN, ThreadState.COMMITTED):
        thread.state = ThreadState.CLOSED  # over, so a buyer who does not answer is no ghost
        session.add(thread)
    unhandled = select(Message).where(
        Message.thread_id == thread_id,
        Message.direction == Direction.INBOUND,
        col(Message.handled).is_(False),
    )
    for inbound in session.exec(unhandled).all():
        inbound.handled = True
        session.add(inbound)
    session.add(message)
    return True
