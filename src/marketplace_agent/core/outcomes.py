"""One line of facts per finished item, kept as a global memory so later drafts and wakes can
calibrate against what really happened. Facts only: the model draws the lessons.

| Fact | Source |
|---|---|
| What | Item title (else the hint) and category, in [brackets]: the part recall matches on |
| Final price | The owner's stated price, else the COMPLETED deal's; asking price when withdrawn |
| Platforms | Every platform the item was posted on |
| List price at publish | The asking price before the first change after the first post |
| Days | First post to now |
| Price changes | PriceChange rows since the first post |
| Buyers who wrote | Distinct buyers with an inbound message on the item |

The owner's price for an item sold in a bundle is what the whole bundle fetched, split over its
items in proportion to the deal's prices, so the lines add up to the cash.

Written once per item when it becomes SOLD or WITHDRAWN (control.mark_sold and withdraw,
complete_sale). Calling again is a no-op, except that a withdrawn item the owner then marks sold
gets its one line rewritten: the last state is the fact.
"""
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, col, func, select

from marketplace_agent.core.clock import Clock
from marketplace_agent.core.items import get_goal
from marketplace_agent.guard.ledger import allocate
from marketplace_agent.core.models import (
    Commitment,
    CommitmentKind,
    Direction,
    Item,
    ItemStatus,
    Listing,
    Memory,
    MemoryScope,
    Message,
    PriceChange,
    Thread,
)
from marketplace_agent.memory.store import MAX_LESSON_CHARS, OUTCOME_SOURCE

SUBJECT_MIN_CHARS = 20


def record_outcome(
    session: Session, clock: Clock, item_id: int, *, final_price_cents: int | None = None
) -> Memory | None:
    """Write the item's outcome line, or rewrite a "Withdrawn unsold" one now that it is SOLD.
    None when it is not SOLD or WITHDRAWN, or its line already says how it ended.
    final_price_cents: what the owner says it sold for, which beats the ledger's price. Flushes;
    the caller commits."""
    item = session.get(Item, item_id)
    if item is None or item.status not in (ItemStatus.SOLD, ItemStatus.WITHDRAWN):
        return None
    lesson = outcome_line(session, clock, item, final_price_cents=final_price_cents)
    memory = outcome_of(session, item_id)
    if memory is not None and _head(memory.lesson) == _head(lesson):
        return None
    if memory is None:
        memory = Memory(
            scope=MemoryScope.GLOBAL,
            item_id=item_id,
            lesson=lesson,
            source=OUTCOME_SOURCE,
            created_at=clock.now(),
        )
    else:  # withdrawn, then sold after all: one line per item, and the sale is the fact
        memory.lesson, memory.created_at = lesson, clock.now()
    session.add(memory)
    session.flush()
    return memory


def record_outcomes(
    session: Session, clock: Clock, item_id: int, *, final_price_cents: int | None = None
) -> list[Memory]:
    """record_outcome for the item and for every item that shared a bundle deal with it and is
    finished too (closing one bundled item closes the bundle). When item_id's sale completed a
    bundle, final_price_cents is what the owner got for all of it: each item is recorded with
    its share, split in proportion to the deal's prices."""
    prices = _owner_prices(session, item_id, final_price_cents)
    written = [record_outcome(session, clock, item_id, final_price_cents=prices.get(item_id))]
    for other in bundle_partners(session, item_id):
        written.append(record_outcome(session, clock, other, final_price_cents=prices.get(other)))
    return [memory for memory in written if memory is not None]


def _owner_prices(session: Session, item_id: int, total_cents: int | None) -> dict[int, int]:
    """The owner's price per item: all of it for item_id, or its split over a sold bundle."""
    if total_cents is None:
        return {}
    done = _completed(session, item_id)
    accepted = session.get(Commitment, done.ref_id) if done and done.ref_id is not None else None
    if accepted is None or accepted.bundle_key is None:
        return {item_id: total_cents}
    rows = session.exec(
        select(Commitment)
        .where(
            Commitment.bundle_key == accepted.bundle_key,
            Commitment.kind == CommitmentKind.ACCEPTED,
        )
        .order_by(col(Commitment.id))
    ).all()
    weights = [max(row.price_cents or 0, 0) for row in rows]
    if len(rows) < 2 or sum(weights) <= 0:
        return {item_id: total_cents}
    shares = allocate(total_cents, weights)
    return {row.item_id: share for row, share in zip(rows, shares, strict=True)}


def outcome_of(session: Session, item_id: int) -> Memory | None:
    query = select(Memory).where(
        Memory.scope == MemoryScope.GLOBAL,
        Memory.source == OUTCOME_SOURCE,
        Memory.item_id == item_id,
    )
    return session.exec(query).first()


def bundle_partners(session: Session, item_id: int) -> list[int]:
    """Other items in any bundle deal this item was part of."""
    keys = select(Commitment.bundle_key).where(
        Commitment.item_id == item_id,
        Commitment.kind == CommitmentKind.ACCEPTED,
        col(Commitment.bundle_key).is_not(None),
    )
    query = (
        select(Commitment.item_id)
        .where(col(Commitment.bundle_key).in_(keys), Commitment.item_id != item_id)
        .distinct()
        .order_by(col(Commitment.item_id))
    )
    return list(session.exec(query).all())


def outcome_line(
    session: Session, clock: Clock, item: Item, *, final_price_cents: int | None = None
) -> str:
    """The facts, one line, at most MAX_LESSON_CHARS; the subject is shortened to fit."""
    assert item.id is not None
    listings = session.exec(
        select(Listing)
        .where(Listing.item_id == item.id, col(Listing.published_at).is_not(None))
        .order_by(col(Listing.published_at), col(Listing.id))
    ).all()
    first_post = _utc(listings[0].published_at) if listings else None
    platforms = list(dict.fromkeys(listing.platform for listing in listings))
    changes = _price_changes(session, item.id, first_post)
    goal = get_goal(session, item.id)
    asking = goal.list_price_cents if goal else None
    listed_at = changes[0].old_cents if changes else asking

    sold = item.status == ItemStatus.SOLD
    said: list[str] = []
    if sold:
        price, bundle = _sale(session, item.id)
        price = final_price_cents if final_price_cents is not None else price
        said.append(f"for {_dollars(price)}" if price is not None else "at an unrecorded price")
        if bundle > 1:
            said.append(f"as its share of a {bundle}-item bundle")
    if first_post is not None:
        said += [f"on {' and '.join(platforms)}", f"after {_days(clock.now() - first_post)}"]
    else:
        said.append("(never posted)")
    facts = [f"listed at {_dollars(listed_at)}" if listed_at is not None else "no asking price"]
    if not sold and asking is not None and asking != listed_at:
        facts.append(f"last asked {_dollars(asking)}")
    facts.append(_count(len(changes), "price change"))
    facts.append(_count(_buyers_who_wrote(session, item.id), "buyer") + " wrote")

    head = "Sold" if sold else "Withdrawn unsold"
    tail = f" {' '.join(said)}: {', '.join(facts)}."
    room = MAX_LESSON_CHARS - len(head) - len(" []") - len(tail)
    return f"{head} [{_subject(item, max(room, SUBJECT_MIN_CHARS))}]{tail}"[:MAX_LESSON_CHARS]


def _price_changes(
    session: Session, item_id: int, since: datetime | None
) -> list[PriceChange]:
    query = select(PriceChange).where(PriceChange.item_id == item_id)
    if since is not None:
        query = query.where(col(PriceChange.created_at) >= since)
    rows = session.exec(query.order_by(col(PriceChange.created_at), col(PriceChange.id))).all()
    return list(rows)


def _completed(session: Session, item_id: int) -> Commitment | None:
    """The item's latest COMPLETED row."""
    return session.exec(
        select(Commitment)
        .where(Commitment.item_id == item_id, Commitment.kind == CommitmentKind.COMPLETED)
        .order_by(col(Commitment.created_at).desc(), col(Commitment.id).desc())
    ).first()


def _head(lesson: str) -> str:
    """How the item ended, as its outcome line opens: "Sold" or "Withdrawn unsold"."""
    return lesson.split(" [", 1)[0]


def _sale(session: Session, item_id: int) -> tuple[int | None, int]:
    """The latest COMPLETED deal's price and how many items its bundle held (1 when none)."""
    done = _completed(session, item_id)
    if done is None:
        return None, 1
    accepted = session.get(Commitment, done.ref_id) if done.ref_id is not None else None
    price = done.price_cents
    if price is None and accepted is not None:
        price = accepted.price_cents
    key = (accepted.bundle_key if accepted else None) or done.bundle_key
    if key is None:
        return price, 1
    size = session.exec(
        select(func.count(func.distinct(Commitment.item_id))).where(Commitment.bundle_key == key)
    ).one()
    return price, max(int(size), 1)


def _buyers_who_wrote(session: Session, item_id: int) -> int:
    query = (
        select(func.count(func.distinct(Thread.buyer_id)))
        .select_from(Thread)
        .join(Message, col(Message.thread_id) == col(Thread.id))
        .where(Thread.item_id == item_id, Message.direction == Direction.INBOUND)
    )
    return int(session.exec(query).one())


def _subject(item: Item, limit: int) -> str:
    title = " ".join((item.title or item.hint).split()) or f"item {item.id}"
    category = " ".join((item.category or "").split())
    text = f"{title}, {category}" if category else title
    text = text.replace("[", "(").replace("]", ")")
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rsplit(" ", 1)[0].rstrip(",;:")
    return (cut or text[: limit - 3]) + "..."


def _dollars(cents: int) -> str:
    whole, rest = divmod(cents, 100)
    return f"${whole:,}" if not rest else f"${whole:,}.{rest:02d}"


def _days(span: timedelta) -> str:
    days = round(max(span, timedelta()) / timedelta(days=1), 1)
    return _count(days, "day")


def _count(n: float, noun: str) -> str:
    shown = f"{n:g}"
    return f"{shown} {noun}" if n == 1 else f"{shown} {noun}s"


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
