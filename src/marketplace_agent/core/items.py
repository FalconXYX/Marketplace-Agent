"""Item, photo, and goal services shared by the CLI, web intake, and agent tools."""
import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlmodel import Session, select

from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import Firmness, Goal, Item, ItemStatus, Photo


class ItemNotFound(LookupError):
    pass


MAX_PRICE_CENTS = 1_000_000_000  # $10M: anything above is a typo, and overflows SQLite past 2**63

# An amount counts as a price only when marked as money: "$120", "CAD 75", "120 dollars", "90 obo".
# French Canadian groups thousands with a space, NBSP or narrow NBSP: "1 200 $".
_PRICE = re.compile(
    r"""
    (?:
        (?P<prefix> (?:CAD|USD|CA|US|C)?\$ | (?<![a-z])(?:CAD|USD|asking)\b:? ) \s*
      | (?<![\w.,])
    )
    (?P<amount>
        \d{1,3}(?:[\   ]\d{3})+(?:[.,]\d{1,2})?(?![^\W\d_])
      | \d{1,3}(?:,\d{3})+(?:\.\d{1,2})?
      | \d+(?:\.\d{1,2})?
    )
    (?P<thousands> k\b )?
    (?![.,]?\d)
    (?P<suffix> \s*(?: \$(?!\s*\d) | (?:dollars?|bucks|cad|usd|obo)\b ) )?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_GROUP_SPACE = re.compile(r"[\   ]")


def create_item(session: Session, clock: Clock, hint: str) -> Item:
    now = clock.now()
    item = Item(hint=hint.strip(), created_at=now, updated_at=now)
    session.add(item)
    session.flush()
    return item


def get_item(session: Session, item_id: int) -> Item:
    item = session.get(Item, item_id)
    if item is None:
        raise ItemNotFound(f"Item {item_id} not found")
    return item


def list_items(session: Session, status: ItemStatus | None = None) -> list[Item]:
    query = select(Item).order_by(Item.id)
    if status is not None:
        query = query.where(Item.status == status)
    return list(session.exec(query).all())


def item_photos(session: Session, item_id: int) -> list[Photo]:
    query = select(Photo).where(Photo.item_id == item_id).order_by(Photo.position, Photo.id)
    return list(session.exec(query).all())


def parse_hint_price(hint: str) -> int | None:
    """First positive amount in the hint that is marked as money, in cents.

    None rather than a misread when that amount is ambiguous or implausible:

    | Hint | Why |
    |---|---|
    | "1.200" | $1.20, or 1200 with a European separator |
    | "1 200 $" | $1,200 in French, but "iPhone 12 400$" is $400 |
    | "$99999999999" | Above MAX_PRICE_CENTS |
    """
    for match in _PRICE.finditer(hint):
        if not (match["prefix"] or match["suffix"]):
            continue  # bare number: "2 drawers", "iPhone 12"
        text = match["amount"]
        if _GROUP_SPACE.search(text):
            if not match["prefix"]:
                return None
            digits = _GROUP_SPACE.sub("", text).replace(",", ".")  # "1 200,50 $" uses a comma
        else:
            digits = text.replace(",", "")
        cents = Decimal(digits) * (100_000 if match["thousands"] else 100)
        if cents > MAX_PRICE_CENTS:
            return None
        if cents > 0:
            return int(cents)
    return None


def get_attributes(item: Item) -> dict[str, str]:
    raw: dict[str, object] = json.loads(item.attributes_json or "{}")
    return {str(key): str(value) for key, value in raw.items() if value is not None}


def set_attributes(item: Item, attrs: dict[str, str]) -> None:
    """Replace all attributes. Blank values are dropped: unknown is absent, not empty."""
    kept = {key: value.strip() for key, value in attrs.items() if value.strip()}
    item.attributes_json = json.dumps(kept, ensure_ascii=False, sort_keys=True)


def set_goal(
    session: Session,
    clock: Clock,
    item_id: int,
    *,
    list_price_cents: int,
    floor_price_cents: int,
    target_price_cents: int | None = None,
    deadline: datetime | None = None,
    deadline_days: int = 14,
    firmness: Firmness = Firmness.SLIGHT,
    notes: str = "",
) -> Goal:
    """Create or replace the item's goal. Flushes; the caller commits."""
    item = get_item(session, item_id)
    target = list_price_cents if target_price_cents is None else target_price_cents
    check_goal_prices(list_price_cents, floor_price_cents, target)
    now = clock.now()
    due = check_deadline(deadline or now + timedelta(days=deadline_days), now)
    fields = {
        "list_price_cents": list_price_cents,
        "floor_price_cents": floor_price_cents,
        "target_price_cents": target,
        "deadline": due,
        "firmness": firmness,
        "notes": notes,
    }
    goal = get_goal(session, item_id)
    if goal is None:
        goal = Goal(item_id=item_id, **fields)
    else:
        goal.sqlmodel_update(fields)
    session.add(goal)
    touch(item, clock)
    session.flush()
    return goal


def get_goal(session: Session, item_id: int) -> Goal | None:
    return session.exec(select(Goal).where(Goal.item_id == item_id)).first()


def touch(item: Item, clock: Clock) -> None:
    item.updated_at = clock.now()


def check_goal_prices(list_cents: int, floor_cents: int, target_cents: int) -> None:
    for name, cents in (("list", list_cents), ("floor", floor_cents), ("target", target_cents)):
        if cents <= 0:
            raise ValueError(f"{name} price must be greater than zero")
    if floor_cents > list_cents:
        raise ValueError("floor price cannot be above the list price")
    if floor_cents > target_cents:
        raise ValueError("floor price cannot be above the target price")


def check_deadline(deadline: datetime, now: datetime) -> datetime:
    if deadline.tzinfo is None:
        raise ValueError("deadline must be timezone-aware")
    if deadline <= now:
        raise ValueError("deadline must be in the future")
    return deadline.astimezone(UTC)
