"""Form input for the web UI: dollars, and the draft form that confirms an item for sale.

Fields arrive as typed, so a rejected form goes back to the owner exactly as they left it.
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import get_args

from pydantic import BaseModel

from marketplace_agent.ai.drafting import TITLE_MAX_CHARS, Condition, ListingDraft
from marketplace_agent.core.items import (
    MAX_PRICE_CENTS,
    check_goal_prices,
    get_attributes,
    parse_hint_price,
)
from marketplace_agent.core.models import Firmness, Goal, Item
from marketplace_agent.tools.impl.common import NEEDS_CONDITION

CONDITIONS: tuple[str, ...] = get_args(Condition)
DEFAULT_DEADLINE_DAYS = 14
MAX_DEADLINE_DAYS = 3650
_DOLLARS = re.compile(r"\d+(?:\.\d{1,2})?")
_SECONDS_PER_DAY = 86_400


class FormError(ValueError):
    """Input the owner must fix; the message says how."""


class DraftFields(BaseModel):
    """The draft form as submitted. attributes holds one "name: value" per line."""

    title: str = ""
    description: str = ""
    category: str = ""
    condition: str = ""
    attributes: str = ""
    price: str = ""
    floor: str = ""
    target: str = ""
    firmness: str = Firmness.SLIGHT.value
    deadline_days: str = str(DEFAULT_DEADLINE_DAYS)
    notes: str = ""
    platforms: list[str] = []


@dataclass(frozen=True)
class ConfirmedDraft:
    title: str
    description: str
    category: str | None
    condition: str | None
    attributes: dict[str, str]
    list_price_cents: int
    floor_price_cents: int
    target_price_cents: int | None
    firmness: Firmness
    deadline_days: int
    notes: str
    platforms: list[str]


def parse_dollars(text: str, label: str) -> int:
    """ "$1,200.50" -> 120050. Raises FormError naming the field."""
    cleaned = text.strip().removeprefix("$").replace(",", "").strip()
    if not _DOLLARS.fullmatch(cleaned):
        raise FormError(f"{label}: enter dollars, like 120 or 120.50.")
    cents = int(Decimal(cleaned) * 100)
    if cents <= 0:
        raise FormError(f"{label} must be more than $0.")
    if cents > MAX_PRICE_CENTS:
        raise FormError(f"{label} is too high.")
    return cents


def optional_dollars(text: str, label: str) -> int | None:
    return parse_dollars(text, label) if text.strip() else None


def dollars_value(cents: int | None) -> str:
    """Cents as an input value: 12000 -> "120", 12050 -> "120.50"."""
    if cents is None:
        return ""
    whole, rest = divmod(cents, 100)
    return f"{whole}.{rest:02d}" if rest else str(whole)


def prefill(
    item: Item,
    goal: Goal | None,
    enabled_platforms: Sequence[str],
    now: datetime,
    suggestion: ListingDraft | None = None,
) -> DraftFields:
    """The form for a draft: the item's fields, then the goal if one is set. Without a goal the
    list price comes from the hint ("desk $120"), else from the draft's suggestion."""
    fields = DraftFields(
        title=item.title,
        description=item.description,
        category=item.category or "",
        condition=item.condition or "",
        attributes="\n".join(f"{key}: {value}" for key, value in get_attributes(item).items()),
        platforms=list(enabled_platforms),
    )
    if goal is None:
        fields.price = dollars_value(parse_hint_price(item.hint) or _suggested(suggestion))
        return fields
    fields.price = dollars_value(goal.list_price_cents)
    fields.floor = dollars_value(goal.floor_price_cents)
    fields.target = dollars_value(goal.target_price_cents)
    fields.firmness = goal.firmness.value
    fields.deadline_days = str(_days_left(goal.deadline, now))
    fields.notes = goal.notes
    return fields


def check_draft(
    fields: DraftFields, enabled_platforms: Sequence[str]
) -> ConfirmedDraft | dict[str, str]:
    """The confirmed draft, or the problems keyed by field name."""
    errors: dict[str, str] = {}

    def field[T](name: str, parse: Callable[[], T]) -> T | None:
        try:
            return parse()
        except FormError as exc:
            errors[name] = str(exc)
            return None

    title = field("title", lambda: _title(fields.title))
    condition = field("condition", lambda: _condition(fields.condition))
    attributes = field("attributes", lambda: parse_attributes(fields.attributes))
    price = field("price", lambda: parse_dollars(fields.price, "Price"))
    floor = field("floor", lambda: parse_dollars(fields.floor, "Floor"))
    target = field("target", lambda: optional_dollars(fields.target, "Target"))
    firmness = field("firmness", lambda: parse_firmness(fields.firmness))
    days = field("deadline_days", lambda: _deadline_days(fields.deadline_days))
    platforms = field("platforms", lambda: _platforms(fields.platforms, enabled_platforms))
    if price is not None and floor is not None and "target" not in errors:
        field("floor", lambda: _goal_prices(price, floor, target))
    if "condition" not in errors and condition is None:
        needs = [name for name in platforms or () if name in NEEDS_CONDITION]
        if needs:
            errors["condition"] = (
                f"{', '.join(n.capitalize() for n in needs)} needs a condition. Pick one, or "
                "untick it."
            )
    if errors:
        return errors
    assert title is not None and attributes is not None and price is not None
    assert floor is not None and firmness is not None and days is not None
    assert platforms is not None
    return ConfirmedDraft(
        title=title,
        description=fields.description.strip(),
        category=fields.category.strip() or None,
        condition=condition,
        attributes=attributes,
        list_price_cents=price,
        floor_price_cents=floor,
        target_price_cents=target,
        firmness=firmness,
        deadline_days=days,
        notes=fields.notes.strip(),
        platforms=platforms,
    )


def parse_attributes(text: str) -> dict[str, str]:
    """ "Brand: IKEA" per line. Blank lines are skipped; a later duplicate name wins."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, colon, value = line.partition(":")
        if not colon or not name.strip():
            raise FormError(f'Write each detail as "name: value", one per line (not "{line}").')
        found[name.strip()] = value.strip()
    return found


def parse_firmness(text: str) -> Firmness:
    try:
        return Firmness(text)
    except ValueError as exc:
        choices = ", ".join(level.value for level in Firmness)
        raise FormError(f"Firmness must be one of: {choices}.") from exc


def _title(text: str) -> str:
    title = " ".join(text.split())
    if not title:
        raise FormError("Give it a title.")
    if len(title) > TITLE_MAX_CHARS:
        raise FormError(f"Keep the title under {TITLE_MAX_CHARS} characters.")
    return title


def _condition(text: str) -> str | None:
    if not text:
        return None
    return parse_condition(text)


def parse_condition(text: str) -> str:
    if text not in CONDITIONS:
        raise FormError("Pick a condition from the list.")
    return text


def _deadline_days(text: str) -> int:
    if not text.strip().isdigit() or not 1 <= int(text) <= MAX_DEADLINE_DAYS:
        raise FormError(f"Days to sell: a whole number from 1 to {MAX_DEADLINE_DAYS}.")
    return int(text)


def _platforms(chosen: Sequence[str], enabled: Sequence[str]) -> list[str]:
    if not enabled:
        raise FormError("No platform is enabled. Set enabled_platforms first.")
    picked = [name for name in enabled if name in chosen]
    if not picked:
        raise FormError("Pick at least one platform.")
    return picked


def _goal_prices(price: int, floor: int, target: int | None) -> None:
    try:
        check_goal_prices(price, floor, price if target is None else target)
    except ValueError as exc:
        raise FormError(str(exc).capitalize() + ".") from exc


def _suggested(suggestion: ListingDraft | None) -> int | None:
    if suggestion is None:
        return None
    return suggestion.suggested_price_high_cents or suggestion.suggested_price_low_cents


def _days_left(deadline: datetime, now: datetime) -> int:
    seconds = (deadline - now).total_seconds()
    return min(MAX_DEADLINE_DAYS, max(1, math.ceil(seconds / _SECONDS_PER_DAY)))
