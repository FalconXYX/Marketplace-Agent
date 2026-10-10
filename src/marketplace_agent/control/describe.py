"""A parked tool call in plain words, for the owner: the Discord approval card and the web page
both show "Accept $110.00 from Lucas K.", never "thread id: 1"."""

from collections.abc import Callable, Mapping
from typing import Any

from sqlmodel import Session

from marketplace_agent.config import Settings
from marketplace_agent.core.items import get_goal
from marketplace_agent.core.models import Item, Thread
from marketplace_agent.guard.invariants import parse_local_datetime
from marketplace_agent.tools.impl.common import buyer_name, fmt_local, fmt_money

QUOTE_CHARS = 300


def describe_call(
    session: Session, settings: Settings, item_id: int, tool: str, args: Mapping[str, Any]
) -> str:
    """A parked tool call in plain words, e.g. "Change the asking price from $120.00 to $105.00"."""
    describe = _CALLS.get(tool)
    if describe is not None:
        try:
            return describe(session, settings, item_id, args)
        except (KeyError, TypeError, ValueError):
            pass
    pairs = ", ".join(f"{key.replace('_', ' ')} {value}" for key, value in sorted(args.items()))
    return f"Run {tool.replace('_', ' ')}" + (f" with {pairs}" if pairs else "")


def _send_message(session: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    return f"Send {_thread_buyer(session, args['thread_id'])} this message:\n{quote(args['body'])}"


def _update_price(session: Session, _s: Settings, item_id: int, args: Mapping[str, Any]) -> str:
    new = fmt_money(int(args["price_cents"]))
    goal = get_goal(session, item_id)
    if goal is None:
        return f"Change the asking price to {new}"
    return f"Change the asking price from {fmt_money(goal.list_price_cents)} to {new}"


def _rewrite_listing(_db: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    parts, quoted = [], []
    if args.get("title"):
        parts.append(f'title "{args["title"]}"')
    if args.get("description"):
        parts.append("description")
        quoted.append(quote(args["description"]))
    if args.get("photo_order"):
        parts.append("photo order " + ", ".join(str(p) for p in args["photo_order"]))
    head = "Rewrite the listing: new " + ", new ".join(parts) if parts else "Rewrite the listing"
    return "\n".join([head, *quoted])


def _publish(_db: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    return f"Post the item on {args['platform']}"


def _relist(_db: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    return f"Relist the item on {args['platform']}"


def _accept_offer(session: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    price = fmt_money(int(args["price_cents"]))
    return f"Accept {price} from {_thread_buyer(session, args['thread_id'])}"


def _accept_bundle(session: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    total = fmt_money(int(args["total_cents"]))
    items = []
    for item_id in args["item_ids"]:
        item = session.get(Item, int(item_id))
        items.append(f"{(item.title or item.hint) if item else 'item'} (#{int(item_id)})")
    buyer = _thread_buyer(session, args["thread_id"])
    return f"Accept {total} from {buyer} for a bundle of {', '.join(items)}"


def _book_pickup(session: Session, settings: Settings, _i: int, args: Mapping[str, Any]) -> str:
    when = fmt_local(settings, parse_local_datetime(settings, str(args["start"])))
    minutes = int(args.get("duration_minutes", 60))
    buyer = _thread_buyer(session, args["thread_id"])
    return f"Book {buyer}'s pickup for {when} ({minutes} minutes)"


def _release(session: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    how = " as a no-show" if args.get("no_show") else ""
    buyer = _thread_buyer(session, args["thread_id"])
    return f"Release the deal with {buyer}{how}: {args['reason']}"


def _complete_sale(session: Session, _s: Settings, _i: int, args: Mapping[str, Any]) -> str:
    return f"Mark the sale to {_thread_buyer(session, args['thread_id'])} as done"


_CALLS: dict[str, Callable[[Session, Settings, int, Mapping[str, Any]], str]] = {
    "send_message": _send_message,
    "update_price": _update_price,
    "rewrite_listing": _rewrite_listing,
    "publish": _publish,
    "relist": _relist,
    "accept_offer": _accept_offer,
    "accept_bundle": _accept_bundle,
    "book_pickup": _book_pickup,
    "release_commitment": _release,
    "complete_sale": _complete_sale,
}


def _thread_buyer(session: Session, thread_id: Any) -> str:
    thread = session.get(Thread, int(thread_id))
    return buyer_name(session, thread.buyer_id) if thread is not None else "the buyer"


def quote(text: str) -> str:
    """Text as a quote ("> " per line), cut to QUOTE_CHARS."""
    clipped = text.strip()
    if len(clipped) > QUOTE_CHARS:
        clipped = clipped[: QUOTE_CHARS - 3] + "..."
    return "\n".join(f"> {line}" for line in clipped.splitlines())
