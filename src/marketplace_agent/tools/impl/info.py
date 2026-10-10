"""Read-only tools: get_stats and query_comps. Only query_comps writes, to the comps cache."""
from datetime import datetime, timedelta
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, StringConstraints
from sqlmodel import Session, col, select

from marketplace_agent.core.comps import (
    cached_comps,
    lookup_comps,
    normalize_query,
    save_comps,
    searching,
)
from marketplace_agent.core.models import Listing, ListingStatsSnapshot, ListingStatus
from marketplace_agent.core.stats import gained_since
from marketplace_agent.platforms.base import ListingStats
from marketplace_agent.tools.base import Allow, Reject, Tool, ToolContext, ToolResult, Verdict
from marketplace_agent.tools.impl.common import ToolArgs, fmt_local, fmt_money

WINDOWS = (("24h", timedelta(hours=24)), ("7d", timedelta(days=7)))
FRESH = timedelta(hours=1)  # older results say when they were searched
_METRICS = (("views", "views"), ("saves", "saves"), ("messages", "message_count"))


class NoArgs(ToolArgs):
    pass


class QueryCompsArgs(ToolArgs):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=100)]


class GetStatsTool(Tool):
    name: ClassVar[str] = "get_stats"
    description: ClassVar[str] = (
        "Views, saves, and buyer conversations for each listing of this item, with the change "
        "over the last 24 hours and 7 days and how long it has been up. Use it to tell whether "
        "the problem is traffic (presentation) or conversion (price)."
    )
    Args: ClassVar[type[BaseModel]] = NoArgs
    read_only: ClassVar[bool] = True

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        return Allow()

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        now = ctx.clock.now()
        query = (
            select(Listing)
            .where(Listing.item_id == ctx.item_id, col(Listing.published_at).is_not(None))
            .order_by(col(Listing.id))
        )
        rows = [_listing_stats(ctx.session, lst, now) for lst in ctx.session.exec(query).all()]
        summary = "\n".join(_stats_line(row) for row in rows) or "No listing has been posted yet."
        return ToolResult(ok=True, summary=summary, data={"listings": rows})


class QueryCompsTool(Tool):
    name: ClassVar[str] = "query_comps"
    description: ClassVar[str] = (
        "Search the connected marketplaces for comparable listings: count, median and middle-"
        "half asking prices, and a few examples. Use a short query like 'ikea malm desk'. These "
        "are asking prices, not sale prices; your own listings are left out. Results are kept "
        "for 24 hours, so asking again the same day returns the same answer."
    )
    Args: ClassVar[type[BaseModel]] = QueryCompsArgs
    read_only: ClassVar[bool] = True

    def check(self, ctx: ToolContext, args: BaseModel) -> Verdict:
        assert isinstance(args, QueryCompsArgs)
        if not normalize_query(args.query):
            return Reject("The query has no words to search for. Use a few keywords.")
        if searching(ctx.platforms) or cached_comps(ctx.session, args.query) is not None:
            return Allow()
        if not ctx.platforms:
            return Reject("No platform is connected, so there is nothing to search. Skip comps.")
        return Reject(
            f"{', '.join(ctx.platforms)} cannot search other sellers' listings, so there is "
            "nothing to search. Skip comps."
        )

    def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        assert isinstance(args, QueryCompsArgs)
        lookup = lookup_comps(ctx.session, ctx.clock, ctx.platforms, args.query)
        save_comps(ctx.session, lookup)
        comps = lookup.summary
        data: dict[str, Any] = {
            "query": lookup.query,
            "count": comps.count,
            "median_cents": comps.median_cents,
            "p25_cents": comps.p25_cents,
            "p75_cents": comps.p75_cents,
            "samples": [
                {key: sample.get(key) for key in ("title", "price_cents", "platform")}
                for sample in comps.samples
            ],
            "skipped": comps.skipped,
            "searched_at": lookup.fetched_at.isoformat() if lookup.fetched_at else None,
        }
        summary = comps.describe(lookup.query)
        if lookup.fetched_at is not None and ctx.clock.now() - lookup.fetched_at > FRESH:
            summary += f" Searched {fmt_local(ctx.settings, lookup.fetched_at)}."
        return ToolResult(ok=True, summary=summary, data=data)


def _listing_stats(session: Session, listing: Listing, now: datetime) -> dict[str, Any]:
    query = (
        select(ListingStatsSnapshot)
        .where(ListingStatsSnapshot.listing_id == listing.id)
        .order_by(col(ListingStatsSnapshot.captured_at), col(ListingStatsSnapshot.id))
    )
    snapshots = list(session.exec(query).all())
    latest = _counts(snapshots[-1]) if snapshots else None
    row: dict[str, Any] = {
        "listing_id": listing.id,
        "platform": listing.platform,
        "status": listing.status.value,
        "price_cents": listing.price_cents,
        "url": listing.url,
        "days_live": _days_live(listing, now),
    }
    for key, field in _METRICS:
        row[key] = getattr(latest, field) if latest else None
    for label, window in WINDOWS:
        gained = gained_since(
            snapshots,
            now - window,
            published_at=listing.published_at,
            relisted_at=listing.last_relisted_at,
        )
        for index, (key, _) in enumerate(_METRICS):
            row[f"{key}_{label}"] = gained[index] if gained is not None else None
    return row


def _counts(snapshot: ListingStatsSnapshot) -> ListingStats:
    return ListingStats(snapshot.views, snapshot.saves, snapshot.message_count)


def _days_live(listing: Listing, now: datetime) -> float | None:
    if listing.published_at is None:
        return None
    end = now
    if listing.status == ListingStatus.DELISTED and listing.delisted_at is not None:
        end = listing.delisted_at
    return round((end - listing.published_at) / timedelta(days=1), 1)


def _stats_line(row: dict[str, Any]) -> str:
    head = (
        f"{row['platform']} listing {row['listing_id']} ({row['status']}, "
        f"{fmt_money(row['price_cents'])}, {row['days_live']} days)"
    )
    metrics = [_metric_text(row, key) for key, _ in _METRICS]
    return f"{head}: " + "; ".join(metrics)


def _metric_text(row: dict[str, Any], key: str) -> str:
    if row[key] is None:
        return f"{key} not measured yet"
    changes = ", ".join(f"{_signed(row[f'{key}_{label}'])} in {label}" for label, _ in WINDOWS)
    return f"{key} {row[key]} ({changes})"


def _signed(value: int | None) -> str:
    return "n/a" if value is None else f"{value:+d}"
