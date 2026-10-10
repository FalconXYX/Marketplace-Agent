"""Comparable listings: what other sellers ask for similar items. No model calls.

Searches every connected platform, drops our own posts and unpriced ones, and keeps a summary
per normalized query in CompsCache for 24 hours. Asking prices, not sale prices.

| Function | Network | Writes |
|---|---|---|
| fetch_comps | when the cache is stale | the cache |
| lookup_comps | when the cache is stale | nothing (save it with save_comps) |
| cached_comps | no | nothing |
| item_comps | no | nothing |
| refresh_due | stale queries of LISTED items | the cache |

A search that no platform answered is never cached, so the next call retries it; meanwhile a
stale summary is better evidence than none and is returned with its age. An adapter that cannot
search declares `searches = False` (Facebook: scraping its search risks the account) and is not
asked at all: its empty answer would say nothing about the market, yet read as "none for sale".
"""
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Any

from sqlalchemy.dialects.sqlite import insert
from sqlmodel import Session, col, select

from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import CompsCache, Item, ItemStatus, Listing
from marketplace_agent.platforms.base import PlatformAdapter, PlatformError, SearchResult

logger = logging.getLogger(__name__)

MAX_AGE_HOURS = 24
SEARCH_LIMIT = 20       # results asked of each platform
MAX_SAMPLES = 5
SAMPLE_TITLE_CHARS = 80
QUERY_WORDS = 6         # long titles find nothing; the first few keywords find the item type
QUERY_MAX_CHARS = 100
REFRESH_BATCH = 5       # searches per refresh_due call, so one tick never fans out unbounded

_NOT_WORD = re.compile(r"[^\w]+|_+")
_MONEY = re.compile(
    r"(?:(?<![a-z])(?:cad|usd|ca|us|c))?\$\s*\d[\d,.]*k?\b"
    r"|\b\d[\d,.]*\s*(?:\$|k\b|dollars?\b|bucks\b|cad\b|usd\b)"
    r"|\basking\s*\$?\s*\d[\d,.]*",
    re.IGNORECASE,
)
_FILLER = frozenset(
    "a an and the of for with in on at to by from or is it its this that my our your "
    "obo firm asking negotiable price sale selling sell must go great good excellent nice "
    "condition used new like mint perfect barely gently lightly works working pickup only "
    "cash please".split()
)


@dataclass
class CompsSummary:
    count: int
    median_cents: int | None
    p25_cents: int | None
    p75_cents: int | None
    samples: list[dict] = field(default_factory=list)  # title, price_cents, platform, url
    skipped: list[str] = field(default_factory=list)    # "platform: why it was not searched"

    @classmethod
    def empty(cls, skipped: Sequence[str] = ()) -> "CompsSummary":
        return cls(0, None, None, None, [], list(skipped))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "CompsSummary":
        """A damaged or foreign row reads as no evidence rather than an error."""
        try:
            data = json.loads(raw or "{}")
            summary = cls(
                count=int(data.get("count", 0)),
                median_cents=_int_or_none(data.get("median_cents")),
                p25_cents=_int_or_none(data.get("p25_cents")),
                p75_cents=_int_or_none(data.get("p75_cents")),
                samples=[
                    {**s, "price_cents": int(s["price_cents"])}
                    for s in data.get("samples", [])
                    if isinstance(s, dict) and "price_cents" in s
                ],
                skipped=[str(s) for s in data.get("skipped", [])],
            )
        except (TypeError, ValueError, AttributeError):
            return cls.empty()
        prices = (summary.median_cents, summary.p25_cents, summary.p75_cents)
        if summary.count < 0 or (summary.count and None in prices):
            return cls.empty()
        return summary

    def describe(self, query: str) -> str:
        """One paragraph for a prompt or a tool result."""
        skipped = f" Not searched: {'; '.join(self.skipped)}." if self.skipped else ""
        median, low, high = self.median_cents, self.p25_cents, self.p75_cents
        if not self.count or median is None or low is None or high is None:
            return f"No comparable listings found for {query!r}.{skipped}"
        examples = "; ".join(
            f"{s.get('title', '?')} ({money(s['price_cents'])}, {s.get('platform', '?')})"
            for s in self.samples
        )
        noun = "listing" if self.count == 1 else "listings"
        return (
            f"{self.count} comparable {noun} for {query!r}: median {money(median)}, middle half "
            f"{money(low)} to {money(high)}. Examples: {examples}.{skipped}"
        )


@dataclass
class CompsLookup:
    query: str                    # normalized; the cache key and what was searched
    summary: CompsSummary
    fetched_at: datetime | None   # None: nothing searched or cached
    searched: bool                # a platform answered just now; save_comps stores it


def searching(platforms: Mapping[str, PlatformAdapter]) -> dict[str, PlatformAdapter]:
    """The platforms comps asks: all but adapters that say `searches = False`."""
    return {
        name: adapter
        for name, adapter in platforms.items()
        if getattr(adapter, "searches", True) is not False
    }


def normalize_query(q: str) -> str:
    """Casefolded words separated by single spaces: the cache key and the search text."""
    words = _NOT_WORD.sub(" ", q.casefold()).split()
    text = " ".join(words)
    return text[:QUERY_MAX_CHARS].rsplit(" ", 1)[0] if len(text) > QUERY_MAX_CHARS else text


def comps_query(text: str) -> str:
    """A short search for free text such as a hint or a title: money and filler words dropped,
    the first QUERY_WORDS keywords kept. A bare number of 3+ digits is dropped as a likely price
    or year ("desk 120"); shorter ones are often a model ("iphone 12"). Empty when nothing
    searchable is left."""
    words = normalize_query(_MONEY.sub(" ", text)).split()
    kept = [w for w in words if w not in _FILLER and not (w.isdigit() and len(w) > 2)]
    return " ".join(kept[:QUERY_WORDS])


def comps_query_for_item(item: Item) -> str:
    """From the title once drafted, else from the owner's hint."""
    return comps_query(item.title) or comps_query(item.hint)


def cached_comps(session: Session, query: str) -> tuple[CompsSummary, datetime] | None:
    """The stored summary for query and when it was fetched, whatever its age."""
    row = _row(session, normalize_query(query))
    if row is None:
        return None
    return CompsSummary.from_json(row.summary_json), _utc(row.fetched_at)


def item_comps(session: Session, item: Item) -> CompsLookup | None:
    """The cached comps to show beside an item's price fields: its title's search, else its
    hint's (not drafted yet, or drafted with nothing connected to search the new title).
    Whatever their age; None when neither was ever searched."""
    for query in dict.fromkeys((comps_query(item.title), comps_query(item.hint))):
        cached = cached_comps(session, query) if query else None
        if cached is not None:
            return CompsLookup(query, cached[0], cached[1], False)
    return None


def lookup_comps(
    session: Session,
    clock: Clock,
    platforms: Mapping[str, PlatformAdapter],
    query: str,
    *,
    max_age_hours: float = MAX_AGE_HOURS,
) -> CompsLookup:
    """The cached summary while fresh, else a new search. Writes nothing, so a caller about to
    make a slow call (a model request) does not hold the database write lock meanwhile."""
    key = normalize_query(query)
    if not key:
        return CompsLookup(key, CompsSummary.empty(), None, False)
    cached = cached_comps(session, key)
    now = clock.now()
    if cached is not None and now - cached[1] <= timedelta(hours=max_age_hours):
        return CompsLookup(key, cached[0], cached[1], False)
    summary, answered = _search(session, platforms, key)
    if answered:
        return CompsLookup(key, summary, now, True)
    if cached is not None:
        return CompsLookup(key, cached[0], cached[1], False)
    return CompsLookup(key, summary, None, False)


def save_comps(session: Session, lookup: CompsLookup) -> None:
    """Store a fresh search; anything else is already stored or not worth storing. Flushes."""
    if not lookup.searched or lookup.fetched_at is None:
        return
    values = {
        "query": lookup.query,
        "fetched_at": lookup.fetched_at,
        "summary_json": lookup.summary.to_json(),
    }
    upsert = insert(CompsCache).values(**values)
    upsert = upsert.on_conflict_do_update(
        index_elements=[CompsCache.query],
        set_={"fetched_at": values["fetched_at"], "summary_json": values["summary_json"]},
    )
    session.execute(upsert)
    session.flush()


def fetch_comps(
    session: Session,
    clock: Clock,
    platforms: Mapping[str, PlatformAdapter],
    query: str,
    *,
    max_age_hours: float = MAX_AGE_HOURS,
) -> CompsSummary:
    """Comps for query, from the cache while fresh. Flushes a new search; the caller commits."""
    lookup = lookup_comps(session, clock, platforms, query, max_age_hours=max_age_hours)
    save_comps(session, lookup)
    return lookup.summary


def refresh_due(
    session: Session,
    clock: Clock,
    platforms: Mapping[str, PlatformAdapter],
    *,
    max_age_hours: float = MAX_AGE_HOURS,
    batch: int = REFRESH_BATCH,
) -> int:
    """Search again for LISTED, unpaused items whose comps are missing or stale, never-searched
    and oldest first, at most batch searches. Stops at the first search no platform answers:
    they are down, and the next call retries. All searches run before any write. Returns how
    many summaries were refreshed; the caller commits."""
    if not searching(platforms) or batch <= 0:
        return 0
    items = session.exec(
        select(Item).where(Item.status == ItemStatus.LISTED, col(Item.paused).is_(False))
    ).all()
    queries = dict.fromkeys(q for q in (comps_query_for_item(i) for i in items) if q)
    now = clock.now()
    due: list[tuple[datetime | None, str]] = []
    for query in queries:
        cached = cached_comps(session, query)
        if cached is None or now - cached[1] > timedelta(hours=max_age_hours):
            due.append((cached[1] if cached else None, query))
    due.sort(key=lambda pair: (pair[0] is not None, pair[0] or now, pair[1]))
    fresh: list[CompsLookup] = []
    for _, query in due[:batch]:
        summary, answered = _search(session, platforms, query)
        if not answered:
            break
        fresh.append(CompsLookup(query, summary, now, True))
    for lookup in fresh:
        save_comps(session, lookup)
    return len(fresh)


def percentile(ordered: Sequence[int], q: Fraction | float) -> int | None:
    """Linear interpolation between closest ranks, rounded half up to a cent. ordered must be
    sorted ascending."""
    if not ordered:
        return None
    position = (len(ordered) - 1) * Fraction(q)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    exact = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return math.floor(exact + Fraction(1, 2))


def summarize(
    found: Sequence[tuple[str, SearchResult]], skipped: Sequence[str] = ()
) -> CompsSummary:
    """found: (platform, result) pairs, any order."""
    ordered = sorted(found, key=lambda pair: (pair[1].price_cents, pair[1].title))
    prices = [result.price_cents for _, result in ordered]
    samples = [
        {
            "title": _clip(result.title),
            "price_cents": result.price_cents,
            "platform": platform,
            "url": result.url,
        }
        for platform, result in _spread(ordered, MAX_SAMPLES)
    ]
    return CompsSummary(
        count=len(prices),
        median_cents=percentile(prices, Fraction(1, 2)),
        p25_cents=percentile(prices, Fraction(1, 4)),
        p75_cents=percentile(prices, Fraction(3, 4)),
        samples=samples,
        skipped=list(skipped),
    )


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def _search(
    session: Session, platforms: Mapping[str, PlatformAdapter], query: str
) -> tuple[CompsSummary, bool]:
    """Every searching platform's results for query, ours and unpriced ones dropped. answered is
    False when no platform could be searched."""
    own = {url for url in session.exec(select(Listing.url)).all() if url}
    found: list[tuple[str, SearchResult]] = []
    seen: set[str] = set()
    skipped: list[str] = []
    answered = False
    for name, adapter in searching(platforms).items():
        try:
            results = adapter.search(query, SEARCH_LIMIT)
        except NotImplementedError as exc:
            skipped.append(f"{name}: {str(exc) or 'search not supported'}")
            continue
        except PlatformError as exc:
            skipped.append(f"{name}: {str(exc) or type(exc).__name__}")
            continue
        except Exception as exc:  # an adapter bug must not stop drafting or the runner
            logger.warning("comps search on %s raised %s", name, type(exc).__name__)
            skipped.append(f"{name}: search failed ({type(exc).__name__})")
            continue
        answered = True
        for result in results:
            if result.price_cents <= 0 or (result.url and result.url in own):
                continue
            if result.url:
                if result.url in seen:
                    continue
                seen.add(result.url)
            found.append((name, result))
    return summarize(found, skipped), answered


def _row(session: Session, key: str) -> CompsCache | None:
    if not key:
        return None
    query = select(CompsCache).where(CompsCache.query == key)
    return session.exec(query.execution_options(populate_existing=True)).first()


def _spread[T](ordered: Sequence[T], k: int) -> list[T]:
    """Up to k entries evenly spaced from cheapest to dearest."""
    if len(ordered) <= k:
        return list(ordered)
    step = (len(ordered) - 1) / (k - 1)
    return [ordered[round(i * step)] for i in range(k)]


def _clip(title: str) -> str:
    flat = " ".join(title.split())
    return flat if len(flat) <= SAMPLE_TITLE_CHARS else flat[: SAMPLE_TITLE_CHARS - 3] + "..."


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _utc(dt: datetime) -> datetime:
    """SQLite drops tzinfo on the way back; every stored datetime is UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
