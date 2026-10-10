"""Listing traffic over a time window, from stored stats snapshots.

A relist posts a fresh copy, and the platform counts it from zero, while the snapshots continue on
the same Listing row. A window that spans a relist is summed in segments, each from its own start:

| Snapshot after the previous one | Counts from |
|---|---|
| Same post | The previous snapshot |
| The post was relisted in between, or views went down | Zero |
"""
from collections.abc import Sequence
from datetime import datetime

from marketplace_agent.core.models import ListingStatsSnapshot

Counts = tuple[int, int, int]  # views, saves, messages


def gained_since(
    snapshots: Sequence[ListingStatsSnapshot],
    since: datetime,
    *,
    published_at: datetime | None,
    relisted_at: datetime | None,
) -> Counts | None:
    """Views, saves, and messages gained since `since`, oldest snapshot first. None when nothing
    is known that old: no snapshot at or before it, and the listing went up before it."""
    if not snapshots:
        return None
    before = [s for s in snapshots if s.captured_at <= since]
    if before:
        previous, previous_at = _counts(before[-1]), before[-1].captured_at
    elif published_at is not None and published_at >= since:
        previous, previous_at = (0, 0, 0), since
    else:
        return None
    gained = [0, 0, 0]
    for snapshot in snapshots[len(before) :]:
        current = _counts(snapshot)
        relisted = relisted_at is not None and previous_at < relisted_at <= snapshot.captured_at
        start = (0, 0, 0) if relisted or current[0] < previous[0] else previous
        for i in range(3):
            gained[i] += current[i] - start[i]
        previous, previous_at = current, snapshot.captured_at
    return gained[0], gained[1], gained[2]


def _counts(snapshot: ListingStatsSnapshot) -> Counts:
    return snapshot.views, snapshot.saves, snapshot.message_count
