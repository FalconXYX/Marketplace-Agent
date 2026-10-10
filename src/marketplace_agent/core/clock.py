"""Injectable time. Every timestamp in the system comes from a Clock.

Convention: timezone-aware UTC datetimes everywhere (SQLModel stores and
returns them as UTC). Convert to Settings.timezone only at the availability
boundary and when rendering for humans.
"""
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    """Manually advanced clock for tests and simulation."""

    def __init__(self, start: datetime | None = None):
        start = start or datetime(2026, 10, 1, 16, 0, tzinfo=UTC)
        self._now = start if start.tzinfo else start.replace(tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta | None = None, **kwargs: float) -> datetime:
        self._now += delta if delta is not None else timedelta(**kwargs)
        return self._now

    def set(self, when: datetime) -> None:
        self._now = when if when.tzinfo else when.replace(tzinfo=UTC)
