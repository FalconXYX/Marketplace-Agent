"""Request-count quota per provider. Free tiers cap requests per minute and per day, not tokens.

| State | Where it lives |
|---|---|
| Daily counts | `QuotaUsage` rows, so they survive restarts |
| Minute window, cooldowns, suspended models | Memory |

A provider's day ends at midnight in its reset zone, so local counts reset when the provider's do:

| Provider | Day ends | Source |
|---|---|---|
| gemini | Midnight Pacific | Gemini rate-limits docs |
| Others | Midnight UTC | OpenRouter limits docs |

`QuotaUsage.day` is the UTC date on which that day began, which for Pacific is the Pacific date.
"""
import logging
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, func
from sqlalchemy.exc import OperationalError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel import Session, col, select

from marketplace_agent.ai.base import QuotaState
from marketplace_agent.config import ProviderLimit
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import QuotaUsage

_MINUTE = timedelta(minutes=1)
SHORT_WAIT = timedelta(minutes=5)  # a hold this short is throttling, not exhaustion
_DEFAULT_COOLDOWN_S = 60.0
_MAX_COOLDOWN_S = 86_400.0

RESET_ZONES: dict[str, tzinfo] = {"gemini": ZoneInfo("America/Los_Angeles")}

logger = logging.getLogger("marketplace_agent.ai")


class QuotaTracker:
    def __init__(
        self,
        engine: Engine,
        clock: Clock,
        limits: Mapping[str, ProviderLimit],
        constrained_fraction: float = 0.25,
        reset_zones: Mapping[str, tzinfo] = RESET_ZONES,
    ):
        self.engine = engine
        self.clock = clock
        self.limits = dict(limits)
        self.constrained_fraction = constrained_fraction
        self.reset_zones = dict(reset_zones)
        self._recent: defaultdict[str, deque[datetime]] = defaultdict(deque)
        self._cooldown_until: dict[str, datetime] = {}
        self._suspended_until: dict[tuple[str, str], datetime] = {}

    def can_request(self, provider: str, model: str | None = None) -> bool:
        now = self.clock.now()
        until = self._cooldown_until.get(provider)
        if until is not None and now < until:
            return False
        if model is not None and self.is_suspended(provider, model):
            return False
        limit = self.limits.get(provider)
        if limit is None:
            return True
        return (
            self._requests_last_minute(provider, now) < limit.rpm
            and self.requests_today(provider) < limit.rpd
        )

    def record(self, provider: str, model: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Count one request. Commits on its own connection, so it survives a caller's rollback.
        A database too busy to take the count loses that count, not the paid-for response."""
        now = self.clock.now()
        stmt = sqlite_insert(QuotaUsage).values(
            provider=provider,
            model=model,
            day=self._day(provider, now),
            requests=1,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "model", "day"],
            set_={
                "requests": col(QuotaUsage.requests) + stmt.excluded.requests,
                "tokens_in": col(QuotaUsage.tokens_in) + stmt.excluded.tokens_in,
                "tokens_out": col(QuotaUsage.tokens_out) + stmt.excluded.tokens_out,
            },
        )
        self._recent[provider].append(now)
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt)
        except OperationalError as exc:
            logger.warning("request to %s not counted, database busy: %s", provider, exc.orig)

    def mark_rate_limited(self, provider: str, retry_after_s: float | None = None) -> None:
        """Cool down for retry_after_s, capped at a day. Missing, NaN or infinite: one minute."""
        if retry_after_s is None or not math.isfinite(retry_after_s):
            seconds = _DEFAULT_COOLDOWN_S
        else:
            seconds = min(max(retry_after_s, 0.0), _MAX_COOLDOWN_S)
        self._cooldown_until[provider] = self.clock.now() + timedelta(seconds=seconds)

    def suspend(self, provider: str, model: str) -> None:
        """Skip one model until the provider's day ends: its daily quota is spent or it does not
        exist. Other models of the provider stay usable."""
        zone = self.reset_zones.get(provider, UTC)
        self._suspended_until[provider, model] = _midnight(self.clock.now(), zone, days_ahead=1)

    def is_suspended(self, provider: str, model: str) -> bool:
        until = self._suspended_until.get((provider, model))
        return until is not None and self.clock.now() < until

    def requests_today(self, provider: str) -> int:
        query = select(func.coalesce(func.sum(QuotaUsage.requests), 0)).where(
            QuotaUsage.provider == provider,
            QuotaUsage.day == self._day(provider, self.clock.now()),
        )
        with Session(self.engine) as session:
            return int(session.exec(query).one())

    def remaining_today(self, provider: str) -> int | None:
        """None means the provider has no configured limit."""
        limit = self.limits.get(provider)
        if limit is None:
            return None
        return max(0, limit.rpd - self.requests_today(provider))

    def state(self, providers: Iterable[str]) -> QuotaState:
        names = list(providers)
        if not any(self.can_request(name) for name in names):
            return QuotaState.EXHAUSTED
        if any(name not in self.limits for name in names):
            return QuotaState.HEALTHY
        budget = sum(self.limits[name].rpd for name in names)
        left = sum(self.remaining_today(name) or 0 for name in names)
        if left < self.constrained_fraction * budget:
            return QuotaState.CONSTRAINED
        return QuotaState.HEALTHY

    def throttled(self, providers: Iterable[str]) -> bool:
        """No provider can take a request now, but one is held back only briefly: by its
        per-minute window or a cooldown ending within SHORT_WAIT, with budget left today. state()
        calls that EXHAUSTED too; the difference is that it clears by the next tick."""
        names = list(providers)
        if not names or any(self.can_request(name) for name in names):
            return False
        return any(self._back_soon(name) for name in names)

    def _back_soon(self, provider: str) -> bool:
        limit = self.limits.get(provider)
        if limit is not None and self.requests_today(provider) >= limit.rpd:
            return False
        until = self._cooldown_until.get(provider)
        return until is None or until - self.clock.now() <= SHORT_WAIT

    def usage_today(self) -> list[QuotaUsage]:
        """Each provider's rows for its own current day."""
        now = self.clock.now()
        days = {_day_label(now, zone) for zone in (UTC, *self.reset_zones.values())}
        query = (
            select(QuotaUsage)
            .where(col(QuotaUsage.day).in_(days))
            .order_by(col(QuotaUsage.provider), col(QuotaUsage.model))
        )
        with Session(self.engine) as session:
            rows = session.exec(query).all()
        return [row for row in rows if row.day == self._day(row.provider, now)]

    def _requests_last_minute(self, provider: str, now: datetime) -> int:
        window = self._recent[provider]
        while window and window[0] <= now - _MINUTE:
            window.popleft()
        return len(window)

    def _day(self, provider: str, now: datetime) -> str:
        return _day_label(now, self.reset_zones.get(provider, UTC))


def _midnight(now: datetime, zone: tzinfo, days_ahead: int = 0) -> datetime:
    """Midnight in zone starting the day that contains now, or a later day, as UTC."""
    day = now.astimezone(zone).date() + timedelta(days=days_ahead)
    return datetime.combine(day, time(), tzinfo=zone).astimezone(UTC)


def _day_label(now: datetime, zone: tzinfo) -> str:
    return _midnight(now, zone).date().isoformat()
