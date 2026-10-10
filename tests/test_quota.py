from datetime import UTC, datetime

import pytest

from marketplace_agent.ai.base import QuotaState
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.config import ProviderLimit


@pytest.fixture
def tracker(engine, clock) -> QuotaTracker:
    return QuotaTracker(engine, clock, {"a": ProviderLimit(rpm=2, rpd=8), "b": ProviderLimit(rpm=10, rpd=8)})


@pytest.fixture
def real(engine, clock) -> QuotaTracker:
    """Real provider names, so the default reset zones apply."""
    limits = {name: ProviderLimit(rpm=1000, rpd=3) for name in ("gemini", "openrouter")}
    return QuotaTracker(engine, clock, limits)


def utc(month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def test_rpm_window_slides_with_clock(tracker, clock):
    tracker.record("a", "m")
    clock.advance(seconds=30)
    tracker.record("a", "m")
    assert not tracker.can_request("a")
    clock.advance(seconds=29)
    assert not tracker.can_request("a")
    clock.advance(seconds=1)  # first request is now exactly 60s old
    assert tracker.can_request("a")


def test_rpd_persists_across_restart(engine, clock):
    limits = {"a": ProviderLimit(rpm=100, rpd=3)}
    first = QuotaTracker(engine, clock, limits)
    for _ in range(3):
        first.record("a", "m")
    assert not first.can_request("a")

    restarted = QuotaTracker(engine, clock, limits)
    assert restarted.requests_today("a") == 3
    assert not restarted.can_request("a")
    assert restarted.remaining_today("a") == 0


def test_daily_count_resets_on_new_utc_day(tracker, clock):
    for _ in range(8):
        tracker.record("b", "m")
    assert not tracker.can_request("b")
    clock.advance(days=1)
    assert tracker.requests_today("b") == 0
    assert tracker.can_request("b")


def test_requests_today_sums_models_and_usage_lists_rows(tracker, clock):
    tracker.record("a", "flash", tokens_in=100, tokens_out=10)
    tracker.record("a", "flash", tokens_in=50, tokens_out=5)
    tracker.record("a", "lite")
    assert tracker.requests_today("a") == 3

    rows = {(u.provider, u.model): u for u in tracker.usage_today()}
    assert set(rows) == {("a", "flash"), ("a", "lite")}
    flash = rows["a", "flash"]
    assert (flash.requests, flash.tokens_in, flash.tokens_out) == (2, 150, 15)

    clock.advance(days=1)
    assert tracker.usage_today() == []


def test_cooldown_expires(tracker, clock):
    tracker.mark_rate_limited("a", 30)
    assert not tracker.can_request("a")
    clock.advance(seconds=29)
    assert not tracker.can_request("a")
    clock.advance(seconds=1)
    assert tracker.can_request("a")


def test_default_cooldown_applies_to_unlimited_provider(tracker, clock):
    tracker.mark_rate_limited("unknown")
    clock.advance(seconds=59)
    assert not tracker.can_request("unknown")
    clock.advance(seconds=1)
    assert tracker.can_request("unknown")


def test_unlimited_provider(tracker):
    for _ in range(50):
        tracker.record("free", "m")
    assert tracker.can_request("free")
    assert tracker.remaining_today("free") is None
    assert tracker.state(["free"]) is QuotaState.HEALTHY


def test_state_healthy_constrained_exhausted(engine, clock):
    tracker = QuotaTracker(engine, clock, {"a": ProviderLimit(rpm=100, rpd=8)}, constrained_fraction=0.25)
    for _ in range(6):
        tracker.record("a", "m")
    assert tracker.state(["a"]) is QuotaState.HEALTHY  # 2 left, threshold is 2
    tracker.record("a", "m")
    assert tracker.state(["a"]) is QuotaState.CONSTRAINED
    tracker.record("a", "m")
    assert tracker.state(["a"]) is QuotaState.EXHAUSTED


def test_state_sums_budget_across_providers(tracker):
    for _ in range(8):
        tracker.record("a", "m")
    assert tracker.state(["a", "b"]) is QuotaState.HEALTHY  # 8 of 16 left
    for _ in range(5):
        tracker.record("b", "m")
    assert tracker.state(["a", "b"]) is QuotaState.CONSTRAINED  # 3 of 16 left


def test_state_exhausted_when_all_cooling_or_empty(tracker, clock):
    assert tracker.state([]) is QuotaState.EXHAUSTED
    tracker.mark_rate_limited("a")
    tracker.mark_rate_limited("b")
    assert tracker.state(["a", "b"]) is QuotaState.EXHAUSTED
    clock.advance(minutes=1)
    assert tracker.state(["a", "b"]) is QuotaState.HEALTHY


def test_unlimited_provider_keeps_state_healthy(tracker):
    for _ in range(7):
        tracker.record("b", "m")
    assert tracker.state(["b"]) is QuotaState.CONSTRAINED
    assert tracker.state(["b", "free"]) is QuotaState.HEALTHY


# Gemini's requests-per-day quota resets at midnight Pacific, not UTC.


def test_gemini_day_does_not_end_at_utc_midnight(real, clock):
    clock.set(utc(10, 9, 23))  # 16:00 PDT
    for _ in range(3):
        real.record("gemini", "flash")

    clock.set(utc(10, 10, 0, 30))  # 17:30 PDT, still Google's Oct 9
    assert not real.can_request("gemini")
    assert real.remaining_today("gemini") == 0
    assert real.state(["gemini"]) is QuotaState.EXHAUSTED
    clock.set(utc(10, 10, 6, 59))
    assert not real.can_request("gemini")

    clock.set(utc(10, 10, 7))  # midnight PDT
    assert real.can_request("gemini")
    assert real.remaining_today("gemini") == 3


def test_gemini_requests_after_utc_midnight_belong_to_the_pacific_day(real, clock):
    clock.set(utc(10, 10, 1))  # Oct 9, 18:00 PDT
    for _ in range(3):
        real.record("gemini", "flash")

    clock.set(utc(10, 10, 8))  # Oct 10, 01:00 PDT: Google has reset
    assert real.requests_today("gemini") == 0
    assert real.can_request("gemini")


def test_gemini_day_follows_daylight_saving(real, clock):
    clock.set(utc(12, 1, 7, 30))  # Nov 30, 23:30 PST
    for _ in range(3):
        real.record("gemini", "flash")

    clock.set(utc(12, 1, 7, 59))
    assert not real.can_request("gemini")
    clock.set(utc(12, 1, 8))  # midnight PST
    assert real.can_request("gemini")


def test_other_providers_reset_at_utc_midnight(real, clock):
    clock.set(utc(10, 9, 23))
    for _ in range(3):
        real.record("openrouter", "m")
    assert not real.can_request("openrouter")

    clock.set(utc(10, 10, 0))
    assert real.can_request("openrouter")


def test_usage_today_uses_each_providers_own_day(real, clock):
    clock.set(utc(10, 9, 23))
    real.record("gemini", "flash")
    real.record("openrouter", "m")

    clock.set(utc(10, 10, 1))
    assert [(u.provider, u.day) for u in real.usage_today()] == [("gemini", "2026-10-09")]


# Retry-After values a provider should never send


@pytest.mark.parametrize("retry_after", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_retry_after_cools_down_for_a_minute(tracker, clock, retry_after):
    tracker.mark_rate_limited("a", retry_after)

    clock.advance(seconds=59)
    assert not tracker.can_request("a")
    clock.advance(seconds=1)
    assert tracker.can_request("a")


def test_huge_retry_after_is_capped_at_a_day(tracker, clock):
    tracker.mark_rate_limited("a", 1e12)

    clock.advance(hours=23, minutes=59)
    assert not tracker.can_request("a")
    clock.advance(minutes=1)
    assert tracker.can_request("a")


@pytest.mark.parametrize("retry_after", [0, -5])
def test_retry_after_zero_means_retry_now(tracker, retry_after):
    tracker.mark_rate_limited("a", retry_after)

    assert tracker.can_request("a")


# Suspended models: daily quota spent, or the model does not exist


def test_suspended_model_waits_for_its_providers_day_to_end(real, clock):
    clock.set(utc(10, 9, 20))  # 13:00 PDT
    real.suspend("gemini", "flash")

    assert real.is_suspended("gemini", "flash")
    assert not real.can_request("gemini", "flash")
    assert real.can_request("gemini", "flash-lite")
    assert real.can_request("gemini")
    clock.set(utc(10, 10, 6, 59))
    assert not real.can_request("gemini", "flash")
    clock.set(utc(10, 10, 7))  # midnight PDT
    assert real.can_request("gemini", "flash")


def test_suspension_on_a_utc_provider_ends_at_utc_midnight(real, clock):
    clock.set(utc(10, 9, 20))
    real.suspend("openrouter", "dead-model")

    clock.set(utc(10, 9, 23, 59))
    assert not real.can_request("openrouter", "dead-model")
    clock.set(utc(10, 10, 0))
    assert real.can_request("openrouter", "dead-model")
