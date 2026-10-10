import pytest
from sqlmodel import select

from marketplace_agent.core.models import TrustRecord, TrustTier
from marketplace_agent.guard import trust
from marketplace_agent.guard.trust import (
    DEFAULT_TIERS,
    current_tier,
    record_correction,
    record_executed,
    summary,
    tier,
)

GATED_ON_FACEBOOK = ("publish", "relist", "rewrite_listing", "update_price")
NOTIFIED = (
    "update_price",
    "rewrite_listing",
    "relist",
    "publish",
    "accept_offer",
    "book_pickup",
    "release_commitment",
    "complete_sale",
)


def _record(session, platform, action) -> TrustRecord:
    query = select(TrustRecord).where(
        TrustRecord.platform == platform, TrustRecord.action == action
    )
    return session.exec(query).one()


def test_default_tiers_table():
    assert DEFAULT_TIERS[("*", "send_message")] == TrustTier.SILENT
    for action in NOTIFIED:
        assert DEFAULT_TIERS[("*", action)] == TrustTier.ACT_AND_NOTIFY
    for action in GATED_ON_FACEBOOK:
        assert DEFAULT_TIERS[("facebook", action)] == TrustTier.ASK


@pytest.mark.parametrize(
    ("platform", "action", "expected"),
    [
        ("*", "send_message", TrustTier.SILENT),
        ("facebook", "send_message", TrustTier.SILENT),  # falls back to "*"
        ("kijiji", "publish", TrustTier.ACT_AND_NOTIFY),  # falls back to "*"
        ("facebook", "publish", TrustTier.ASK),  # platform-specific wins
        ("facebook", "update_price", TrustTier.ASK),
        ("facebook", "accept_offer", TrustTier.ACT_AND_NOTIFY),
        ("*", "never_heard_of_it", TrustTier.ASK),
        ("kijiji", "never_heard_of_it", TrustTier.ASK),
    ],
)
def test_tier_defaults_with_star_fallback(session, clock, platform, action, expected):
    assert tier(session, clock, platform, action) == expected
    record = _record(session, platform, action)
    assert (record.platform, record.action, record.tier) == (platform, action, expected)


def test_tier_creates_the_record_once_and_then_reads_it(session, clock):
    tier(session, clock, "kijiji", "publish")
    _record(session, "kijiji", "publish").tier = TrustTier.ASK
    session.flush()

    assert tier(session, clock, "kijiji", "publish") == TrustTier.ASK
    assert len(session.exec(select(TrustRecord)).all()) == 1


def test_current_tier_reads_without_writing(session, clock):
    """The guard asks it before a platform call, so it must not take the write lock."""
    assert current_tier(session, "facebook", "publish") == TrustTier.ASK
    assert current_tier(session, "kijiji", "publish") == TrustTier.ACT_AND_NOTIFY
    assert session.exec(select(TrustRecord)).all() == []
    record_correction(session, clock, "kijiji", "publish")
    assert current_tier(session, "kijiji", "publish") == TrustTier.ASK


def test_promotion_happens_exactly_at_the_threshold(session, clock, settings):
    settings = settings.model_copy(update={"trust_promote_after": 3})
    platform, action = "facebook", "publish"

    assert record_executed(session, clock, settings, platform, action) == TrustTier.ASK
    assert record_executed(session, clock, settings, platform, action) == TrustTier.ASK
    assert record_executed(session, clock, settings, platform, action) == TrustTier.ACT_AND_NOTIFY
    record = _record(session, platform, action)
    assert (record.executed, record.consecutive_clean) == (3, 0)

    for _ in range(2):
        assert (
            record_executed(session, clock, settings, platform, action) == TrustTier.ACT_AND_NOTIFY
        )
    assert record_executed(session, clock, settings, platform, action) == TrustTier.SILENT


def test_promotion_stops_at_silent(session, clock, settings):
    settings = settings.model_copy(update={"trust_promote_after": 1})
    for _ in range(5):
        assert record_executed(session, clock, settings, "*", "send_message") == TrustTier.SILENT
    assert _record(session, "*", "send_message").executed == 5


def test_correction_demotes_one_tier_and_resets_the_streak(session, clock, settings):
    settings = settings.model_copy(update={"trust_promote_after": 3})
    for _ in range(2):
        record_executed(session, clock, settings, "*", "update_price")
    assert _record(session, "*", "update_price").consecutive_clean == 2

    assert record_correction(session, clock, "*", "update_price") == TrustTier.ASK
    record = _record(session, "*", "update_price")
    assert (record.corrected, record.consecutive_clean, record.executed) == (1, 0, 2)

    # The two clean runs before the correction no longer count toward promotion.
    record_executed(session, clock, settings, "*", "update_price")
    assert record_executed(session, clock, settings, "*", "update_price") == TrustTier.ASK
    assert (
        record_executed(session, clock, settings, "*", "update_price") == TrustTier.ACT_AND_NOTIFY
    )


def test_demotion_floors_at_ask(session, clock):
    assert record_correction(session, clock, "*", "send_message") == TrustTier.ACT_AND_NOTIFY
    assert record_correction(session, clock, "*", "send_message") == TrustTier.ASK
    assert record_correction(session, clock, "*", "send_message") == TrustTier.ASK
    assert _record(session, "*", "send_message").corrected == 3


def test_records_track_updated_at(session, clock, settings):
    tier(session, clock, "*", "publish")
    clock.advance(hours=1)
    record_executed(session, clock, settings, "*", "publish")
    assert _record(session, "*", "publish").updated_at == clock.now()


def test_platform_records_are_independent(session, clock):
    record_correction(session, clock, "kijiji", "publish")
    assert tier(session, clock, "kijiji", "publish") == TrustTier.ASK
    assert tier(session, clock, "*", "publish") == TrustTier.ACT_AND_NOTIFY


def test_summary_lists_every_record_in_order(session, clock):
    for platform, action in [
        ("kijiji", "relist"),
        ("*", "send_message"),
        ("facebook", "publish"),
        ("*", "accept_offer"),
    ]:
        tier(session, clock, platform, action)

    assert [(r.platform, r.action) for r in summary(session)] == [
        ("*", "accept_offer"),
        ("*", "send_message"),
        ("facebook", "publish"),
        ("kijiji", "relist"),
    ]
    assert trust.summary is summary
