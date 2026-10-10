"""Trust ladder per (platform, action): ASK -> ACT_AND_NOTIFY -> SILENT.

A record is created on first use from DEFAULT_TIERS, trying the exact platform, then "*", then ASK.
trust_promote_after clean executions in a row promote one tier; one human correction demotes one.
"""
from sqlmodel import Session, col, select

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import TrustRecord, TrustTier

_NOTIFIED_ACTIONS = (
    "update_price",
    "rewrite_listing",
    "relist",
    "publish",
    "accept_offer",
    "book_pickup",
    "release_commitment",
    "complete_sale",
)
_FACEBOOK_GATED = ("publish", "relist", "rewrite_listing", "update_price")

DEFAULT_TIERS: dict[tuple[str, str], TrustTier] = {
    ("*", "send_message"): TrustTier.SILENT,
    **{("*", action): TrustTier.ACT_AND_NOTIFY for action in _NOTIFIED_ACTIONS},
    **{("facebook", action): TrustTier.ASK for action in _FACEBOOK_GATED},
}


def tier(session: Session, clock: Clock, platform: str, action: str) -> TrustTier:
    return TrustTier(_record(session, clock, platform, action).tier)


def current_tier(session: Session, platform: str, action: str) -> TrustTier:
    """The stored tier, else the default a first use stores. Reads only: the guard asks before a
    platform call, and a write then would hold SQLite's write lock for as long as it takes."""
    record = session.exec(_query(platform, action)).first()
    return TrustTier(record.tier) if record is not None else _default_tier(platform, action)


def record_executed(
    session: Session, clock: Clock, settings: Settings, platform: str, action: str
) -> TrustTier:
    record = _record(session, clock, platform, action)
    record.executed += 1
    record.consecutive_clean += 1
    if record.consecutive_clean >= settings.trust_promote_after and record.tier < TrustTier.SILENT:
        record.tier += 1
        record.consecutive_clean = 0
    return _save(session, clock, record)


def record_correction(session: Session, clock: Clock, platform: str, action: str) -> TrustTier:
    record = _record(session, clock, platform, action)
    record.corrected += 1
    record.consecutive_clean = 0
    record.tier = max(TrustTier.ASK, record.tier - 1)
    return _save(session, clock, record)


def summary(session: Session) -> list[TrustRecord]:
    query = select(TrustRecord).order_by(col(TrustRecord.platform), col(TrustRecord.action))
    return list(session.exec(query).all())


def _default_tier(platform: str, action: str) -> TrustTier:
    return DEFAULT_TIERS.get((platform, action), DEFAULT_TIERS.get(("*", action), TrustTier.ASK))


def _query(platform: str, action: str):
    return select(TrustRecord).where(TrustRecord.platform == platform, TrustRecord.action == action)


def _record(session: Session, clock: Clock, platform: str, action: str) -> TrustRecord:
    record = session.exec(_query(platform, action)).first()
    if record is None:
        record = TrustRecord(
            platform=platform,
            action=action,
            tier=_default_tier(platform, action),
            updated_at=clock.now(),
        )
        session.add(record)
        session.flush()
    return record


def _save(session: Session, clock: Clock, record: TrustRecord) -> TrustTier:
    record.updated_at = clock.now()
    session.add(record)
    session.flush()
    return TrustTier(record.tier)
