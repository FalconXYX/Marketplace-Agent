"""Database models.

Conventions: money is integer cents, datetimes are timezone-aware UTC from a
Clock (never datetime.now()). No ORM relationships; query explicitly.
Commitment is an append-only ledger: rows are never updated or deleted.
"""
from datetime import datetime
from enum import IntEnum, StrEnum

from sqlmodel import Field, SQLModel, UniqueConstraint


class ItemStatus(StrEnum):
    DRAFT = "draft"
    LISTED = "listed"
    COMMITTED = "committed"
    SOLD = "sold"
    WITHDRAWN = "withdrawn"


class Firmness(StrEnum):
    FIRM = "firm"          # decline anything under list price, no counters
    SLIGHT = "slight"      # small movement allowed
    FLEXIBLE = "flexible"  # negotiate freely down to floor


class ListingStatus(StrEnum):
    PENDING = "pending"
    AWAITING_HUMAN = "awaiting_human"  # Facebook assist mode: form filled, human clicks Post
    LIVE = "live"
    DELISTED = "delisted"
    FAILED = "failed"


class ThreadState(StrEnum):
    OPEN = "open"
    NEGOTIATING = "negotiating"
    COMMITTED = "committed"
    CLOSED = "closed"
    FROZEN = "frozen"  # scam or safety stop; agent never auto-replies


class Direction(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CommitmentKind(StrEnum):
    ACCEPTED = "accepted"            # item promised to buyer at price_cents
    PICKUP_BOOKED = "pickup_booked"  # ref_id -> ACCEPTED row
    RELEASED = "released"            # ref_id -> ACCEPTED row (no-show, buyer backed out)
    UNDONE = "undone"                # ref_id -> ACCEPTED row (human undo within window)
    COMPLETED = "completed"          # ref_id -> ACCEPTED row (sold, cash in hand)


class MemoryScope(StrEnum):
    ITEM = "item"
    BUYER = "buyer"
    GLOBAL = "global"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TrustTier(IntEnum):
    ASK = 0             # needs human approval first
    ACT_AND_NOTIFY = 1  # act, then tell the human
    SILENT = 2          # act, log only


class Item(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    hint: str = ""                 # the user's raw input, e.g. "IKEA desk $120"
    title: str = ""
    description: str = ""
    category: str | None = None
    condition: str | None = None   # new | like_new | good | fair | poor
    attributes_json: str = "{}"    # brand, model, dimensions... only what is verifiable
    status: ItemStatus = Field(default=ItemStatus.DRAFT, index=True)
    paused: bool = False           # human pause: agent takes no actions
    created_at: datetime
    updated_at: datetime
    last_agent_run_at: datetime | None = None
    last_wake_views: int = 0       # views at last stats-triggered wake


class Photo(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    path: str                      # relative to Settings.data_dir
    position: int = 0
    width: int
    height: int
    sha256: str


class Goal(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", unique=True)
    list_price_cents: int          # current asking price; agent moves it via update_price
    target_price_cents: int        # what the human hopes for
    floor_price_cents: int         # hard minimum, enforced by guard
    deadline: datetime
    firmness: Firmness = Firmness.SLIGHT
    notes: str = ""


class Listing(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    platform: str = Field(index=True)
    remote_id: str | None = None
    url: str | None = None
    status: ListingStatus = ListingStatus.PENDING
    price_cents: int
    title: str
    description: str
    published_at: datetime | None = None
    last_relisted_at: datetime | None = None
    delisted_at: datetime | None = None
    last_error: str | None = None


class ListingStatsSnapshot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    listing_id: int = Field(foreign_key="listing.id", index=True)
    captured_at: datetime
    views: int = 0
    saves: int = 0
    message_count: int = 0


class Buyer(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("platform", "handle"),)
    id: int | None = Field(default=None, primary_key=True)
    platform: str
    handle: str                    # stable remote user id
    display_name: str = ""
    completed_count: int = 0
    no_show_count: int = 0
    ghost_count: int = 0
    flagged_scam: bool = False
    notes: str = ""


class Thread(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("platform", "remote_id"),)
    id: int | None = Field(default=None, primary_key=True)
    platform: str = Field(index=True)
    remote_id: str
    item_id: int | None = Field(default=None, foreign_key="item.id", index=True)
    buyer_id: int = Field(foreign_key="buyer.id", index=True)
    state: ThreadState = ThreadState.OPEN
    last_offer_cents: int | None = None
    created_at: datetime
    last_inbound_at: datetime | None = None
    last_outbound_at: datetime | None = None
    frozen_reason: str | None = None


class Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    thread_id: int = Field(foreign_key="thread.id", index=True)
    direction: Direction
    body: str
    sent_at: datetime
    remote_id: str | None = Field(default=None, index=True)
    handled: bool = False          # inbound: processed by cache or agent
    via: str = ""                  # buyer | pattern_cache | agent | human | system
    intent: str | None = None      # availability | question | offer | scheduling | lowball | scam | spam | other


class Commitment(SQLModel, table=True):
    """Append-only ledger of promises made to humans. Never update or delete rows."""
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    thread_id: int | None = Field(default=None, foreign_key="thread.id")
    buyer_id: int | None = Field(default=None, foreign_key="buyer.id")
    kind: CommitmentKind
    ref_id: int | None = None      # the ACCEPTED row this event refers to
    price_cents: int | None = None
    pickup_start: datetime | None = None
    pickup_end: datetime | None = None
    undo_deadline: datetime | None = None
    reason: str = ""
    actor: str = "agent"           # agent | human | system
    bundle_key: str | None = Field(default=None, index=True)  # rows of one multi-item deal share it
    created_at: datetime


class CompsCache(SQLModel, table=True):
    """Comparable listings per normalized query, refreshed at most daily. No model calls."""
    id: int | None = Field(default=None, primary_key=True)
    query: str = Field(unique=True)
    fetched_at: datetime
    summary_json: str = "{}"


class Memory(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    scope: MemoryScope = Field(index=True)
    item_id: int | None = Field(default=None, foreign_key="item.id", index=True)
    buyer_id: int | None = Field(default=None, foreign_key="buyer.id", index=True)
    lesson: str
    source: str = "agent"          # agent | undo | correction | outcome
    uses: int = 0
    created_at: datetime


class TrustRecord(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("platform", "action"),)
    id: int | None = Field(default=None, primary_key=True)
    platform: str                  # "*" for platform-independent actions
    action: str                    # publish | send_message | update_price | accept_offer | ...
    tier: int = TrustTier.ASK
    executed: int = 0
    corrected: int = 0
    consecutive_clean: int = 0
    updated_at: datetime


class PendingApproval(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    platform: str = "*"
    tool: str
    args_json: str
    rationale: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    agent_run_id: int | None = None
    created_at: datetime
    decided_at: datetime | None = None


class Escalation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    item_id: int | None = Field(default=None, foreign_key="item.id", index=True)
    thread_id: int | None = Field(default=None, foreign_key="thread.id")
    reason: str
    created_at: datetime
    resolved_at: datetime | None = None
    resolution: str | None = None


class WakeRequest(SQLModel, table=True):
    """Out-of-band wake, e.g. a human override from Discord or the web UI."""
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    reason: str
    created_at: datetime
    consumed_at: datetime | None = None


class AgentRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    started_at: datetime
    finished_at: datetime | None = None
    triggers_json: str = "[]"
    rationale: str = ""
    plan_json: str = ""
    results_json: str = ""
    provider: str | None = None
    requests_used: int = 0
    quota_state: str = "healthy"
    error: str | None = None


class QuotaUsage(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("provider", "model", "day"),)
    id: int | None = Field(default=None, primary_key=True)
    provider: str
    model: str
    day: str                       # YYYY-MM-DD the provider's quota day began (Gemini: Pacific)
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class PriceChange(SQLModel, table=True):
    """Audit trail of asking-price moves. Drives the 24h change limit and correction detection."""
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    old_cents: int
    new_cents: int
    actor: str = "agent"           # agent | human
    created_at: datetime


class PatternEntry(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    intent: str = Field(index=True)
    normalized: str = Field(unique=True)
    hits: int = 0
    source: str = "seed"           # seed | learned
    created_at: datetime
