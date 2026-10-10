"""Platform state into the database: threads, buyers, messages, and listing stats.

Ingest is idempotent: messages are deduped by remote id within their thread, so fetching the
same state twice stores nothing new.

| Remote message | Stored as |
|---|---|
| From the buyer, thread not frozen | INBOUND, unhandled, via "buyer" |
| From the buyer, freeze-level scam | INBOUND, handled, intent "scam"; freeze, flag, escalate |
| From the buyer, thread frozen | INBOUND, handled |
| First message from a flagged buyer | The new thread is frozen and escalated at once |
| From the seller, matching one we sent | Nothing (echo of our own message) |
| From the seller, unknown | OUTBOUND, handled, via "human" (typed on the platform) |

Every stored body has the pickup address redacted to the placeholder.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlmodel import Session, col, select

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import (
    Buyer,
    Direction,
    Escalation,
    Item,
    ItemStatus,
    Listing,
    ListingStatsSnapshot,
    ListingStatus,
    Message,
    Thread,
    ThreadState,
    WakeRequest,
)
from marketplace_agent.guard.invariants import detect_scam, offer_cents, redact_address
from marketplace_agent.notify.base import Notice, NoticeKind, Notifier
from marketplace_agent.platforms.base import (
    ListingStats,
    PlatformAdapter,
    PlatformAuthError,
    PlatformError,
    PlatformRef,
    RemoteMessage,
    RemoteThread,
)
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.tools.impl.common import gone_notice, mark_gone

ECHO_WINDOW = timedelta(minutes=10)
STATS_REFRESH = timedelta(hours=6)
FLAGGED_BUYER_REASON = "this buyer was already flagged as a scammer in another conversation"

logger = logging.getLogger("marketplace_agent.runner")


@dataclass
class IngestResult:
    platform: str
    new_inbound: list[int] = field(default_factory=list)
    new_threads: int = 0
    frozen_threads: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def ingest(
    session: Session,
    clock: Clock,
    settings: Settings,
    notifier: Notifier,
    adapter: PlatformAdapter,
    since: datetime | None,
) -> IngestResult:
    """Store what adapter.fetch_threads(since) returns. Flushes; the caller commits."""
    ingestor = _Ingestor(session, clock, settings, notifier, IngestResult(adapter.name))
    for remote in adapter.fetch_threads(since):
        ingestor.thread(remote)
    session.flush()
    return ingestor.result


def snapshot_stats(
    session: Session, clock: Clock, adapter: PlatformAdapter, notifier: Notifier | None = None
) -> int:
    """One snapshot per LIVE listing on the adapter's platform whose numbers changed or whose
    latest snapshot is STATS_REFRESH old. An auth error stops the pass; any other failure skips
    that listing. A post gone from the platform is marked DELISTED, alerted, and wakes the agent
    so it can put the item back up. Returns the number stored. Every platform call comes first,
    then the writes, so no write lock is held while the platform answers. Flushes; the caller
    commits."""
    query = select(Listing).where(
        Listing.platform == adapter.name,
        Listing.status == ListingStatus.LIVE,
        col(Listing.remote_id).is_not(None),
    )
    fetched: list[tuple[Listing, ListingStats | ListingGone]] = []
    for listing in session.exec(query).all():
        assert listing.remote_id is not None and listing.id is not None
        try:
            stats = adapter.stats(PlatformRef(listing.platform, listing.remote_id, listing.url))
        except ListingGone as exc:
            fetched.append((listing, exc))
            continue
        except PlatformAuthError:
            raise
        except PlatformError as exc:
            logger.warning("stats failed for listing %s: %s", listing.id, exc)
            continue
        fetched.append((listing, stats))
    stored = 0
    for listing, found in fetched:
        assert listing.id is not None
        if isinstance(found, ListingGone):
            _vanished(session, clock, notifier, listing, found)
            continue
        latest = _latest_snapshot(session, listing.id)
        numbers = (found.views, found.saves, found.message_count)
        if _fresh(latest, numbers, clock.now()):
            continue
        session.add(
            ListingStatsSnapshot(
                listing_id=listing.id,
                captured_at=clock.now(),
                views=found.views,
                saves=found.saves,
                message_count=found.message_count,
            )
        )
        stored += 1
    session.flush()
    return stored


def _vanished(
    session: Session, clock: Clock, notifier: Notifier | None, listing: Listing, why: ListingGone
) -> None:
    mark_gone(clock, listing, why)
    session.add(listing)
    item = session.get(Item, listing.item_id)
    assert item is not None
    if notifier is not None and item.status not in (ItemStatus.SOLD, ItemStatus.WITHDRAWN):
        notifier.notify(gone_notice(item, listing, why))  # a closed item's post is meant to go
    if item.status == ItemStatus.LISTED:
        session.add(
            WakeRequest(
                item_id=item.id,
                reason=f"The {listing.platform} post is gone from {listing.platform} (removed or "
                f"expired: {why}), so the item is not on {listing.platform} now. Relist or "
                f"publish on {listing.platform} to put it back up.",
                created_at=clock.now(),
            )
        )


def _fresh(
    latest: ListingStatsSnapshot | None, numbers: tuple[int, int, int], now: datetime
) -> bool:
    """The latest snapshot already holds these numbers and is younger than STATS_REFRESH."""
    if latest is None or now - latest.captured_at >= STATS_REFRESH:
        return False
    return numbers == (latest.views, latest.saves, latest.message_count)


def _latest_snapshot(session: Session, listing_id: int) -> ListingStatsSnapshot | None:
    query = (
        select(ListingStatsSnapshot)
        .where(ListingStatsSnapshot.listing_id == listing_id)
        .order_by(col(ListingStatsSnapshot.captured_at).desc(), col(ListingStatsSnapshot.id).desc())
    )
    return session.exec(query).first()


class _Ingestor:
    def __init__(
        self,
        session: Session,
        clock: Clock,
        settings: Settings,
        notifier: Notifier,
        result: IngestResult,
    ):
        self.session = session
        self.clock = clock
        self.settings = settings
        self.notifier = notifier
        self.result = result

    @property
    def platform(self) -> str:
        return self.result.platform

    def thread(self, remote: RemoteThread) -> None:
        messages = self._usable(remote)
        buyer = self._buyer(remote)
        thread = self._thread(remote, buyer, messages)
        for message in sorted(messages, key=lambda m: m.sent_at):
            if self._known(thread, message.remote_id):
                continue
            if message.from_buyer:
                self._inbound(thread, buyer, message)
            else:
                self._outbound(thread, message)
        self.session.flush()

    def _usable(self, remote: RemoteThread) -> list[RemoteMessage]:
        """Messages with an aware timestamp; the database rejects naive ones."""
        usable = []
        for message in remote.messages:
            if message.sent_at.tzinfo is None:
                self.result.errors.append(
                    f"{self.platform} thread {remote.remote_id}: skipped message "
                    f"{message.remote_id} with a timestamp that has no timezone"
                )
                continue
            usable.append(message)
        return usable

    def _buyer(self, remote: RemoteThread) -> Buyer:
        query = select(Buyer).where(
            Buyer.platform == self.platform, Buyer.handle == remote.buyer_handle
        )
        buyer = self.session.exec(query).first()
        if buyer is None:
            buyer = Buyer(platform=self.platform, handle=remote.buyer_handle)
        if remote.buyer_name and remote.buyer_name != buyer.display_name:
            buyer.display_name = remote.buyer_name
        self.session.add(buyer)
        self.session.flush()
        return buyer

    def _thread(self, remote: RemoteThread, buyer: Buyer, messages: list[RemoteMessage]) -> Thread:
        query = select(Thread).where(
            Thread.platform == self.platform, Thread.remote_id == remote.remote_id
        )
        thread = self.session.exec(query).first()
        if thread is not None:
            if thread.item_id is None:
                thread.item_id = self._item_for(remote.listing_remote_id)
                self.session.add(thread)
            return thread
        assert buyer.id is not None
        thread = Thread(
            platform=self.platform,
            remote_id=remote.remote_id,
            item_id=self._item_for(remote.listing_remote_id),
            buyer_id=buyer.id,
            created_at=min((m.sent_at for m in messages), default=self.clock.now()),
        )
        self.session.add(thread)
        self.session.flush()
        self.result.new_threads += 1
        if buyer.flagged_scam:
            self._freeze(thread, buyer, FLAGGED_BUYER_REASON)
        return thread

    def _item_for(self, listing_remote_id: str | None) -> int | None:
        if listing_remote_id is None:
            return None
        query = select(Listing.item_id).where(
            Listing.platform == self.platform, Listing.remote_id == listing_remote_id
        )
        return self.session.exec(query).first()

    def _known(self, thread: Thread, remote_id: str) -> bool:
        query = select(Message.id).where(
            Message.thread_id == thread.id, Message.remote_id == remote_id
        )
        return self.session.exec(query).first() is not None

    def _inbound(self, thread: Thread, buyer: Buyer, remote: RemoteMessage) -> None:
        message = Message(
            thread_id=thread.id,
            direction=Direction.INBOUND,
            body=redact_address(remote.body, self.settings),
            sent_at=remote.sent_at,
            remote_id=remote.remote_id,
            via="buyer",
        )
        arrived = _arrival(thread, remote.sent_at)
        if thread.last_inbound_at is None or arrived > thread.last_inbound_at:
            thread.last_inbound_at = arrived
        if thread.state == ThreadState.FROZEN:
            message.handled = True
        else:
            signal = detect_scam(remote.body)
            if signal is not None and signal.severity == "freeze":
                message.handled = True
                message.intent = "scam"
                self._freeze(thread, buyer, signal.reason)
            else:
                _note_offer(thread, remote.body)
        self.session.add_all([message, thread])
        self.session.flush()
        assert message.id is not None
        self.result.new_inbound.append(message.id)

    def _outbound(self, thread: Thread, remote: RemoteMessage) -> None:
        body = redact_address(remote.body, self.settings)
        echo = self._echo_of(thread, body, remote.sent_at)
        if echo is not None:
            if echo.remote_id is None:
                echo.remote_id = remote.remote_id
                self.session.add(echo)
            return
        self.session.add(
            Message(
                thread_id=thread.id,
                direction=Direction.OUTBOUND,
                body=body,
                sent_at=remote.sent_at,
                remote_id=remote.remote_id,
                handled=True,
                via="human",
            )
        )
        if thread.last_outbound_at is None or remote.sent_at > thread.last_outbound_at:
            thread.last_outbound_at = remote.sent_at
        self.session.add(thread)

    def _echo_of(self, thread: Thread, body: str, sent_at: datetime) -> Message | None:
        """A stored outbound with the same text sent within ECHO_WINDOW, preferring one that has
        no remote id yet."""
        query = select(Message).where(
            Message.thread_id == thread.id,
            Message.direction == Direction.OUTBOUND,
            col(Message.sent_at) >= sent_at - ECHO_WINDOW,
            col(Message.sent_at) <= sent_at + ECHO_WINDOW,
        )
        wanted = _flat(body)
        matches = [m for m in self.session.exec(query).all() if _flat(m.body) == wanted]
        matches.sort(key=lambda m: m.remote_id is not None)
        return matches[0] if matches else None

    def _freeze(self, thread: Thread, buyer: Buyer, reason: str) -> None:
        thread.state = ThreadState.FROZEN
        thread.frozen_reason = reason
        buyer.flagged_scam = True
        escalation = Escalation(
            item_id=thread.item_id,
            thread_id=thread.id,
            reason=f"Froze a conversation with {_name(buyer)}: {reason}.",
            created_at=self.clock.now(),
        )
        self.session.add_all([thread, buyer, escalation])
        self.session.flush()
        assert thread.id is not None
        self.result.frozen_threads.append(thread.id)
        self.notifier.notify(
            Notice(
                kind=NoticeKind.ALERT,
                title=f"Possible scam on {self.platform}",
                body=f"{_name(buyer)}: {reason}. The agent will not reply in this conversation.",
                item_id=thread.item_id,
                ref_id=escalation.id,
            )
        )


def _arrival(thread: Thread, sent_at: datetime) -> datetime:
    """When a new inbound message counts as written, for turn-taking: never before our last
    outbound. The platform stamps buyer messages on its own clock, often to the minute, so a
    reply that arrives after our message can carry an earlier time; arriving later, it is the
    buyer's answer to it, as outbound_streak (ordered by arrival) also reads it."""
    last_out = thread.last_outbound_at
    if last_out is not None and sent_at <= last_out:
        return last_out + timedelta(microseconds=1)
    return sent_at


def _note_offer(thread: Thread, body: str) -> None:
    offer = offer_cents(body)
    if offer is not None:
        thread.last_offer_cents = offer
    if thread.state == ThreadState.CLOSED or (
        thread.state == ThreadState.OPEN and offer is not None
    ):
        thread.state = ThreadState.NEGOTIATING


def _flat(text: str) -> str:
    return " ".join(text.split())


def _name(buyer: Buyer) -> str:
    return buyer.display_name or buyer.handle
