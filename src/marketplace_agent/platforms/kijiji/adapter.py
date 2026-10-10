"""PlatformAdapter for Kijiji over KijijiClient.

| Method | Source |
|---|---|
| publish, update, delist | legacy ad forms (low confidence) |
| fetch_threads, send_message | GraphQL conversations, seller side only |
| stats | public view count, plus buyer conversations about the ad; Kijiji shows no saves |
| search | public GraphQL search in Buy & Sell near kijiji_location_id |
| healthcheck | postal code set, conversations readable (sign in) |
| probe | probe.run_probe on this client, read only, for `agent doctor` |
"""
from collections.abc import Iterator
from datetime import datetime, timedelta

from marketplace_agent.core.clock import Clock
from marketplace_agent.platforms.base import (
    ListingPayload,
    ListingStats,
    PlatformError,
    PlatformRef,
    RemoteThread,
    SearchResult,
)
from marketplace_agent.platforms.kijiji import api_notes as api
from marketplace_agent.platforms.kijiji.client import KijijiAd, KijijiClient, KijijiConversation
from marketplace_agent.platforms.kijiji.mapping import (
    search_url,
    to_ad,
    to_search_result,
    to_thread,
)
from marketplace_agent.platforms.kijiji.probe import ProbeResult, run_probe

CONVERSATION_PAGE = 25
MAX_CONVERSATION_PAGES = 8
CONVERSATIONS_FRESH_FOR = timedelta(minutes=5)  # stats reuse the list fetch_threads just read


class KijijiAdapter:
    name = "kijiji"

    def __init__(
        self,
        client: KijijiClient,
        clock: Clock,
        *,
        location_id: int | None = None,
        postal_code: str | None = None,
    ):
        self._client = client
        self._clock = clock
        self._location_id = location_id
        self._postal_code = postal_code
        self._conversations: tuple[datetime, list[KijijiConversation]] | None = None

    def publish(self, listing: ListingPayload, *, assist: bool = False) -> PlatformRef:
        """Posts straight away; assist does not apply to Kijiji."""
        if not self._postal_code and self._location_id is None:
            raise PlatformError(
                "kijiji: set MA_KIJIJI_POSTAL_CODE (and MA_KIJIJI_LOCATION_ID) so ads have a "
                "location"
            )
        ad_id = self._client.post_ad(self._ad(listing), listing.photo_paths)
        return PlatformRef(self.name, ad_id, api.VIEW_URL.format(ad_id=ad_id))

    def update(self, ref: PlatformRef, listing: ListingPayload) -> None:
        self._client.edit_ad(ref.remote_id, self._ad(listing))

    def delist(self, ref: PlatformRef) -> None:
        self._client.delete_ad(ref.remote_id)

    def fetch_threads(self, since: datetime | None) -> list[RemoteThread]:
        """Seller conversations with activity at or after since (all when None), each with its
        latest messages."""
        threads = []
        for conversation in self._fresh_conversations():
            last = conversation.last_message_at
            if since is not None and last is not None and last < since:
                continue
            messages = self._client.conversation_messages(conversation.id)
            threads.append(to_thread(conversation, messages))
        return threads

    def send_message(self, thread_remote_id: str, body: str) -> str:
        return self._client.send_message(thread_remote_id, body)

    def stats(self, ref: PlatformRef) -> ListingStats:
        listing = self._client.listing(ref.remote_id)
        about = [c for c in self._cached_conversations() if c.listing_id == ref.remote_id]
        return ListingStats(views=listing.views or 0, saves=0, message_count=len(about))

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        url = search_url(query, self._location_id)
        if url is None or limit <= 0:
            return []
        hits = (to_search_result(hit) for hit in self._client.search(url, limit))
        return [hit for hit in hits if hit is not None]

    def healthcheck(self) -> list[str]:
        problems = []
        if not self._postal_code and self._location_id is None:
            problems.append("kijiji: MA_KIJIJI_POSTAL_CODE is not set, so ads cannot be posted")
        try:
            self._client.conversations(limit=1)
        except PlatformError as exc:
            problems.append(str(exc))
        return problems

    def probe(self) -> list[ProbeResult]:
        """Every endpoint the adapter reads, without writing anything. Reuses the signed-in
        session."""
        return run_probe(self._client, self._location_id)

    def _ad(self, listing: ListingPayload) -> KijijiAd:
        return to_ad(listing, location_id=self._location_id, postal_code=self._postal_code)

    def _fresh_conversations(self) -> list[KijijiConversation]:
        conversations = list(self._all_conversations())
        self._conversations = (self._clock.now(), conversations)
        return conversations

    def _cached_conversations(self) -> list[KijijiConversation]:
        if self._conversations is not None:
            read_at, conversations = self._conversations
            if self._clock.now() - read_at < CONVERSATIONS_FRESH_FOR:
                return conversations
        return self._fresh_conversations()

    def _all_conversations(self) -> Iterator[KijijiConversation]:
        offset = 0
        for _ in range(MAX_CONVERSATION_PAGES):
            page = self._client.conversations(limit=CONVERSATION_PAGE, offset=offset)
            yield from page.items
            offset += len(page.items)
            if not page.items or offset >= page.total:
                return
