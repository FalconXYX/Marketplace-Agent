"""The adapter boundary. Every marketplace implements PlatformAdapter; nothing above it knows which."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PlatformRef:
    platform: str
    remote_id: str
    url: str | None = None


@dataclass
class ListingPayload:
    title: str
    description: str
    price_cents: int
    photo_paths: list[Path]
    location: str                      # pickup area / cross street, never an exact address
    category: str | None = None
    condition: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class RemoteMessage:
    remote_id: str
    body: str
    sent_at: datetime                  # aware UTC
    from_buyer: bool


@dataclass
class RemoteThread:
    remote_id: str
    listing_remote_id: str | None
    buyer_handle: str
    buyer_name: str
    messages: list[RemoteMessage]      # ingestion dedupes by remote_id, so returning extras is safe


@dataclass
class ListingStats:
    views: int = 0
    saves: int = 0
    message_count: int = 0


@dataclass
class SearchResult:
    title: str
    price_cents: int
    url: str | None = None
    posted_at: datetime | None = None


class PlatformError(Exception):
    """Recoverable platform failure (network, unexpected response)."""


class ListingGone(PlatformError):
    """The post is not on the platform any more (removed, expired, deleted).

    Raised by update, delist, and stats. Callers treat it as "already down", never as a
    failure to retry. An adapter that cannot tell a missing post from a timeout raises
    plain PlatformError instead.
    """


class PlatformAuthError(PlatformError):
    """Session expired or credentials rejected. Needs a human."""


class AwaitingHuman(PlatformError):
    """Assist mode: the adapter prepared the action and a human must finish it."""


class PlatformAdapter(Protocol):
    name: str

    def publish(self, listing: ListingPayload, *, assist: bool = False) -> PlatformRef:
        """Create a listing. assist=True: prepare it and leave the final submit to a human."""
        ...

    def update(self, ref: PlatformRef, listing: ListingPayload) -> None: ...

    def delist(self, ref: PlatformRef) -> None: ...

    def fetch_threads(self, since: datetime | None) -> list[RemoteThread]: ...

    def send_message(self, thread_remote_id: str, body: str) -> str | None:
        """Send to a buyer. Returns the remote message id when the platform provides one."""
        ...

    def stats(self, ref: PlatformRef) -> ListingStats: ...

    def search(self, query: str, limit: int = 20) -> list[SearchResult]: ...

    def healthcheck(self) -> list[str]:
        """Problems found; empty list means healthy."""
        ...
