"""Real Kijiji calls, read only. Needs MA_LIVE_TESTS=1.

The public checks need no account. The account checks also need MA_KIJIJI_EMAIL and
MA_KIJIJI_PASSWORD; they sign in and read, never post, send, or mark anything read.
"""
import os

import pytest

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import SystemClock
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.platforms.kijiji import build_client
from marketplace_agent.platforms.kijiji.client import KijijiClient
from marketplace_agent.platforms.kijiji.mapping import search_url

LIVE = os.environ.get("MA_LIVE_TESTS") == "1"
SETTINGS = Settings() if LIVE else None
HAS_ACCOUNT = bool(SETTINGS and SETTINGS.kijiji_email and SETTINGS.kijiji_password)

pytestmark = [pytest.mark.live, pytest.mark.skipif(not LIVE, reason="needs MA_LIVE_TESTS=1")]
needs_account = pytest.mark.skipif(
    not HAS_ACCOUNT, reason="needs MA_KIJIJI_EMAIL and MA_KIJIJI_PASSWORD"
)


@pytest.fixture(scope="module")
def public_client() -> KijijiClient:
    return KijijiClient("", "", clock=SystemClock())


def test_public_search_and_listing_views(public_client):
    url = search_url("desk", None)
    assert url is not None

    hits = public_client.search(url, 5)

    assert hits, "Kijiji search returned nothing for 'desk'"
    listing = public_client.listing(hits[0].id)
    assert listing.views is not None


def test_missing_listing_is_gone(public_client):
    with pytest.raises(ListingGone):
        public_client.listing("1000000001")


@needs_account
def test_account_reads():
    assert SETTINGS is not None
    client = build_client(SETTINGS, SystemClock())

    assert client.login()
    client.my_listings()
    client.conversations(limit=5)
