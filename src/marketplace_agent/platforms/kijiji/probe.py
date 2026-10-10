"""Read-only probe of every Kijiji endpoint the adapter reads. Shared by `agent kijiji probe`
and `agent doctor`.

Nothing is posted, sent, edited, deleted, or marked read. The post form is only fetched.
"""
from collections.abc import Callable
from dataclasses import dataclass

from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.platforms.kijiji import api_notes as api
from marketplace_agent.platforms.kijiji.client import KijijiClient
from marketplace_agent.platforms.kijiji.mapping import search_url

PROBE_QUERY = "chair"
ACCOUNT_CHECKS = ("my ads", "conversations", "messages", "post form")


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool | None  # None: skipped
    detail: str

    def line(self) -> str:
        label = "skip" if self.ok is None else "ok" if self.ok else "FAIL"
        return f"{label:<4}  {self.name}: {self.detail}"


def run_probe(client: KijijiClient, location_id: int | None) -> list[ProbeResult]:
    checks = [
        _check("search (comps)", lambda: _search(client, location_id)),
        _check("location", lambda: _location(client, location_id)),
        _check("default category", lambda: _category(client)),
    ]
    sign_in = _check("sign in", lambda: f"signed in as user {client.user_id()}")
    checks.append(sign_in)
    if not sign_in.ok:
        return checks + [ProbeResult(name, None, "needs sign in") for name in ACCOUNT_CHECKS]
    checks.append(_check("my ads", lambda: _my_ads(client)))
    checks.append(_check("conversations", lambda: _conversations(client)))
    checks.append(_check("messages", lambda: _messages(client)))
    checks.append(_check("post form", lambda: _post_form(client)))
    return checks


class _Skip(Exception):
    """A check with nothing to look at."""


def _check(name: str, run: Callable[[], str]) -> ProbeResult:
    try:
        return ProbeResult(name, True, run())
    except _Skip as skip:
        return ProbeResult(name, None, str(skip))
    except PlatformError as exc:
        return ProbeResult(name, False, str(exc))
    except Exception as exc:  # a parsing bug must not hide the other checks
        return ProbeResult(name, False, f"{type(exc).__name__}: {exc}")


def _search(client: KijijiClient, location_id: int | None) -> str:
    url = search_url(PROBE_QUERY, location_id)
    assert url is not None
    hits = client.search(url, 3)
    priced = sum(hit.price_cents is not None for hit in hits)
    return f"{len(hits)} results for {PROBE_QUERY!r}, {priced} with a fixed price"


def _location(client: KijijiClient, location_id: int | None) -> str:
    if location_id is None:
        raise _Skip("MA_KIJIJI_LOCATION_ID not set; search covers all of Canada")
    name = client.location_name(location_id)
    if name is None:
        raise PlatformError(f"Kijiji has no location {location_id}")
    return f"{location_id} is {name}"


def _category(client: KijijiClient) -> str:
    return " > ".join(client.category_path(api.DEFAULT_CATEGORY_ID))


def _my_ads(client: KijijiClient) -> str:
    ads = client.my_listings()
    if not ads:
        return "0 active ads"
    newest = client.listing(ads[0].id)
    views = "unknown" if newest.views is None else str(newest.views)
    return f"{len(ads)} active ads; the newest ({newest.id}) has {views} views"


def _conversations(client: KijijiClient) -> str:
    return f"{client.conversations(limit=5).total} conversations as seller"


def _messages(client: KijijiClient) -> str:
    page = client.conversations(limit=1)
    if not page.items:
        raise _Skip("no conversations to read")
    messages = client.conversation_messages(page.items[0].id)
    inbound = sum(m.inbound for m in messages)
    return f"{len(messages)} messages in the latest conversation, {inbound} from the buyer"


def _post_form(client: KijijiClient) -> str:
    fields = client.post_form_fields(api.DEFAULT_CATEGORY_ID)
    missing = [key for key, found in fields.items() if not found]
    required = [key for key in api.REQUIRED_POST_FIELDS if key in missing]
    if required:
        raise PlatformError(f"post form lacks {', '.join(required)}; publishing will fail")
    found = len(fields) - len(missing)
    extra = f"; not on this form: {', '.join(missing)}" if missing else ""
    return f"{found}/{len(fields)} fields found{extra}"
