"""HTTP client for Kijiji's web endpoints. api_notes says which ones are verified.

| Rule | How |
|---|---|
| One request at a time | every public method holds one lock |
| Polite | a jittered gap of pace_s seconds between requests |
| Retries | 5xx and network errors, MAX_RETRIES times, reads and photo uploads only |
| No retry | posts, edits, deletes, messages: a retry could duplicate them |
| Session | signed in lazily; a signed-out answer signs in once more, then PlatformAuthError |
| Secrets | the email, password, and cookies never reach a log or an exception |
"""
import functools
import mimetypes
import random
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from marketplace_agent.core.clock import Clock
from marketplace_agent.platforms.base import ListingGone, PlatformAuthError, PlatformError
from marketplace_agent.platforms.kijiji import api_notes as api

MAX_RETRIES = 2
BACKOFF_S = 2.0
DEFAULT_PACE_S = (1.0, 3.0)
MESSAGE_PAGE = 50
MY_ADS_CHECKED = 50  # newest ads read before and after posting, to find the new one
_EXCERPT_CHARS = 200
_AD_ID = re.compile(r"(?:[?&]adId=|/)(\d{8,})(?:[/?#&]|$)")


@dataclass(frozen=True)
class KijijiListing:
    id: str
    title: str
    url: str | None
    status: str | None
    price_cents: int | None
    views: int | None


@dataclass(frozen=True)
class KijijiConversation:
    id: str
    listing_id: str | None
    counterparty_id: str | None
    counterparty_name: str
    last_message_at: datetime | None


@dataclass(frozen=True)
class ConversationPage:
    items: list[KijijiConversation]
    total: int


@dataclass(frozen=True)
class KijijiMessage:
    id: str
    body: str
    sent_at: datetime
    inbound: bool  # from the other party, not the seller


@dataclass(frozen=True)
class KijijiSearchHit:
    id: str
    title: str
    url: str | None
    price_cents: int | None  # None unless the ad has a fixed price
    posted_at: datetime | None


@dataclass(frozen=True)
class KijijiAd:
    """What the legacy post and edit forms take. Built by mapping.to_ad."""

    title: str
    description: str
    price: str  # dollars, as typed into the form
    category_id: int
    condition: str | None = None
    postal_code: str | None = None
    location_id: int | None = None


class _SignedOut(PlatformAuthError):
    """The session is gone; the caller signs in once more."""


def _serialized[**P, T](method: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        self = args[0]
        assert isinstance(self, KijijiClient)
        with self._lock:
            return method(*args, **kwargs)

    return wrapper


class KijijiClient:
    def __init__(
        self,
        email: str,
        password: str,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Clock,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        pace_s: tuple[float, float] = DEFAULT_PACE_S,
        timeout_s: float = 30.0,
    ):
        self._email = email
        self._password = password
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._pace_s = pace_s
        self._http = httpx.Client(
            base_url=api.BASE_URL,
            headers=api.COMMON_HEADERS,
            transport=transport,
            timeout=timeout_s,
            follow_redirects=True,
        )
        self._lock = threading.RLock()
        self._user_id: str | None = None
        self._last_request: datetime | None = None

    # Session

    @_serialized
    def login(self) -> str:
        """Sign in now, even with a live session. Returns the Kijiji user id."""
        self._user_id = None
        return self._signed_in()

    @_serialized
    def user_id(self) -> str:
        """The Kijiji user id, signing in only when there is no session yet."""
        return self._signed_in()

    # Public reads (no account needed)

    @_serialized
    def listing(self, listing_id: str) -> KijijiListing:
        """The ad as buyers see it, with its view count. ListingGone when it is not up."""
        variables = {"listingId": listing_id}
        data, codes, message = self._graphql("GetListing", api.LISTING_QUERY, variables)
        raw = data.get("listing")
        if raw is None and api.NOT_FOUND in codes:
            raise ListingGone(f"kijiji: ad {listing_id} not found")
        listing = _listing(_obj(raw, "GetListing", message))
        status = listing.status or ""
        if status in api.GONE_STATUSES:
            raise ListingGone(f"kijiji: ad {listing_id} is {status.lower()}")
        return listing

    @_serialized
    def search(self, url: str, limit: int) -> list[KijijiSearchHit]:
        page = {"offset": 0, "limit": limit}
        variables = {
            "searchResultsByUrlInput": {"url": url, "pagination": page},
            "pagination": page,
        }
        data, _, message = self._graphql("GetSearchResultsPageByUrl", api.SEARCH_QUERY, variables)
        found = _obj(data.get("searchResultsPageByUrl"), "search", message)
        results = _obj(found.get("results"), "search", message)
        hits = _list(results.get("mainListings"), "search")
        return [_search_hit(_obj(raw, "search")) for raw in hits]

    @_serialized
    def category_path(self, category_id: int) -> list[str]:
        """Names from the top category down, e.g. ["Buy & Sell", "Other"]."""
        variables = {"categoryId": category_id}
        data, _, message = self._graphql("GetCategories", api.CATEGORY_QUERY, variables)
        category = data.get("category")
        if category is None:
            raise PlatformError(f"kijiji: category {category_id} does not exist")
        paths = _list(_obj(category, "category", message).get("categoryPaths"), "category")
        return [_text(_obj(p, "category").get("localizedName"), "category") for p in paths]

    @_serialized
    def location_name(self, location_id: int) -> str | None:
        variables = {"locationId": location_id}
        data, _, message = self._graphql("getLocationFromId", api.LOCATION_QUERY, variables)
        location = data.get("location")
        if location is None:
            return None
        return _text(_obj(location, "location", message).get("localizedName"), "location")

    # Account reads and replies

    @_serialized
    def my_listings(self, limit: int = 50) -> list[KijijiListing]:
        """Active ads on the signed-in account, newest first, without view counts."""

        def fetch() -> list[KijijiListing]:
            variables = {
                "profileId": self._signed_in(),
                "sort": api.PROFILE_SORT,
                "pagination": {"offset": 0, "limit": limit},
            }
            query = api.PROFILE_LISTINGS_QUERY
            data, _, message = self._graphql("GetPosterOtherAds", query, variables)
            profile = _obj(data.get("profile"), "my ads", message)
            ads = _list(profile.get("listings"), "my ads")
            return [_listing(_obj(raw, "my ads")) for raw in ads]

        return self._with_session(fetch)

    @_serialized
    def conversations(self, limit: int = 25, offset: int = 0) -> ConversationPage:
        """Conversations where the account is the seller, most recent activity first."""

        def fetch() -> ConversationPage:
            variables = {"limit": limit, "offset": offset, "role": api.SELLER_ROLE}
            query = api.CONVERSATIONS_QUERY
            data, _, message = self._graphql("GetConversations", query, variables)
            found = _obj(data.get("conversations"), "conversations", message)
            raws = _list(found.get("items"), "conversations")
            items = [_conversation(_obj(raw, "conversations")) for raw in raws]
            return ConversationPage(items, _int(found.get("totalCount"), "conversations"))

        return self._with_session(fetch)

    @_serialized
    def conversation_messages(
        self, conversation_id: str, limit: int = MESSAGE_PAGE
    ) -> list[KijijiMessage]:
        """The latest messages, oldest first. Leaves the conversation's unread state alone."""

        def fetch() -> list[KijijiMessage]:
            variables = {"id": conversation_id, "markAsRead": False, "limit": limit}
            data, _, message = self._graphql("GetConversation", api.CONVERSATION_QUERY, variables)
            found = _obj(data.get("conversation"), "conversation", message)
            raws = _list(_obj(found.get("messages"), "conversation").get("items"), "conversation")
            messages = [_message(_obj(raw, "conversation")) for raw in raws]
            return sorted(messages, key=lambda m: m.sent_at)

        return self._with_session(fetch)

    @_serialized
    def send_message(self, conversation_id: str, body: str) -> str:
        def send() -> str:
            variables = {"conversationId": conversation_id, "message": body}
            data, _, message = self._graphql(
                "SendMessage", api.SEND_MESSAGE_MUTATION, variables, mutation=True
            )
            sent = _obj(data.get("sendMessage"), "send message", message)
            return _text(sent.get("messageId"), "send message")

        return self._with_session(send)

    # Legacy ad forms (low confidence; see api_notes)

    @_serialized
    def post_ad(self, ad: KijijiAd, photos: Sequence[Path]) -> str:
        """Post a new ad and return its id."""
        return self._with_session(lambda: self._post_ad(ad, photos))

    @_serialized
    def edit_ad(self, ad_id: str, ad: KijijiAd) -> None:
        """Change title, description, and price. ListingGone when the ad is not up."""
        self.listing(ad_id)
        self._with_session(lambda: self._edit_ad(ad_id, ad))

    @_serialized
    def delete_ad(self, ad_id: str) -> None:
        """ListingGone when the ad is already down."""
        self.listing(ad_id)
        self._with_session(lambda: self._delete_ad(ad_id))

    @_serialized
    def post_form_fields(self, category_id: int) -> dict[str, bool]:
        """Read-only check of the post form: which fields this client fills are present."""

        def check() -> dict[str, bool]:
            page = self._page("GET", api.POST_AD_PATH, params={"categoryId": category_id})
            form = _ad_form(page)
            return {key: _field_name(form, key) is not None for key in api.POST_FIELDS}

        return self._with_session(check)

    # Internals: ad forms

    def _post_ad(self, ad: KijijiAd, photos: Sequence[Path]) -> str:
        """The new id from the redirect, else an ad with this title that was not in My Ads
        before the submit: an older ad with the same title is never taken for the new one."""
        page = self._page("GET", api.POST_AD_PATH, params={"categoryId": ad.category_id})
        form = _ad_form(page)
        before = {listing.id for listing in self.my_listings(limit=MY_ADS_CHECKED)}
        images = [self._upload_image(photo, form) for photo in photos]
        values = _ad_values(ad) | {"images": ",".join(images) if images else None}
        result = self._submit(page, form, values)
        ad_id = _new_ad_id(result)
        if ad_id is None and _find_ad_form(result.text) is not None:
            raise PlatformError(
                "kijiji: Kijiji sent the post form back, so the ad was not posted (a rejected "
                "field, or a duplicate of one of your ads). Check the item on kijiji.ca."
            )
        ad_id = ad_id or self._my_ad_titled(ad.title, exclude=before)
        if ad_id is None:
            raise PlatformError(
                "kijiji: the ad was not confirmed (no new ad id and no new matching ad in My "
                "Ads). Check My Ads on kijiji.ca before retrying, then run `agent kijiji probe`."
            )
        return ad_id

    def _edit_ad(self, ad_id: str, ad: KijijiAd) -> None:
        page = self._page("GET", api.EDIT_AD_PATH, params={"adId": ad_id})
        form = _ad_form(page)
        result = self._submit(page, form, _ad_values(ad))
        if _find_ad_form(result.text) is not None:
            raise PlatformError(f"kijiji: Kijiji sent back the edit form for ad {ad_id}; not saved")

    def _delete_ad(self, ad_id: str) -> None:
        mutation = api.GENERATE_XSRF_MUTATION
        data, _, message = self._graphql("generateXsrf", mutation, {}, mutation=True)
        token = _text(_obj(data.get("generateXsrf"), "xsrf", message).get("xsrf"), "xsrf")
        reason = f'[{{"adId":"{ad_id}","reason":"{api.DELETE_REASON}","otherReason":""}}]'
        fields = {
            "Action": "DELETE_ADS",
            "Mode": "ACTIVE",
            "needsRedirect": "false",
            "ads": reason,
            api.XSRF_FIELD: token,
        }
        result = self._page("POST", api.DELETE_AD_PATH, data=fields, idempotent=False)
        text = result.text.strip()
        if api.DELETE_OK_MARKER not in text or text.startswith("<"):  # an HTML page is no answer
            raise PlatformError(f"kijiji: Kijiji did not confirm deleting ad {ad_id}")

    def _upload_image(self, photo: Path, form: "_Form") -> str:
        try:
            content = photo.read_bytes()
        except OSError as exc:
            raise PlatformError(f"kijiji: cannot read photo {photo.name}: {exc.strerror}") from None
        kind = mimetypes.guess_type(photo.name)[0] or "image/jpeg"
        token = form.fields.get(api.XSRF_FIELD)
        headers = {api.XSRF_HEADER: token} if token else {}
        response = self._page(
            "POST",
            api.UPLOAD_IMAGE_PATH,
            files={"file": (photo.name, content, kind)},
            headers=headers,
        )
        body = _json(response, "photo upload")
        for key in api.UPLOAD_URL_KEYS:
            url = body.get(key) if isinstance(body, dict) else None
            if isinstance(url, str) and url:
                return url
        raise PlatformError("kijiji: photo upload returned no image url")

    def _submit(
        self, page: httpx.Response, form: "_Form", values: dict[str, str | None]
    ) -> httpx.Response:
        fields = dict(form.fields)
        for key, value in values.items():
            if value is None:
                continue
            name = _field_name(form, key)
            if name is None:
                if key in api.REQUIRED_POST_FIELDS or key == "images":
                    names = ", ".join(api.POST_FIELDS[key])
                    raise PlatformError(
                        f"kijiji: the ad form has no {key} field (looked for {names}). Kijiji "
                        "changed the form; run `agent kijiji probe`."
                    )
                continue
            fields[name] = value
        action = urljoin(str(page.url), form.action or str(page.url))
        return self._page("POST", action, data=fields, idempotent=False)

    def _my_ad_titled(self, title: str, *, exclude: set[str]) -> str | None:
        for listing in self.my_listings(limit=MY_ADS_CHECKED):
            if listing.id not in exclude and listing.title.strip() == title.strip():
                return listing.id
        return None

    # Internals: session

    def _signed_in(self) -> str:
        if self._user_id is None:
            self._sign_in()
        assert self._user_id is not None
        return self._user_id

    def _with_session[T](self, call: Callable[[], T]) -> T:
        self._signed_in()
        try:
            return call()
        except _SignedOut:
            self._user_id = None
            self._signed_in()
            try:
                return call()
            except _SignedOut:
                raise PlatformAuthError("kijiji: signed out again right after signing in") from None

    def _sign_in(self) -> None:
        csrf = _json_field(self._request("GET", api.AUTH_CSRF_PATH), "csrfToken")
        start = self._request(
            "POST",
            api.AUTH_SIGNIN_PATH,
            data={"csrfToken": csrf, "callbackUrl": f"{api.BASE_URL}/", "json": "true"},
        )
        page = self._request("GET", _json_field(start, "url"))
        if page.url.host == api.LOGIN_HOST:
            page = self._send_credentials(page)
        if page.url.host == api.LOGIN_HOST:
            raise PlatformAuthError(_sign_in_problem(page))
        self._user_id = self._session_user()

    def _send_credentials(self, page: httpx.Response) -> httpx.Response:
        form = _find_form(page.text, lambda f: f.id == api.LOGIN_FORM_ID)
        if form is None:
            raise PlatformError("kijiji: the sign-in page has no login form; Kijiji changed it")
        fields = dict(form.fields)
        fields[api.LOGIN_USERNAME_FIELD] = self._email
        fields[api.LOGIN_PASSWORD_FIELD] = self._password
        fields[api.LOGIN_TRUST_FIELD] = "on"
        action = urljoin(str(page.url), form.action or str(page.url))
        return self._request("POST", action, data=fields, idempotent=False)

    def _session_user(self) -> str:
        body = _json(self._request("GET", api.AUTH_SESSION_PATH), "session")
        user = body.get("user") if isinstance(body, dict) else None
        if not isinstance(user, dict) or user.get("error") == api.SESSION_REFRESH_ERROR:
            raise PlatformAuthError("kijiji: sign in finished but Kijiji did not start a session")
        user_id = _optional_id(user.get("sub"))
        if user_id is None:
            raise PlatformAuthError("kijiji: the session has no user id")
        return user_id

    # Internals: transport

    def _graphql(
        self, operation: str, query: str, variables: dict[str, Any], *, mutation: bool = False
    ) -> tuple[dict[str, Any], set[str], str | None]:
        """data, error codes, and the first error message (redacted)."""
        payload = {"operationName": operation, "query": query, "variables": variables}
        response = self._request(
            "POST",
            api.GRAPHQL_PATH,
            json=payload,
            headers=api.GRAPHQL_HEADERS,
            idempotent=not mutation,
            passthrough=api.GRAPHQL_ERROR_STATUS,
        )
        body = _obj(_json(response, operation), operation)
        errors = [e for e in body.get("errors") or [] if isinstance(e, dict)]
        codes = {str((e.get("extensions") or {}).get("code")) for e in errors}
        message = self._redact(str(errors[0].get("message"))) if errors else None
        if api.UNAUTHENTICATED in codes:
            raise _SignedOut(f"kijiji: {operation} needs a signed-in session")
        data = body.get("data")
        if not isinstance(data, dict):
            raise PlatformError(f"kijiji: {operation} failed: {message or 'no data returned'}")
        return data, codes, message

    def _page(
        self, method: str, url: str, *, idempotent: bool = True, **kwargs: Any
    ) -> httpx.Response:
        """A legacy page or form post; a bounce to the sign-in page means signed out."""
        response = self._request(method, url, idempotent=idempotent, **kwargs)
        url_now = response.url
        if url_now.host == api.LOGIN_HOST or url_now.path.startswith(api.CONSUMER_LOGIN_PATH):
            raise _SignedOut("kijiji: the page asked to sign in")
        return response

    def _request(
        self,
        method: str,
        url: str,
        *,
        idempotent: bool = True,
        passthrough: int | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """passthrough: a 4xx status whose body the caller reads instead of failing here."""
        attempts = 1 + (MAX_RETRIES if idempotent else 0)
        failure = PlatformError("kijiji: no request was made")
        for attempt in range(attempts):
            if attempt:
                self._sleep(BACKOFF_S * 2 ** (attempt - 1))
            self._pace()
            try:
                response = self._http.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                failure = PlatformError(f"kijiji: network error ({type(exc).__name__})")
                continue
            if response.status_code == passthrough:
                return response
            if response.status_code < 500:
                return _checked(response)
            failure = PlatformError(f"kijiji: HTTP {response.status_code} from {response.url.path}")
        raise failure

    def _pace(self) -> None:
        now = self._clock.now()
        if self._last_request is not None:
            wait = self._rng.uniform(*self._pace_s) - (now - self._last_request).total_seconds()
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock.now()

    def _redact(self, text: str) -> str:
        for secret in (self._password, self._email):
            if secret:
                text = text.replace(secret, "***")
        return text[:_EXCERPT_CHARS]


def _checked(response: httpx.Response) -> httpx.Response:
    status = response.status_code
    where = response.url.path
    if status == 401:
        raise _SignedOut(f"kijiji: HTTP 401 from {where}")
    if status == 403:
        raise PlatformError(
            f"kijiji: HTTP 403 from {where}; Kijiji's bot protection is refusing this network"
        )
    if status == 429:
        raise PlatformError(f"kijiji: rate limited (HTTP 429 from {where}); slow down")
    if status >= 400:
        raise PlatformError(f"kijiji: HTTP {status} from {where}")
    return response


def _sign_in_problem(page: httpx.Response) -> str:
    if _find_form(page.text, lambda f: f.id == api.LOGIN_FORM_ID) is not None:
        return (
            "kijiji: Kijiji rejected the email or password (or locked the account). Sign in at "
            "kijiji.ca in a browser to check, update MA_KIJIJI_PASSWORD, then restart `agent "
            "serve` (or `agent run`): it reads the password only when it starts."
        )
    return (
        "kijiji: Kijiji asked for an extra sign-in step, usually a code it emailed you, which "
        "the agent cannot complete. It signs in again by itself on every tick; if Kijiji keeps "
        "asking, sign in once at kijiji.ca in a browser on this network."
    )


# Parsing


def _json(response: httpx.Response, operation: str) -> Any:
    try:
        return response.json()
    except ValueError:
        raise PlatformError(f"kijiji: {operation} did not answer with JSON") from None


def _json_field(response: httpx.Response, key: str) -> str:
    return _text(_obj(_json(response, "sign in"), "sign in").get(key), "sign in")


def _unexpected(operation: str, message: str | None = None) -> PlatformError:
    suffix = f": {message}" if message else ""
    return PlatformError(f"kijiji: unexpected response from {operation}{suffix}")


def _obj(value: Any, operation: str, message: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _unexpected(operation, message)
    return value


def _list(value: Any, operation: str) -> list[Any]:
    if not isinstance(value, list):
        raise _unexpected(operation)
    return value


def _text(value: Any, operation: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str) or not value:
        raise _unexpected(operation)
    return value


def _int(value: Any, operation: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _unexpected(operation)
    return value


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_id(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    return str(value) or None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _when(value: Any) -> datetime | None:
    """ISO 8601 from Kijiji ("2026-10-09T19:27:06.000Z") as aware UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _fixed_cents(price: Any) -> int | None:
    if not isinstance(price, dict) or price.get("type") != api.FIXED_PRICE:
        return None
    return _optional_int(price.get("amount"))


def _listing(raw: dict[str, Any]) -> KijijiListing:
    metrics = raw.get("metrics")
    return KijijiListing(
        id=_text(raw.get("id"), "listing"),
        title=_optional_str(raw.get("title")) or "",
        url=_optional_str(raw.get("url")),
        status=_optional_str(raw.get("status")),
        price_cents=_fixed_cents(raw.get("price")),
        views=_optional_int(metrics.get("views")) if isinstance(metrics, dict) else None,
    )


def _search_hit(raw: dict[str, Any]) -> KijijiSearchHit:
    return KijijiSearchHit(
        id=_text(raw.get("id"), "search"),
        title=_optional_str(raw.get("title")) or "",
        url=_optional_str(raw.get("url")),
        price_cents=_fixed_cents(raw.get("price")),
        posted_at=_when(raw.get("activationDate")),
    )


def _conversation(raw: dict[str, Any]) -> KijijiConversation:
    counterparty = raw.get("counterparty") if isinstance(raw.get("counterparty"), dict) else {}
    listing = raw.get("listing") if isinstance(raw.get("listing"), dict) else {}
    return KijijiConversation(
        id=_text(raw.get("id"), "conversations"),
        listing_id=_optional_id(listing.get("id")),
        counterparty_id=_optional_id(counterparty.get("userId")),
        counterparty_name=_optional_str(counterparty.get("name")) or "",
        last_message_at=_when(raw.get("lastMessageAt")),
    )


def _message(raw: dict[str, Any]) -> KijijiMessage:
    sent_at = _when(raw.get("sentAt"))
    if sent_at is None:
        raise _unexpected("conversation")
    return KijijiMessage(
        id=_text(raw.get("id"), "conversation"),
        body=raw.get("body") if isinstance(raw.get("body"), str) else "",
        sent_at=sent_at,
        inbound=raw.get("direction") == api.INBOUND,
    )


def _new_ad_id(response: httpx.Response) -> str | None:
    """The id from the redirect chain or its kjrva cookie. Only this response's own cookies count:
    the jar may still hold one from an earlier post."""
    chain = (*response.history, response)
    for step in chain:
        url = str(step.url)
        found = _AD_ID.search(url)
        if found and api.POST_AD_PATH not in url:
            return found.group(1)
    for step in chain:
        value = step.cookies.get(api.NEW_AD_COOKIE)
        if value and value.isdigit():
            return value
    return None


# Legacy forms


def _ad_values(ad: KijijiAd) -> dict[str, str | None]:
    return {
        "title": ad.title,
        "description": ad.description,
        "price": ad.price,
        "price_type": api.FIXED_PRICE,
        "postal_code": ad.postal_code,
        "location_id": str(ad.location_id) if ad.location_id is not None else None,
        "condition": ad.condition,
        "for_sale_by": api.FOR_SALE_BY_OWNER,
        "terms": "true",
        "package": api.BASIC_PACKAGE,
    }


def _field_name(form: "_Form", key: str) -> str | None:
    return next((name for name in api.POST_FIELDS[key] if name in form.names), None)


def _ad_form(page: httpx.Response) -> "_Form":
    form = _find_ad_form(page.text)
    if form is None:
        raise PlatformError(
            f"kijiji: no ad form on {page.url.path}; Kijiji changed the page. "
            "Run `agent kijiji probe`."
        )
    return form


def _find_ad_form(html: str) -> "_Form | None":
    titles = api.POST_FIELDS["title"]
    return _find_form(html, lambda f: any(name in f.names for name in titles))


def _find_form(html: str, wanted: Callable[["_Form"], bool]) -> "_Form | None":
    parser = _FormParser()
    parser.feed(html)
    parser.close()
    return next((form for form in parser.forms if wanted(form)), None)


@dataclass
class _Form:
    id: str | None
    action: str
    fields: dict[str, str] = field(default_factory=dict)  # what a browser would submit as is
    names: set[str] = field(default_factory=set)  # every named control, chosen or not


_SKIPPED_INPUTS = {"submit", "button", "image", "reset", "file"}


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[_Form] = []
        self._form: _Form | None = None
        self._select: str | None = None
        self._first_option: str | None = None
        self._textarea: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "form":
            self._form = _Form(id=values.get("id"), action=values.get("action", ""))
            self.forms.append(self._form)
            return
        name = values.get("name")
        if self._form is None:
            return
        if tag == "input" and name:
            self._input(name, values)
        elif tag == "select" and name:
            self._form.names.add(name)
            self._select, self._first_option = name, None
        elif tag == "option" and self._select is not None:
            option = values.get("value", "")
            if self._first_option is None:
                self._first_option = option
            if "selected" in values:
                self._form.fields[self._select] = option
        elif tag == "textarea" and name:
            self._form.names.add(name)
            self._textarea, self._text = name, []

    def _input(self, name: str, values: dict[str, str]) -> None:
        assert self._form is not None
        kind = values.get("type", "text").lower()
        if kind in _SKIPPED_INPUTS:
            return
        self._form.names.add(name)
        if kind in ("checkbox", "radio"):
            if "checked" in values:
                self._form.fields[name] = values.get("value") or "on"
            return
        self._form.fields[name] = values.get("value", "")

    def handle_data(self, data: str) -> None:
        if self._textarea is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._form is None:
            return
        if tag == "textarea" and self._textarea is not None:
            self._form.fields[self._textarea] = "".join(self._text)
            self._textarea = None
        elif tag == "select" and self._select is not None:
            if self._select not in self._form.fields and self._first_option is not None:
                self._form.fields[self._select] = self._first_option
            self._select = None
        elif tag == "form":
            self._form = None
