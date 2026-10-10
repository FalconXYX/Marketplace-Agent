"""A fake www.kijiji.ca and id.kijiji.ca for httpx.MockTransport. No tests here.

Fixture provenance (tests/fixtures/kijiji): auth_*, cis_login.html, graphql_listing*,
graphql_unauthenticated, graphql_search, graphql_profile_listings, graphql_generate_xsrf,
graphql_category and graphql_location copy the shapes of live responses captured on 2026-10-10
with synthetic ids and text. graphql_conversation(s) and graphql_send_message follow the query
documents in Kijiji's v1.94.7 web bundle. post_ad_form, edit_ad_form, upload_image and
cis_verification are rebuilt from older code and are low confidence (see api_notes).
"""
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx

from marketplace_agent.core.clock import FakeClock
from marketplace_agent.platforms.kijiji.client import KijijiClient

FIXTURES = Path(__file__).parent / "fixtures" / "kijiji"
EMAIL = "seller@example.com"
PASSWORD = "hunter2-Secret!"
USER_ID = "1043690001"
SESSION_COOKIE = "__Secure-next-auth.session-token"
SSO_COOKIE = "TGC"
NEW_AD_ID = "1744800001"
AUTH_OPERATIONS = {"GetConversations", "GetConversation", "SendMessage"}


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def fixture_json(name: str) -> Any:
    return json.loads(fixture_text(name))


@dataclass
class FakeKijiji:
    """State a test can inspect or steer. login_outcome: ok | bad_password | challenge."""

    login_outcome: str = "ok"
    sso: bool = False  # id.kijiji.ca remembers the browser and skips the password form
    session: str | None = None  # the valid session token, once signed in
    sign_ins: int = 0
    password_posts: int = 0
    requests: list[httpx.Request] = field(default_factory=list)
    graphql_calls: list[str] = field(default_factory=list)
    listings: dict[str, dict[str, Any]] = field(default_factory=dict)
    conversations: list[dict[str, Any]] = field(default_factory=list)
    sent: list[tuple[str, str]] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    submitted: list[dict[str, str]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    submit_location: str = f"/v-buy-sell-desks/city-of-toronto/ikea-malm-desk-white/{NEW_AD_ID}"
    submit_cookie: str | None = None  # a kjrva cookie on the redirect after posting
    delete_reply: str = '{"status":"OK"}'
    queued: dict[str, list[httpx.Response | Exception]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self) -> None:
        listing = fixture_json("graphql_listing.json")["data"]["listing"]
        self.listings.setdefault(listing["id"], listing)
        if not self.conversations:
            self.conversations = fixture_json("graphql_conversations.json")["data"][
                "conversations"
            ]["items"]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def queue(self, key: str, *responses: httpx.Response | Exception) -> None:
        """Answer the next calls to key (a GraphQL operation or a path) with these first; an
        exception is raised as if the network failed."""
        self.queued[key].extend(responses)

    def expire_session(self) -> None:
        self.session = None

    def calls(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == path)

    # Routing

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = self._key(request)
        if request.url.path == "/anvil/api":
            self.graphql_calls.append(key)
        if self.queued.get(key):
            queued = self.queued[key].pop(0)
            if isinstance(queued, Exception):
                raise queued
            return queued
        host, path = request.url.host, request.url.path
        if host == "id.kijiji.ca":
            return self._identity(request, path)
        routes = {
            ("GET", "/api/auth/csrf"): lambda: _json("auth_csrf.json"),
            ("POST", "/api/auth/signin/cis"): lambda: self._signin(request),
            ("GET", "/api/auth/callback/cis"): self._callback,
            ("GET", "/api/auth/session"): lambda: self._session(request),
            ("POST", "/anvil/api"): lambda: self._graphql(request),
            ("GET", "/p-post-ad.html"): lambda: self._page(request, "post_ad_form.html"),
            ("GET", "/p-edit-ad.html"): lambda: self._page(request, "edit_ad_form.html"),
            ("POST", "/p-upload-image.html"): lambda: self._upload(request),
            ("POST", "/p-submit-ad.html"): lambda: self._submit(request),
            ("POST", "/j-delete-ad.json"): lambda: self._delete(request),
        }
        route = routes.get((request.method, path))
        if route is not None:
            return route()
        if request.method == "GET":
            return httpx.Response(200, html="<html><body>page</body></html>")
        return httpx.Response(404, text="not found")

    def _key(self, request: httpx.Request) -> str:
        if request.url.path == "/anvil/api":
            return json.loads(request.content)["operationName"]
        return request.url.path

    # id.kijiji.ca

    def _identity(self, request: httpx.Request, path: str) -> httpx.Response:
        if path == "/oidc/authorize":
            if self.sso and SSO_COOKIE in request.headers.get("cookie", ""):
                return _redirect("https://id.kijiji.ca/oauth2.0/callbackAuthorize?ticket=ST-1")
            return _redirect("https://id.kijiji.ca/login?service=x&state=fixture-state")
        if path == "/login" and request.method == "GET":
            return httpx.Response(200, html=fixture_text("cis_login.html"))
        if path == "/login":
            return self._password(request)
        if path == "/oauth2.0/callbackAuthorize":
            return _redirect(
                "https://www.kijiji.ca/api/auth/callback/cis?code=abc&state=fixture-state"
            )
        return httpx.Response(404)

    def _password(self, request: httpx.Request) -> httpx.Response:
        self.password_posts += 1
        form = _form(request)
        assert form.get("execution") == "fixture-execution-token", "hidden fields were not replayed"
        assert form.get("_eventId") == "submit"
        if self.login_outcome == "challenge":
            return httpx.Response(200, html=fixture_text("cis_verification.html"))
        if self.login_outcome == "bad_password" or form.get("password") != PASSWORD:
            return httpx.Response(200, html=fixture_text("cis_login.html"))
        assert form.get("username") == EMAIL
        return _redirect(
            "https://id.kijiji.ca/oauth2.0/callbackAuthorize?ticket=ST-1",
            cookies=[f"{SSO_COOKIE}=sso-1; Path=/; Secure; HttpOnly"],
        )

    # www.kijiji.ca

    def _signin(self, request: httpx.Request) -> httpx.Response:
        form = _form(request)
        assert form.get("csrfToken") == fixture_json("auth_csrf.json")["csrfToken"]
        assert form.get("json") == "true"
        return _json("auth_signin.json")

    def _callback(self) -> httpx.Response:
        self.sign_ins += 1
        self.session = f"session-{self.sign_ins}"
        cookie = f"{SESSION_COOKIE}={self.session}; Path=/; Secure; HttpOnly; SameSite=Lax"
        return _redirect("https://www.kijiji.ca/", cookies=[cookie])

    def _signed_in(self, request: httpx.Request) -> bool:
        cookie = request.headers.get("cookie", "")
        return self.session is not None and f"{SESSION_COOKIE}={self.session}" in cookie

    def _session(self, request: httpx.Request) -> httpx.Response:
        name = "auth_session.json" if self._signed_in(request) else "auth_session_signed_out.json"
        return _json(name)

    def _page(self, request: httpx.Request, fixture: str) -> httpx.Response:
        if not self._signed_in(request):
            return _redirect("https://www.kijiji.ca/consumer/login?redirectUrl=%2Fp-post-ad.html")
        return httpx.Response(200, html=fixture_text(fixture))

    def _upload(self, request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-Ebay-Box-Token") == "fixture-form-xsrf"
        self.uploads.append(request.headers["content-type"])
        return _json("upload_image.json")

    def _submit(self, request: httpx.Request) -> httpx.Response:
        if not self._signed_in(request):
            return _redirect("https://www.kijiji.ca/consumer/login")
        form = _form(request)
        self.submitted.append(form)
        if "postAdForm.adId" in form:
            return _redirect("https://www.kijiji.ca/m-my-ads/active/1")
        cookies = [f"kjrva={self.submit_cookie}; Path=/"] if self.submit_cookie else []
        return _redirect(self.submit_location, cookies=cookies)

    def _delete(self, request: httpx.Request) -> httpx.Response:
        form = _form(request)
        assert form.get("ca.kijiji.xsrf.token") == "fixture-xsrf-token"
        ad_id = re.search(r'"adId":"(\d+)"', form["ads"])
        assert ad_id is not None and form["Action"] == "DELETE_ADS"
        self.deleted.append(ad_id.group(1))
        return httpx.Response(200, text=self.delete_reply)

    # GraphQL

    def _graphql(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        operation, variables = payload["operationName"], payload["variables"]
        assert request.headers.get("apollo-require-preflight") == "true"
        if operation in AUTH_OPERATIONS and not self._signed_in(request):
            return _json("graphql_unauthenticated.json")
        handler = getattr(self, f"_op_{operation}", None)
        assert handler is not None, f"unexpected operation {operation}"
        return handler(variables)

    def _op_GetListing(self, variables: dict[str, Any]) -> httpx.Response:
        listing = self.listings.get(variables["listingId"])
        if listing is None:
            return _json("graphql_listing_not_found.json")
        return httpx.Response(200, json={"data": {"listing": listing}})

    def _op_GetSearchResultsPageByUrl(self, variables: dict[str, Any]) -> httpx.Response:
        return _json("graphql_search.json")

    def _op_GetCategories(self, variables: dict[str, Any]) -> httpx.Response:
        if variables["categoryId"] != 26:
            return httpx.Response(200, json={"data": {"category": None}})
        return _json("graphql_category.json")

    def _op_getLocationFromId(self, variables: dict[str, Any]) -> httpx.Response:
        if variables["locationId"] != 1700273:
            return httpx.Response(200, json={"data": {"location": None}})
        return _json("graphql_location.json")

    def _op_GetPosterOtherAds(self, variables: dict[str, Any]) -> httpx.Response:
        assert variables["profileId"] == USER_ID
        return _json("graphql_profile_listings.json")

    def _op_GetConversations(self, variables: dict[str, Any]) -> httpx.Response:
        assert variables["role"] == "SELLER"
        start, limit = variables["offset"], variables["limit"]
        items = self.conversations[start : start + limit]
        body = {"totalCount": len(self.conversations), "items": items}
        return httpx.Response(200, json={"data": {"conversations": body}})

    def _op_GetConversation(self, variables: dict[str, Any]) -> httpx.Response:
        assert variables["markAsRead"] is False
        body = fixture_json("graphql_conversation.json")
        body["data"]["conversation"]["id"] = variables["id"]
        return httpx.Response(200, json=body)

    def _op_SendMessage(self, variables: dict[str, Any]) -> httpx.Response:
        self.sent.append((variables["conversationId"], variables["message"]))
        return _json("graphql_send_message.json")

    def _op_generateXsrf(self, variables: dict[str, Any]) -> httpx.Response:
        return _json("graphql_generate_xsrf.json")


def make_client(
    fake: FakeKijiji, *, clock: FakeClock | None = None
) -> tuple[KijijiClient, list[float]]:
    """A client over the fake that records its sleeps instead of sleeping."""
    sleeps: list[float] = []
    client = KijijiClient(
        EMAIL,
        PASSWORD,
        transport=fake.transport(),
        clock=clock or FakeClock(),
        sleep=sleeps.append,
        rng=random.Random(7),
    )
    return client, sleeps


def _json(name: str) -> httpx.Response:
    return httpx.Response(200, json=fixture_json(name))


def _redirect(location: str, cookies: list[str] | None = None) -> httpx.Response:
    headers = [("Location", location)] + [("Set-Cookie", c) for c in cookies or []]
    return httpx.Response(302, headers=headers)


def _form(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.content.decode(), keep_blank_values=True).items()}
