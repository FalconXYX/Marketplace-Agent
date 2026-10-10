import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from marketplace_agent.platforms.base import ListingGone, PlatformAuthError, PlatformError
from marketplace_agent.platforms.kijiji.client import KijijiAd, MAX_RETRIES
from test_kijiji_support import (
    EMAIL,
    NEW_AD_ID,
    PASSWORD,
    USER_ID,
    FakeKijiji,
    fixture_json,
    fixture_text,
    make_client,
)

AD = KijijiAd(
    title="IKEA Malm desk, white",
    description="White Malm desk. Pickup near Bloor and Ossington.",
    price="120",
    category_id=239,
    condition="usedgood",
    postal_code="M6J 1Z1",
    location_id=1700273,
)


def _photos(tmp_path: Path, count: int = 2) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / f"photo{index}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
        paths.append(path)
    return paths


# Sign in


def test_login_walks_the_cis_flow_and_returns_the_user_id():
    fake = FakeKijiji()
    client, _ = make_client(fake)

    assert client.login() == USER_ID

    assert fake.sign_ins == 1 and fake.password_posts == 1
    posts = [r for r in fake.requests if r.url.host == "id.kijiji.ca" and r.method == "POST"]
    (password_post,) = posts
    assert b"trustThisDevice=on" in password_post.content


def test_session_is_reused_across_calls():
    fake = FakeKijiji()
    client, _ = make_client(fake)

    client.conversations()
    client.conversations()
    client.my_listings()

    assert fake.calls("/api/auth/csrf") == 1


def test_user_id_signs_in_only_without_a_session_and_login_always_does():
    fake = FakeKijiji()
    client, _ = make_client(fake)

    assert client.user_id() == USER_ID
    client.conversations()
    assert client.user_id() == USER_ID
    assert fake.sign_ins == 1

    assert client.login() == USER_ID
    assert fake.sign_ins == 2


def test_signs_in_again_once_when_the_session_expires():
    fake = FakeKijiji(sso=True)
    client, _ = make_client(fake)
    client.conversations()
    fake.expire_session()

    page = client.conversations()

    assert page.total == 2
    assert fake.sign_ins == 2
    assert fake.password_posts == 1  # the identity server remembered the browser


def test_unauthenticated_after_a_fresh_sign_in_is_an_auth_error():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    unauthenticated = httpx.Response(200, json=fixture_json("graphql_unauthenticated.json"))
    fake.queue("GetConversations", unauthenticated, unauthenticated)

    with pytest.raises(PlatformAuthError):
        client.conversations()

    assert fake.sign_ins == 2


def test_http_401_triggers_one_sign_in():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    client.login()
    fake.queue("GetConversations", httpx.Response(401))

    assert client.conversations().total == 2
    assert fake.sign_ins == 2


def test_wrong_password_is_an_auth_error():
    fake = FakeKijiji(login_outcome="bad_password")
    client, _ = make_client(fake)

    with pytest.raises(PlatformAuthError, match="rejected the email or password") as raised:
        client.login()
    assert "restart `agent serve`" in str(raised.value)  # the password is read at start only


def test_verification_step_is_an_auth_error():
    """There is no Kijiji login command: the message says the agent retries by itself."""
    fake = FakeKijiji(login_outcome="challenge")
    client, _ = make_client(fake)

    with pytest.raises(PlatformAuthError, match="extra sign-in step") as raised:
        client.conversations()
    assert "signs in again by itself" in str(raised.value)


def test_session_without_user_is_an_auth_error():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("/api/auth/session", httpx.Response(200, json={}))

    with pytest.raises(PlatformAuthError, match="did not start a session"):
        client.login()


# Transport


def test_5xx_is_retried_with_backoff():
    fake = FakeKijiji()
    client, sleeps = make_client(fake)
    fake.queue("GetListing", httpx.Response(503), httpx.Response(502))

    assert client.listing("1744700001").views == 37
    assert 2.0 in sleeps and 4.0 in sleeps


def test_5xx_after_retries_is_a_platform_error():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("GetListing", *[httpx.Response(503)] * (MAX_RETRIES + 1))

    with pytest.raises(PlatformError, match="HTTP 503") as caught:
        client.listing("1744700001")

    assert not isinstance(caught.value, ListingGone)
    assert fake.graphql_calls.count("GetListing") == MAX_RETRIES + 1


def test_network_errors_are_retried():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("GetListing", httpx.ConnectError("boom"))

    assert client.listing("1744700001").id == "1744700001"


def test_network_error_after_retries_is_a_platform_error():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("GetListing", *[httpx.ReadTimeout("slow")] * (MAX_RETRIES + 1))

    with pytest.raises(PlatformError, match="network error"):
        client.listing("1744700001")


def test_messages_are_never_retried():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    client.login()
    fake.queue("SendMessage", httpx.Response(503))

    with pytest.raises(PlatformError, match="HTTP 503"):
        client.send_message("c-9001", "Still available.")

    assert fake.graphql_calls.count("SendMessage") == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>maintenance</html>"),
        httpx.Response(200, json={"data": {"listing": {"title": "no id"}}}),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"errors": [{"message": "boom"}], "data": None}),
    ],
    ids=["html", "missing-id", "list", "error-without-data"],
)
def test_malformed_bodies_are_platform_errors(response):
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("GetListing", response)

    with pytest.raises(PlatformError) as caught:
        client.listing("1744700001")

    assert type(caught.value) is PlatformError


@pytest.mark.parametrize("status", [403, 429, 404])
def test_client_errors_are_platform_errors(status):
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("GetListing", httpx.Response(status))

    with pytest.raises(PlatformError, match=f"{status}"):
        client.listing("1744700001")


def test_schema_drift_names_the_field():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    rejected = {
        "errors": [
            {
                "message": 'Cannot query field "metrics" on type "StandardListing".',
                "extensions": {"code": "GRAPHQL_VALIDATION_FAILED"},
            }
        ]
    }
    fake.queue("GetListing", httpx.Response(400, json=rejected))

    with pytest.raises(PlatformError, match='Cannot query field "metrics"'):
        client.listing("1744700001")


def test_requests_are_paced_with_jitter():
    fake = FakeKijiji()
    client, sleeps = make_client(fake)

    client.listing("1744700001")
    client.listing("1744700001")
    client.listing("1744700001")

    assert len(sleeps) == 2
    assert all(1.0 <= s <= 3.0 for s in sleeps)
    assert sleeps[0] != sleeps[1]


def test_one_request_at_a_time_across_threads():
    fake = FakeKijiji()
    in_flight = 0
    most = 0
    guard = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, most
        with guard:
            in_flight += 1
            most = max(most, in_flight)
        time.sleep(0.005)
        try:
            return fake.handle(request)
        finally:
            with guard:
                in_flight -= 1

    client, _ = make_client(fake)
    client._http._transport = httpx.MockTransport(handler)  # noqa: SLF001 - observe concurrency
    threads = [threading.Thread(target=client.conversations) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert most == 1
    assert fake.sign_ins == 1


def test_credentials_never_appear_in_errors():
    leaky = {"errors": [{"message": f"bad user {EMAIL} with {PASSWORD}"}], "data": None}
    failures = []
    for setup in (
        lambda f: f.queue("GetListing", httpx.Response(200, json=leaky)),
        lambda f: f.queue("GetListing", *[httpx.Response(500, text=PASSWORD)] * 3),
        lambda f: setattr(f, "login_outcome", "bad_password"),
        lambda f: setattr(f, "login_outcome", "challenge"),
    ):
        fake = FakeKijiji()
        setup(fake)
        client, _ = make_client(fake)
        with pytest.raises(PlatformError) as caught:
            client.listing("1744700001")
            client.conversations()
        failures.append(str(caught.value))

    assert len(failures) == 4
    for text in failures:
        assert EMAIL not in text and PASSWORD not in text


# Reads


def test_listing_parses_views_price_and_status():
    client, _ = make_client(FakeKijiji())

    listing = client.listing("1744700001")

    assert listing.views == 37
    assert listing.price_cents == 12000
    assert listing.status == "ACTIVE"
    assert listing.url and listing.url.endswith("/1744700001")


def test_missing_listing_is_listing_gone():
    client, _ = make_client(FakeKijiji())

    with pytest.raises(ListingGone):
        client.listing("1744700099")


@pytest.mark.parametrize("status", ["DELETED", "EXPIRED", "USER_DELETED", "SOLD"])
def test_listing_in_a_gone_state_is_listing_gone(status):
    fake = FakeKijiji()
    fake.listings["1744700001"]["status"] = status
    client, _ = make_client(fake)

    with pytest.raises(ListingGone):
        client.listing("1744700001")


def test_listing_pending_review_is_still_up():
    fake = FakeKijiji()
    fake.listings["1744700001"]["status"] = "PENDING"
    client, _ = make_client(fake)

    assert client.listing("1744700001").status == "PENDING"


def test_search_keeps_fixed_prices_in_cents():
    client, _ = make_client(FakeKijiji())

    hits = client.search("https://www.kijiji.ca/b-buy-sell/canada/desk/k0c10l0", 10)

    assert [h.price_cents for h in hits] == [9000, None, 4550, None]
    assert hits[0].posted_at == datetime(2026, 10, 9, 19, 27, 6, tzinfo=UTC)


def test_my_listings_reads_the_signed_in_profile():
    client, _ = make_client(FakeKijiji())

    ads = client.my_listings()

    assert [a.id for a in ads] == ["1744700001", "1744415555"]
    assert ads[0].views is None and ads[0].price_cents == 12000


def test_conversations_and_messages():
    client, _ = make_client(FakeKijiji())

    page = client.conversations()
    messages = client.conversation_messages("c-9001")

    first, second = page.items
    assert (first.id, first.listing_id) == ("c-9001", "1744700001")
    assert first.counterparty_id == "2000111"
    assert second.counterparty_id == "2000222"  # an integer id comes back as text
    assert first.last_message_at == datetime(2026, 10, 1, 17, 5, tzinfo=UTC)
    assert [m.id for m in messages] == ["m-1", "m-2", "m-3"]
    assert [m.inbound for m in messages] == [True, False, True]


def test_send_message_returns_the_message_id():
    fake = FakeKijiji()
    client, _ = make_client(fake)

    assert client.send_message("c-9001", "Yes, still available.") == "m-4"
    assert fake.sent == [("c-9001", "Yes, still available.")]


def test_category_and_location_lookups():
    client, _ = make_client(FakeKijiji())

    assert client.category_path(26) == ["Buy & Sell", "Other"]
    assert client.location_name(1700273) == "City of Toronto"
    assert client.location_name(5) is None
    with pytest.raises(PlatformError, match="category 999"):
        client.category_path(999)


# Legacy ad forms


def test_post_ad_fills_the_form_and_returns_the_new_id(tmp_path):
    fake = FakeKijiji()
    client, _ = make_client(fake)

    ad_id = client.post_ad(AD, _photos(tmp_path))

    assert ad_id == NEW_AD_ID
    assert len(fake.uploads) == 2
    (form,) = fake.submitted
    assert form["postAdForm.title"] == AD.title
    assert form["postAdForm.description"] == AD.description
    assert form["postAdForm.priceAmount"] == "120"
    assert form["postAdForm.priceType"] == "FIXED"
    assert form["postAdForm.attributeMap[condition_s]"] == "usedgood"
    assert form["postAdForm.attributeMap[forsaleby_s]"] == "ownr"
    assert form["postAdForm.postalCode"] == "M6J 1Z1"
    assert form["postAdForm.locationId"] == "1700273"
    assert form["postAdForm.confirmedTerms"] == "true"
    assert form["images"].count("media.kijiji.ca") == 2
    assert form["postAdForm.fraudToken"] == "fixture-fraud-token"  # hidden fields replayed
    assert "file" not in form


def _my_ads(*ids: str) -> httpx.Response:
    """My Ads holding only these of the fixture's ads."""
    body = fixture_json("graphql_profile_listings.json")
    ads = body["data"]["profile"]["listings"]
    body["data"]["profile"]["listings"] = [ad for ad in ads if ad["id"] in ids]
    return httpx.Response(200, json=body)


def test_post_ad_finds_the_id_in_my_ads_when_the_redirect_has_none(tmp_path):
    fake = FakeKijiji(submit_location="/m-my-ads/active/1")
    client, _ = make_client(fake)
    fake.queue("GetPosterOtherAds", _my_ads("1744415555"))  # before posting; then the fixture

    assert client.post_ad(AD, _photos(tmp_path, 1)) == "1744700001"


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(302, headers={"Location": "https://www.kijiji.ca/m-my-ads/active/1"}),
        httpx.Response(302, headers={"Location": "https://www.kijiji.ca/p-post-ad.html?e=dup"}),
        httpx.Response(200, html="<html><body>Something went wrong</body></html>"),
    ],
)
def test_post_ad_never_takes_an_older_ad_with_the_same_title(answer):
    """Two identical items, or one posted by hand earlier: My Ads already holds "IKEA Malm desk,
    white" (1744700001), and the submit brings no new id."""
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("/p-submit-ad.html", answer)

    with pytest.raises(PlatformError, match="not confirmed|not posted"):
        client.post_ad(AD, [])


def test_post_ad_sent_back_as_the_form_was_not_posted():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("/p-submit-ad.html", httpx.Response(200, html=fixture_text("post_ad_form.html")))

    with pytest.raises(PlatformError, match="sent the post form back"):
        client.post_ad(AD, [])


def test_post_ad_reads_the_id_from_the_kjrva_cookie():
    fake = FakeKijiji(submit_location="/m-my-ads/active/1", submit_cookie="1744800777")
    client, _ = make_client(fake)

    assert client.post_ad(AD, []) == "1744800777"


def test_post_ad_without_confirmation_is_an_error(tmp_path):
    fake = FakeKijiji(submit_location="/m-my-ads/active/1")
    client, _ = make_client(fake)
    other = KijijiAd(**{**AD.__dict__, "title": "Something never posted"})

    with pytest.raises(PlatformError, match="not confirmed"):
        client.post_ad(other, [])


def test_post_ad_on_a_changed_form_fails_before_submitting():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    client.login()
    fake.queue("/p-post-ad.html", httpx.Response(200, html="<form><input name='q'></form>"))

    with pytest.raises(PlatformError, match="no ad form"):
        client.post_ad(AD, [])

    assert fake.submitted == []


def test_post_ad_signs_in_again_when_the_page_bounces_to_login(tmp_path):
    fake = FakeKijiji(sso=True)
    client, _ = make_client(fake)
    client.login()
    fake.expire_session()

    assert client.post_ad(AD, _photos(tmp_path, 1)) == NEW_AD_ID
    assert fake.sign_ins == 2


def test_post_ad_is_not_retried_on_5xx():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    fake.queue("/p-submit-ad.html", httpx.Response(503))

    with pytest.raises(PlatformError, match="HTTP 503"):
        client.post_ad(AD, [])

    assert fake.calls("/p-submit-ad.html") == 1


def test_unreadable_photo_is_a_platform_error(tmp_path):
    client, _ = make_client(FakeKijiji())

    with pytest.raises(PlatformError, match="cannot read photo"):
        client.post_ad(AD, [tmp_path / "missing.jpg"])


def test_edit_ad_changes_fields_and_keeps_photos():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    cheaper = KijijiAd(**{**AD.__dict__, "price": "95", "title": "IKEA Malm desk"})

    client.edit_ad("1744700001", cheaper)

    (form,) = fake.submitted
    assert form["postAdForm.adId"] == "1744700001"
    assert form["postAdForm.priceAmount"] == "95"
    assert form["postAdForm.title"] == "IKEA Malm desk"
    assert form["images"].endswith("existing-photo")


def test_edit_ad_that_comes_back_with_the_form_is_an_error():
    fake = FakeKijiji()
    client, _ = make_client(fake)
    client.login()
    form_again = httpx.Response(200, html=fixture_text("edit_ad_form.html"))
    fake.queue("/p-submit-ad.html", form_again)

    with pytest.raises(PlatformError, match="not saved"):
        client.edit_ad("1744700001", AD)


def test_edit_or_delete_of_a_missing_ad_is_listing_gone():
    fake = FakeKijiji()
    client, _ = make_client(fake)

    with pytest.raises(ListingGone):
        client.edit_ad("1744700099", AD)
    with pytest.raises(ListingGone):
        client.delete_ad("1744700099")

    assert fake.submitted == [] and fake.deleted == []


def test_delete_ad_sends_the_reason_and_xsrf_token():
    fake = FakeKijiji()
    client, _ = make_client(fake)

    client.delete_ad("1744700001")

    assert fake.deleted == ["1744700001"]


def test_delete_without_ok_is_an_error():
    fake = FakeKijiji(delete_reply='{"status":"ERROR"}')
    client, _ = make_client(fake)

    with pytest.raises(PlatformError, match="did not confirm"):
        client.delete_ad("1744700001")


def test_post_form_fields_is_read_only():
    fake = FakeKijiji()
    client, _ = make_client(fake)

    fields = client.post_form_fields(26)

    assert all(fields.values())
    assert fake.submitted == [] and fake.uploads == []
