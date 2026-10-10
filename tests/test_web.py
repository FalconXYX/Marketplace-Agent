"""web: intake end to end, drafting, confirming, pages, photos, upload limits, and what no page
may ever show."""

import re
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError
from sqlmodel import select

from marketplace_agent.ai.base import Capability, ProviderError
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    AgentRun,
    Commitment,
    CommitmentKind,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Listing,
    Photo,
)
from marketplace_agent.platforms.base import PlatformError, SearchResult
from marketplace_agent.web import pages
from test_web_support import (
    ADDRESS,
    CONFIRM,
    accepting,
    build_world,
    has_gps,
    jpeg_with_gps,
)


STATIC = Path(pages.__file__).parent / "static"


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def _undo_url(html: str) -> str:
    found = re.search(r'action="(/commitments/\d+/undo)"', html)
    assert found, "no Undo button on the page"
    return found[1]


def test_photo_to_listing_to_deal_and_undo(tmp_path):
    world = build_world(tmp_path, accepting(11_000))
    photo = jpeg_with_gps()
    assert has_gps(photo), "fixture must carry GPS"

    item_id = world.upload("IKEA desk $120", [photo])
    page = world.client.get(f"/items/{item_id}").text
    assert f'hx-post="/items/{item_id}/draft"' in page and 'hx-trigger="load"' in page

    drafted = world.htmx(f"/items/{item_id}/draft")
    assert drafted.status_code == 200
    assert 'value="IKEA Linnmon desk"' in drafted.text
    assert 'name="price" value="120"' in drafted.text  # from the hint, over the suggestion
    assert "$90.00 to $110.00" in drafted.text and "exact dimensions" in drafted.text
    assert 'hx-trigger="load"' not in drafted.text
    with world.session() as session:
        [stored] = session.exec(select(Photo)).all()
    served = world.client.get(f"/photos/{stored.id}")
    assert served.headers["content-type"] == "image/jpeg" and not has_gps(served.content)

    confirmed = world.htmx(f"/items/{item_id}/confirm", CONFIRM)
    assert confirmed.status_code == 200, confirmed.text
    assert "Confirmed for sale." in confirmed.text
    assert "https://fake.invalid/listings/fake-l1" in confirmed.text
    [posted] = world.market.listings.values()
    assert posted.payload.title == "IKEA Linnmon desk, white"
    assert posted.payload.price_cents == 12_000
    assert posted.payload.attributes == {"Brand": "IKEA", "Colour": "white"}
    with world.session() as session:
        item = session.get(Item, item_id)
        goal = session.exec(select(Goal)).one()
    assert item.status == ItemStatus.LISTED and item.condition == "good"
    assert (goal.floor_price_cents, goal.target_price_cents) == (9_500, 11_000)
    assert goal.firmness == Firmness.SLIGHT and goal.notes == "No holds."

    world.buyer_writes(item_id, "Would you take $110? I can pick it up tonight.")
    page = world.client.get(f"/items/{item_id}").text
    assert "Would you take $110? I can pick it up tonight." in page
    assert "Promised to" in page and "Undo deal (until" in page

    undone = world.htmx(_undo_url(page))
    assert undone.status_code == 200
    assert "Undid the $110.00 deal" in undone.text
    assert "Undo deal" not in undone.text
    with world.session() as session:
        kinds = [c.kind for c in session.exec(select(Commitment).order_by(Commitment.id))]
        assert session.get(Item, item_id).status == ItemStatus.LISTED
    assert kinds == [CommitmentKind.ACCEPTED, CommitmentKind.UNDONE]


# ---------------------------------------------------------------- drafting


def test_without_a_vision_model_the_form_says_how_to_get_one(tmp_path):
    world = build_world(tmp_path, capabilities=frozenset({Capability.REASONING}))
    item_id = world.upload("Desk $80")

    response = world.htmx(f"/items/{item_id}/draft")

    assert response.status_code == 200
    assert "Set GEMINI_API_KEY" in response.text
    assert "fill it in yourself" in response.text
    assert 'name="price" value="80"' in response.text and 'name="title" value=""' in response.text
    assert 'hx-trigger="load"' not in response.text and "Try the draft again" not in response.text


def test_a_failed_draft_offers_to_try_again_and_keeps_the_form(tmp_path):
    def broken(request):
        raise ProviderError("model unavailable")

    world = build_world(tmp_path, broken)
    item_id = world.upload("Desk")

    response = world.htmx(f"/items/{item_id}/draft")

    assert response.status_code == 200
    assert "The draft did not work: model unavailable" in response.text
    assert "Try the draft again" in response.text and 'action="/items/1/confirm"' in response.text


def _locked(*_args, **_kwargs):
    """What SQLite raises when `agent run` holds the write lock past busy_timeout, as during
    a slow platform push."""
    raise OperationalError("UPDATE item", {}, sqlite3.OperationalError("database is locked"))


def test_a_busy_database_while_drafting_offers_to_try_again(world, monkeypatch):
    """A 500 is not swapped in by htmx: the drafting note would pulse forever."""
    item_id = world.upload()
    monkeypatch.setattr(pages, "draft_item", _locked)

    response = world.htmx(f"/items/{item_id}/draft")

    assert response.status_code == 200
    assert "The agent is busy writing to the database" in response.text
    assert "Try the draft again" in response.text and 'action="/items/1/confirm"' in response.text


def test_a_busy_database_on_confirm_keeps_what_was_typed(world, monkeypatch):
    item_id = world.upload()
    monkeypatch.setattr(pages, "_save_draft", _locked)

    response = world.htmx(f"/items/{item_id}/confirm", {**CONFIRM, "title": "My own title"})

    assert response.status_code == 409  # swapped in, unlike a 5xx
    assert "busy writing to the database" in response.text
    assert "Nothing was saved yet" in response.text
    assert 'value="My own title"' in response.text and world.market.listings == {}


def test_a_busy_database_on_upload_asks_to_try_again(world, monkeypatch):
    monkeypatch.setattr(pages, "_store_item", _locked)
    files = [("photos", ("p.jpg", jpeg_with_gps(), "image/jpeg"))]

    response = world.client.post("/items", data={"hint": "Desk $80"}, files=files)

    assert response.status_code == 409 and "busy writing to the database" in response.text
    assert 'value="Desk $80"' in response.text


def test_a_server_error_is_shown_instead_of_dropped():
    """htmx drops a 5xx response; app.js puts a note in the toast area instead."""
    script = (STATIC / "app.js").read_text()
    assert 'addEventListener("htmx:responseError"' in script
    assert 'getElementById("toast")' in script


def test_without_htmx_the_draft_answers_with_the_whole_page(world):
    item_id = world.upload()

    response = world.client.post(f"/items/{item_id}/draft")

    assert response.status_code == 200 and response.text.startswith("<!doctype html>")
    assert 'value="IKEA Linnmon desk"' in response.text


def test_without_a_hint_price_the_suggestion_prefills_the_price(world):
    item_id = world.upload("IKEA desk")

    response = world.htmx(f"/items/{item_id}/draft")

    assert 'name="price" value="110"' in response.text


def test_the_draft_form_shows_comparable_listings_beside_the_price(world):
    world.market.search_catalog = [
        SearchResult("IKEA Linnmon desk, white", 8_000, url="https://fake.test/a"),
        SearchResult("IKEA Linnmon desk", 10_000, url="javascript:alert(1)"),
        SearchResult("Linnmon desk <b>IKEA</b>", 12_000, url="https://fake.test/c"),
    ]
    item_id = world.upload()

    form = world.htmx(f"/items/{item_id}/draft").text

    shown = form.index("Similar listings:")
    assert form.index('name="floor"') < shown < form.index('name="description"')
    assert '3 for "ikea linnmon desk", median $100.00, middle half $90.00 to $110.00' in form
    assert 'href="https://fake.test/a"' in form and "javascript:" not in form
    assert "&lt;b&gt;IKEA&lt;/b&gt;" in form and "<b>IKEA" not in form
    world.market.search_catalog = []
    page = world.client.get(f"/items/{item_id}").text
    assert "median $100.00" in page  # from the cache, not a new search


def test_the_draft_form_says_when_a_search_found_nothing(world):
    item_id = world.upload()

    form = world.htmx(f"/items/{item_id}/draft").text

    assert 'none found for "ikea linnmon desk"' in form


def test_a_failing_search_still_drafts_and_shows_no_comps(world, monkeypatch):
    def down(query, limit=20):
        raise PlatformError("search is down")

    monkeypatch.setattr(world.market, "search", down)
    item_id = world.upload()

    form = world.htmx(f"/items/{item_id}/draft").text

    assert 'value="IKEA Linnmon desk"' in form and "Similar listings" not in form


def test_a_listed_item_is_not_drafted_again(world):
    item_id = world.listed()

    response = world.htmx(f"/items/{item_id}/draft")

    assert response.status_code == 409 and response.headers["HX-Retarget"] == "#toast"
    with world.session() as session:
        assert session.get(Item, item_id).title == CONFIRM["title"]


def test_a_goal_set_before_drafting_fills_the_form(world):
    item_id = world.upload("Desk $80")
    with world.session() as session:
        set_goal(
            session,
            world.clock,
            item_id,
            list_price_cents=9_000,
            floor_price_cents=7_000,
            target_price_cents=8_500,
            deadline=world.clock.now() + timedelta(days=9, hours=1),
            firmness=Firmness.FIRM,
            notes="Weekends only.",
        )
        session.commit()

    form = world.htmx(f"/items/{item_id}/draft").text

    for name, value in (("price", "90"), ("floor", "70"), ("target", "85"), ("deadline_days", "10")):
        assert f'name="{name}" value="{value}"' in form
    assert '<option value="firm" selected>' in form and "Weekends only." in form


# ---------------------------------------------------------------- confirming


def test_an_invalid_form_comes_back_marked_with_what_was_typed(world):
    item_id = world.upload()
    form = {**CONFIRM, "price": "1oo", "floor": "", "deadline_days": "0", "platforms": []}

    response = world.htmx(f"/items/{item_id}/confirm", form)

    assert response.status_code == 422
    text = response.text
    assert "Price: enter dollars" in text and "Floor: enter dollars" in text
    assert "Days to sell" in text and "Pick at least one platform." in text
    assert 'value="1oo"' in text and 'value="IKEA Linnmon desk, white"' in text
    assert "<details class=\"more\" open>" in text  # the days field is in there
    assert world.market.listings == {}
    with world.session() as session:
        assert session.get(Item, item_id).status == ItemStatus.DRAFT
        assert session.exec(select(Goal)).first() is None


def test_a_floor_above_the_price_is_refused(world):
    item_id = world.upload()

    response = world.htmx(f"/items/{item_id}/confirm", {**CONFIRM, "floor": "130"})

    assert response.status_code == 422
    assert "Floor price cannot be above the list price." in response.text
    assert world.market.listings == {}


def test_only_enabled_platforms_are_offered_and_used(world):
    item_id = world.upload()

    form = world.htmx(f"/items/{item_id}/draft").text
    response = world.htmx(f"/items/{item_id}/confirm", {**CONFIRM, "platforms": ["kijiji"]})

    assert 'name="platforms" value="fake"' in form and "kijiji" not in form
    assert response.status_code == 422 and "Pick at least one platform." in response.text


def test_facebook_without_a_condition_is_sent_back_before_anything_is_posted(tmp_path):
    """Facebook refuses a post without a condition, and once confirmed this form is gone."""
    world = build_world(tmp_path, enabled_platforms=["fake", "facebook"])
    item_id = world.upload()
    form = {**CONFIRM, "condition": "", "platforms": ["fake", "facebook"]}

    response = world.htmx(f"/items/{item_id}/confirm", form)

    assert response.status_code == 422
    assert "Facebook needs a condition. Pick one, or untick it." in response.text
    assert world.market.listings == {}
    with world.session() as session:
        assert session.get(Item, item_id).status == ItemStatus.DRAFT
    fake_only = world.htmx(f"/items/{item_id}/confirm", {**form, "platforms": ["fake"]})
    assert fake_only.status_code == 200  # "Not sure" is fine where no condition is needed


def test_the_condition_can_be_set_once_listed(tmp_path):
    world = build_world(tmp_path, enabled_platforms=["fake", "facebook"])
    item_id = world.listed(condition="")
    page = world.client.get(f"/items/{item_id}").text
    assert f'action="/items/{item_id}/condition"' in page

    response = world.htmx(f"/items/{item_id}/condition", {"condition": "like_new"})

    assert response.status_code == 200 and "Condition set to like new." in response.text
    with world.session() as session:
        assert session.get(Item, item_id).condition == "like_new"
    bad = world.htmx(f"/items/{item_id}/condition", {"condition": "mint"})
    assert bad.status_code == 422 and "Pick a condition from the list." in bad.text


def test_confirming_twice_does_not_post_twice(world):
    item_id = world.listed()

    response = world.htmx(f"/items/{item_id}/confirm", CONFIRM)

    assert response.status_code == 409 and "only a draft is edited here" in response.text
    assert len(world.market.listings) == 1


def test_a_failed_post_is_shown_per_platform(world):
    world.market.fail_next["publish"] = PlatformError("fake is down")
    item_id = world.upload()

    response = world.htmx(f"/items/{item_id}/confirm", CONFIRM)

    assert response.status_code == 409
    assert "nothing was posted" in response.text and "fake is down" in response.text


# ---------------------------------------------------------------- pages


def test_the_home_page_groups_items_and_flags_what_needs_the_owner(world):
    listed = world.listed()
    world.upload("Lamp $20")
    now = world.clock.now()
    with world.session() as session:
        session.add(Escalation(item_id=listed, reason="Buyer asks for delivery.", created_at=now))
        session.add(Escalation(reason="Froze a chat with no listing.", created_at=now))
        session.commit()

    page = world.client.get("/").text

    assert page.index("For sale") < page.index("Drafts")
    assert "IKEA Linnmon desk, white" in page and "$120.00" in page and "Lamp $20" in page
    assert "1 needs you" in page
    assert 'href="https://fake.invalid/listings/fake-l1"' in page
    assert "Needs you" in page and "Froze a chat with no listing." in page
    assert 'href="/items/new"' in page


def test_a_platform_url_that_is_not_http_is_never_a_link(world):
    item_id = world.listed()
    with world.session() as session:
        listing = session.exec(select(Listing)).one()
        listing.url = "javascript:alert(1)"
        session.add(listing)
        session.commit()

    for page in (world.client.get("/").text, world.client.get(f"/items/{item_id}").text):
        assert "javascript:" not in page


def test_the_new_item_form_opens_the_camera_and_takes_many_photos(world):
    page = world.client.get("/items/new").text

    assert 'enctype="multipart/form-data"' in page
    assert 'accept="image/*" capture="environment" multiple' in page
    assert 'name="hint"' in page and 'placeholder="IKEA desk $120"' in page


def test_the_agent_runs_and_waiting_items_show_on_the_item_page(world):
    item_id = world.listed()
    with world.session() as session:
        session.add(
            AgentRun(
                item_id=item_id,
                started_at=world.clock.now(),
                triggers_json='[{"kind": "idle", "detail": "3 days quiet"}]',
                rationale="Traffic is fine but nobody writes, so the price is the problem.",
                results_json='[{"tool": "update_price", "status": "executed", "detail": "$110"}]',
            )
        )
        session.commit()

    page = world.client.get(f"/items/{item_id}").text

    assert "idle (3 days quiet)" in page and "so the price is the problem." in page
    assert "update_price executed: $110" in page


def test_unknown_pages_and_ids_are_plain_html_errors(world):
    assert world.client.get("/items/99").status_code == 404
    assert "There is no item with id 99." in world.client.get("/items/99").text
    assert world.client.get("/nope").status_code == 404
    assert world.client.get("/items/abc").status_code == 422
    assert world.client.get("/items/1/confirm").status_code == 405


def test_pages_send_security_headers_and_serve_htmx_locally(world):
    response = world.client.get("/")

    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert '<script src="/static/htmx.min.js"' in response.text and "unpkg" not in response.text
    htmx = world.client.get("/static/htmx.min.js")
    assert htmx.status_code == 200 and 'version:"2.0.11"' in htmx.text


# ---------------------------------------------------------------- upload limits


def _items(world) -> list[Item]:
    with world.session() as session:
        return list(session.exec(select(Item)).all())


def _post_photos(world, photos, hint="Desk $50"):
    files = [("photos", (name, data, "image/jpeg")) for name, data in photos]
    return world.client.post("/items", data={"hint": hint}, files=files, follow_redirects=False)


def test_too_many_photos_are_refused_and_nothing_is_saved(tmp_path):
    world = build_world(tmp_path, web_max_photos=2)
    photos = [(f"{i}.jpg", jpeg_with_gps(color=(i, 0, 0))) for i in range(3)]

    response = _post_photos(world, photos)

    assert response.status_code == 413
    assert "You picked 3 photos. Up to 2 photos" in response.text
    assert 'value="Desk $50"' in response.text  # the hint survives
    assert _items(world) == []


def test_a_photo_over_the_size_limit_is_refused(tmp_path):
    world = build_world(tmp_path, web_max_photo_mb=1)
    big = jpeg_with_gps() + b"\0" * (1024 * 1024)

    response = _post_photos(world, [("big.jpg", big)])

    assert response.status_code == 413 and "big.jpg is too big" in response.text
    assert _items(world) == []


def test_an_upload_bigger_than_every_photo_allowed_is_refused_before_reading(tmp_path):
    world = build_world(tmp_path, web_max_photos=1, web_max_photo_mb=1)
    huge = b"\0" * (3 * 1024 * 1024)

    response = _post_photos(world, [("huge.jpg", huge)])

    assert response.status_code == 413 and "too much to upload at once" in response.text
    assert _items(world) == []


def test_a_file_that_is_not_a_photo_saves_nothing(world):
    response = _post_photos(world, [("a.jpg", jpeg_with_gps()), ("notes.txt", b"not an image")])

    assert response.status_code == 422 and "not a supported image" in response.text
    assert _items(world) == []
    assert not any(world.runtime.settings.photos_dir.rglob("*.jpg"))


def test_an_item_needs_a_photo(world):
    response = _post_photos(world, [])

    assert response.status_code == 422 and "Add at least one photo." in response.text
    assert _items(world) == []


def test_an_empty_picker_is_ignored(world):
    response = _post_photos(world, [("desk.jpg", jpeg_with_gps()), ("", b"")])

    assert response.status_code == 303
    with world.session() as session:
        assert len(session.exec(select(Photo)).all()) == 1


# ---------------------------------------------------------------- photos


def test_a_photo_is_served_by_id_only_from_the_photos_dir(world, tmp_path):
    item_id = world.upload()
    secret = tmp_path / "data" / "secret.jpg"
    secret.write_bytes(b"SECRET")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"SECRET")
    link = world.runtime.settings.photos_dir / "link.jpg"
    link.symlink_to(outside)
    paths = ["secret.jpg", "photos/../secret.jpg", "../outside.jpg", str(outside)]
    paths.append("photos/link.jpg")
    with world.session() as session:
        rows = [Photo(item_id=item_id, path=path, width=1, height=1, sha256=path) for path in paths]
        session.add_all(rows)
        session.commit()
        ids = [row.id for row in rows]

    for photo_id in ids:
        response = world.client.get(f"/photos/{photo_id}")
        assert response.status_code == 404 and b"SECRET" not in response.content
    assert world.client.get("/photos/1").status_code == 200
    assert world.client.get("/photos/999").status_code == 404
    assert world.client.get("/photos/..%2Fsecret.jpg").status_code in (404, 422)
    assert world.client.get("/photos/0").status_code == 422


# ---------------------------------------------------------------- what pages never show


def test_a_buyer_message_is_text_never_markup(world):
    item_id = world.listed()
    attack = '<script>alert("x")</script><img src=x onerror="alert(1)"> & "quotes"'

    world.buyer_writes(item_id, attack)
    page = world.client.get(f"/items/{item_id}").text

    assert "<script>alert" not in page and "<img src=x" not in page
    assert "&lt;script&gt;alert(&#34;x&#34;)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=&#34;alert(1)&#34;&gt; &amp; &#34;quotes&#34;" in page


def test_listing_text_that_breaks_a_rule_is_sent_back_before_anything_is_posted(world):
    item_id = world.upload()

    response = world.htmx(
        f"/items/{item_id}/confirm", {**CONFIRM, "description": "Call 416-555-0199 or come by."}
    )

    assert response.status_code == 422
    assert "contains a phone number" in response.text
    assert world.market.listings == {}
    with world.session() as session:
        assert session.get(Item, item_id).status == ItemStatus.DRAFT


def test_no_page_shows_the_pickup_address(world):
    item_id = world.upload(f"Desk, pick up at {ADDRESS} $120")
    pages = [world.client.get("/").text, world.client.get(f"/items/{item_id}").text]
    pages.append(world.htmx(f"/items/{item_id}/draft").text)
    refused = world.htmx(
        f"/items/{item_id}/confirm", {**CONFIRM, "title": f"Desk at {ADDRESS}", "notes": ADDRESS}
    )
    assert refused.status_code == 422 and "must never contain the pickup address" in refused.text
    pages.append(refused.text)
    world.htmx(f"/items/{item_id}/confirm", {**CONFIRM, "notes": f"Buyers come to {ADDRESS}"})
    thread = world.buyer_writes(item_id, f"Is it still at {ADDRESS}?")
    with world.session() as session:
        now = world.clock.now()
        reason = f"Buyer named {ADDRESS}."
        session.add(Escalation(item_id=item_id, thread_id=thread.id, reason=reason, created_at=now))
        session.add(AgentRun(item_id=item_id, started_at=now, rationale=f"Meet at {ADDRESS}."))
        session.commit()
    pages += [world.client.get("/").text, world.client.get(f"/items/{item_id}").text]
    reply = f"/threads/{thread.id}/reply"
    pages.append(world.htmx(reply, {"body": f"Come to {ADDRESS}"}).text)
    pages.append(world.client.post(reply, data={"body": f"I am at {ADDRESS}"}).text)

    assert "[PICKUP_ADDRESS]" in pages[1]  # the hint, redacted
    for page in pages:
        assert "123 example" not in page.lower()


# ---------------------------------------------------------------- phone layout

LONG_LINK = (
    "https://www.paypal.com/paypalme/somebody-else/110?currency=CAD&note=deposit-for-the-desk"
    "-before-pickup-tomorrow-morning"
)


def _phone_width(html: str) -> int:
    """The page's scroll width in Chromium at a 375 px phone viewport, with the real CSS."""
    sync_api = pytest.importorskip("playwright.sync_api")
    css = (STATIC / "app.css").read_text()
    html = html.replace('<link rel="stylesheet" href="/static/app.css">', f"<style>{css}</style>")
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chromium")
        except sync_api.Error as exc:
            pytest.skip(f"Chromium is not available: {exc}")
        try:
            page = browser.new_page(viewport={"width": 375, "height": 800}, is_mobile=True)
            page.route("**/*", lambda route: route.abort())
            page.set_content(html)
            return int(page.evaluate("document.documentElement.scrollWidth"))
        finally:
            browser.close()


def test_a_long_link_in_a_reason_never_makes_the_phone_page_scroll_sideways(world):
    item_id = world.listed()
    with world.session() as session:
        now = world.clock.now()
        session.add(Escalation(item_id=item_id, reason=f"Buyer wants {LONG_LINK}", created_at=now))
        session.add(Escalation(item_id=None, reason=f"Froze a chat: {LONG_LINK}", created_at=now))
        session.add(AgentRun(item_id=item_id, started_at=now, rationale=f"Saw {LONG_LINK}"))
        session.commit()

    for url in (f"/items/{item_id}", "/"):
        page = world.client.get(url).text
        assert LONG_LINK.replace("&", "&amp;") in page
        assert _phone_width(page) <= 375, url
