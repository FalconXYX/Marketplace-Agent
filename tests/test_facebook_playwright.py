"""PlaywrightDriver and the default selectors in real Chromium, against local HTML fixtures.

Every facebook.com request is answered from tests/fixtures/facebook; anything else is aborted,
so nothing leaves the machine. Skipped when Playwright or its Chromium build is missing. The
fixtures follow what open source automation reports about Facebook's pages; passing here proves
the plumbing and the locator strategy, not that facebook.com still looks like this.
"""

from pathlib import Path
from urllib.parse import urlsplit

import pytest

from marketplace_agent.platforms.base import (
    AwaitingHuman,
    ListingGone,
    ListingPayload,
    PlatformAuthError,
    PlatformRef,
)
from marketplace_agent.platforms.facebook import flow_notes as fb
from marketplace_agent.platforms.facebook.adapter import FacebookAdapter
from marketplace_agent.platforms.facebook.driver import (
    DriverError,
    PacedDriver,
    PlaywrightDriver,
    ProfileInUse,
)
from marketplace_agent.platforms.facebook.pacing import Pacer

FIXTURES = Path(__file__).parent / "fixtures" / "facebook"
PAGES = {
    "/marketplace/create/item": "create_item.html",
    "/marketplace/you/selling": "selling.html",
    "/messages/t/": "inbox.html",
    "/messages/t/777/": "thread.html",
}


def _serve(route) -> None:
    url = route.request.url
    page = PAGES.get(urlsplit(url).path) if url.startswith(fb.BASE_URL) else None
    if page is None:
        route.fulfill(status=404, body="<html><body>This page isn't available</body></html>")
        return
    route.fulfill(status=200, content_type="text/html", body=(FIXTURES / page).read_text())


def _offline_site(context) -> None:
    context.route("**/*", lambda route: route.abort())
    context.route(f"{fb.BASE_URL}/**", _serve)
    context.add_cookies([{"name": fb.SESSION_COOKIE, "value": "1", "url": fb.BASE_URL}])


@pytest.fixture(scope="module")
def browser(tmp_path_factory):
    pytest.importorskip("playwright.sync_api")
    pacer = Pacer((0.0, 0.0), key_delay_ms=(0, 5))
    driver = PlaywrightDriver(
        tmp_path_factory.mktemp("profile"),
        headless=True,
        pacer=pacer,
        timeout_s=5,
        on_context=_offline_site,
    )
    try:
        driver.goto(fb.CREATE_URL)
    except DriverError as exc:
        driver.close()
        pytest.skip(f"Chromium is not available: {exc}")
    yield PacedDriver(driver, pacer)
    driver.close()


@pytest.fixture
def adapter(settings, clock, browser) -> FacebookAdapter:
    browser.release_page()  # a fresh tab per test, as after assist mode
    return FacebookAdapter(settings, clock, browser, None, find_timeout_s=3.0)


@pytest.fixture
def payload(tmp_path) -> ListingPayload:
    photo = tmp_path / "desk.jpg"
    photo.write_bytes(b"\xff\xd8\xff\xd9")
    return ListingPayload(
        title="IKEA desk",
        description="Solid desk.\nWhite.",
        price_cents=12050,
        photo_paths=[photo],
        location="Bloor St W & Ossington Ave",
        category="Furniture",
        condition="like_new",
    )


def test_default_selectors_match_the_fixture_pages(adapter):
    assert adapter.validate_selectors() == []
    assert adapter.healthcheck() == []


def test_publish_end_to_end(adapter, browser, payload):
    ref = adapter.publish(payload)
    assert ref.remote_id == "2222"
    card = browser.find({"text": "IKEA desk"})
    assert card and "$121" in browser.text_of(browser.find({"css": ".price"})[0])


def test_assist_leaves_a_filled_form(adapter, browser, payload):
    with pytest.raises(AwaitingHuman):
        adapter.publish(payload, assist=True)
    browser.goto(fb.CREATE_URL)  # a new tab: the filled one stays with the human
    assert browser.find({"css": "#title"})[0] is not None


def test_fill_select_and_read_back(browser):
    browser.goto(fb.CREATE_URL)
    title = browser.find({"label": "Title", "exact": True})[0]
    browser.fill(title, "Lamp")
    browser.fill(title, "Desk lamp")  # replaces, never appends
    assert 'textbox "Title": Desk lamp' in browser.aria_snapshot()

    description = browser.find({"css": "#description"})[0]
    browser.fill(description, "Line one\nLine two")
    category = browser.find({"role": "combobox", "name": "Category"})[0]
    browser.select_option(category, "Bicycles")  # plain text entries, scrolled into view
    assert browser.text_of(browser.find({"css": "#category-value"})[0]) == "Bicycles"
    condition = browser.find({"role": "combobox", "name": "Condition"})[0]
    browser.select_option(condition, "New", exact=True)
    assert browser.text_of(browser.find({"css": "#condition-value"})[0]) == "New"
    with pytest.raises(DriverError, match="No option"):
        browser.select_option(category, "Spaceships")

    assert browser.find({"css": "input[type='file']"}) == []  # hidden
    assert len(browser.find({"css": "input[type='file']", "attached": True})) == 1
    assert browser.screenshot().startswith(b"\x89PNG")
    assert browser.wait_for([{"css": "#nothing"}], 0.2) is False


def test_inbox_and_reply_round_trip(adapter):
    [thread] = adapter.fetch_threads(None)  # the personal chat is never opened
    assert (thread.remote_id, thread.listing_remote_id, thread.buyer_handle, thread.buyer_name) == (
        "777",
        "2222",
        "555",
        "Lucas",
    )
    assert [(m.body, m.from_buyer) for m in thread.messages] == [
        ("Is this available?", True),
        ("Yes it is", False),
        ("Would you take $100?", True),
    ]
    sent = adapter.send_message("777", "Yes, $110 works.\nTonight?")
    assert sent is not None and sent.startswith("777-")


def test_a_missing_listing_is_gone(adapter):
    with pytest.raises(ListingGone):
        adapter.update(PlatformRef("facebook", "4040"), ListingPayload("x", "y", 100, [], "z"))


def _driver(profile: Path) -> PlaywrightDriver:
    pacer = Pacer((0.0, 0.0), key_delay_ms=(0, 0))
    return PlaywrightDriver(
        profile, headless=True, pacer=pacer, timeout_s=5, on_context=_offline_site
    )


def test_a_closed_window_is_reopened_on_the_next_call(browser, tmp_path):
    driver = _driver(tmp_path / "profile")
    try:
        driver.goto(fb.CREATE_URL)
        driver._run(lambda: driver._context.close())  # the human closed the window
        with pytest.raises(DriverError, match="window was closed"):
            driver.goto(fb.CREATE_URL)
        driver.goto(fb.CREATE_URL)
        assert driver.find({"label": "Title", "exact": True})
    finally:
        driver.close()


def test_a_profile_in_use_says_which_process_to_use(browser, tmp_path):
    first, second = _driver(tmp_path / "profile"), _driver(tmp_path / "profile")
    try:
        first.goto(fb.CREATE_URL)
        with pytest.raises(ProfileInUse, match="in use by another process"):
            second.goto(fb.CREATE_URL)
    finally:
        second.close()
        first.close()


def test_a_profile_in_use_needs_the_owner_and_says_what_to_stop(tmp_path):
    """No browser needed: Playwright's launch error becomes ProfileInUse, which the runner
    alerts on as it does a sign-out, naming the processes that may hold the profile."""
    error = type("Error", (Exception,), {"__module__": "playwright._impl._errors"})
    driver = PlaywrightDriver(tmp_path, headless=True, pacer=Pacer((0.0, 0.0)))

    found = driver._translate(error("Failed to create a ProcessSingleton for your profile"))

    assert isinstance(found, ProfileInUse) and isinstance(found, PlatformAuthError)
    assert "`agent serve`, `agent run`, `agent web`, or `agent facebook login`" in str(found)
    assert "Stop that one (Ctrl+C)" in str(found)
