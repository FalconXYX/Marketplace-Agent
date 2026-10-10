"""FacebookAdapter flows against FakeDriver, a state machine standing in for facebook.com."""

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fixtures.facebook.fake_driver import FakeDriver, FakeSite
from PIL import Image
from sqlmodel import select

from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.control.service import ControlService
from marketplace_agent.core.comps import searching
from marketplace_agent.core.items import create_item, set_goal
from marketplace_agent.core.models import (
    Direction,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    Thread,
)
from marketplace_agent.intake.photos import ingest_photos
from marketplace_agent.platforms.base import (
    AwaitingHuman,
    ListingGone,
    ListingPayload,
    ListingStats,
    PlatformAdapter,
    PlatformAuthError,
    PlatformError,
    PlatformRef,
)
from marketplace_agent.platforms.facebook import flow_notes as fb
from marketplace_agent.platforms.facebook.adapter import (
    INBOX_INTERVAL,
    STATS_TTL,
    THREADS_PER_POLL,
    FacebookAdapter,
    category_option,
    condition_option,
    parse_bubbles,
    parse_sent,
    row_key,
    signed_in,
    split_heading,
    whole_dollars,
)
from marketplace_agent.platforms.facebook.driver import DriverError
from marketplace_agent.runner.ingest import ingest
from marketplace_agent.tools.registry import default_registry


@pytest.fixture
def site() -> FakeSite:
    return FakeSite()


@pytest.fixture
def driver(site) -> FakeDriver:
    return FakeDriver(site)


@pytest.fixture
def adapter(settings, clock, driver) -> FacebookAdapter:
    return FacebookAdapter(settings, clock, driver, None, find_timeout_s=1.0)


@pytest.fixture
def photos(tmp_path) -> list[Path]:
    paths = [tmp_path / f"p{i}.jpg" for i in range(12)]
    for path in paths:
        path.write_bytes(b"jpeg")
    return paths


def _payload(photos: list[Path], **changes: object) -> ListingPayload:
    fields: dict = dict(
        title="IKEA Malm desk",
        description="Solid desk.\nWhite, light scratches.",
        price_cents=12050,
        photo_paths=photos,
        location="Bloor St W & Ossington Ave",
        category="Furniture",
        condition="good",
    )
    fields.update(changes)
    return ListingPayload(**fields)


def _ref(listing_id: str) -> PlatformRef:
    return PlatformRef("facebook", listing_id, fb.ITEM_URL.format(listing_id=listing_id))


def test_adapter_satisfies_the_platform_protocol(adapter):
    platform: PlatformAdapter = adapter
    assert platform.name == "facebook"
    assert platform.search("desk") == []


def test_comps_never_asks_facebook(adapter):
    """Its search is a stub, so an empty answer must not read as "none for sale"."""
    assert adapter.searches is False
    assert "facebook" not in searching({"facebook": adapter})


# ---------------------------------------------------------------- publish


def test_publish_fills_the_form_posts_and_returns_the_new_id(adapter, driver, site, photos):
    ref = adapter.publish(_payload(photos))

    listing = site.listings[ref.remote_id]
    assert (listing.title, listing.price, listing.category, listing.condition) == (
        "IKEA Malm desk",
        "121",  # $120.50 rounds up, never below the floor
        "Furniture",
        "Used - Good",
    )
    assert listing.description == "Solid desk.\nWhite, light scratches."
    assert len(listing.photos) == fb.MAX_PHOTOS
    assert ref == PlatformRef(
        "facebook", listing.listing_id, fb.ITEM_URL.format(listing_id=listing.listing_id)
    )
    assert driver.kinds().index("upload") < driver.actions.index(("click", "next"))
    assert driver.actions.index(("click", "next")) < driver.actions.index(("click", "publish"))


def test_publish_dismisses_a_boost_prompt(adapter, driver, site, photos):
    site.boost_prompt = True
    ref = adapter.publish(_payload(photos))
    assert ("click", "not_now") in driver.actions
    assert ref.remote_id in site.listings


def test_assist_fills_everything_then_leaves_the_tab_for_the_human(adapter, driver, site, photos):
    with pytest.raises(AwaitingHuman, match="Next and Publish"):
        adapter.publish(_payload(photos), assist=True)

    assert not site.listings
    assert ("click", "next") not in driver.actions
    filled = {action[1] for action in driver.actions if action[0] in ("fill", "select", "upload")}
    assert filled == {"photos", "title", "price", "description", "category", "condition"}
    assert driver.released == [fb.CREATE_URL]


def test_assist_refuses_a_headless_browser(settings, clock, driver, photos):
    settings.facebook_headless = True
    adapter = FacebookAdapter(settings, clock, driver, None)
    with pytest.raises(PlatformError, match="visible browser"):
        adapter.publish(_payload(photos), assist=True)
    assert driver.actions == []


def test_publish_needs_a_condition_before_touching_the_browser(adapter, driver, photos):
    with pytest.raises(PlatformError, match="condition"):
        adapter.publish(_payload(photos, condition=None))
    assert driver.actions == []


def test_publish_falls_back_to_miscellaneous_when_the_category_is_not_offered(
    adapter, driver, site, photos
):
    ref = adapter.publish(_payload(photos, category="Jewelry", title="Silver ring"))
    assert site.listings[ref.remote_id].category == "Miscellaneous"
    assert ("select", "category", "Jewelry") in driver.actions


def test_publish_stops_when_facebook_keeps_next_disabled(adapter, driver, site, photos):
    driver.upload = lambda handle, paths: None  # Facebook rejected the photos
    with pytest.raises(PlatformError, match=r"create\.next disabled"):
        adapter.publish(_payload(photos))
    assert not site.listings


def test_republishing_the_same_title_returns_the_new_listing(adapter, photos):
    first = adapter.publish(_payload(photos))
    second = adapter.publish(_payload(photos))
    assert second.remote_id != first.remote_id


def test_publish_without_a_card_afterwards_says_to_check_before_retrying(adapter, driver, photos):
    driver._publish = lambda: setattr(driver, "url", fb.SELLING_URL)  # posted nothing
    with pytest.raises(PlatformError, match="before publishing again"):
        adapter.publish(_payload(photos))


@pytest.mark.parametrize(
    ("title", "status"),
    [
        ("IKEA Malm desk chair", "active"),  # holds the new title as a substring
        ("IKEA Malm desk", "sold"),  # a relist's predecessor, after a restart
    ],
)
def test_publish_never_takes_an_older_listing_for_the_new_one(
    adapter, driver, site, photos, title, status
):
    """The new card is missing (a slow list); another card must not stand in for it."""
    site.add_listing(title, status=status)
    driver._publish = lambda: setattr(driver, "url", fb.SELLING_URL)  # posted nothing

    with pytest.raises(PlatformError, match="before publishing again"):
        adapter.publish(_payload(photos))


def test_publish_finds_a_card_whose_title_is_cut_short(adapter, driver, site, photos):
    real_page = driver._selling_page

    def cut_titles():
        cards = real_page()
        for card in cards:
            card.text = card.text.replace("IKEA Malm desk with drawers", "IKEA Malm desk wi…")
        return cards

    driver._selling_page = cut_titles
    ref = adapter.publish(_payload(photos, title="IKEA Malm desk with drawers"))
    assert site.listings[ref.remote_id].title == "IKEA Malm desk with drawers"


@pytest.mark.parametrize(
    ("category", "title", "option"),
    [
        ("Furniture", "Desk", "Furniture"),
        (None, "Trek mountain bike", "Bicycles"),
        ("Electronics", "Sony TV", "Electronics"),
        (None, "Video games bundle", "Video Games"),
        ("Kitchen", "Kettle", "Household"),
        (None, "Mystery box", "Miscellaneous"),
        (None, "Smart cart", "Miscellaneous"),  # "art" only as a whole word
    ],
)
def test_category_option(category, title, option, photos):
    assert category_option(_payload(photos, category=category, title=title)) == option


def test_condition_and_price_helpers():
    assert condition_option("new") == ("New", True)
    assert condition_option("poor") == ("Fair", False)
    with pytest.raises(PlatformError):
        condition_option("mint")
    assert [whole_dollars(c) for c in (0, 100, 101, 9999)] == [0, 1, 2, 100]


# ---------------------------------------------------------------- update, delist, stats


def test_update_rewrites_title_price_and_description(adapter, site, photos):
    listing = site.add_listing("Desk", "130")
    adapter.update(
        _ref(listing.listing_id), _payload(photos, title="Desk, white", price_cents=9500)
    )
    assert (listing.title, listing.price) == ("Desk, white", "95")
    assert listing.description.startswith("Solid desk.")


def test_update_that_facebook_never_confirms_is_an_error(adapter, driver, site, photos):
    listing = site.add_listing("Desk", "130")
    driver._save = lambda listing: None  # the Update click does nothing
    with pytest.raises(PlatformError, match="did not confirm"):
        adapter.update(_ref(listing.listing_id), _payload(photos))
    assert listing.price == "130"


def test_update_of_a_missing_post_raises_listing_gone(adapter, photos):
    with pytest.raises(ListingGone):
        adapter.update(_ref("404"), _payload(photos))


def test_delist_marks_the_listing_sold(adapter, site):
    listing = site.add_listing("Desk")
    adapter.delist(_ref(listing.listing_id))
    assert listing.status == "sold"


def test_delist_finishes_the_who_bought_it_dialog(adapter, driver, site):
    site.sold_dialog = True
    listing = site.add_listing("Desk")
    adapter.delist(_ref(listing.listing_id))
    assert listing.status == "sold"
    assert ("click", "skip") in driver.actions


def test_delist_of_a_sold_listing_does_nothing(adapter, driver, site):
    listing = site.add_listing("Desk", status="sold")
    adapter.delist(_ref(listing.listing_id))
    assert driver.kinds() == ["goto"]


def test_delist_deletes_when_mark_as_sold_is_not_offered(adapter, driver, site):
    site.offers_mark_sold = False
    listing = site.add_listing("Desk")
    adapter.delist(_ref(listing.listing_id))
    assert listing.status == "deleted"
    assert [a[1] for a in driver.actions if a[0] == "click"] == [
        "more",
        "delete_item",
        "delete_confirm",
    ]


def test_delist_of_a_missing_post_raises_listing_gone(adapter, site):
    listing = site.add_listing("Desk", status="deleted")
    with pytest.raises(ListingGone):
        adapter.delist(_ref(listing.listing_id))


def test_stats_reads_clicks_once_per_ttl(adapter, driver, site, clock):
    listing = site.add_listing("Desk", clicks=42)
    assert adapter.stats(_ref(listing.listing_id)) == ListingStats(views=42)
    gotos = driver.kinds().count("goto")

    listing.clicks = 50
    assert adapter.stats(_ref(listing.listing_id)).views == 42
    assert driver.kinds().count("goto") == gotos

    clock.advance(STATS_TTL)
    assert adapter.stats(_ref(listing.listing_id)).views == 50


def test_stats_of_a_listing_sold_on_facebook_raises_listing_gone(adapter, site):
    listing = site.add_listing("Desk", status="sold")
    with pytest.raises(ListingGone, match="sold"):
        adapter.stats(_ref(listing.listing_id))


def test_stats_of_a_deleted_listing_raises_listing_gone(adapter, site):
    listing = site.add_listing("Desk", status="deleted")
    with pytest.raises(ListingGone):
        adapter.stats(_ref(listing.listing_id))


def test_stats_without_a_card_but_a_live_post_are_zero(adapter, driver, site):
    listing = site.add_listing("Desk", clicks=7)
    driver.break_target(adapter.registry, "selling.card")
    assert adapter.stats(_ref(listing.listing_id)) == ListingStats()


# ---------------------------------------------------------------- inbox


@pytest.fixture
def chats(site):
    listing = site.add_listing("IKEA desk", "120")
    lucas = site.add_chat(
        "Lucas",
        "IKEA desk",
        ("Lucas", "Is this available?"),
        ("You", "Yes it is"),
        ("Lucas", "Would you take $100?"),
        listing_id=listing.listing_id,
        profile_id="555",
    )
    maya = site.add_chat("Maya", "IKEA desk", ("Maya", "Hi, still there?"), unread=False)
    mom = site.add_chat("Mom", "", ("Mom", "Call me"), marketplace=False)
    return listing, lucas, maya, mom


def test_fetch_threads_reads_marketplace_chats_only(adapter, driver, chats, clock):
    listing, lucas, maya, mom = chats
    adapter.stats(_ref(listing.listing_id))  # loads Your listings, so Maya's chat maps by title

    threads = {t.remote_id: t for t in adapter.fetch_threads(None)}

    assert set(threads) == {lucas.chat_id, maya.chat_id}
    assert ("goto", fb.THREAD_URL.format(thread_id=mom.chat_id)) not in driver.actions
    from_lucas = threads[lucas.chat_id]
    assert (from_lucas.listing_remote_id, from_lucas.buyer_handle, from_lucas.buyer_name) == (
        listing.listing_id,
        "555",
        "Lucas",
    )
    assert [(m.body, m.from_buyer) for m in from_lucas.messages] == [
        ("Is this available?", True),
        ("Yes it is", False),
        ("Would you take $100?", True),
    ]
    assert all(m.sent_at <= clock.now() and m.sent_at.tzinfo for m in from_lucas.messages)
    assert len({m.remote_id for m in from_lucas.messages}) == 3
    from_maya = threads[maya.chat_id]
    assert (from_maya.listing_remote_id, from_maya.buyer_handle) == (
        listing.listing_id,
        f"thread-{maya.chat_id}",
    )


def test_fetch_threads_polls_politely_and_opens_only_changed_chats(adapter, driver, chats, clock):
    _, lucas, _, _ = chats
    first = {t.remote_id: t for t in adapter.fetch_threads(None)}
    since = clock.now()

    actions = len(driver.actions)
    assert adapter.fetch_threads(since) == []
    assert len(driver.actions) == actions  # within INBOX_INTERVAL: no browser at all

    clock.advance(INBOX_INTERVAL)
    assert adapter.fetch_threads(since) == []  # nothing changed: no chat opened

    lucas.messages.append(("Just now", "Lucas", "I can pick up tonight"))
    lucas.unread = True
    clock.advance(INBOX_INTERVAL)
    [again] = adapter.fetch_threads(since)
    assert again.remote_id == lucas.chat_id
    assert [m.remote_id for m in again.messages[:3]] == [
        m.remote_id for m in first[lucas.chat_id].messages
    ]
    assert again.messages[-1].body == "I can pick up tonight"


def test_a_chat_the_owner_read_elsewhere_is_still_opened_when_its_last_line_changes(
    adapter, chats, clock
):
    _, lucas, _, _ = chats
    adapter.fetch_threads(None)
    lucas.messages.append(("Just now", "Lucas", "Hello?"))
    lucas.unread = False  # read on the owner's phone
    clock.advance(INBOX_INTERVAL)
    assert [t.remote_id for t in adapter.fetch_threads(clock.now())] == [lucas.chat_id]


def test_a_fresh_start_opens_the_newest_chats_and_takes_older_read_ones_as_seen(
    adapter, site, clock
):
    oldest = site.add_chat("Ann", "Lamp", ("Ann", "Price?"), unread=True)
    for n in range(THREADS_PER_POLL + 1):
        site.add_chat(f"Buyer{n}", "Lamp", ("Buyer", f"hello {n}"), unread=False)

    first = adapter.fetch_threads(None)
    assert len(first) == THREADS_PER_POLL
    assert oldest.chat_id not in {t.remote_id for t in first}

    clock.advance(INBOX_INTERVAL)
    later = adapter.fetch_threads(clock.now())
    assert [t.remote_id for t in later] == [oldest.chat_id]  # unread, so not skipped


def test_a_chat_that_fails_to_load_is_skipped_and_retried(adapter, driver, chats, clock):
    _, lucas, maya, _ = chats
    goto = driver.goto
    failing = fb.THREAD_URL.format(thread_id=maya.chat_id)

    def flaky(url: str) -> None:
        if url == failing:
            driver.goto = goto
            raise DriverError("Browser action failed: net::ERR_TIMED_OUT")
        goto(url)

    driver.goto = flaky
    assert [t.remote_id for t in adapter.fetch_threads(None)] == [lucas.chat_id]
    clock.advance(INBOX_INTERVAL)
    assert [t.remote_id for t in adapter.fetch_threads(clock.now())] == [maya.chat_id]


def test_send_message_returns_the_id_fetch_reports(adapter, chats, clock):
    _, lucas, _, _ = chats
    sent_id = adapter.send_message(lucas.chat_id, "Yes, $110 works.\nTonight at 7?")
    assert lucas.messages[-1] == ("Just now", "You", "Yes, $110 works.\nTonight at 7?")

    [thread] = [t for t in adapter.fetch_threads(None) if t.remote_id == lucas.chat_id]
    mine = [m for m in thread.messages if not m.from_buyer]
    assert mine[-1].remote_id == sent_id
    assert mine[-1].body == "Yes, $110 works. Tonight at 7?"


def test_send_presses_enter_when_there_is_no_send_button(adapter, driver, chats):
    _, lucas, _, _ = chats
    driver.break_target(adapter.registry, "thread.send")
    assert adapter.send_message(lucas.chat_id, "Still here") is not None
    assert ("press", "Enter") in driver.actions
    assert lucas.messages[-1][2] == "Still here"


def test_send_reports_messengers_limit(adapter, site, chats):
    _, lucas, _, _ = chats
    site.rate_limited = True
    with pytest.raises(PlatformError, match="limit"):
        adapter.send_message(lucas.chat_id, "Hello")
    assert lucas.messages[-1][2] != "Hello"


def test_ingest_and_human_reply_round_trip_without_duplicates(
    adapter, chats, engine, session, settings, clock, notifier
):
    listing, lucas, _, _ = chats
    item = Item(
        hint="desk",
        title="IKEA desk",
        status=ItemStatus.LISTED,
        created_at=clock.now(),
        updated_at=clock.now(),
    )
    session.add(item)
    session.flush()
    session.add(
        Listing(
            item_id=item.id,
            platform="facebook",
            remote_id=listing.listing_id,
            status=ListingStatus.LIVE,
            price_cents=12000,
            title="IKEA desk",
            description="",
        )
    )
    session.commit()

    ingest(session, clock, settings, notifier, adapter, None)
    session.commit()
    thread = session.exec(select(Thread).where(Thread.remote_id == lucas.chat_id)).one()
    assert thread.item_id == item.id

    control = ControlService(
        engine, clock, settings, {"facebook": adapter}, notifier, default_registry()
    )
    assert control.reply_as_human(thread.id, "Yes, tonight works").ok

    clock.advance(INBOX_INTERVAL)
    result = ingest(session, clock, settings, notifier, adapter, clock.now() - timedelta(minutes=1))
    session.commit()
    assert result.new_inbound == []
    bodies = session.exec(
        select(Message.body, Message.direction).where(Message.thread_id == thread.id)
    ).all()
    assert bodies.count(("Yes, tonight works", Direction.OUTBOUND)) == 1


def _live_on_facebook(session, clock, listing_id: str) -> Item:
    item = Item(
        hint="desk",
        title="IKEA desk",
        status=ItemStatus.LISTED,
        created_at=clock.now(),
        updated_at=clock.now(),
    )
    session.add(item)
    session.flush()
    session.add(
        Listing(
            item_id=item.id,
            platform="facebook",
            remote_id=listing_id,
            status=ListingStatus.LIVE,
            price_cents=12000,
            title="IKEA desk",
            description="",
        )
    )
    session.commit()
    return item


def test_a_repeat_of_a_message_messenger_no_longer_renders_is_still_new(
    adapter, driver, site, session, settings, clock, notifier
):
    """Messenger renders the latest 4 messages; older ones load when scrolled to."""
    listing = site.add_listing("IKEA desk", "120")
    chat = site.add_chat(
        "Lucas",
        "IKEA desk",
        ("Lucas", "Is this available?"),
        ("You", "Yes it is"),
        ("Lucas", "ok"),
        listing_id=listing.listing_id,
        window=4,
    )
    _live_on_facebook(session, clock, listing.listing_id)
    first = ingest(session, clock, settings, notifier, adapter, None)
    session.commit()
    assert len(first.new_inbound) == 2

    chat.messages += [("4:05 PM", "You", f"line {n}") for n in range(3)]
    chat.messages.append(("4:20 PM", "Lucas", "ok"))  # the first "ok" is out of the window
    chat.unread = True
    clock.advance(INBOX_INTERVAL)
    second = ingest(session, clock, settings, notifier, adapter, clock.now())
    session.commit()

    assert [session.get(Message, m).body for m in second.new_inbound] == ["ok"]
    assert ("scroll", "bubble") in driver.actions


def test_a_message_sent_in_a_long_chat_gets_the_id_fetch_reports(adapter, site, clock):
    chat = site.add_chat(
        "Lucas", "IKEA desk", ("You", "Sure"), ("Lucas", "Tonight?"), ("Lucas", "7pm"), window=2
    )

    sent_id = adapter.send_message(chat.chat_id, "Sure")

    [thread] = adapter.fetch_threads(None)
    first, again = [m.remote_id for m in thread.messages if m.body == "Sure"]
    assert sent_id == again != first


def test_owner_confirmed_publish_goes_live_through_the_publish_tool(
    adapter, site, engine, session, settings, clock, notifier
):
    settings.enabled_platforms = ["facebook"]
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (150, 100, 60)).save(buffer, "JPEG")
    item = create_item(session, clock, "IKEA desk $120")
    item.title, item.description = "IKEA desk", "Solid desk."
    item.category, item.condition = "Furniture", "good"
    ingest_photos(session, settings, item.id, [buffer.getvalue()])
    set_goal(session, clock, item.id, list_price_cents=12000, floor_price_cents=9500)
    session.commit()
    control = ControlService(
        engine, clock, settings, {"facebook": adapter}, notifier, default_registry()
    )

    [result] = control.confirm_and_publish(item.id, ["facebook"])

    assert result.ok, result.summary
    listing = session.exec(select(Listing).where(Listing.item_id == item.id)).one()
    posted = site.listings[listing.remote_id]
    assert listing.status == ListingStatus.LIVE
    assert (posted.title, posted.price, len(posted.photos)) == ("IKEA desk", "120", 1)


# ---------------------------------------------------------------- session and health


@pytest.mark.parametrize(
    "call",
    [
        lambda a, p: a.publish(_payload(p)),
        lambda a, p: a.update(_ref("1"), _payload(p)),
        lambda a, p: a.delist(_ref("1")),
        lambda a, p: a.fetch_threads(None),
        lambda a, p: a.send_message("1", "hi"),
        lambda a, p: a.stats(_ref("1")),
    ],
)
def test_signed_out_raises_auth_error_pointing_at_login(adapter, site, photos, call):
    site.logged_in = False
    with pytest.raises(PlatformAuthError, match="agent facebook login"):
        call(adapter, photos)


@pytest.mark.parametrize("problem", ["logged_in", "pin_prompt"])
def test_a_sign_out_frees_the_profile_for_the_login(adapter, driver, site, problem):
    """`agent facebook login` needs the profile, which `agent serve` holds while its browser is
    open: the adapter closes it and says the login works while serve runs on."""
    setattr(site, problem, not getattr(site, problem))

    with pytest.raises(PlatformAuthError, match="works while `agent serve` keeps running"):
        adapter.fetch_threads(None)

    assert driver.closed


def test_signed_in_needs_the_cookie_and_a_normal_page(driver, site):
    driver.url = "https://www.facebook.com/checkpoint/123/"
    assert not signed_in(driver)
    driver.url = fb.HOME_URL
    assert signed_in(driver)
    site.logged_in = False
    assert not signed_in(driver)


def test_messenger_pin_prompt_needs_the_owner(adapter, site):
    site.pin_prompt = True
    with pytest.raises(PlatformAuthError, match="PIN"):
        adapter.fetch_threads(None)
    assert "PIN" in adapter.healthcheck()[0]


def _healing_adapter(settings, clock, engine, site) -> tuple[FacebookAdapter, ScriptedProvider]:
    """Pages draw after load (one wait), and a vision model would heal any selector."""
    provider = ScriptedProvider(lambda request: {"css": "#anything"})
    router = ProviderRouter([provider], QuotaTracker(engine, clock, {}))
    driver = FakeDriver(site, load_waits=1)
    return FacebookAdapter(settings, clock, driver, router, find_timeout_s=1.0), provider


@pytest.mark.parametrize("site", [FakeSite(pin_prompt=True), FakeSite(pin_overlay=True)])
def test_a_pin_prompt_drawn_after_load_needs_the_owner(settings, clock, engine, site):
    adapter, provider = _healing_adapter(settings, clock, engine, site)

    with pytest.raises(PlatformAuthError, match="PIN"):
        adapter.fetch_threads(None)
    with pytest.raises(PlatformAuthError, match="PIN"):
        adapter.send_message("1", "Hello")
    assert "PIN" in adapter.healthcheck()[0]
    assert provider.requests == []  # nothing healed: the page is hidden, not changed


def test_a_blocked_notice_drawn_after_load_stops_publishing(settings, clock, engine, photos):
    site = FakeSite(blocked=True)
    adapter, provider = _healing_adapter(settings, clock, engine, site)

    with pytest.raises(PlatformError, match="cannot use Marketplace"):
        adapter.publish(_payload(photos))
    assert site.listings == {} and provider.requests == []


def test_a_notice_stops_healing(settings, clock, engine):
    """The PIN box is drawn after the chats, while the folder is waited for: no heal is spent."""
    site = FakeSite()
    adapter, provider = _healing_adapter(settings, clock, engine, site)
    driver = adapter.driver
    driver.break_target(adapter.registry, "inbox.marketplace_folder")
    waits = []
    page_wait = driver.wait_for

    def pin_drawn_on_second_wait(specs, timeout_s):
        waits.append(timeout_s)
        site.pin_overlay = len(waits) > 1  # the first wait is for the page itself
        return page_wait(specs, timeout_s)

    driver.wait_for = pin_drawn_on_second_wait

    with pytest.raises(PlatformAuthError, match="PIN"):
        adapter.fetch_threads(None)
    assert provider.requests == []


def test_a_blocked_account_is_a_platform_error(adapter, site, photos):
    site.blocked = True
    with pytest.raises(PlatformError, match="cannot use Marketplace"):
        adapter.publish(_payload(photos))


def test_healthcheck(adapter, site):
    assert adapter.healthcheck() == []
    site.logged_in = False
    [problem] = adapter.healthcheck()
    assert "agent facebook login" in problem


def test_validate_selectors_is_read_only(adapter, driver):
    assert adapter.validate_selectors() == []
    assert set(driver.kinds()) <= {"goto", "wait_for"}
    assert [a[1] for a in driver.actions if a[0] == "goto"] == [fb.CREATE_URL, fb.INBOX_URL]


def test_validate_selectors_names_what_no_longer_matches(adapter, driver):
    driver.break_target(adapter.registry, "create.title")
    [problem] = adapter.validate_selectors()
    assert problem.startswith("create.title (the Title text input")


def test_close_closes_the_browser(adapter, driver):
    adapter.close()
    assert driver.closed


# ---------------------------------------------------------------- page text


ZONE = __import__("zoneinfo").ZoneInfo("America/Toronto")
NOW = datetime(2026, 10, 9, 22, 0, tzinfo=UTC)  # Friday 18:00 in Toronto


@pytest.mark.parametrize(
    ("label", "local"),
    [
        ("4:01 PM", datetime(2026, 10, 9, 16, 1)),
        ("4:01\u202fPM", datetime(2026, 10, 9, 16, 1)),
        ("Yesterday at 9:15 AM", datetime(2026, 10, 8, 9, 15)),
        ("Mon 4:01 PM", datetime(2026, 10, 5, 16, 1)),
        ("Friday 8:00 AM", datetime(2026, 10, 2, 8, 0)),  # a weekday name is never today
        ("Oct 3, 2026, 4:01 PM", datetime(2026, 10, 3, 16, 1)),
        ("October 3, 2026 at 4:01 PM", datetime(2026, 10, 3, 16, 1)),
        ("Dec 30, 9:15 AM", datetime(2025, 12, 30, 9, 15)),
        ("14:30", datetime(2026, 10, 9, 14, 30)),
    ],
)
def test_parse_sent(label, local):
    assert parse_sent(label, NOW, ZONE) == local.replace(tzinfo=ZONE).astimezone(UTC)


@pytest.mark.parametrize(
    "label", ["", "Just now", "13:01 PM", "Smarch 3, 4:01 PM", "Feb 30, 1:00 PM"]
)
def test_parse_sent_gives_up_on_what_it_cannot_read(label):
    assert parse_sent(label, NOW, ZONE) is None


def test_parse_sent_never_returns_the_future():
    assert parse_sent("11:59 PM", NOW, ZONE) == NOW


def test_page_text_helpers():
    assert split_heading("Conversation titled Lucas · IKEA desk · white") == (
        "Lucas",
        "IKEA desk · white",
    )
    assert row_key("Lucas · Desk\n1 unread message\nLucas: hi · 2m") == row_key(
        "Lucas · Desk\nLucas: hi · 3h"
    )
    bubbles = parse_bubbles(
        [
            "Enter, Message sent 4:01 PM by Lucas: Is it: available?",
            "Enter, Message sent 4:01 PM by Lucas: Is it: available?",  # rendered twice
            "Enter, Message sent 4:02 PM by You: Yes",
            "Seen by Lucas",
        ]
    )
    assert [(b.sender, b.text) for b in bubbles] == [("Lucas", "Is it: available?"), ("You", "Yes")]
