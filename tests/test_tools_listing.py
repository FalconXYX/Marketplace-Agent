"""update_price, rewrite_listing, publish, relist."""
import itertools
from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import event
from sqlmodel import select

from marketplace_agent.core.items import get_goal, item_photos, set_goal
from marketplace_agent.core.models import (
    Firmness,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Photo,
    PriceChange,
)
from marketplace_agent.guard.invariants import ADDRESS_PLACEHOLDER
from marketplace_agent.guard.ledger import LedgerConflict
from marketplace_agent.guard.validate import validate
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import (
    AwaitingHuman,
    ListingStats,
    PlatformAuthError,
    PlatformError,
    PlatformRef,
)
from marketplace_agent.platforms.base import ListingGone
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.tools.base import Allow, NeedsApproval, Reject, ToolContext
from marketplace_agent.tools.impl.common import latest_listing
from marketplace_agent.tools.impl.listing import (
    PlatformArgs,
    PublishTool,
    RelistTool,
    RewriteListingArgs,
    RewriteListingTool,
)
from marketplace_agent.tools.impl.pricing import UpdatePriceArgs, UpdatePriceTool

LIST, FLOOR = 12_000, 9_500
PRICE, REWRITE, PUBLISH, RELIST = (
    UpdatePriceTool(),
    RewriteListingTool(),
    PublishTool(),
    RelistTool(),
)


class StubAdapter:
    """Records calls; fail[method] raises once."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.calls: list[tuple[str, object]] = []
        self.fail: dict[str, Exception] = {}
        self._ids = itertools.count(1)

    def _call(self, method: str, arg: object) -> None:
        self.calls.append((method, arg))
        if method in self.fail:
            raise self.fail.pop(method)

    def called(self, method: str) -> list:
        return [arg for name, arg in self.calls if name == method]

    def publish(self, listing, *, assist=False):
        self._call("publish", listing)
        n = next(self._ids)
        return PlatformRef(self.name, f"{self.name}-{n}", f"https://{self.name}.test/{n}")

    def update(self, ref, listing):
        self._call("update", (ref, listing))

    def delist(self, ref):
        self._call("delist", ref)

    def fetch_threads(self, since):
        return []

    def send_message(self, thread_remote_id, body):
        self._call("send_message", (thread_remote_id, body))
        return None

    def stats(self, ref):
        return ListingStats()

    def search(self, query, limit=20):
        return []

    def healthcheck(self):
        return []


@pytest.fixture
def two_platforms(settings):
    return settings.model_copy(update={"enabled_platforms": ["fake", "kijiji", "facebook"]})


@pytest.fixture
def adapters() -> dict[str, StubAdapter]:
    return {name: StubAdapter(name) for name in ("fake", "kijiji", "facebook")}


@pytest.fixture
def item(session, clock) -> Item:
    return _item(session, clock)


@pytest.fixture
def ctx(session, clock, two_platforms, notifier, item, adapters) -> ToolContext:
    return _ctx(session, clock, two_platforms, notifier, item, adapters)


def _ctx(session, clock, settings, notifier, item, platforms, actor="agent") -> ToolContext:
    return ToolContext(
        session=session,
        clock=clock,
        settings=settings,
        platforms=platforms,
        notifier=notifier,
        item_id=item.id,
        actor=actor,
    )


def _item(
    session,
    clock,
    *,
    firmness: Firmness = Firmness.SLIGHT,
    status: ItemStatus = ItemStatus.LISTED,
    goal: bool = True,
    photos: int = 2,
) -> Item:
    item = Item(
        title="IKEA desk",
        description="Solid desk, light wear on one corner.",
        condition="good",
        status=status,
        created_at=clock.now(),
        updated_at=clock.now(),
    )
    session.add(item)
    session.flush()
    if goal:
        set_goal(
            session,
            clock,
            item.id,
            list_price_cents=LIST,
            floor_price_cents=FLOOR,
            firmness=firmness,
        )
    for index in range(photos):
        session.add(
            Photo(
                item_id=item.id,
                path=f"photos/{item.id}/{index}.jpg",
                position=index,
                width=10,
                height=10,
                sha256=f"{item.id}-{index}",
            )
        )
    session.commit()
    return item


def _listing(
    session,
    clock,
    item: Item,
    platform: str = "fake",
    status: ListingStatus = ListingStatus.LIVE,
    hours_ago: float = 48,
) -> Listing:
    posted = clock.now() - timedelta(hours=hours_ago)
    listing = Listing(
        item_id=item.id,
        platform=platform,
        remote_id=f"{platform}-old-{item.id}",
        url=f"https://{platform}.test/old",
        status=status,
        price_cents=LIST,
        title=item.title,
        description=item.description,
        published_at=posted,
        delisted_at=posted + timedelta(hours=1) if status == ListingStatus.DELISTED else None,
    )
    session.add(listing)
    session.commit()
    return listing


def _reject(tool, ctx: ToolContext, args) -> str:
    verdict = tool.check(ctx, args)
    assert isinstance(verdict, Reject), verdict
    return verdict.reason


def _run(tool, ctx: ToolContext, args):
    assert tool.check(ctx, args) == Allow()
    result = tool.execute(ctx, args)
    ctx.session.commit()
    return result


def _price_changes(session, item: Item) -> list[PriceChange]:
    return list(session.exec(select(PriceChange).where(PriceChange.item_id == item.id)).all())


def _listings(session, item: Item, platform: str) -> list[Listing]:
    query = select(Listing).where(Listing.item_id == item.id, Listing.platform == platform)
    return list(session.exec(query.order_by(Listing.id)).all())


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (PRICE, UpdatePriceArgs(price_cents=11_000)),
        (PRICE, UpdatePriceArgs(price_cents=100)),
        (REWRITE, RewriteListingArgs(title="Desk", photo_order=[999])),
        (REWRITE, RewriteListingArgs(description="Call 416-555-0199")),
        (PUBLISH, PlatformArgs(platform="kijiji")),
        (RELIST, PlatformArgs(platform="fake")),
    ],
)
def test_check_writes_nothing(ctx, session, clock, item, tool, args):
    _listing(session, clock, item)
    flushes = []
    event.listen(session, "after_flush", lambda *_: flushes.append(1))
    tool.check(ctx, args)
    assert flushes == [] and not session.new and not session.dirty


def test_trust_keys():
    assert (PRICE.trust_action, REWRITE.trust_action) == ("update_price", "rewrite_listing")
    assert (PUBLISH.trust_action, RELIST.trust_action) == ("publish", "relist")
    assert not any(tool.read_only for tool in (PRICE, REWRITE, PUBLISH, RELIST))


# ---------------------------------------------------------------- update_price


def test_update_price_moves_goal_and_every_live_listing(ctx, session, clock, item, adapters):
    fake = _listing(session, clock, item, "fake")
    kijiji = _listing(session, clock, item, "kijiji")
    _listing(session, clock, item, "facebook", ListingStatus.DELISTED)

    result = _run(PRICE, ctx, UpdatePriceArgs(price_cents=10_500))

    assert result.ok and result.data == {"errors": [], "gone": []}
    assert result.summary == "Price changed from $120.00 to $105.00."
    assert get_goal(session, item.id).list_price_cents == 10_500
    [change] = _price_changes(session, item)
    assert (change.old_cents, change.new_cents, change.actor) == (LIST, 10_500, "agent")
    for listing, adapter in ((fake, adapters["fake"]), (kijiji, adapters["kijiji"])):
        [(ref, payload)] = adapter.called("update")
        assert ref.remote_id == listing.remote_id and payload.price_cents == 10_500
        assert listing.price_cents == 10_500
    assert adapters["facebook"].calls == []
    assert PRICE.trust_platform(ctx, UpdatePriceArgs(price_cents=10_500)) == "*"


def test_update_price_with_nothing_published_changes_the_goal_only(ctx, session, item):
    assert _run(PRICE, ctx, UpdatePriceArgs(price_cents=11_000)).ok
    assert get_goal(session, item.id).list_price_cents == 11_000


def test_update_price_partial_failure_keeps_the_true_state(ctx, session, clock, item, adapters):
    fake = _listing(session, clock, item, "fake")
    kijiji = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["update"] = PlatformError("kijiji timed out")

    result = _run(PRICE, ctx, UpdatePriceArgs(price_cents=10_500))

    assert not result.ok
    assert result.data["errors"] == ["kijiji: kijiji timed out"]
    assert "Some listings still show the old price" in result.summary
    assert fake.price_cents == 10_500 and fake.last_error is None
    assert kijiji.price_cents == LIST and kijiji.last_error == "kijiji timed out"
    assert get_goal(session, item.id).list_price_cents == 10_500


def test_update_price_total_failure_raises_and_rolls_back(ctx, session, clock, item, adapters):
    _listing(session, clock, item, "fake")
    adapters["fake"].fail["update"] = PlatformError("down")

    with pytest.raises(PlatformError, match="No listing could be updated. fake: down"):
        PRICE.execute(ctx, UpdatePriceArgs(price_cents=10_500))
    session.rollback()

    assert get_goal(session, item.id).list_price_cents == LIST
    assert _price_changes(session, item) == []


def test_update_price_needs_a_goal(session, clock, two_platforms, notifier, adapters):
    item = _item(session, clock, goal=False)
    ctx = _ctx(session, clock, two_platforms, notifier, item, adapters)
    reason = _reject(PRICE, ctx, UpdatePriceArgs(price_cents=10_000))
    assert "only the owner can set one. Escalate." in reason


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ItemStatus.DRAFT, "still a draft"),
        (ItemStatus.COMMITTED, "change the price only if that deal is released"),
        (ItemStatus.SOLD, "The item is sold; there is nothing more to change the price."),
    ],
)
def test_update_price_needs_a_listed_item(
    session, clock, two_platforms, notifier, adapters, status, expected
):
    item = _item(session, clock, status=status)
    ctx = _ctx(session, clock, two_platforms, notifier, item, adapters)
    assert expected in _reject(PRICE, ctx, UpdatePriceArgs(price_cents=10_000))


def test_update_price_never_below_the_floor(ctx):
    reason = _reject(PRICE, ctx, UpdatePriceArgs(price_cents=9_000))
    assert reason == "$90.00 is below the floor of $95.00. Choose $95.00 or more."


def test_update_price_must_change_something(ctx):
    reason = _reject(PRICE, ctx, UpdatePriceArgs(price_cents=LIST))
    assert "already $120.00. Choose a different price." in reason


def test_update_price_rejects_non_positive_prices():
    with pytest.raises(ValidationError):
        PRICE.parse_args({"price_cents": 0})


def test_update_price_firm_never_goes_down(session, clock, two_platforms, notifier, adapters):
    item = _item(session, clock, firmness=Firmness.FIRM)
    ctx = _ctx(session, clock, two_platforms, notifier, item, adapters)
    reason = _reject(PRICE, ctx, UpdatePriceArgs(price_cents=11_000))
    assert "firm at $120.00: you may raise it but never lower it" in reason
    assert PRICE.check(ctx, UpdatePriceArgs(price_cents=13_000)) == Allow()


def test_update_price_once_per_24_hours(ctx, clock):
    _run(PRICE, ctx, UpdatePriceArgs(price_cents=11_000))
    clock.advance(hours=23, minutes=59)

    reason = _reject(PRICE, ctx, UpdatePriceArgs(price_cents=10_500))
    assert "last changed Thu Oct 1 12:00; it may change once per 24 hours" in reason
    assert "Leave it until Fri Oct 2 12:00" in reason

    clock.advance(minutes=1)
    assert PRICE.check(ctx, UpdatePriceArgs(price_cents=10_500)) == Allow()


def test_the_owner_may_reprice_firm_items_anytime(
    session, clock, two_platforms, notifier, adapters
):
    item = _item(session, clock, firmness=Firmness.FIRM)
    human = _ctx(session, clock, two_platforms, notifier, item, adapters, actor="human")
    _run(PRICE, human, UpdatePriceArgs(price_cents=11_000))
    assert _run(PRICE, human, UpdatePriceArgs(price_cents=10_000)).ok
    assert [c.actor for c in _price_changes(session, item)] == ["human", "human"]
    assert "below the floor" in _reject(PRICE, human, UpdatePriceArgs(price_cents=9_000))


def test_an_owner_change_starts_the_agents_24_hours(
    session, clock, two_platforms, notifier, adapters, item
):
    human = _ctx(session, clock, two_platforms, notifier, item, adapters, actor="human")
    agent = _ctx(session, clock, two_platforms, notifier, item, adapters)
    _run(PRICE, human, UpdatePriceArgs(price_cents=11_000))
    assert "once per 24 hours" in _reject(PRICE, agent, UpdatePriceArgs(price_cents=10_500))


# ---------------------------------------------------------------- rewrite_listing


def test_rewrite_updates_item_photos_and_live_listings(ctx, session, clock, item, adapters):
    listing = _listing(session, clock, item, "kijiji")
    first, second = (photo.id for photo in item_photos(session, item.id))
    args = RewriteListingArgs(
        title="IKEA Malm desk, 140x65 cm",
        description="Sturdy white desk. Pickup near Bloor and Ossington.",
        photo_order=[second, first],
    )

    result = _run(REWRITE, ctx, args)

    assert result.ok and result.summary == "Rewrote the title, description, photo order."
    assert item.title == "IKEA Malm desk, 140x65 cm"
    assert [p.id for p in item_photos(session, item.id)] == [second, first]
    [(ref, payload)] = adapters["kijiji"].called("update")
    assert ref.remote_id == listing.remote_id
    assert payload.title == args.title and payload.price_cents == LIST
    assert [path.name for path in payload.photo_paths] == ["1.jpg", "0.jpg"]
    assert all(path.is_absolute() for path in payload.photo_paths)
    assert listing.title == args.title and listing.description == args.description


def test_rewrite_title_only_keeps_the_description(ctx, item):
    assert _run(REWRITE, ctx, RewriteListingArgs(title="  White desk  ")).ok
    assert item.title == "White desk"
    assert item.description == "Solid desk, light wear on one corner."


def test_update_price_trust_platform_follows_live_facebook(ctx, session, clock, item):
    args = UpdatePriceArgs(price_cents=10_500)
    _listing(session, clock, item, "fake")
    assert PRICE.trust_platform(ctx, args) == "*"
    _listing(session, clock, item, "facebook")
    assert PRICE.trust_platform(ctx, args) == "facebook"  # ("facebook", "update_price") is ASK


def test_rewrite_trust_platform_follows_live_facebook(ctx, session, clock, item):
    args = RewriteListingArgs(title="White desk")
    _listing(session, clock, item, "kijiji")
    assert REWRITE.trust_platform(ctx, args) == "*"
    _listing(session, clock, item, "facebook")
    assert REWRITE.trust_platform(ctx, args) == "facebook"


def test_rewrite_partial_failure_reports_the_listing(ctx, session, clock, item, adapters):
    _listing(session, clock, item, "fake")
    kijiji = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["update"] = PlatformError("captcha")

    result = _run(REWRITE, ctx, RewriteListingArgs(title="White desk"))

    assert not result.ok and result.data["errors"] == ["kijiji: captcha"]
    assert kijiji.title == "IKEA desk" and kijiji.last_error == "captcha"


def test_rewrite_total_failure_raises(ctx, session, clock, item, adapters):
    _listing(session, clock, item, "fake")
    adapters["fake"].fail["update"] = PlatformError("down")
    with pytest.raises(PlatformError):
        REWRITE.execute(ctx, RewriteListingArgs(title="White desk"))
    session.rollback()
    assert session.get(Item, item.id).title == "IKEA desk"


@pytest.mark.parametrize(
    ("tool", "args"),
    [(REWRITE, RewriteListingArgs(title="New")), (RELIST, PlatformArgs(platform="fake"))],
)
def test_rewrite_and_relist_need_a_goal(session, clock, settings, notifier, adapters, tool, args):
    item = _item(session, clock, goal=False)
    _listing(session, clock, item, "fake")
    ctx = _ctx(session, clock, settings, notifier, item, adapters)
    assert "only the owner can set one. Escalate." in _reject(tool, ctx, args)


def test_rewrite_needs_a_listed_item(session, clock, two_platforms, notifier, adapters):
    item = _item(session, clock, status=ItemStatus.COMMITTED)
    ctx = _ctx(session, clock, two_platforms, notifier, item, adapters)
    reason = _reject(REWRITE, ctx, RewriteListingArgs(title="New"))
    assert "rewrite the listing only if that deal is released" in reason


@pytest.mark.parametrize(
    "args",
    [RewriteListingArgs(), RewriteListingArgs(title="IKEA desk", photo_order=None)],
)
def test_rewrite_must_change_something(ctx, args):
    assert "exactly as it is" in _reject(REWRITE, ctx, args)


def test_rewrite_same_photo_order_is_no_change(ctx, session, item):
    order = [p.id for p in item_photos(session, item.id)]
    assert "exactly as it is" in _reject(REWRITE, ctx, RewriteListingArgs(photo_order=order))


@pytest.mark.parametrize("field", ["title", "description"])
def test_rewrite_rejects_blank_text(ctx, field):
    reason = _reject(REWRITE, ctx, RewriteListingArgs(**{field: "   "}))
    assert f"The {field} cannot be blank. Leave it out to keep the current one." == reason


@pytest.mark.parametrize("order", [lambda a, b: [a], lambda a, b: [a, a], lambda a, b: [a, b, 99]])
def test_rewrite_photo_order_must_be_a_permutation(ctx, session, item, order):
    first, second = (p.id for p in item_photos(session, item.id))
    reason = _reject(REWRITE, ctx, RewriteListingArgs(photo_order=order(first, second)))
    assert f"each of this item's photo ids exactly once, in any order: {[first, second]}" in reason


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (RewriteListingArgs(description="Text 416-555-0199"), "In the description: The listing"),
        (RewriteListingArgs(title="Desk at 123 Example St"), "In the title: Listing text must"),
        (RewriteListingArgs(title=f"Desk near {ADDRESS_PLACEHOLDER}"), "Mention only the pickup"),
        (RewriteListingArgs(description="Was $150, now $80!"), "below the floor of $95.00"),
        (RewriteListingArgs(description="Pay by money order"), "matches a scam pattern"),
    ],
)
def test_rewrite_text_rules(ctx, args, expected):
    assert expected in _reject(REWRITE, ctx, args)


def test_rewrite_title_limit_is_in_the_schema():
    with pytest.raises(ValidationError):
        REWRITE.parse_args({"title": "x" * 101})


# ---------------------------------------------------------------- publish


def test_publish_goes_live(ctx, session, clock, item, adapters, two_platforms):
    result = _run(PUBLISH, ctx, PlatformArgs(platform="kijiji"))

    [listing] = _listings(session, item, "kijiji")
    assert result.ok and result.summary == "Published on kijiji at $120.00: https://kijiji.test/1"
    assert result.data == {"listing_id": listing.id, "status": "live", "url": listing.url}
    assert listing.status == ListingStatus.LIVE and listing.remote_id == "kijiji-1"
    assert listing.published_at == clock.now() and listing.price_cents == LIST
    [payload] = adapters["kijiji"].called("publish")
    assert payload.location == two_platforms.pickup_area
    assert payload.title == "IKEA desk" and payload.price_cents == LIST
    assert all(path.is_absolute() for path in payload.photo_paths)
    assert len(payload.photo_paths) == 2
    assert PUBLISH.trust_platform(ctx, PlatformArgs(platform="kijiji")) == "kijiji"


def test_publish_failure_marks_failed_and_allows_a_retry(ctx, session, item, adapters):
    adapters["kijiji"].fail["publish"] = PlatformError("ad rejected")

    result = _run(PUBLISH, ctx, PlatformArgs(platform="kijiji"))

    assert not result.ok and result.summary == "Publishing on kijiji failed: ad rejected"
    [failed] = _listings(session, item, "kijiji")
    assert failed.status == ListingStatus.FAILED and failed.last_error == "ad rejected"
    assert _run(PUBLISH, ctx, PlatformArgs(platform="kijiji")).ok
    assert [lst.status for lst in _listings(session, item, "kijiji")] == [
        ListingStatus.FAILED,
        ListingStatus.LIVE,
    ]


def test_publish_never_takes_the_id_of_another_items_post(ctx, session, clock, item, adapters):
    """A platform that cannot confirm the new post may answer with an older one that has the
    same title; that id must not become this item's listing."""
    other = _item(session, clock)
    _listing(session, clock, other, "kijiji")
    adapters["kijiji"].publish = lambda listing, *, assist=False: PlatformRef(  # type: ignore
        "kijiji", f"kijiji-old-{other.id}", "https://kijiji.test/old"
    )

    result = _run(PUBLISH, ctx, PlatformArgs(platform="kijiji"))

    assert not result.ok and f"already item #{other.id}'s listing" in result.summary
    [failed] = _listings(session, item, "kijiji")
    assert failed.status == ListingStatus.FAILED and failed.remote_id is None


def test_relist_never_takes_back_the_id_it_took_down(ctx, session, clock, item, adapters):
    listing = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].publish = lambda payload, *, assist=False: PlatformRef(  # type: ignore
        "kijiji", listing.remote_id, listing.url
    )

    result = _run(RELIST, ctx, PlatformArgs(platform="kijiji"))

    assert not result.ok and "was not confirmed" in result.summary
    assert listing.status == ListingStatus.DELISTED


@pytest.mark.parametrize("tool", [PUBLISH, RELIST])
def test_facebook_needs_a_condition_before_posting(ctx, session, clock, item, tool):
    """The adapter would refuse it every time; the agent is told to ask the owner instead."""
    if tool is RELIST:
        _listing(session, clock, item, "facebook")
    item.condition = None
    session.commit()

    reason = _reject(tool, ctx, PlatformArgs(platform="facebook"))

    assert reason.startswith("Facebook needs the item's condition, and none is set.")
    assert "Escalate so the owner sets it" in reason
    if tool is PUBLISH:  # elsewhere no condition is needed
        assert tool.check(ctx, PlatformArgs(platform="kijiji")) == Allow()


def test_publish_auth_failure_is_recorded_too(ctx, session, item, adapters):
    adapters["kijiji"].fail["publish"] = PlatformAuthError("session expired")
    assert not _run(PUBLISH, ctx, PlatformArgs(platform="kijiji")).ok
    assert _listings(session, item, "kijiji")[0].status == ListingStatus.FAILED


def test_publish_in_assist_mode_waits_for_the_owner(ctx, session, notifier, item, adapters):
    adapters["facebook"].fail["publish"] = AwaitingHuman("form filled")

    result = _run(PUBLISH, ctx, PlatformArgs(platform="facebook"))

    assert result.ok and "the owner must click Post" in result.summary
    [listing] = _listings(session, item, "facebook")
    assert listing.status == ListingStatus.AWAITING_HUMAN and listing.remote_id is None
    assert listing.published_at is None
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "Finish posting on facebook" and alert.item_id == item.id
    assert "waiting for the owner" in _reject(PUBLISH, ctx, PlatformArgs(platform="facebook"))


def test_publish_on_facebook_needs_approval_at_ask(ctx, session, adapters):
    args = PlatformArgs(platform="facebook")
    verdict = validate(ctx, PUBLISH, args)
    assert isinstance(verdict, NeedsApproval)
    assert "publish on facebook needs the owner's approval" in verdict.reason
    assert adapters["facebook"].calls == []
    assert validate(ctx, PUBLISH, PlatformArgs(platform="kijiji")) == Allow()


def test_the_owner_publishes_on_facebook_without_approval(
    session, clock, two_platforms, notifier, adapters, item
):
    human = _ctx(session, clock, two_platforms, notifier, item, adapters, actor="human")
    assert validate(human, PUBLISH, PlatformArgs(platform="facebook")) == Allow()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ListingStatus.LIVE, "already live on kijiji. Use relist to refresh it"),
        (ListingStatus.PENDING, "already in progress. Do not publish again"),
        (ListingStatus.AWAITING_HUMAN, "waiting for the owner to finish it"),
    ],
)
def test_publish_once_per_platform(ctx, session, clock, item, status, expected):
    _listing(session, clock, item, "kijiji", status)
    assert expected in _reject(PUBLISH, ctx, PlatformArgs(platform="kijiji"))


def test_publish_again_after_a_delisting(ctx, session, clock, item):
    _listing(session, clock, item, "kijiji", ListingStatus.DELISTED)
    assert PUBLISH.check(ctx, PlatformArgs(platform="kijiji")) == Allow()


@pytest.mark.parametrize("platform", ["craigslist", "kijiji"])
def test_publish_needs_an_enabled_connected_platform(
    session, clock, settings, notifier, item, platform
):
    ctx = _ctx(session, clock, settings, notifier, item, {"fake": StubAdapter()})
    reason = _reject(PUBLISH, ctx, PlatformArgs(platform=platform))
    assert f"{platform!r} is not an enabled, connected platform. Choose one of: fake." == reason


def test_publish_cap(session, clock, two_platforms, notifier, adapters, item):
    capped = two_platforms.model_copy(update={"daily_publish_cap": {"kijiji": 1}})
    other = _item(session, clock)
    _listing(session, clock, other, "kijiji", hours_ago=1)
    ctx = _ctx(session, clock, capped, notifier, item, adapters)

    reason = _reject(PUBLISH, ctx, PlatformArgs(platform="kijiji"))

    assert "daily limit of 1 posts on kijiji is used up (1 today)" in reason
    assert "Post nothing more there until Thu Oct 1 20:00" in reason


def test_publish_needs_a_listed_item_with_goal_photos_and_text(
    session, clock, two_platforms, notifier, adapters
):
    def reason_for(item: Item) -> str:
        ctx = _ctx(session, clock, two_platforms, notifier, item, adapters)
        return _reject(PUBLISH, ctx, PlatformArgs(platform="kijiji"))

    assert "still a draft" in reason_for(_item(session, clock, status=ItemStatus.DRAFT))
    assert "only the owner can set one. Escalate." in reason_for(_item(session, clock, goal=False))
    assert "no photos" in reason_for(_item(session, clock, photos=0))
    untitled = _item(session, clock)
    untitled.description = " "
    assert "Write them with rewrite_listing first" in reason_for(untitled)


# ---------------------------------------------------------------- relist


def test_relist_takes_down_and_reposts(ctx, session, clock, item, adapters):
    listing = _listing(session, clock, item, "kijiji", hours_ago=24)

    result = _run(RELIST, ctx, PlatformArgs(platform="kijiji"))

    assert result.ok and result.summary.startswith("Relisted on kijiji at $120.00")
    [old_ref] = adapters["kijiji"].called("delist")
    assert old_ref.remote_id == f"kijiji-old-{item.id}"
    assert len(adapters["kijiji"].called("publish")) == 1
    assert listing.status == ListingStatus.LIVE and listing.remote_id == "kijiji-1"
    assert listing.last_relisted_at == clock.now() and listing.delisted_at is None
    current, replaced = _listings(session, item, "kijiji")
    assert current is listing and latest_listing(session, item.id, "kijiji") is listing
    assert (replaced.remote_id, replaced.status) == (old_ref.remote_id, ListingStatus.DELISTED)
    assert replaced.published_at is None  # history only: never shown, never relisted
    assert RELIST.trust_platform(ctx, PlatformArgs(platform="kijiji")) == "kijiji"


def test_relist_a_delisted_listing_skips_the_takedown(ctx, session, clock, item, adapters):
    listing = _listing(session, clock, item, "kijiji", ListingStatus.DELISTED)
    assert _run(RELIST, ctx, PlatformArgs(platform="kijiji")).ok
    assert adapters["kijiji"].called("delist") == []
    assert listing.status == ListingStatus.LIVE


def test_relist_within_24_hours_is_rejected(ctx, session, clock, item):
    listing = _listing(session, clock, item, "kijiji", hours_ago=30)
    listing.last_relisted_at = clock.now() - timedelta(hours=2)
    session.commit()
    reason = _reject(RELIST, ctx, PlatformArgs(platform="kijiji"))
    assert "last posted Thu Oct 1 10:00" in reason
    assert "Relist no earlier than Fri Oct 2 10:00" in reason


@pytest.mark.parametrize("status", [None, ListingStatus.FAILED, ListingStatus.AWAITING_HUMAN])
def test_relist_needs_a_live_or_delisted_listing(ctx, session, clock, item, status):
    if status is not None:
        _listing(session, clock, item, "kijiji", status)
    reason = _reject(RELIST, ctx, PlatformArgs(platform="kijiji"))
    assert "no live or delisted listing on kijiji to relist" in reason
    assert "Use publish to post it there" in reason
    if status is not None:
        assert f"(the latest one is {status.value})" in reason


def test_relist_counts_toward_the_publish_cap(session, clock, two_platforms, notifier, adapters):
    capped = two_platforms.model_copy(update={"daily_publish_cap": {"kijiji": 1}})
    item = _item(session, clock)
    _listing(session, clock, item, "kijiji", hours_ago=25)
    _listing(session, clock, _item(session, clock), "kijiji", hours_ago=1)
    ctx = _ctx(session, clock, capped, notifier, item, adapters)
    assert "daily limit of 1 posts on kijiji" in _reject(
        RELIST, ctx, PlatformArgs(platform="kijiji")
    )


def test_relist_needs_a_listed_item_and_platform(session, clock, settings, notifier, adapters):
    item = _item(session, clock, status=ItemStatus.COMMITTED)
    ctx = _ctx(session, clock, settings, notifier, item, adapters)
    assert "relist only if that deal is released" in _reject(
        RELIST, ctx, PlatformArgs(platform="fake")
    )
    listed = _item(session, clock)
    ctx = _ctx(session, clock, settings, notifier, listed, adapters)
    assert "not an enabled, connected platform" in _reject(
        RELIST, ctx, PlatformArgs(platform="kijiji")
    )


def test_relist_stops_when_the_old_post_cannot_be_taken_down(
    ctx, session, clock, notifier, item, adapters
):
    """The contract has no not-found error, so a timeout looks the same: reposting would leave
    two copies up, with buyers on the old one mapped to nothing."""
    listing = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["delist"] = PlatformError("timeout")

    result = _run(RELIST, ctx, PlatformArgs(platform="kijiji"))

    assert not result.ok and "nothing was reposted" in result.summary
    assert adapters["kijiji"].called("publish") == []
    assert listing.status == ListingStatus.LIVE and listing.remote_id == f"kijiji-old-{item.id}"
    assert listing.last_error == "could not take the old post down: timeout"
    assert len(_listings(session, item, "kijiji")) == 1
    assert RELIST.check(ctx, PlatformArgs(platform="kijiji")) == Allow()  # may try again


def test_relist_stops_on_an_auth_failure(ctx, session, clock, item, adapters):
    listing = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["delist"] = PlatformAuthError("logged out")
    with pytest.raises(PlatformAuthError):
        RELIST.execute(ctx, PlatformArgs(platform="kijiji"))
    assert adapters["kijiji"].called("publish") == []
    assert listing.status == ListingStatus.LIVE


def test_relist_publish_failure_records_the_delisting(ctx, session, clock, item, adapters):
    listing = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["publish"] = PlatformError("rate limited")

    result = _run(RELIST, ctx, PlatformArgs(platform="kijiji"))

    assert not result.ok and result.summary == (
        "Relisting on kijiji failed: rate limited. The old listing is down, so the item is not "
        "on kijiji now."
    )
    assert listing.status == ListingStatus.DELISTED and listing.delisted_at == clock.now()
    assert listing.last_error == "rate limited"
    assert RELIST.check(ctx, PlatformArgs(platform="kijiji")) == Allow()


# ---------------------------------------------------------------- a post gone from its platform


def test_relist_reposts_when_the_old_post_is_already_gone(
    ctx, session, clock, notifier, item, adapters
):
    """Removed or expired on the platform: nothing to take down, so the fresh copy goes up."""
    listing = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["delist"] = ListingGone("kijiji: listing not found")

    result = _run(RELIST, ctx, PlatformArgs(platform="kijiji"))

    assert result.ok and result.summary.startswith("Relisted on kijiji at $120.00")
    assert len(adapters["kijiji"].called("publish")) == 1
    assert listing.status == ListingStatus.LIVE and listing.remote_id == "kijiji-1"
    assert listing.last_error is None
    current, replaced = _listings(session, item, "kijiji")
    assert replaced.remote_id == f"kijiji-old-{item.id}"  # buyers there still map to the item
    assert notifier.of_kind(NoticeKind.ALERT) == []


def test_relist_of_a_gone_post_on_the_real_fake_market(session, clock, two_platforms, notifier):
    market = FakeMarketplace(clock, name="kijiji")
    item = _item(session, clock)
    ctx = _ctx(session, clock, two_platforms, notifier, item, {"kijiji": market})
    assert _run(PUBLISH, ctx, PlatformArgs(platform="kijiji")).ok
    [listing] = _listings(session, item, "kijiji")
    old = listing.remote_id
    market.remove_listing(old)
    clock.advance(hours=25)

    assert _run(RELIST, ctx, PlatformArgs(platform="kijiji")).ok

    assert listing.status == ListingStatus.LIVE and listing.remote_id != old
    assert market.listings[listing.remote_id].live


def test_update_price_marks_a_gone_post_delisted_and_still_reprices(
    ctx, session, clock, notifier, item, adapters
):
    fake = _listing(session, clock, item, "fake")
    kijiji = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["update"] = ListingGone("kijiji: listing not found")

    result = _run(PRICE, ctx, UpdatePriceArgs(price_cents=10_500))

    assert result.ok and result.data == {"errors": [], "gone": ["kijiji"]}
    assert "The kijiji post is gone from kijiji" in result.summary
    assert "Relist or publish on kijiji" in result.summary
    assert fake.price_cents == 10_500
    assert kijiji.status == ListingStatus.DELISTED and kijiji.delisted_at == clock.now()
    assert kijiji.last_error == "gone from kijiji: kijiji: listing not found"
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "The kijiji post is gone" and alert.item_id == item.id
    assert RELIST.check(ctx, PlatformArgs(platform="kijiji")) == Allow()
    assert PUBLISH.check(ctx, PlatformArgs(platform="kijiji")) == Allow()


def test_update_price_when_the_only_post_is_gone(ctx, session, clock, item, adapters):
    """Not a total failure: the price changes and the listing stops counting as live."""
    listing = _listing(session, clock, item, "fake")
    adapters["fake"].fail["update"] = ListingGone("fake: listing was delisted")

    result = _run(PRICE, ctx, UpdatePriceArgs(price_cents=10_500))

    assert result.ok and get_goal(session, item.id).list_price_cents == 10_500
    assert listing.status == ListingStatus.DELISTED


def test_a_gone_post_does_not_hide_a_total_failure(ctx, session, clock, item, adapters):
    _listing(session, clock, item, "fake")
    _listing(session, clock, item, "kijiji")
    adapters["fake"].fail["update"] = ListingGone("gone")
    adapters["kijiji"].fail["update"] = PlatformError("down")

    with pytest.raises(PlatformError, match="No listing could be updated. kijiji: down"):
        PRICE.execute(ctx, UpdatePriceArgs(price_cents=10_500))


def test_rewrite_reports_a_gone_post(ctx, session, clock, item, adapters):
    listing = _listing(session, clock, item, "kijiji")
    adapters["kijiji"].fail["update"] = ListingGone("not found")

    result = _run(REWRITE, ctx, RewriteListingArgs(title="White desk"))

    assert result.ok and result.data["gone"] == ["kijiji"]
    assert listing.status == ListingStatus.DELISTED


# ---------------------------------------------------------------- stored text is checked too


def _store_text(session, item: Item, *, title: str | None = None, description: str | None = None):
    if title is not None:
        item.title = title
    if description is not None:
        item.description = description
    session.commit()


def _raise_floor(session, item: Item, floor: int) -> None:
    goal = get_goal(session, item.id)
    goal.floor_price_cents = floor
    session.commit()


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Solid desk. Text 416-555-0199 for details.", "phone number"),
        ("Solid desk, $60 or best offer.", "below the floor of $95.00"),
        ("Solid desk. I can ship it anywhere.", "other than cash at pickup"),
        (f"Pickup at {ADDRESS_PLACEHOLDER}", "pickup address"),
    ],
)
def test_publish_rejects_stored_text_that_breaks_a_rule(ctx, session, item, description, expected):
    _store_text(session, item, description=description)

    reason = _reject(PUBLISH, ctx, PlatformArgs(platform="kijiji"))

    assert reason.startswith("The listing text cannot be posted as it is. In the description:")
    assert expected in reason and reason.endswith("Fix it with rewrite_listing first.")


def test_the_owner_is_told_to_fix_the_text_not_to_call_a_tool(
    session, clock, two_platforms, notifier, adapters, item
):
    _store_text(session, item, title="Desk, call 416-555-0199")
    human = _ctx(session, clock, two_platforms, notifier, item, adapters, actor="human")

    reason = _reject(PUBLISH, human, PlatformArgs(platform="kijiji"))

    assert "In the title:" in reason and reason.endswith("Fix the title or description first.")


def test_relist_rejects_text_under_a_floor_raised_since_it_was_written(ctx, session, clock, item):
    _store_text(session, item, description="Solid desk, $100 OBO.")
    _listing(session, clock, item, "kijiji")
    assert RELIST.check(ctx, PlatformArgs(platform="kijiji")) == Allow()

    _raise_floor(session, item, 11_000)

    reason = _reject(RELIST, ctx, PlatformArgs(platform="kijiji"))
    assert "mentions $100.00, below the floor of $110.00" in reason
    assert ctx.platforms["kijiji"].calls == []


def test_update_price_rejects_reposting_text_under_the_floor(ctx, session, clock, item, adapters):
    _store_text(session, item, description="Solid desk, $100 OBO.")
    _listing(session, clock, item, "kijiji")
    _raise_floor(session, item, 11_000)

    reason = _reject(PRICE, ctx, UpdatePriceArgs(price_cents=11_500))

    assert reason.startswith("Changing the price re-posts the listing text, which breaks a rule.")
    assert "In the description:" in reason and "below the floor of $110.00" in reason
    assert "Fix the text with rewrite_listing first, leaving prices out of it" in reason
    assert adapters["kijiji"].calls == []


def test_update_price_checks_firm_text_against_the_new_price(
    session, clock, two_platforms, notifier, adapters
):
    item = _item(session, clock, firmness=Firmness.FIRM)
    _store_text(session, item, description="Asking $120, firm.")
    _listing(session, clock, item, "kijiji")
    ctx = _ctx(session, clock, two_platforms, notifier, item, adapters)

    reason = _reject(PRICE, ctx, UpdatePriceArgs(price_cents=13_000))

    assert "mentions $120.00, below the firm price of $130.00" in reason


def test_update_price_without_a_live_listing_posts_no_text(ctx, session, item):
    _store_text(session, item, description="Solid desk, $60 or best offer.")
    assert PRICE.check(ctx, UpdatePriceArgs(price_cents=11_000)) == Allow()


def test_update_price_rechecks_the_text_under_the_write_lock(ctx, session, clock, item, adapters):
    """The owner raised the floor between check() and execute()."""
    _store_text(session, item, description="Solid desk, $100 OBO.")
    _listing(session, clock, item, "kijiji")
    args = UpdatePriceArgs(price_cents=11_500)
    assert PRICE.check(ctx, args) == Allow()
    get_goal(session, item.id).floor_price_cents = 10_500
    session.flush()

    with pytest.raises(LedgerConflict, match="below the floor of \\$105.00"):
        PRICE.execute(ctx, args)
    assert adapters["kijiji"].calls == []


def test_rewrite_of_one_field_checks_the_other_too(ctx, session, clock, item):
    _store_text(session, item, description="Solid desk, $100 OBO.")
    _listing(session, clock, item, "kijiji")
    _raise_floor(session, item, 11_000)

    reason = _reject(REWRITE, ctx, RewriteListingArgs(title="White desk"))
    assert reason.startswith("In the current description (rewrite it too): The listing text")

    fixed = RewriteListingArgs(title="White desk", description="Solid desk, light wear.")
    assert REWRITE.check(ctx, fixed) == Allow()
