"""control.service: every human override, through the real tools and ledger."""

import json
from datetime import timedelta

import pytest
from sqlmodel import Session, select

from marketplace_agent.control.service import ControlError, ControlService
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    ApprovalStatus,
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Memory,
    Message,
    PendingApproval,
    Photo,
    PriceChange,
    Thread,
    ThreadState,
    TrustRecord,
    TrustTier,
    WakeRequest,
)
from marketplace_agent.guard import handover, ledger
from marketplace_agent.guard.invariants import ADDRESS_PLACEHOLDER
from marketplace_agent.notify.base import NoticeKind
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.runner.takedown import TakedownState, takedown_state
from marketplace_agent.tools.registry import default_registry

LIST, FLOOR = 12_000, 9_500


@pytest.fixture
def market(clock) -> FakeMarketplace:
    return FakeMarketplace(clock)


@pytest.fixture
def control(engine, clock, settings, notifier, market) -> ControlService:
    return ControlService(engine, clock, settings, {"fake": market}, notifier, default_registry())


def _draft(session: Session, clock, *, title="IKEA desk", goal=True) -> int:
    now = clock.now()
    item = Item(hint="desk $120", title=title, description="Solid.", created_at=now, updated_at=now)
    session.add(item)
    session.flush()
    session.add(Photo(item_id=item.id, path="photos/1.jpg", width=640, height=480, sha256="ab"))
    if goal:
        set_goal(session, clock, item.id, list_price_cents=LIST, floor_price_cents=FLOOR)
    session.commit()
    return item.id


def _listed(session: Session, control: ControlService, clock) -> int:
    item_id = _draft(session, clock)
    [result] = control.confirm_and_publish(item_id)
    assert result.ok, result.summary
    return item_id


def _thread(session: Session, clock, item_id: int, handle="maya") -> Thread:
    buyer = Buyer(platform="fake", handle=handle, display_name=handle.title())
    session.add(buyer)
    session.flush()
    thread = Thread(
        platform="fake",
        remote_id=f"t-{handle}",
        item_id=item_id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    session.add(
        Message(
            thread_id=thread.id,
            direction=Direction.INBOUND,
            body="Would you take $110?",
            sent_at=clock.now(),
            handled=True,
            via="buyer",
        )
    )
    session.commit()
    return thread


def _deal(session: Session, clock, item_id: int, thread: Thread) -> int:
    accepted = ledger.accept(
        session,
        clock,
        item_id=item_id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=11_000,
        undo_minutes=10,
    )
    session.commit()
    return accepted.id


def _rows(session: Session, model) -> list:
    session.expire_all()
    return list(session.exec(select(model).order_by(model.id)).all())


def _wake_reasons(session: Session) -> list[str]:
    return [request.reason for request in _rows(session, WakeRequest)]


def _trust(session: Session, platform: str, action: str) -> TrustRecord:
    session.expire_all()
    query = select(TrustRecord).where(
        TrustRecord.platform == platform, TrustRecord.action == action
    )
    return session.exec(query).one()


def _park(session: Session, clock, item_id: int, tool: str, args: dict, platform="*") -> int:
    approval = PendingApproval(
        item_id=item_id,
        platform=platform,
        tool=tool,
        args_json=json.dumps(args, sort_keys=True),
        created_at=clock.now(),
    )
    session.add(approval)
    session.commit()
    return approval.id


# ---------------------------------------------------------------- confirm_and_publish


def test_confirm_and_publish_lists_the_draft_and_posts_it(session, control, clock, market):
    item_id = _draft(session, clock)

    [result] = control.confirm_and_publish(item_id)

    assert result.ok and "Published on fake at $120.00" in result.summary
    item = session.get(Item, item_id, populate_existing=True)
    [listing] = _rows(session, Listing)
    assert item.status == ItemStatus.LISTED
    assert (listing.status, listing.published_at) == (ListingStatus.LIVE, clock.now())
    assert market.listings[listing.remote_id].live
    assert _wake_reasons(session) == ["The owner confirmed the item for sale; posted on fake."]


def test_confirm_and_publish_reports_platforms_it_cannot_use(session, control, clock):
    item_id = _draft(session, clock)

    [result] = control.confirm_and_publish(item_id, ["kijiji"])

    assert not result.ok and "'kijiji' is not an enabled, connected platform" in result.summary
    assert session.get(Item, item_id, populate_existing=True).status == ItemStatus.LISTED
    assert _wake_reasons(session) == [
        "The owner confirmed the item for sale; posted on no platform yet."
    ]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"goal": False}, "has no goal"), ({"title": ""}, "has no title")],
)
def test_confirm_and_publish_needs_goal_and_title(session, control, clock, kwargs, error):
    item_id = _draft(session, clock, **kwargs)

    with pytest.raises(ControlError, match=error):
        control.confirm_and_publish(item_id)
    assert session.get(Item, item_id, populate_existing=True).status == ItemStatus.DRAFT


def test_confirm_and_publish_refuses_sold_and_missing_items(session, control, clock):
    item_id = _draft(session, clock)
    session.get(Item, item_id).status = ItemStatus.SOLD
    session.commit()

    with pytest.raises(ControlError, match="is sold"):
        control.confirm_and_publish(item_id)
    with pytest.raises(ControlError, match="no item with id 99"):
        control.confirm_and_publish(99)


# ---------------------------------------------------------------- undo


def test_undo_reopens_the_item_and_teaches_the_agent(session, control, clock):
    item_id = _listed(session, control, clock)
    thread = _thread(session, clock, item_id)
    deal = _deal(session, clock, item_id, thread)

    message = control.undo_commitment(deal)

    assert message == (
        f"Undid the $110.00 deal with Maya. Item {item_id} is listed again; the agent will re-plan."
    )
    session.expire_all()
    assert session.get(Item, item_id).status == ItemStatus.LISTED
    assert session.get(Thread, thread.id).state == ThreadState.NEGOTIATING
    undone = _rows(session, Commitment)[-1]
    assert (undone.kind, undone.ref_id, undone.actor) == (CommitmentKind.UNDONE, deal, "human")
    [memory] = _rows(session, Memory)
    assert (memory.scope.value, memory.item_id, memory.source) == ("item", item_id, "undo")
    assert "undid your $110.00 deal with Maya" in memory.lesson
    record = _trust(session, "*", "accept_offer")
    assert (record.corrected, record.tier) == (1, TrustTier.ASK)
    assert _wake_reasons(session)[-1] == (
        "The owner undid your $110.00 deal with Maya; the item is available again."
    )


def test_undo_after_the_window_is_refused(session, control, clock):
    item_id = _listed(session, control, clock)
    deal = _deal(session, clock, item_id, _thread(session, clock, item_id))
    clock.advance(minutes=11)

    with pytest.raises(ControlError, match=f"Too late to undo deal {deal}"):
        control.undo_commitment(deal)
    assert _rows(session, Memory) == []


def test_undo_refuses_unknown_and_closed_deals(session, control, clock):
    item_id = _listed(session, control, clock)
    deal = _deal(session, clock, item_id, _thread(session, clock, item_id))
    control.undo_commitment(deal)

    with pytest.raises(ControlError, match="already released, undone, or completed"):
        control.undo_commitment(deal)
    with pytest.raises(ControlError, match="no deal with id 999"):
        control.undo_commitment(999)


# ---------------------------------------------------------------- approvals


def test_approve_runs_the_parked_call_as_the_owner(session, control, clock, market):
    item_id = _draft(session, clock)
    session.get(Item, item_id).status = ItemStatus.LISTED
    session.commit()
    approval_id = _park(session, clock, item_id, "publish", {"platform": "fake"}, "fake")

    result = control.approve(approval_id)

    assert result.ok
    [listing] = _rows(session, Listing)
    assert listing.status == ListingStatus.LIVE
    approval = session.get(PendingApproval, approval_id, populate_existing=True)
    assert (approval.status, approval.decided_at) == (ApprovalStatus.APPROVED, clock.now())
    record = _trust(session, "fake", "publish")
    assert (record.executed, record.consecutive_clean) == (1, 1)
    assert _wake_reasons(session)[-1].startswith("The owner approved publish; Published on fake")


def test_approve_of_a_call_that_no_longer_passes_reports_why(session, control, clock):
    item_id = _listed(session, control, clock)
    approval_id = _park(session, clock, item_id, "update_price", {"price_cents": 5_000})

    result = control.approve(approval_id)

    assert not result.ok and "below the floor" in result.summary
    assert _rows(session, PriceChange) == []
    approval = session.get(PendingApproval, approval_id, populate_existing=True)
    assert approval.status == ApprovalStatus.APPROVED
    assert session.exec(select(TrustRecord)).all() == []
    assert "it could not run" in _wake_reasons(session)[-1]


def test_approve_twice_or_unknown_is_refused(session, control, clock):
    item_id = _listed(session, control, clock)
    approval_id = _park(session, clock, item_id, "update_price", {"price_cents": 11_000})
    control.approve(approval_id)

    with pytest.raises(ControlError, match="already approved"):
        control.approve(approval_id)
    with pytest.raises(ControlError, match="no approval with id 42"):
        control.approve(42)


def test_reject_records_a_correction_and_the_note(session, control, clock, settings):
    item_id = _listed(session, control, clock)
    approval_id = _park(session, clock, item_id, "update_price", {"price_cents": 10_000})

    control.reject(approval_id, f"Too soon,  we only listed it at {settings.pickup_address}")

    approval = session.get(PendingApproval, approval_id, populate_existing=True)
    assert approval.status == ApprovalStatus.REJECTED
    assert _trust(session, "*", "update_price").corrected == 1
    [memory] = _rows(session, Memory)
    assert memory.source == "correction"
    assert memory.lesson == (
        'The owner rejected update_price {"price_cents": 10000}: Too soon, we only listed it at '
        f"{ADDRESS_PLACEHOLDER}"
    )
    assert _wake_reasons(session)[-1].endswith(
        f"Their note: Too soon, we only listed it at {ADDRESS_PLACEHOLDER}"
    )


def test_reject_without_a_note_writes_no_memory(session, control, clock):
    item_id = _listed(session, control, clock)
    approval_id = _park(session, clock, item_id, "publish", {"platform": "fake"}, "fake")

    control.reject(approval_id)

    assert _rows(session, Memory) == []
    assert _trust(session, "fake", "publish").corrected == 1
    assert _wake_reasons(session)[-1] == 'The owner rejected publish {"platform": "fake"}.'


# ---------------------------------------------------------------- price and goal


def test_set_price_updates_goal_and_live_listings(session, control, clock, market):
    item_id = _listed(session, control, clock)

    result = control.set_price(item_id, 11_500)

    assert result.ok
    [listing] = _rows(session, Listing)
    assert market.listings[listing.remote_id].payload.price_cents == 11_500
    [change] = _rows(session, PriceChange)
    assert (change.old_cents, change.new_cents, change.actor) == (LIST, 11_500, "human")
    assert session.exec(select(TrustRecord)).all() == []
    assert _rows(session, Memory) == []
    assert _wake_reasons(session)[-1] == "The owner set the asking price to $115.00."


def test_set_price_soon_after_an_agent_change_is_a_correction(session, control, clock):
    item_id = _listed(session, control, clock)
    session.add(
        PriceChange(
            item_id=item_id,
            old_cents=LIST,
            new_cents=10_000,
            actor="agent",
            created_at=clock.now() - timedelta(hours=47),
        )
    )
    session.commit()

    control.set_price(item_id, 11_000)

    assert _trust(session, "*", "update_price").corrected == 1
    [memory] = _rows(session, Memory)
    assert (
        memory.source == "correction" and "overrode your price change to $100.00" in memory.lesson
    )


def test_set_price_long_after_an_agent_change_is_not_a_correction(session, control, clock):
    item_id = _listed(session, control, clock)
    session.add(
        PriceChange(
            item_id=item_id,
            old_cents=LIST,
            new_cents=10_000,
            actor="agent",
            created_at=clock.now() - timedelta(hours=49),
        )
    )
    session.commit()

    control.set_price(item_id, 11_000)

    assert session.exec(select(TrustRecord)).all() == []


def test_set_price_below_floor_changes_nothing(session, control, clock):
    item_id = _listed(session, control, clock)
    before = _wake_reasons(session)

    result = control.set_price(item_id, 5_000)

    assert not result.ok and "below the floor" in result.summary
    assert _rows(session, PriceChange) == []
    assert _wake_reasons(session) == before


def test_set_floor_and_firmness(session, control, clock):
    item_id = _listed(session, control, clock)

    control.set_floor(item_id, 10_000)
    control.set_firmness(item_id, Firmness.FIRM)

    [goal] = _rows(session, Goal)
    assert (goal.floor_price_cents, goal.firmness) == (10_000, Firmness.FIRM)
    assert _wake_reasons(session)[-2:] == [
        "The owner set the floor to $100.00.",
        "The owner set the price firmness to firm.",
    ]


@pytest.mark.parametrize(("floor", "error"), [(0, "more than"), (12_001, "cannot be above")])
def test_set_floor_rejects_impossible_floors(session, control, clock, floor, error):
    item_id = _listed(session, control, clock)

    with pytest.raises(ControlError, match=error):
        control.set_floor(item_id, floor)


def test_goal_changes_need_a_goal(session, control, clock):
    item_id = _draft(session, clock, goal=False)

    with pytest.raises(ControlError, match="has no goal"):
        control.set_floor(item_id, 5_000)
    with pytest.raises(ControlError, match="has no goal"):
        control.set_firmness(item_id, Firmness.FIRM)


def test_set_goal_on_a_listed_item_goes_through_the_tools(session, control, clock, market):
    item_id = _listed(session, control, clock)

    goal = control.set_goal(
        item_id,
        list_price_cents=15_000,
        floor_price_cents=13_000,
        target_price_cents=14_000,
        deadline_days=7,
        firmness=Firmness.FLEXIBLE,
    )

    [stored] = _rows(session, Goal)
    assert (stored.list_price_cents, stored.floor_price_cents, stored.target_price_cents) == (
        15_000,
        13_000,
        14_000,
    )
    assert stored.firmness == Firmness.FLEXIBLE and goal.list_price_cents == 15_000
    assert stored.deadline.replace(tzinfo=clock.now().tzinfo) == clock.now() + timedelta(days=7)
    [listing] = _rows(session, Listing)
    assert market.listings[listing.remote_id].payload.price_cents == 15_000
    [change] = _rows(session, PriceChange)
    assert (change.old_cents, change.new_cents, change.actor) == (LIST, 15_000, "human")
    assert _wake_reasons(session)[-1] == (
        "The owner set the goal: list $150.00, target $140.00, floor $130.00, flexible."
    )


def test_set_goal_lowering_price_and_floor_together(session, control, clock, market):
    item_id = _listed(session, control, clock)

    control.set_goal(item_id, list_price_cents=8_000, floor_price_cents=6_000)

    [stored] = _rows(session, Goal)
    assert (stored.list_price_cents, stored.floor_price_cents) == (8_000, 6_000)


def test_set_goal_soon_after_an_agent_reprice_is_a_correction(session, control, clock):
    item_id = _listed(session, control, clock)
    session.add(
        PriceChange(
            item_id=item_id,
            old_cents=LIST,
            new_cents=10_000,
            actor="agent",
            created_at=clock.now() - timedelta(hours=1),
        )
    )
    session.commit()

    control.set_goal(item_id, list_price_cents=11_000, floor_price_cents=FLOOR)

    assert _trust(session, "*", "update_price").corrected == 1


def test_set_goal_that_cannot_reprice_changes_nothing(session, control, clock):
    """A promised item cannot be repriced, so the floor and firmness stay as they were too."""
    item_id = _listed(session, control, clock)
    _deal(session, clock, item_id, _thread(session, clock, item_id))
    before = _wake_reasons(session)

    with pytest.raises(ControlError, match="promised to a buyer"):
        control.set_goal(
            item_id, list_price_cents=10_000, floor_price_cents=8_000, firmness=Firmness.FIRM
        )

    [stored] = _rows(session, Goal)
    assert (stored.list_price_cents, stored.floor_price_cents) == (LIST, FLOOR)
    assert stored.firmness == Firmness.SLIGHT and _rows(session, PriceChange) == []
    assert _wake_reasons(session) == before


def test_set_goal_that_no_listing_takes_puts_the_whole_goal_back(session, control, clock, market):
    """The new goal is committed before the price push; when every listing refuses the price,
    the floor and firmness go back too."""
    item_id = _listed(session, control, clock)
    market.fail_next["update"] = PlatformError("down")
    before = _wake_reasons(session)

    with pytest.raises(ControlError, match="could not be changed, so nothing was"):
        control.set_goal(
            item_id, list_price_cents=10_000, floor_price_cents=8_000, firmness=Firmness.FIRM
        )

    session.expire_all()
    [stored] = _rows(session, Goal)
    assert (stored.list_price_cents, stored.floor_price_cents) == (LIST, FLOOR)
    assert stored.firmness == Firmness.SLIGHT and _rows(session, PriceChange) == []
    assert _wake_reasons(session) == before


def test_set_goal_on_a_promised_item_may_keep_the_price(session, control, clock):
    item_id = _listed(session, control, clock)
    _deal(session, clock, item_id, _thread(session, clock, item_id))

    control.set_goal(item_id, list_price_cents=LIST, floor_price_cents=10_000)

    assert _rows(session, Goal)[0].floor_price_cents == 10_000
    assert _wake_reasons(session)[-1].startswith("The owner set the goal: list $120.00")


@pytest.mark.parametrize(
    ("price", "floor", "error"),
    [(10_000, 11_000, "floor price cannot be above the list price"), (0, 0, "greater than zero")],
)
def test_set_goal_rejects_impossible_goals(session, control, clock, price, floor, error):
    item_id = _listed(session, control, clock)

    with pytest.raises(ControlError, match=error):
        control.set_goal(item_id, list_price_cents=price, floor_price_cents=floor)


def test_set_goal_on_a_draft_writes_the_goal_only(session, control, clock):
    item_id = _draft(session, clock, goal=False)

    control.set_goal(item_id, list_price_cents=9_000, floor_price_cents=7_000)

    [stored] = _rows(session, Goal)
    assert (stored.list_price_cents, stored.floor_price_cents) == (9_000, 7_000)
    assert _wake_reasons(session) == [] and _rows(session, PriceChange) == []


def test_raising_the_floor_under_the_listing_text_tells_the_agent(session, control, clock):
    item_id = _draft(session, clock)
    session.get(Item, item_id).description = "Solid. $100 OBO."
    session.commit()
    [result] = control.confirm_and_publish(item_id)
    assert result.ok

    control.set_floor(item_id, 11_000)

    reason = _wake_reasons(session)[-1]
    assert reason.startswith("The owner set the floor to $110.00. The listing text now breaks")
    assert "below the floor of $110.00" in reason and "rewrite_listing" in reason
    blocked = control.set_price(item_id, 11_500)
    assert not blocked.ok and "re-posts the listing text" in blocked.summary


def test_mark_listing_gone_lets_the_agent_post_again(session, control, clock):
    item_id = _listed(session, control, clock)

    message = control.mark_listing_gone(item_id)

    [listing] = _rows(session, Listing)
    assert listing.status == ListingStatus.DELISTED
    assert listing.last_error == "gone from fake: the owner reported it gone"
    assert "Marked the fake listing of item" in message
    assert "Relist or publish on fake" in _wake_reasons(session)[-1]
    with pytest.raises(ControlError, match="has no live listing"):
        control.mark_listing_gone(item_id)


def test_mark_listing_gone_needs_the_platform_when_live_on_several(
    engine, clock, settings, notifier, session
):
    both = settings.model_copy(update={"enabled_platforms": ["fake", "kijiji"]})
    markets = {"fake": FakeMarketplace(clock), "kijiji": FakeMarketplace(clock, name="kijiji")}
    control = ControlService(engine, clock, both, markets, notifier, default_registry())
    item_id = _draft(session, clock)
    assert all(result.ok for result in control.confirm_and_publish(item_id))

    with pytest.raises(ControlError, match="live on fake, kijiji. Say which"):
        control.mark_listing_gone(item_id)
    with pytest.raises(ControlError, match="no live listing on facebook"):
        control.mark_listing_gone(item_id, "facebook")
    control.mark_listing_gone(item_id, "kijiji")

    assert [lst.status for lst in _rows(session, Listing)] == [
        ListingStatus.LIVE,
        ListingStatus.DELISTED,
    ]


# ---------------------------------------------------------------- pause, resume


def test_pause_and_resume(session, control, clock):
    item_id = _listed(session, control, clock)
    before = len(_wake_reasons(session))

    control.pause(item_id)
    assert session.get(Item, item_id, populate_existing=True).paused
    assert len(_wake_reasons(session)) == before

    control.resume(item_id)
    control.resume(item_id)

    assert not session.get(Item, item_id, populate_existing=True).paused
    assert _wake_reasons(session)[before:] == [
        "The owner resumed the agent on this item after a pause."
    ]


# ---------------------------------------------------------------- sold, withdrawn


def test_mark_sold_completes_the_deal_and_delists(session, control, clock, market):
    item_id = _listed(session, control, clock)
    thread = _thread(session, clock, item_id)
    deal = _deal(session, clock, item_id, thread)

    control.mark_sold(item_id)

    session.expire_all()
    done = _rows(session, Commitment)[-1]
    assert (done.kind, done.ref_id, done.actor) == (CommitmentKind.COMPLETED, deal, "human")
    assert session.get(Item, item_id).status == ItemStatus.SOLD
    assert session.get(Buyer, thread.buyer_id).completed_count == 1
    [listing] = _rows(session, Listing)
    assert (listing.status, listing.delisted_at) == (ListingStatus.DELISTED, clock.now())
    assert not market.listings[listing.remote_id].live


def _booked(session, control, clock) -> tuple[int, Thread, int]:
    """A listed item with Maya's deal and a pickup tomorrow 18:00-19:00 local."""
    item_id = _listed(session, control, clock)
    thread = _thread(session, clock, item_id)
    deal = _deal(session, clock, item_id, thread)
    start = clock.now() + timedelta(days=1, hours=6)
    ledger.book_pickup(
        session, clock, accepted_id=deal, start=start, end=start + timedelta(hours=1)
    )
    session.commit()
    return item_id, thread, deal


def test_mark_no_show_releases_the_deal_on_the_owners_word(session, control, clock):
    item_id, thread, deal = _booked(session, control, clock)
    with pytest.raises(ControlError, match="has not started yet"):
        control.mark_no_show(item_id)
    clock.advance(days=1, hours=6, minutes=10)  # ten minutes into the pickup

    message = control.mark_no_show(item_id)

    assert message.startswith("Released the deal with Maya as a no-show")
    session.expire_all()
    released = _rows(session, Commitment)[-1]
    assert (released.kind, released.ref_id, released.actor) == (
        CommitmentKind.RELEASED,
        deal,
        "human",
    )
    assert released.reason.startswith("no-show")
    assert session.get(Item, item_id).status == ItemStatus.LISTED
    assert session.get(Buyer, thread.buyer_id).no_show_count == 1
    assert "did not come" in _wake_reasons(session)[-1]
    with pytest.raises(ControlError, match="no active deal"):
        control.mark_no_show(item_id)


def test_mark_no_show_needs_a_booked_pickup(session, control, clock):
    item_id = _listed(session, control, clock)
    _deal(session, clock, item_id, _thread(session, clock, item_id))
    with pytest.raises(ControlError, match="no booked pickup"):
        control.mark_no_show(item_id)


def test_the_owners_answer_resolves_the_pickup_check(session, control, clock, settings, notifier):
    item_id, _, _ = _booked(session, control, clock)
    clock.advance(days=1, hours=7)  # the pickup is over
    handover.ask_owner(session, clock, settings, notifier)
    session.commit()
    [check] = [e for e in _rows(session, Escalation) if e.reason.startswith("Pickup check")]
    assert check.resolved_at is None

    control.mark_sold(item_id, 11_000)

    session.expire_all()
    check = session.get(Escalation, check.id)
    assert (check.resolved_at, check.resolution) == (clock.now(), "The owner marked it sold.")


def test_the_owner_is_asked_once_per_pickup_even_after_answering_some_other_way(
    session, control, clock, settings, notifier
):
    item_id, _, _ = _booked(session, control, clock)
    clock.advance(days=1, hours=7)
    assert handover.ask_owner(session, clock, settings, notifier) == 1
    session.commit()
    [check] = [e for e in _rows(session, Escalation) if e.reason.startswith("Pickup check")]
    control.resolve_escalation(check.id, "She is coming tomorrow instead")

    assert handover.ask_owner(session, clock, settings, notifier) == 0


def test_mark_sold_without_a_deal_keeps_the_price(session, control, clock):
    item_id = _listed(session, control, clock)

    control.mark_sold(item_id, 10_500)

    assert session.get(Item, item_id, populate_existing=True).status == ItemStatus.SOLD
    [memory] = _rows(session, Memory)
    assert (memory.scope.value, memory.source) == ("global", "outcome")
    assert memory.lesson.startswith("Sold [IKEA desk] for $105 on fake")
    with pytest.raises(ControlError, match="already sold"):
        control.mark_sold(item_id)


def test_mark_sold_alerts_when_a_listing_will_not_come_down(
    session, control, clock, market, notifier
):
    item_id = _listed(session, control, clock)
    market.fail_next["delist"] = PlatformError("timeout")

    control.mark_sold(item_id)

    [listing] = _rows(session, Listing)
    assert listing.status == ListingStatus.LIVE
    assert takedown_state(listing) == TakedownState(0, clock.now(), "timeout", alerted=True)
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert "Take down the fake listing" in alert.title
    assert session.get(Item, item_id, populate_existing=True).status == ItemStatus.SOLD


def test_mark_sold_treats_a_post_already_gone_as_down(session, control, clock, market, notifier):
    item_id = _listed(session, control, clock)
    market.remove_listing(_rows(session, Listing)[0].remote_id)

    control.mark_sold(item_id)

    [listing] = _rows(session, Listing)
    assert listing.status == ListingStatus.DELISTED
    assert listing.last_error.startswith("gone from fake")
    assert notifier.of_kind(NoticeKind.ALERT) == []


def test_withdraw_releases_the_deal_and_tells_the_owner(session, control, clock, market, notifier):
    item_id = _listed(session, control, clock)
    deal = _deal(session, clock, item_id, _thread(session, clock, item_id))

    control.withdraw(item_id)

    released = _rows(session, Commitment)[-1]
    assert (released.kind, released.ref_id, released.actor) == (
        CommitmentKind.RELEASED,
        deal,
        "human",
    )
    assert session.get(Item, item_id, populate_existing=True).status == ItemStatus.WITHDRAWN
    [listing] = _rows(session, Listing)
    assert listing.status == ListingStatus.DELISTED
    [alert] = notifier.of_kind(NoticeKind.ALERT)
    assert alert.title == "Tell Maya the deal is off"
    with pytest.raises(ControlError, match="already withdrawn"):
        control.withdraw(item_id)


# ---------------------------------------------------------------- escalations, replies


def test_resolve_escalation_wakes_the_agent_with_the_answer(session, control, clock):
    item_id = _listed(session, control, clock)
    escalation = Escalation(item_id=item_id, reason="Can it ship?", created_at=clock.now())
    session.add(escalation)
    session.commit()

    control.resolve_escalation(escalation.id, "No shipping,   pickup only.")

    session.refresh(escalation)
    assert (escalation.resolved_at, escalation.resolution) == (
        clock.now(),
        "No shipping, pickup only.",
    )
    assert _wake_reasons(session)[-1] == (
        f"The owner answered escalation #{escalation.id}: No shipping, pickup only."
    )
    with pytest.raises(ControlError, match="already resolved"):
        control.resolve_escalation(escalation.id, "again")
    with pytest.raises(ControlError, match="no escalation"):
        control.resolve_escalation(999, "x")


def test_reply_as_human_may_share_the_address_but_not_contact_info(
    session, control, clock, market, settings
):
    item_id = _listed(session, control, clock)
    thread = _thread(session, clock, item_id)
    market_thread = market.add_buyer("scripted", _rows(session, Listing)[0].remote_id)
    thread.remote_id = market_thread
    session.add(thread)
    session.commit()

    sent = control.reply_as_human(thread.id, f"Come by {ADDRESS_PLACEHOLDER} at 6.")
    blocked = control.reply_as_human(thread.id, "Text me at 416-555-0199")

    assert sent.ok and not blocked.ok
    assert market.sent[-1][1] == f"Come by {settings.pickup_address} at 6."
    outbound = [m for m in _rows(session, Message) if m.direction == Direction.OUTBOUND]
    assert [(m.body, m.via) for m in outbound] == [
        (f"Come by {ADDRESS_PLACEHOLDER} at 6.", "human")
    ]
    assert _wake_reasons(session)[-1] == f"The owner replied in thread {thread.id} themselves."


def test_reply_as_human_needs_a_known_linked_thread(session, control, clock):
    buyer = Buyer(platform="fake", handle="x")
    session.add(buyer)
    session.flush()
    orphan = Thread(platform="fake", remote_id="t9", buyer_id=buyer.id, created_at=clock.now())
    session.add(orphan)
    session.commit()

    with pytest.raises(ControlError, match="not linked"):
        control.reply_as_human(orphan.id, "hi")
    with pytest.raises(ControlError, match="no conversation"):
        control.reply_as_human(999, "hi")
