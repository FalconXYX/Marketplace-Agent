"""notify.discord_views: cards for every notice, buttons through a real ControlService, digest.

FakeClock starts Thu 2026-10-01 16:00 UTC, 12:00 in Toronto. Thursday pickups run 17:00-20:00
local (21:00-24:00 UTC).
"""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from marketplace_agent.ai.factory import build_router
from marketplace_agent.control.service import ControlService
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    AgentRun,
    ApprovalStatus,
    Buyer,
    Direction,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Listing,
    Memory,
    Message,
    PendingApproval,
    Photo,
    QuotaUsage,
    Thread,
    ThreadState,
    TrustRecord,
)
from marketplace_agent.guard import handover
from marketplace_agent.guard.validate import validate
from marketplace_agent.notify import discord_views as views
from marketplace_agent.notify.base import Notice, NoticeKind
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.tools.base import Allow, ToolContext, ToolResult
from marketplace_agent.tools.registry import default_registry

OWNER, STRANGER = 4242, 777
LIST, FLOOR = 12_000, 9_500
UNDO_UNTIL = "Thu Oct 1 12:10"  # 16:00 UTC + 10 minutes, in Toronto
PICKUP = "2026-10-01T18:00"
PICKUP_TEXT = "Thu Oct 1 18:00 to 19:00"
AFTER_PICKUP = datetime(2026, 10, 1, 23, 30, tzinfo=UTC)


@pytest.fixture
def market(clock) -> FakeMarketplace:
    return FakeMarketplace(clock)


@pytest.fixture
def control(engine, clock, settings, notifier, market) -> ControlService:
    return ControlService(engine, clock, settings, {"fake": market}, notifier, default_registry())


@pytest.fixture
def world(session, control, clock, market) -> "World":
    return World(session, control, clock, market)


class World:
    """A listed item on the fake marketplace, buyers, and the agent's tools run as the agent."""

    def __init__(self, session: Session, control: ControlService, clock, market) -> None:
        self.session = session
        self.control = control
        self.clock = clock
        self.market = market

    def listed(self, title: str = "IKEA desk") -> int:
        now = self.clock.now()
        item = Item(
            hint="desk $120", title=title, description="Solid.", created_at=now, updated_at=now
        )
        self.session.add(item)
        self.session.flush()
        self.session.add(Photo(item_id=item.id, path="photos/1.jpg", width=8, height=8, sha256="a"))
        set_goal(self.session, self.clock, item.id, list_price_cents=LIST, floor_price_cents=FLOOR)
        self.session.commit()
        [result] = self.control.confirm_and_publish(item.id)
        assert result.ok, result.summary
        return item.id

    def thread(self, item_id: int, name: str = "Maya", says: str = "Would you take $110?") -> int:
        listing = self.session.exec(select(Listing).where(Listing.item_id == item_id)).first()
        remote_id = self.market.add_buyer("scripted", listing.remote_id)
        buyer = Buyer(platform="fake", handle=name.lower(), display_name=name)
        self.session.add(buyer)
        self.session.flush()
        thread = Thread(
            platform="fake",
            remote_id=remote_id,
            item_id=item_id,
            buyer_id=buyer.id,
            state=ThreadState.NEGOTIATING,
            last_offer_cents=11_000,
            created_at=self.clock.now(),
            last_inbound_at=self.clock.now(),
        )
        self.session.add(thread)
        self.session.flush()
        self.session.add(
            Message(
                thread_id=thread.id,
                direction=Direction.INBOUND,
                body=says,
                sent_at=self.clock.now(),
                handled=True,
                via="buyer",
            )
        )
        self.session.commit()
        return thread.id

    def tool(self, item_id: int, name: str, **args) -> ToolResult:
        control = self.control
        tool = default_registry().get(name)
        parsed = tool.parse_args(args)
        ctx = ToolContext(
            session=self.session,
            clock=self.clock,
            settings=control.settings,
            platforms=control.platforms,
            notifier=control.notifier,
            item_id=item_id,
        )
        verdict = validate(ctx, tool, parsed)
        assert isinstance(verdict, Allow), verdict
        result = tool.execute(ctx, parsed)
        assert result.ok, result.summary
        self.session.commit()
        return result

    def deal(self, item_id: int, thread_id: int, *, pickup: bool = False) -> int:
        result = self.tool(item_id, "accept_offer", thread_id=thread_id, price_cents=11_000)
        if pickup:
            self.tool(item_id, "book_pickup", thread_id=thread_id, start=PICKUP)
        return result.data["commitment_id"]

    def pickup_check(self) -> Escalation:
        """The deal's pickup is over and the runner asks the owner whether the buyer came."""
        self.clock.set(AFTER_PICKUP)
        handover.ask_owner(self.session, self.clock, self.control.settings, self.control.notifier)
        self.session.commit()
        return self.session.exec(select(Escalation)).one()

    def park(self, item_id: int, tool: str, args: dict, why: str = "") -> Notice:
        """What the agent loop does when a call needs the owner's approval."""
        args_json = json.dumps(args, sort_keys=True)
        approval = PendingApproval(
            item_id=item_id,
            tool=tool,
            args_json=args_json,
            rationale=why,
            created_at=self.clock.now(),
        )
        self.session.add(approval)
        self.session.commit()
        return Notice(
            kind=NoticeKind.APPROVAL,
            title=f"Approve {tool}?",
            body=f"{args_json}\n{why}".strip(),
            item_id=item_id,
            ref_id=approval.id,
        )

    def card(self, notice: Notice) -> views.Card:
        self.session.expire_all()
        return views.render_notice(self.session, self.clock, self.control.settings, notice)

    def act(self, action: str, ref: int, form: dict | None = None, user: int = OWNER):
        button = views.custom_id(action, ref)
        response = views.handle_action(self.control, button, user, OWNER, form)
        self.session.expire_all()
        return response

    def get(self, model, row_id: int):
        return self.session.get(model, row_id, populate_existing=True)


def _texts(card: views.Card) -> str:
    parts = [card.title, card.description]
    parts += [f"{field.name}: {field.value}" for field in card.fields]
    parts += [button.label for button in card.buttons]
    return "\n".join(parts)


def _buttons(card: views.Card) -> list[tuple[str, str, str, bool]]:
    return [(b.label, b.custom_id, b.style, b.disabled) for b in card.buttons]


def _field(card: views.Card, name: str) -> str:
    [value] = [field.value for field in card.fields if field.name == name]
    return value


# ---------------------------------------------------------------- custom ids and the owner


@pytest.mark.parametrize("action", sorted(views.ACTIONS))
def test_custom_id_round_trips_for_every_action(action):
    made = views.custom_id(action, 31)

    assert made == f"ma:{action}:31"
    assert views.parse_custom_id(made) == (action, 31)
    assert views.parse_custom_id(views.custom_id(action, 2**63 - 1)) == (action, 2**63 - 1)


@pytest.mark.parametrize(
    "value",
    ["ma:undo:0", "ma:undo:-1", "ma:undo:abc", "ma:nope:1", "xx:undo:1", "ma:undo:1:2", "ma:undo:"],
)
def test_parse_custom_id_rejects_anything_else(value):
    with pytest.raises(ValueError):
        views.parse_custom_id(value)


def test_custom_id_refuses_unknown_actions_and_ids():
    with pytest.raises(ValueError):
        views.custom_id("nope", 1)
    with pytest.raises(ValueError):
        views.custom_id("undo", 0)


def test_only_the_owner_may_act(world):
    item_id = world.listed()

    stranger = world.act("pause", item_id, user=STRANGER)
    nobody = views.handle_action(world.control, views.custom_id("pause", item_id), OWNER, None)

    assert stranger == views.ActionResponse(views.NOT_OWNER, ephemeral=True)
    assert nobody == views.ActionResponse(views.NO_OWNER, ephemeral=True)
    assert not world.get(Item, item_id).paused
    assert views.check_owner(OWNER, OWNER) is None


def test_an_unknown_button_is_refused_politely(world):
    response = views.handle_action(world.control, "ma:launch:1", OWNER, OWNER)

    assert response == views.ActionResponse(views.UNKNOWN_BUTTON)


def test_modals_are_for_reject_reply_and_resolve_only():
    reject = views.modal_for("ma:reject:4")
    reply = views.modal_for("ma:reply:9")

    assert reject.title == "Reject #4" and [f.key for f in reject.fields] == ["note"]
    assert not reject.fields[0].required
    assert reply.title == "Reply as you #9" and reply.fields[0].required
    assert "[PICKUP_ADDRESS]" in reply.fields[0].placeholder
    assert views.modal_for("ma:resolve:2").fields[0].key == "answer"
    assert views.modal_for("ma:approve:4") is None
    assert views.modal_for("garbage") is None
    labels = [f.label for spec in views.MODALS.values() for f in spec.fields]
    assert all(len(label) <= 45 for label in labels)


@pytest.mark.parametrize(
    ("text", "cents"), [("120", 12_000), ("120.5", 12_050), ("$1,200.99", 120_099), (" 95 ", 9_500)]
)
def test_parse_dollars(text, cents):
    assert views.parse_dollars(text) == cents


@pytest.mark.parametrize("text", ["", "abc", "12.345", "0", "-5", "99999999999"])
def test_parse_dollars_refuses_bad_amounts(text):
    with pytest.raises(views.InvalidInput):
        views.parse_dollars(text)


# ---------------------------------------------------------------- COMMITMENT


def test_commitment_card_shows_deal_buyer_and_undo_deadline(world, notifier, settings):
    item_id = world.listed()
    deal = world.deal(item_id, world.thread(item_id))
    [notice] = notifier.of_kind(NoticeKind.COMMITMENT)

    card = world.card(notice)

    assert card.title == "Deal: IKEA desk"
    assert card.description == (
        f"Deal #{deal}: promised to Maya for $110.00.\nUndo until {UNDO_UNTIL}"
    )  # the number /undo asks for
    assert card.color == views.COLORS[NoticeKind.COMMITMENT]
    assert _field(card, "Pickup") == "Not booked yet"
    listing = world.session.exec(select(Listing)).one()
    assert _field(card, "Item") == (
        f"[#{item_id} IKEA desk](http://127.0.0.1:8765/items/{item_id})\nfake: {listing.url}"
    )
    assert _buttons(card) == [("Undo", f"ma:undo:{deal}", "danger", False)]


def test_commitment_card_shows_a_booked_pickup(world, notifier):
    item_id = world.listed()
    world.deal(item_id, world.thread(item_id), pickup=True)

    card = world.card(notifier.of_kind(NoticeKind.COMMITMENT)[0])

    assert _field(card, "Pickup") == PICKUP_TEXT


def test_commitment_card_after_the_undo_window_has_its_button_off(world, notifier, clock):
    item_id = world.listed()
    world.deal(item_id, world.thread(item_id))
    clock.advance(minutes=11)

    card = world.card(notifier.of_kind(NoticeKind.COMMITMENT)[0])

    assert f"Undo closed at {UNDO_UNTIL}" in card.description
    assert [b.disabled for b in card.buttons] == [True]


def test_undo_button_undoes_the_deal_and_switches_off(world, notifier):
    item_id = world.listed()
    deal = world.deal(item_id, world.thread(item_id))

    response = world.act("undo", deal)

    assert response == views.ActionResponse(
        f"Undid the $110.00 deal with Maya. Item {item_id} is listed again; the agent will "
        "re-plan.",
        ephemeral=False,
        disable_buttons=True,
    )
    assert world.get(Item, item_id).status == ItemStatus.LISTED
    card = world.card(notifier.of_kind(NoticeKind.COMMITMENT)[0])
    assert "This deal is closed." in card.description
    assert [b.disabled for b in card.buttons] == [True]
    again = world.act("undo", deal)
    assert again.ephemeral and not again.disable_buttons
    assert "already released, undone, or completed" in again.text


# ---------------------------------------------------------------- pickup check


def test_pickup_check_card_has_picked_up_and_no_show(world, notifier):
    item_id = world.listed()
    world.deal(item_id, world.thread(item_id), pickup=True)
    escalation = world.pickup_check()
    [notice] = notifier.of_kind(NoticeKind.ESCALATION)

    card = world.card(notice)

    assert notice.ref_id == escalation.id
    assert card.title == "Did Maya pick up IKEA desk?"
    assert card.description == (
        "Tap Picked up if Maya took it, or No-show if they did not come.\n"
        "Without an answer, the agent may treat them as a no-show from Fri Oct 2 01:00."
    )
    assert _field(card, "Deal") == "$110.00 with Maya"
    assert _field(card, "Pickup") == PICKUP_TEXT
    assert _buttons(card) == [
        ("Picked up", f"ma:pickedup:{escalation.id}", "success", False),
        ("No-show", f"ma:noshow:{escalation.id}", "danger", False),
    ]
    assert "agent sold" not in _texts(card)


def test_picked_up_marks_the_item_sold(world, notifier, market):
    item_id = world.listed()
    world.deal(item_id, world.thread(item_id), pickup=True)
    escalation = world.pickup_check()

    response = world.act("pickedup", escalation.id)

    assert response == views.ActionResponse(
        f"Marked item #{item_id} sold and took its listings down.",
        ephemeral=False,
        disable_buttons=True,
    )
    assert world.get(Item, item_id).status == ItemStatus.SOLD
    assert world.get(Escalation, escalation.id).resolved_at is not None
    assert not any(listing.live for listing in market.listings.values())
    card = world.card(notifier.of_kind(NoticeKind.ESCALATION)[0])
    assert all(button.disabled for button in card.buttons)


def test_no_show_releases_the_deal(world):
    item_id = world.listed()
    world.deal(item_id, world.thread(item_id), pickup=True)
    escalation = world.pickup_check()

    response = world.act("noshow", escalation.id)

    assert response.disable_buttons and not response.ephemeral
    assert response.text.startswith("Released the deal with Maya as a no-show.")
    assert world.get(Item, item_id).status == ItemStatus.LISTED
    assert world.session.exec(select(Buyer)).one().no_show_count == 1
    assert world.get(Escalation, escalation.id).resolved_at is not None


def _owner_answers_no_show(world, item_id: int, _thread_id: int) -> None:
    world.control.mark_no_show(item_id)  # on the web or with `agent noshow`


def _agent_releases_no_show(world, item_id: int, thread_id: int) -> None:
    world.clock.set(datetime(2026, 10, 2, 5, 30, tzinfo=UTC))  # past no_show_from
    world.tool(item_id, "release_commitment", thread_id=thread_id, reason="No-show", no_show=True)


def _check_card_then_next_deal(world, notifier, first_answer) -> tuple[int, views.Card]:
    """Aisha's pickup check is posted, answered elsewhere, and Noah's deal follows for Fri
    18:00 local. The posted card is never edited, so its buttons stay as they were."""
    item_id = world.listed()
    aisha = world.thread(item_id, "Aisha")
    world.deal(item_id, aisha, pickup=True)
    world.pickup_check()
    posted = world.card(notifier.of_kind(NoticeKind.ESCALATION)[0])
    first_answer(world, item_id, aisha)
    handover.ask_owner(world.session, world.clock, world.control.settings, world.control.notifier)
    world.session.commit()
    noah = world.thread(item_id, "Noah")
    world.tool(item_id, "accept_offer", thread_id=noah, price_cents=11_000)
    world.tool(item_id, "book_pickup", thread_id=noah, start="2026-10-02T18:00")
    return item_id, posted


def _press(world, card: views.Card, label: str) -> views.ActionResponse:
    [button] = [b for b in card.buttons if b.label == label]
    assert not button.disabled
    response = views.handle_action(world.control, button.custom_id, OWNER, OWNER)
    world.session.expire_all()
    return response


@pytest.mark.parametrize("first_answer", [_owner_answers_no_show, _agent_releases_no_show])
def test_an_old_no_show_button_never_releases_a_later_deal(world, notifier, first_answer):
    item_id, posted = _check_card_then_next_deal(world, notifier, first_answer)
    world.clock.set(datetime(2026, 10, 2, 22, 15, tzinfo=UTC))  # Noah's pickup is under way

    response = _press(world, posted, "No-show")

    assert response.ephemeral and response.disable_buttons
    assert "already answered" in response.text and "Nothing was changed." in response.text
    noah = world.session.exec(select(Buyer).where(Buyer.display_name == "Noah")).one()
    assert noah.no_show_count == 0
    assert world.get(Item, item_id).status == ItemStatus.COMMITTED


@pytest.mark.parametrize("first_answer", [_owner_answers_no_show, _agent_releases_no_show])
def test_an_old_picked_up_button_never_completes_a_later_deal(
    world, notifier, market, first_answer
):
    item_id, posted = _check_card_then_next_deal(world, notifier, first_answer)

    response = _press(world, posted, "Picked up")

    assert response.ephemeral and response.disable_buttons
    assert "already answered" in response.text
    assert world.get(Item, item_id).status == ItemStatus.COMMITTED
    assert all(listing.live for listing in market.listings.values())


def test_a_check_whose_deal_closed_unanswered_is_refused(world, notifier, market):
    """The agent released the deal; the runner has not resolved the check yet."""
    item_id = world.listed()
    aisha = world.thread(item_id, "Aisha")
    world.deal(item_id, aisha, pickup=True)
    check = world.pickup_check()
    _agent_releases_no_show(world, item_id, aisha)
    noah = world.thread(item_id, "Noah")
    world.tool(item_id, "accept_offer", thread_id=noah, price_cents=11_000)

    response = world.act("pickedup", check.id)

    assert response.disable_buttons and "no longer applies" in response.text
    assert world.get(Item, item_id).status == ItemStatus.COMMITTED
    assert world.act("pickedup", 99).text == "There is no pickup check with id 99."


def test_picked_up_says_which_listing_is_still_up(world, market):
    item_id = world.listed()
    world.deal(item_id, world.thread(item_id), pickup=True)
    check = world.pickup_check()
    market.fail_next["delist"] = PlatformError("503")

    response = world.act("pickedup", check.id)

    [listing] = world.session.exec(select(Listing)).all()
    assert world.get(Item, item_id).status == ItemStatus.SOLD
    assert "took its listings down" not in response.text
    assert response.text == (
        f"Marked item #{item_id} sold. Could not take it down on fake ({listing.url}): remove it "
        "there by hand."
    )


def test_sold_command_says_which_listing_is_still_up(world, market):
    item_id = world.listed()
    market.fail_next["delist"] = PlatformError("503")

    response = world.act("sold", item_id)

    assert response.text.startswith(f"Marked item #{item_id} sold. Could not take it down on fake")
    assert "remove it there by hand" in response.text


# ---------------------------------------------------------------- other escalations


def test_escalation_card_offers_resolve_reply_and_pause(world, notifier):
    item_id = world.listed()
    thread_id = world.thread(item_id, says="Can you deliver it to my place?")
    world.tool(item_id, "escalate", reason="The buyer wants delivery.", thread_id=thread_id)
    [notice] = notifier.of_kind(NoticeKind.ESCALATION)

    card = world.card(notice)

    assert card.title == "Needs you: IKEA desk"
    assert card.description == "The buyer wants delivery."
    assert _field(card, "Last from Maya") == "> Can you deliver it to my place?"
    assert _buttons(card) == [
        ("Resolve", f"ma:resolve:{notice.ref_id}", "primary", False),
        ("Reply as me", f"ma:reply:{thread_id}", "secondary", False),
        ("Pause item", f"ma:pause:{item_id}", "secondary", False),
    ]


def test_resolve_records_the_answer(world, notifier):
    item_id = world.listed()
    world.tool(item_id, "escalate", reason="Is delivery ok?")
    escalation_id = notifier.of_kind(NoticeKind.ESCALATION)[0].ref_id

    response = world.act("resolve", escalation_id, {"answer": "  No delivery, pickup only. "})

    assert response == views.ActionResponse(
        f"Resolved escalation #{escalation_id}; the agent will re-plan. Answer: No delivery, "
        "pickup only.",
        ephemeral=False,
        disable_buttons=True,
    )
    escalation = world.get(Escalation, escalation_id)
    assert escalation.resolution == "No delivery, pickup only."
    assert "already resolved" in world.act("resolve", escalation_id).text


def test_reply_sends_as_the_owner(world, market):
    item_id = world.listed()
    thread_id = world.thread(item_id)

    response = world.act("reply", thread_id, {"text": "Yes, it is still available."})

    assert not response.ephemeral and not response.disable_buttons
    assert [body for _, body, _ in market.sent] == ["Yes, it is still available."]
    sent = world.session.exec(
        select(Message).where(Message.direction == Direction.OUTBOUND)
    ).one()
    assert sent.via == "human"


def test_reply_needs_text(world):
    item_id = world.listed()
    thread_id = world.thread(item_id)

    assert world.act("reply", thread_id, {"text": "  "}) == views.ActionResponse(
        "Write the message to send first."
    )
    assert world.act("reply", thread_id).text == "Write the message to send first."


def test_pause_and_resume(world):
    item_id = world.listed()

    paused = world.act("pause", item_id)
    assert world.get(Item, item_id).paused
    resumed = world.act("resume", item_id)

    assert paused == views.ActionResponse(
        f"Paused item #{item_id}. The agent takes no action on it until you resume it.",
        ephemeral=False,
    )
    assert resumed.text == f"Resumed item #{item_id}." and not resumed.disable_buttons
    assert not world.get(Item, item_id).paused


# ---------------------------------------------------------------- APPROVAL


def test_approval_card_says_what_and_why_in_plain_words(world):
    item_id = world.listed()
    notice = world.park(
        item_id, "update_price", {"price_cents": 10_500}, why="Forty views, no messages."
    )

    card = world.card(notice)

    assert card.title == f"Approve #{notice.ref_id}? IKEA desk"  # the number /approve asks for
    assert card.description == (
        "The agent wants to:\nChange the asking price from $120.00 to $105.00\n\n"
        "Why: Forty views, no messages."
    )
    assert "{" not in _texts(card) and "price_cents" not in _texts(card)
    assert _field(card, "Expires") == "Fri Oct 2 12:00"
    assert _buttons(card) == [
        ("Approve", f"ma:approve:{notice.ref_id}", "success", False),
        ("Reject", f"ma:reject:{notice.ref_id}", "danger", False),
    ]


def test_approve_runs_the_parked_call(world):
    item_id = world.listed()
    notice = world.park(item_id, "update_price", {"price_cents": 10_500})

    response = world.act("approve", notice.ref_id)

    assert response.disable_buttons and not response.ephemeral
    assert response.text.startswith(f"Approved #{notice.ref_id}: ")
    assert world.get(PendingApproval, notice.ref_id).status == ApprovalStatus.APPROVED
    assert world.session.exec(select(Goal)).one().list_price_cents == 10_500
    card = world.card(notice)
    assert "Already approved." in card.description
    assert all(button.disabled for button in card.buttons)


def test_approve_that_cannot_run_still_closes_the_card(world):
    item_id = world.listed()
    notice = world.park(item_id, "update_price", {"price_cents": 5_000})

    response = world.act("approve", notice.ref_id)

    assert response.disable_buttons
    assert f"Approved #{notice.ref_id}, but it could not run:" in response.text
    assert "floor" in response.text


def test_reject_keeps_the_note_for_the_agent(world):
    item_id = world.listed()
    notice = world.park(item_id, "update_price", {"price_cents": 10_500})

    response = world.act("reject", notice.ref_id, {"note": "Too early to drop."})

    assert response == views.ActionResponse(
        f"Rejected #{notice.ref_id}. Note: Too early to drop.",
        ephemeral=False,
        disable_buttons=True,
    )
    assert world.get(PendingApproval, notice.ref_id).status == ApprovalStatus.REJECTED
    [memory] = world.session.exec(select(Memory)).all()
    assert memory.lesson.endswith("Too early to drop.")
    assert "already rejected" in world.act("reject", notice.ref_id).text


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("send_message", {"thread_id": "T", "body": "Hi\nStill there?"},
         "Send Maya this message:\n> Hi\n> Still there?"),
        ("update_price", {"price_cents": 9_999}, "Change the asking price from $120.00 to $99.99"),
        ("rewrite_listing",
         {"title": "Oak desk", "description": "Like new.", "photo_order": [2, 1]},
         'Rewrite the listing: new title "Oak desk", new description, new photo order 2, 1\n'
         "> Like new."),
        ("publish", {"platform": "kijiji"}, "Post the item on kijiji"),
        ("relist", {"platform": "fake"}, "Relist the item on fake"),
        ("accept_offer", {"thread_id": "T", "price_cents": 10_000}, "Accept $100.00 from Maya"),
        ("book_pickup", {"thread_id": "T", "start": PICKUP, "duration_minutes": 30},
         "Book Maya's pickup for Thu Oct 1 18:00 (30 minutes)"),
        ("release_commitment", {"thread_id": "T", "reason": "Gone quiet", "no_show": True},
         "Release the deal with Maya as a no-show: Gone quiet"),
        ("complete_sale", {"thread_id": "T"}, "Mark the sale to Maya as done"),
        ("mystery_tool", {"x_y": 1}, "Run mystery tool with x y 1"),
        ("book_pickup", {"thread_id": "T", "start": "soon"},
         "Run book pickup with start soon, thread id T"),
    ],
)
def test_describe_call_in_plain_words(world, settings, tool, args, expected):
    item_id = world.listed()
    thread_id = world.thread(item_id)
    args = {key: thread_id if value == "T" else value for key, value in args.items()}
    expected = expected.replace("thread id T", f"thread id {thread_id}")

    assert views.describe_call(world.session, settings, item_id, tool, args) == expected


def test_approval_waiting_on_its_commit_falls_back_to_the_notice(world, session, engine):
    item_id = world.listed()
    notice = Notice(
        kind=NoticeKind.APPROVAL,
        title="Approve publish?",
        body='{"platform": "fake"}\nTry it.',
        item_id=item_id,
        ref_id=99,
    )

    card = world.card(notice)

    assert not views.notice_ready(session, notice)
    assert card.description == notice.body
    assert [b.custom_id for b in card.buttons] == ["ma:approve:99", "ma:reject:99"]
    committed = world.park(item_id, "publish", {"platform": "fake"})
    with Session(engine) as other:
        assert views.notice_ready(other, committed)
    assert views.notice_ready(session, Notice(kind=NoticeKind.INFO, title="t", body="b"))


# ---------------------------------------------------------------- INFO, ALERT, SOLD, DIGEST


def test_info_card_offers_pause(world):
    item_id = world.listed()
    notice = Notice(kind=NoticeKind.INFO, title="Price changed: IKEA desk", body="Now $110.00.",
                    item_id=item_id)

    card = world.card(notice)

    assert (card.title, card.description) == ("Price changed: IKEA desk", "Now $110.00.")
    assert _buttons(card) == [("Pause item", f"ma:pause:{item_id}", "secondary", False)]
    assert card.color == views.COLORS[NoticeKind.INFO]


def test_alert_card_from_a_withdrawn_deal(world, notifier):
    item_id = world.listed()
    world.deal(item_id, world.thread(item_id))
    world.control.withdraw(item_id)
    [notice] = notifier.of_kind(NoticeKind.ALERT)

    card = world.card(notice)

    assert card.title == "Tell Maya the deal is off"
    assert card.buttons == () and card.color == views.COLORS[NoticeKind.ALERT]
    assert _field(card, "Item").startswith(f"[#{item_id} IKEA desk]")


def test_scam_alert_card_resolves_its_escalation(world, session):
    """runner/ingest.py freezes a scam conversation with an escalation and an ALERT naming it."""
    item_id = world.listed()
    thread_id = world.thread(item_id, says="Send me the 6 digit code first.")
    escalation = Escalation(
        item_id=item_id,
        thread_id=thread_id,
        reason="Froze a conversation with Maya: asks for a verification code.",
        created_at=world.clock.now(),
    )
    notice = Notice(
        kind=NoticeKind.ALERT,
        title="Possible scam on fake",
        body="Maya: asks for a verification code. The agent will not reply in this conversation.",
        item_id=item_id,
        ref_id=99,
    )
    assert not views.notice_ready(session, notice)  # waits for the escalation's commit
    session.add(escalation)
    session.commit()
    notice = replace(notice, ref_id=escalation.id)
    assert views.notice_ready(session, notice)

    card = world.card(notice)

    assert card.color == views.COLORS[NoticeKind.ALERT]
    assert _buttons(card) == [("Resolve", f"ma:resolve:{escalation.id}", "primary", False)]
    response = world.act("resolve", escalation.id, {"answer": "Ignore them."})
    assert response.disable_buttons
    assert _buttons(world.card(notice)) == [
        ("Resolve", f"ma:resolve:{escalation.id}", "primary", True)
    ]


def test_sold_card(world, notifier):
    item_id = world.listed()
    thread_id = world.thread(item_id)
    world.deal(item_id, thread_id, pickup=True)
    world.clock.set(AFTER_PICKUP)
    world.tool(item_id, "complete_sale", thread_id=thread_id)
    [notice] = notifier.of_kind(NoticeKind.SOLD)

    card = world.card(notice)

    assert (card.title, card.description) == ("Sold: IKEA desk", "Sold to Maya for $110.00.")
    assert card.buttons == ()
    assert _field(card, "Item") == f"[#{item_id} IKEA desk](http://127.0.0.1:8765/items/{item_id})"


def test_digest_notice_card(world):
    card = world.card(Notice(kind=NoticeKind.DIGEST, title="Digest", body="All quiet."))

    assert (card.title, card.description, card.fields, card.buttons) == (
        "Digest", "All quiet.", (), ()
    )


def test_no_web_link_when_the_app_binds_every_interface(world, settings):
    item_id = world.listed()
    settings.web_host = "0.0.0.0"

    card = world.card(Notice(kind=NoticeKind.ALERT, title="t", body="b", item_id=item_id))

    assert _field(card, "Item").startswith(f"#{item_id} IKEA desk\nfake: https://")
    assert views.item_url(settings, item_id) is None


def test_cards_never_show_the_pickup_address(world, settings):
    item_id = world.listed()
    body = f"Buyer asks if {settings.pickup_address} is right. " + "x" * 9000
    card = world.card(Notice(kind=NoticeKind.ESCALATION, title="Address?", body=body,
                             item_id=item_id, ref_id=12345))

    assert settings.pickup_address not in _texts(card)
    assert "[PICKUP_ADDRESS]" in card.description
    total = len(card.title) + len(card.description)
    total += sum(len(f.name) + len(f.value) for f in card.fields)
    assert len(card.description) <= views.DESCRIPTION_MAX and total <= views.EMBED_MAX
    assert card.description.endswith("...")


# ---------------------------------------------------------------- slash command actions


def test_price_floor_and_firmness(world):
    item_id = world.listed()

    price = world.act("price", item_id, {"amount": "110"})
    floor = world.act("floor", item_id, {"amount": "$90"})
    firm = world.act("firm", item_id, {"level": "Flexible"})

    goal = world.session.exec(select(Goal)).one()
    assert (goal.list_price_cents, goal.floor_price_cents, goal.firmness) == (
        11_000, 9_000, Firmness.FLEXIBLE
    )
    assert not price.ephemeral and "$110.00" in price.text
    assert floor.text == f"Set the floor of item #{item_id} to $90.00. The agent will re-plan."
    assert firm.text == f"Set item #{item_id} to flexible. The agent will re-plan."


@pytest.mark.parametrize(
    ("action", "form", "says"),
    [
        ("price", {"amount": "ten"}, "Expected dollars like 120"),
        ("price", {"amount": "50"}, "floor"),
        ("floor", {"amount": "500"}, "The floor cannot be above the list price"),
        ("firm", {"level": "wobbly"}, "Firmness is firm, slight, or flexible."),
        ("sold", {"price": "lots"}, "Expected dollars like 120"),
    ],
)
def test_bad_values_come_back_as_ephemeral_text(world, action, form, says):
    item_id = world.listed()

    response = world.act(action, item_id, form)

    assert response.ephemeral and not response.disable_buttons
    assert says in response.text
    goal = world.session.exec(select(Goal)).one()
    assert (goal.list_price_cents, goal.floor_price_cents) == (LIST, FLOOR)
    assert world.get(Item, item_id).status == ItemStatus.LISTED


def test_sold_with_a_price_remembers_it(world):
    item_id = world.listed()

    world.act("sold", item_id, {"price": "100"})

    assert world.get(Item, item_id).status == ItemStatus.SOLD
    [memory] = world.session.exec(select(Memory)).all()
    assert (memory.scope.value, memory.source) == ("global", "outcome")
    assert memory.lesson.startswith("Sold [IKEA desk] for $100 on fake")


def test_missing_records_are_refused_with_the_reason(world):
    assert world.act("undo", 99).text == "There is no deal with id 99."
    assert world.act("approve", 99).text == "There is no approval with id 99."
    assert world.act("pause", 99).text == "There is no item with id 99."


# ---------------------------------------------------------------- item views


def test_items_overview(world, session, settings):
    desk = world.listed()
    lamp = world.listed("Lamp")
    world.park(desk, "update_price", {"price_cents": 10_500})
    world.control.pause(lamp)
    sold = world.listed("Chair")
    world.control.mark_sold(sold)

    text = views.items_overview(session, settings)

    assert text.splitlines() == [
        f"#{desk} IKEA desk: listed, $120.00, on fake, 1 approval(s) waiting",
        f"#{lamp} Lamp: listed, paused, $120.00, on fake",
        "1 sold",
    ]


def test_items_overview_when_empty(session, settings):
    assert views.items_overview(session, settings).startswith("No items yet.")


def test_item_card(world, notifier, clock, settings):
    item_id = world.listed()
    thread_id = world.thread(item_id)
    deal = world.deal(item_id, thread_id, pickup=True)
    world.tool(item_id, "escalate", reason="Wants a receipt.")
    world.session.add(AgentRun(item_id=item_id, started_at=clock.now(), rationale="Took $110."))
    world.session.commit()

    card = views.item_card(world.session, clock, settings, item_id)

    assert card.title == f"#{item_id} IKEA desk"
    assert card.description == (
        "Status: committed\nPrice: $120.00, target $120.00, floor $95.00, slight\n"
        "Deadline: Thu Oct 15 12:00"
    )
    assert _field(card, "Listings").startswith("fake: live, https://fake.invalid/listings/")
    assert _field(card, "Deal") == (
        f"#{deal}: $110.00 with Maya\nPickup: {PICKUP_TEXT}\nUndo until {UNDO_UNTIL}"
    )
    assert _field(card, "Waiting for you").startswith("Escalation #")
    assert _field(card, "Last agent run") == "Thu Oct 1 12:00: Took $110."
    assert _buttons(card) == [("Pause", f"ma:pause:{item_id}", "secondary", False)]
    world.control.pause(item_id)
    paused = views.item_card(world.session, clock, settings, item_id)
    assert _buttons(paused) == [("Resume", f"ma:resume:{item_id}", "secondary", False)]


def test_status_text(world, session, engine, clock, settings):
    router = build_router(settings, engine, clock)
    session.add(TrustRecord(platform="*", action="update_price", tier=1, executed=3, corrected=1,
                            updated_at=clock.now()))
    session.add(QuotaUsage(provider="gemini", model="m", day="2026-10-01", requests=7))
    session.commit()

    text = views.status_text(session, settings, router)

    assert text.splitlines() == [
        "Quota: no model configured (set GEMINI_API_KEY)",
        "gemini: 7 of 250 requests today",
        "",
        "Trust:",
        "* update_price: act and notify (executed 3, corrected 1)",
    ]


# ---------------------------------------------------------------- digest


def test_next_digest_at_is_the_next_local_digest_hour(settings):
    morning = datetime(2026, 10, 1, 16, tzinfo=UTC)  # 12:00 local
    night = datetime(2026, 10, 2, 1, tzinfo=UTC)  # 21:00 local
    dst_end = datetime(2026, 11, 1, 3, tzinfo=UTC)  # 23:00 Oct 31 local, before the clocks fall

    assert views.next_digest_at(settings, morning) == datetime(2026, 10, 2, 0, tzinfo=UTC)
    assert views.next_digest_at(settings, night) == datetime(2026, 10, 3, 0, tzinfo=UTC)
    assert views.next_digest_at(settings, dst_end) == datetime(2026, 11, 2, 1, tzinfo=UTC)


def test_digest_with_real_data(world, session, clock, settings):
    sold = world.listed("Chair")
    world.deal(sold, world.thread(sold, name="Sam"))
    world.control.mark_sold(sold)
    desk = world.listed()
    world.deal(desk, world.thread(desk), pickup=True)
    table = world.listed("Oak table")
    world.thread(table, name="Ann")
    world.park(table, "update_price", {"price_cents": 10_500})
    world.tool(table, "escalate", reason="Asks about delivery.")
    session.add(AgentRun(
        item_id=table,
        started_at=clock.now(),
        results_json=json.dumps([
            {"tool": "send_message", "status": "executed"},
            {"tool": "send_message", "status": "executed"},
            {"tool": "update_price", "status": "rejected"},
        ]),
    ))
    session.add(QuotaUsage(provider="gemini", model="m", day="2026-10-01", requests=12))
    stale = world.listed("Lamp")
    listing = session.exec(select(Listing).where(Listing.item_id == stale)).one()
    listing.published_at = clock.now() - timedelta(days=5)
    session.commit()

    digest = views.build_digest(session, clock, settings)

    assert digest == "\n\n".join([
        "Digest for Thu Oct 1",
        f"Sold today\n#{sold} Chair: $110.00 to Sam",
        "Live negotiations\n"
        f"#{desk} IKEA desk: deal with Maya at $110.00, pickup {PICKUP_TEXT}\n"
        f"#{table} Oak table: talking with Ann (offered $110.00)",
        "Agent actions today\n1 wake(s)\nsend message: 2",
        "Waiting for you\n"
        f"Approval #1 on #{table}: Change the asking price from $120.00 to $105.00\n"
        f"Escalation #1 on #{table}: Asks about delivery.",
        f"Going stale\n#{stale} Lamp: listed 5 days, 0 views, no buyer messages",
        "Model requests today\ngemini: 12 of 250 requests today",
    ])
    assert "\u2014" not in digest


def test_digest_on_a_quiet_day(session, clock, settings):
    digest = views.build_digest(session, clock, settings)

    assert digest.split("\n\n") == [
        "Digest for Thu Oct 1",
        "Sold today\nnone",
        "Live negotiations\nnone",
        "Agent actions today\nnone",
        "Waiting for you\nnone",
        "Going stale\nnone",
        "Model requests today\nNo model requests today.",
    ]
