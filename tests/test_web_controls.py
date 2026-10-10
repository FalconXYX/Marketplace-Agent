"""web.controls: every owner control through the UI, each one a ControlService call, with the
outcome shown where the owner asked."""

import json
import re
from datetime import timedelta

import pytest
from sqlmodel import select

from marketplace_agent.core.models import (
    ApprovalStatus,
    Buyer,
    Commitment,
    CommitmentKind,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Memory,
    Message,
    PendingApproval,
)
from marketplace_agent.guard import ledger
from test_web_support import HX, accepting, build_world


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


@pytest.fixture
def item_id(world) -> int:
    return world.listed()


def _goal(world, item_id: int) -> Goal:
    with world.session() as session:
        return session.exec(select(Goal).where(Goal.item_id == item_id)).one()


def _item(world, item_id: int) -> Item:
    with world.session() as session:
        return session.get(Item, item_id)


def _flash(html: str) -> str:
    start = html.index('<div class="flash')
    return html[start : html.index("</div>", start)]


def _park(world, item_id: int, tool: str, args: dict) -> int:
    with world.session() as session:
        approval = PendingApproval(
            item_id=item_id,
            tool=tool,
            args_json=json.dumps(args),
            rationale="Nobody bites at $120.",
            created_at=world.clock.now(),
        )
        session.add(approval)
        session.commit()
        return approval.id


def _escalate(world, item_id: int | None, reason: str, thread_id: int | None = None) -> int:
    with world.session() as session:
        escalation = Escalation(
            item_id=item_id, thread_id=thread_id, reason=reason, created_at=world.clock.now()
        )
        session.add(escalation)
        session.commit()
        return escalation.id


# ---------------------------------------------------------------- prices and agent control


def test_price(world, item_id):
    response = world.htmx(f"/items/{item_id}/price", {"price": "$110"})

    assert response.status_code == 200 and 'id="item"' in response.text
    assert "$110.00" in _flash(response.text)
    assert _goal(world, item_id).list_price_cents == 11_000
    [listing] = world.market.listings.values()
    assert listing.payload.price_cents == 11_000


def test_a_price_under_the_floor_is_refused_inline(world, item_id):
    response = world.htmx(f"/items/{item_id}/price", {"price": "50"})

    assert response.status_code == 409
    flash = _flash(response.text)
    assert 'class="flash error"' in flash and "floor" in flash
    assert response.text.index('id="controls"') < response.text.index('class="flash error"')
    assert _goal(world, item_id).list_price_cents == 12_000


def test_a_price_that_is_not_dollars_is_refused(world, item_id):
    response = world.htmx(f"/items/{item_id}/price", {"price": "cheap"})

    assert response.status_code == 422 and "Price: enter dollars" in response.text


def test_floor(world, item_id):
    response = world.htmx(f"/items/{item_id}/floor", {"floor": "100"})

    assert response.status_code == 200 and "Floor set to $100.00." in response.text
    assert _goal(world, item_id).floor_price_cents == 10_000


def test_a_floor_above_the_price_is_refused(world, item_id):
    response = world.htmx(f"/items/{item_id}/floor", {"floor": "500"})

    assert response.status_code == 409 and "cannot be above the list price" in response.text
    assert _goal(world, item_id).floor_price_cents == 9_500


def test_firmness(world, item_id):
    response = world.htmx(f"/items/{item_id}/firmness", {"firmness": "firm"})

    assert response.status_code == 200 and "Firmness set to firm" in response.text
    assert _goal(world, item_id).firmness == Firmness.FIRM
    bad = world.htmx(f"/items/{item_id}/firmness", {"firmness": "stubborn"})
    assert bad.status_code == 422 and "Firmness must be one of" in bad.text


def test_pause_and_resume(world, item_id):
    paused = world.htmx(f"/items/{item_id}/pause")
    assert paused.status_code == 200 and "Paused." in paused.text
    assert "Resume the agent" in paused.text and _item(world, item_id).paused

    resumed = world.htmx(f"/items/{item_id}/resume")
    assert resumed.status_code == 200 and "Pause the agent" in resumed.text
    assert not _item(world, item_id).paused


def test_mark_sold_takes_the_listing_down(world, item_id):
    response = world.htmx(f"/items/{item_id}/sold", {"price": "115"})

    assert response.status_code == 200 and "Marked sold." in response.text
    assert 'id="controls"' not in response.text  # nothing left to control
    assert _item(world, item_id).status == ItemStatus.SOLD
    [listing] = world.market.listings.values()
    assert not listing.live
    with world.session() as session:
        [memory] = session.exec(select(Memory)).all()
        assert (memory.scope.value, memory.source) == ("global", "outcome")
        assert memory.lesson.startswith("Sold [IKEA Linnmon desk, white, Furniture] for $115 on")


def test_withdraw(world, item_id):
    response = world.htmx(f"/items/{item_id}/withdraw")

    assert response.status_code == 200 and "Withdrawn." in response.text
    assert _item(world, item_id).status == ItemStatus.WITHDRAWN
    again = world.htmx(f"/items/{item_id}/withdraw")
    assert again.status_code == 409 and "already withdrawn" in again.text


def test_a_control_on_an_unknown_item_is_a_not_found(world):
    response = world.htmx("/items/42/pause")

    assert response.status_code == 404 and response.headers["HX-Retarget"] == "#toast"
    assert "There is no item with id 42." in response.text


def test_without_htmx_a_control_answers_with_the_whole_page(world, item_id):
    response = world.client.post(f"/items/{item_id}/pause")

    assert response.status_code == 200 and response.text.startswith("<!doctype html>")
    assert "Paused." in response.text


# ---------------------------------------------------------------- approvals


def test_approve_runs_the_parked_call(world, item_id):
    approval_id = _park(world, item_id, "update_price", {"price_cents": 11_000})
    page = world.client.get(f"/items/{item_id}").text
    assert "Update price" in page and "Nobody bites" in page
    assert "Change the asking price from $120.00 to $110.00" in page

    response = world.htmx(f"/approvals/{approval_id}/approve")

    assert response.status_code == 200 and "$110.00" in _flash(response.text)
    assert f"approval-{approval_id}" not in response.text
    assert _goal(world, item_id).list_price_cents == 11_000
    with world.session() as session:
        assert session.get(PendingApproval, approval_id).status == ApprovalStatus.APPROVED


def test_approvals_name_the_buyer_and_the_local_time(world, item_id):
    """Two offers waiting: the owner must see whose each one is, as Discord shows it."""
    first = world.buyer_writes(item_id, "Would you take $110?")
    second = world.buyer_writes(item_id, "$105 and I come today")
    one, two = thread_buyer(world, first), thread_buyer(world, second)
    assert one != two
    _park(world, item_id, "accept_offer", {"thread_id": first.id, "price_cents": 11_000})
    _park(world, item_id, "accept_offer", {"thread_id": second.id, "price_cents": 10_500})
    _park(world, item_id, "book_pickup", {"thread_id": first.id, "start": "2026-10-10T13:30"})

    page = world.client.get(f"/items/{item_id}").text

    assert f"Accept $110.00 from {one}" in page and f"Accept $105.00 from {two}" in page
    assert f"Book {one}&#39;s pickup for Sat Oct 10 13:30 (60 minutes)" in page
    assert "thread id" not in page


def test_reject_with_a_note_teaches_the_agent(world, item_id):
    approval_id = _park(world, item_id, "update_price", {"price_cents": 9_900})

    response = world.htmx(f"/approvals/{approval_id}/reject", {"note": "Too soon to drop."})

    assert response.status_code == 200 and "Rejected." in response.text
    with world.session() as session:
        assert session.get(PendingApproval, approval_id).status == ApprovalStatus.REJECTED
        lessons = [m.lesson for m in session.exec(select(Memory))]
    assert any("Too soon to drop." in lesson for lesson in lessons)


def test_an_approval_decided_already_is_refused(world, item_id):
    approval_id = _park(world, item_id, "update_price", {"price_cents": 11_000})
    world.htmx(f"/approvals/{approval_id}/reject")

    response = world.htmx(f"/approvals/{approval_id}/approve")

    assert response.status_code == 409 and "already rejected" in response.text


def test_an_unknown_approval_is_a_not_found(world):
    response = world.htmx("/approvals/77/approve")

    assert response.status_code == 404 and "There is no approval with id 77." in response.text


# ---------------------------------------------------------------- escalations and replies


def test_resolve_an_escalation(world, item_id):
    escalation_id = _escalate(world, item_id, "Buyer asks if you deliver.")

    response = world.htmx(
        f"/escalations/{escalation_id}/resolve", {"resolution": "No delivery, pickup only."}
    )

    assert response.status_code == 200 and "Resolved." in response.text
    assert "Buyer asks if you deliver." not in response.text
    with world.session() as session:
        escalation = session.get(Escalation, escalation_id)
    assert escalation.resolved_at is not None
    assert escalation.resolution == "No delivery, pickup only."


def test_an_escalation_about_no_item_is_resolved_from_the_home_page(world):
    escalation_id = _escalate(world, None, "Froze a chat that names no listing.")
    assert "Froze a chat that names no listing." in world.client.get("/").text

    response = world.htmx(f"/escalations/{escalation_id}/resolve")

    assert response.status_code == 200
    assert response.text.startswith('<section id="attention"') and "Resolved." in response.text
    assert "Froze a chat" not in response.text


def test_reply_as_me_from_a_conversation_or_an_escalation(world, item_id):
    thread = world.buyer_writes(item_id, "Does it wobble?")
    _escalate(world, item_id, "Buyer asks about wobble.", thread_id=thread.id)
    page = world.client.get(f"/items/{item_id}").text
    assert f'href="#thread-{thread.id}">Reply as me' in page

    response = world.htmx(f"/threads/{thread.id}/reply", {"body": "Solid, no wobble."})

    assert response.status_code == 200
    assert world.market.sent[-1][1] == "Solid, no wobble."
    assert "Solid, no wobble." in response.text and "you</p>" in response.text
    with world.session() as session:
        sent = session.exec(select(Message).where(Message.via == "human")).one()
    assert sent.thread_id == thread.id


def test_a_refused_reply_shows_under_its_conversation(world, item_id):
    thread = world.buyer_writes(item_id, "Does it come apart for transport?")
    sent = len(world.market.sent)

    response = world.htmx(f"/threads/{thread.id}/reply", {"body": "Text me at 416-555-0199"})

    assert response.status_code == 409
    text = response.text
    assert text.index(f'id="thread-{thread.id}"') < text.index('class="flash error"')
    assert len(world.market.sent) == sent


@pytest.mark.parametrize("close", ["withdraw", "sold"])
def test_a_closed_item_says_where_to_reply_instead_of_offering_a_reply(tmp_path, close):
    """After Withdraw the owner is told to tell the buyer; the guard (rule 3) stops every send
    on a closed item, so the page says where to write rather than offering a form that fails
    with a message meant for the agent."""
    world = build_world(tmp_path, accepting(11_000))
    item_id = world.listed()
    thread = world.buyer_writes(item_id, "I'll take it for $110.")
    sent = len(world.market.sent)

    assert world.htmx(f"/items/{item_id}/{close}").status_code == 200

    page = world.client.get(f"/items/{item_id}").text
    assert f'action="/threads/{thread.id}/reply"' not in page
    assert f"so nothing more is sent from here. Write to {thread_buyer(world, thread)}" in page
    response = world.htmx(f"/threads/{thread.id}/reply", {"body": "Sorry, it is gone."})
    assert response.status_code == 409 and "in the fake app or site instead" in response.text
    assert "escalate to the human" not in response.text and len(world.market.sent) == sent
    if close == "withdraw":
        [alert] = [n for n in world.notifier.notices if n.title.endswith("the deal is off")]
        assert alert.body.endswith("Write to them yourself on fake.")


def thread_buyer(world, thread) -> str:
    with world.session() as session:
        return session.get(Buyer, thread.buyer_id).display_name


# ---------------------------------------------------------------- deals


def test_undo_after_the_window_is_refused_inline(tmp_path):
    world = build_world(tmp_path, accepting(11_000))
    item_id = world.listed()
    world.buyer_writes(item_id, "I'll take it for $110.")
    with world.session() as session:
        [accepted] = session.exec(select(Commitment)).all()
    world.clock.advance(minutes=11)
    page = world.client.get(f"/items/{item_id}").text
    assert "Undo deal" not in page  # the window has closed

    response = world.htmx(f"/commitments/{accepted.id}/undo")

    assert response.status_code == 409 and "Too late to undo" in response.text
    assert _item(world, item_id).status == ItemStatus.COMMITTED


def _pickup_over(world, item_id: int) -> int:
    """A deal whose booked pickup ended an hour ago, and the runner's question about it."""
    thread = world.buyer_writes(item_id, "Deal at $110?")
    with world.session() as session:
        accepted = ledger.accept(
            session,
            world.clock,
            item_id=item_id,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            price_cents=11_000,
            undo_minutes=10,
        )
        start = world.clock.now() + timedelta(hours=1)
        end = start + timedelta(hours=1)
        ledger.book_pickup(session, world.clock, accepted_id=accepted.id, start=start, end=end)
        session.commit()
    world.clock.advance(hours=3)
    world.runtime.runner.tick()
    return accepted.id


def test_the_pickup_check_asks_picked_up_or_no_show(world, item_id):
    _pickup_over(world, item_id)

    page = world.client.get(f"/items/{item_id}").text

    assert "pick it up?" in page
    assert f'action="/items/{item_id}/sold"' in page and 'name="anchor" value="deal"' in page
    assert f'action="/items/{item_id}/noshow"' in page
    check = _form_data(page, f"/items/{item_id}/noshow")["check"]
    assert _form_data(page, f"/items/{item_id}/sold") == {"anchor": "deal", "check": check}
    assert "Questions for you" not in page  # the check is answered in the deal, not resolved


def test_no_show_releases_the_deal(world, item_id):
    accepted_id = _pickup_over(world, item_id)

    page = world.client.get(f"/items/{item_id}").text

    response = world.htmx(f"/items/{item_id}/noshow", _form_data(page, f"/items/{item_id}/noshow"))

    assert response.status_code == 200 and "as a no-show" in response.text
    assert _item(world, item_id).status == ItemStatus.LISTED
    with world.session() as session:
        closing = session.exec(select(Commitment).where(Commitment.ref_id == accepted_id))
        assert CommitmentKind.RELEASED in [c.kind for c in closing]
        assert all(e.resolved_at for e in session.exec(select(Escalation)))


def test_picked_up_marks_it_sold(world, item_id):
    _pickup_over(world, item_id)

    page = world.client.get(f"/items/{item_id}").text

    response = world.htmx(f"/items/{item_id}/sold", _form_data(page, f"/items/{item_id}/sold"))

    assert response.status_code == 200 and "Marked sold." in response.text
    assert _item(world, item_id).status == ItemStatus.SOLD


def _form_data(page: str, action: str) -> dict[str, str]:
    """The hidden fields of the form posting to action, as the browser would send them."""
    start = page.index(f'action="{action}"')
    form = page[start : page.index("</form>", start)]
    return dict(re.findall(r'type="hidden" name="(\w+)" value="([^"]*)"', form))


def _next_deal_with_pickup(world, item_id: int, accepted_id: int) -> int:
    """A second buyer's deal, its pickup an hour from now."""
    with world.session() as session:
        first = session.get(Commitment, accepted_id)
        ledger.release(session, world.clock, accepted_id=accepted_id, reason="x", actor="agent")
        session.commit()
    thread = world.buyer_writes(item_id, "Still available? $105?")
    assert thread.buyer_id != first.buyer_id
    with world.session() as session:
        accepted = ledger.accept(
            session,
            world.clock,
            item_id=item_id,
            thread_id=thread.id,
            buyer_id=thread.buyer_id,
            price_cents=10_500,
            undo_minutes=10,
        )
        start = world.clock.now() + timedelta(minutes=5)
        end = start + timedelta(hours=1)
        ledger.book_pickup(session, world.clock, accepted_id=accepted.id, start=start, end=end)
        session.commit()
    world.runtime.runner.tick()  # resolves the first deal's check
    return accepted.id


@pytest.mark.parametrize("answer", ["noshow", "sold"])
def test_an_old_pickup_check_page_never_answers_for_a_later_deal(world, item_id, answer):
    first = _pickup_over(world, item_id)
    action = f"/items/{item_id}/{answer}"
    old_form = _form_data(world.client.get(f"/items/{item_id}").text, action)
    second = _next_deal_with_pickup(world, item_id, first)
    world.clock.advance(minutes=30)  # the second pickup is under way

    response = world.htmx(action, old_form)

    assert response.status_code == 409 and "already answered" in response.text
    with world.session() as session:
        assert ledger.active_commitment(session, item_id).accepted.id == second
    assert _item(world, item_id).status == ItemStatus.COMMITTED


def test_no_show_without_a_deal_is_refused(world, item_id):
    response = world.htmx(f"/items/{item_id}/noshow")

    assert response.status_code == 409 and "has no active deal" in response.text


def test_every_control_needs_a_post(world, item_id):
    paths = [
        f"/items/{item_id}/{name}"
        for name in ("price", "floor", "firmness", "pause", "resume", "sold", "noshow", "withdraw")
    ]
    paths += ["/commitments/1/undo", "/approvals/1/approve", "/approvals/1/reject"]
    paths += ["/escalations/1/resolve", "/threads/1/reply"]
    for path in paths:
        assert world.client.get(path, headers=HX).status_code == 405, path
