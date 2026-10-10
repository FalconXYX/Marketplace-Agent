"""sim.audit: each invariant broken once is reported; a clean history reports nothing."""
import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel

from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Message,
    PriceChange,
    Thread,
    ThreadState,
)
from marketplace_agent.sim import audit as audit_module
from marketplace_agent.sim.audit import audit

T0 = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)  # Thu 12:00 in America/Toronto
THU_6PM = datetime(2026, 10, 1, 22, 0, tzinfo=UTC)  # inside the Mon-Fri 17:00-20:00 window
SAT_11AM = datetime(2026, 10, 3, 15, 0, tzinfo=UTC)  # inside the Sat-Sun 10:00-18:00 window
ADDRESS = "123 Example St"  # conftest settings.pickup_address


def at(minutes: float = 0, *, hours: float = 0) -> datetime:
    return T0 + timedelta(hours=hours, minutes=minutes)


class World:
    """Writes rows directly, bypassing guard/ and tools/, so any state can be built."""

    def __init__(self, session: Session):
        self.session = session
        self._serial = 0

    def _add[R: SQLModel](self, row: R) -> R:
        self.session.add(row)
        self.session.flush()
        return row

    def _next(self) -> int:
        self._serial += 1
        return self._serial

    def item(
        self, *, list_cents: int = 12000, floor_cents: int = 9500,
        firmness: Firmness = Firmness.SLIGHT,
    ) -> Item:
        item = self._add(
            Item(title="Desk", status=ItemStatus.LISTED, created_at=T0, updated_at=T0)
        )
        self._add(
            Goal(
                item_id=item.id, list_price_cents=list_cents, target_price_cents=list_cents,
                floor_price_cents=floor_cents, deadline=T0 + timedelta(days=14),
                firmness=firmness,
            )
        )
        return item

    def buyer(self) -> Buyer:
        return self._add(Buyer(platform="fake", handle=f"buyer-{self._next()}"))

    def thread(self, item: Item, buyer: Buyer | None = None) -> Thread:
        buyer = buyer or self.buyer()
        return self._add(
            Thread(
                platform="fake", remote_id=f"t-{self._next()}", item_id=item.id,
                buyer_id=buyer.id, created_at=T0,
            )
        )

    def inbound(
        self, thread: Thread, when: datetime, body: str = "Is this still available?",
        intent: str | None = None,
    ) -> Message:
        return self._add(
            Message(
                thread_id=thread.id, direction=Direction.INBOUND, body=body, sent_at=when,
                via="buyer", handled=True, intent=intent,
            )
        )

    def outbound(
        self, thread: Thread, when: datetime, body: str = "Yes, it's available.",
        via: str = "agent",
    ) -> Message:
        return self._add(
            Message(
                thread_id=thread.id, direction=Direction.OUTBOUND, body=body, sent_at=when,
                via=via,
            )
        )

    def accept(self, thread: Thread, when: datetime, price_cents: int = 10500) -> Commitment:
        return self._add(
            Commitment(
                item_id=thread.item_id, thread_id=thread.id, buyer_id=thread.buyer_id,
                kind=CommitmentKind.ACCEPTED, price_cents=price_cents,
                undo_deadline=when + timedelta(minutes=10), created_at=when,
            )
        )

    def book(
        self, accepted: Commitment, when: datetime, start: datetime = THU_6PM,
        minutes: int = 60,
    ) -> Commitment:
        return self._event(
            accepted, CommitmentKind.PICKUP_BOOKED, when,
            pickup_start=start, pickup_end=start + timedelta(minutes=minutes),
        )

    def close(self, accepted: Commitment, kind: CommitmentKind, when: datetime) -> Commitment:
        return self._event(accepted, kind, when)

    def _event(
        self, accepted: Commitment, kind: CommitmentKind, when: datetime, **fields: datetime
    ) -> Commitment:
        return self._add(
            Commitment(
                item_id=accepted.item_id, thread_id=accepted.thread_id,
                buyer_id=accepted.buyer_id, kind=kind, ref_id=accepted.id, created_at=when,
                **fields,
            )
        )

    def price_change(self, item: Item, old: int, new: int, when: datetime) -> None:
        self._add(PriceChange(item_id=item.id, old_cents=old, new_cents=new, created_at=when))


@pytest.fixture
def world(session) -> World:
    return World(session)


def run(session: Session, settings, **kwargs) -> list[str]:
    session.commit()
    session.expire_all()  # read back from SQLite, which drops tzinfo
    return audit(session, settings, **kwargs)


def only(violations: list[str], invariant: str) -> str:
    assert len(violations) == 1, violations
    assert violations[0].split(" ")[1] == f"{invariant}:", violations
    return violations[0]


# Clean history


def test_clean_multi_item_history_reports_nothing(session, settings, world):
    desk = world.item()
    alice = world.thread(desk)
    world.inbound(alice, at(0), intent="availability")
    world.outbound(alice, at(1), "Hi! Yes, it's still available. Pickup is near Bloor St W.",
                   via="pattern_cache")
    world.inbound(alice, at(30), "would you take 90?", intent="offer")
    world.outbound(alice, at(35), "I can't go that low, but I could do $105.")
    world.inbound(alice, at(40), "Deal at 105")
    desk_deal = world.accept(alice, at(45), 10500)
    world.outbound(alice, at(46), "Great, it's yours for $105. Can you come today after 5?")
    world.inbound(alice, at(50), "6pm today")
    world.book(desk_deal, at(60), THU_6PM)
    world.outbound(alice, at(61), f"See you at 6. Address: {audit_module.PLACEHOLDER}")
    world.inbound(alice, at(hours=7), "Picked it up, thanks!")
    world.close(desk_deal, CommitmentKind.COMPLETED, at(hours=7))

    bob = world.thread(desk)
    world.inbound(bob, at(10))
    world.outbound(bob, at(12), "Yes! Asking $120.")
    world.outbound(bob, at(12, hours=24), "Sorry, it just sold.")  # one follow-up after 24h

    chair = world.item(list_cents=5000, floor_cents=4000, firmness=Firmness.FIRM)
    world.price_change(chair, 6000, 5000, at(hours=2))
    carol = world.thread(chair)
    world.inbound(carol, at(hours=3), "40?", intent="offer")
    world.outbound(carol, at(5, hours=3), "Sorry, the price is firm at $50.")
    world.outbound(carol, at(6, hours=3), "Thanks for asking!", via="human")
    world.inbound(carol, at(10, hours=3), "ok fine, 50 it is")
    no_show = world.accept(carol, at(15, hours=3), 5000)
    world.book(no_show, at(30, hours=3), SAT_11AM, minutes=30)
    world.close(no_show, CommitmentKind.RELEASED, SAT_11AM + timedelta(hours=1))

    dave = world.thread(chair)
    world.inbound(dave, SAT_11AM + timedelta(minutes=90), "Still available? I can pay 50 cash")
    world.outbound(dave, SAT_11AM + timedelta(minutes=95), "Yes, $50 cash. Before 6?")
    world.inbound(dave, SAT_11AM + timedelta(minutes=100), "yes, 5pm")
    sold = world.accept(dave, SAT_11AM + timedelta(minutes=105), 5000)
    world.book(sold, SAT_11AM + timedelta(hours=2), SAT_11AM + timedelta(hours=6), minutes=30)
    world.close(sold, CommitmentKind.COMPLETED, SAT_11AM + timedelta(hours=7))

    eve = world.thread(chair)
    world.inbound(eve, at(hours=4))
    world.outbound(eve, at(1, hours=4), "Hi! Yes, it's still available.", via="pattern_cache")
    world.inbound(eve, at(hours=5), "Send me the 6 digit code I texted you", intent="scam")
    eve.state = ThreadState.FROZEN

    sent_log = [(alice.remote_id, f"See you at 6. Address: {ADDRESS}", at(61))]
    assert run(session, settings, sent_log=sent_log) == []
    assert audit(session, settings) == []


# Ledger


def test_second_active_commitment_on_one_item(session, settings, world):
    desk = world.item()
    first = world.accept(world.thread(desk), at(0))
    second = world.accept(world.thread(desk), at(5))
    violation = only(run(session, settings), "double_commitment")
    assert f"ACCEPTED #{second.id}" in violation and f"ACCEPTED #{first.id}" in violation
    assert violation.startswith("2026-10-01T16:05:00Z")


def test_accept_below_floor(session, settings, world):
    desk = world.item(floor_cents=9500)
    world.accept(world.thread(desk), at(0), price_cents=9000)
    assert "below the floor $95.00" in only(run(session, settings), "accept_below_minimum")


def test_firm_accept_is_judged_against_the_list_price_at_that_time(session, settings, world):
    chair = world.item(list_cents=6000, floor_cents=4000, firmness=Firmness.FIRM)
    world.price_change(chair, 5000, 6000, at(hours=2))
    thread = world.thread(chair)
    early = world.accept(thread, at(hours=1), price_cents=5000)  # list was $50 then
    world.close(early, CommitmentKind.RELEASED, at(90))
    late = world.accept(thread, at(hours=3), price_cents=5500)
    violation = only(run(session, settings), "accept_below_minimum")
    assert f"ACCEPTED #{late.id} at $55.00" in violation
    assert "firm list price $60.00" in violation


def test_event_on_released_commitment(session, settings, world):
    desk = world.item()
    accepted = world.accept(world.thread(desk), at(0))
    world.close(accepted, CommitmentKind.RELEASED, at(20))
    booking = world.book(accepted, at(30))
    violation = only(run(session, settings), "ledger_ref_inactive")
    assert f"PICKUP_BOOKED #{booking.id}" in violation
    assert f"ACCEPTED #{accepted.id}" in violation


def test_undo_after_deadline(session, settings, world):
    accepted = world.accept(world.thread(world.item()), at(0))
    world.close(accepted, CommitmentKind.UNDONE, at(11))
    assert "undo deadline 2026-10-01T16:10:00Z" in only(run(session, settings), "late_undo")


def test_undo_inside_window_is_fine(session, settings, world):
    accepted = world.accept(world.thread(world.item()), at(0))
    world.close(accepted, CommitmentKind.UNDONE, at(10))
    assert run(session, settings) == []


def test_pickup_with_empty_interval(session, settings, world):
    accepted = world.accept(world.thread(world.item()), at(0))
    world.book(accepted, at(20), minutes=0)
    only(run(session, settings), "pickup_invalid")


def test_pickup_booked_in_the_past(session, settings, world):
    accepted = world.accept(world.thread(world.item()), at(0))
    world.book(accepted, THU_6PM + timedelta(minutes=5), THU_6PM)
    assert "Thu 2026-10-01 18:00 local" in only(run(session, settings), "pickup_in_past")


@pytest.mark.parametrize(
    "start, minutes",
    [
        (T0 + timedelta(days=1), 60),  # Fri 12:00 local, before the 17:00 window
        (THU_6PM + timedelta(hours=1, minutes=30), 60),  # 19:30 to 20:30 runs past 20:00
        (SAT_11AM + timedelta(hours=6, minutes=30), 60),  # Sat 17:30 to 18:30
    ],
)
def test_pickup_outside_availability(session, settings, world, start, minutes):
    accepted = world.accept(world.thread(world.item()), at(0))
    world.book(accepted, at(20), start, minutes)
    only(run(session, settings), "pickup_outside_availability")


def test_overlapping_pickups_across_items(session, settings, world):
    first = world.accept(world.thread(world.item()), at(0))
    second = world.accept(world.thread(world.item()), at(1))
    first_booking = world.book(first, at(20), THU_6PM)
    second_booking = world.book(second, at(21), THU_6PM + timedelta(minutes=30))
    violation = only(run(session, settings), "pickup_overlap")
    assert f"#{second_booking.id}" in violation and f"#{first_booking.id}" in violation
    assert "Thu 2026-10-01 18:30 local" in violation


def test_back_to_back_rebooked_and_released_pickups_do_not_overlap(session, settings, world):
    first = world.accept(world.thread(world.item()), at(0))
    second = world.accept(world.thread(world.item()), at(1))
    third = world.accept(world.thread(world.item()), at(2))
    world.book(first, at(20), THU_6PM + timedelta(minutes=30))
    world.book(first, at(21), THU_6PM)  # rebooked over its own slot: 18:00 to 19:00
    world.book(second, at(22), THU_6PM + timedelta(hours=1))  # starts as first now ends
    world.close(second, CommitmentKind.RELEASED, at(23))
    world.book(third, at(24), THU_6PM + timedelta(hours=1))  # second's slot, now free
    assert run(session, settings) == []


# Message rules


def test_follow_up_within_24h(session, settings, world):
    thread = world.thread(world.item())
    world.inbound(thread, at(0))
    first = world.outbound(thread, at(1))
    follow_up = world.outbound(thread, at(1, hours=23))
    violation = only(run(session, settings), "outbound_streak")
    assert f"follow-up #{follow_up.id}" in violation and f"message #{first.id}" in violation


def test_third_message_without_reply(session, settings, world):
    thread = world.thread(world.item())
    world.inbound(thread, at(0))
    world.outbound(thread, at(1))
    world.outbound(thread, at(1, hours=25))
    third = world.outbound(thread, at(1, hours=50))
    violation = only(run(session, settings), "outbound_streak")
    assert f"message #{third.id} is outbound number 3" in violation


def test_buyer_reply_resets_streak_and_human_is_exempt(session, settings, world):
    thread = world.thread(world.item())
    world.inbound(thread, at(0))
    world.outbound(thread, at(1))
    world.outbound(thread, at(2), via="human")
    world.outbound(thread, at(3), via="human")
    world.inbound(thread, at(4))
    world.outbound(thread, at(5))
    assert run(session, settings) == []


def test_daily_outbound_cap(session, settings, world):
    capped = settings.model_copy(update={"daily_outbound_cap": {"fake": 2}})
    item = world.item()
    for minute in range(3):
        thread = world.thread(item)
        world.inbound(thread, at(minute))
        world.outbound(thread, at(minute, hours=1), via="human")  # humans do not count
        world.inbound(thread, at(minute, hours=2))
        last = world.outbound(thread, at(minute, hours=3))
    for minute in range(2):  # a new UTC day starts at T0 + 8h
        thread = world.thread(item)
        world.inbound(thread, at(minute, hours=9))
        world.outbound(thread, at(minute, hours=9))
    violation = only(run(session, capped), "outbound_cap")
    assert f"message #{last.id} is outbound number 3 on fake on 2026-10-01" in violation


@pytest.mark.parametrize(
    "body, flagged",
    [
        ("I could do $80.", ["$80.00"]),
        ("Lowest is 80 bucks", ["$80.00"]),
        ("How about 85?", ["$85.00"]),
        ("I can't take 90, sorry", ["$90.00"]),
        ("Not $1,200.50, but I'd do 90 cash", ["$90.00"]),
        ("I can do $95 or $1.2k for both", []),
        ("Can you come at 6? Saturday 10? Oct 12?", []),
        ("Pickup 5:30 or 5pm works, it's 48x24 and 30\" deep", []),
        ("Call me at 2 for 1 more photo", []),
        ("Can you do 6? Would you take 7? How about 12?", []),  # hours, as the guard reads them
        ("I'd do 9 cash", ["$9.00"]),
        ("Would you take 13?", ["$13.00"]),
        # Agreeing to a price is stating it: the guard and this audit both missed these once.
        ("Deal at 50! It's yours, when can you come by?", ["$50.00"]),
        ("Sure, 80 is fine.", ["$80.00"]),
        ("Ok, 80 it is!", ["$80.00"]),
        ("80 would work for me.", ["$80.00"]),
        ("Yours for 80", ["$80.00"]),
        ("Sold for 80!", ["$80.00"]),
        ("80 sounds good", ["$80.00"]),
        ("Happy with 80", ["$80.00"]),
        ("I am ok with 80", ["$80.00"]),
        ("agreed at 80", ["$80.00"]),
        ("I can go to 80", ["$80.00"]),
        ("I would let it go at 80", ["$80.00"]),
        ("Ok, 80,00 $ then", ["$80.00"]),
        ("\u20ac80 is fine", ["$80.00"]),
        ("fifty bucks and it's yours", ["$50.00"]),
        ("Saturday at 6 works, see you then", []),
    ],
)
def test_amounts_below_floor_in_outbound_text(session, settings, world, body, flagged):
    thread = world.thread(world.item(floor_cents=9500))
    world.inbound(thread, at(0))
    world.outbound(thread, at(1), body)
    violations = run(session, settings)
    if flagged:
        assert f"states {', '.join(flagged)}, below the floor $95.00" in only(
            violations, "text_below_minimum"
        )
    else:
        assert violations == []


def test_amount_below_firm_list_price_in_text(session, settings, world):
    thread = world.thread(world.item(list_cents=12000, floor_cents=9500, firmness=Firmness.FIRM))
    world.inbound(thread, at(0), "100?")
    world.outbound(thread, at(1), "Sorry, $100 is too low, it's firm at $120.")
    assert "below the firm list price $120.00" in only(
        run(session, settings), "text_below_minimum"
    )


@pytest.mark.parametrize(
    "body",
    ["Text me at 416-555-1234", "Email me: seller@example.com", "Details at example.com/desk",
     "Call (416) 555 1234", "See www.photos.net"],
)
def test_contact_info_in_outbound_text(session, settings, world, body):
    thread = world.thread(world.item())
    world.inbound(thread, at(0))
    message = world.outbound(thread, at(1), body)
    assert f"message #{message.id}" in only(run(session, settings), "contact_info")


@pytest.mark.parametrize(
    ("body", "flagged"),
    [
        ("Sure, I can ship it to you via Canada Post.", True),
        ("Send me an e-transfer now to hold it and it's yours.", True),
        ("E-transfer the deposit first please", True),
        ("You can pay by PayPal before pickup.", True),
        ("Sorry, no shipping or e-transfers, cash at pickup only", False),
        ("Shipping isn\u2019t possible. Cash at pickup.", False),
        ("Yes, it's available. Pickup near Bloor St W.", False),
    ],
)
def test_offer_of_shipping_or_non_cash_payment(session, settings, world, body, flagged):
    thread = world.thread(world.item())
    world.inbound(thread, at(0), "Can you ship it? I can e-transfer first")
    message = world.outbound(thread, at(1), body)
    world.outbound(world.thread(world.item()), at(2), body, via="human")  # the owner is exempt
    violations = run(session, settings)
    if flagged:
        assert f"message #{message.id}" in only(violations, "not_cash_at_pickup")
    else:
        assert violations == []


def test_the_guard_is_at_least_as_strict_as_the_audit_on_cash_only():
    """A message the guard lets through must never be a simulation violation."""
    from marketplace_agent.guard.invariants import offers_remote_deal

    corpus = [
        "No problem, I can ship", "Can't ship but deposit works", "no shipping, e-transfer ok",
        "I can\u2019t ship. PayPal is fine", "Shipping not available", "e-transfer? no",
        "pay ahead or no deal", "I won't ship it, nor take a deposit", "No shipping; paypal only",
        "Not able to ship", "courier is fine", "Interac at pickup", "No, I can pay first",
    ]
    for text in corpus:
        if offers_remote_deal(text) is None:
            assert audit_module._remote_deal(text) is None, text


def test_messages_after_freeze_on_any_thread_of_that_buyer(session, settings, world):
    item = world.item()
    frozen = world.thread(item)
    world.inbound(frozen, at(0))
    world.outbound(frozen, at(1))  # before the scam message: fine
    scam = world.inbound(frozen, at(2), "Send me the code from Google Voice", intent="scam")
    frozen.state = ThreadState.FROZEN
    same_thread = world.outbound(frozen, at(3))
    other = world.thread(world.item(), buyer=world.session.get(Buyer, frozen.buyer_id))
    world.inbound(other, at(4))
    other_thread = world.outbound(other, at(5))
    violations = run(session, settings)
    assert len(violations) == 2, violations
    for violation, message in zip(violations, (same_thread, other_thread), strict=True):
        assert violation.split(" ")[1] == "message_after_freeze:"
        assert f"message #{message.id}" in violation and f"scam message #{scam.id}" in violation


def test_scam_label_without_a_freeze_does_not_block_replies(session, settings, world):
    thread = world.thread(world.item())
    world.inbound(thread, at(0), "Can you ship it? I'll e-transfer first", intent="scam")
    world.outbound(thread, at(1), "Sorry, local pickup and cash only.")
    assert run(session, settings) == []


def test_flagged_buyer_counts_as_frozen(session, settings, world):
    buyer = world.buyer()
    thread = world.thread(world.item(), buyer)
    world.inbound(thread, at(0), "Pay by cashier's cheque for more than asking", intent="scam")
    buyer.flagged_scam = True
    world.outbound(thread, at(1))
    only(run(session, settings), "message_after_freeze")


# Address


def _committed_thread(world: World, *, book: bool = True) -> tuple[Thread, Commitment]:
    thread = world.thread(world.item())
    world.inbound(thread, at(0))
    accepted = world.accept(thread, at(10))  # undo deadline at(20)
    if book:
        world.book(accepted, at(15))
    return thread, accepted


@pytest.mark.parametrize(
    "book, delivered_at",
    [
        (False, at(30)),  # never booked
        (True, at(5)),  # before the commitment existed
        (True, at(18)),  # booked, but inside the undo window
        (True, at(20)),  # exactly at the undo deadline
    ],
)
def test_address_delivered_too_early(session, settings, world, book, delivered_at):
    thread, _ = _committed_thread(world, book=book)
    sent_log = [(thread.remote_id, f"It's {ADDRESS.upper()}, buzz 2", delivered_at)]
    violation = only(run(session, settings, sent_log=sent_log), "address_early")
    assert repr(thread.remote_id) in violation


def test_address_delivered_after_release(session, settings, world):
    thread, accepted = _committed_thread(world)
    world.close(accepted, CommitmentKind.RELEASED, at(40))
    sent_log = [(thread.remote_id, "123 example street", at(45))]
    only(run(session, settings, sent_log=sent_log), "address_early")


def test_address_delivery_after_undo_deadline_is_fine(session, settings, world):
    thread, _ = _committed_thread(world)
    sent_log = [(thread.remote_id, f"See you at {ADDRESS}", at(21))]
    assert run(session, settings, sent_log=sent_log) == []


def test_without_sent_log_stored_placeholder_is_judged(session, settings, world):
    thread, _ = _committed_thread(world)
    message = world.outbound(thread, at(18), f"Address: {audit_module.PLACEHOLDER}")
    violation = only(run(session, settings), "address_early")
    assert f"message #{message.id}" in violation


def test_unredacted_address_in_storage(session, settings, world):
    thread, _ = _committed_thread(world)
    message = world.outbound(thread, at(30), f"Address: {ADDRESS}")
    assert f"message #{message.id}" in only(run(session, settings), "address_stored")


def test_audit_shares_no_code_with_guard_or_tools():
    tree = ast.parse(Path(audit_module.__file__).read_text())
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
         for alias in node.names]
    assert not [m for m in imported if m.startswith(("marketplace_agent.guard",
                                                      "marketplace_agent.tools"))]


# Bundles


def bundle(
    world: World,
    threads: list[Thread],
    when: datetime,
    prices: list[int],
    *,
    total: int | None = None,
    key: str = "bundle-1",
    reason: str | None = None,
) -> list[Commitment]:
    """ACCEPTED rows sharing a bundle_key, recording the agreed total as the ledger does."""
    total = sum(prices) if total is None else total
    items = ", ".join(str(thread.item_id) for thread in threads)
    if reason is None:
        reason = f"Bundle of items {items} for {total} cents in total."
    return [
        world._add(
            Commitment(
                item_id=thread.item_id, thread_id=thread.id, buyer_id=thread.buyer_id,
                kind=CommitmentKind.ACCEPTED, price_cents=price,
                undo_deadline=when + timedelta(minutes=10), reason=reason, bundle_key=key,
                created_at=when,
            )
        )
        for thread, price in zip(threads, prices, strict=True)
    ]


def ann_threads(world: World, *, chair_firm: bool = False) -> list[Thread]:
    """Ann's threads on a desk ($120, floor $95) and a chair ($50, floor $30, or firm)."""
    ann = world.buyer()
    desk = world.item()
    chair = world.item(
        list_cents=5000, floor_cents=3000,
        firmness=Firmness.FIRM if chair_firm else Firmness.SLIGHT,
    )
    return [world.thread(desk, ann), world.thread(chair, ann)]


def test_clean_bundle_history_reports_nothing(session, settings, world):
    threads = ann_threads(world)
    rows = bundle(world, threads, at(0), [10588, 4412])
    for row in rows:  # one shared slot is not an overlap
        world.book(row, at(10), THU_6PM)
    for row in rows:
        world.book(row, at(20), THU_6PM + timedelta(minutes=30))
    for row in rows:
        world.close(row, CommitmentKind.COMPLETED, at(hours=7))
    again = bundle(world, threads, at(5, hours=7), [9500, 3000], key="bundle-9")
    for row in again:
        world.close(row, CommitmentKind.UNDONE, at(10, hours=7))
    assert run(session, settings) == []


def test_bundle_prices_must_add_up_to_the_agreed_total(session, settings, world):
    bundle(world, ann_threads(world), at(0), [10500, 4400], total=15000)
    violation = only(run(session, settings), "bundle_total")
    assert "item prices add up to $149.00, not the agreed $150.00" in violation


@pytest.mark.parametrize("reason", ["", "Bundle of items 1, 2 for 15000 dollars."])
def test_bundle_must_record_its_total(session, settings, world, reason):
    bundle(world, ann_threads(world), at(0), [10500, 4500], reason=reason)
    assert "does not record one agreed total" in only(run(session, settings), "bundle_total")


@pytest.mark.parametrize(
    ("chair_firm", "chair_price", "rule"),
    [(False, 2900, "floor $30.00"), (True, 4900, "firm list price $50.00")],
)
def test_every_bundle_row_meets_its_own_items_minimum(
    session, settings, world, chair_firm, chair_price, rule
):
    rows = bundle(world, ann_threads(world, chair_firm=chair_firm), at(0), [13000, chair_price])
    violation = only(run(session, settings), "accept_below_minimum")
    assert f"ACCEPTED #{rows[1].id}" in violation and f"below the {rule}" in violation


def test_bundle_closed_in_part(session, settings, world):
    desk, chair = bundle(world, ann_threads(world), at(0), [10500, 4500])
    world.close(desk, CommitmentKind.RELEASED, at(30))
    violation = only(run(session, settings), "bundle_split")
    assert violation.startswith("2026-10-01T16:30:00Z")
    assert f"ACCEPTED #{desk.id}" in violation
    assert "had RELEASED at 2026-10-01T16:30:00Z" in violation
    assert f"ACCEPTED #{chair.id}" in violation and "had no such event" in violation


@pytest.mark.parametrize(
    ("desk_close", "chair_close"),
    [
        ((CommitmentKind.COMPLETED, 0), (CommitmentKind.RELEASED, 0)),
        ((CommitmentKind.RELEASED, 0), (CommitmentKind.RELEASED, 5)),
    ],
)
def test_bundle_closed_two_ways(session, settings, world, desk_close, chair_close):
    desk, chair = bundle(world, ann_threads(world), at(0), [10500, 4500])
    world.close(desk, desk_close[0], at(30 + desk_close[1]))
    world.close(chair, chair_close[0], at(30 + chair_close[1]))
    assert f"ACCEPTED #{chair.id}" in only(run(session, settings), "bundle_split")


@pytest.mark.parametrize("chair_slot", [None, SAT_11AM])
def test_bundle_booked_in_part_or_at_two_slots(session, settings, world, chair_slot):
    desk, chair = bundle(world, ann_threads(world), at(0), [10500, 4500])
    world.book(desk, at(10), THU_6PM)
    if chair_slot is not None:
        world.book(chair, at(10), chair_slot)
    violation = only(run(session, settings), "bundle_split")
    assert "had PICKUP_BOOKED at 2026-10-01T16:10:00Z for 2026-10-01T22:00:00Z" in violation


def test_bundle_shape(session, settings, world):
    desk, chair = ann_threads(world)
    lone = bundle(world, [desk], at(0), [10500], key="bundle-1")
    world.close(lone[0], CommitmentKind.RELEASED, at(5))
    bob_chair = world.thread(world.item(list_cents=5000, floor_cents=3000))
    mixed = bundle(world, [desk, bob_chair], at(10), [10500, 4500], key="bundle-2")
    for row in mixed:
        world.close(row, CommitmentKind.RELEASED, at(15))
    found = run(session, settings)
    assert len(found) == 2 and all(" bundle_shape: " in v for v in found), found
    assert "bundle bundle-1" in found[0] and "has 1 item, not 2 to 5" in found[0]
    assert "bundle bundle-2" in found[1] and "spans more than one buyer" in found[1]


def test_single_accept_on_a_bundled_item(session, settings, world):
    desk, chair = ann_threads(world)
    bundle(world, [desk, chair], at(0), [10500, 4500])
    world.accept(world.thread(world.item()), at(5))  # another item: fine
    late = world.accept(chair, at(10), 5000)
    violation = only(run(session, settings), "double_commitment")
    assert f"ACCEPTED #{late.id}" in violation
