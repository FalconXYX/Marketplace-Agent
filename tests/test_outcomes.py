"""core/outcomes.py: one line of facts per finished item, and where it is read back."""
import io
from datetime import timedelta

import pytest
from PIL import Image
from sqlmodel import Session, select

from marketplace_agent.agent.context import build_context
from marketplace_agent.ai.base import QuotaState
from marketplace_agent.ai.drafting import draft_item
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.control.service import ControlService
from marketplace_agent.core.items import create_item, get_goal, set_goal
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Memory,
    MemoryScope,
    Message,
    Photo,
    PriceChange,
    Thread,
    ThreadState,
)
from marketplace_agent.core.outcomes import (
    bundle_partners,
    outcome_line,
    record_outcome,
    record_outcomes,
)
from marketplace_agent.guard import ledger
from marketplace_agent.intake.photos import ingest_photo
from marketplace_agent.memory.store import (
    MAX_LESSON_CHARS,
    MAX_OUTCOMES_RECALLED,
    recall,
    recall_outcomes,
    remember,
)
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.tools.registry import default_registry

LIST, FLOOR = 12_000, 9_000


def _item(session, clock, title="IKEA Malm desk", category="Furniture", **kw) -> Item:
    item = Item(
        title=title,
        category=category,
        hint=kw.pop("hint", "desk $120"),
        description=kw.pop("description", "Solid."),
        status=kw.pop("status", ItemStatus.LISTED),
        created_at=clock.now(),
        updated_at=clock.now(),
        **kw,
    )
    session.add(item)
    session.flush()
    set_goal(session, clock, item.id, list_price_cents=LIST, floor_price_cents=FLOOR)
    return item


def _post(session, clock, item, platform, *, after=timedelta(), status=ListingStatus.LIVE):
    listing = Listing(
        item_id=item.id,
        platform=platform,
        remote_id=f"{platform}-{item.id}",
        url=f"https://{platform}.test/{item.id}",
        status=status,
        price_cents=LIST,
        title=item.title,
        description=item.description,
        published_at=clock.now() + after if status == ListingStatus.LIVE else None,
    )
    session.add(listing)
    session.flush()
    return listing


def _thread(session, clock, item, handle, *, inbound=1, outbound=0, platform="kijiji") -> Thread:
    buyer = session.exec(select(Buyer).where(Buyer.handle == handle)).first()
    if buyer is None:
        buyer = Buyer(platform=platform, handle=handle, display_name=handle.title())
        session.add(buyer)
        session.flush()
    thread = Thread(
        platform=platform,
        remote_id=f"{platform}-{handle}-{item.id}",
        item_id=item.id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    for n in range(inbound + outbound):
        session.add(
            Message(
                thread_id=thread.id,
                direction=Direction.INBOUND if n < inbound else Direction.OUTBOUND,
                body="Is it available?",
                sent_at=clock.now(),
            )
        )
    session.flush()
    return thread


def _reprice(session, clock, item, new_cents):
    goal = get_goal(session, item.id)
    old = goal.list_price_cents
    session.add(
        PriceChange(item_id=item.id, old_cents=old, new_cents=new_cents, created_at=clock.now())
    )
    goal.list_price_cents = new_cents
    session.add(goal)
    session.flush()


def _sell(session, clock, item, thread, price_cents) -> Commitment:
    accepted = ledger.accept(
        session,
        clock,
        item_id=item.id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=price_cents,
        undo_minutes=10,
    )
    return ledger.complete(session, clock, accepted_id=accepted.id)


def _outcomes(session) -> list[Memory]:
    return list(
        session.exec(select(Memory).where(Memory.source == "outcome").order_by(Memory.id)).all()
    )


@pytest.fixture
def sold(session, clock) -> Item:
    """Posted on kijiji then facebook, one price cut, three buyers wrote (one twice), sold."""
    item = _item(session, clock)
    before_any_post = clock.now() - timedelta(hours=1)
    session.add(
        PriceChange(item_id=item.id, old_cents=13_000, new_cents=LIST, created_at=before_any_post)
    )
    _post(session, clock, item, "kijiji")
    _post(session, clock, item, "facebook", after=timedelta(hours=1))
    _post(session, clock, item, "fake", status=ListingStatus.FAILED)
    alice = _thread(session, clock, item, "alice", inbound=2)
    _thread(session, clock, item, "alice", platform="facebook")
    _thread(session, clock, item, "bob")
    _thread(session, clock, item, "carol", inbound=0, outbound=1)
    clock.advance(days=3)
    _reprice(session, clock, item, 11_000)
    clock.advance(days=6, hours=12)
    _sell(session, clock, item, alice, 10_500)
    return item


# ---------------------------------------------------------------- content


def test_sold_line_holds_the_facts(session, clock, sold):
    memory = record_outcome(session, clock, sold.id)

    assert memory is not None
    assert (memory.scope, memory.source) == (MemoryScope.GLOBAL, "outcome")
    assert memory.item_id == sold.id
    assert memory.created_at == clock.now()
    assert memory.lesson == (
        "Sold [IKEA Malm desk, Furniture] for $105 on kijiji and facebook after 9.5 days: "
        "listed at $120, 1 price change, 2 buyers wrote."
    )


def test_written_exactly_once_per_item(session, clock, sold):
    first = record_outcome(session, clock, sold.id)
    clock.advance(days=1)

    assert record_outcome(session, clock, sold.id) is None
    assert record_outcomes(session, clock, sold.id, final_price_cents=1) == []
    assert _outcomes(session) == [first]


def test_nothing_is_written_for_an_item_still_for_sale(session, clock):
    item = _item(session, clock)

    assert record_outcome(session, clock, item.id) is None
    assert record_outcome(session, clock, 999) is None
    assert _outcomes(session) == []


def test_the_owners_price_beats_the_ledger(session, clock, sold):
    memory = record_outcome(session, clock, sold.id, final_price_cents=10_050)

    assert "for $100.50 on kijiji" in memory.lesson


def test_withdrawn_never_posted_and_unknown_price(session, clock):
    item = _item(session, clock, title="Oak chair", category=None, status=ItemStatus.WITHDRAWN)
    sold = _item(session, clock, title="Lamp", category="Lighting", status=ItemStatus.SOLD)

    assert outcome_line(session, clock, item) == (
        "Withdrawn unsold [Oak chair] (never posted): listed at $120, 0 price changes, "
        "0 buyers wrote."
    )
    assert outcome_line(session, clock, sold) == (
        "Sold [Lamp, Lighting] at an unrecorded price (never posted): listed at $120, "
        "0 price changes, 0 buyers wrote."
    )


def test_the_hint_stands_in_for_a_missing_title(session, clock):
    item = _item(
        session, clock, title="", category=None, hint="old bike $50", status=ItemStatus.WITHDRAWN
    )

    assert outcome_line(session, clock, item).startswith("Withdrawn unsold [old bike $50] ")


def test_a_long_title_is_shortened_to_keep_one_short_line(session, clock, sold):
    sold.title = "Solid oak [antique] writing desk with brass handles " * 8
    memory = record_outcome(session, clock, sold.id)

    assert len(memory.lesson) <= MAX_LESSON_CHARS and "\n" not in memory.lesson
    assert memory.lesson.startswith("Sold [Solid oak (antique) writing desk")
    assert "...] for $105 on kijiji and facebook after 9.5 days: listed at $120" in memory.lesson


def test_a_bundled_item_says_its_price_was_a_share(session, clock):
    desk, chair = _item(session, clock), _item(session, clock, title="Office chair")
    _post(session, clock, desk, "kijiji")
    thread = _thread(session, clock, desk, "dana")
    now = clock.now()
    shared = {"thread_id": thread.id, "buyer_id": thread.buyer_id, "bundle_key": "b1"}
    rows = [
        Commitment(
            item_id=item.id,
            kind=CommitmentKind.ACCEPTED,
            price_cents=price,
            undo_deadline=now,
            created_at=now,
            **shared,
        )
        for item, price in ((desk, 9_500), (chair, 4_500))
    ]
    session.add_all(rows)
    session.flush()
    for row in rows:
        session.add(
            Commitment(
                item_id=row.item_id,
                kind=CommitmentKind.COMPLETED,
                ref_id=row.id,
                price_cents=row.price_cents,
                created_at=now,
                **shared,
            )
        )
        session.get(Item, row.item_id).status = ItemStatus.SOLD
    session.flush()

    assert bundle_partners(session, desk.id) == [chair.id]
    written = record_outcomes(session, clock, desk.id)

    assert [m.item_id for m in written] == [desk.id, chair.id]
    assert "for $95 as its share of a 2-item bundle on kijiji" in written[0].lesson
    assert "for $45 as its share of a 2-item bundle (never posted)" in written[1].lesson


# ---------------------------------------------------------------- control hooks


@pytest.fixture
def market(clock) -> FakeMarketplace:
    return FakeMarketplace(clock)


@pytest.fixture
def control(engine, clock, settings, notifier, market) -> ControlService:
    return ControlService(engine, clock, settings, {"fake": market}, notifier, default_registry())


def _listed(session: Session, control: ControlService, clock, title="IKEA desk") -> int:
    item = _item(session, clock, title=title, category=None, status=ItemStatus.DRAFT)
    session.add(Photo(item_id=item.id, path="photos/1.jpg", width=640, height=480, sha256="ab"))
    session.commit()
    [result] = control.confirm_and_publish(item.id)
    assert result.ok, result.summary
    return item.id


def test_mark_sold_writes_the_outcome_once(session, control, clock):
    item_id = _listed(session, control, clock)
    clock.advance(days=4)

    control.mark_sold(item_id, 11_500)

    session.expire_all()
    [memory] = _outcomes(session)
    assert memory.lesson == (
        "Sold [IKEA desk] for $115 on fake after 4 days: listed at $120, 0 price changes, "
        "0 buyers wrote."
    )


def test_mark_sold_completes_the_deal_at_its_price(session, control, clock):
    item_id = _listed(session, control, clock)
    item = session.get(Item, item_id)
    thread = _thread(session, clock, item, "maya", platform="fake")
    ledger.accept(
        session,
        clock,
        item_id=item_id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=11_000,
        undo_minutes=10,
    )
    session.commit()

    control.mark_sold(item_id)

    session.expire_all()
    [memory] = _outcomes(session)
    assert "for $110 on fake" in memory.lesson and "1 buyer wrote" in memory.lesson


def test_withdraw_writes_the_outcome_with_the_last_asking_price(session, control, clock):
    item_id = _listed(session, control, clock)
    clock.advance(days=2)
    assert control.set_price(item_id, 9_500).ok
    clock.advance(days=12)

    control.withdraw(item_id)

    session.expire_all()
    [memory] = _outcomes(session)
    assert memory.lesson == (
        "Withdrawn unsold [IKEA desk] on fake after 14 days: listed at $120, last asked $95, "
        "1 price change, 0 buyers wrote."
    )


def test_the_owners_price_for_a_bundle_is_what_the_whole_bundle_fetched(session, control, clock):
    """The pickup check asks about the bundle and `agent sold` sells all of it, so the owner
    enters the cash they got for everything. Each item keeps its share, and the shares add up
    to that cash, not to it plus the partner's agreed price."""
    desk_id = _listed(session, control, clock)
    chair_id = _listed(session, control, clock, title="Office chair")
    set_goal(session, clock, chair_id, list_price_cents=6_000, floor_price_cents=4_000)
    thread = _thread(session, clock, session.get(Item, desk_id), "maya", platform="fake")
    ledger.accept_bundle(
        session,
        clock,
        item_ids=[desk_id, chair_id],
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        total_cents=18_000,  # $120 for the desk, $60 for the chair
        undo_minutes=10,
    )
    session.commit()
    clock.advance(days=3)

    control.mark_sold(desk_id, 15_000)  # haggled down at the door

    session.expire_all()
    desk, chair = _outcomes(session)
    assert (desk.item_id, chair.item_id) == (desk_id, chair_id)
    assert desk.lesson.startswith("Sold [IKEA desk] for $100 as its share of a 2-item bundle on")
    assert chair.lesson.startswith("Sold [Office chair] for $50 as its share of a 2-item bundle")


def test_a_withdrawn_item_the_owner_then_sells_is_remembered_as_sold(session, control, clock):
    item_id = _listed(session, control, clock)
    clock.advance(days=2)
    control.withdraw(item_id)  # a friend says they will take it
    clock.advance(days=1)

    control.mark_sold(item_id, 9_000)

    session.expire_all()
    [memory] = _outcomes(session)
    assert memory.lesson == (
        "Sold [IKEA desk] for $90 on fake after 3 days: listed at $120, 0 price changes, "
        "0 buyers wrote."
    )
    assert memory.created_at == clock.now()
    assert record_outcome(session, clock, item_id) is None  # still once per item


# ---------------------------------------------------------------- read back


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 80, 40)).save(buffer, "JPEG")
    return buffer.getvalue()


_DRAFT = {
    "title": "IKEA Micke desk",
    "description": "White desk.",
    "category": "Furniture",
    "condition": "good",
    "attributes": [],
    "suggested_price_low_dollars": 90,
    "suggested_price_high_dollars": 110,
    "price_reasoning": "A similar desk sold for $105.",
    "uncertain": [],
}


def test_a_later_similar_items_draft_sees_the_outcome(session, settings, clock, sold):
    desk = record_outcome(session, clock, sold.id)
    shelf = _item(
        session, clock, title="Oak bookshelf", category="Shelving", status=ItemStatus.SOLD
    )
    other = record_outcome(session, clock, shelf.id)
    new = create_item(session, clock, "ikea desk $100")
    ingest_photo(session, settings, new.id, _jpeg())
    router = ScriptedProvider([_DRAFT])

    draft_item(session, router, settings, clock, new.id)

    [request] = router.requests
    assert "Your past sales" in request.prompt
    assert f"- {desk.lesson}" in request.prompt
    assert other.lesson not in request.prompt


def test_a_draft_shows_at_most_five_past_sales_best_match_first(session, settings, clock):
    lessons = []
    for n in range(7):
        item = _item(session, clock, title=f"Desk number{n}", status=ItemStatus.SOLD)
        lessons.append(record_outcome(session, clock, item.id).lesson)
        clock.advance(hours=1)
    exact = _item(session, clock, title="Walnut desk", status=ItemStatus.SOLD)
    best = record_outcome(session, clock, exact.id).lesson
    new = create_item(session, clock, "walnut desk")
    ingest_photo(session, settings, new.id, _jpeg())
    router = ScriptedProvider([_DRAFT])

    draft_item(session, router, settings, clock, new.id)

    lines = router.requests[0].prompt.splitlines()
    past = [line[2:] for line in lines if line.startswith("- Sold")]
    assert past == [best, *reversed(lessons[-4:])]


def test_an_address_in_an_old_hint_never_reaches_a_later_draft(session, settings, clock):
    """A never-drafted item's outcome names it by its hint, which may hold the pickup address."""
    hint = f"IKEA desk, pick up at {settings.pickup_address}"
    gone = _item(session, clock, title="", hint=hint, status=ItemStatus.WITHDRAWN)
    record_outcome(session, clock, gone.id)
    new = create_item(session, clock, "IKEA desk white $90")
    ingest_photo(session, settings, new.id, _jpeg())
    router = ScriptedProvider([_DRAFT])

    draft_item(session, router, settings, clock, new.id)

    prompt = router.requests[0].prompt
    assert "Your past sales" in prompt and "pick up at [PICKUP_ADDRESS]" in prompt
    assert settings.pickup_address not in prompt


def test_no_past_sales_section_without_a_match(session, settings, clock, sold):
    record_outcome(session, clock, sold.id)
    new = create_item(session, clock, "bike $50")
    ingest_photo(session, settings, new.id, _jpeg())
    router = ScriptedProvider([_DRAFT])

    draft_item(session, router, settings, clock, new.id)

    assert "Your past sales" not in router.requests[0].prompt


def test_a_later_similar_items_context_sees_the_outcome(session, settings, clock, sold):
    desk = record_outcome(session, clock, sold.id)
    shelf = _item(
        session, clock, title="Oak bookshelf", category="Shelving", status=ItemStatus.SOLD
    )
    other = record_outcome(session, clock, shelf.id)
    new = _item(session, clock, title="IKEA Micke desk", category="Furniture")

    text = build_context(
        session, settings, clock, new.id, [], quota_state=QuotaState.HEALTHY, tool_specs=[]
    ).text

    assert f"- {desk.lesson}" in text
    assert other.lesson not in text


# ---------------------------------------------------------------- recall ranking


def test_outcomes_match_on_their_subject_only(session, clock):
    sold = _item(
        session, clock, title="Oak bookshelf", category="Shelving", status=ItemStatus.SOLD
    )
    record_outcome(session, clock, sold.id)

    wording = "sold price change buyers wrote listed days kijiji facebook"
    assert recall_outcomes(session, wording) == []
    assert recall(session, item_id=sold.id, buyer_ids=[], item_text=wording).global_ == []
    assert len(recall_outcomes(session, "bookshelf")) == 1


def test_outcomes_never_crowd_out_other_global_lessons(session, clock):
    for n in range(MAX_OUTCOMES_RECALLED + 3):
        item = _item(session, clock, title=f"Desk {n}", status=ItemStatus.SOLD)
        record_outcome(session, clock, item.id)
    lesson = remember(session, clock, MemoryScope.GLOBAL, "Dimensions in the title get replies")
    new = _item(session, clock, title="Standing desk")

    recalled = recall(session, item_id=new.id, buyer_ids=[], item_text="Standing desk")

    outcomes = [m for m in recalled.global_ if m.source == "outcome"]
    assert len(outcomes) == MAX_OUTCOMES_RECALLED
    assert lesson in recalled.global_
