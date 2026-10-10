from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlmodel import select

from marketplace_agent.core.items import (
    ItemNotFound,
    create_item,
    get_attributes,
    get_goal,
    get_item,
    item_photos,
    list_items,
    parse_hint_price,
    set_attributes,
    set_goal,
    touch,
)
from marketplace_agent.core.models import Firmness, Goal, ItemStatus, Photo


def test_create_and_get_item(session, clock):
    item = create_item(session, clock, "  IKEA desk $120 ")

    assert item.id is not None
    assert (item.hint, item.status) == ("IKEA desk $120", ItemStatus.DRAFT)
    assert item.created_at == item.updated_at == clock.now()
    assert get_item(session, item.id) is item


def test_get_missing_item_raises(session):
    with pytest.raises(ItemNotFound, match="Item 42 not found"):
        get_item(session, 42)


def test_list_items_filters_by_status(session, clock):
    desk = create_item(session, clock, "desk")
    chair = create_item(session, clock, "chair")
    chair.status = ItemStatus.LISTED

    assert list_items(session) == [desk, chair]
    assert list_items(session, ItemStatus.LISTED) == [chair]
    assert list_items(session, ItemStatus.SOLD) == []


def test_item_photos_ordered_by_position(session, clock):
    item = create_item(session, clock, "desk")
    for pos in (2, 0, 1):
        session.add(
            Photo(item_id=item.id, path=f"p{pos}.jpg", position=pos, width=1, height=1, sha256="")
        )
    session.flush()

    assert [p.path for p in item_photos(session, item.id)] == ["p0.jpg", "p1.jpg", "p2.jpg"]


@pytest.mark.parametrize(
    ("hint", "cents"),
    [
        ("$120", 12000),
        ("120$", 12000),
        ("120 dollars", 12000),
        ("$1,200.50", 120050),
        ("CAD 75", 7500),
        ("75 cad", 7500),
        ("C$75", 7500),
        ("120 bucks", 12000),
        ("asking 90", 9000),
        ("90 obo", 9000),
        ("$1.2k", 120000),
        ("4 chairs $40", 4000),
        ("2 $40 chairs", 4000),
        ("iPhone 12 $400", 40000),
        ("Desk $120, paid $300", 12000),
        ("IKEA desk 2 drawers", None),
        ("desk 120", None),
        ("4k monitor", None),
        ("$0", None),
        ("1.200", None),
        ("", None),
        # French Canadian thousands: space, NBSP, narrow NBSP
        ("desk $1 200", 120000),
        ("$1 200", 120000),
        ("desk $2 500 obo", 250000),
        ("$1 200", 120000),
        ("C$1 200", 120000),
        ("$1 200,50", 120050),
        ("asking 1 200", 120000),
        ("desk $120 200cm", 12000),  # a unit follows: two numbers
        # suffix only: "1 200 $" is $1,200 in French, "iPhone 12 400$" is $400 in English
        ("bureau 1 200 $", None),
        ("sofa 1 500$", None),
        ("bureau 1 200 $", None),
        ("iPhone 12 400$", None),
        ("desk 1 200", None),
        ("1 200 lamps $50", 5000),
        # implausible
        ("$10000000", 1_000_000_000),
        ("$10000000.01", None),
    ],
)
def test_parse_hint_price(hint, cents):
    assert parse_hint_price(hint) == cents


def test_parse_hint_price_refuses_more_digits_than_int_conversion_allows():
    assert parse_hint_price("desk $" + "9" * 4400) is None


def test_attributes_roundtrip(session, clock):
    item = create_item(session, clock, "desk")
    assert get_attributes(item) == {}

    set_attributes(item, {"width": "120 cm", "brand": " IKEA ", "model": "  "})

    assert get_attributes(item) == {"brand": "IKEA", "width": "120 cm"}


def test_get_attributes_coerces_non_string_values(session, clock):
    item = create_item(session, clock, "desk")
    item.attributes_json = '{"drawers": 2, "color": null}'

    assert get_attributes(item) == {"drawers": "2"}


def test_set_goal_defaults(session, clock):
    item = create_item(session, clock, "desk")
    clock.advance(hours=1)

    goal = set_goal(session, clock, item.id, list_price_cents=12000, floor_price_cents=9500)

    assert goal.id is not None
    assert (goal.target_price_cents, goal.firmness, goal.notes) == (12000, Firmness.SLIGHT, "")
    assert goal.deadline == clock.now() + timedelta(days=14)
    assert item.updated_at == clock.now()
    assert get_goal(session, item.id) is goal


def test_set_goal_upserts(session, clock):
    item = create_item(session, clock, "desk")
    first = set_goal(session, clock, item.id, list_price_cents=12000, floor_price_cents=9500)
    deadline = datetime(2026, 11, 1, 12, 0, tzinfo=timezone(timedelta(hours=-4)))

    second = set_goal(
        session,
        clock,
        item.id,
        list_price_cents=11000,
        floor_price_cents=9000,
        target_price_cents=10500,
        deadline=deadline,
        firmness=Firmness.FIRM,
        notes="must go",
    )

    assert second.id == first.id
    assert len(session.exec(select(Goal)).all()) == 1
    prices = (second.list_price_cents, second.floor_price_cents, second.target_price_cents)
    assert prices == (11000, 9000, 10500)
    assert second.deadline == deadline and second.deadline.tzinfo == UTC
    assert (second.firmness, second.notes) == (Firmness.FIRM, "must go")


@pytest.mark.parametrize(
    ("list_cents", "floor_cents", "target_cents", "message"),
    [
        (0, 0, None, "list price must be greater than zero"),
        (100, -5, None, "floor price must be greater than zero"),
        (100, 50, 0, "target price must be greater than zero"),
        (100, 150, None, "floor price cannot be above the list price"),
        (100, 90, 80, "floor price cannot be above the target price"),
    ],
)
def test_set_goal_rejects_bad_prices(
    session, clock, list_cents, floor_cents, target_cents, message
):
    item = create_item(session, clock, "desk")
    with pytest.raises(ValueError, match=message):
        set_goal(
            session,
            clock,
            item.id,
            list_price_cents=list_cents,
            floor_price_cents=floor_cents,
            target_price_cents=target_cents,
        )
    assert get_goal(session, item.id) is None


@pytest.mark.parametrize(
    ("deadline", "message"),
    [
        (datetime(2026, 12, 1), "timezone-aware"),
        (datetime(2026, 9, 1, tzinfo=UTC), "in the future"),
    ],
)
def test_set_goal_rejects_bad_deadline(session, clock, deadline, message):
    item = create_item(session, clock, "desk")
    with pytest.raises(ValueError, match=message):
        set_goal(
            session, clock, item.id, list_price_cents=100, floor_price_cents=50, deadline=deadline
        )


def test_set_goal_for_missing_item_raises(session, clock):
    with pytest.raises(ItemNotFound):
        set_goal(session, clock, 7, list_price_cents=100, floor_price_cents=50)


def test_touch_sets_updated_at(session, clock):
    item = create_item(session, clock, "desk")
    clock.advance(days=2)

    touch(item, clock)

    assert item.updated_at == clock.now()
    assert item.created_at < item.updated_at
