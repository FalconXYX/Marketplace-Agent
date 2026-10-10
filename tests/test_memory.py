from datetime import timedelta

import pytest
from sqlmodel import select

from marketplace_agent.core.models import Buyer, Item, Memory, MemoryScope
from marketplace_agent.memory.store import MAX_LESSON_CHARS, recall, remember


@pytest.fixture
def ids(session, clock) -> tuple[int, int, int, int]:
    items = [
        Item(title=t, created_at=clock.now(), updated_at=clock.now()) for t in ("Desk", "Chair")
    ]
    buyers = [Buyer(platform="fake", handle=h) for h in ("alice", "bob")]
    session.add_all(items + buyers)
    session.flush()
    return items[0].id, items[1].id, buyers[0].id, buyers[1].id


def _all(session) -> list[Memory]:
    return list(session.exec(select(Memory)).all())


# ---------------------------------------------------------------- remember


def test_remember_stores_a_cleaned_lesson(session, clock, ids):
    desk, _, _, _ = ids
    memory = remember(
        session, clock, MemoryScope.ITEM, "  Buyers ask about\n the drawers  ", item_id=desk
    )

    assert memory is not None and memory.id is not None
    assert (memory.lesson, memory.item_id, memory.source, memory.uses) == (
        "Buyers ask about the drawers",
        desk,
        "agent",
        0,
    )
    assert memory.created_at == clock.now()


@pytest.mark.parametrize(
    "duplicate",
    [
        "Buyers ask about the drawers",
        "buyers ask about the drawers.",
        "BUYERS   ask about the DRAWERS!!",
        "Buyers asks about the drawers",  # near-duplicate, ratio >= 0.9
        "Buyers ask about the drawer",
    ],
)
def test_near_duplicates_in_the_same_scope_and_target_are_dropped(session, clock, ids, duplicate):
    desk, _, _, _ = ids
    remember(session, clock, MemoryScope.ITEM, "Buyers ask about the drawers", item_id=desk)

    assert remember(session, clock, MemoryScope.ITEM, duplicate, item_id=desk) is None
    assert len(_all(session)) == 1


def test_different_lessons_and_targets_are_kept(session, clock, ids):
    desk, chair, alice, bob = ids
    lesson = "Buyers ask about the drawers"
    remember(session, clock, MemoryScope.ITEM, lesson, item_id=desk)

    assert remember(session, clock, MemoryScope.ITEM, "Buyers ask about the legs", item_id=desk)
    assert remember(session, clock, MemoryScope.ITEM, lesson, item_id=chair)
    assert remember(session, clock, MemoryScope.GLOBAL, lesson)
    assert remember(session, clock, MemoryScope.BUYER, "Haggles hard", buyer_id=alice)
    assert remember(session, clock, MemoryScope.BUYER, "Haggles hard", buyer_id=bob)
    assert remember(session, clock, MemoryScope.BUYER, "Haggles hard", buyer_id=alice) is None
    assert len(_all(session)) == 6


def test_global_lessons_dedupe_across_items(session, clock, ids):
    desk, chair, _, _ = ids
    remember(
        session,
        clock,
        MemoryScope.GLOBAL,
        "Dimensions in the title help furniture sell",
        item_id=desk,
    )
    assert (
        remember(
            session,
            clock,
            MemoryScope.GLOBAL,
            "dimensions in the title help furniture sell!",
            item_id=chair,
        )
        is None
    )


def test_scope_needs_its_target(session, clock):
    with pytest.raises(ValueError):
        remember(session, clock, MemoryScope.ITEM, "lesson")
    with pytest.raises(ValueError):
        remember(session, clock, MemoryScope.BUYER, "lesson")
    with pytest.raises(ValueError):
        remember(session, clock, MemoryScope.GLOBAL, "   \n ")


def test_long_lessons_are_cut_at_a_word_boundary(session, clock):
    memory = remember(session, clock, MemoryScope.GLOBAL, "word " * 100, source="correction")
    assert memory is not None and memory.source == "correction"
    assert len(memory.lesson) <= MAX_LESSON_CHARS
    assert memory.lesson.endswith("word")

    unbroken = remember(session, clock, MemoryScope.GLOBAL, "x" * 400)
    assert unbroken is not None and len(unbroken.lesson) == MAX_LESSON_CHARS


# ---------------------------------------------------------------- recall

DESK_LESSONS = [
    "Lowballers drop off after one firm reply",
    "Weekday evening pickups suit most buyers",
    "Mentioning the solid wood top drew offers",
    "Price drop to 110 doubled the views",
]


def test_recall_returns_item_and_buyer_lessons_newest_first(session, clock, ids):
    desk, chair, alice, bob = ids
    for lesson in DESK_LESSONS:
        remember(session, clock, MemoryScope.ITEM, lesson, item_id=desk)
        clock.advance(minutes=1)
    remember(session, clock, MemoryScope.ITEM, "chair only", item_id=chair)
    remember(session, clock, MemoryScope.BUYER, "alice is reliable", buyer_id=alice)
    clock.advance(minutes=1)
    remember(session, clock, MemoryScope.BUYER, "alice pays cash", buyer_id=alice)

    recalled = recall(session, item_id=desk, buyer_ids=[alice, bob, alice], limit_item=3)

    assert [m.lesson for m in recalled.item] == DESK_LESSONS[:0:-1]
    assert [m.lesson for m in recalled.buyers[alice]] == ["alice pays cash", "alice is reliable"]
    assert recalled.buyers[bob] == []
    assert list(recalled.buyers) == [alice, bob]


def test_recall_limits_buyer_lessons(session, clock, ids):
    _, _, alice, _ = ids
    for lesson in DESK_LESSONS:
        remember(session, clock, MemoryScope.BUYER, lesson, buyer_id=alice)
        clock.advance(minutes=1)
    assert len(recall(session, item_id=ids[0], buyer_ids=[alice], limit_buyer=2).buyers[alice]) == 2


def test_global_ranking_is_overlap_then_uses_then_recency(session, clock, ids):
    desk = ids[0]
    old_popular = remember(
        session, clock, MemoryScope.GLOBAL, "Weekend mornings get the most replies"
    )
    old_popular.uses = 10
    clock.advance(days=1)
    unrelated_new = remember(session, clock, MemoryScope.GLOBAL, "Short replies close deals faster")
    clock.advance(days=1)
    one_hit = remember(
        session, clock, MemoryScope.GLOBAL, "Desks sell better with a photo of the drawers open"
    )
    clock.advance(days=1)
    two_hits = remember(
        session, clock, MemoryScope.GLOBAL, "Put desk dimensions in the title for a standing desk"
    )
    session.flush()

    recalled = recall(session, item_id=desk, buyer_ids=[], item_text="Standing desk, 30x60, IKEA")

    assert [m.id for m in recalled.global_] == [
        two_hits.id,
        one_hit.id,
        old_popular.id,
        unrelated_new.id,
    ]


def test_recall_counts_uses_only_for_returned_global_lessons(session, clock, ids):
    lessons = []
    for n, text in enumerate(
        ["alpha lesson about sofas", "beta lesson about lamps", "gamma lesson about desks"]
    ):
        lessons.append(remember(session, clock, MemoryScope.GLOBAL, text))
        clock.advance(minutes=n + 1)
    item_lesson = remember(session, clock, MemoryScope.ITEM, "item lesson", item_id=ids[0])

    recalled = recall(session, item_id=ids[0], buyer_ids=[], item_text="desks", limit_global=2)

    assert [m.lesson for m in recalled.global_] == [
        "gamma lesson about desks",
        "beta lesson about lamps",
    ]
    assert [m.uses for m in lessons] == [0, 1, 1]
    assert item_lesson.uses == 0

    again = recall(session, item_id=ids[0], buyer_ids=[], item_text="", limit_global=2)
    assert [m.lesson for m in again.global_] == [
        "gamma lesson about desks",
        "beta lesson about lamps",
    ]
    assert recall(session, item_id=ids[0], buyer_ids=[], limit_global=0).global_ == []


def test_recall_on_an_empty_store(session, ids):
    recalled = recall(session, item_id=ids[0], buyer_ids=[ids[2]])
    assert (recalled.item, recalled.buyers, recalled.global_) == ([], {ids[2]: []}, [])


def test_keyword_overlap_ignores_stopwords_and_plurals(session, clock, ids):
    stop = remember(session, clock, MemoryScope.GLOBAL, "the and for with this that")
    clock.advance(minutes=1)
    plural = remember(session, clock, MemoryScope.GLOBAL, "chairs need close-up photos")
    clock.advance(timedelta(minutes=1))
    remember(session, clock, MemoryScope.GLOBAL, "newest unrelated lesson")

    recalled = recall(
        session, item_id=ids[0], buyer_ids=[], item_text="The chair for this room", limit_global=1
    )

    assert [m.id for m in recalled.global_] == [plural.id]
    assert stop.uses == 0
