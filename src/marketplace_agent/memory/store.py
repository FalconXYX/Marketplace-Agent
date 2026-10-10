"""Scoped lessons the agent writes for itself: item, buyer, and global.

| Scope | Target | Recall order |
|---|---|---|
| item | item_id | Newest first |
| buyer | buyer_id | Newest first |
| global | none (item_id kept as provenance) | Keyword overlap with the item, uses, newest |

Outcome lessons (source "outcome", core/outcomes.py) are facts about one past item. They match on
their bracketed subject only, are recalled only when it shares a keyword with the item, and at
most MAX_OUTCOMES_RECALLED of them join the global lessons, so a long sales history never crowds
out the rest.
"""
import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from sqlmodel import Session, col, select

from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import Memory, MemoryScope

MAX_LESSON_CHARS = 300
DUPLICATE_RATIO = 0.9
OUTCOME_SOURCE = "outcome"
MAX_OUTCOMES_RECALLED = 5

_WORD = re.compile(r"[^\W_]+")
_SUBJECT = re.compile(r"\[([^\]]*)\]")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do for from has have if in into is it its of on or so "
    "than that the their them then there these they this to too was were will with you your "
    "when what which who why how not no yes very more most less just".split()
)


@dataclass
class Recalled:
    item: list[Memory]
    buyers: dict[int, list[Memory]]
    global_: list[Memory]


def remember(
    session: Session,
    clock: Clock,
    scope: MemoryScope,
    lesson: str,
    *,
    item_id: int | None = None,
    buyer_id: int | None = None,
    source: str = "agent",
) -> Memory | None:
    """Store a lesson; None when the same scope and target already hold a near duplicate.
    Lessons longer than MAX_LESSON_CHARS are cut at a word boundary."""
    text = _clip(" ".join(lesson.split()))
    if not text:
        raise ValueError("lesson is empty")
    if scope == MemoryScope.ITEM and item_id is None:
        raise ValueError("an item lesson needs item_id")
    if scope == MemoryScope.BUYER and buyer_id is None:
        raise ValueError("a buyer lesson needs buyer_id")
    known = _same_target(session, scope, item_id, buyer_id)
    if any(_near_duplicate(text, memory.lesson) for memory in known):
        return None
    memory = Memory(
        scope=scope,
        item_id=item_id,
        buyer_id=buyer_id,
        lesson=text,
        source=source,
        created_at=clock.now(),
    )
    session.add(memory)
    session.flush()
    return memory


def recall(
    session: Session,
    *,
    item_id: int,
    buyer_ids: Iterable[int],
    item_text: str = "",
    limit_item: int = 10,
    limit_buyer: int = 5,
    limit_global: int = 12,
) -> Recalled:
    """Lessons for one wake. Every buyer asked for gets an entry, possibly empty.
    Increments uses on the global lessons returned."""
    item = _newest(session, MemoryScope.ITEM, Memory.item_id == item_id, limit=limit_item)
    buyers = {
        buyer: _newest(session, MemoryScope.BUYER, Memory.buyer_id == buyer, limit=limit_buyer)
        for buyer in dict.fromkeys(buyer_ids)
    }
    global_ = _ranked_global(session, item_text, limit_global)
    for memory in global_:
        memory.uses += 1
        session.add(memory)
    session.flush()
    return Recalled(item=item, buyers=buyers, global_=global_)


def _same_target(
    session: Session, scope: MemoryScope, item_id: int | None, buyer_id: int | None
) -> list[Memory]:
    query = select(Memory).where(Memory.scope == scope)
    if scope == MemoryScope.ITEM:
        query = query.where(Memory.item_id == item_id)
    elif scope == MemoryScope.BUYER:
        query = query.where(Memory.buyer_id == buyer_id)
    return list(session.exec(query).all())


def _near_duplicate(a: str, b: str) -> bool:
    left, right = _normalize(a), _normalize(b)
    if left == right:
        return True
    matcher = difflib.SequenceMatcher(None, left, right)
    return (
        matcher.real_quick_ratio() >= DUPLICATE_RATIO
        and matcher.quick_ratio() >= DUPLICATE_RATIO
        and matcher.ratio() >= DUPLICATE_RATIO
    )


def _newest(session: Session, scope: MemoryScope, target: object, *, limit: int) -> list[Memory]:
    query = (
        select(Memory)
        .where(Memory.scope == scope, target)
        .order_by(col(Memory.created_at).desc(), col(Memory.id).desc())
        .limit(limit)
    )
    return list(session.exec(query).all())


def recall_outcomes(
    session: Session, text: str, *, limit: int = MAX_OUTCOMES_RECALLED
) -> list[Memory]:
    """Outcome lessons whose subject shares a keyword with text: most shared, then newest.
    Read only: uses are not counted."""
    if limit <= 0:
        return []
    wanted = _keywords(text)
    query = select(Memory).where(
        Memory.scope == MemoryScope.GLOBAL, Memory.source == OUTCOME_SOURCE
    )
    scored = [(len(wanted & _subject_keywords(m)), m) for m in session.exec(query).all()]
    ranked = sorted(
        ((score, m) for score, m in scored if score),
        key=lambda pair: (pair[0], pair[1].created_at, pair[1].id or 0),
        reverse=True,
    )
    return [m for _, m in ranked[:limit]]


def _ranked_global(session: Session, item_text: str, limit: int) -> list[Memory]:
    if limit <= 0:
        return []
    wanted = _keywords(item_text)
    lessons = session.exec(select(Memory).where(Memory.scope == MemoryScope.GLOBAL)).all()
    scored = [(len(wanted & _subject_keywords(m)), m) for m in lessons]
    ranked = sorted(
        ((score, m) for score, m in scored if score or m.source != OUTCOME_SOURCE),
        key=lambda pair: (pair[0], pair[1].uses, pair[1].created_at, pair[1].id or 0),
        reverse=True,
    )
    chosen: list[Memory] = []
    outcomes = 0
    for _, memory in ranked:
        if memory.source == OUTCOME_SOURCE:
            if outcomes >= MAX_OUTCOMES_RECALLED:
                continue
            outcomes += 1
        chosen.append(memory)
        if len(chosen) == limit:
            break
    return chosen


def _subject_keywords(memory: Memory) -> set[str]:
    """An outcome's subject is its first [bracketed] part; the rest is the same wording on every
    outcome and would match any item."""
    if memory.source == OUTCOME_SOURCE:
        subject = _SUBJECT.search(memory.lesson)
        return _keywords(subject[1]) if subject else set()
    return _keywords(memory.lesson)


def _normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def _keywords(text: str) -> set[str]:
    words = (_singular(word) for word in _WORD.findall(text.casefold()))
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _clip(text: str) -> str:
    if len(text) <= MAX_LESSON_CHARS:
        return text
    cut = text[: MAX_LESSON_CHARS + 1].rsplit(" ", 1)[0]
    return cut if cut and len(cut) <= MAX_LESSON_CHARS else text[:MAX_LESSON_CHARS]
