"""Answers "is this still available?" with a template and no model call.

A false positive is worse than a miss: replying "yes, available" to "available? would you take $40"
ignores an offer. So a message matches only when its normalized text equals a stored pattern, or
when it is short, single-clause, and within a typo of one.

| Check | Exact | Near match | Learn |
|---|---|---|---|
| First message in the thread | yes | yes | |
| No digits, currency, offer or time words, URL, email | yes | yes | yes |
| At most one question | yes | yes | yes |
| One clause, no conjunctions or question words | | yes | yes |
| At most 8 words | | yes | yes |
| Each word is in the pattern or a typo of one of its words | | yes | |
| Only words already in the seeds | | | yes |

Typos keep the meaning; added words change it. "Do you still have two?" asks about quantity,
"Is the bed still available?" may be about another listing, and "still available tmw?" is about
a time. Patterns apply to every item, so none of these can share the canned reply.
"""
import difflib
import re
import unicodedata
from functools import cache

from sqlmodel import Session, col, select
from sqlmodel.sql.expression import SelectOfScalar

from marketplace_agent.core.clock import Clock
from marketplace_agent.core.models import PatternEntry

CACHEABLE_INTENTS = frozenset({"availability"})
NEAR_MATCH_RATIO = 0.9
MAX_NEAR_WORDS = 8
TYPO_RATIO = 0.8
MIN_TYPO_CHARS = 4  # shorter words sit one letter from unrelated ones: "hat" and "that"

SEED_PATTERNS: dict[str, tuple[str, ...]] = {
    "availability": (
        "Hi, is this available?",  # Facebook Marketplace prefilled first message
        "Is this still available?",  # Facebook shortcut button, Kijiji app quick message
        "Hi, I am interested! Please contact me if this is still available.",  # Kijiji web form
        "Hi, I'm interested! Please contact me if this is still available.",
        "Is it still available?",
        "Is it available?",
        "Is this item still available?",
        "Is this item available?",
        "Is the item still available?",
        "Is that still available?",
        "Is this available still?",
        "Still available?",
        "Available?",
        "Is this still for sale?",
        "Is it still for sale?",
        "Still for sale?",
        "Do you still have this?",
        "Do you still have it?",
        "Are you still selling this?",
        "I'm interested in this item.",
        "I'm interested.",
        "Interested!",
    ),
}

_APOSTROPHES = frozenset("'‘’ʼ")
_GREETING = re.compile(
    r"^(?:(?:(?:hi+|hey+|hello+)(?: there)?|good (?:morning|afternoon|evening))(?: |$))+"
)
_SIGN_OFF = re.compile(r"(?:(?:^| )(?:thanks|thank you|thx|please|pls))+$")
_CLAUSE = re.compile(r"([^.?!;,\n]*)([.?!;,\n]*)")
_URL = re.compile(r"https?:|www\.|\w\.(?:com|ca|net|org|io|ly|co|me|app|link)\b")

_OFFER_WORDS = frozenset(
    "offer offering obo take price lowest negotiable negotiate firm cash pay dollar dollars buck "
    "bucks cad usd discount deal cheaper less trade swap transfer etransfer emt interac paypal "
    "venmo zelle".split()
)
_TIME_WORDS = frozenset(
    "today tonight tomorrow tmrw tmr weekend weekday morning afternoon evening night noon "
    "midnight pm oclock hour hours hr hrs minute minutes min mins asap now later soon week month "
    "monday tuesday wednesday thursday friday saturday sunday "
    "mon tue tues wed thu thur thurs fri sat sun".split()
)
_QUESTION_STARTERS = frozenset(
    "is are am do does did can could would will should may when where what how why which who "
    "any".split()
)
_CLAUSE_WORDS = frozenset(
    "and or but also if when because so then plus though although unless while nor btw cuz "
    "cause what where how why which who can could would will should may".split()
)


def normalize(text: str) -> str:
    """Casefolded words only: no accents, punctuation, or emoji, and no greeting or sign-off."""
    return _plain(_fold(text))


def seed(session: Session, clock: Clock) -> int:
    """Insert any SEED_PATTERNS not yet stored. Flushes; the caller commits."""
    known = set(session.exec(select(PatternEntry.normalized)).all())
    added = 0
    for intent, phrases in SEED_PATTERNS.items():
        for normalized in map(normalize, phrases):
            if normalized and normalized not in known:
                session.add(_entry(clock, intent, normalized, "seed"))
                known.add(normalized)
                added += 1
    session.flush()
    return added


def match(session: Session, text: str, *, first_contact: bool) -> str | None:
    """Intent of the stored pattern this message safely matches, counting the hit; else None."""
    entry = find(session, text, first_contact=first_contact)
    if entry is None:
        return None
    count_hit(session, entry)
    return entry.intent


def find(session: Session, text: str, *, first_contact: bool) -> PatternEntry | None:
    """The stored pattern this message safely matches, else None. Reads only, so a caller about
    to make a platform call can count the hit after it, not hold SQLite's write lock over it.

    first_contact: the thread has no outbound message yet. Later in a thread the same words can
    mean something else: "I'm interested." after a counteroffer accepts it.
    """
    if not first_contact:
        return None
    normalized = normalize(text)
    if not normalized or not _is_safe(text):
        return None
    entry = session.exec(_cacheable().where(PatternEntry.normalized == normalized)).first()
    if entry is None and _is_simple(text):
        entry = _nearest(session, normalized)
    return entry


def count_hit(session: Session, entry: PatternEntry) -> None:
    """Flushes; the caller commits."""
    entry.hits += 1
    session.add(entry)
    session.flush()


def learn(session: Session, clock: Clock, text: str, intent: str) -> bool:
    """Cache a phrasing the agent labeled. Flushes; the caller commits.

    Patterns apply to every item, so learned text may only recombine seed words: "is the blue
    one available" is right for one listing and wrong for the next.
    """
    normalized = normalize(text)
    if intent not in CACHEABLE_INTENTS or not normalized or not _is_simple(text):
        return False
    if not set(normalized.split()) <= _seed_vocabulary():
        return False
    exists = select(PatternEntry.id).where(PatternEntry.normalized == normalized)
    if session.exec(exists).first() is not None:
        return False
    session.add(_entry(clock, intent, normalized, "learned"))
    session.flush()
    return True


def _entry(clock: Clock, intent: str, normalized: str, source: str) -> PatternEntry:
    return PatternEntry(intent=intent, normalized=normalized, source=source, created_at=clock.now())


def _cacheable() -> SelectOfScalar[PatternEntry]:
    return select(PatternEntry).where(col(PatternEntry.intent).in_(CACHEABLE_INTENTS))


def _nearest(session: Session, normalized: str) -> PatternEntry | None:
    entries = {entry.normalized: entry for entry in session.exec(_cacheable())}
    if not entries:
        return None
    words = normalized.split()
    candidates = difflib.get_close_matches(
        normalized, entries, n=len(entries), cutoff=NEAR_MATCH_RATIO
    )
    for candidate in candidates:
        if _only_typos(words, frozenset(candidate.split())):
            return entries[candidate]
    return None


def _only_typos(words: list[str], pattern: frozenset[str]) -> bool:
    """Every word is in the pattern or a typo of one of its words."""
    return all(
        word in pattern
        or (
            len(word) >= MIN_TYPO_CHARS
            and bool(difflib.get_close_matches(word, pattern, n=1, cutoff=TYPO_RATIO))
        )
        for word in words
    )


def _is_safe(text: str) -> bool:
    """False when a canned reply would ignore something: an offer, a time, contact details,
    or a second question."""
    folded = _fold(text)
    words = set(normalize(text).split())
    return not (
        any(c.isdigit() or c == "@" or unicodedata.category(c) == "Sc" for c in folded)
        or _URL.search(folded)
        or words & (_OFFER_WORDS | _TIME_WORDS)
        or sum(_is_question(clause, asked) for clause, asked in _clauses(folded)) > 1
    )


def _is_simple(text: str) -> bool:
    """Short and single-clause: the only shape safe to fuzzy match or learn."""
    words = normalize(text).split()
    return (
        _is_safe(text)
        and len(words) <= MAX_NEAR_WORDS
        and len(_clauses(_fold(text))) == 1
        and not _CLAUSE_WORDS.intersection(words)
    )


def _is_question(clause: str, asked: bool) -> bool:
    return asked or clause.split()[0] in _QUESTION_STARTERS


def _clauses(folded: str) -> list[tuple[str, bool]]:
    """(plain words, ended with "?") per clause; clauses of only courtesy or emoji are skipped."""
    found = []
    for body, end in _CLAUSE.findall(folded):
        plain = _plain(body)
        if plain:
            found.append((plain, "?" in end))
    return found


def _fold(text: str) -> str:
    """Casefold and drop accents and apostrophes ("it's" -> "its"); other punctuation stays."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        c for c in decomposed if not unicodedata.combining(c) and c not in _APOSTROPHES
    ).casefold()


def _plain(folded: str) -> str:
    spaced = "".join(c if c.isalnum() else " " for c in folded)
    return _SIGN_OFF.sub("", _GREETING.sub("", " ".join(spaced.split()))).strip()


@cache
def _seed_vocabulary() -> frozenset[str]:
    phrases = (phrase for group in SEED_PATTERNS.values() for phrase in group)
    return frozenset(word for phrase in phrases for word in normalize(phrase).split())
