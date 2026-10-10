import pytest
from sqlmodel import select

from marketplace_agent.ai.pattern_cache import (
    CACHEABLE_INTENTS,
    SEED_PATTERNS,
    learn,
    match,
    normalize,
    seed,
)
from marketplace_agent.ai.templates import availability_reply
from marketplace_agent.core.models import PatternEntry

FACEBOOK_DEFAULT = "Hi, is this available?"
KIJIJI_DEFAULT = "Hi, I am interested! Please contact me if this is still available."

UNSAFE = [
    "is this available would you take 40",
    "is this available? when can I pick up",
    "still available? 416-555-0100",
    "is the desk available or the chair",
    "Is this available? I'll take it",
    "is this still available, $80 cash?",
    "Is this available today?",
    "Still available? I can come Saturday",
    "is this available? 5pm works",
    "is this available? www.example.com",
    "is this available? email me jo@example.com",
    "Is this available? Is it?",
    "Is this available, is the price negotiable",
]

# Within NEAR_MATCH_RATIO of a seed, but the extra word is not a typo and changes the question.
NOT_A_TYPO = [
    "Do you still have two?",  # quantity
    "Do you still have six?",
    "Do you still have ten?",
    "Are you still selling both?",
    "Do you still have these?",
    "Do you still have them?",
    "Is the other still available?",  # another item
    "Do you still have mine?",
    "Is the bed still available?",  # an item noun may name a different listing
    "Is this desk still available?",
    "Is that hat still available?",  # "hat" is one letter from "that"
    "Is it available tn?",  # tonight
    "still available tn?",
    "Is this still available tmw?",  # tomorrow
    "Is this still available tom?",
    "Is this still available tdy?",  # today
    "Is this still available eve?",
    "Is this still available neg?",  # negotiable
    "Is this still available nego?",
    "Is this still available ono?",  # or nearest offer
]


@pytest.fixture
def seeded(session, clock):
    seed(session, clock)
    return session


def _entry(session, text: str) -> PatternEntry | None:
    query = select(PatternEntry).where(PatternEntry.normalized == normalize(text))
    return session.exec(query).first()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hi, is this available?", "is this available"),
        ("  IS THIS STILL AVAILABLE??  ", "is this still available"),
        ("Hey there! Is it still available? Thanks 🙏", "is it still available"),
        ("Good morning, is this still available please", "is this still available"),
        ("Hiii is it available, thank you!", "is it available"),
        ("I’m interested 👍🏽", "im interested"),
        ("Ｉｓ ｔｈｉｓ ａｖａｉｌａｂｌｅ？", "is this available"),
        ("Est-ce disponible, café?", "est ce disponible cafe"),
        ("hiking boots, available?", "hiking boots available"),
        ("Hi!", ""),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_seed_is_idempotent(session, clock):
    added = seed(session, clock)

    assert added == len(session.exec(select(PatternEntry)).all()) > 0
    assert seed(session, clock) == 0
    entries = session.exec(select(PatternEntry)).all()
    assert len(entries) == added
    assert {e.source for e in entries} == {"seed"}
    assert {e.intent for e in entries} <= CACHEABLE_INTENTS
    assert all(e.created_at == clock.now() for e in entries)


@pytest.mark.parametrize("phrase", [p for group in SEED_PATTERNS.values() for p in group])
def test_every_seed_phrase_matches_itself(seeded, phrase):
    assert match(seeded, phrase, first_contact=True) == "availability"


@pytest.mark.parametrize(
    "text",
    [
        FACEBOOK_DEFAULT,
        KIJIJI_DEFAULT,
        "Is this still available??",
        "hi is this available",
        "Hello, is it still available? Thanks!",
        "is this stil available",  # typo: near match
        "Is that still avalable",
        "Are you still sellng this?",
        "Do you stil have it?",
    ],
)
def test_matches_platform_defaults_and_variants(seeded, text):
    assert match(seeded, text, first_contact=True) == "availability"


@pytest.mark.parametrize("text", UNSAFE + NOT_A_TYPO)
def test_refuses_messages_a_template_would_ignore(seeded, text):
    assert match(seeded, text, first_contact=True) is None


@pytest.mark.parametrize(
    "text", ["I'm interested.", "Interested!", "Is this still available?", "is this stil available"]
)
def test_no_match_after_first_contact(seeded, text):
    """After "I can do $90 if you pick up Saturday", "I'm interested." accepts the counteroffer."""
    assert match(seeded, text, first_contact=True) == "availability"
    hits = sum(e.hits for e in seeded.exec(select(PatternEntry)))

    assert match(seeded, text, first_contact=False) is None
    assert sum(e.hits for e in seeded.exec(select(PatternEntry))) == hits


@pytest.mark.parametrize("text", ["", "Hi!", "thanks", "🙂", "Where is pickup?", "Is this sold?"])
def test_no_match_for_other_messages(seeded, text):
    assert match(seeded, text, first_contact=True) is None


def test_hits_increment_on_exact_and_near_match(seeded):
    match(seeded, "Is this still available?", first_contact=True)
    match(seeded, "is this stil available", first_contact=True)

    assert _entry(seeded, "is this still available").hits == 2
    assert _entry(seeded, FACEBOOK_DEFAULT).hits == 0


def test_refused_match_counts_no_hit(seeded):
    match(seeded, "is this still available? would you take 40", first_contact=True)

    assert sum(e.hits for e in seeded.exec(select(PatternEntry))) == 0


def test_only_cacheable_intents_match(session, clock):
    entry = PatternEntry(intent="offer", normalized="is this available", created_at=clock.now())
    session.add(entry)
    session.flush()

    assert match(session, "is this available", first_contact=True) is None


def test_learn_adds_safe_cacheable_phrasing(seeded, clock):
    text = "Do you still have the item?"
    assert match(seeded, text, first_contact=True) is None

    assert learn(seeded, clock, text, "availability") is True

    entry = _entry(seeded, text)
    assert (entry.source, entry.intent, entry.hits) == ("learned", "availability", 0)
    assert match(seeded, text, first_contact=True) == "availability"
    assert learn(seeded, clock, text, "availability") is False


@pytest.mark.parametrize("intent", ["offer", "scheduling", "question", "scam", "other"])
def test_learn_refuses_non_cacheable_intents(session, clock, intent):
    assert learn(session, clock, "Are you still selling it?", intent) is False
    assert _entry(session, "Are you still selling it?") is None


@pytest.mark.parametrize(
    "text",
    [
        *UNSAFE,
        *NOT_A_TYPO,
        KIJIJI_DEFAULT,  # two clauses: exact match only
        "is this item still available for sale in the store do you have it",  # too long
        "Is the blue one still available?",  # item specific: words outside the seeds
        "Hi!",
    ],
)
def test_learn_refuses_unsafe_text(session, clock, text):
    assert learn(session, clock, text, "availability") is False
    assert session.exec(select(PatternEntry)).first() is None


def test_available_reply_names_pickup_area_only():
    reply = availability_reply(committed=False, pickup_area=" Bloor St W & Ossington Ave ")

    assert reply == "Hi! Yes, it's still available. Pickup is near Bloor St W & Ossington Ave."


def test_available_reply_without_pickup_area():
    assert availability_reply(committed=False, pickup_area="") == "Hi! Yes, it's still available."


def test_committed_reply_offers_to_follow_up():
    reply = availability_reply(committed=True, pickup_area="Bloor St W & Ossington Ave")

    assert "pending" in reply and "falls through" in reply
    assert "Bloor" not in reply


@pytest.mark.parametrize("committed", [True, False])
def test_replies_never_mention_price_or_use_em_dashes(committed):
    reply = availability_reply(committed=committed, pickup_area="Queen & Spadina")

    assert not any(c.isdigit() for c in reply)
    assert "$" not in reply and "—" not in reply and "price" not in reply.lower()
