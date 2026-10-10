import re
import time as time_module
from datetime import UTC, datetime, time, timedelta

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from sqlmodel import Session

from marketplace_agent.config import AvailabilityWindow, Settings
from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    Buyer,
    Direction,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Message,
    Thread,
    ThreadState,
)
from marketplace_agent.guard import ledger
from marketplace_agent.guard.invariants import (
    ADDRESS_PLACEHOLDER,
    availability_text,
    check_listing_text,
    check_outbound_text,
    contains_contact_info,
    detect_scam,
    leaks_address,
    money_amounts_cents,
    offer_cents,
    offers_remote_deal,
    seller_money_cents,
    outbound_count_today,
    outbound_streak,
    parse_local_datetime,
    publish_count_today,
    redact_address,
    to_local,
    within_availability,
)

# ---------------------------------------------------------------- money

MONEY = [
    ("$80", [8000]),
    ("$ 80", [8000]),
    ("80$", [8000]),
    ("80 $", [8000]),
    ("80 bucks", [8000]),
    ("80 dollars", [8000]),
    ("80 CAD", [8000]),
    ("CA$80", [8000]),
    ("$80.50", [8050]),
    ("$1,200.50", [120050]),
    ("$1,200", [120000]),
    ("1.2k", [120000]),
    ("$2k", [200000]),
    ("would you take 80", [8000]),
    ("Would you take 80 for it?", [8000]),
    ("would you be willing to take 80?", [8000]),
    ("will you accept 75", [7500]),
    ("do 80?", [8000]),
    ("Can you do 85?", [8500]),
    ("can you do it for 85", [8500]),
    ("80?", [8000]),
    ("Hi! 80?", [8000]),
    ("offer 80", [8000]),
    ("I can offer 70", [7000]),
    ("80 cash", [8000]),
    ("80 cash today", [8000]),
    ("80 firm", [8000]),
    ("80 obo", [8000]),
    ("70 or best offer", [7000]),
    ("I can do 75 tonight", [7500]),
    ("how about 70", [7000]),
    ("what about 65?", [6500]),
    ("I'd pay 80", [8000]),
    ("I'll give you 80", [8000]),
    ("best I can do is 70", [7000]),
    ("my final offer: 65", [6500]),
    ("70 is my max", [7000]),
    ("Is 80 ok?", [8000]),
    ("Would 85 work?", [8500]),
    ("Could I get it for 80?", [8000]),
    ("both for 150", [15000]),
    ("Would you let it go for 90", [9000]),
    ("would you take 2000", [200000]),
    ("Can you do 1,200?", [120000]),
    ("would you take 4k?", [400000]),
    ("6 cash", [600]),
    ("Sorry, 50 is too low for me", [5000]),
    ("60 won't work", [6000]),
    ("Lowest I'd go is 85", [8500]),
    ("I'm firm at 95", [9500]),
    ("Asking 120", [12000]),
    ("Price: 120", [12000]),
    ("I paid 300 for it", [30000]),
    ("$5 off", [500]),
    ("I can't do 80, how about 70?", [8000, 7000]),
    ("you're asking $100, would you take 80?", [10000, 8000]),
    ("Hi, would you take 60 for it? I can pick up at 5:30 tomorrow", [6000]),
    ("call 416 555 0100 if 90 works", [9000]),
    ("$0", [0]),
    # Agreeing to a price states it.
    ("Deal at 80!", [8000]),
    ("Sure, 80 is fine.", [8000]),
    ("Ok, 80 it is!", [8000]),
    ("80 would work for me.", [8000]),
    ("Yours for 80", [8000]),
    ("it's yours for 80, come grab it", [8000]),
    ("Sold for 80!", [8000]),
    ("80 sounds good", [8000]),
    ("Happy with 80", [8000]),
    ("I am ok with 80", [8000]),
    ("agreed at 80", [8000]),
    ("I can go to 80", [8000]),
    ("I would let it go at 80", [8000]),
    ("make it 80 and it's a deal", [8000]),
    # Other ways to mark money.
    ("80,00 $", [8000]),
    ("$80,00", [8000]),
    ("\u20ac80", [8000]),
    ("\uff0480", [8000]),
    ("fifty bucks", [5000]),
    ("ninety-five dollars", [9500]),
    ("a hundred and ten bucks", [11000]),
]

NOT_MONEY = [
    "5:30",
    "can you come at 5:30?",
    "17:00 works",
    "416-555-0100",
    "(416) 555-0100",
    "416.555.0100",
    "+1 416 555 0100",
    "4165550100",
    "call 555-0100",
    "30x60 inches",
    '30" x 60"',
    "30 x 60 x 29",
    "30\u00d760 desk",
    "it is 30 inches wide",
    "6ft tall",
    "2020 model",
    "it's a 2020 model, bought in 2021",
    "2 chairs",
    "2 chairs and 1 table",
    "pick up at 6",
    "can I pick up at 6 tonight?",
    "apt 4",
    "unit 12, buzz 1204",
    "can you do 6pm",
    "can you do 6 pm tomorrow",
    "I can do 6 tonight",
    "Do 2 of the drawers work?",
    "how about 6?",
    "can you do 2000?",
    "Oct 15 works",
    "on 10/15",
    "October 3rd",
    "the 12th",
    "I have 2 kids",
    "4k TV",
    "iPhone 12 Pro",
    "PS5",
    "for kids under 5",
    "used for 3 months",
    "about 2 years old",
    "it seats 6 people",
    "is it still available?",
    "my email is joe80@example.com",
    "see https://example.com/item/8080",
    "Mon-Fri 17:00-20:00",
    "I'll take it at 7",
    "lmk by 9",
    "see you at five thirty",
    "half past six works",
    "a 27-inch monitor",
    "one of them is scratched",
    "someone else asked first",
    "Saturday at 6 sounds good",
    "2020 is fine",
]


@pytest.mark.parametrize(("text", "expected"), MONEY)
def test_money_amounts_reads_marked_and_offer_phrased_numbers(text, expected):
    assert money_amounts_cents(text) == expected


@pytest.mark.parametrize("text", NOT_MONEY)
def test_money_amounts_ignores_times_dates_phones_dimensions_and_counts(text):
    assert money_amounts_cents(text) == []


OFFERS = [
    ("would you take 80", 8000),
    ("do 80?", 8000),
    ("80?", 8000),
    ("$80?", 8000),
    ("80 cash today", 8000),
    ("80 firm", 8000),
    ("I can do 75 tonight", 7500),
    ("I can't do 80, how about 70?", 7000),
    ("you're asking $100, would you take 80?", 8000),
    ("Would you take 60? I can pick up at 5:30", 6000),
    ("Is $80 your lowest? I could do $70 cash", 7000),
    ("1.2k?", 120000),
    ("Deal at 105", 10500),
    ("I'd be happy with 90", 9000),
    ("would you take seventy-five?", 7500),
]
NOT_OFFERS = [
    "Is $80 your lowest?",
    "I saw one listed for $90 on Kijiji",
    "I paid 300 for mine",
    "is it still available?",
    "can I pick up at 6?",
    "2 chairs",
    "Is it under 30 inches?",
    "$70-80 is what they go for",
]


@pytest.mark.parametrize(("text", "expected"), OFFERS)
def test_offer_cents_is_the_last_offer_phrased_amount(text, expected):
    assert offer_cents(text) == expected


@pytest.mark.parametrize("text", NOT_OFFERS)
def test_offer_cents_is_none_for_questions_prices_and_non_money(text):
    assert offer_cents(text) is None


@hyp_settings(max_examples=200, database=None, deadline=None)
@given(
    dollars=st.integers(min_value=13, max_value=99_999),
    cents=st.integers(min_value=0, max_value=99),
)
def test_money_round_trips_any_amount(dollars, cents):
    assert money_amounts_cents(f"would you take {dollars}") == [dollars * 100]
    assert offer_cents(f"${dollars:,}.{cents:02d}?") == dollars * 100 + cents
    assert money_amounts_cents(f"{dollars} bucks") == [dollars * 100]


@hyp_settings(max_examples=200, database=None, deadline=None)
@given(
    hour=st.integers(min_value=1, max_value=12), minute=st.sampled_from(["00", "15", "30", "45"])
)
def test_times_never_read_as_money(hour, minute):
    for phrase in (
        f"can you do {hour}:{minute}?",
        f"do {hour}pm?",
        f"how about {hour}?",
        f"at {hour}",
    ):
        assert money_amounts_cents(phrase) == [], phrase


# Seller text: every bare number may be a price unless it reads as something else.
SELLER_MONEY = [  # phrasings the buyer-side reading does not count
    ("50 and done, come grab it", [5000]),
    ("Tell you what: 50 and we are square", [5000]),
    ("Final answer 50 if you come today", [5000]),
    ("Throw in the chair, 60 total", [6000]),
    ("IKEA Desk - only today 50, first come first served", [5000]),
    ("IKEA Desk - Sale 50!", [5000]),
]
SELLER_NOT_MONEY = [
    "Yes it's the iPhone 13 Pro, 128GB, battery at 91%",
    "It fits a Dell 27 monitor",
    "MacBook Pro 16, barely used",
    "Size 42, worn twice",
    "It's model 42 with the #3 bracket",
    "Canon 80D body only",
    "Takes a 27-inch monitor",
    "See you at five thirty, or 17:00 on Mon-Fri 17:00-20:00",
    "Saturday at 6 works",
    "Bought it in 2019, it's about 3 years old",
    "It has 2 drawers and seats 4 people",
    "It's 105 x 50 cm",
    "Pickup on Oct 15 or the 16th",
]


@pytest.mark.parametrize(("text", "expected"), SELLER_MONEY)
def test_seller_text_reads_any_bare_number_as_money(text, expected):
    assert money_amounts_cents(text) == []
    assert seller_money_cents(text) == expected


@pytest.mark.parametrize(("text", "expected"), MONEY)
def test_seller_text_reads_everything_buyer_text_does(text, expected):
    assert set(expected) <= set(seller_money_cents(text))


@pytest.mark.parametrize("text", SELLER_NOT_MONEY)
def test_seller_text_skips_names_labels_units_times_and_dates(text):
    assert seller_money_cents(text) == []


def test_seller_text_skips_numbers_in_the_pickup_area():
    assert seller_money_cents("Pickup near Yonge & 401") == [40100]
    assert seller_money_cents("Pickup near Yonge & 401", known=["Yonge & 401"]) == []


@hyp_settings(max_examples=200, database=None, deadline=None)
@given(
    dollars=st.integers(min_value=13, max_value=94),
    phrase=st.sampled_from(
        (
            "Deal at {}!",
            "Sure, {} is fine.",
            "Ok, {} it is!",
            "{} would work for me.",
            "Yours for {}",
            "Sold for {}!",
            "{} sounds good",
            "Happy with {}",
            "agreed at {}",
            "I can go to {}",
            "I would let it go at {}",
            "{} and it's yours",
            "Come get it, {}",
        )
    ),
)
def test_any_sub_floor_acceptance_is_caught_in_seller_text(dollars, phrase):
    if dollars in range(1900, 2100):
        return
    assert seller_money_cents(phrase.format(dollars)) == [dollars * 100]


# ---------------------------------------------------------------- contact info

CONTACT = [
    "call me at 416-555-0100",
    "(416) 555-0100",
    "+1 416 555 0100",
    "my number 4165550100",
    "text 555-0100",
    "joe.smith+ads@gmail.com",
    "joe (at) gmail.com",
    "joe at gmail dot com",
    "https://example.com/pay",
    "www.example.ca",
    "check bit.ly/abc123",
]
NOT_CONTACT = [
    "$1,200.50",
    "would you take 80",
    "pickup near Bloor St W & Ossington Ave",
    "30x60 inches",
    "available Mon-Fri 17:00-20:00",
    "I'm at work. Net is fine",
    "e.g. the desk",
    "[PICKUP_ADDRESS]",
]


@pytest.mark.parametrize("text", CONTACT)
def test_contact_info_found(text):
    assert contains_contact_info(text)


@pytest.mark.parametrize("text", NOT_CONTACT)
def test_contact_info_not_found(text):
    assert not contains_contact_info(text)


# ---------------------------------------------------------------- scams

FREEZE = [
    "Before I drive over I need to know you're real. I'll text you a code from Google Voice, "
    "just tell me what it says",
    "verify you're real with the code I sent",
    "Can you send me the 6 digit code I just texted you?",
    "I sent a verification code to your phone, what is it?",
    "what's the six digit number I sent",
    "Let's use google voice",
    "Tell me the code please",
    "my mover will collect it and I'll mail a cheque for extra",
    "My courier will pick it up, payment already sent through PayPal",
    "I'll send a certified cheque",
    "Can I pay with a money order?",
    "I overpaid by mistake, please send back the difference",
    "pay through this link",
    "Click the link to accept the payment",
    "Here's the payment link: https://paypa1-secure.com/claim",
    "Please log in to your bank so I can send the deposit",
    "What's your banking password",
    "Send your account number and transit number",
    "what's your SIN",
    "I need your social insurance number for the transfer",
    "Whats your email so I can send the payment through PayPal",
    "Can I pay in bitcoin?",
    "I'll pay with a gift card",
]
WARN = [
    ("Can you ship it to Calgary?", "ship"),
    ("Does UPS deliver there?", "ship"),
    ("I'll send a courier", "ship"),
    ("Can I e-transfer you now to hold it?", "advance"),
    ("can I send a deposit", "advance"),
    ("I can pay by PayPal", "advance"),
    ("can I pay by cheque?", "cheque"),
    ("text me on WhatsApp", "platform"),
    ("what's your number?", "platform"),
    ("My driver will pick it up and pay cash", "third party"),
    ("Check this out https://bit.ly/abc", "link"),
]
CLEAN = [
    "can I pick up with my truck",
    "does the lock come with the key",
    "I'll bring cash",
    "is the zip code area near the subway",
    "Is it still available?",
    "Can you confirm the pickup time?",
    "My son will pick it up and pay cash",
    "what's the buzzer code when I arrive?",
    "Can you give me the code for the building door?",
    "Does it have a 4 digit combination lock?",
    "I'll text you when I'm outside",
    "I'll give you a check",
    "What are the dimensions?",
    "Do you deliver? I can pay extra for gas",
    "I sin a lot",
    "Would you take 80 cash?",
    "Is the postal code M5V?",
    "Any scratches on the top?",
    "Can you send more pictures?",
]


@pytest.mark.parametrize("text", FREEZE)
def test_scams_freeze(text):
    signal = detect_scam(text)
    assert signal is not None and signal.severity == "freeze", signal


@pytest.mark.parametrize(("text", "hint"), WARN)
def test_risky_requests_warn(text, hint):
    signal = detect_scam(text)
    assert signal is not None and signal.severity == "warn", signal
    assert signal.reason


@pytest.mark.parametrize("text", CLEAN)
def test_ordinary_messages_are_not_scams(text):
    assert detect_scam(text) is None


# ---------------------------------------------------------------- address

LEAKS = [
    "123 Example St",
    "123 example st",
    "123 EXAMPLE ST",
    "123   Example    St.",
    "123 Example Street",
    "123 Example Ave",
    "123Example St",
    "123 example",
    "Come to 123 Example at 6",
    "Unit 4, 123 Example St",
    "4-123 Example St",
    "#4 123 example street",
    "1 2 3 Example St",
    "123, Example St",
]
SAFE = [
    "Example St",
    "near Example St",
    "1234 Example St",
    "23 Example St",
    "123 Examples Road",
    "123 Exampleton Ave",
    "Bloor St W & Ossington Ave",
    "would you take 123",
    ADDRESS_PLACEHOLDER,
]


@pytest.mark.parametrize("text", LEAKS)
def test_address_variants_leak(settings, text):
    assert leaks_address(text, settings)


@pytest.mark.parametrize("text", SAFE)
def test_non_address_text_does_not_leak(settings, text):
    assert not leaks_address(text, settings)


@pytest.mark.parametrize("text", LEAKS)
def test_redaction_removes_every_variant(settings, text):
    redacted = redact_address(text, settings)
    assert ADDRESS_PLACEHOLDER in redacted
    assert not leaks_address(redacted, settings)
    assert redact_address(redacted, settings) == redacted


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Come to 123 Example Street at 6", f"Come to {ADDRESS_PLACEHOLDER} at 6"),
        ("123 example st.", f"{ADDRESS_PLACEHOLDER}."),
        ("123 Example St or 123 example st", f"{ADDRESS_PLACEHOLDER} or {ADDRESS_PLACEHOLDER}"),
        ("near Example St", "near Example St"),
    ],
)
def test_redaction_output(settings, text, expected):
    assert redact_address(text, settings) == expected


def test_full_address_with_unit_city_and_postal_code(settings):
    full = settings.model_copy(
        update={"pickup_address": "Unit 4, 123 Example St, Toronto, ON M5V 2T6"}
    )

    for text in ("M5V 2T6", "m5v2t6", "123 Example", "unit 4 123 example st toronto on m5v 2t6"):
        assert leaks_address(text, full), text
    assert (
        redact_address("Unit 4, 123 Example St, Toronto, ON M5V 2T6!", full)
        == f"{ADDRESS_PLACEHOLDER}!"
    )
    assert not leaks_address("Toronto, ON", full)


def test_multi_word_street_names(settings):
    st_clair = settings.model_copy(update={"pickup_address": "77 St Clair Ave W"})

    assert leaks_address("77 st. clair avenue west", st_clair)
    assert not leaks_address("77 St George St", st_clair)
    assert redact_address("at 77 St Clair Avenue W", st_clair) == f"at {ADDRESS_PLACEHOLDER} W"


def test_redaction_repeats_until_no_copy_is_left(settings):
    highway = settings.model_copy(update={"pickup_address": "12 Highway 7"})
    text = "12 Highway 712 Highway 7 or 12highway7"

    redacted = redact_address(text, highway)

    assert redacted.count(ADDRESS_PLACEHOLDER) == 3
    assert "12" not in redacted and "highway" not in redacted.lower()
    assert not leaks_address(redacted, highway)


def test_no_pickup_address_configured_never_leaks(settings):
    blank = settings.model_copy(update={"pickup_address": ""})
    assert not leaks_address("123 Example St", blank)
    assert redact_address("123 Example St", blank) == "123 Example St"


_FRAGMENTS = st.sampled_from(
    [
        "123",
        "Example",
        "St",
        "Street",
        "example",
        " ",
        "  ",
        ",",
        "-",
        "#4",
        "unit",
        "123Example",
        "at 6",
        "x",
    ]
)


_HOME = Settings(_env_file=None, pickup_address="123 Example St", pickup_area="Bloor & Ossington")


@hyp_settings(max_examples=300, database=None, deadline=None)
@given(parts=st.lists(_FRAGMENTS, max_size=12))
def test_redacted_text_never_leaks(parts):
    text = "".join(parts)
    assert not leaks_address(redact_address(text, _HOME), _HOME)


@hyp_settings(max_examples=200, database=None, deadline=None)
@given(text=st.text(max_size=60))
def test_redaction_is_total_on_arbitrary_text(text):
    assert not leaks_address(redact_address(text, _HOME), _HOME)


# ---------------------------------------------------------------- untrusted input size


@pytest.mark.parametrize("unit", ["a.", "x", "1 ", "1", "a@", "1-", "word ", "$1 "])
def test_long_adversarial_text_is_scanned_in_linear_time(unit):
    """Buyer text is untrusted; 20k chars of "a.a.a." once took over 10 seconds."""
    text = unit * (20_000 // len(unit))
    started = time_module.perf_counter()
    for check in (contains_contact_info, detect_scam, money_amounts_cents, offer_cents):
        check(text)
    leaks_address(redact_address(text, _HOME), _HOME)
    assert time_module.perf_counter() - started < 2.0


# ---------------------------------------------------------------- time


def _at(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    ("start", "minutes", "expected"),
    [
        # Fri Oct 30 2026, EDT (UTC-4): window 17:00-20:00 local = 21:00-00:00 UTC
        (_at(2026, 10, 30, 21), 60, True),
        (_at(2026, 10, 30, 20, 59), 60, False),
        (_at(2026, 10, 30, 23), 60, True),  # ends exactly at close
        (_at(2026, 10, 30, 23), 61, False),
        # Mon Nov 2 2026, after fall-back, EST (UTC-5): 17:00 local = 22:00 UTC
        (_at(2026, 11, 2, 21, 30), 60, False),  # 16:30 local; true with a stale EDT offset
        (_at(2026, 11, 2, 22), 60, True),
        (_at(2026, 11, 3, 0), 60, True),  # 19:00-20:00 Monday local, Tuesday in UTC
        (_at(2026, 11, 3, 0, 30), 60, False),
        # Sun Nov 1 2026, the fall-back day: 10:00-18:00 EST = 15:00-23:00 UTC
        (_at(2026, 11, 1, 14, 30), 60, False),
        (_at(2026, 11, 1, 15), 60, True),
        # Spring forward Sun Mar 8 2026. Fri Mar 6 is EST, Mon Mar 9 is EDT.
        (_at(2026, 3, 6, 21), 60, False),
        (_at(2026, 3, 6, 22), 60, True),
        (_at(2026, 3, 9, 21), 60, True),
        (_at(2026, 3, 9, 20), 60, False),
        # Crossing from Friday evening into Saturday morning
        (_at(2026, 10, 30, 23, 30), 11 * 60, False),
    ],
)
def test_within_availability_default_windows(settings, start, minutes, expected):
    assert within_availability(settings, start, start + timedelta(minutes=minutes)) is expected


def test_interval_crossing_two_windows_is_outside(settings):
    split = settings.model_copy(
        update={
            "availability": [
                AvailabilityWindow(weekday=5, start=time(10), end=time(14)),
                AvailabilityWindow(weekday=5, start=time(14), end=time(18)),
                AvailabilityWindow(weekday=6, start=time(10), end=time(12)),
                AvailabilityWindow(weekday=6, start=time(13), end=time(15)),
            ]
        }
    )
    sat = datetime(2026, 10, 10, tzinfo=UTC) + timedelta(hours=4)  # local midnight, EDT
    sun = sat + timedelta(days=1)

    assert not within_availability(split, sat + timedelta(hours=13), sat + timedelta(hours=15))
    assert within_availability(split, sat + timedelta(hours=14), sat + timedelta(hours=15))
    assert within_availability(split, sat + timedelta(hours=10), sat + timedelta(hours=14))
    assert not within_availability(
        split, sun + timedelta(hours=11, minutes=30), sun + timedelta(hours=13, minutes=30)
    )
    assert not within_availability(split, sun + timedelta(hours=12), sun + timedelta(hours=13))


def test_window_spanning_the_fall_back_hour(settings):
    night = settings.model_copy(
        update={"availability": [AvailabilityWindow(weekday=6, start=time(0), end=time(4))]}
    )
    # Sun Nov 1 2026: 00:00 EDT = 04:00Z, 04:00 EST = 09:00Z; 01:30 happens twice.
    assert within_availability(night, _at(2026, 11, 1, 5, 30), _at(2026, 11, 1, 6, 30))
    assert within_availability(night, _at(2026, 11, 1, 4), _at(2026, 11, 1, 9))
    assert not within_availability(night, _at(2026, 11, 1, 8, 30), _at(2026, 11, 1, 9, 30))


def test_empty_or_reversed_intervals_are_never_available(settings):
    start = _at(2026, 10, 30, 22)
    assert not within_availability(settings, start, start)
    assert not within_availability(settings, start, start - timedelta(minutes=30))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-10T18:00", _at(2026, 10, 10, 22)),
        ("2026-10-10 18:00", _at(2026, 10, 10, 22)),
        (" 2026-11-02T18:00 ", _at(2026, 11, 2, 23)),
        ("2026-10-10T18:00Z", _at(2026, 10, 10, 18)),
        ("2026-10-10T18:00-07:00", _at(2026, 10, 11, 1)),
        ("2026-11-01T01:30", _at(2026, 11, 1, 5, 30)),  # ambiguous: first occurrence
    ],
)
def test_parse_local_datetime(settings, text, expected):
    parsed = parse_local_datetime(settings, text)
    assert parsed == expected
    assert parsed.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "tomorrow at 6",
        "6pm",
        "2026-10-10",
        "18:00",
        "2026-13-01T10:00",
        "2026-10-10T25:00",
        "soon",
        "2026-03-08T02:30",
    ],
)
def test_parse_local_datetime_rejects_junk_and_skipped_times(settings, text):
    with pytest.raises(ValueError):
        parse_local_datetime(settings, text)


def test_to_local_converts_and_refuses_naive(settings):
    local = to_local(settings, _at(2026, 10, 10, 22))
    assert (local.hour, local.utcoffset()) == (18, timedelta(hours=-4))
    with pytest.raises(ValueError):
        to_local(settings, datetime(2026, 10, 10, 22))


def test_availability_text(settings):
    assert availability_text(settings) == "Mon-Fri 17:00-20:00, Sat-Sun 10:00-18:00"
    custom = settings.model_copy(
        update={
            "availability": [
                AvailabilityWindow(weekday=0, start=time(13), end=time(17)),
                AvailabilityWindow(weekday=0, start=time(9), end=time(12)),
                AvailabilityWindow(weekday=2, start=time(18), end=time(20)),
                AvailabilityWindow(weekday=5, start=time(10), end=time(14)),
            ]
        }
    )
    assert availability_text(custom) == (
        "Mon 09:00-12:00 and 13:00-17:00, Wed 18:00-20:00, Sat 10:00-14:00"
    )
    assert (
        availability_text(settings.model_copy(update={"availability": []})) == "no pickup times set"
    )


# ---------------------------------------------------------------- counts


def _thread(
    session: Session, clock: FakeClock, platform: str = "fake", handle: str = "alice"
) -> Thread:
    item = Item(
        title="Desk", status=ItemStatus.LISTED, created_at=clock.now(), updated_at=clock.now()
    )
    buyer = Buyer(platform=platform, handle=handle)
    session.add_all([item, buyer])
    session.flush()
    thread = Thread(
        platform=platform,
        remote_id=f"t-{handle}",
        item_id=item.id,
        buyer_id=buyer.id,
        state=ThreadState.NEGOTIATING,
        created_at=clock.now(),
    )
    session.add(thread)
    session.flush()
    return thread


def _message(session, thread, direction, at, via="agent") -> Message:
    message = Message(thread_id=thread.id, direction=direction, body="hi", sent_at=at, via=via)
    session.add(message)
    session.flush()
    return message


def test_outbound_count_today_counts_automated_sends_since_utc_midnight(session, clock):
    clock.set(_at(2026, 10, 9, 15))
    thread = _thread(session, clock)
    other = _thread(session, clock, platform="kijiji", handle="bob")
    midnight = _at(2026, 10, 9, 0)
    _message(session, thread, Direction.OUTBOUND, midnight - timedelta(seconds=1))
    _message(session, thread, Direction.OUTBOUND, midnight)
    _message(
        session, thread, Direction.OUTBOUND, midnight + timedelta(hours=3), via="pattern_cache"
    )
    _message(session, thread, Direction.OUTBOUND, midnight + timedelta(hours=4), via="system")
    _message(session, thread, Direction.OUTBOUND, midnight + timedelta(hours=5), via="human")
    _message(session, thread, Direction.INBOUND, midnight + timedelta(hours=6), via="buyer")
    _message(session, other, Direction.OUTBOUND, midnight + timedelta(hours=7))

    assert outbound_count_today(session, clock, "fake") == 3
    assert outbound_count_today(session, clock, "kijiji") == 1
    assert outbound_count_today(session, clock, "facebook") == 0


def test_publish_count_today_counts_publishes_and_relists(session, clock):
    clock.set(_at(2026, 10, 9, 15))
    item = Item(
        title="Desk", status=ItemStatus.LISTED, created_at=clock.now(), updated_at=clock.now()
    )
    session.add(item)
    session.flush()
    today, yesterday = _at(2026, 10, 9, 1), _at(2026, 10, 8, 23)

    def listing(platform, published_at=None, relisted_at=None, status=ListingStatus.LIVE):
        session.add(
            Listing(
                item_id=item.id,
                platform=platform,
                price_cents=100,
                title="t",
                description="d",
                status=status,
                published_at=published_at,
                last_relisted_at=relisted_at,
            )
        )

    listing("kijiji", published_at=today)
    listing("kijiji", published_at=yesterday - timedelta(days=3), relisted_at=today)
    listing("kijiji", published_at=yesterday)
    listing("kijiji", status=ListingStatus.FAILED)
    listing("facebook", published_at=today)
    session.flush()

    assert publish_count_today(session, clock, "kijiji") == 2
    assert publish_count_today(session, clock, "facebook") == 1


def test_outbound_streak(session, clock):
    thread = _thread(session, clock)
    t0 = clock.now()
    assert outbound_streak(session, thread.id) == (0, None)

    _message(session, thread, Direction.OUTBOUND, t0)
    assert outbound_streak(session, thread.id) == (1, t0)

    _message(session, thread, Direction.INBOUND, t0 + timedelta(minutes=5), via="buyer")
    assert outbound_streak(session, thread.id) == (0, None)

    _message(session, thread, Direction.OUTBOUND, t0 + timedelta(minutes=6), via="human")
    _message(session, thread, Direction.OUTBOUND, t0 + timedelta(minutes=7))
    assert outbound_streak(session, thread.id) == (2, t0 + timedelta(minutes=7))


def test_outbound_streak_follows_arrival_order_not_stamps(session, clock):
    """A buyer reply that reached us after our message ends the streak, whatever its stamp."""
    thread = _thread(session, clock)
    t0 = clock.now()
    _message(session, thread, Direction.OUTBOUND, t0 + timedelta(microseconds=51))
    _message(session, thread, Direction.INBOUND, t0, via="buyer")  # minute-precision stamp

    assert outbound_streak(session, thread.id) == (0, None)


# ---------------------------------------------------------------- text rules


@pytest.fixture
def deal(session, clock):
    thread = _thread(session, clock)
    item = session.get(Item, thread.item_id)
    goal = set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
    return item, goal, thread


def _check(session, settings, clock, deal, body):
    item, goal, thread = deal
    return check_outbound_text(
        session, settings, clock, item=item, goal=goal, thread=thread, body=body
    )


def test_clean_message_passes(session, settings, clock, deal):
    body = (
        "Hi! Yes it's available. I could do 110, pickup near Bloor and Ossington, "
        "weekdays 17:00-20:00."
    )
    assert _check(session, settings, clock, deal, body) == []


def test_typed_address_is_always_a_violation_and_never_echoed(session, settings, clock, deal):
    reasons = _check(session, settings, clock, deal, "Come to 123 example street at 6")

    assert len(reasons) == 1
    assert ADDRESS_PLACEHOLDER in reasons[0]
    assert settings.pickup_area in reasons[0]
    assert "123" not in reasons[0] and "example" not in reasons[0].lower()


def test_placeholder_needs_a_booked_pickup_past_the_undo_window(session, settings, clock, deal):
    item, _, thread = deal
    body = f"See you at {ADDRESS_PLACEHOLDER}"

    assert "accept an offer" in _check(session, settings, clock, deal, body)[0]

    accepted = ledger.accept(
        session,
        clock,
        item_id=item.id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=10_000,
        undo_minutes=10,
    )
    assert "book_pickup" in _check(session, settings, clock, deal, body)[0]

    start = clock.now() + timedelta(days=1)
    ledger.book_pickup(
        session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(hours=1)
    )
    clock.advance(minutes=10)
    reasons = _check(session, settings, clock, deal, body)
    assert "undo" in reasons[0] and "12:10" in reasons[0]  # undo deadline in local time

    clock.advance(seconds=1)
    assert _check(session, settings, clock, deal, body) == []


def test_placeholder_is_blocked_in_a_thread_without_the_commitment(session, settings, clock, deal):
    item, goal, thread = deal
    rival = Thread(
        platform="fake",
        remote_id="t-rival",
        item_id=item.id,
        buyer_id=thread.buyer_id,
        created_at=clock.now(),
    )
    session.add(rival)
    session.flush()
    accepted = ledger.accept(
        session,
        clock,
        item_id=item.id,
        thread_id=thread.id,
        buyer_id=thread.buyer_id,
        price_cents=10_000,
        undo_minutes=0,
    )
    start = clock.now() + timedelta(days=1)
    ledger.book_pickup(
        session, clock, accepted_id=accepted.id, start=start, end=start + timedelta(hours=1)
    )
    clock.advance(minutes=1)

    reasons = check_outbound_text(
        session, settings, clock, item=item, goal=goal, thread=rival, body=f"{ADDRESS_PLACEHOLDER}"
    )
    assert len(reasons) == 1


def test_only_address_rule_reasons_mention_the_placeholder(session, settings, clock, deal):
    body = (
        "Come to 123 Example St or [PICKUP_ADDRESS], call 416-555-0100, I'd take $10, "
        "send me the code you got. " + "x" * 1000
    )
    reasons = _check(session, settings, clock, deal, body)

    address = [r for r in reasons if ADDRESS_PLACEHOLDER in r]
    assert len(address) == 2 and len(reasons) == 6


@pytest.mark.parametrize(
    "body",
    ["Text me at 416-555-0100", "email me: seller@example.com", "details at www.example.com"],
)
def test_contact_info_is_a_violation(session, settings, clock, deal, body):
    reasons = _check(session, settings, clock, deal, body)
    assert any("on the platform" in r for r in reasons)


@pytest.mark.parametrize(
    "body",
    [
        "Sorry, I can't do 80",
        "would you do $90?",
        "I can't go below 90",
        "50 is too low, sorry",
        "I could knock $10 off",
        # An injected model promising a sub-floor price, as it might word it.
        "Deal at 50! It's yours, when can you come by?",
        "Sure, 80 is fine.",
        "Ok, 80 it is!",
        "80 would work for me.",
        "Yours for 80",
        "Sold for 80!",
        "80 sounds good",
        "Happy with 80",
        "I am ok with 80",
        "agreed at 80",
        "I can go to 80",
        "I would let it go at 80",
        "80,00 $ et c'est \u00e0 vous",
        "\u20ac80 is fine",
        "fifty bucks and it's yours",
        "Fine, 50 and we're done",
    ],
)
def test_sub_floor_amounts_are_violations(session, settings, clock, deal, body):
    reasons = _check(session, settings, clock, deal, body)
    assert len(reasons) == 1 and "floor of $95.00" in reasons[0]


def test_amounts_at_or_above_floor_pass(session, settings, clock, deal):
    assert (
        _check(session, settings, clock, deal, "I can do 95 cash, or $120 delivered to the curb")
        == []
    )


def test_firm_price_rejects_anything_below_list(session, settings, clock, deal):
    _, goal, _ = deal
    goal.firmness = Firmness.FIRM
    reasons = _check(session, settings, clock, deal, "I could do 110")
    assert len(reasons) == 1 and "firm price of $120.00" in reasons[0]
    assert _check(session, settings, clock, deal, "It's firm at $120, thanks") == []


def test_scam_shaped_outbound_text_is_rejected_but_warnings_pass(session, settings, clock, deal):
    assert any(
        "scam" in r for r in _check(session, settings, clock, deal, "Sure, send me the code")
    )
    assert (
        _check(
            session, settings, clock, deal, "Sorry, no shipping or e-transfers, cash at pickup only"
        )
        == []
    )


# Guardrail 8 on the seller's own text: offering shipping or non-cash payment is rejected,
# declining it is not.
REMOTE_DEAL_OFFERS = [
    "Sure, I can ship it to you via Canada Post for $15 extra.",
    "Send me an e-transfer of $110 now to hold it and it's yours.",
    "E-transfer the deposit first please",
    "You can pay by PayPal before pickup.",
    "I can't hold it without a deposit",
    "No problem, I can ship it",
    "No, but I can mail it to you",
    "Can't ship, but I can hold it with a deposit",
    "Shipping available anywhere in Canada",
    "Local pickup or shipping at your cost",
    "UPS works for me",
    "Cash or e-transfer at pickup is fine",
    "I don\u2019t mind shipping it",
    "Happy to send it by courier",
    "Pay now and it's yours",
    "Etransfer is fine",
    "E-transfer works but not on weekends",
]
REMOTE_DEAL_DECLINES = [
    "Sorry, no shipping or e-transfers, cash at pickup only",
    "Sorry, no shipping, e-transfers, or deposits. Cash at pickup only.",
    "Shipping isn't possible, sorry",
    "Shipping is not available",
    "I can\u2019t do PayPal, cash only",
    "I don't ship, pickup near Bloor and Ossington",
    "Cash at pickup only \u2014 no e-transfers please",
    "Deposits are not needed, just come by",
    "No holds or deposits, first come first served",
    "Cash please rather than e-transfer",
    "No need to pay before pickup",
    "There are a few pick ups booked this week",
    "Let me check and get back to you",
]


@pytest.mark.parametrize("body", REMOTE_DEAL_OFFERS)
def test_offering_shipping_or_non_cash_payment_is_a_violation(session, settings, clock, deal, body):
    reasons = _check(session, settings, clock, deal, body)
    [reason] = [r for r in reasons if "other than cash at pickup" in r]
    assert "no shipping, e-transfers, PayPal, cheques, deposits, or paying ahead" in reason
    _, goal, _ = deal
    assert any(
        "other than cash at pickup" in r for r in check_listing_text(settings, goal=goal, text=body)
    )


@pytest.mark.parametrize("body", REMOTE_DEAL_DECLINES)
def test_declining_shipping_or_non_cash_payment_passes(session, settings, clock, deal, body):
    assert offers_remote_deal(body) is None
    assert _check(session, settings, clock, deal, body) == []


def test_the_suggested_decline_passes_every_rule(session, settings, clock, deal):
    [reason] = _check(session, settings, clock, deal, "Happy to ship it!")
    suggested = re.search(r'e\.g\. "([^"]+)"', reason)[1]
    assert suggested == "Sorry, no shipping, cash at pickup only."
    assert _check(session, settings, clock, deal, suggested) == []


def test_a_decline_does_not_cover_a_later_offer_in_another_sentence():
    assert offers_remote_deal("No shipping. Deposit holds it though.") == "Deposit"
    assert offers_remote_deal("Shipping isn't an option; e-transfer is fine") == "e-transfer"


@pytest.mark.parametrize(
    ("body", "ok"), [("", False), ("   \n", False), ("x" * 1000, True), ("x" * 1001, False)]
)
def test_message_length(session, settings, clock, deal, body, ok):
    assert (_check(session, settings, clock, deal, body) == []) is ok


def test_listing_text_rules(settings, clock):
    goal = Goal(
        item_id=1,
        list_price_cents=12_000,
        target_price_cents=12_000,
        floor_price_cents=9_500,
        deadline=clock.now() + timedelta(days=14),
    )

    assert (
        check_listing_text(
            settings, goal=goal, text="Solid oak desk, 30x60 inches, $120 or best offer."
        )
        == []
    )
    assert check_listing_text(settings, goal=goal, text="x" * 5000) == []
    assert len(check_listing_text(settings, goal=goal, text="x" * 5001)) == 1
    assert (
        ADDRESS_PLACEHOLDER
        in check_listing_text(settings, goal=goal, text=f"Pickup at {ADDRESS_PLACEHOLDER}")[0]
    )
    assert len(check_listing_text(settings, goal=goal, text="Pickup at 123 Example St")) == 1
    assert len(check_listing_text(settings, goal=goal, text="Call 416-555-0100")) == 1
    assert "floor" in check_listing_text(settings, goal=goal, text="Only $60!")[0]
    assert len(check_listing_text(settings, goal=goal, text="Pay through the link to reserve")) == 1
    goal.firmness = Firmness.FIRM
    assert "firm" in check_listing_text(settings, goal=goal, text="Asking 110")[0]


def test_listing_text_cannot_promise_a_sub_floor_price(settings, clock):
    goal = Goal(
        item_id=1,
        list_price_cents=12_000,
        target_price_cents=12_000,
        floor_price_cents=9_500,
        deadline=clock.now() + timedelta(days=14),
    )
    [reason] = check_listing_text(
        settings, goal=goal, text="IKEA Desk - yours for 50, sold for 50 to first taker"
    )
    assert "mentions $50.00, below the floor of $95.00" in reason
    assert "give it its unit" in reason
    title = "IKEA Micke Desk, White, 105 x 50 cm, fits a 27-inch monitor"
    assert check_listing_text(settings, goal=goal, text=title) == []
    area = settings.pickup_area
    assert check_listing_text(settings, goal=goal, text=f"Pickup near {area}") == []
