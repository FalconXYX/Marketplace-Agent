"""Hard rules on text, time, and counts. Pure checks: nothing here writes.

Money in text is read conservatively in one direction: an amount marked as money ($, €, £,
dollars, bucks, 1.2k, "80,00 $") always counts, a bare number counts only in offer or price
phrasing, and anything that reads as a time, date, phone number, dimension, quantity, or model
year never counts. Spelled-out numbers ("fifty bucks") read as their digits.

| Bare number | Money? |
|---|---|
| After an offer, price, or deal lead: "would you take 80", "asking 80", "yours for 80" | Yes |
| Before a firm trailer: "80 cash", "80 firm", "80 obo" | Yes, even 1-12 |
| After a weak lead, or before a weak or agreeing trailer: "do 80?", "80 is fine" | Unless a year |
| Alone in its clause: "80?" | Unless a year (1900-2099) |
| 1-12, unless before a firm trailer | No: an hour or a count ("can you do 6") |
| Before a unit or plural noun: "2 chairs", "30 inches" | No |

The "No" rows win over the "Yes" rows.

Text the seller sends (messages, listing text) is read the other way round, since a sub-floor
promise there is the harm and a false alarm only makes the model reword: there every bare number
is money unless it reads as a time, date, unit, count, year, or a model or label number ("iPhone
13", "size 42", "80D"). seller_money_cents does that reading.
"""
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_
from sqlmodel import Session, col, select

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.items import MAX_PRICE_CENTS
from marketplace_agent.core.models import (
    Direction,
    Firmness,
    Goal,
    Item,
    Listing,
    Message,
    Thread,
)
from marketplace_agent.guard.ledger import active_commitment_for_thread

ADDRESS_PLACEHOLDER = "[PICKUP_ADDRESS]"
MAX_MESSAGE_CHARS = 1000
MAX_LISTING_CHARS = 5000

_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class ScamSignal:
    severity: Literal["freeze", "warn"]
    reason: str


# ---------------------------------------------------------------- contact info

# Every unbounded run starts only where its run starts (the lookbehinds), which keeps scans
# linear: buyer text is untrusted and "a.a.a.a..." must not cost quadratic time.
_EMAIL = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|(?<![\w.+-])[\w.+-]+\s*(?:\(at\)|\[at\])\s*[\w-]+\s*(?:\.|\(dot\)|\[dot\]|\s+dot\s+)\s*"
    r"(?:com|ca|net|org)\b"
    r"|(?<![\w.+-])[\w.+-]+\s+at\s+[\w-]+\s*(?:\(dot\)|\[dot\]|\s+dot\s+)\s*"
    r"(?:com|ca|net|org)\b",
    re.IGNORECASE,
)
_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|(?<![\w.-])[\w-]+(?:\.[\w-]+)*\.(?:com|ca|net|org|io|ly|co|me|app|link|info|biz|xyz|shop"
    r"|site|online|gl|to)\b(?:/\S*)?",
    re.IGNORECASE,
)
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s*|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
    r"|(?<![\d$,.])\d{3}-\d{4}(?!\d)"
)


def contains_contact_info(text: str) -> bool:
    return any(pattern.search(text) for pattern in (_PHONE, _EMAIL, _URL))


# ---------------------------------------------------------------- money

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?(?![a-z])"
)
_DAY = r"(?:[12]\d|3[01]|0?[1-9])"
_LENGTH_UNIT = r"(?:\"|''|'|in(?:ch(?:es)?)?\.?|cm|mm|ft|feet|m)"
_TIMES = r"(?:x|\u00d7|\*|by)"  # 30x60, 30 by 60, and the multiplication sign
_MEASURE = rf"\d+(?:\.\d+)?\s*{_LENGTH_UNIT}?"
_DIMENSION = rf"(?<![\d.]){_MEASURE}\s*{_TIMES}\s*{_MEASURE}(?:\s*{_TIMES}\s*{_MEASURE})?"
_UNITS = r"(?:one|two|three|four|five|six|seven|eight|nine)"
_TEENS = r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
_TENS_WORD = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_HOUR_WORD = rf"(?:{_UNITS}|ten|eleven|twelve)"
_SPELLED_TIME = (
    rf"\b(?:{_HOUR_WORD}[\s-]+(?:o'?\s?clock|fifteen|thirty|forty[\s-]five|oh[\s-]{_UNITS})"
    rf"|(?:half|quarter)\s+(?:past|to|after)\s+{_HOUR_WORD})\b"
)
_NOT_MONEY_SPANS = re.compile(
    "|".join(
        (
            _URL.pattern,
            _EMAIL.pattern,
            _PHONE.pattern,
            r"(?<!\d)\d{1,2}(?::\d{2})+(?:\s*[ap]\.?m\b\.?)?",  # 5:30, 17:00, 5:30pm
            r"(?<!\d)\d{1,2}\s*(?:[ap]\.?m\b\.?|o'?\s?clock)",  # 6pm, 6 a.m., 6 o'clock
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # 2026-10-09
            r"(?<!\d)\d{1,2}/\d{1,2}(?:/\d{2,4})?(?!\d)",  # 10/09, 10/09/2026
            rf"\b{_MONTH}\s*{_DAY}(?:st|nd|rd|th)?(?!\d)(?:,?\s+\d{{4}})?",  # Oct 9, October 9th
            rf"(?<!\d){_DAY}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}",  # 9 Oct, 9th of October
            _DIMENSION,  # 30x60, 30" x 60", 30 x 60 x 29
            _SPELLED_TIME,  # five thirty, half past six
        )
    ),
    re.IGNORECASE,
)
_NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)(?!\d|[.,]\d)"
)
_BELOW_100 = rf"(?:{_TENS_WORD}(?:[\s-]+{_UNITS})?|{_TEENS}|{_UNITS})"
_HUNDREDS = rf"(?:a|{_UNITS})[\s-]+hundred(?:[\s-]+(?:and[\s-]+)?{_BELOW_100})?"
_BELOW_1000 = rf"(?:{_HUNDREDS}|{_BELOW_100})"
_SPELLED = re.compile(
    rf"\b(?:(?:a|{_BELOW_1000})[\s-]+thousand(?:[\s-]+(?:and[\s-]+)?{_BELOW_1000})?"
    rf"|{_BELOW_1000})\b"
)
_SMALL = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_DECADES = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_WORD_VALUES = (
    {word: n for n, word in enumerate(_SMALL)}
    | {word: 10 * n for n, word in enumerate(_DECADES, start=2)}
    | {"a": 1}
)
_CURRENCY_SIGNS = re.compile("[\u20ac\u00a3\uff04\ufe69]")  # euro, pound, wide dollar signs
# French Canadian "80,00 $" and "$80,00": a comma before exactly two digits next to a $ sign.
_DECIMAL_COMMA = re.compile(r"\$\s*\d+,\d{2}(?![\d,])|(?<![\d,])\d+,\d{2}(?=\s*\$)")
_THOUSANDS = re.compile(r"k(?![a-z])")
_CURRENCY_BEFORE = re.compile(r"(?:[a-z]{0,3}\$|\b(?:cad|usd|cdn))\s*$")
_CURRENCY_AFTER = re.compile(r"\s*(?:\$(?!\s*\d)|(?:dollars?|bucks?|cad|usd|cdn|loonies)(?![a-z]))")
_UNIT_AFTER = re.compile(
    r"(?:%|°|\"|''|'(?![a-z])|(?:in|ft|m|g|l|w|v|h|lb|kg|cm|mm|gb|tb|yo)(?![a-z])"
    r"|[\s-]*(?:inch(?:es)?|ft|feet|foot|cm|mm|km|kms|kgs?|lbs?|pounds?|oz|ml|litres?|liters?"
    r"|gal(?:lon)?s?|watts?|volts?|hz|gb|tb|mb|mah|mph|sq|pcs|pieces?|pack|seats?|seater|people"
    r"|persons?|years?|yrs?|months?|mos|weeks?|wks?|days?|hours?|hrs?|minutes?|mins?|seconds?"
    r"|secs?|times|miles?|views?|st|nd|rd|th|of|am|pm|a\.m|p\.m|o'?clock|x|\u00d7|percent|degrees?"
    r"|year[\s-]old)(?![a-z]))"
)
_PLURAL_AFTER = re.compile(
    r"\s+(?!(?:plus|bucks|dollars|cents|thanks|thx|this|is|was|its|yes|works|sounds|seems|has"
    r"|does|as|us|pls|plz|cuz|bc|less)\b)[a-z]{2,}s\b"
)
_OBJECT = r"(?:\s+(?:you|u|ya|it|this|that|them|both|all|everything|the\s+\w+))?"
_LEAD_TAIL = rf"{_OBJECT}(?:\s+(?:for|at))?(?:\s+(?:like|maybe|around|about|approx|~))?\s*$"
_SUPERLATIVE_TAIL = (
    r"(?:\s+(?:offer|price))?"
    r"(?:\s+(?:i|we)(?:\s+can|\s+could|\s+would|\s+will|'d|'ll)?"
    r"\s+(?:do|go|offer|pay|take|accept))?"
    r"(?:\s+(?:is|would\s+be)|\s*[=:])"
)
_LEAD = re.compile(
    r"\b(?:"
    r"(?P<offer>take|accept|pay|paying|offer(?:ing|\s+of)?|give|sell|deal|yours"
    r"|let\s+(?:it|this|them|that)\s+go\s+(?:for|at)|go\s+(?:as\s+low\s+as|down\s+to|for|to|at)"
    r"|agreed?(?:\s+(?:on|to))?|(?:happy|ok(?:ay)?|fine|good|cool)\s+with"
    r"|make\s+it|call\s+it|let'?s\s+(?:say|do)"
    r"|come\s+down\s+to|drop\s+(?:it\s+|the\s+price\s+)?to|lower\s+(?:it\s+|the\s+price\s+)?to"
    r"|settle\s+(?:for|on|at)|counter(?:\s+(?:at|with|of))?"
    rf"|(?:best|max|final|highest|top){_SUPERLATIVE_TAIL})"
    r"|(?P<price>asking|price(?:\s+is|\s+of|\s*:)|priced\s+at|listed\s+(?:at|for)|sold"
    r"|selling\s+(?:it\s+)?for|paid|bought\s+(?:it\s+)?for|retail(?:s|ed)?(?:\s+(?:for|at))?"
    r"|worth|originally|msrp"
    r"|costs?|firm\s+(?:at|on)"
    rf"|lowest{_SUPERLATIVE_TAIL})"
    r"|(?P<maybe>do|would|could|maybe|have|got|how\s+about|what\s+about|wbu|only|just"
    r"|meet\s+(?:you\s+)?(?:in\s+the\s+middle\s+)?at"
    r"|(?:it|this|that|them|both|all|everything)\s+for)"
    r"|(?P<bound>under|below|less\s+than|lower\s+than|at\s+least|more\s+than|over)"
    rf"){_LEAD_TAIL}"
)
_STRONG_TRAIL = re.compile(
    r"\s*(?:cash|firm|obo|o\.b\.o|or\s+best|final|tops|max|flat|even"
    r"|is\s+(?:my|the|our)\s+(?:best|final|max|lowest|limit|budget|offer|price))(?![a-z])"
)
# Judging a number names it as a price: "50 is too low", "60 won't work".
_PRICE_TRAIL = re.compile(
    r"\s*(?:(?:is|was|seems|feels|sounds)"
    r"\s+(?:a\s+bit\s+|way\s+|too\s+|kind\s+of\s+|pretty\s+|really\s+)*"
    r"(?:low|little|cheap|high|steep|much|a\s+lowball)"
    r"|(?:won'?t|doesn'?t|does\s+not|will\s+not)\s+work)(?![a-z])"
)
_WEAK_TRAIL = re.compile(
    r"\s*(?:ok(?:ay)?|work|works|good|fine|for\s+(?:it|this|that|them|both|all|the)|each"
    r"|tonight|today|tomorrow|now|then|instead)(?![a-z])"
)
# Agreeing to a number names it as the price: "80 is fine", "80 it is", "80 sounds good".
_ACCEPT_TRAIL = re.compile(
    r"\s*,?\s*(?:(?:is|'s|would\s+be|will\s+be)\s+(?:fine|good|great|perfect|ok(?:ay)?|fair"
    r"|doable|reasonable|acceptable|cool|no\s+problem)"
    r"|(?:it\s+is|it'?s)(?=\s*(?:[.!?,;\n]|$))|(?:and\s+)?(?:it'?s|its|it\s+is)\s+yours|deal|done"
    r"|(?:sounds|seems)\s+(?:good|great|fair|fine|ok(?:ay)?|right|reasonable|perfect)"
    r"|(?:would|will|should|could|can|does|do)\s+(?:work|do)|works?\s+for\s+me)(?![a-z])"
)
# A label or a product name before a number makes it a name, not a price, in seller text.
_LABEL = re.compile(
    r"(?:#|model|size|sz|no|number|nr|gen|generation|series|version|ver|v|mk|mark|rev|room|rm"
    r"|unit|apt|suite|ste|highway|hwy|route|rte|exit|bus|line|page|pg|level|lvl|grade|season"
    r"|part|pt|vol|volume|chapter|ch|type|class|item|lot|set|kit|ref|sku|id|age|ages|rated"
    r"|rating|track|disc|episode|ep|row|section|sec|seat|issue|edition|floor|storey|step|stage"
    r"|tier|zone|channel|code|batch|serial|lane|gate|door|bay|aisle|locker|box|bin|gauge|grit)"
    r"[.:#]?$"
)
_CUES = frozenset(
    "deal sold yours ok okay sure fine done price offer only just now how would can could i it "
    "its it's that this yes yeah great perfect happy cool take let make call asking firm cash "
    "final best lowest agreed go do pay give accept settle counter say about around like maybe "
    "is was be and or but so then hi hey thanks you me we our my your the a an at for to "
    "sale steal bargain special clearance reduced discount total cheap priced buy get grab "
    "all both each per".split()
)
_CLAUSE_BREAK = re.compile(r"[.!?\n;,]")
_FILLER = re.compile(
    r"(?:(?:ok(?:ay)?|so|um+|hmm+|hi|hey|hello|well|maybe|or|and|but|also|then|yes|yeah|sure"
    r"|please|pls)\b\s*)*"
)
_CLAUSE_END = re.compile(r"\s*(?:\?+|!+|\.)?\s*(?:[\n.!?;,]|$)")


@dataclass(frozen=True)
class _Amount:
    cents: int
    offer: bool  # phrased as an offer, not a stated price or bound


def money_amounts_cents(text: str) -> list[int]:
    return [amount.cents for amount in _amounts(text)]


def seller_money_cents(text: str, *, known: Sequence[str] = ()) -> list[int]:
    """Amounts in text the seller sends: every bare number that does not read as something else
    (see the module docstring). known: phrases whose numbers are never prices, such as the
    pickup area ("Yonge & 401")."""
    return [amount.cents for amount in _amounts(text, seller=True, known=known)]


def offer_cents(text: str) -> int | None:
    """The last amount phrased as an offer, so "can't do 80, how about 70?" gives 70."""
    offers = [amount.cents for amount in _amounts(text) if amount.offer]
    return offers[-1] if offers else None


def _amounts(text: str, *, seller: bool = False, known: Sequence[str] = ()) -> list[_Amount]:
    folded = text.casefold()
    for phrase in known:
        if phrase.strip():
            folded = folded.replace(phrase.casefold(), " " * len(phrase.casefold()))
    masked = _NOT_MONEY_SPANS.sub(lambda m: " " * len(m.group()), folded)
    masked = _SPELLED.sub(_as_digits, masked)
    masked = _CURRENCY_SIGNS.sub("$", masked)
    masked = _DECIMAL_COMMA.sub(lambda m: m.group().replace(",", "."), masked)
    original = text if seller and len(text) == len(folded) else None
    found = []
    for m in _NUMBER.finditer(masked):
        before = _words_before(masked, m.start())
        name = _name_before(masked, original, m.start()) if seller else None
        after = masked[m.end() : m.end() + 60]
        found.append(_read_amount(before, m.group(1), after, seller=seller, name=name))
    return [amount for amount in found if amount is not None]


def _as_digits(match: re.Match[str]) -> str:
    """A spelled-out number as digits, padded to keep every later index in place."""
    total = current = 0
    for word in re.split(r"[\s-]+", match.group()):
        if word == "hundred":
            current = (current or 1) * 100
        elif word == "thousand":
            total, current = total + (current or 1) * 1000, 0
        elif word != "and":
            current += _WORD_VALUES[word]
    return str(total + current).ljust(len(match.group()))


def _name_before(masked: str, original: str | None, start: int) -> bool:
    """The word right before a number makes it a name: a label ("size 42", "#3") or a product
    name ("iPhone 13", "a Dell 27"), unless that word is a money cue ("Deal 80")."""
    found = re.search(r"(\S+)[ \t]*$", masked[:start])
    if found is None:
        return False
    word = found.group(1)
    if _LABEL.match(word):
        return True
    if original is None or word.strip(".,!?:;") in _CUES:
        return False
    shown = original[found.start(1) : found.end(1)]
    if any(ch.isupper() for ch in shown[1:]):  # iPhone, MacBook, PS
        return True
    starts_sentence = not masked[: found.start(1)].strip() or bool(
        re.search(r"[.!?\n]\s*$", masked[: found.start(1)])
    )
    return shown[:1].isupper() and not starts_sentence


def _words_before(text: str, end: int, size: int = 80) -> str:
    """The last few words before a number, never starting mid-word."""
    if end <= size:
        return text[:end]
    return re.sub(r"^\S*", "", text[end - size : end])


def _read_amount(
    before: str, digits: str, after: str, *, seller: bool = False, name: bool | None = None
) -> _Amount | None:
    thousands = bool(_THOUSANDS.match(after))
    if thousands:
        after = after[1:]
    if _UNIT_AFTER.match(after):
        return None
    cents = int(Decimal(digits.replace(",", "")) * (100_000 if thousands else 100))
    if cents > MAX_PRICE_CENTS:
        return None
    prefix = _CURRENCY_BEFORE.search(before)
    suffix = _CURRENCY_AFTER.match(after)
    core = before[: prefix.start()] if prefix else before
    rest = after[suffix.end() :] if suffix else after
    lead = _LEAD.search(core)
    kind = lead.lastgroup if lead else None
    strong_trail = bool(_STRONG_TRAIL.match(rest))
    alone = _alone(core, rest)
    trailed = bool(_WEAK_TRAIL.match(rest) or _ACCEPT_TRAIL.match(rest))
    offer = kind in ("offer", "maybe") or strong_trail or trailed or alone
    if prefix or suffix or (thousands and "." in digits):
        return _Amount(cents, offer)
    if _PLURAL_AFTER.match(rest):
        return None
    if not (kind or offer or _PRICE_TRAIL.match(rest)):
        if not seller or name or re.match(r"[a-z]", rest):  # "80d", "13t" are model names
            return None
    strong = kind in ("offer", "price") or strong_trail
    if _is_hour_or_count(digits, thousands) and not strong_trail:
        return None
    if _is_year(digits, thousands) and not strong:
        return None
    return _Amount(cents, offer)


def _alone(before: str, after: str) -> bool:
    head = _CLAUSE_BREAK.split(before)[-1].strip()
    return bool(_FILLER.fullmatch(head)) and bool(_CLAUSE_END.match(after))


def _is_hour_or_count(digits: str, thousands: bool) -> bool:
    return not thousands and digits.isdigit() and 0 <= int(digits) <= 12


def _is_year(digits: str, thousands: bool) -> bool:
    return not thousands and digits.isdigit() and 1900 <= int(digits) <= 2099


# ---------------------------------------------------------------- scams

_I = re.IGNORECASE
_BENIGN_CODES = re.compile(
    r"\b(?:zip|postal|area|dress|promo|promotional|discount|coupon|bar|qr|source|error|model"
    r"|product|colou?r|lock|door|gate|buzzer|entry|parking|referral|cheat)\s+codes?\b",
    _I,
)
_PREMISES = re.compile(r"\b(?:building|door|buzzer|gate|lobby|elevator|garage|parking|lock)\b", _I)
_CODE_REQUEST = re.compile(
    r"\b(?:verif(?:y|ication)|confirm(?:ation)?|security|authenticat(?:e|ion)|auth|activation"
    r"|one[\s-]?time|otp|2fa|sms|google)\b[^.!?\n]{0,40}\bcodes?\b"
    r"|\bcodes?\b[^.!?\n]{0,40}\b(?:i|we|they|google)\s+(?:just\s+)?(?:sent|send|texted|txted)\b"
    r"|\b(?:sent|send|sending|text|texted|texting|txt)\s+(?:you|u|to\s+your\s+(?:phone|number))"
    r"\s+(?:a|the|an|my)?\s*(?:\w+[\s-]?digit\s+)?(?:verification\s+)?codes?\b"
    r"|\bgoogle\s*voice\b"
    r"|\bverify\s+(?:that\s+)?(?:you(?:'re|\s+are)|ur)\s+(?:a\s+)?(?:real|human|legit|genuine"
    r"|not\s+a\s+(?:bot|scam(?:mer)?))",
    _I,
)
# Weaker shapes that a buyer asking for a building or lock code can also produce.
_CODE_HANDOVER = re.compile(
    r"\b(?:tell|give|read|share|forward|send)\s+(?:(?:me|it|us|over|back)\s+)*"
    r"(?:the|that|this|your|my)?\s*codes?\b"
    r"|\b(?:\d|four|six|eight)[\s-]?digits?(?:\s+(?:code|number|pin))?\s+(?:i|we)\s+(?:sent|texted)"
    r"|\b(?:\d|four|six|eight)[\s-]?digit\s+(?:code|number|pin)\b",
    _I,
)
_PAYMENT = re.compile(
    r"\b(?:pay|paid|paying|payment|cheque|transfer|e-?transfer|etransfer|paypal|funds|money"
    r"|deposit|refund|invoice|zelle|venmo)\b",
    _I,
)
# Payment arranged away from the pickup. "I'll pay cash" is not one.
_REMOTE_PAYMENT = re.compile(
    r"\b(?:paid|payment|cheque|transfer|e-?transfer|etransfer|paypal|funds|deposit|refund"
    r"|invoice|zelle|venmo|wire|send\s+(?:you\s+)?(?:the\s+)?money)\b",
    _I,
)
_YOUR_EMAIL = re.compile(r"\b(?:your|ur)\s+e-?mail\b", _I)
_LINK = re.compile(r"\b(?:link|url|website)\b", _I)
_LINK_PURPOSE = re.compile(
    r"\b(?:pay|payment|paid|deposit|transfer|funds|money|claim|accept|receive|release|confirm"
    r"|verify|ship|shipping|deliver|delivery|courier|click|sign\s*in|log\s*in|insurance)",
    _I,
)
_FORM = re.compile(r"\bfill\s+(?:out|in)\s+(?:the|this|a)?\s*form\b", _I)
_THIRD_PARTY = re.compile(
    r"\b(?:my|our|a|the|his|her)\s+(?:shipping\s+)?(?:agent|mover|movers|moving\s+company|courier"
    r"|shipper|driver|delivery\s+(?:guy|man|person|service|company|agent)"
    r"|pick\s*-?\s*up\s+(?:agent|guy|person|service)|assistant|secretary|representative|rep"
    r"|logistics(?:\s+company)?|transport(?:er|ation)?(?:\s+company)?"
    r"|freight\s+(?:company|agent))\b",
    _I,
)
_FREEZE = (
    (
        re.compile(
            r"\b(?:bank(?:ing)?\s+(?:login|log\s*in|password|username|user\s*name|credentials"
            r"|card|account\s+(?:number|details|info|login|password))"
            r"|online\s+banking\s+(?:login|log\s*in|password|username|details|info)"
            r"|(?:log|sign)\s*(?:in|into)\s+(?:to\s+)?(?:your\s+)?(?:bank|online\s+banking)"
            r"|(?:account|routing|transit|institution)\s+number"
            r"|(?:your|bank|account|email)\s+password|pin\s+(?:number|code)"
            r"|credit\s+card\s+(?:number|details|info)|card\s+number|cvv|cvc|date\s+of\s+birth"
            r"|mother'?s\s+maiden)\b",
            _I,
        ),
        "asks for banking or card details",
    ),
    (
        # The bare acronym is case-sensitive: "sin" is also a word.
        re.compile(
            r"(?i:\b(?:social\s+insurance|social\s+security|ssn|sin\s*(?:number|#|no\b)"
            r"|your\s+sin)\b)|(?<![A-Za-z])S\.?I\.?N\.?(?![A-Za-z])"
        ),
        "asks for a SIN or SSN",
    ),
    (
        re.compile(
            r"\b(?:cashier'?s?\s+(?:cheque|check)|certified\s+(?:cheque|check)|money\s+order"
            r"|bank\s+draft|over\s*pa(?:y|id|yment)|(?:cheque|check|payment)\s+for\s+(?:extra|more)"
            r"|(?:send|give|wire)\s+(?:me\s+)?(?:back\s+)?the\s+(?:difference|extra|rest|balance)"
            r"|refund\s+(?:me\s+)?the\s+(?:difference|extra)"
            r"|(?:mail|post|courier|ship)(?:ing)?\s+(?:you\s+)?(?:a|the|my)?\s*(?:cheque|check))\b",
            _I,
        ),
        "proposes a cheque by mail, a money order, or an overpayment",
    ),
    (
        re.compile(
            r"\b(?:gift\s*cards?|bitcoin|btc|crypto(?:currency)?|usdt|western\s+union|moneygram"
            r"|wire\s+transfer)\b",
            _I,
        ),
        "proposes gift cards, crypto, or a wire transfer",
    ),
)
_SHIP_REASON = "asks to ship; pickup is local and in person only"
_WARN = (
    (
        re.compile(
            r"\b(?:ship(?:s|ped|ping|ment)?|mail\s+it|post\s+it\s+to|send\s+it\s+(?:to|via|by)"
            r"|courier|canada\s+post|purolator|fedex|delivery\s+service)\b",
            _I,
        ),
        _SHIP_REASON,
    ),
    (re.compile(r"\b(?:UPS|DHL)\b"), _SHIP_REASON),  # case-sensitive: "pick ups"
    (
        re.compile(
            r"\b(?:e-?\s?transfers?|etransfers?|emt|interac|paypal|zelle|venmo|cash\s*app|deposit"
            r"|pay\s+(?:you\s+)?(?:now|first|upfront|up\s+front|in\s+advance|ahead|before))\b",
            _I,
        ),
        "wants to pay electronically or in advance; payment is cash at pickup",
    ),
    (re.compile(r"\bcheques?\b", _I), "wants to pay by cheque; payment is cash at pickup"),
    (
        re.compile(
            r"\b(?:whats\s?app|telegram|wechat|viber|kik|hangouts|imessage|signal\s+app|on\s+signal"
            r"|text\s+me|txt\s+me|call\s+me|email\s+me|e-mail\s+me"
            r"|(?:my|your|ur)\s+(?:cell|phone|number|mobile|email|e-mail)"
            r"|reach\s+me\s+(?:at|on)|contact\s+me\s+(?:at|on|via|by))\b",
            _I,
        ),
        "wants to move the conversation off the platform",
    ),
    (_THIRD_PARTY, "a third party would collect the item"),
)


def detect_scam(text: str) -> ScamSignal | None:
    """Freeze-level patterns first, then warn-level. None when nothing matches."""
    reason = _freeze_reason(text)
    if reason:
        return ScamSignal("freeze", reason)
    reason = _warn_reason(text)
    return ScamSignal("warn", reason) if reason else None


def _freeze_reason(text: str) -> str | None:
    if _asks_for_code(text):
        return "asks for a verification code (Google Voice scam)"
    for pattern, why in _FREEZE:
        if pattern.search(text):
            return why
    if _YOUR_EMAIL.search(text) and _PAYMENT.search(text):
        return "asks for an email address to send payment"
    if _THIRD_PARTY.search(text) and _REMOTE_PAYMENT.search(text):
        return "a third party will collect the item while payment is arranged remotely"
    if _URL.search(text) and (_PAYMENT.search(text) or _LINK_PURPOSE.search(text)):
        return "sends a payment or shipping link"
    if _FORM.search(text) or _in_one_sentence(text, _LINK, _LINK_PURPOSE):
        return "asks to pay or confirm through a link"
    return None


def _warn_reason(text: str) -> str | None:
    for pattern, why in _WARN:
        if pattern.search(text):
            return why
    if contains_contact_info(text):
        return "shares contact details or a link to move off the platform"
    return None


def _asks_for_code(text: str) -> bool:
    cleaned = _BENIGN_CODES.sub(" ", text)
    if _CODE_REQUEST.search(cleaned):
        return True
    return bool(_CODE_HANDOVER.search(cleaned)) and not _PREMISES.search(cleaned)


def _in_one_sentence(text: str, first: re.Pattern[str], second: re.Pattern[str]) -> bool:
    return any(first.search(s) and second.search(s) for s in re.split(r"[.!?\n]+", text))


# ---------------------------------------------------------------- local cash only (seller text)

# DESIGN guardrail 8 on what the seller sends. A buyer asking is only a warning (detect_scam);
# the seller offering it is the harm. A mention passes only when its own clause declines it:
# a negation before it with no "but", "if", "can", "I"... in between, or a negated verb right
# after it.
#
# | Seller text | Verdict |
# |---|---|
# | "I can ship it via Canada Post", "E-transfer the deposit first" | Offer: rejected |
# | "Sorry, no shipping, e-transfers, or deposits. Cash at pickup only" | Decline: passes |
# | "Shipping isn't possible", "I can't do PayPal" | Decline: passes |
# | "I can't hold it without a deposit", "No problem, I can ship" | Offer: rejected |
_REMOTE_DEAL = re.compile(
    r"\b(?:ship(?:s|ped|ping|ment)?|mail(?:ing|ed)?\s+(?:it|them|this)|post\s+(?:it|them)\s+to"
    r"|send\s+(?:it|them)\s+(?:to|via|by|through)|couriers?|canada\s+post|purolator|fedex"
    r"|delivery\s+service|e-?\s?transfers?|etransfers?|emt|interac|paypal|zelle|venmo|cash\s*app"
    r"|cheques?|money\s+orders?|deposits?"
    r"|pay(?:ing|ment)?\s+(?:me\s+|you\s+|it\s+)?(?:now|first|upfront|up\s+front|in\s+advance"
    r"|ahead|before|online)"
    r"|send\s+(?:me\s+|us\s+)?(?:the\s+)?(?:money|payment|funds))\b",
    re.IGNORECASE,
)
_REMOTE_DEAL_CASED = re.compile(r"\b(?:UPS|DHL)\b")  # case-sensitive: "pick ups"
_SELLER_CLAUSE = re.compile(r"[.!?;:\n()\u2014\u2013]|\s-\s")
# "No problem, I can ship": these negations agree rather than decline.
_AGREEING_NEGATION = re.compile(
    r"\b(?:no|not|never|\w+n't)\s+(?:a\s+)?(?:mind|object(?:ion)?|opposed|problems?|issues?"
    r"|worries|worry|trouble|rush|pressure|hassle|hesitat\w*|doubt|sure|bad|questions)\b"
    r"|\bnot\s+(?:only|just)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|nor|cannot|unable|cant|wont|dont|doesnt|isnt|arent|didnt|couldnt"
    r"|wouldnt|instead\s+of|rather\s+than)\b|\b\w+n't\b",
    re.IGNORECASE,
)
# Between a negation and the term, these end the decline: "no, but I can ship".
_UNDOES_NEGATION = re.compile(
    r"\b(?:but|however|though|although|unless|until|except|without|if|first|then|so|and\s+then"
    r"|can|will|could|would|happy|glad|ok(?:ay)?|sure|yes|fine|i|i'm|i'll|we|we'll|you)\b",
    re.IGNORECASE,
)
_NEGATED_AFTER = re.compile(
    r"\s*(?:(?:is|are|was|were|will|would|does|do|can|could|really|just|also)\s+)*"
    r"(?:not|never|cannot|cant|wont|dont|doesnt|isnt|arent|\w+n't)\b",
    re.IGNORECASE,
)


def offers_remote_deal(text: str) -> str | None:
    """The first shipping or non-cash payment term in seller text that its clause does not
    decline, or None. Read conservatively: rewording a false alarm is cheap."""
    for clause in _SELLER_CLAUSE.split(text.replace("\u2019", "'")):
        clause = _AGREEING_NEGATION.sub(" ", clause)
        for match in (*_REMOTE_DEAL.finditer(clause), *_REMOTE_DEAL_CASED.finditer(clause)):
            if not _declined(clause, match):
                return match.group()
    return None


def _declined(clause: str, match: re.Match[str]) -> bool:
    if _NEGATED_AFTER.match(clause, match.end()):
        return True
    before = clause[: match.start()]
    negations = list(_NEGATION.finditer(before))
    return bool(negations) and not _UNDOES_NEGATION.search(before[negations[-1].end() :])


# ---------------------------------------------------------------- address

_STREET_TYPES = (
    "st|street|ave|av|avenue|rd|road|blvd|boulevard|dr|drive|cres|crescent|ct|crt|court|pl|place"
    "|ln|lane|way|terr|terrace|tce|pkwy|parkway|cir|circle|hwy|highway|trail|trl|sq|square|gate"
    "|gdns|gardens|grove|grv|row|path|line|close|mews|hts|heights|park|walk|cove|ridge|hill"
)
_NAME_END = re.compile(rf"^(?:{_STREET_TYPES}|n|s|e|w|ne|nw|se|sw|north|south|east|west)\.?$")
_HOUSE = re.compile(r"(?:^|[\s,#])(?:\d+[a-z]?-)?(?P<number>\d+[a-z]?)\s+(?P<rest>[a-z][^,\d]*)")
_POSTAL = re.compile(r"\b[a-z]\d[a-z][\s-]?\d[a-z]\d\b")
_NO_LETTER_NEXT = r"(?![^\W\d_])"
_STREET_TAIL = rf"(?:[\W_]*(?:{_STREET_TYPES}){_NO_LETTER_NEXT}\.?|{_NO_LETTER_NEXT})"


def leaks_address(text: str, settings: Settings) -> bool:
    return any(pattern.search(text) for pattern in _address_patterns(settings.pickup_address))


def redact_address(text: str, settings: Settings) -> str:
    """Repeats until stable. With an address ending in a digit, such as "12 Highway 7", the
    second copy in "12 Highway 712 Highway 7" only matches once the first is replaced."""
    patterns = _address_patterns(settings.pickup_address)
    previous = None
    while text != previous:
        previous = text
        for pattern in patterns:
            text = pattern.sub(ADDRESS_PLACEHOLDER, text)
    return text


def _address_patterns(address: str) -> list[re.Pattern[str]]:
    """Full address, house number + street name, and postal code, each ignoring case, spacing,
    and punctuation. After the house number and name, any street type is swallowed too, so
    "Street" and "St" redact alike."""
    lowered = address.casefold()
    keys = [(_key(lowered), _NO_LETTER_NEXT), (_house_key(lowered), _STREET_TAIL)]
    postal = _POSTAL.search(lowered)
    if postal:
        keys.append((_key(postal.group()), _NO_LETTER_NEXT))
    return [_flexible(key, tail) for key, tail in keys if key]


def _house_key(address: str) -> str:
    match = _HOUSE.search(address)
    if match is None:
        return ""
    first, *more = match["rest"].split()
    name = [first]
    for word in more:
        if _NAME_END.match(word):
            break
        name.append(word)
    return _key(match["number"] + "".join(name))


def _key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text)


def _flexible(key: str, tail: str) -> re.Pattern[str]:
    """Matches key with any run of spaces or punctuation between its characters."""
    start = r"(?<!\d)" if key[0].isdigit() else r"(?<![^\W_])"
    body = r"[\W_]*".join(re.escape(ch) for ch in key)
    return re.compile(start + body + tail, re.IGNORECASE)


# ---------------------------------------------------------------- counts


def outbound_count_today(session: Session, clock: Clock, platform: str) -> int:
    query = (
        select(func.count(col(Message.id)))
        .join(Thread, col(Thread.id) == Message.thread_id)
        .where(
            Thread.platform == platform,
            Message.direction == Direction.OUTBOUND,
            Message.via != "human",
            Message.sent_at >= _utc_midnight(clock),
        )
    )
    return int(session.exec(query).one())


def publish_count_today(session: Session, clock: Clock, platform: str) -> int:
    since = _utc_midnight(clock)
    query = select(func.count(col(Listing.id))).where(
        Listing.platform == platform,
        or_(col(Listing.published_at) >= since, col(Listing.last_relisted_at) >= since),
    )
    return int(session.exec(query).one())


def outbound_streak(session: Session, thread_id: int) -> tuple[int, datetime | None]:
    """Outbound messages since the buyer last wrote, and when the latest one was sent.

    Turns follow the order messages reached the database (Message.id), not their stamps: an
    inbound stamp is the platform's clock, often minute-precision, while an outbound stamp is
    ours, so a reply 40s after our message can carry an earlier time than it."""
    query = (
        select(Message.direction, Message.sent_at)
        .where(Message.thread_id == thread_id)
        .order_by(col(Message.id).desc())
    )
    count, last = 0, None
    for direction, sent_at in session.exec(query):
        if direction == Direction.INBOUND:
            break
        count += 1
        last = last or sent_at
    return count, last


def _utc_midnight(clock: Clock) -> datetime:
    return clock.now().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------- time


def to_local(settings: Settings, dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(ZoneInfo(settings.timezone))


def parse_local_datetime(settings: Settings, text: str) -> datetime:
    """ISO 8601 with a time of day. Naive means Settings.timezone; a local time skipped by a DST
    change is rejected. Returns aware UTC."""
    raw = text.strip()
    junk = ValueError(f"{text!r} is not an ISO 8601 date and time, e.g. 2026-10-10T18:00")
    if not re.search(r"\d[T ]\d", raw):
        raise junk
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        raise junk from None
    if parsed.tzinfo is not None:
        return parsed.astimezone(UTC)
    zone = ZoneInfo(settings.timezone)
    local = parsed.replace(tzinfo=zone)
    if local.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != parsed:
        raise ValueError(f"{text!r} does not exist in {settings.timezone} (clocks skip it for DST)")
    return local.astimezone(UTC)


def within_availability(settings: Settings, start: datetime, end: datetime) -> bool:
    """True when [start, end] fits inside a single window on start's local day."""
    if not start < end:
        return False
    zone = ZoneInfo(settings.timezone)
    day = to_local(settings, start).date()
    return any(
        window.weekday == day.weekday()
        and _at(day, window.start, zone) <= start
        and end <= _at(day, window.end, zone)
        for window in settings.availability
    )


def availability_text(settings: Settings) -> str:
    by_day: list[tuple[tuple[time, time], ...]] = [
        tuple(sorted((w.start, w.end) for w in settings.availability if w.weekday == day))
        for day in range(7)
    ]
    groups: list[tuple[int, int]] = []
    for day in range(7):
        if groups and groups[-1][1] == day - 1 and by_day[day] == by_day[day - 1]:
            groups[-1] = (groups[-1][0], day)
        else:
            groups.append((day, day))
    parts = [_window_group(first, last, by_day[first]) for first, last in groups if by_day[first]]
    return ", ".join(parts) or "no pickup times set"


def _window_group(first: int, last: int, spans: tuple[tuple[time, time], ...]) -> str:
    days = _DAY_NAMES[first] if first == last else f"{_DAY_NAMES[first]}-{_DAY_NAMES[last]}"
    hours = " and ".join(f"{start:%H:%M}-{end:%H:%M}" for start, end in spans)
    return f"{days} {hours}"


def _at(day: date, at: time, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, at, tzinfo=zone).astimezone(UTC)


# ---------------------------------------------------------------- text rules


def check_outbound_text(
    session: Session,
    settings: Settings,
    clock: Clock,
    *,
    item: Item,
    goal: Goal | None,
    thread: Thread,
    body: str,
) -> list[str]:
    """Violations in a message to a buyer, each telling the model what to do instead.
    Address-rule reasons, and only those, contain ADDRESS_PLACEHOLDER, so a human reply can
    drop them: [r for r in reasons if ADDRESS_PLACEHOLDER not in r]."""
    reasons = []
    if leaks_address(body, settings):
        reasons.append(
            f"The message contains the exact pickup address. Never type it: write "
            f"{ADDRESS_PLACEHOLDER} and it is filled in once allowed. Until then share only the "
            f"pickup area ({settings.pickup_area})."
        )
    if ADDRESS_PLACEHOLDER in body:
        blocked = _placeholder_blocked(session, settings, clock, item, thread)
        if blocked:
            reasons.append(
                f"{ADDRESS_PLACEHOLDER} cannot be sent yet: {blocked}. Share the pickup area "
                f"({settings.pickup_area}) instead for now."
            )
    reasons += _shared_text_rules(settings, goal, body, "message")
    stripped = body.strip()
    if not stripped:
        reasons.append("The message is empty. Write what you want to say to the buyer.")
    elif len(stripped) > MAX_MESSAGE_CHARS:
        reasons.append(
            f"The message is {len(stripped)} characters; the limit is {MAX_MESSAGE_CHARS}. "
            "Say it in fewer words."
        )
    return reasons


def check_listing_text(settings: Settings, *, goal: Goal | None, text: str) -> list[str]:
    reasons = []
    if ADDRESS_PLACEHOLDER in text or leaks_address(text, settings):
        reasons.append(
            "Listing text must never contain the pickup address or "
            f"{ADDRESS_PLACEHOLDER}. Mention only the pickup area ({settings.pickup_area})."
        )
    reasons += _shared_text_rules(settings, goal, text, "listing text")
    if len(text) > MAX_LISTING_CHARS:
        reasons.append(
            f"The listing text is {len(text)} characters; the limit is {MAX_LISTING_CHARS}. "
            "Shorten it."
        )
    return reasons


def _shared_text_rules(settings: Settings, goal: Goal | None, text: str, what: str) -> list[str]:
    reasons = []
    if contains_contact_info(text):
        reasons.append(
            f"The {what} contains a phone number, email address, or link. Keep the conversation "
            "on the platform and remove it."
        )
    money = _below_minimum(goal, text, what, known=(settings.pickup_area,))
    if money:
        reasons.append(money)
    signal = detect_scam(text)
    if signal and signal.severity == "freeze":
        reasons.append(
            f"The {what} matches a scam pattern ({signal.reason}). Do not mention codes, links, "
            "cheques, or remote payment; to decline, just say it is cash at pickup only."
        )
    offered = offers_remote_deal(text)
    if offered:
        reasons.append(
            f"The {what} offers shipping or payment other than cash at pickup ({offered!r}). "
            "Sales are local and cash at pickup only: no shipping, e-transfers, PayPal, cheques, "
            "deposits, or paying ahead. To decline, say so plainly, e.g. \"Sorry, no shipping, "
            "cash at pickup only.\""
        )
    return reasons


def _below_minimum(
    goal: Goal | None, text: str, what: str, *, known: Sequence[str] = ()
) -> str | None:
    """Seller text is read with seller_money_cents: any bare number may be a price."""
    if goal is None:
        return None
    firm = goal.firmness == Firmness.FIRM
    minimum = goal.list_price_cents if firm else goal.floor_price_cents
    low = sorted({cents for cents in seller_money_cents(text, known=known) if cents < minimum})
    if not low:
        return None
    amounts = ", ".join(_money(cents) for cents in low)
    not_price = (
        " If one of those numbers is not a price, give it its unit (27 inch, 6pm, 3 years old) "
        "or leave it out."
    )
    if firm:
        return (
            f"The {what} mentions {amounts}, below the firm price of {_money(minimum)}. The price "
            f"is firm: decline politely without naming any lower number.{not_price}"
        )
    return (
        f"The {what} mentions {amounts}, below the floor of {_money(minimum)}. Do not restate "
        f"numbers under the floor: decline, or counter at {_money(minimum)} or more. State full "
        f"prices, not discounts.{not_price}"
    )


def _placeholder_blocked(
    session: Session, settings: Settings, clock: Clock, item: Item, thread: Thread
) -> str | None:
    active = active_commitment_for_thread(session, thread.id) if thread.id is not None else None
    if active is None or active.accepted.item_id != item.id:
        return "no deal is agreed in this thread; accept an offer first"
    if active.pickup is None:
        return "no pickup is booked; book one with book_pickup first"
    deadline = active.accepted.undo_deadline
    if deadline is None or clock.now() <= deadline:
        until = f" until {to_local(settings, deadline):%H:%M}" if deadline else ""
        return f"the owner can still undo the deal{until}; send the address after that"
    return None


def _money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"
