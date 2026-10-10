"""Independent re-check of every hard invariant, used to test the guard code.

Shares no code with guard/ or tools/ on purpose: a bug there must not hide a
violation here. Each violation reads "<UTC time> <invariant>: <what happened>".

| Scope | Rule |
|---|---|
| Messages | Replayed in storage order, which is the order the system learned of them |
| Human | Outbound messages via "human" are exempt; they still count toward a streak |
| Goal | Floor and firmness have no history, so their current values apply throughout |
| Freeze | A FROZEN thread or flagged buyer is frozen from its first inbound intent "scam" message |
| Buyer | A freeze covers every thread of that buyer |
| sent_log | Deliveries carry no sender, so an address the human sent counts too |
| Cash only | A sentence offering shipping or non-cash payment with no negation in it |
| Bundle | ACCEPTED rows sharing a bundle_key are one deal of 2 to 5 items, one buyer, one instant |
| Bundle total | Each bundle row's reason records the agreed total: "... for <N> cents in total." |
| Bundle events | All rows of a bundle get the same events at the same instants |
| Bundle slot | Rows of one bundle share a pickup slot; they do not overlap each other |
| Bundle prices | Each row meets its own item's floor (list price when firm), as any ACCEPTED |
"""
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from marketplace_agent.config import Settings
from marketplace_agent.core.models import (
    Buyer,
    Commitment,
    CommitmentKind,
    Direction,
    Firmness,
    Goal,
    Message,
    PriceChange,
    Thread,
    ThreadState,
)

SentRecord = tuple[str, str, datetime]  # (thread_remote_id, delivered_body, sent_at)

PLACEHOLDER = "[PICKUP_ADDRESS]"
FOLLOW_UP_AFTER = timedelta(hours=24)
HUMAN = "human"
SCAM_INTENT = "scam"
BUNDLE_SIZE = range(2, 6)
_BUNDLE_TOTAL = re.compile(r"\bfor (\d+) cents in total\b")


def audit(
    session: Session, settings: Settings, *, sent_log: Iterable[SentRecord] | None = None
) -> list[str]:
    """Every invariant violation in the database, oldest first. Empty means clean.

    With sent_log, address timing is judged on what the platform actually delivered;
    without it, on stored outbound bodies that carry the placeholder or the address.
    """
    facts = _load(session)
    ledger = _Ledger(facts, settings)
    commitments = list(session.exec(select(Commitment)).all())
    ledger.replay(commitments)
    return sorted(
        [
            *ledger.violations,
            *_bundle_violations(commitments),
            *_streak_violations(facts),
            *_cap_violations(facts, settings),
            *_freeze_violations(facts),
            *_text_violations(facts),
            *_address_violations(facts, ledger, settings, sent_log),
        ]
    )


# Formatting


def _utc(dt: datetime) -> datetime:
    """SQLite drops tzinfo on the way back; every stored datetime is UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _stamp(dt: datetime) -> str:
    return f"{dt:%Y-%m-%dT%H:%M:%SZ}"


def _violation(when: datetime, invariant: str, detail: str) -> str:
    return f"{_stamp(when)} {invariant}: {detail}"


def _money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


# Database facts


def _chronological(row: Commitment | PriceChange) -> tuple[datetime, int]:
    return _utc(row.created_at), row.id or 0


@dataclass
class _Facts:
    threads: dict[int, Thread]
    goals: dict[int, Goal]
    price_changes: dict[int, list[PriceChange]]
    messages: list[Message]
    frozen_buyers: set[int]  # flagged as scammers, or owning a FROZEN thread

    def minimum_cents(self, item_id: int | None, when: datetime) -> tuple[int, str] | None:
        """Lowest price the seller may agree to or state at `when`, with its rule."""
        goal = self.goals.get(item_id) if item_id is not None else None
        if goal is None:
            return None
        floor = goal.floor_price_cents
        if goal.firmness == Firmness.FIRM:
            listed = self._list_price_at(goal, when)
            if listed > floor:
                return listed, f"firm list price {_money(listed)}"
        return floor, f"floor {_money(floor)}"

    def _list_price_at(self, goal: Goal, when: datetime) -> int:
        changes = self.price_changes.get(goal.item_id, [])
        before = [c for c in changes if _utc(c.created_at) <= when]
        if before:
            return before[-1].new_cents
        return changes[0].old_cents if changes else goal.list_price_cents

    def system_outbound(self) -> Iterator[tuple[Message, Thread]]:
        for message in self.messages:
            if message.direction == Direction.OUTBOUND and message.via != HUMAN:
                yield message, self.threads[message.thread_id]


def _load(session: Session) -> _Facts:
    changes: dict[int, list[PriceChange]] = defaultdict(list)
    for change in sorted(session.exec(select(PriceChange)).all(), key=_chronological):
        changes[change.item_id].append(change)
    threads = {t.id: t for t in session.exec(select(Thread)).all() if t.id is not None}
    flagged = {b.id for b in session.exec(select(Buyer)).all() if b.flagged_scam and b.id}
    return _Facts(
        threads=threads,
        goals={g.item_id: g for g in session.exec(select(Goal)).all()},
        price_changes=changes,
        messages=list(session.exec(select(Message).order_by(Message.id)).all()),
        frozen_buyers=flagged
        | {t.buyer_id for t in threads.values() if t.state == ThreadState.FROZEN},
    )


# Commitment ledger


@dataclass
class _Deal:
    accepted: Commitment
    slot: tuple[datetime, datetime] | None = None
    pickup_id: int | None = None
    booked_at: datetime | None = None
    closed_at: datetime | None = None

    def address_allowed(self, when: datetime) -> bool:
        deadline = self.accepted.undo_deadline
        return (
            self.booked_at is not None
            and self.booked_at <= when
            and (self.closed_at is None or when < self.closed_at)
            and (deadline is None or when > _utc(deadline))
        )


@dataclass
class _Ledger:
    facts: _Facts
    settings: Settings
    deals: dict[int | None, _Deal] = field(default_factory=dict)  # by ACCEPTED id
    active: dict[int | None, _Deal] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def replay(self, events: Iterable[Commitment]) -> None:
        for event in sorted(events, key=_chronological):
            when = _utc(event.created_at)
            deal = self.active.get(event.ref_id)
            if event.kind == CommitmentKind.ACCEPTED:
                self._accept(event, when)
            elif deal is None:
                self._flag(
                    when,
                    "ledger_ref_inactive",
                    f"{str(event.kind).upper()} #{event.id} (item {event.item_id}) refers to "
                    f"ACCEPTED #{event.ref_id}, which is not an active commitment",
                )
            elif event.kind == CommitmentKind.PICKUP_BOOKED:
                self._book(deal, event, when)
            else:
                self._close(deal, event, when)

    def _flag(self, when: datetime, invariant: str, detail: str) -> None:
        self.violations.append(_violation(when, invariant, detail))

    def _local(self, dt: datetime) -> str:
        return f"{dt.astimezone(ZoneInfo(self.settings.timezone)):%a %Y-%m-%d %H:%M} local"

    def _accept(self, event: Commitment, when: datetime) -> None:
        for other in self.active.values():
            if other.accepted.item_id == event.item_id:
                self._flag(
                    when,
                    "double_commitment",
                    f"item {event.item_id}: ACCEPTED #{event.id} (thread {event.thread_id}, "
                    f"buyer {event.buyer_id}) while ACCEPTED #{other.accepted.id} "
                    f"(thread {other.accepted.thread_id}, buyer {other.accepted.buyer_id}) "
                    "was still active",
                )
        price = event.price_cents or 0
        minimum = self.facts.minimum_cents(event.item_id, when)
        if minimum is not None and price < minimum[0]:
            self._flag(
                when,
                "accept_below_minimum",
                f"item {event.item_id}: ACCEPTED #{event.id} at {_money(price)} "
                f"is below the {minimum[1]}",
            )
        self.deals[event.id] = self.active[event.id] = _Deal(event)

    def _book(self, deal: _Deal, event: Commitment, when: datetime) -> None:
        slot = _slot(event)
        if slot is None:
            self._flag(
                when,
                "pickup_invalid",
                f"PICKUP_BOOKED #{event.id} for ACCEPTED #{deal.accepted.id} "
                "lacks a start before its end",
            )
            return
        start, end = slot
        if start <= when:
            self._flag(
                when,
                "pickup_in_past",
                f"PICKUP_BOOKED #{event.id} (item {event.item_id}) starts at "
                f"{self._local(start)}, not after the booking time",
            )
        if not _inside_availability(self.settings, start, end):
            self._flag(
                when,
                "pickup_outside_availability",
                f"PICKUP_BOOKED #{event.id} (item {event.item_id}) "
                f"{self._local(start)} to {self._local(end)} is not inside one availability window",
            )
        for other in self.active.values():
            if other is deal or other.slot is None or _same_bundle(other, deal):
                continue
            if start < other.slot[1] and other.slot[0] < end:
                self._flag(
                    when,
                    "pickup_overlap",
                    f"PICKUP_BOOKED #{event.id} (item {event.item_id}) at {self._local(start)} "
                    f"overlaps PICKUP_BOOKED #{other.pickup_id} (item {other.accepted.item_id}) "
                    f"at {self._local(other.slot[0])}",
                )
        deal.slot, deal.pickup_id = slot, event.id
        deal.booked_at = deal.booked_at or when

    def _close(self, deal: _Deal, event: Commitment, when: datetime) -> None:
        deadline = deal.accepted.undo_deadline
        if event.kind == CommitmentKind.UNDONE and deadline and when > _utc(deadline):
            self._flag(
                when,
                "late_undo",
                f"UNDONE #{event.id} for ACCEPTED #{deal.accepted.id} came after its undo "
                f"deadline {_stamp(_utc(deadline))}",
            )
        deal.closed_at = when
        del self.active[deal.accepted.id]


def _same_bundle(one: _Deal, other: _Deal) -> bool:
    key = one.accepted.bundle_key
    return key is not None and key == other.accepted.bundle_key


# Bundles

Step = tuple[str, datetime, datetime | None, datetime | None]  # kind, when, pickup start, end


def _bundle_violations(commitments: Sequence[Commitment]) -> Iterator[str]:
    bundles: dict[str, list[Commitment]] = defaultdict(list)
    steps: dict[int | None, list[Step]] = defaultdict(list)
    for row in sorted(commitments, key=_chronological):
        if row.kind == CommitmentKind.ACCEPTED:
            if row.bundle_key is not None:
                bundles[row.bundle_key].append(row)
            continue
        start, end = row.pickup_start, row.pickup_end
        steps[row.ref_id].append(
            (
                str(row.kind).upper(),
                _utc(row.created_at),
                _utc(start) if start else None,
                _utc(end) if end else None,
            )
        )
    for key, rows in bundles.items():
        yield from _bundle_shape(key, rows)
        yield from _bundle_total(key, rows)
        yield from _bundle_split(key, rows, steps)


def _bundle_shape(key: str, rows: list[Commitment]) -> Iterator[str]:
    first = rows[0]
    when = _utc(first.created_at)
    items = [row.item_id for row in rows]
    problems = []
    if len(rows) not in BUNDLE_SIZE:
        problems.append(f"has {len(rows)} item{'' if len(rows) == 1 else 's'}, not 2 to 5")
    repeated = sorted({item for item in items if items.count(item) > 1})
    if repeated:
        problems.append(f"promises item {', '.join(map(str, repeated))} more than once")
    if len({row.buyer_id for row in rows}) > 1:
        problems.append("spans more than one buyer")
    if len({_utc(row.created_at) for row in rows}) > 1:
        problems.append("was accepted at more than one instant")
    if len({row.undo_deadline and _utc(row.undo_deadline) for row in rows}) > 1:
        problems.append("has more than one undo deadline")
    for problem in problems:
        yield _violation(when, "bundle_shape", f"bundle {key} (ACCEPTED #{first.id}) {problem}")


def _bundle_total(key: str, rows: list[Commitment]) -> Iterator[str]:
    when = _utc(rows[0].created_at)
    totals = {_recorded_total(row.reason) for row in rows}
    if None in totals or len(totals) != 1:
        yield _violation(
            when, "bundle_total", f"bundle {key} does not record one agreed total on every row"
        )
        return
    [total] = totals
    assert total is not None
    prices = sum(row.price_cents or 0 for row in rows)
    if prices != total:
        yield _violation(
            when,
            "bundle_total",
            f"bundle {key}: its item prices add up to {_money(prices)}, not the agreed "
            f"{_money(total)}",
        )


def _recorded_total(reason: str) -> int | None:
    found = _BUNDLE_TOTAL.search(reason)
    return int(found[1]) if found else None


def _bundle_split(
    key: str, rows: list[Commitment], steps: dict[int | None, list[Step]]
) -> Iterator[str]:
    """Each row's events against the first row's: the first difference is a split bundle."""
    first = rows[0]
    expected = steps.get(first.id, [])
    for row in rows[1:]:
        got = steps.get(row.id, [])
        for index in range(max(len(expected), len(got))):
            want = expected[index] if index < len(expected) else None
            have = got[index] if index < len(got) else None
            if want == have:
                continue
            when = min(step[1] for step in (want, have) if step is not None)
            yield _violation(
                when,
                "bundle_split",
                f"bundle {key}: ACCEPTED #{first.id} (item {first.item_id}) "
                f"{_describe(want)} while ACCEPTED #{row.id} (item {row.item_id}) "
                f"{_describe(have)}",
            )
            break


def _describe(step: Step | None) -> str:
    if step is None:
        return "had no such event"
    kind, when, start, end = step
    slot = f" for {_stamp(start)} to {_stamp(end)}" if start and end else ""
    return f"had {kind} at {_stamp(when)}{slot}"


def _slot(event: Commitment) -> tuple[datetime, datetime] | None:
    if event.pickup_start is None or event.pickup_end is None:
        return None
    start, end = _utc(event.pickup_start), _utc(event.pickup_end)
    return (start, end) if start < end else None


def _inside_availability(settings: Settings, start: datetime, end: datetime) -> bool:
    zone = ZoneInfo(settings.timezone)
    local_start, local_end = start.astimezone(zone), end.astimezone(zone)
    if local_start.date() != local_end.date():
        return False
    return any(
        window.weekday == local_start.weekday()
        and window.start <= local_start.time()
        and local_end.time() <= window.end
        for window in settings.availability
    )


# Message rules


def _streak_violations(facts: _Facts) -> Iterator[str]:
    trailing: dict[int, tuple[int, datetime, int | None]] = {}  # count, last sent_at, last id
    for message in facts.messages:
        if message.direction == Direction.INBOUND:
            trailing.pop(message.thread_id, None)
            continue
        when = _utc(message.sent_at)
        count, last_at, last_id = trailing.get(message.thread_id, (0, when, None))
        if message.via != HUMAN and count >= 2:
            yield _violation(
                when,
                "outbound_streak",
                f"message #{message.id} is outbound number {count + 1} in a row "
                f"to thread {message.thread_id}; at most one follow-up is allowed",
            )
        elif message.via != HUMAN and count == 1 and when - last_at < FOLLOW_UP_AFTER:
            hours = (when - last_at).total_seconds() / 3600
            yield _violation(
                when,
                "outbound_streak",
                f"follow-up #{message.id} to thread {message.thread_id} came {hours:.1f}h "
                f"after message #{last_id}; a follow-up needs 24h without a buyer reply",
            )
        trailing[message.thread_id] = (count + 1, when, message.id)


def _cap_violations(facts: _Facts, settings: Settings) -> Iterator[str]:
    sent: Counter[tuple[str, date]] = Counter()
    for message, thread in facts.system_outbound():
        when = _utc(message.sent_at)
        key = (thread.platform, when.date())
        sent[key] += 1
        cap = settings.daily_outbound_cap.get(thread.platform)
        if cap is not None and sent[key] == cap + 1:
            yield _violation(
                when,
                "outbound_cap",
                f"message #{message.id} is outbound number {cap + 1} on {thread.platform} "
                f"on {when.date()} (UTC); the daily cap is {cap}",
            )


def _freeze_violations(facts: _Facts) -> Iterator[str]:
    frozen_by: dict[int, Message] = {}  # buyer id -> first scam message
    for message in facts.messages:
        thread = facts.threads[message.thread_id]
        if (
            message.direction == Direction.INBOUND
            and message.intent == SCAM_INTENT
            and thread.buyer_id in facts.frozen_buyers
        ):
            frozen_by.setdefault(thread.buyer_id, message)
        elif message.direction == Direction.OUTBOUND and message.via != HUMAN:
            scam = frozen_by.get(thread.buyer_id)
            if scam is not None:
                yield _violation(
                    _utc(message.sent_at),
                    "message_after_freeze",
                    f"message #{message.id} to thread {thread.id} (buyer {thread.buyer_id}) "
                    f"after scam message #{scam.id} in thread {scam.thread_id} froze that buyer",
                )


def _text_violations(facts: _Facts) -> Iterator[str]:
    for message, thread in facts.system_outbound():
        when = _utc(message.sent_at)
        minimum = facts.minimum_cents(thread.item_id, when)
        if minimum is not None:
            low = sorted({c for c in _money_amounts_cents(message.body) if c < minimum[0]})
            if low:
                yield _violation(
                    when,
                    "text_below_minimum",
                    f"message #{message.id} to thread {thread.id} (item {thread.item_id}) "
                    f"states {', '.join(map(_money, low))}, below the {minimum[1]}",
                )
        if _has_contact_info(message.body):
            yield _violation(
                when,
                "contact_info",
                f"message #{message.id} to thread {thread.id} contains a phone number, "
                "email, or URL",
            )
        offered = _remote_deal(message.body)
        if offered:
            yield _violation(
                when,
                "not_cash_at_pickup",
                f"message #{message.id} to thread {thread.id} offers shipping or payment other "
                f"than cash at pickup ({offered!r}) without declining it",
            )


def _address_violations(
    facts: _Facts, ledger: _Ledger, settings: Settings, sent_log: Iterable[SentRecord] | None
) -> Iterator[str]:
    address = settings.pickup_address
    if not address.strip():
        return
    for message in facts.messages:
        if _leaks_address(message.body, address):
            yield _violation(
                _utc(message.sent_at),
                "address_stored",
                f"message #{message.id} in thread {message.thread_id} is stored with the "
                "real pickup address instead of the placeholder",
            )
    if sent_log is None:
        deliveries = _stored_deliveries(facts, address)
    else:
        deliveries = _logged_deliveries(facts, address, sent_log)
    for label, thread_ids, when in deliveries:
        if not any(
            deal.address_allowed(when)
            for deal in ledger.deals.values()
            if deal.accepted.thread_id in thread_ids
        ):
            yield _violation(
                when,
                "address_early",
                f"{label} carried the pickup address before the thread had a booked "
                "pickup past its undo deadline",
            )


Delivery = tuple[str, set[int], datetime]  # label, matching thread ids, when


def _stored_deliveries(facts: _Facts, address: str) -> Iterator[Delivery]:
    for message, thread in facts.system_outbound():
        if PLACEHOLDER in message.body or _leaks_address(message.body, address):
            label = f"message #{message.id} to thread {thread.id}"
            yield label, {message.thread_id}, _utc(message.sent_at)


def _logged_deliveries(
    facts: _Facts, address: str, sent_log: Iterable[SentRecord]
) -> Iterator[Delivery]:
    by_remote: dict[str, set[int]] = defaultdict(set)
    for thread_id, thread in facts.threads.items():
        by_remote[thread.remote_id].add(thread_id)
    for remote_id, body, sent_at in sent_log:
        if _leaks_address(body, address):
            ids = by_remote.get(remote_id, set())
            label = f"delivery to remote thread {remote_id!r} (thread {sorted(ids) or 'unknown'})"
            yield label, ids, _utc(sent_at)


# Text parsing, written independently of guard/invariants.py

_AMOUNT = r"(?P<n>\d{1,3}(?:,\d{3})+|\d+)(?P<f>\.\d{1,2})?"
_BARE = r"(?<![\w.,:$/-])" + _AMOUNT + r"(?![\w%:/'\"-]|\.\d)"
_MONEY = [
    re.compile(r"\$\s?" + _AMOUNT + r"(?:\s?(?P<k>k)\b)?", re.I),
    re.compile(
        r"(?<![\w.,])" + _AMOUNT + r"(?:\s?(?P<k>k)\b)?\s?(?:\$|(?:bucks|dollars?|cad)\b)", re.I
    ),
    re.compile(r"(?<![\w.,$])" + _AMOUNT + r"(?P<k>k)\b", re.I),
    re.compile(_BARE + r"\s+(?:cash|firm|obo)\b", re.I),
]
# Bare numbers in offer phrasing. 1 to 12 there is an hour or a count ("can you do 6?"), the
# rule the guard enforces too; only a firm trailer ("6 cash") makes a small bare number money.
_OFFER = re.compile(
    r"\b(?:take|do|offer(?:ed|ing)?|accept|counter)\s+(?:at\s+)?" + _BARE
    + r"(?=\s*(?:$|[?!.,;)])|\s+(?:cash|firm|obo|for|if|and|or|then)\b)",
    re.I,
)
# Agreeing to a price: "deal at 80", "yours for 80", "80 is fine", "80 sounds good".
_AGREED = [
    re.compile(
        r"\b(?:deal(?:\s+(?:at|for))?|yours\s+for|sold\s+(?:it\s+)?(?:for|at)"
        r"|let\s+it\s+go\s+(?:for|at)|go\s+(?:to|at|for)|agreed?\s+(?:at|on|to)"
        r"|(?:happy|ok|okay|fine|good)\s+with|make\s+it|call\s+it)\s+" + _BARE,
        re.I,
    ),
    re.compile(
        _BARE + r"\s*,?\s+(?:is\s+(?:fine|good|great|perfect|ok|okay)|it\s+is|sounds\s+good"
        r"|would\s+work|works|and\s+it'?s\s+yours)\b",
        re.I,
    ),
]
_SIGNS = str.maketrans({"\u20ac": "$", "\u00a3": "$", "\uff04": "$", "\ufe69": "$"})
_COMMA_CENTS = re.compile(r"(\d),(\d\d)(?=\s?\$)|(\$\s?\d+),(\d\d)\b")
_WORDS = {
    **dict(zip("one two three four five six seven eight nine".split(), range(1, 10), strict=True)),
    **dict(zip("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
               .split(), range(10, 20), strict=True)),
    **dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(),
               range(20, 100, 10), strict=True)),
}
_SPELLED_MONEY = re.compile(
    r"\b((?:(?:a|" + "|".join(_WORDS) + r")[\s-]+hundred(?:[\s-]+and)?[\s-]*)?"
    r"(?:(?:" + "|".join(_WORDS) + r")(?:[\s-]+(?:" + "|".join(_WORDS) + r"))?)?)"
    r"\s*(?:bucks|dollars?)\b",
    re.I,
)
_QUESTION = re.compile(_BARE + r"\s*\?")
HOUR_OR_COUNT_MAX = 12
# "N?" is an offer unless a time or date word precedes it: "at 6?", "Saturday 10?", "Oct 12?".
_NOT_PRICE_BEFORE = frozenset(
    "at by around after before until till from on the since between and or to "
    "mon tue wed thu fri sat sun monday tuesday wednesday thursday friday saturday sunday "
    "jan feb mar apr may jun jul aug sep sept oct nov dec january february march april "
    "june july august september october november december".split()
)
_CONTACT = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"\bhttps?://|\bwww\.|\b[\w-]+\.(?:com|ca|net|org|io|ly|info|biz)\b", re.I),
    re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    re.compile(r"(?<![\d$,.])\d{3}[.-]\d{4}(?!\d)"),
]


def _cents(match: re.Match[str]) -> int:
    value = Decimal(match["n"].replace(",", "") + (match["f"] or ""))
    if match.groupdict().get("k"):
        value *= 1000
    return int(value * 100)


def _word_before(text: str, index: int) -> str:
    found = re.search(r"([a-z]+)[^a-z\d]*$", text[:index].lower())
    return found[1] if found else ""


def _hour_or_count(match: re.Match[str]) -> bool:
    whole = match["n"]
    return "," not in whole and int(whole) <= HOUR_OR_COUNT_MAX


def _spelled_cents(words: str) -> int | None:
    parts = [w for w in re.split(r"[\s-]+", words.lower()) if w and w != "and"]
    if not parts:
        return None
    value = 0
    for word in parts:
        value = value * 100 if word == "hundred" else value + (1 if word == "a" else _WORDS[word])
    return value * 100


def _money_amounts_cents(text: str) -> list[int]:
    text = _COMMA_CENTS.sub(lambda m: m.group().replace(",", "."), text.translate(_SIGNS))
    amounts = [_cents(m) for pattern in _MONEY for m in pattern.finditer(text)]
    amounts += [_cents(m) for m in _OFFER.finditer(text) if not _hour_or_count(m)]
    amounts += [
        _cents(m) for pattern in _AGREED for m in pattern.finditer(text) if not _hour_or_count(m)
    ]
    amounts += [
        cents
        for m in _SPELLED_MONEY.finditer(text)
        if (cents := _spelled_cents(m[1])) is not None
    ]
    amounts += [
        _cents(m)
        for m in _QUESTION.finditer(text)
        if _word_before(text, m.start()) not in _NOT_PRICE_BEFORE and not _hour_or_count(m)
    ]
    return amounts


def _has_contact_info(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CONTACT)


# Guardrail 8: shipping, e-transfer, PayPal, deposits, or paying ahead, in a sentence with no
# negation at all. Looser than the guard, which also wants the negation to govern the term.
_REMOTE_TERMS = re.compile(
    r"\b(?:ship(?:ping|ped)?|canada\s+post|couriers?|e-?transfers?|interac|paypal|venmo|zelle"
    r"|deposits?|pay\s+(?:first|upfront|up\s+front|in\s+advance|ahead|before))\b",
    re.I,
)
_DECLINES = re.compile(
    r"\b(?:no|not|never|nor|cannot|unable|cant|wont|dont|doesnt|isnt|arent|didnt|couldnt"
    r"|wouldnt|instead\s+of|rather\s+than)\b|n['\u2019]t\b",
    re.I,
)


def _remote_deal(text: str) -> str | None:
    for sentence in re.split(r"[.!?\n]+", text):
        term = _REMOTE_TERMS.search(sentence)
        if term and not _DECLINES.search(sentence):
            return term.group()
    return None


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _leaks_address(text: str, address: str) -> bool:
    """The full address ignoring case and spacing, or its house number and street name."""
    if _squash(address) in _squash(text):
        return True
    head = re.match(r"\s*(\d+[a-z]?)\s+([^\W\d]+)", address, re.I)
    return head is not None and bool(
        re.search(rf"\b{re.escape(head[1])}\s+{re.escape(head[2])}\b", text, re.I)
    )
