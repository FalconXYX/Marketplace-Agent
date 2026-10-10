"""Offline stand-in for the reasoning model, so the whole agent runs without spending quota.

Simulation only: the production runtime never imports this module. HeuristicPolicy is a
ScriptedProvider responder. It reads ITEM_ID from the prompt's first line and the rest from the
database, and answers in the model's wire format, so its plans go through the same agent loop,
guard, and tools as a real model's. It plays a competent seller:

| Situation | Plan |
|---|---|
| A buyer names a price at or above the bar | accept_offer, book_pickup (next free slot), confirm |
| The owner approved a parked accept_offer | book_pickup, confirm |
| A buyer names a lower price | Counter between their price and the last ask, never under the bar |
| Question or availability | Answer from the item's attributes |
| Deal booked, undo window closed, buyer writes | Send the [PICKUP_ADDRESS] placeholder |
| Buyer says they picked it up | Thank them, complete_sale, remember the outcome |
| Pickup over, no word, the owner may still answer | Wait: silence is no evidence of a no-show |
| The owner's time to answer is up, no word | Release as a no-show, re-engage other buyers |
| Others write while the item is promised | "It's pending", once per deal |
| Prompt injection or a warn-level scam signal | Escalate once, reply with the real terms |
| Quiet, comps never checked | query_comps, then remember the median |
| Quiet, listed 2+ days, nothing tuned for 36h | get_stats, then rewrite, cut the price, or relist |
| Quiet, a buyer silent 24h after our message | One follow-up |
| A follow-up will be allowed later | schedule_wake for then |
| Quiet, deadline passed | Drop the price to the floor |
| A buyer wants other items too (has threads on them, names them) | One price for all; accept_bundle |
| That buyer's bundle talk is in another item's thread | Say so once and leave it there |

The bar starts halfway between list and floor and falls linearly to the floor at the deadline;
a firm item holds its list price. A bundle's bar is its items' bars added up, and every price
said about a bundle is its total. The bundle is run from the item whose thread it was agreed in
(the first item_ids entry): the other items' wakes leave the deal alone.
"""

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import Engine
from sqlmodel import Session, col, func, select

from marketplace_agent.ai.base import LLMRequest
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.core.items import get_attributes, get_goal, get_item, item_photos
from marketplace_agent.core.wakes import pending_scheduled
from marketplace_agent.core.models import (
    AgentRun,
    Buyer,
    Direction,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Listing,
    ListingStatsSnapshot,
    ListingStatus,
    Message,
    PriceChange,
    Thread,
    ThreadState,
)
from marketplace_agent.guard.invariants import (
    ADDRESS_PLACEHOLDER,
    detect_scam,
    money_amounts_cents,
    offer_cents,
    outbound_streak,
    to_local,
)
from marketplace_agent.guard.handover import no_show_from
from marketplace_agent.guard.ledger import (
    ActiveCommitment,
    active_commitment,
    active_pickups,
    deal_rows,
)

RESULTS_HEADER = "## RESULTS OF YOUR LAST PLAN"
SLOT_LENGTH = timedelta(minutes=60)
SLOT_LEAD = timedelta(hours=3)  # time for the buyer to answer and the undo window to close
SLOT_STEP = timedelta(minutes=30)
SLOT_SEARCH_DAYS = 8
FOLLOW_UP_AFTER = timedelta(hours=24)
COMPLETE_EARLIEST = timedelta(hours=1)
TUNE_AFTER = timedelta(days=2)  # listing age before the first stale-listing move
TUNE_EVERY = timedelta(hours=36)
STATS_EVERY = timedelta(hours=24)
PRICE_CHANGE_EVERY = timedelta(hours=24)
RELIST_EVERY = timedelta(hours=72)
PRICE_CUT = 0.95
REENGAGE = 2  # buyers messaged when a deal falls through
BUNDLE_ASK = 0.95  # of the list prices together, when a buyer wants several and names no price
DEFER_MARK = "I'll answer about"  # in the reply that points a buyer to their bundle thread

_ITEM_ID = re.compile(r"\AITEM_ID: (\d+)")
_STEP = re.compile(r"This is step (\d+) of")
_OUTCOME = re.compile(r"^- step (\d+) (\w+) .*? -> (\w+): (.*)$", re.MULTILINE)
_MEDIAN = re.compile(r'"median_cents":\s*(\d+)')
_COUNT = re.compile(r'"count":\s*(\d+)')
_INJECTION = re.compile(
    r"ignore (?:all |any )?(?:previous|prior) instructions|\bsystem\s*:|\bassistant\s*:"
    r"|accept_offer|price_cents|owner (?:override|approved)|buyer_message"
    r"|reveal the (?:pickup )?address|seller (?:already )?agreed",
    re.IGNORECASE,
)
_WALK = re.compile(
    r"\b(?:i'll pass|pass\.|no worries|good luck|too rich|more than i want|thanks anyway"
    r"|falls? through)",
    re.IGNORECASE,
)
_BACKUP = re.compile(r"\bfalls? through\b", re.IGNORECASE)  # "let me know if it falls through"
_PICKED_UP = re.compile(r"\bpicked (?:it |them |both |everything )?up\b", re.IGNORECASE)
_PENDING = re.compile(r"\bpending\b", re.IGNORECASE)
_AVAILABILITY = re.compile(r"\bavailable\b|\bstill (?:have|for sale)\b", re.IGNORECASE)
_SCHEDULING = re.compile(
    r"\b(?:today|tonight|tomorrow|morning|afternoon|evening|weekend|pick (?:it )?up|pickup"
    r"|come get)\b|\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b",
    re.IGNORECASE,
)
_WORDS = re.compile(r"[A-Za-z]+")
_STOPWORDS = frozenset({"a", "an", "and", "for", "of", "the", "with"})
_DIGITS = re.compile(r"\S*\d\S*")
_CONDITIONS = {
    "new": "new",
    "like_new": "like new",
    "good": "good",
    "fair": "fair",
    "poor": "rough",
}
_ANSWERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"dimension|size|how (?:big|wide|tall|long)|measure", re.IGNORECASE), "dimensions"),
    (re.compile(r"scratch|damage|condition|marks|dents", re.IGNORECASE), "condition"),
    (re.compile(r"how old|\bage\b|bought it", re.IGNORECASE), "age"),
    (re.compile(r"smok|pets?\b", re.IGNORECASE), "home"),
    (re.compile(r"\bwork|function", re.IGNORECASE), "works"),
    (_AVAILABILITY, "available"),
)

Outcome = tuple[int, str, str, str]  # step, tool, status, detail


class HeuristicPolicy:
    """ScriptedProvider responder: ScriptedProvider(HeuristicPolicy(engine, settings, clock))."""

    def __init__(self, engine: Engine, settings: Settings, clock: Clock):
        self.engine = engine
        self.settings = settings
        self.clock = clock

    def __call__(self, request: LLMRequest) -> dict[str, Any]:
        found = _ITEM_ID.match(request.prompt)
        if found is None:
            return _Plan(["The briefing has no ITEM_ID line, so I cannot act."]).wire()
        with Session(self.engine) as session:
            board = _Board.load(session, self.settings, self.clock.now(), int(found[1]))
            if board is None:
                return _Plan(["The item has no goal or is not for sale; nothing to do."]).wire()
            if RESULTS_HEADER in request.prompt:
                return _after_results(board, _last_outcomes(request.prompt)).wire()
            return _Planner(board).run().wire()


# ---------------------------------------------------------------- what the policy sees


@dataclass
class _Conversation:
    thread: Thread
    buyer: Buyer
    messages: list[Message]  # oldest first
    streak: int  # our messages since the buyer last wrote
    last_sent_at: datetime | None

    @property
    def id(self) -> int:
        assert self.thread.id is not None
        return self.thread.id

    @property
    def name(self) -> str:
        return (self.buyer.display_name or self.buyer.handle).split()[0]

    @property
    def blocked(self) -> bool:
        return self.thread.state == ThreadState.FROZEN or self.buyer.flagged_scam

    @property
    def new(self) -> list[Message]:
        if self.thread.state == ThreadState.FROZEN:
            return []
        return [m for m in self.messages if m.direction == Direction.INBOUND and not m.handled]

    @property
    def new_text(self) -> str:
        return " ".join(m.body for m in self.new)

    def may_message(self, now: datetime) -> bool:
        if self.blocked or self.streak >= 2:
            return False
        return self.streak == 0 or (
            self.last_sent_at is not None and now - self.last_sent_at >= FOLLOW_UP_AFTER
        )

    def last_ask(self) -> int | None:
        """The price in our latest message that named one."""
        for message in reversed(self.messages):
            if message.direction == Direction.OUTBOUND:
                amounts = money_amounts_cents(message.body)
                if amounts:
                    return max(amounts)
        return None

    def said_since(self, pattern: re.Pattern[str] | str, since: datetime) -> bool:
        def matches(body: str) -> bool:
            return pattern in body if isinstance(pattern, str) else bool(pattern.search(body))

        return any(
            m.direction == Direction.OUTBOUND and m.sent_at >= since and matches(m.body)
            for m in self.messages
        )

    def backup(self) -> bool:
        """Asked to hear if the current deal falls through."""
        return any(
            m.direction == Direction.INBOUND and _BACKUP.search(m.body) for m in self.messages
        )

    def interested(self) -> bool:
        """A buyer worth another message: named a price or asked something, has not walked away
        (unless as a backup), never no-showed, and never sent an injection."""
        inbound = [m.body for m in self.messages if m.direction == Direction.INBOUND]
        if not inbound or self.buyer.no_show_count or _INJECTION.search(" ".join(inbound)):
            return False
        if _WALK.search(inbound[-1]) and not self.backup():
            return False
        return self.thread.last_offer_cents is not None or any("?" in body for body in inbound)


@dataclass
class _Other:
    """Another of the owner's items a buyer could take with this one: listed, unpaused, with a
    goal, and promised to nobody."""

    item: Item
    goal: Goal
    start: datetime  # first published, else created: where its bar starts
    keywords: frozenset[str]  # title words that this item's title does not have

    @property
    def id(self) -> int:
        assert self.item.id is not None
        return self.item.id

    def bar(self, now: datetime) -> int:
        return _bar(self.goal, self.start, now)


@dataclass
class _BundleOffer:
    total: int
    conversation: _Conversation
    partners: list[_Other]


@dataclass
class _Board:
    settings: Settings
    now: datetime
    item: Item
    goal: Goal
    deal: ActiveCommitment | None
    conversations: list[_Conversation]
    photo_ids: list[int]
    live: list[Listing]
    first_published: datetime | None
    busy: list[tuple[datetime, datetime]]  # every booked pickup, all items
    history: list[tuple[datetime, str, str]]  # (run start, tool, status) of finished runs
    escalated: frozenset[int]  # thread ids with an open escalation
    last_price_change: datetime | None
    views_last_day: int
    scheduled: datetime | None  # our pending scheduled wake
    others: list[_Other] = field(default_factory=list)
    other_threads: dict[int, dict[int, int]] = field(default_factory=dict)  # buyer: item: thread
    deal_items: list[int] = field(default_factory=list)  # the active deal's items, lead first
    deal_total: int = 0  # the active deal's price; a bundle's total

    @classmethod
    def load(
        cls, session: Session, settings: Settings, now: datetime, item_id: int
    ) -> "_Board | None":
        item = get_item(session, item_id)
        goal = get_goal(session, item_id)
        if (
            goal is None
            or item.paused
            or item.status not in (ItemStatus.LISTED, ItemStatus.COMMITTED)
        ):
            return None
        listings = session.exec(select(Listing).where(Listing.item_id == item_id)).all()
        published = [x.published_at for x in listings if x.published_at is not None]
        deal = active_commitment(session, item_id)
        rows = deal_rows(session, deal.accepted) if deal else []
        others = _others(session, item)
        return cls(
            settings=settings,
            now=now,
            item=item,
            goal=goal,
            deal=deal,
            conversations=_conversations(session, item_id),
            photo_ids=[p.id for p in item_photos(session, item_id) if p.id is not None],
            live=[x for x in listings if x.status == ListingStatus.LIVE],
            first_published=min(published) if published else None,
            busy=[
                (a.pickup.pickup_start, a.pickup.pickup_end)
                for a in active_pickups(session)
                if a.pickup and a.pickup.pickup_start and a.pickup.pickup_end
            ],
            history=_history(session, item_id),
            escalated=frozenset(
                e.thread_id
                for e in session.exec(
                    select(Escalation).where(
                        Escalation.item_id == item_id, col(Escalation.resolved_at).is_(None)
                    )
                ).all()
                if e.thread_id is not None
            ),
            last_price_change=session.exec(
                select(PriceChange.created_at)
                .where(PriceChange.item_id == item_id)
                .order_by(col(PriceChange.created_at).desc())
            ).first(),
            views_last_day=_views_since(session, listings, now - timedelta(days=1)),
            scheduled=next((w.created_at for w in pending_scheduled(session, item_id)), None),
            others=others,
            other_threads=_other_threads(session, [other.id for other in others]),
            deal_items=[row.item_id for row in rows],
            deal_total=sum(row.price_cents or 0 for row in rows),
        )

    def conversation(self, thread_id: int | None) -> _Conversation | None:
        return next((c for c in self.conversations if c.id == thread_id), None)

    @property
    def firm(self) -> bool:
        return self.goal.firmness == Firmness.FIRM

    def bar(self) -> int:
        """Lowest price worth accepting now."""
        return _bar(self.goal, self.first_published or self.item.created_at, self.now)

    @property
    def leads_deal(self) -> bool:
        """This item's wakes run the active deal: a single deal, or the bundle's first item."""
        return not self.deal_items or self.deal_items[0] == self.item.id

    def partners(self, conversation: _Conversation) -> list[_Other]:
        """The other items this buyer wants too: they have an open thread about it, or name it."""
        threads = self.other_threads.get(conversation.buyer.id or 0, {})
        inbound = " ".join(
            m.body for m in conversation.messages if m.direction == Direction.INBOUND
        )
        words = {w.lower() for w in _WORDS.findall(inbound)}
        return [o for o in self.others if o.id in threads or _names(words, o.keywords)]

    def bundle_lead(self, conversation: _Conversation, partners: list[_Other]) -> _Other | None:
        """The partner whose thread with this buyer is older, where the bundle is talked
        through; None when it is this conversation."""
        threads = self.other_threads.get(conversation.buyer.id or 0, {})
        older = [
            (threads[o.id], o)
            for o in partners
            if threads.get(o.id, conversation.id) < conversation.id
        ]
        return min(older, key=lambda pair: pair[0])[1] if older else None

    def executed(self) -> Counter[str]:
        return Counter(tool for _, tool, status in self.history if status == "executed")

    def last_time(self, tools: tuple[str, ...], status: str) -> datetime | None:
        times = [at for at, tool, s in self.history if tool in tools and s == status]
        return max(times) if times else None

    def local(self, when: datetime) -> str:
        local = to_local(self.settings, when)
        return f"{local:%a %b} {local.day} at {local:%H:%M}"


def _conversations(session: Session, item_id: int) -> list[_Conversation]:
    found = []
    threads = session.exec(
        select(Thread).where(Thread.item_id == item_id).order_by(col(Thread.id))
    ).all()
    for thread in threads:
        assert thread.id is not None
        buyer = session.get(Buyer, thread.buyer_id)
        assert buyer is not None
        messages = session.exec(
            select(Message)
            .where(Message.thread_id == thread.id)
            .order_by(col(Message.sent_at), col(Message.id))
        ).all()
        streak, last = outbound_streak(session, thread.id)
        found.append(_Conversation(thread, buyer, list(messages), streak, last))
    return found


def _others(session: Session, item: Item) -> list[_Other]:
    candidates = session.exec(
        select(Item)
        .where(Item.id != item.id, Item.status == ItemStatus.LISTED, col(Item.paused).is_(False))
        .order_by(col(Item.id))
    ).all()
    own = _keywords(item.title)
    found = []
    for other in candidates:
        assert other.id is not None
        goal = get_goal(session, other.id)
        if goal is None or active_commitment(session, other.id) is not None:
            continue
        published = session.exec(
            select(func.min(Listing.published_at)).where(Listing.item_id == other.id)
        ).one()
        start = published or other.created_at
        found.append(_Other(other, goal, start, _keywords(other.title) - own))
    return found


def _other_threads(session: Session, item_ids: list[int]) -> dict[int, dict[int, int]]:
    """buyer id -> item id -> the buyer's oldest open thread about it, for item_ids."""
    found: dict[int, dict[int, int]] = {}
    if not item_ids:
        return found
    threads = session.exec(
        select(Thread)
        .where(
            col(Thread.item_id).in_(item_ids),
            col(Thread.state).not_in((ThreadState.CLOSED, ThreadState.FROZEN)),
        )
        .order_by(col(Thread.id))
    ).all()
    for thread in threads:
        assert thread.id is not None and thread.item_id is not None
        found.setdefault(thread.buyer_id, {}).setdefault(thread.item_id, thread.id)
    return found


def _history(session: Session, item_id: int) -> list[tuple[datetime, str, str]]:
    runs = session.exec(
        select(AgentRun)
        .where(AgentRun.item_id == item_id, col(AgentRun.finished_at).is_not(None))
        .order_by(col(AgentRun.started_at), col(AgentRun.id))
    ).all()
    history = []
    for run in runs:
        outcomes = json.loads(run.results_json or "[]")
        history += [(run.started_at, str(o.get("tool")), str(o.get("status"))) for o in outcomes]
    return history


def _views_since(session: Session, listings: Sequence[Listing], since: datetime) -> int:
    total = 0
    for listing in listings:
        snapshots = session.exec(
            select(ListingStatsSnapshot)
            .where(ListingStatsSnapshot.listing_id == listing.id)
            .order_by(col(ListingStatsSnapshot.captured_at), col(ListingStatsSnapshot.id))
        ).all()
        if snapshots:
            before = [s.views for s in snapshots if s.captured_at <= since]
            total += snapshots[-1].views - (before[-1] if before else 0)
    return total


# ---------------------------------------------------------------- the plan


@dataclass
class _Plan:
    """Actions run in three groups: ledger moves, then messages, then wrap-up (complete_sale,
    remember, escalate), so a sale never blocks a message planned in the same wake."""

    notes: list[str] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    first: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    last: list[dict[str, Any]] = field(default_factory=list)
    messaged: set[int] = field(default_factory=set)

    def add(self, group: list[dict[str, Any]], tool: str, why: str, **args: Any) -> None:
        group.append({"tool": tool, "args_json": json.dumps(args), "why": why})

    def send(self, conversation: _Conversation, body: str, why: str) -> None:
        self.messaged.add(conversation.id)
        self.add(self.messages, "send_message", why, thread_id=conversation.id, body=body)

    def wire(self) -> dict[str, Any]:
        return {
            "rationale": " ".join(self.notes) or "Nothing needs doing right now.",
            "message_labels": self.labels,
            "actions": self.first + self.messages + self.last,
        }


class _Planner:
    """Step 1 of a wake: everything the briefing calls for, in one plan."""

    def __init__(self, board: _Board):
        self.b = board
        self.out = _Plan()
        self.holder: int | None = board.deal.accepted.thread_id if board.deal else None
        self.sold = False
        self.reopened = False

    def run(self) -> _Plan:
        self._label()
        if self.b.deal is not None:
            self._tend_deal(self.b.deal)
        self._answer_buyers()
        if self.reopened and self.holder is None:
            self._reengage()
        if not any(c.new for c in self.b.conversations) and not self.out.first:
            self._initiative()
        self._schedule_follow_up()
        return self.out

    def _schedule_follow_up(self) -> None:
        """Nothing wakes the agent when a follow-up becomes allowed, so ask for a wake then."""
        if self.b.deal is not None or self.holder is not None or self.sold:
            return
        due = []
        for conversation in self.b.conversations:
            if conversation.blocked or not conversation.interested():
                continue
            if self._deferred(conversation):
                continue
            if conversation.id in self.out.messaged:
                due.append((self.b.now + FOLLOW_UP_AFTER, conversation))
            elif conversation.streak == 1 and not conversation.may_message(self.b.now):
                assert conversation.last_sent_at is not None
                due.append((conversation.last_sent_at + FOLLOW_UP_AFTER, conversation))
        if not due:
            return
        when, conversation = min(due, key=lambda pair: (pair[0], pair[1].id))
        at = when.replace(second=0, microsecond=0) + timedelta(minutes=2)
        if self.b.scheduled is not None and self.b.scheduled <= at:
            return
        local = to_local(self.b.settings, at)
        self.out.add(
            self.out.last,
            "schedule_wake",
            "A follow-up is allowed then",
            at=f"{local:%Y-%m-%dT%H:%M}",
            reason=f"Follow up with {conversation.name} if they have not replied",
        )

    # ------------------------------------------------------------ messages

    def _label(self) -> None:
        floor = self.b.goal.floor_price_cents
        for conversation in self.b.conversations:
            for message in conversation.new:
                intent = _intent(message.body, floor)
                self.out.labels.append({"message_id": message.id, "intent": intent})

    def _tend_deal(self, deal: ActiveCommitment) -> None:
        conversation = self.b.conversation(deal.accepted.thread_id)
        if not self.b.leads_deal or conversation is None:
            lead = self.b.deal_items[0] if self.b.deal_items else "?"
            self.out.notes.append(f"This item is in a bundle deal that item {lead}'s wakes run.")
            return
        pickup = deal.pickup
        text = conversation.new_text
        now = self.b.now
        if pickup is None:
            # Untold: the deal came from the owner approving a parked accept_offer or
            # accept_bundle, whose confirmation the guard held back.
            untold = not conversation.said_since("", deal.accepted.created_at)
            if (conversation.new or untold) and conversation.may_message(now):
                self._book_and_confirm(conversation, self.b.deal_total, _lot(self.b.deal_items))
            return
        assert pickup.pickup_start is not None and pickup.pickup_end is not None
        if _PICKED_UP.search(text) and now >= pickup.pickup_start - COMPLETE_EARLIEST:
            self._complete(conversation, deal)
        elif now >= no_show_from(self.b.settings, pickup.pickup_end):
            self._release(conversation, "no-show", no_show=True, when=pickup.pickup_start)
        elif now >= pickup.pickup_end:
            self.out.notes.append(
                f"{conversation.name} has not written since the pickup, which proves nothing; "
                "waiting for the owner's answer."
            )
        elif conversation.new and _WALK.search(text):
            self._release(conversation, "the buyer backed out", no_show=False)
        elif conversation.new and conversation.may_message(now):
            self._reply_to_holder(conversation, deal, pickup.pickup_start)

    def _reply_to_holder(
        self, conversation: _Conversation, deal: ActiveCommitment, start: datetime
    ) -> None:
        deadline = deal.accepted.undo_deadline
        undo_over = deadline is not None and self.b.now > deadline
        when = self.b.local(start)
        if undo_over and not conversation.said_since(ADDRESS_PLACEHOLDER, deal.accepted.created_at):
            body = f"See you {when}! The address is {ADDRESS_PLACEHOLDER}."
            self.out.send(conversation, body, "Deal booked and past the undo window")
            self.out.notes.append(f"Sent {conversation.name} the address for the pickup.")
        elif "?" in conversation.new_text:
            self.out.send(conversation, f"Yes, still on for {when}.", "Confirm the pickup")

    def _complete(self, conversation: _Conversation, deal: ActiveCommitment) -> None:
        price = self.b.deal_total
        bundled = len(self.b.deal_items) > 1
        if conversation.may_message(self.b.now):
            enjoy = "enjoy them" if bundled else "enjoy it"
            self.out.send(conversation, f"Thanks {conversation.name}, {enjoy}!", "Sign off")
        self.out.add(
            self.out.last, "complete_sale", "Buyer has the item", thread_id=conversation.id
        )
        days = (self.b.now - (self.b.first_published or self.b.item.created_at)).days
        what = f"{self.b.item.title} and {len(self.b.deal_items) - 1} other item(s)"
        lesson = (
            f"{what if bundled else self.b.item.title} sold "
            f"{'together ' if bundled else ''}for {_dollars(price)} after {days} day(s) on the "
            f"market, list price then {_dollars(self.b.goal.list_price_cents)}."
        )
        self.out.add(
            self.out.last, "remember", "Outcome for future items", scope="global", lesson=lesson
        )
        them = "them" if bundled else "it"
        self.out.notes.append(f"{conversation.name} picked {them} up; recording the sale.")
        self.holder = None
        self.sold = True

    def _release(
        self,
        conversation: _Conversation,
        reason: str,
        *,
        no_show: bool,
        when: datetime | None = None,
    ) -> None:
        why = f"no-show for the {self.b.local(when)} pickup, no word" if when else reason
        self.out.add(
            self.out.first,
            "release_commitment",
            reason,
            thread_id=conversation.id,
            reason=why,
            no_show=no_show,
        )
        if no_show:
            self.out.add(
                self.out.last,
                "remember",
                "Be careful with this buyer",
                scope="buyer",
                buyer_id=conversation.buyer.id,
                lesson=f"Booked a pickup and did not show up or write ({why}).",
            )
        self.out.notes.append(f"Released the deal with {conversation.name}: {why}.")
        self.holder = None
        self.reopened = True

    def _answer_buyers(self) -> None:
        offers: list[tuple[int, _Conversation]] = []
        bundles: list[_BundleOffer] = []
        for conversation in self.b.conversations:
            if conversation.blocked or not conversation.new:
                continue
            if conversation.id == self._dealt() or conversation.id in self.out.messaged:
                continue
            offer = self._answer(conversation)
            if isinstance(offer, _BundleOffer):
                bundles.append(offer)
            elif offer is not None:
                offers.append((offer, conversation))
        if not offers and not bundles:
            return
        offers.sort(key=lambda pair: (-pair[0], pair[1].id))
        bundles.sort(key=lambda b: (-len(b.partners), -b.total, b.conversation.id))
        if self.holder is None and not self.sold:
            if bundles:  # more items sold at once beats a better price for this one alone
                self._accept_bundle(bundles[0])
            else:
                price, buyer = offers[0]
                self._accept(buyer, price)
        waiting = [conversation for _, conversation in offers]
        waiting += [bundle.conversation for bundle in bundles]
        for conversation in waiting:
            if conversation.id not in self.out.messaged:
                self._pending(conversation)

    def _dealt(self) -> int | None:
        """The thread whose deal this plan already handled."""
        return self.b.deal.accepted.thread_id if self.b.deal else None

    def _answer(self, conversation: _Conversation) -> int | _BundleOffer | None:
        """Reply to one buyer's new messages. Returns an acceptable price, or bundle, to weigh
        against the other buyers' instead of replying."""
        text = conversation.new_text
        if _INJECTION.search(text):
            self._deflect(conversation, "is writing instructions aimed at an AI (prompt injection)")
            return None
        signal = detect_scam(text)
        if signal is not None:
            self._deflect(conversation, signal.reason)
            return None
        if _WALK.search(text):
            self.out.notes.append(f"{conversation.name} is out for now.")
            return None
        if not conversation.may_message(self.b.now):
            return None
        if self.holder is not None or self.sold:
            self._pending(conversation)
            return None
        partners = self.b.partners(conversation)
        if partners:
            return self._answer_bundle(conversation, partners)
        price = _buyer_price(text)
        if price is not None and price >= self.b.bar():
            return price
        if price is not None:
            self._counter(conversation, price)
        else:
            self._inform(conversation, text)
        return None

    def _answer_bundle(
        self, conversation: _Conversation, partners: list[_Other]
    ) -> _BundleOffer | None:
        """Every price is the total for this item and partners together."""
        lead = self.b.bundle_lead(conversation, partners)
        if lead is not None:
            self._defer(conversation, lead, _lot([self.b.item.id, *(o.id for o in partners)]))
            return None
        now = self.b.now
        bar = self.b.bar() + sum(other.bar(now) for other in partners)
        price = _buyer_price(conversation.new_text)
        if price is not None and price >= bar:
            return _BundleOffer(price, conversation, partners)
        listed = self.b.goal.list_price_cents + sum(o.goal.list_price_cents for o in partners)
        things = self._things(partners)
        lot = _lot([self.b.item.id, *(other.id for other in partners)])
        if price is None:
            ask = min(listed, max(_round_up(bar), _round_down(listed * BUNDLE_ASK)))
            facts = _facts(self.b)
            text = conversation.new_text
            answers = [
                facts[key]
                for pattern, key in _ANSWERS
                if key != "available" and pattern.search(text) and facts.get(key)
            ]
            body = " ".join(
                [
                    f"Hi {conversation.name}!",
                    *dict.fromkeys(answers),
                    f"Yes, {lot} are available. I could do {things} together for "
                    f"{_dollars(ask)}.",
                ]
            )
        else:
            anchor = min(conversation.last_ask() or listed, listed)
            counter = min(listed, max(_round_up(bar), _round_up((price + anchor) / 2)))
            body = (
                f"Thanks {conversation.name}! I can't go that low for {lot}, but I could do "
                f"{things} together for {_dollars(counter)}."
            )
        self.out.send(conversation, body, "They want several items: one price for all")
        self.out.notes.append(f"Offered {conversation.name} a bundle.")
        return None

    def _defer(self, conversation: _Conversation, lead: _Other, lot: str) -> None:
        """The buyer's older thread about lead is where their bundle is talked through."""
        if conversation.said_since(DEFER_MARK, conversation.thread.created_at):
            return
        body = (
            f"Hi {conversation.name}! Yes, it's still available. {DEFER_MARK} {lot} in your "
            f"message about the {_plain(lead.item.title)}."
        )
        self.out.send(conversation, body, "Their bundle is talked through in another thread")
        self.out.notes.append(f"Pointed {conversation.name} to their other thread.")

    def _things(self, partners: list[_Other]) -> str:
        titles = [self.b.item.title, *(other.item.title for other in partners)]
        return " and ".join(f"the {_plain(title)}" for title in titles)

    def _deflect(self, conversation: _Conversation, reason: str) -> None:
        if conversation.id not in self.b.escalated:
            self.out.add(
                self.out.last,
                "escalate",
                "Suspicious buyer",
                thread_id=conversation.id,
                reason=f"{conversation.name} in thread {conversation.id} {reason}. I am "
                "ignoring it and offering only cash at pickup at the real price.",
            )
        if conversation.may_message(self.b.now) and self.holder is None and not self.sold:
            body = (
                f"Hi {conversation.name}, it's {_dollars(self.b.goal.list_price_cents)}, cash "
                f"at pickup near {self.b.settings.pickup_area}. I share the address once a "
                "pickup is booked."
            )
            self.out.send(conversation, body, "State the real terms")
        self.out.notes.append(f"{conversation.name}'s messages look suspicious; escalated.")

    def _pending(self, conversation: _Conversation) -> None:
        since = self.b.deal.accepted.created_at if self.b.deal else self.b.now
        if self.sold:
            body = f"Sorry {conversation.name}, it just sold."
        elif conversation.said_since(_PENDING, since):
            return
        else:
            body = (
                f"Hi {conversation.name}, it's pending with someone else right now. Want me to "
                "message you if it falls through?"
            )
        if conversation.may_message(self.b.now):
            self.out.send(conversation, body, "Item is promised to another buyer")
            self.out.notes.append(f"Told {conversation.name} it is pending.")

    def _counter(self, conversation: _Conversation, offer: int) -> None:
        goal = self.b.goal
        if self.b.firm:
            body = (
                f"Thanks {conversation.name}, but the price is firm at "
                f"{_dollars(goal.list_price_cents)}."
            )
        else:
            anchor = min(conversation.last_ask() or goal.list_price_cents, goal.list_price_cents)
            counter = min(
                goal.list_price_cents,
                max(_round_up(self.b.bar()), _round_up((offer + anchor) / 2)),
            )
            body = (
                f"Thanks {conversation.name}! I can't go that low, but I could do "
                f"{_dollars(counter)}."
            )
        self.out.send(conversation, body, "Counter above the bar")
        self.out.notes.append(f"Countered {conversation.name}'s offer.")

    def _inform(self, conversation: _Conversation, text: str) -> None:
        facts = _facts(self.b)
        answers = [
            facts[key] for pattern, key in _ANSWERS if pattern.search(text) and facts.get(key)
        ]
        if not answers:
            answers = [facts["condition"], f"Pickup is near {self.b.settings.pickup_area}."]
        body = f"Hi {conversation.name}! " + " ".join(dict.fromkeys(answers))
        self.out.send(conversation, body, "Answer their question")

    def _accept(self, conversation: _Conversation, price: int) -> None:
        self.out.add(
            self.out.first,
            "accept_offer",
            "Offer clears the bar",
            thread_id=conversation.id,
            price_cents=price,
        )
        self.holder = conversation.id
        self._book_and_confirm(conversation, price)
        self.out.notes.append(f"Accepted {_dollars(price)} from {conversation.name}.")

    def _accept_bundle(self, offer: _BundleOffer) -> None:
        conversation = offer.conversation
        ids = [self.b.item.id, *(other.id for other in offer.partners)]
        self.out.add(
            self.out.first,
            "accept_bundle",
            "The total clears every item's bar together",
            thread_id=conversation.id,
            item_ids=ids,
            total_cents=offer.total,
        )
        self.holder = conversation.id
        self._book_and_confirm(conversation, offer.total, _lot(ids))
        self.out.notes.append(
            f"Accepted {_dollars(offer.total)} from {conversation.name} for "
            f"{self._things(offer.partners)} together."
        )

    def _book_and_confirm(self, conversation: _Conversation, price: int, lot: str = "") -> None:
        """lot: "both" or "all N" for a bundle, whose price is its total."""
        yours = f"{lot} are yours" if lot else "it's yours"
        start = _next_slot(self.b.settings, self.b.now, self.b.busy)
        if start is None:
            body = f"Deal, {yours} for {_dollars(price)}. When suits you for pickup?"
        else:
            local = to_local(self.b.settings, start)
            self.out.add(
                self.out.first,
                "book_pickup",
                "Next free slot inside availability",
                thread_id=conversation.id,
                start=f"{local:%Y-%m-%dT%H:%M}",
                duration_minutes=int(SLOT_LENGTH.total_seconds() // 60),
            )
            body = (
                f"Deal, {yours} for {_dollars(price)}. I've booked you for "
                f"{self.b.local(start)}, near {self.b.settings.pickup_area}. Message me when "
                "you're heading over and I'll send the address."
            )
        self.out.send(conversation, body, "Confirm the deal and pickup")

    def _reengage(self) -> None:
        waiting = [
            c
            for c in self.b.conversations
            if c.id not in self.out.messaged
            and c.id != self._dealt()
            and c.interested()
            and c.may_message(self.b.now)
        ]
        waiting.sort(key=lambda c: (not c.backup(), -(c.thread.last_offer_cents or 0), c.id))
        for conversation in waiting[:REENGAGE]:
            body = (
                f"Hi {conversation.name}, it's available again if you're still interested. "
                f"Pickup is near {self.b.settings.pickup_area}."
            )
            self.out.send(conversation, body, "Deal fell through; back to the next buyer")
            self.out.notes.append(f"Told {conversation.name} it is available again.")

    # ------------------------------------------------------------ initiative

    def _initiative(self) -> None:
        if self.b.deal is not None:
            return
        self._follow_ups()
        floor = _deadline_price(self.b)
        if floor is not None:
            self.out.add(self.out.first, "update_price", "Deadline passed", price_cents=floor)
            self.out.notes.append(f"The deadline passed; dropping to {_dollars(floor)}.")
        elif self._comps_due():
            query = _comps_query(self.b.item)
            self.out.add(self.out.last, "query_comps", "What similar items ask", query=query)
            self.out.notes.append("Checking comparable listings first.")
        elif self._tune_due():
            self.out.add(self.out.last, "get_stats", "Is the listing stale?")
            self.out.notes.append("Quiet listing; checking stats before changing anything.")

    def _follow_ups(self) -> None:
        for conversation in self.b.conversations:
            if conversation.streak != 1 or not conversation.interested():
                continue
            if self._deferred(conversation):
                continue
            if not conversation.may_message(self.b.now):
                continue
            ask = conversation.last_ask()
            offer = f" I could still do {_dollars(ask)}." if ask and ask >= self.b.bar() else ""
            body = f"Hi {conversation.name}, just checking in. It's still available.{offer}"
            self.out.send(conversation, body, "One follow-up after 24h of silence")
            self.out.notes.append(f"Followed up with {conversation.name}.")

    def _deferred(self, conversation: _Conversation) -> bool:
        """The buyer's bundle is talked through in their thread about another item."""
        partners = self.b.partners(conversation)
        return bool(partners) and self.b.bundle_lead(conversation, partners) is not None

    def _comps_due(self) -> bool:
        never = self.b.last_time(("query_comps",), "observed") is None
        return never and len(_comps_query(self.b.item)) >= 3

    def _tune_due(self) -> bool:
        b = self.b
        if b.first_published is None or b.now - b.first_published < TUNE_AFTER:
            return False
        tuned = b.last_time(("rewrite_listing", "update_price", "relist"), "executed")
        looked = b.last_time(("get_stats",), "observed")
        return (
            (tuned is None or b.now - tuned >= TUNE_EVERY)
            and (looked is None or b.now - looked >= STATS_EVERY)
            and _tune_move(b) is not None
        )


# ---------------------------------------------------------------- steps 2 and 3


def _last_outcomes(prompt: str) -> list[Outcome]:
    results = prompt[prompt.index(RESULTS_HEADER) :]
    found = _STEP.search(results)
    step = int(found[1]) if found else 2
    return [
        (int(m[1]), m[2], m[3], m[4]) for m in _OUTCOME.finditer(results) if int(m[1]) == step - 1
    ]


def _after_results(board: _Board, outcomes: list[Outcome]) -> _Plan:
    plan = _Plan()
    problems = [
        f"{tool} {status} ({detail})"
        for _, tool, status, detail in outcomes
        if status in ("rejected", "failed")
    ]
    if problems:
        plan.notes.append("Leaving these for the next wake: " + "; ".join(problems))
        return plan
    for _, tool, status, detail in outcomes:
        if status != "observed":
            continue
        if tool == "query_comps":
            _learn_comps(board, plan, detail)
        elif tool == "get_stats":
            _tune(board, plan)
    return plan


def _learn_comps(board: _Board, plan: _Plan, detail: str) -> None:
    median, count = _MEDIAN.search(detail), _COUNT.search(detail)
    if median is None or count is None or int(count[1]) == 0:
        plan.notes.append("No comparable listings found.")
        return
    query = _comps_query(board.item)
    lesson = (
        f"Comparable {query} listings ask about {_dollars(int(median[1]))} (median of {count[1]})."
    )
    plan.add(plan.last, "remember", "Market reference", scope="global", lesson=lesson)
    plan.notes.append(lesson)


def _tune(board: _Board, plan: _Plan) -> None:
    move = _tune_move(board)
    if move is None:
        plan.notes.append("Nothing left to tune right now.")
        return
    tool, args, what = move
    plan.add(plan.first, tool, what, **args)
    age = (board.now - (board.first_published or board.item.created_at)).days
    lesson = (
        f"Day {age}: {board.views_last_day} views in a day at "
        f"{_dollars(board.goal.list_price_cents)} and no deal, so I {what}."
    )
    plan.add(plan.last, "remember", "Judge this change later", scope="item", lesson=lesson)
    plan.notes.append(lesson)


def _tune_move(board: _Board) -> tuple[str, dict[str, Any], str] | None:
    """The next stale-listing move: rewrite once, then alternate price cuts and relists."""
    executed = board.executed()
    rewrite = _rewrite_args(board)
    if executed["rewrite_listing"] == 0 and rewrite:
        return "rewrite_listing", rewrite, "rewrote the title and reordered the photos"
    moves: list[tuple[int, str, dict[str, Any], str]] = []
    cut = _price_cut(board)
    if cut is not None:
        moves.append(
            (
                executed["update_price"],
                "update_price",
                {"price_cents": cut},
                f"cut the price to {_dollars(cut)}",
            )
        )
    platform = _relist_platform(board)
    if platform is not None:
        moves.append(
            (executed["relist"], "relist", {"platform": platform}, f"relisted on {platform}")
        )
    if not moves:
        return None
    _, tool, args, what = min(moves, key=lambda move: move[0])
    return tool, args, what


def _rewrite_args(board: _Board) -> dict[str, Any]:
    item = board.item
    extra = get_attributes(item).get("dimensions") or f"{_condition(item)} condition"
    args: dict[str, Any] = {}
    if extra.lower() not in item.title.lower():
        args["title"] = f"{item.title}, {extra}"[:100]
    if len(board.photo_ids) > 1:
        args["photo_order"] = list(reversed(board.photo_ids))
    return args


def _price_cut(board: _Board) -> int | None:
    goal = board.goal
    if board.firm or board.deal is not None:
        return None
    if board.last_price_change and board.now - board.last_price_change < PRICE_CHANGE_EVERY:
        return None
    cut = max(goal.floor_price_cents, _round_down(goal.list_price_cents * PRICE_CUT))
    return cut if cut < goal.list_price_cents else None


def _deadline_price(board: _Board) -> int | None:
    """The floor, once the deadline has passed and the price may still move."""
    floor = board.goal.floor_price_cents
    if board.now < board.goal.deadline or floor >= board.goal.list_price_cents:
        return None
    return floor if _price_cut(board) is not None else None


def _relist_platform(board: _Board) -> str | None:
    for listing in board.live:
        posted = listing.last_relisted_at or listing.published_at
        if posted is not None and board.now - posted >= RELIST_EVERY:
            return listing.platform
    return None


# ---------------------------------------------------------------- reading and writing text


def _intent(text: str, floor_cents: int) -> str:
    if _INJECTION.search(text):
        return "spam"
    if detect_scam(text) is not None or _PICKED_UP.search(text) or _WALK.search(text):
        return "other"
    price = _buyer_price(text)
    if price is not None:
        return "lowball" if price < floor_cents else "offer"
    if _AVAILABILITY.search(text) and not re.search(r"\d", text):
        return "availability"
    if _SCHEDULING.search(text):
        return "scheduling"
    return "question" if "?" in text else "other"


def _buyer_price(text: str) -> int | None:
    """The price a buyer named: their offer, else the last amount they stated."""
    offer = offer_cents(text)
    if offer is not None:
        return offer
    amounts = money_amounts_cents(text)
    return amounts[-1] if amounts else None


def _facts(board: _Board) -> dict[str, str]:
    attributes = get_attributes(board.item)
    facts = {
        "condition": f"It's in {_condition(board.item)} condition.",
        "home": "It's from a smoke-free, pet-free home.",
        "works": "Everything works as it should.",
        "available": "Yes, it's still available.",
    }
    if "dimensions" in attributes:
        facts["dimensions"] = f"It's {attributes['dimensions']}."
    if "age" in attributes:
        facts["age"] = f"It's {attributes['age']}."
    return facts


def _bar(goal: Goal, start: datetime, now: datetime) -> int:
    """Lowest price worth accepting now for one item: halfway from floor to list at start,
    falling linearly to the floor at the deadline; the list price when firm."""
    if goal.firmness == Firmness.FIRM:
        return goal.list_price_cents
    span = (goal.deadline - start).total_seconds()
    elapsed = (now - start).total_seconds()
    left = 0.0 if span <= 0 else 1 - min(1.0, max(0.0, elapsed / span))
    give = goal.list_price_cents - goal.floor_price_cents
    return goal.floor_price_cents + round(give / 2 * left)


def _keywords(title: str) -> frozenset[str]:
    return frozenset(
        word.lower()
        for word in _WORDS.findall(title)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    )


def _names(words: set[str], keywords: frozenset[str]) -> bool:
    """The buyer named the item: at least half of its distinctive title words."""
    return bool(keywords) and len(words & keywords) >= max(1, (len(keywords) + 1) // 2)


def _plain(title: str) -> str:
    """A title fit for seller text: before its first comma, without number-like words, which the
    guard would read as prices."""
    head = title.split(",")[0]
    return " ".join(_DIGITS.sub(" ", head).split()) or "item"


_LOTS = {2: "both", 3: "all three", 4: "all four", 5: "all five"}


def _lot(item_ids: Sequence[int]) -> str:
    """How a message names a bundle's items together, in words (a digit would read as a price);
    empty for one item."""
    return "" if len(item_ids) < 2 else _LOTS.get(len(item_ids), "all of them")


def _comps_query(item: Item) -> str:
    words = [w for w in _WORDS.findall(item.title) if w.lower() not in _STOPWORDS]
    return " ".join(words[:3])


def _condition(item: Item) -> str:
    return _CONDITIONS.get(item.condition or "", "good")


def _next_slot(
    settings: Settings, now: datetime, busy: list[tuple[datetime, datetime]]
) -> datetime | None:
    """Earliest start, on a half hour, at least SLOT_LEAD from now, whose whole SLOT_LENGTH fits
    in one availability window and overlaps no booked pickup."""
    zone = ZoneInfo(settings.timezone)
    earliest = now + SLOT_LEAD
    today = now.astimezone(zone).date()
    for offset in range(SLOT_SEARCH_DAYS):
        day = today + timedelta(days=offset)
        windows = sorted(
            (w for w in settings.availability if w.weekday == day.weekday()), key=lambda w: w.start
        )
        for window in windows:
            start = datetime.combine(day, window.start, tzinfo=zone)
            close = datetime.combine(day, window.end, tzinfo=zone)
            while start + SLOT_LENGTH <= close:
                begin, end = start.astimezone(UTC), (start + SLOT_LENGTH).astimezone(UTC)
                if begin >= earliest and not any(begin < b and a < end for a, b in busy):
                    return begin
                start += SLOT_STEP
    return None


def _step_cents(cents: float) -> int:
    return 500 if cents >= 5_000 else 100


def _round_up(cents: float) -> int:
    step = _step_cents(cents)
    return -(-int(cents) // step) * step


def _round_down(cents: float) -> int:
    step = _step_cents(cents)
    return int(cents) // step * step


def _dollars(cents: int) -> str:
    return f"${cents // 100:,}" if cents % 100 == 0 else f"${cents // 100:,}.{cents % 100:02d}"
