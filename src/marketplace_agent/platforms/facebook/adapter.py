"""FacebookAdapter: PlatformAdapter over a browser that uses facebook.com as the owner.

Slow on purpose. One browser action at a time behind a lock, a human pause before each one
(PacedDriver), at most THREADS_PER_POLL chats opened per inbox read, the inbox read at most
every INBOX_INTERVAL and Your listings every STATS_TTL. Posting is gated by the core's trust
ladder: facebook publish, relist, rewrite_listing, and update_price start at ASK.

| Method | Flow (flow_notes has the URLs) |
|---|---|
| publish | create form; assist fills it, leaves the tab open, and raises AwaitingHuman |
| update | edit form: title, price, description; photos are not changed |
| delist | listing page: Mark as sold, else More options, Delete listing |
| fetch_threads | Messenger's Marketplace folder: new, unread, or changed chats |
| send_message | chat page: type, send, read the message back for its id |
| stats | Your listings: clicks on listing as views |
| search | [] (comps come from Kijiji; scraping Facebook search adds ban risk) |
| healthcheck | signed in, Messenger inbox loads |

Messenger shows no message ids, so a message id hashes the thread, the sender side, the text,
and how many identical messages came before it. Messenger renders only the latest messages of a
long chat, so the adapter scrolls up until no older ones load before it counts. Times come from
the bubble labels when they parse, else the time the message was first read.
"""

import hashlib
import logging
import re
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.platforms.base import (
    AwaitingHuman,
    ListingGone,
    ListingPayload,
    ListingStats,
    PlatformAuthError,
    PlatformError,
    PlatformRef,
    RemoteMessage,
    RemoteThread,
    SearchResult,
)
from marketplace_agent.platforms.facebook import flow_notes as fb
from marketplace_agent.platforms.facebook.driver import BrowserDriver, DriverError
from marketplace_agent.platforms.facebook.healer import SelectorHealer
from marketplace_agent.platforms.facebook.selectors import (
    FIND_TIMEOUT_S,
    OVERRIDES_FILE,
    Finder,
    SelectorRegistry,
)

INBOX_INTERVAL = timedelta(minutes=5)
STATS_TTL = timedelta(hours=1)
THREADS_PER_POLL = 8
PROMPT_WAIT_S = 3.0
SUBMIT_POLLS = 30  # one second apart
FOLDER_SETTLE_S = 3.0
HISTORY_PASSES = 20  # scrolls to the oldest message, each loading an older batch
HISTORY_SETTLE_S = 1.5
TITLE_SHOWN_MIN = 12  # characters of a title cut short on a card, to count as that title
# The adapter closes its browser before raising these, so the login can open the profile
SIGNED_OUT = (
    "Facebook is signed out or wants you to confirm it is you: run `agent facebook login`. "
    "The agent closed its browser, so this works while `agent serve` keeps running."
)
PIN_PROMPT = (
    "Messenger is asking for your chat history PIN: run `agent facebook login` and enter it "
    "in the browser window. The agent closed its browser, so this works while `agent serve` "
    "keeps running."
)
BLOCKED = (
    "Facebook says this account cannot use Marketplace right now. Open "
    "facebook.com/marketplace in the browser to see why."
)
# Notices that replace or cover a page, checked before its content, in this order
NOTICES = ("login.form", "page.blocked", "page.pin_prompt")
DOCTOR_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        fb.CREATE_URL,
        (
            "create.photos",
            "create.title",
            "create.price",
            "create.category",
            "create.condition",
            "create.description",
            "create.next",
        ),
    ),
    (fb.INBOX_URL, ("inbox.thread_list", "inbox.marketplace_folder")),
)
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_CLOCK = r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<half>[ap]m)?"

logger = logging.getLogger("marketplace_agent.facebook")


@dataclass(frozen=True)
class _Card:
    text: str
    stats: ListingStats
    sold: bool


@dataclass(frozen=True)
class _Row:
    thread_id: str
    key: str  # row text without the moving "2m" age: changes when a message arrives
    unread: bool


@dataclass(frozen=True)
class Bubble:
    when: str
    sender: str
    text: str


class FacebookAdapter:
    name = "facebook"
    searches = False  # search() is a stub: scraping Facebook search risks the account

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        driver: BrowserDriver,
        router: ProviderRouter | None,
        *,
        registry: SelectorRegistry | None = None,
        find_timeout_s: float = FIND_TIMEOUT_S,
    ):
        self.settings = settings
        self.clock = clock
        self.driver = driver
        self.registry = registry or SelectorRegistry.load(settings.data_dir / OVERRIDES_FILE)
        healer = (
            SelectorHealer(driver, self.registry, router, settings, clock)
            if router is not None
            else None
        )
        self.finder = Finder(
            driver,
            self.registry,
            healer,
            timeout_s=find_timeout_s,
            before_heal=self._check_notices,
        )
        self._zone = ZoneInfo(settings.timezone)
        self._lock = threading.RLock()
        self._titles: dict[str, str] = {}  # listing id -> title we posted
        self._cards: dict[str, _Card] = {}
        self._cards_at: datetime | None = None
        self._stats: dict[str, tuple[datetime, ListingStats]] = {}
        self._rows: dict[str, str] = {}  # thread id -> row key when last read
        self._polled_at: datetime | None = None

    # ------------------------------------------------------------ PlatformAdapter

    def publish(self, listing: ListingPayload, *, assist: bool = False) -> PlatformRef:
        with self._lock:
            if assist and self.settings.facebook_headless:
                raise PlatformError(
                    "Facebook assist mode needs a visible browser: set facebook_headless to false."
                )
            if not listing.photo_paths:
                raise PlatformError("Facebook needs at least one photo.")
            condition, exact = condition_option(listing.condition)
            before = set() if assist else set(self._read_cards())
            self._open(fb.CREATE_URL, ["create.photos"])
            photos = listing.photo_paths[: fb.MAX_PHOTOS]
            self.driver.upload(self.finder.find("create.photos"), photos)
            self._fill_text(listing)
            self._choose_category(listing)
            self.driver.select_option(self.finder.find("create.condition"), condition, exact=exact)
            if assist:
                self.driver.release_page()
                raise AwaitingHuman(
                    "Facebook: the listing form is filled in the open browser window. Check it, "
                    "then click Next and Publish."
                )
            self._submit_new()
            ref = self._new_ref(listing.title, before)
            self._titles[ref.remote_id] = listing.title
            return ref

    def update(self, ref: PlatformRef, listing: ListingPayload) -> None:
        with self._lock:
            url = fb.EDIT_URL.format(listing_id=ref.remote_id)
            if self._open(url, ["item.gone", "create.title"]) == "item.gone":
                raise ListingGone(f"Facebook listing {ref.remote_id} no longer exists.")
            self._fill_text(listing)
            self._click_enabled("edit.save")
            self._await_leave(
                "/marketplace/edit", "Facebook did not confirm the changes", button="edit.save"
            )
            self._titles[ref.remote_id] = listing.title
            self._stats.pop(ref.remote_id, None)

    def delist(self, ref: PlatformRef) -> None:
        """Mark as sold; delete when that is not offered or does not take. Already sold: done."""
        with self._lock:
            state = self._open_listing(ref.remote_id)
            if state == "item.mark_available":
                return
            if state == "item.mark_sold" and self._mark_sold():
                return
            self._delete()

    def fetch_threads(self, since: datetime | None) -> list[RemoteThread]:
        """Chats to read: with since None (a fresh start) the newest THREADS_PER_POLL, else
        unread ones and ones whose last line changed. Returns [] when polled again within
        INBOX_INTERVAL. Each chat comes back whole; ingest skips messages it already has. A
        chat that fails to load is skipped and tried again next time."""
        with self._lock:
            now = self.clock.now()
            if since is not None and self._polled_at and now - self._polled_at < INBOX_INTERVAL:
                return []
            self._polled_at = now
            rows = self._chat_rows()
            threads = []
            for row in self._due(rows, fresh_start=since is None or not self._rows):
                try:
                    thread = self._read_thread(row.thread_id)
                except PlatformAuthError:
                    raise
                except PlatformError as exc:
                    logger.warning("facebook chat %s skipped: %s", row.thread_id, exc)
                    continue
                self._rows[row.thread_id] = row.key
                if thread is not None:
                    threads.append(thread)
            return threads

    def send_message(self, thread_remote_id: str, body: str) -> str | None:
        with self._lock:
            self._open(fb.THREAD_URL.format(thread_id=thread_remote_id), ["thread.message"])
            self._load_history()  # so the id read back counts from the chat's first message
            self.driver.fill(self.finder.find("thread.input"), body)
            send = self.finder.probe("thread.send")
            if send is not None:
                self.driver.click(send)
            else:
                self.driver.press("Enter")
            if self.finder.probe("thread.rate_limited", PROMPT_WAIT_S) is not None:
                raise PlatformError(
                    "Facebook: Messenger's sending limit was reached; the message was not sent."
                )
            return self._sent_id(thread_remote_id, body)

    def stats(self, ref: PlatformRef) -> ListingStats:
        """From the Your listings card, read at most every STATS_TTL. A listing without a card
        is looked up on its own page: gone raises ListingGone, else zeros."""
        with self._lock:
            now = self.clock.now()
            cached = self._stats.get(ref.remote_id)
            if cached is not None and now - cached[0] < STATS_TTL:
                return cached[1]
            if self._cards_at is None or now - self._cards_at >= STATS_TTL:
                self._read_cards()
            card = self._cards.get(ref.remote_id)
            if card is not None and card.sold:
                raise ListingGone(f"Facebook listing {ref.remote_id} is marked as sold.")
            if card is None and self._open_listing(ref.remote_id) == "item.mark_available":
                raise ListingGone(f"Facebook listing {ref.remote_id} is marked as sold.")
            stats = card.stats if card is not None else ListingStats()
            self._stats[ref.remote_id] = (now, stats)
            return stats

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        return []

    def healthcheck(self) -> list[str]:
        with self._lock:
            try:
                if self._open(fb.INBOX_URL, ["inbox.thread_list"]) is None:
                    return ["facebook: the Messenger inbox did not load. Run `agent doctor`."]
                self._check_notices()  # one drawn over the chats just after them
            except PlatformError as exc:
                return [f"facebook: {exc}"]
            return []

    # ------------------------------------------------------------ doctor and lifecycle

    def validate_selectors(self) -> list[str]:
        """Read only: opens the create form and the inbox, submits nothing, never heals.
        Returns one line per target that matches nothing."""
        with self._lock:
            problems = []
            try:
                for url, names in DOCTOR_CHECKS:
                    self._open(url, names)
                    path = urlsplit(url).path
                    for name in names:
                        if self.finder.probe(name, self.finder.timeout_s) is None:
                            purpose = self.registry.target(name).purpose
                            problems.append(f"{name} ({purpose}) matches nothing on {path}")
            except PlatformError as exc:
                problems.append(str(exc))
            return problems

    def close(self) -> None:
        self.driver.close()

    # ------------------------------------------------------------ session

    def _open(self, url: str, ready: Sequence[str]) -> str | None:
        """Go to url and wait for it to render: the first of ready to show, or None after the
        wait. Facebook draws pages after load, so a notice (signed out, blocked, PIN) is waited
        for along with the content, and raises."""
        self.driver.goto(url)
        if not signed_in(self.driver):
            raise self._needs_owner(SIGNED_OUT)
        shown = self.finder.first_of([*NOTICES, *ready], self.finder.timeout_s)
        self._raise_for(shown)
        return shown

    def _check_notices(self) -> None:
        """Raise for a notice on the page now. Before healing too: a selector is not broken
        when a notice hides the page."""
        self._raise_for(next((n for n in NOTICES if self.finder.probe(n) is not None), None))

    def _raise_for(self, shown: str | None) -> None:
        if shown == "login.form":
            raise self._needs_owner(SIGNED_OUT)
        if shown == "page.pin_prompt":
            raise self._needs_owner(PIN_PROMPT)
        if shown == "page.blocked":
            raise PlatformError(BLOCKED)

    def _needs_owner(self, problem: str) -> PlatformAuthError:
        """Close the browser, which frees the profile for `agent facebook login` while this
        process runs on; the next call opens it again."""
        try:
            self.driver.close()
        except PlatformError as exc:
            logger.warning("facebook: could not close the browser: %s", exc)
        return PlatformAuthError(problem)

    # ------------------------------------------------------------ listings

    def _fill_text(self, listing: ListingPayload) -> None:
        self.driver.fill(self.finder.find("create.title"), listing.title)
        self.driver.fill(self.finder.find("create.price"), str(whole_dollars(listing.price_cents)))
        self.driver.fill(self.finder.find("create.description"), listing.description)

    def _choose_category(self, listing: ListingPayload) -> None:
        dropdown = self.finder.find("create.category")
        option = category_option(listing)
        try:
            self.driver.select_option(dropdown, option)
        except PlatformError:
            if option == fb.FALLBACK_CATEGORY:
                raise
            self.driver.select_option(self.finder.find("create.category"), fb.FALLBACK_CATEGORY)

    def _submit_new(self) -> None:
        first = self.finder.first_of(["create.next", "create.publish"], self.finder.timeout_s)
        if first != "create.publish":
            self._click_enabled("create.next")
        self._click_enabled("create.publish")
        later = self.finder.probe("dialog.not_now", PROMPT_WAIT_S)
        if later is not None:
            self.driver.click(later)
        self._await_leave("/marketplace/create", "Facebook did not confirm the new listing")

    def _click_enabled(self, name: str) -> None:
        button = self.finder.find(name)
        if self.driver.attribute(button, "aria-disabled") == "true":
            raise PlatformError(
                f"Facebook kept {name} disabled: a required field is missing or was rejected. "
                "Check the form in the browser."
            )
        self.driver.click(button)

    def _await_leave(self, path_part: str, failure: str, *, button: str | None = None) -> None:
        """Wait until the page leaves path_part or, when given, button disappears."""
        for _ in range(SUBMIT_POLLS):
            if path_part not in urlsplit(self.driver.current_url()).path:
                return
            if button is not None and self.finder.probe(button) is None:
                return
            self.driver.wait(1.0)
        raise PlatformError(f"{failure}. Look at the browser before trying again.")

    def _new_ref(self, title: str, before: set[str]) -> PlatformRef:
        """The card that appeared since publishing (its id not in before) with this title as a
        line of its own, preferring cards still for sale; page order (newest first) breaks
        ties. An older card, such as "IKEA Malm desk chair" or a sold one with the same title,
        is never taken for the new listing."""
        cards = self._read_cards()
        matches = [
            lid for lid, card in cards.items() if lid not in before and titled(card.text, title)
        ]
        if not matches:
            raise PlatformError(
                "Facebook: Publish was clicked, but the listing is not under Your listings. "
                f"Check {fb.SELLING_URL} before publishing again."
            )
        best = min(matches, key=lambda lid: cards[lid].sold)
        return PlatformRef(self.name, best, fb.ITEM_URL.format(listing_id=best))

    def _read_cards(self) -> dict[str, _Card]:
        self._open(fb.SELLING_URL, ["selling.card"])
        cards: dict[str, _Card] = {}
        for card in self.finder.find_all("selling.card"):
            link = self.finder.probe("selling.item_link", within=card)
            href = self.driver.attribute(link, "href") if link is not None else None
            match = re.search(fb.LISTING_ID_PATTERN, href or "")
            if match is not None and match.group(1) not in cards:
                cards[match.group(1)] = _parse_card(self.driver.text_of(card))
        self._cards, self._cards_at = cards, self.clock.now()
        return cards

    def _open_listing(self, listing_id: str) -> str | None:
        """Opens the listing page; which control it shows. Raises ListingGone when it is gone."""
        url = fb.ITEM_URL.format(listing_id=listing_id)
        state = self._open(url, ["item.gone", "item.mark_sold", "item.mark_available"])
        if state == "item.gone":
            self._titles.pop(listing_id, None)
            raise ListingGone(f"Facebook listing {listing_id} no longer exists.")
        return state

    def _mark_sold(self) -> bool:
        """True once the page offers Mark as available (or no longer offers Mark as sold)."""
        self.driver.click(self.finder.find("item.mark_sold"))
        for _ in range(3):
            step = self.finder.probe("sold.confirm", PROMPT_WAIT_S)
            if step is None:
                break
            self.driver.click(step)
        if self.finder.probe("item.mark_available", PROMPT_WAIT_S) is not None:
            return True
        return self.finder.probe("item.mark_sold") is None

    def _delete(self) -> None:
        self.driver.click(self.finder.find("item.more"))
        self.driver.click(self.finder.find("item.delete"))
        self.driver.click(self.finder.find("delete.confirm"))

    # ------------------------------------------------------------ messages

    def _chat_rows(self) -> list[_Row]:
        self._open(fb.INBOX_URL, ["inbox.thread_list"])
        self.finder.find("inbox.thread_list")
        folder = self.finder.find("inbox.marketplace_folder")
        try:
            self.driver.click(folder)
        except DriverError:
            self._check_notices()  # a PIN box drawn over the chats takes the click
            raise
        self.driver.wait(FOLDER_SETTLE_S)
        rows: list[_Row] = []
        for link in self.finder.find_all("inbox.thread_link"):
            match = re.search(fb.THREAD_ID_PATTERN, self.driver.attribute(link, "href") or "")
            text = self.driver.text_of(link)
            if match is None or any(r.thread_id == match.group(1) for r in rows):
                continue
            if fb.TITLE_SEPARATOR not in text.strip().split("\n")[0]:
                continue  # not a Marketplace chat ("Buyer · Listing"): never opened
            unread = self.finder.probe("inbox.unread", within=link) is not None
            rows.append(_Row(match.group(1), row_key(text), unread))
        return rows

    def _due(self, rows: list[_Row], *, fresh_start: bool) -> list[_Row]:
        """On a fresh start the newest rows, plus unread ones; the rest are taken as read."""
        if fresh_start:
            due = [*rows[:THREADS_PER_POLL], *(r for r in rows[THREADS_PER_POLL:] if r.unread)]
            for row in rows:
                if row not in due:
                    self._rows[row.thread_id] = row.key
        else:
            due = [r for r in rows if r.unread or self._rows.get(r.thread_id) != r.key]
        return due[:THREADS_PER_POLL]

    def _read_thread(self, thread_id: str) -> RemoteThread | None:
        self._open(fb.THREAD_URL.format(thread_id=thread_id), ["thread.message"])
        self._load_history()
        bubbles = self._bubbles()
        if not bubbles:
            logger.warning("facebook chat %s: no messages found; skipped", thread_id)
            return None
        heading = self.finder.probe("thread.heading")
        buyer, listing_title = split_heading(self.driver.text_of(heading) if heading else "")
        buyer = buyer or next((b.sender for b in bubbles if b.sender != fb.OWN_SENDER), "")
        return RemoteThread(
            remote_id=thread_id,
            listing_remote_id=self._listing_for(listing_title),
            buyer_handle=self._buyer_handle(thread_id),
            buyer_name=buyer,
            messages=self._messages(thread_id, bubbles),
        )

    def _load_history(self) -> None:
        """Scroll to the oldest message until no older ones load. Counting from the chat's
        first message keeps ids stable: counted from the first one rendered, a repeated "ok"
        would get the id of an earlier "ok" that scrolled out of view, and be dropped."""
        count = -1
        for _ in range(HISTORY_PASSES):
            bubbles = self.finder.find_all("thread.message")
            if not bubbles or len(bubbles) == count:
                return
            count = len(bubbles)
            try:
                self.driver.scroll_into_view(bubbles[0])
            except DriverError:
                count = -1  # the bubble re-rendered mid-scroll: count again
            self.driver.wait(HISTORY_SETTLE_S)
        logger.warning("facebook chat: older messages still loading after %d passes", count)

    def _bubbles(self) -> list[Bubble]:
        found = self.finder.find_all("thread.message")
        labels = [self.driver.attribute(bubble, "aria-label") for bubble in found]
        return parse_bubbles([label for label in labels if label])

    def _messages(self, thread_id: str, bubbles: Sequence[Bubble]) -> list[RemoteMessage]:
        now = self.clock.now()
        seen: Counter[tuple[bool, str]] = Counter()
        messages = []
        for bubble in bubbles:
            from_buyer = bubble.sender != fb.OWN_SENDER
            seen[from_buyer, bubble.text] += 1
            occurrence = seen[from_buyer, bubble.text]
            messages.append(
                RemoteMessage(
                    remote_id=message_id(thread_id, from_buyer, bubble.text, occurrence),
                    body=bubble.text,
                    sent_at=parse_sent(bubble.when, now, self._zone) or now,
                    from_buyer=from_buyer,
                )
            )
        return messages

    def _listing_for(self, listing_title: str) -> str | None:
        link = self.finder.probe("thread.listing_link")
        href = self.driver.attribute(link, "href") if link is not None else None
        match = re.search(fb.LISTING_ID_PATTERN, href or "")
        if match is not None:
            return match.group(1)
        wanted = _flat(listing_title).lower()
        if not wanted:
            return None
        by_title = [lid for lid, title in self._titles.items() if _flat(title).lower() == wanted]
        by_card = [lid for lid, card in self._cards.items() if wanted in _flat(card.text).lower()]
        candidates = by_title or by_card
        return candidates[0] if len(candidates) == 1 else None

    def _buyer_handle(self, thread_id: str) -> str:
        link = self.finder.probe("thread.buyer_link")
        href = (self.driver.attribute(link, "href") if link is not None else None) or ""
        for pattern in fb.PROFILE_ID_PATTERNS:
            match = re.search(pattern, href)
            if match is not None:
                return match.group(1)
        return f"thread-{thread_id}"

    def _sent_id(self, thread_id: str, body: str) -> str | None:
        """The id fetch_threads will give the message just sent; None if it does not show."""
        text = _flat(body)
        for _ in range(3):
            mine = [m for m in self._messages(thread_id, self._bubbles()) if not m.from_buyer]
            sent = [m for m in mine if m.body == text]
            if sent:
                return sent[-1].remote_id
            self.driver.wait(1.0)
        logger.warning("facebook chat %s: sent message not visible yet", thread_id)
        return None


def signed_in(driver: BrowserDriver) -> bool:
    """Facebook's session cookie is set and the page is not a login or checkpoint page."""
    path = urlsplit(driver.current_url()).path
    return driver.has_cookie(fb.SESSION_COOKIE) and not path.startswith(fb.LOGIN_PATHS)


# ---------------------------------------------------------------- form values


def whole_dollars(cents: int) -> int:
    """Facebook takes whole dollars. Rounds up, so a price at the floor never posts below it."""
    return -(-cents // 100)


def condition_option(condition: str | None) -> tuple[str, bool]:
    """(option text, exact) for the Condition dropdown."""
    option = fb.CONDITION_OPTIONS.get(condition or "")
    if option is None:
        known = ", ".join(fb.CONDITION_OPTIONS)
        raise PlatformError(
            f"Facebook requires a condition, and the item has {condition or 'none'}. Set the "
            f"item's condition to one of: {known}."
        )
    return option


def category_option(listing: ListingPayload) -> str:
    """The category option for the item's category words, then its title words."""
    for source in (listing.category or "", listing.title):
        words = f" {' '.join(re.findall(r'[a-z]+', source.lower()))} "
        for keywords, option in fb.CATEGORY_KEYWORDS:
            if any(f" {keyword} " in words for keyword in keywords):
                return option
    return fb.FALLBACK_CATEGORY


# ---------------------------------------------------------------- page text


def parse_bubbles(labels: Sequence[str]) -> list[Bubble]:
    """Bubble aria-labels ("Enter, Message sent 4:01 PM by Lucas: text") in page order;
    repeated labels are one message rendered twice."""
    bubbles: list[Bubble] = []
    for label in dict.fromkeys(labels):
        match = re.search(fb.BUBBLE_PATTERN, label, re.DOTALL)
        if match is not None and _flat(match["text"]):
            when, sender, text = _flat(match["when"]), match["sender"].strip(), _flat(match["text"])
            bubbles.append(Bubble(when, sender, text))
    return bubbles


def message_id(thread_id: str, from_buyer: bool, text: str, occurrence: int) -> str:
    digest = hashlib.sha1(f"{from_buyer}|{text}|{occurrence}".encode()).hexdigest()[:16]
    return f"{thread_id}-{digest}"


def titled(card_text: str, title: str) -> bool:
    """A line of a Your listings card is the title: the same text, or cut short with an
    ellipsis."""
    wanted = _flat(title).casefold()
    for line in card_text.splitlines():
        shown = _flat(line).casefold()
        if shown == wanted:
            return True
        cut = shown.removesuffix("…").removesuffix("...").rstrip()
        if cut != shown and len(cut) >= TITLE_SHOWN_MIN and wanted.startswith(cut):
            return True
    return False


def split_heading(text: str) -> tuple[str, str]:
    """("Lucas", "IKEA desk") from "Conversation titled Lucas · IKEA desk"."""
    text = _flat(text).removeprefix(fb.HEADING_PREFIX)
    buyer, _, listing_title = text.partition(fb.TITLE_SEPARATOR)
    return buyer.strip(), listing_title.strip()


def row_key(text: str) -> str:
    """Row text without its age ("· 2m", "· Mon") and unread markers."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line for line in lines if "unread" not in line.lower()]
    if lines:
        lines[-1] = re.sub(r"\s*·\s*[^·]{1,12}$", "", lines[-1])
    return "\n".join(lines)


def parse_sent(label: str, now: datetime, zone: ZoneInfo) -> datetime | None:
    """Messenger's bubble time ("4:01 PM", "Yesterday 4:01 PM", "Mon 4:01 PM",
    "Oct 3, 2026, 4:01 PM") as aware UTC, never after now; None when it does not parse."""
    text = _flat(label).lower().replace(" at ", " ")
    today = now.astimezone(zone).date()
    clock = re.search(_CLOCK + "$", text)
    if clock is None:
        return None
    moment = _clock_time(clock)
    if moment is None:
        return None
    day = _day(text[: clock.start()].strip(" ,"), today)
    if day is None:
        return None
    return min(datetime.combine(day, moment, tzinfo=zone).astimezone(now.tzinfo), now)


def _clock_time(match: re.Match[str]) -> time | None:
    hour, minute = int(match["hour"]), int(match["minute"])
    if match["half"]:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if match["half"] == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def _day(text: str, today: date) -> date | None:
    if not text or text == "today":
        return today
    if text == "yesterday":
        return today - timedelta(days=1)
    if text[:3] in _WEEKDAYS and len(text.split()) == 1:
        back = (today.weekday() - _WEEKDAYS.index(text[:3])) % 7 or 7
        return today - timedelta(days=back)
    match = re.fullmatch(r"([a-z]{3})[a-z]*\.? (\d{1,2}),?(?: (\d{4}))?", text)
    if match is None or match[1] not in _MONTHS:
        return None
    try:
        day = date(int(match[3] or today.year), _MONTHS.index(match[1]) + 1, int(match[2]))
    except ValueError:
        return None
    if not match[3] and day > today:
        day = day.replace(year=day.year - 1)  # "Dec 30" read in January
    return day


def _parse_card(text: str) -> _Card:
    clicks = re.search(fb.CLICKS_PATTERN, text, re.IGNORECASE)
    saves = re.search(fb.SAVES_PATTERN, text, re.IGNORECASE)
    stats = ListingStats(
        views=int(clicks[1].replace(",", "")) if clicks else 0,
        saves=int(saves[1].replace(",", "")) if saves else 0,
    )
    return _Card(text, stats, re.search(fb.SOLD_CARD_PATTERN, text) is not None)


def _flat(text: str) -> str:
    """Whitespace collapsed; split() also breaks on the narrow no-break space in "4:01 PM"."""
    return " ".join(text.split())
