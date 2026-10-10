"""A fake facebook.com for adapter tests: a small state machine behind the BrowserDriver protocol.

Pages are rendered from FakeSite state on every lookup, so a handle from an earlier lookup still
works after the page changes (its callbacks close over the site). Elements answer the spec kinds
the way Playwright does: role + name, label, text, placeholder (substring, any case, unless
exact) and css (the literal selector string must be in the element's css set). Specs listed in
FakeDriver.broken match nothing, which is how tests simulate Facebook renaming things.
"""

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from marketplace_agent.platforms.facebook import flow_notes as fb
from marketplace_agent.platforms.facebook.driver import DriverError, LocatorSpec
from marketplace_agent.platforms.facebook.selectors import SelectorRegistry

CARD_CSS = SelectorRegistry.load().specs("selling.card")[0]["css"]
CATEGORIES = (
    "Tools",
    "Furniture",
    "Household",
    "Garden",
    "Appliances",
    "Video Games",
    "Books, Movies & Music",
    "Electronics & Computers",
    "Mobile Phones",
    "Bicycles",
    "Toys & Games",
    "Miscellaneous",
)
CONDITIONS = ("New", "Used - Like New", "Used - Good", "Used - Fair")
GONE_TEXT = "This content isn't available right now"
PIN_TEXT = "Enter your PIN to restore your chats"


@dataclass
class El:
    key: str = ""
    role: str | None = None
    name: str = ""
    label: str | None = None
    text: str = ""
    placeholder: str | None = None
    css: frozenset[str] = frozenset()
    attrs: dict[str, str] = field(default_factory=dict)
    visible: bool = True
    children: list["El"] = field(default_factory=list)
    on_click: Callable[[], None] | None = None
    options: tuple[str, ...] = ()

    def all_text(self) -> str:
        return "\n".join(t for t in [self.text, *(c.all_text() for c in self.children)] if t)


def _match(value: str | None, wanted: str, exact: bool) -> bool:
    if value is None:
        return False
    return value == wanted if exact else wanted.lower() in value.lower()


def _matches(el: El, spec: LocatorSpec) -> bool:
    exact = spec.get("exact", False)
    if "role" in spec:
        return el.role == spec["role"] and (
            "name" not in spec or _match(el.name, spec["name"], exact)
        )
    if "label" in spec:
        return _match(el.label, spec["label"], exact)
    if "text" in spec:
        return bool(el.text) and _match(el.text, spec["text"], exact)
    if "placeholder" in spec:
        return _match(el.placeholder, spec["placeholder"], exact)
    return spec["css"] in el.css


@dataclass
class FakeListing:
    listing_id: str
    title: str
    price: str
    description: str
    category: str = ""
    condition: str = ""
    photos: list[Path] = field(default_factory=list)
    status: str = "active"  # active | sold | deleted
    clicks: int = 0


@dataclass
class FakeChat:
    chat_id: str
    buyer: str
    listing_title: str
    messages: list[tuple[str, str, str]]  # (time label, sender, text); sender "You" = seller
    listing_id: str | None = None
    profile_id: str | None = None
    unread: bool = True
    marketplace: bool = True
    window: int | None = None  # render only the latest this many, more per scroll to the top


@dataclass
class FakeSite:
    logged_in: bool = True
    blocked: bool = False
    pin_prompt: bool = False  # the PIN box replaces the chats
    pin_overlay: bool = False  # the PIN box covers the chats and takes every click
    rate_limited: bool = False
    boost_prompt: bool = False
    sold_dialog: bool = False
    offers_mark_sold: bool = True
    listings: dict[str, FakeListing] = field(default_factory=dict)
    chats: dict[str, FakeChat] = field(default_factory=dict)
    ids: itertools.count = field(default_factory=lambda: itertools.count(9001))

    def add_listing(self, title: str, price: str = "100", **kwargs: object) -> FakeListing:
        listing_id = str(next(self.ids))
        listing = FakeListing(listing_id, title, price, "", **kwargs)  # type: ignore[arg-type]
        self.listings[listing.listing_id] = listing
        return listing

    def add_chat(
        self, buyer: str, listing_title: str, *messages: tuple[str, str], **kw: object
    ) -> FakeChat:
        chat_id = str(next(self.ids))
        lines = [("4:01 PM", sender, text) for sender, text in messages]
        chat = FakeChat(chat_id, buyer, listing_title, lines, **kw)  # type: ignore[arg-type]
        self.chats[chat_id] = chat
        return chat


class FakeDriver:
    """BrowserDriver over a FakeSite. actions records every input in order."""

    def __init__(self, site: FakeSite | None = None, *, load_waits: int = 0):
        self.site = site or FakeSite()
        self.url = "about:blank"
        self.actions: list[tuple[str, ...]] = []
        self.broken: set[str] = set()
        self.extra: list[El] = []  # elements added to every page (healing targets, duplicates)
        self.load_waits = load_waits
        self.closed = False
        self.released: list[str] = []
        self.on_wait: Callable[[], None] | None = None
        self.snapshot_extra = ""
        self._loading = 0
        self._form: dict[str, object] = {}
        self._stage = "form"
        self._dialog: str | None = None
        self._menu = False
        self._folder_open = False
        self._history = 0  # older batches loaded by scrolling up in the open chat

    # helpers for tests

    def break_spec(self, spec: LocatorSpec) -> None:
        self.broken.add(json.dumps(spec, sort_keys=True))

    def break_target(self, registry: SelectorRegistry, name: str) -> None:
        for spec in registry.specs(name):
            self.break_spec(spec)

    def kinds(self) -> list[str]:
        return [action[0] for action in self.actions]

    # BrowserDriver

    def goto(self, url: str) -> None:
        self.actions.append(("goto", url))
        self._loading = self.load_waits
        self._dialog, self._menu = None, False
        self._history = 0
        if not self.site.logged_in:
            self.url = f"{fb.LOGIN_URL}/?next={url}"
            return
        self.url = url
        path = urlsplit(url).path
        if path.startswith("/marketplace/create"):
            self._form, self._stage = {}, "form"
        elif path.startswith("/marketplace/edit"):
            listing = self._listing_from_query()
            self._form = (
                {"title": listing.title, "price": listing.price, "description": listing.description}
                if listing
                else {}
            )
        elif path == "/messages/t/":
            self._folder_open = False

    def find(self, spec: LocatorSpec, within: El | None = None) -> list[El]:
        if self._loading > 0 or json.dumps(spec, sort_keys=True) in self.broken:
            return []
        roots = within.children if within is not None else self._page()
        return [
            el for el in _walk(roots) if (el.visible or spec.get("attached")) and _matches(el, spec)
        ]

    def wait_for(self, specs: Sequence[LocatorSpec], timeout_s: float) -> bool:
        self.actions.append(("wait_for", str(timeout_s)))
        self._loading = 0
        return any(self.find(spec) for spec in specs)

    def wait(self, seconds: float) -> None:
        self.actions.append(("wait", str(seconds)))
        self._loading = max(0, self._loading - 1)
        if self.on_wait is not None:
            self.on_wait()

    def click(self, handle: El) -> None:
        self.actions.append(("click", handle.key))
        if self.site.pin_overlay and urlsplit(self.url).path.startswith("/messages/"):
            raise DriverError("Browser action failed: Locator.click: Timeout 30000ms exceeded.")
        if handle.attrs.get("aria-disabled") != "true" and handle.on_click is not None:
            handle.on_click()

    def scroll_into_view(self, handle: El) -> None:
        self.actions.append(("scroll", handle.key))
        bubbles = [el for el in self._page() if el.key == "bubble"]
        if bubbles and handle == bubbles[0]:
            self._history += 1

    def fill(self, handle: El, text: str) -> None:
        self.actions.append(("fill", handle.key, text))
        self._form[handle.key] = text

    def press(self, key: str) -> None:
        self.actions.append(("press", key))
        if key == "Enter" and self._chat() is not None:
            self._send()

    def upload(self, handle: El, paths: Sequence[Path]) -> None:
        self.actions.append(("upload", handle.key, str(len(paths))))
        self._form["photos"] = list(paths)

    def select_option(self, handle: El, option: str, *, exact: bool = False) -> None:
        self.actions.append(("select", handle.key, option))
        chosen = [o for o in handle.options if _match(o, option, exact)]
        if not chosen:
            raise DriverError(f"No option {option!r} in the list.")
        self._form[handle.key] = chosen[0]

    def text_of(self, handle: El) -> str:
        return handle.all_text()

    def attribute(self, handle: El, name: str) -> str | None:
        return handle.attrs.get(name)

    def screenshot(self) -> bytes:
        return b"\x89PNG fake"

    def aria_snapshot(self) -> str:
        lines = [
            f'- {el.role or "generic"} "{el.name or el.label or el.text}"'
            for el in _walk(self._page())
        ]
        return "\n".join([*lines, self.snapshot_extra])

    def current_url(self) -> str:
        return self.url

    def has_cookie(self, name: str) -> bool:
        return name == fb.SESSION_COOKIE and self.site.logged_in

    def release_page(self) -> None:
        self.actions.append(("release",))
        self.released.append(self.url)
        self.url = "about:blank"

    def close(self) -> None:
        self.closed = True

    # pages

    def _page(self) -> list[El]:
        path = urlsplit(self.url).path
        if path.startswith(("/login", "/checkpoint")):
            page = [
                El("email", css=frozenset({"input[name='email']"})),
                El("pass", css=frozenset({"input[name='pass']"})),
            ]
        elif path.startswith("/marketplace/create"):
            page = self._create_page()
        elif path.startswith("/marketplace/edit"):
            page = self._edit_page()
        elif path.startswith("/marketplace/item/"):
            page = self._item_page(path.split("/")[3])
        elif path == "/marketplace/you/selling":
            page = self._selling_page()
        elif path.startswith("/messages/") and self.site.pin_prompt:
            page = [El("pin", text=PIN_TEXT)]  # on every Messenger page, in place of the chats
        elif path == "/messages/t/":
            page = self._inbox_page()
        elif path.startswith("/messages/t/"):
            page = self._thread_page(path.split("/")[3])
        else:
            page = []
        if self.site.pin_overlay and path.startswith("/messages/"):
            page.append(El("pin", text=PIN_TEXT))  # drawn over the chats
        if self.site.blocked:
            page.append(El("blocked", text="You can't use Marketplace"))
        return [*page, *self.extra]

    def _create_page(self) -> list[El]:
        if self._dialog == "boost":
            return [_button("not_now", "Not now", self._to_selling)]
        if self._stage == "review":
            return [_button("publish", "Publish", self._publish)]
        ready = all(
            self._form.get(k) for k in ("photos", "title", "price", "category", "condition")
        )
        return [
            El(
                "photos",
                visible=False,
                css=frozenset({"input[type='file'][accept*='image']", "input[type='file']"}),
            ),
            *self._text_fields(),
            El(
                "category",
                role="combobox",
                name="Category",
                label="Category",
                css=frozenset({"[aria-label='Category']"}),
                options=CATEGORIES,
            ),
            El(
                "condition",
                role="combobox",
                name="Condition",
                label="Condition",
                css=frozenset({"[aria-label='Condition']"}),
                options=CONDITIONS,
            ),
            _button("next", "Next", self._review, disabled=not ready),
        ]

    def _text_fields(self) -> list[El]:
        return [
            El(
                "title",
                role="textbox",
                name="Title",
                label="Title",
                css=frozenset({"input[aria-label='Title']", "#title-v2"}),
                placeholder="What are you selling?",
            ),
            El(
                "price",
                role="textbox",
                name="Price",
                label="Price",
                css=frozenset({"input[aria-label='Price']"}),
            ),
            El(
                "description",
                role="textbox",
                name="Description",
                label="Description",
                css=frozenset({"textarea[aria-label='Description']"}),
                placeholder="Describe your item",
            ),
        ]

    def _edit_page(self) -> list[El]:
        listing = self._listing_from_query()
        if listing is None:
            return [El("gone", text=GONE_TEXT)]
        return [*self._text_fields(), _button("update", "Update", lambda: self._save(listing))]

    def _item_page(self, listing_id: str) -> list[El]:
        listing = self.site.listings.get(listing_id)
        if listing is None or listing.status == "deleted":
            return [El("gone", text=GONE_TEXT)]
        page = [El("title", role="heading", name=listing.title, text=listing.title)]
        if listing.status == "sold":
            page.append(_button("mark_available", "Mark as available", lambda: None))
        elif self.site.offers_mark_sold:
            page.append(_button("mark_sold", "Mark as sold", lambda: self._mark_sold(listing)))
        page.append(_button("more", "More options", self._open_menu))
        if self._menu:
            page.append(
                El(
                    "delete_item",
                    role="menuitem",
                    name="Delete listing",
                    text="Delete listing",
                    on_click=self._confirm_delete,
                )
            )
        if self._dialog == "delete":
            page.append(
                El(
                    "delete_confirm",
                    role="button",
                    name="Delete",
                    css=frozenset({"[role='dialog'] [role='button'][aria-label='Delete']"}),
                    on_click=lambda: self._delete(listing),
                )
            )
        if self._dialog == "sold":
            page.append(
                El(
                    "skip",
                    role="button",
                    name="Skip",
                    css=frozenset({"[role='dialog'] [role='button']:has-text('Skip')"}),
                    on_click=lambda: self._finish_sold(listing),
                )
            )
        return page

    def _selling_page(self) -> list[El]:
        cards = []
        for listing in reversed(self.site.listings.values()):  # newest first
            if listing.status == "deleted":
                continue
            state = "Active · Listed today" if listing.status == "active" else "Sold"
            button = "Mark as sold" if listing.status == "active" else "Mark as available"
            link = El(
                "link",
                css=frozenset({"a[href*='/marketplace/item/']"}),
                attrs={
                    "href": f"https://www.facebook.com/marketplace/item/{listing.listing_id}/?ref=selling"
                },
            )
            cards.append(
                El(
                    f"card-{listing.listing_id}",
                    css=frozenset({CARD_CSS}),
                    text=f"{listing.title}\n${listing.price}\n{state}\n{listing.clicks} clicks on listing\n{button}",
                    children=[link],
                )
            )
        return cards

    def _inbox_page(self) -> list[El]:
        grid = El(
            "grid", role="grid", name="Chats", css=frozenset({"[role='grid'][aria-label='Chats']"})
        )
        folder = El(
            "folder",
            role="button",
            name="Marketplace",
            css=frozenset(
                {
                    "[role='grid'][aria-label='Chats'] [role='row']:has-text('Marketplace') [role='button']"
                }
            ),
            on_click=self._open_folder,
        )
        rows = [
            self._row(chat)
            for chat in reversed(self.site.chats.values())
            if self._folder_open or not chat.marketplace
        ]
        return [grid, folder, *rows]

    def _row(self, chat: FakeChat) -> El:
        _, sender, text = chat.messages[-1]
        first = f"{chat.buyer} · {chat.listing_title}" if chat.marketplace else chat.buyer
        children = (
            [El("unread", visible=False, css=frozenset({"[aria-label*='nread']"}))]
            if chat.unread
            else []
        )
        return El(
            f"row-{chat.chat_id}",
            role="link",
            css=frozenset({"[role='grid'] a[href*='/messages/t/']", "a[href*='/messages/t/']"}),
            attrs={"href": f"https://www.facebook.com/messages/t/{chat.chat_id}/"},
            text=f"{first}\n{sender}: {text} · 2m",
            children=children,
        )

    def _thread_page(self, chat_id: str) -> list[El]:
        chat = self.site.chats.get(chat_id)
        if chat is None:
            return []
        chat.unread = False
        title = f"Conversation titled {chat.buyer} · {chat.listing_title}"
        page = [
            El(
                "heading",
                role="heading",
                name=title,
                text=title,
                css=frozenset({"h3:has-text('Conversation titled')"}),
            )
        ]
        if chat.listing_id:
            page.append(
                El(
                    "details",
                    visible=False,
                    css=frozenset({"a[aria-label='See details'][href*='/marketplace/item/']"}),
                    attrs={"href": f"/marketplace/item/{chat.listing_id}/"},
                )
            )
        if chat.profile_id:
            page.append(
                El(
                    "profile",
                    visible=False,
                    css=frozenset({"[role='main'] a[href*='/marketplace/profile/']"}),
                    attrs={"href": f"/marketplace/profile/{chat.profile_id}/"},
                )
            )
        shown = chat.messages
        if chat.window is not None:
            shown = shown[-chat.window * (self._history + 1) :]
        for when, sender, text in shown:
            label = f"Enter, Message sent {when} by {sender}: {text}"
            page.append(
                El(
                    "bubble",
                    visible=False,
                    attrs={"aria-label": label},
                    css=frozenset(
                        {
                            "[role='main'] [aria-label*='Message sent ']",
                            "[aria-label*='Message sent ']",
                        }
                    ),
                )
            )
        page.append(
            El(
                "input",
                role="textbox",
                name="Message",
                placeholder="Aa",
                css=frozenset({"[role='textbox'][contenteditable='true']"}),
            )
        )
        page.append(_button("send", "Press enter to send", self._send))
        if self._dialog == "limit":
            page.append(El("limit", text="You've reached the messaging limit"))
        return page

    # state changes

    def _listing_from_query(self) -> FakeListing | None:
        listing_id = parse_qs(urlsplit(self.url).query).get("listing_id", [""])[0]
        listing = self.site.listings.get(listing_id)
        return listing if listing is not None and listing.status != "deleted" else None

    def _review(self) -> None:
        self._stage = "review"

    def _publish(self) -> None:
        form = self._form
        listing = self.site.add_listing(
            str(form["title"]),
            str(form["price"]),
            category=str(form["category"]),
            condition=str(form["condition"]),
            photos=list(form["photos"]),  # type: ignore[call-overload]
        )
        listing.description = str(form.get("description", ""))
        if self.site.boost_prompt:
            self._dialog = "boost"
        else:
            self._to_selling()

    def _to_selling(self) -> None:
        self.url = fb.SELLING_URL
        self._dialog = None

    def _save(self, listing: FakeListing) -> None:
        listing.title = str(self._form["title"])
        listing.price = str(self._form["price"])
        listing.description = str(self._form["description"])
        self.url = fb.ITEM_URL.format(listing_id=listing.listing_id)

    def _mark_sold(self, listing: FakeListing) -> None:
        if self.site.sold_dialog:
            self._dialog = "sold"
        else:
            listing.status = "sold"

    def _finish_sold(self, listing: FakeListing) -> None:
        listing.status = "sold"
        self._dialog = None

    def _open_menu(self) -> None:
        self._menu = True

    def _confirm_delete(self) -> None:
        self._menu, self._dialog = False, "delete"

    def _delete(self, listing: FakeListing) -> None:
        listing.status = "deleted"
        self._dialog = None
        self.url = fb.SELLING_URL

    def _open_folder(self) -> None:
        self._folder_open = True

    def _chat(self) -> FakeChat | None:
        path = urlsplit(self.url).path
        if not path.startswith("/messages/t/") or path == "/messages/t/":
            return None
        return self.site.chats.get(path.split("/")[3])

    def _send(self) -> None:
        chat = self._chat()
        body = str(self._form.pop("input", ""))
        if chat is None or not body:
            return
        if self.site.rate_limited:
            self._dialog = "limit"
            return
        chat.messages.append(("Just now", "You", body))


def _button(key: str, name: str, on_click: Callable[[], None], *, disabled: bool = False) -> El:
    attrs = {"aria-disabled": "true"} if disabled else {}
    return El(
        key,
        role="button",
        name=name,
        text=name,
        attrs=attrs,
        on_click=on_click,
        css=frozenset({f"[role='button'][aria-label='{name}']"}),
    )


def _walk(elements: Sequence[El]) -> list[El]:
    found = []
    for el in elements:
        found.append(el)
        found.extend(_walk(el.children))
    return found
