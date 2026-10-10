"""Rendering for the web UI: templates, flash messages, and the responses routes return.

An htmx request (HX-Request header) gets the panel it targets; any other request gets the full
page, so every form also works without JavaScript.

Every `{{ value }}` passes through redact_address before escaping (Jinja finalize), so no page
can show the pickup address, whatever text it came in.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Request, Response
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

from marketplace_agent.ai.drafting import ListingDraft
from marketplace_agent.config import Settings
from marketplace_agent.control.service import ControlError, ControlService
from marketplace_agent.core.comps import CompsLookup, item_comps
from marketplace_agent.core.items import ItemNotFound
from marketplace_agent.core.models import Firmness, ItemStatus
from marketplace_agent.core.runtime import Runtime
from marketplace_agent.runner.takedown import listing_error
from marketplace_agent.guard.invariants import redact_address
from marketplace_agent.tools.base import ToolResult
from marketplace_agent.tools.impl.common import fmt_local, fmt_money
from marketplace_agent.web.forms import (
    CONDITIONS,
    DraftFields,
    FormError,
    dollars_value,
    prefill,
)
from marketplace_agent.web.views import (
    ItemView,
    item_cards,
    item_view,
    loose_escalations,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"

STATUS_LABELS = {
    ItemStatus.DRAFT: "Draft",
    ItemStatus.LISTED: "For sale",
    ItemStatus.COMMITTED: "Promised to a buyer",
    ItemStatus.SOLD: "Sold",
    ItemStatus.WITHDRAWN: "Withdrawn",
}
GROUP_LABELS = {**STATUS_LABELS, ItemStatus.DRAFT: "Drafts"}  # headings on the home page
CONDITION_LABELS = {
    "new": "New",
    "like_new": "Like new",
    "good": "Good",
    "fair": "Fair",
    "poor": "Poor",
}
FIRMNESS_LABELS = {
    Firmness.FIRM: "Firm: no haggling",
    Firmness.SLIGHT: "A little room",
    Firmness.FLEXIBLE: "Flexible, down to the floor",
}
VIA_LABELS = {  # who sent an outbound message
    "agent": "the agent",
    "pattern_cache": "auto-reply",
    "human": "you",
    "system": "the system",
}
DATABASE_BUSY = "The agent is busy writing to the database. Try again in a moment."

Outcome = str | ToolResult | list[ToolResult]
Action = Callable[[ControlService], Outcome]


@dataclass(frozen=True)
class Flash:
    text: str
    ok: bool = True
    anchor: str = "top"  # the section it shows in; see ItemView.anchors
    details: tuple[str, ...] = ()  # one line per platform after publishing


@dataclass
class DraftState:
    """The draft form and what came with it."""

    fields: DraftFields | None = None  # None: from the item, with the suggested price
    errors: dict[str, str] = field(default_factory=dict)
    suggestion: ListingDraft | None = None
    note: str | None = None  # why there is no draft, when drafting did not work
    retry: bool = False  # offer to draft again
    auto: bool = False  # draft on page load: the item has no title yet
    comps: CompsLookup | None = None  # cached comparable listings, shown beside the price


class Ui:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.settings = runtime.settings
        self.templates = Jinja2Templates(env=_environment(runtime.settings))

    def render(
        self, request: Request, name: str, context: dict[str, Any], status: int = 200
    ) -> Response:
        return self.templates.TemplateResponse(request, name, context, status_code=status)

    # ------------------------------------------------------------ pages

    def home(self, request: Request, flash: Flash | None = None, status: int = 200) -> Response:
        """The item list. htmx gets only the "needs you" panel, which its forms target."""
        with Session(self.runtime.engine) as session:
            context = {
                "groups": item_cards(session),
                "loose": loose_escalations(session),
                "flash": flash,
            }
        name = "_attention.html" if is_htmx(request) else "index.html"
        return self.render(request, name, context, status)

    def item(
        self,
        request: Request,
        item_id: int,
        *,
        flash: Flash | None = None,
        draft: DraftState | None = None,
        status: int = 200,
    ) -> Response:
        """The item panel for htmx, else the item page. Raises ItemNotFound."""
        view, draft = self._item_context(item_id, draft)
        if flash is not None and flash.anchor not in view.anchors:
            flash = replace(flash, anchor="top")
        context = {"view": view, "draft": draft, "flash": flash}
        name = "_panel.html" if is_htmx(request) else "item.html"
        return self.render(request, name, context, status)

    def draft(self, request: Request, item_id: int, draft: DraftState) -> Response:
        """The draft form for htmx, else the item page. Raises ItemNotFound."""
        if not is_htmx(request):
            return self.item(request, item_id, draft=draft)
        view, draft = self._item_context(item_id, draft)
        return self.render(request, "_draft.html", {"view": view, "draft": draft})

    def message(self, request: Request, status: int, text: str) -> Response:
        """A message on its own: in the toast area for htmx, else as a page."""
        if is_htmx(request):
            response = self.render(request, "_toast.html", {"text": text}, status)
            response.headers.update({"HX-Retarget": "#toast", "HX-Reswap": "innerHTML"})
            return response
        return self.render(request, "message.html", {"text": text, "status": status}, status)

    # ------------------------------------------------------------ owner actions

    def act(
        self,
        request: Request,
        item_id: int | None,
        anchor: str,
        action: Action,
    ) -> Response:
        """Run a ControlService call and show the outcome where the owner asked for it: on the
        item, or on the home page when it concerns no item."""
        try:
            outcome = action(self.runtime.control)
        except FormError as exc:
            flash, status = Flash(str(exc), ok=False, anchor=anchor), 422
        except ControlError as exc:
            flash, status = Flash(str(exc), ok=False, anchor=anchor), 409
        except OperationalError as exc:
            if not busy(exc):
                raise
            flash, status = Flash(DATABASE_BUSY, ok=False, anchor=anchor), 409
        else:
            flash, status = _outcome_flash(outcome, anchor)
        if item_id is None:
            return self.home(request, flash, status)
        try:
            return self.item(request, item_id, flash=flash, status=status)
        except ItemNotFound:
            return self.message(request, 404, f"There is no item with id {item_id}.")

    # ------------------------------------------------------------ helpers

    def _item_context(
        self, item_id: int, draft: DraftState | None
    ) -> tuple[ItemView, DraftState | None]:
        now = self.runtime.clock.now()
        with Session(self.runtime.engine) as session:
            view = item_view(session, self.settings, now, item_id)
            if view.item.status != ItemStatus.DRAFT:
                return view, None
            comps = item_comps(session, view.item)
        if draft is None:
            draft = DraftState(auto=not view.item.title.strip())
        if draft.fields is None:
            enabled = self.settings.enabled_platforms
            fields = prefill(view.item, view.goal, enabled, now, draft.suggestion)
            draft = replace(draft, fields=fields)
        return view, replace(draft, comps=comps)


def get_ui(request: Request) -> Ui:
    ui: Ui = request.app.state.ui
    return ui


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def _outcome_flash(outcome: Outcome, anchor: str) -> tuple[Flash, int]:
    if isinstance(outcome, ToolResult):
        return Flash(outcome.summary, ok=outcome.ok, anchor=anchor), 200 if outcome.ok else 409
    if isinstance(outcome, list):
        return _published_flash(outcome, anchor)
    return Flash(outcome, anchor=anchor), 200


def _published_flash(results: list[ToolResult], anchor: str) -> tuple[Flash, int]:
    """One line per platform. The item is for sale whatever happened, so only a total failure
    is an error status."""
    details = tuple(result.summary for result in results)
    if all(result.ok for result in results):
        return Flash("Confirmed for sale.", anchor=anchor, details=details), 200
    if any(result.ok for result in results):
        text = "Confirmed for sale, but not every post went up:"
        return Flash(text, ok=False, anchor=anchor, details=details), 200
    text = "Confirmed for sale, but nothing was posted:"
    return Flash(text, ok=False, anchor=anchor, details=details), 409


def busy(exc: OperationalError) -> bool:
    """SQLite's write lock was held past busy_timeout: worth trying again."""
    text = str(exc.orig)
    return "locked" in text or "busy" in text


def _environment(settings: Settings) -> Environment:
    def finalize(value: object) -> object:
        if isinstance(value, str) and not isinstance(value, Markup):
            return redact_address(value, settings)
        return value

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        finalize=finalize,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        money=fmt_money,
        local=lambda when: fmt_local(settings, when),
        dollars=dollars_value,
        web_url=web_url,
        listing_error=listing_error,
    )
    env.globals.update(
        status_labels=STATUS_LABELS,
        group_labels=GROUP_LABELS,
        via_labels=VIA_LABELS,
        conditions=[(value, CONDITION_LABELS[value]) for value in CONDITIONS],
        firmness_levels=[(level.value, FIRMNESS_LABELS[level]) for level in Firmness],
        enabled_platforms=settings.enabled_platforms,
        max_photos=settings.web_max_photos,
        max_photo_mb=settings.web_max_photo_mb,
    )
    return env


def web_url(url: str | None) -> str | None:
    """A platform URL fit for a link: http(s) only, so a javascript: URL never becomes one."""
    if not url:
        return None
    return url if urlsplit(url).scheme in ("http", "https") else None
