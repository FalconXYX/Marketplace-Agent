"""Self healing selectors: when no locator for a target matches, a vision model proposes one.

| Step | What happens |
|---|---|
| Budget | facebook_heal_budget_per_day model calls per local day, counted in data_dir/facebook_heal_budget.json before the call |
| Ask | accessibility snapshot (pickup address redacted) plus a screenshot -> VISION request, purpose "heal" |
| Chat pages | Messenger shows the owner's personal chats there: no screenshot, and only the snapshot's outline (roles, short control labels, the names the old locators used) |
| Check | the answer is a valid LocatorSpec that matches exactly one element |
| Keep | saved as an override, so it is tried first from then on |

Every failure raises PlatformError naming the target and pointing at `agent doctor`.
"""

import json
import logging
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from marketplace_agent.ai.base import Capability, ImageInput, LLMRequest, ProviderError
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock
from marketplace_agent.guard.invariants import redact_address
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.platforms.facebook.driver import BrowserDriver, Handle, LocatorSpec
from marketplace_agent.platforms.facebook.selectors import (
    SPEC_KINDS,
    SelectorRegistry,
    Target,
    parse_spec,
)

BUDGET_FILE = "facebook_heal_budget.json"
SNAPSHOT_LIMIT = 40_000
CHAT_PREFIXES = ("inbox.", "thread.")  # targets on Messenger pages, which show every chat
CHAT_PATH = "/messages"
# Roles whose short names are interface labels ("Message", "Chats"); others carry chat text.
CONTROL_ROLES = frozenset(
    {
        "alertdialog", "banner", "button", "checkbox", "combobox", "complementary",
        "contentinfo", "dialog", "form", "grid", "group", "list", "listbox", "main", "menu",
        "menubar", "menuitem", "navigation", "radio", "region", "search", "searchbox", "switch",
        "tab", "table", "tablist", "textbox", "toolbar", "tree", "treegrid",
    }
)  # fmt: skip
LABEL_MAX_CHARS, LABEL_MAX_WORDS = 30, 4
_SNAPSHOT_LINE = re.compile(r"""^(?P<indent> *)- +['"]?(?P<role>[a-z]+)\b(?P<rest>.*)$""")
_SNAPSHOT_NAME = re.compile(r'\\?"(?P<name>.*?)\\?"')
_CSS_TEXT = re.compile(r"""(?:aria-label[*^$]?=|placeholder=|has-text\()\s*['"]([^'"]+)['"]""")

SYSTEM = (
    "You repair browser automation for Facebook Marketplace. The page changed and a saved "
    "locator stopped matching. Return one locator, as a JSON object, that matches exactly one "
    "element: the one described. Use one of these shapes: "
    '{"role": "<ARIA role>", "name": "<accessible name>"}, {"label": "<label text>"}, '
    '{"placeholder": "<placeholder>"}, {"text": "<visible text>"}, or as a last resort '
    '{"css": "<selector using attributes such as aria-label>"}. Never use generated class '
    "names. Copy roles, names, and text exactly as the snapshot shows them. If the element is "
    "not on the page, return {}."
)
SPEC_SCHEMA = {
    "type": "object",
    "properties": {kind: {"type": ["string", "null"]} for kind in ("name", *SPEC_KINDS)},
}

logger = logging.getLogger("marketplace_agent.facebook")


class HealBudget:
    """Model calls spent on healing per local day, kept in a small JSON file."""

    def __init__(self, path: Path, per_day: int, clock: Clock, timezone: str):
        self.path = path
        self.per_day = per_day
        self.clock = clock
        self.zone = ZoneInfo(timezone)

    def used_today(self) -> int:
        day, used = self._read()
        return used if day == self._today() else 0

    def spend(self) -> bool:
        """Count one call; False, counting nothing, when today's budget is spent."""
        used = self.used_today()
        if used >= self.per_day:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"day": self._today().isoformat(), "used": used + 1}
        self.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return True

    def _today(self) -> date:
        return self.clock.now().astimezone(self.zone).date()

    def _read(self) -> tuple[date | None, int]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return date.fromisoformat(data["day"]), int(data["used"])
        except (OSError, ValueError, KeyError, TypeError):
            return None, 0


class SelectorHealer:
    def __init__(
        self,
        driver: BrowserDriver,
        registry: SelectorRegistry,
        router: ProviderRouter,
        settings: Settings,
        clock: Clock,
    ):
        self.driver = driver
        self.registry = registry
        self.router = router
        self.settings = settings
        self.budget = HealBudget(
            settings.data_dir / BUDGET_FILE,
            settings.facebook_heal_budget_per_day,
            clock,
            settings.timezone,
        )

    def heal(self, target: Target) -> Handle:
        lost = f"Facebook: nothing on the page matches {target.name} ({target.purpose})"
        if not self.router.has_capability(Capability.VISION):
            raise PlatformError(
                f"{lost}, and no vision model is configured to repair it (set GEMINI_API_KEY). "
                "Run `agent doctor`."
            )
        if not self.budget.spend():
            raise PlatformError(
                f"{lost}, and today's repair budget of {self.budget.per_day} model calls is "
                "spent. Run `agent doctor`."
            )
        try:
            response = self.router.generate(self._request(target))
        except ProviderError as exc:
            raise PlatformError(f"{lost}, and the repair request failed ({exc}).") from None
        spec = self._spec(lost, response.data)
        hits = self.driver.find(spec)
        if len(hits) != 1:
            raise PlatformError(
                f"{lost}. The suggested locator {json.dumps(spec)} matches {len(hits)} elements, "
                "not exactly one. Run `agent doctor`."
            )
        self.registry.save_override(target.name, spec)
        logger.warning("healed %s with %s", target.name, json.dumps(spec))
        return hits[0]

    def _request(self, target: Target) -> LLMRequest:
        path = urlsplit(self.driver.current_url()).path
        chats = target.name.startswith(CHAT_PREFIXES) or path.startswith(CHAT_PATH)
        snapshot = self.driver.aria_snapshot()
        if chats:
            snapshot = chat_outline(snapshot, known_names(target))
        snapshot = redact_address(snapshot, self.settings)[:SNAPSHOT_LIMIT]
        prompt = "\n".join(
            [
                f"Target: {target.name}",
                f"What it is: {target.purpose}",
                f"Page: {path}",
                f"Locators that no longer match: {json.dumps(list(target.specs))}",
                "",
                "Accessibility snapshot of the page"
                + (" (outline only: chat text is left out)" if chats else "")
                + ":",
                snapshot,
            ]
        )
        images = []
        if not chats:
            images.append(ImageInput(self.driver.screenshot(), "image/png"))
        return LLMRequest(
            system=SYSTEM,
            prompt=prompt,
            capability=Capability.VISION,
            images=images,
            json_schema=SPEC_SCHEMA,
            max_output_tokens=300,
            temperature=0.0,
            purpose="heal",
        )

    @staticmethod
    def _spec(lost: str, data: dict[str, object] | None) -> LocatorSpec:
        answer = {key: value for key, value in (data or {}).items() if value not in (None, "")}
        if not answer:
            raise PlatformError(
                f"{lost}, and the model could not find it either. Run `agent doctor`."
            )
        try:
            return parse_spec(answer)
        except ValueError as exc:
            raise PlatformError(
                f"{lost}. The model's locator is unusable ({exc}). Run `agent doctor`."
            ) from None


def known_names(target: Target) -> list[str]:
    """The names, texts, and labels the target's own locators look for."""
    found: list[str] = []
    for spec in target.specs:
        found += [str(spec[key]) for key in ("name", "text", "label", "placeholder") if key in spec]
        found += _CSS_TEXT.findall(str(spec.get("css", "")))
    return list(dict.fromkeys(name.strip() for name in found if name.strip()))


def chat_outline(snapshot: str, known: list[str]) -> str:
    """A Messenger page's aria snapshot without its chats: each element's role, plus a name only
    when it is a short control label or one of the known names. Built from those pieces alone,
    so no message, preview, or contact name reaches the model."""
    lines = []
    for line in snapshot.splitlines():
        found = _SNAPSHOT_LINE.match(line)
        if found is None or found["role"] == "text":
            continue
        role = found["role"]
        named = _SNAPSHOT_NAME.search(found["rest"])
        name = _outline_name(role, named["name"] if named else "", known)
        lines.append(f"{found['indent']}- {role}" + (f' "{name}"' if name else ""))
    return "\n".join(lines)


def _outline_name(role: str, name: str, known: list[str]) -> str:
    name = " ".join(name.split())
    if role in CONTROL_ROLES and _is_label(name):
        return name
    folded = name.casefold()
    return next((k for k in known if k.casefold() in folded), "")


def _is_label(name: str) -> bool:
    return (
        0 < len(name) <= LABEL_MAX_CHARS
        and len(name.split()) <= LABEL_MAX_WORDS
        and not any(mark in name for mark in ":·\"'")
    )
