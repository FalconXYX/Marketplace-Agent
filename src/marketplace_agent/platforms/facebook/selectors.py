"""Named targets ("create.title") with ordered locator specs, and the Finder that resolves them.

Defaults ship in selectors.json. Overrides in data_dir/facebook_selectors.json, written by hand
or by the healer, map a target name to a list of specs that are tried before the defaults:

    {"create.title": [{"label": "Listing title"}]}

A target with "heal": false is one whose absence means something (a login form, a gone notice),
so the Finder never asks a model to find it.
"""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Protocol, cast

from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.platforms.facebook.driver import BrowserDriver, Handle, LocatorSpec

OVERRIDES_FILE = "facebook_selectors.json"
SPEC_KINDS = ("role", "label", "text", "placeholder", "css")
_FLAGS = ("exact", "attached")
FIND_TIMEOUT_S = 10.0

logger = logging.getLogger("marketplace_agent.facebook")


@dataclass(frozen=True)
class Target:
    name: str
    purpose: str
    specs: tuple[LocatorSpec, ...]
    heal: bool = True


def parse_spec(raw: object) -> LocatorSpec:
    """A validated LocatorSpec; ValueError says what is wrong."""
    if not isinstance(raw, dict):
        raise ValueError("a locator spec must be a JSON object")
    kinds = [kind for kind in SPEC_KINDS if kind in raw]
    if len(kinds) != 1:
        raise ValueError(f"a locator spec needs exactly one of: {', '.join(SPEC_KINDS)}")
    allowed = {kinds[0], *_FLAGS} | ({"name"} if kinds[0] == "role" else set())
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unexpected keys in a {kinds[0]} spec: {', '.join(unknown)}")
    for key, value in raw.items():
        if key in _FLAGS and not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        if key not in _FLAGS and not (isinstance(value, str) and value.strip()):
            raise ValueError(f"{key} must be a non-empty string")
    return cast(LocatorSpec, dict(raw))


class SelectorRegistry:
    def __init__(self, targets: Mapping[str, Target], override_path: Path | None = None):
        self._targets = dict(targets)
        self.override_path = override_path
        self._overrides = self._read_overrides()

    @classmethod
    def load(cls, override_path: Path | None = None) -> "SelectorRegistry":
        """The packaged defaults, with overrides from override_path when it exists."""
        text = (
            files("marketplace_agent.platforms.facebook")
            .joinpath("selectors.json")
            .read_text(encoding="utf-8")
        )
        return cls(_parse_targets(json.loads(text)), override_path)

    def names(self) -> list[str]:
        return list(self._targets)

    def target(self, name: str) -> Target:
        """The target with its effective specs, overrides included."""
        found = self._default(name)
        return Target(found.name, found.purpose, tuple(self.specs(name)), found.heal)

    def specs(self, name: str) -> list[LocatorSpec]:
        """Overrides first, then the defaults, without repeats."""
        ordered: list[LocatorSpec] = []
        for spec in (*self._overrides.get(name, ()), *self._default(name).specs):
            if spec not in ordered:
                ordered.append(spec)
        return ordered

    def save_override(self, name: str, spec: LocatorSpec) -> None:
        """Put spec first for name and write the overrides file."""
        self._default(name)
        self._overrides[name] = [spec, *(s for s in self._overrides.get(name, ()) if s != spec)]
        if self.override_path is None:
            return
        self.override_path.parent.mkdir(parents=True, exist_ok=True)
        staging = self.override_path.with_name(self.override_path.name + ".tmp")
        staging.write_text(json.dumps(self._overrides, indent=2) + "\n", encoding="utf-8")
        staging.replace(self.override_path)

    def _default(self, name: str) -> Target:
        try:
            return self._targets[name]
        except KeyError:
            raise ValueError(f"unknown Facebook target {name!r}") from None

    def _read_overrides(self) -> dict[str, list[LocatorSpec]]:
        """A broken overrides file is skipped with a warning; the defaults still work."""
        if self.override_path is None or not self.override_path.exists():
            return {}
        try:
            data = json.loads(self.override_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("ignoring %s: %s", self.override_path, exc)
            return {}
        if not isinstance(data, dict):
            logger.warning("ignoring %s: expected an object of target names", self.override_path)
            return {}
        overrides: dict[str, list[LocatorSpec]] = {}
        for name, raw_specs in data.items():
            if name not in self._targets or not isinstance(raw_specs, list):
                logger.warning("ignoring override %r in %s", name, self.override_path)
                continue
            overrides[name] = _valid_specs(name, raw_specs)
        return overrides


def _parse_targets(data: object) -> dict[str, Target]:
    if not isinstance(data, dict):
        raise ValueError("selectors must be an object of target names")
    targets = {}
    for name, entry in data.items():
        specs = tuple(parse_spec(raw) for raw in entry["specs"])
        targets[name] = Target(name, entry["purpose"], specs, entry.get("heal", True))
    return targets


def _valid_specs(name: str, raw_specs: list[object]) -> list[LocatorSpec]:
    specs = []
    for raw in raw_specs:
        try:
            specs.append(parse_spec(raw))
        except ValueError as exc:
            logger.warning("ignoring an override for %s: %s", name, exc)
    return specs


class Healer(Protocol):
    def heal(self, target: Target) -> Handle: ...


class Finder:
    """Resolves targets on the current page: each spec in order, waiting up to timeout_s for
    the page to render, then the healer when no spec matches. before_heal runs first and may
    raise, for a page that is hidden rather than changed."""

    def __init__(
        self,
        driver: BrowserDriver,
        registry: SelectorRegistry,
        healer: Healer | None = None,
        *,
        timeout_s: float = FIND_TIMEOUT_S,
        before_heal: Callable[[], None] | None = None,
    ):
        self.driver = driver
        self.registry = registry
        self.healer = healer
        self.timeout_s = timeout_s
        self.before_heal = before_heal

    def find(self, name: str) -> Handle:
        hit = self.probe(name, self.timeout_s)
        if hit is not None:
            return hit
        target = self.registry.target(name)
        if self.healer is None or not target.heal:
            raise PlatformError(
                f"Facebook: nothing on the page matches {name} ({target.purpose}). "
                "Run `agent doctor`."
            )
        if self.before_heal is not None:
            self.before_heal()
        return self.healer.heal(target)

    def probe(
        self, name: str, timeout_s: float = 0.0, within: Handle | None = None
    ) -> Handle | None:
        """The first match, or None. Never heals."""
        specs = self.registry.specs(name)
        hit = self._scan(specs, within)
        if hit is None and timeout_s > 0 and self.driver.wait_for(specs, timeout_s):
            hit = self._scan(specs, within)
        return hit

    def find_all(self, name: str, within: Handle | None = None) -> list[Handle]:
        """Every match of the first spec that matches anything."""
        for spec in self.registry.specs(name):
            hits = self.driver.find(spec, within)
            if hits:
                return hits
        return []

    def first_of(self, names: Sequence[str], timeout_s: float) -> str | None:
        """The first of names present on the page, waiting up to timeout_s for any of them."""
        found = self._present(names)
        if found is None:
            specs = [spec for name in names for spec in self.registry.specs(name)]
            if self.driver.wait_for(specs, timeout_s):
                found = self._present(names)
        return found

    def _present(self, names: Sequence[str]) -> str | None:
        return next((name for name in names if self.probe(name) is not None), None)

    def _scan(self, specs: Sequence[LocatorSpec], within: Handle | None) -> Handle | None:
        for spec in specs:
            hits = self.driver.find(spec, within)
            if hits:
                return hits[0]
        return None
