"""Browser access for the Facebook adapter.

| Class | Job |
|---|---|
| BrowserDriver | What the adapter needs from a browser; handles are opaque |
| PacedDriver | Wraps a driver: a human pause before every action |
| PlaywrightDriver | Chromium over the persistent profile, started on first use |

A LocatorSpec names one way to find an element, Playwright style:

| Key | Meaning |
|---|---|
| role (+ name) | ARIA role, accessible name |
| label, text, placeholder | Label text, visible text, placeholder |
| css | Any Playwright selector string (css, or "xpath=...") |
| exact | Whole string, case sensitive (default: substring, any case) |
| attached | Also match hidden elements (file inputs) |
"""

import atexit
import queue
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from functools import reduce
from pathlib import Path
from typing import Any, Protocol, TypedDict

from marketplace_agent.platforms.base import PlatformAuthError, PlatformError
from marketplace_agent.platforms.facebook.pacing import Pacer

Handle = Any  # opaque to callers; a Playwright Locator in PlaywrightDriver
_Job = tuple[Callable[..., Any], tuple[Any, ...], Future[Any]]

PROFILE_IN_USE = (
    "The Facebook browser profile at {path} is in use by another process: `agent serve`, "
    "`agent run`, `agent web`, or `agent facebook login`. Only one can use it at a time. Stop "
    "that one (Ctrl+C) or let it finish, then try again. With Facebook on, run `agent serve` "
    "on its own: it does everything in one process."
)
MAX_MATCHES = 1000  # a bound, above any chat's message count or the owner's listings
OPTION_SCROLLS = 8
PLAYWRIGHT_MISSING = (
    "Playwright is not installed. Run `uv sync --extra facebook`, then "
    "`uv run playwright install chromium`."
)


class LocatorSpec(TypedDict, total=False):
    role: str
    name: str
    label: str
    text: str
    placeholder: str
    css: str
    exact: bool
    attached: bool


class DriverError(PlatformError):
    """A browser action failed (timeout, closed window, missing element)."""


class ProfileInUse(DriverError, PlatformAuthError):
    """Another process has the browser profile open. Like a sign-out, only the owner can fix it,
    so it stops the platform for now and the runner alerts the owner."""


class BrowserDriver(Protocol):
    def goto(self, url: str) -> None: ...

    def find(self, spec: LocatorSpec, within: Handle | None = None) -> list[Handle]:
        """Elements matching spec right now, visible ones unless spec["attached"]."""
        ...

    def wait_for(self, specs: Sequence[LocatorSpec], timeout_s: float) -> bool:
        """Wait until any spec matches; False on timeout."""
        ...

    def wait(self, seconds: float) -> None: ...

    def click(self, handle: Handle) -> None: ...

    def scroll_into_view(self, handle: Handle) -> None:
        """Scroll until handle is on screen, which makes lazy lists load what lies past it."""
        ...

    def fill(self, handle: Handle, text: str) -> None:
        """Clear the field and type text like a person; newlines become Shift+Enter."""
        ...

    def press(self, key: str) -> None: ...

    def upload(self, handle: Handle, paths: Sequence[Path]) -> None: ...

    def select_option(self, handle: Handle, option: str, *, exact: bool = False) -> None:
        """Pick option in a native select or a custom dropdown opened by clicking handle."""
        ...

    def text_of(self, handle: Handle) -> str: ...

    def attribute(self, handle: Handle, name: str) -> str | None: ...

    def screenshot(self) -> bytes: ...

    def aria_snapshot(self) -> str: ...

    def current_url(self) -> str: ...

    def has_cookie(self, name: str) -> bool: ...

    def release_page(self) -> None:
        """Leave the current tab open as it is for the human; later calls use a new tab."""
        ...

    def close(self) -> None: ...


class PacedDriver:
    """A pause from pacer before every navigation and every input; reads are not paced."""

    def __init__(self, inner: BrowserDriver, pacer: Pacer):
        self.inner = inner
        self.pacer = pacer

    def goto(self, url: str) -> None:
        self.pacer.pause()
        self.inner.goto(url)

    def find(self, spec: LocatorSpec, within: Handle | None = None) -> list[Handle]:
        return self.inner.find(spec, within)

    def wait_for(self, specs: Sequence[LocatorSpec], timeout_s: float) -> bool:
        return self.inner.wait_for(specs, timeout_s)

    def wait(self, seconds: float) -> None:
        self.inner.wait(seconds)

    def click(self, handle: Handle) -> None:
        self.pacer.pause()
        self.inner.click(handle)

    def scroll_into_view(self, handle: Handle) -> None:
        self.pacer.pause()
        self.inner.scroll_into_view(handle)

    def fill(self, handle: Handle, text: str) -> None:
        self.pacer.pause()
        self.inner.fill(handle, text)

    def press(self, key: str) -> None:
        self.pacer.pause()
        self.inner.press(key)

    def upload(self, handle: Handle, paths: Sequence[Path]) -> None:
        self.pacer.pause()
        self.inner.upload(handle, paths)

    def select_option(self, handle: Handle, option: str, *, exact: bool = False) -> None:
        self.pacer.pause()
        self.inner.select_option(handle, option, exact=exact)

    def text_of(self, handle: Handle) -> str:
        return self.inner.text_of(handle)

    def attribute(self, handle: Handle, name: str) -> str | None:
        return self.inner.attribute(handle, name)

    def screenshot(self) -> bytes:
        return self.inner.screenshot()

    def aria_snapshot(self) -> str:
        return self.inner.aria_snapshot()

    def current_url(self) -> str:
        return self.inner.current_url()

    def has_cookie(self, name: str) -> bool:
        return self.inner.has_cookie(name)

    def release_page(self) -> None:
        self.inner.release_page()

    def close(self) -> None:
        self.inner.close()


class PlaywrightDriver:
    """BrowserDriver over launch_persistent_context(profile_dir), started on first use.

    Playwright's sync API only works on the thread that started it, while the runner, the web
    app, and Discord call in from their own threads. So every call runs on one daemon worker
    thread, one at a time. on_context runs once on the new browser context (tests use it to
    route requests to local fixtures).
    """

    def __init__(
        self,
        profile_dir: Path,
        *,
        headless: bool,
        pacer: Pacer,
        timeout_s: float = 30.0,
        on_context: Callable[[Any], None] | None = None,
    ):
        self.profile_dir = profile_dir
        self.headless = headless
        self.pacer = pacer
        self.timeout_ms = timeout_s * 1000
        self._on_context = on_context
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._playwright: Any = None
        self._context: Any = None
        self._page: Any = None

    # BrowserDriver

    def goto(self, url: str) -> None:
        self._run(lambda: self._current().goto(url, wait_until="domcontentloaded"))

    def find(self, spec: LocatorSpec, within: Handle | None = None) -> list[Handle]:
        def matches() -> list[Handle]:
            locator = self._locator(spec, within)
            return [locator.nth(i) for i in range(min(locator.count(), MAX_MATCHES))]

        return self._run(matches)

    def wait_for(self, specs: Sequence[LocatorSpec], timeout_s: float) -> bool:
        def waited() -> bool:
            combined = reduce(lambda a, b: a.or_(b), (self._locator(s, None) for s in specs))
            try:
                combined.first.wait_for(state="attached", timeout=timeout_s * 1000)
            except Exception as exc:
                if type(exc).__name__ == "TimeoutError":
                    return False
                raise
            return True

        return bool(specs) and self._run(waited)

    def wait(self, seconds: float) -> None:
        self._run(lambda: self._current().wait_for_timeout(seconds * 1000))

    def click(self, handle: Handle) -> None:
        def clicked() -> None:
            handle.scroll_into_view_if_needed(timeout=self.timeout_ms)
            handle.click(timeout=self.timeout_ms)

        self._run(clicked)

    def scroll_into_view(self, handle: Handle) -> None:
        self._run(lambda: handle.scroll_into_view_if_needed(timeout=self.timeout_ms))

    def fill(self, handle: Handle, text: str) -> None:
        self._run(self._type_into, handle, text)

    def press(self, key: str) -> None:
        self._run(lambda: self._current().keyboard.press(key))

    def upload(self, handle: Handle, paths: Sequence[Path]) -> None:
        self._run(lambda: handle.set_input_files([str(p) for p in paths]))

    def select_option(self, handle: Handle, option: str, *, exact: bool = False) -> None:
        self._run(self._choose, handle, option, exact)

    def text_of(self, handle: Handle) -> str:
        return str(self._run(lambda: handle.inner_text(timeout=self.timeout_ms)))

    def attribute(self, handle: Handle, name: str) -> str | None:
        return self._run(lambda: handle.get_attribute(name, timeout=self.timeout_ms))

    def screenshot(self) -> bytes:
        return bytes(self._run(lambda: self._current().screenshot(type="png")))

    def aria_snapshot(self) -> str:
        return str(self._run(lambda: self._current().locator("body").aria_snapshot()))

    def current_url(self) -> str:
        return str(self._run(lambda: self._current().url))

    def has_cookie(self, name: str) -> bool:
        def found() -> bool:
            self._current()
            return any(cookie.get("name") == name for cookie in self._context.cookies())

        return self._run(found)

    def release_page(self) -> None:
        def released() -> None:
            if self._context is not None:
                self._page = self._context.new_page()

        self._run(released)

    def close(self) -> None:
        with self._lock:
            worker = self._worker
        if worker is None:
            return
        try:
            self._run(self._shutdown)
        finally:
            self._queue.put(None)
            worker.join(timeout=10)
            with self._lock:
                self._worker = None
            atexit.unregister(self.close)

    # worker thread

    def _run[T](self, fn: Callable[..., T], *args: Any) -> T:
        self._ensure_worker()
        future: Future[T] = Future()
        self._queue.put((fn, args, future))
        return future.result()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._serve, name="facebook-browser", daemon=True
                )
                self._worker.start()
                atexit.register(self.close)

    def _serve(self) -> None:
        while (job := self._queue.get()) is not None:
            fn, args, future = job
            try:
                future.set_result(fn(*args))
            except Exception as exc:  # handed to the calling thread
                future.set_exception(self._translate(exc))

    def _translate(self, exc: Exception) -> Exception:
        if isinstance(exc, PlatformError) or not type(exc).__module__.startswith("playwright"):
            return exc
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        if re.search(r"ProcessSingleton|SingletonLock|already in use", str(exc)):
            return ProfileInUse(PROFILE_IN_USE.format(path=self.profile_dir))
        if "Executable doesn't exist" in str(exc):
            return DriverError(
                "Chromium for Playwright is not installed: run `uv run playwright install chromium`."
            )
        if "has been closed" in str(exc):
            self._context = self._page = None
            return DriverError("The browser window was closed; try again to reopen it.")
        return DriverError(f"Browser action failed: {first_line}")

    def _current(self) -> Any:
        if self._context is None:
            self._start()
        if self._page is None or self._page.is_closed():
            open_pages = [p for p in self._context.pages if not p.is_closed()]
            self._page = open_pages[-1] if open_pages else self._context.new_page()
        return self._page

    def _start(self) -> None:
        """Launch the browser; the Playwright instance is kept across relaunches, since only
        one may run per thread."""
        if self._playwright is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:
                raise DriverError(PLAYWRIGHT_MISSING) from exc
            self._playwright = sync_playwright().start()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        # The headless shell cannot read a persistent profile; the full build can.
        context = self._playwright.chromium.launch_persistent_context(
            str(self.profile_dir),
            headless=self.headless,
            channel="chromium",
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        context.set_default_timeout(self.timeout_ms)
        if self._on_context is not None:
            self._on_context(context)
        self._context, self._page = context, None

    def _shutdown(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._playwright = self._context = self._page = None

    def _locator(self, spec: LocatorSpec, within: Handle | None) -> Any:
        root = within if within is not None else self._current()
        exact = spec.get("exact", False)
        if "role" in spec:
            name = spec.get("name")
            locator = root.get_by_role(spec["role"], name=name, exact=exact if name else None)
        elif "label" in spec:
            locator = root.get_by_label(spec["label"], exact=exact)
        elif "text" in spec:
            locator = root.get_by_text(spec["text"], exact=exact)
        elif "placeholder" in spec:
            locator = root.get_by_placeholder(spec["placeholder"], exact=exact)
        else:
            locator = root.locator(spec["css"])
        return locator if spec.get("attached") else locator.filter(visible=True)

    def _type_into(self, handle: Any, text: str) -> None:
        handle.click(timeout=self.timeout_ms)
        handle.fill("", timeout=self.timeout_ms)
        keyboard = self._current().keyboard
        for number, line in enumerate(text.split("\n")):
            if number:
                keyboard.press("Shift+Enter")
            for chunk in re.findall(r"\S+\s*|\s+", line):
                keyboard.type(chunk, delay=self.pacer.key_delay_ms())

    def _choose(self, handle: Any, option: str, exact: bool) -> None:
        if handle.evaluate("element => element.tagName") == "SELECT":
            handle.select_option(label=option, timeout=self.timeout_ms)
            return
        page = self._current()
        handle.click(timeout=self.timeout_ms)
        page.wait_for_timeout(700)
        for _ in range(OPTION_SCROLLS):
            by_role = page.get_by_role("option", name=option, exact=exact).filter(visible=True)
            if by_role.count():
                by_role.first.click(timeout=self.timeout_ms)
                return
            by_text = page.get_by_text(option, exact=exact).filter(visible=True)
            if by_text.count():
                by_text.last.click(timeout=self.timeout_ms)
                return
            page.keyboard.press("PageDown")  # long lists render only what is scrolled into view
            page.wait_for_timeout(400)
        page.keyboard.press("Escape")
        raise DriverError(f"No option {option!r} in the list.")
