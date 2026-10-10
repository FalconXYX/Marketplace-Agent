"""Facebook locators: registry, overrides, fallback order, waiting, healing, and pacing."""

import json
import random
import re
from pathlib import Path

import pytest
from fixtures.facebook.fake_driver import El, FakeDriver, FakeSite

from marketplace_agent.ai.base import Capability, ProviderError
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.platforms.base import ListingPayload, PlatformError
from marketplace_agent.platforms.facebook import adapter as adapter_module
from marketplace_agent.platforms.facebook import flow_notes as fb
from marketplace_agent.platforms.facebook.adapter import FacebookAdapter
from marketplace_agent.platforms.facebook.driver import PacedDriver
from marketplace_agent.platforms.facebook.healer import BUDGET_FILE, SelectorHealer
from marketplace_agent.platforms.facebook.pacing import Pacer
from marketplace_agent.platforms.facebook.selectors import (
    OVERRIDES_FILE,
    Finder,
    SelectorRegistry,
    parse_spec,
)


def _router(engine, clock, script, **kwargs) -> tuple[ProviderRouter, ScriptedProvider]:
    provider = ScriptedProvider(script, **kwargs)
    return ProviderRouter([provider], QuotaTracker(engine, clock, {})), provider


@pytest.fixture
def overrides(settings) -> Path:
    return settings.data_dir / OVERRIDES_FILE


@pytest.fixture
def registry(overrides) -> SelectorRegistry:
    return SelectorRegistry.load(overrides)


@pytest.fixture
def driver() -> FakeDriver:
    driver = FakeDriver(FakeSite())
    driver.goto(fb.CREATE_URL)
    driver.actions.clear()
    return driver


def _healer(driver, registry, router, settings, clock) -> SelectorHealer:
    return SelectorHealer(driver, registry, router, settings, clock)


# ---------------------------------------------------------------- specs and registry


@pytest.mark.parametrize(
    "raw",
    [
        {"role": "button", "name": "Next", "exact": True},
        {"label": "Title"},
        {"text": "Mark as sold"},
        {"placeholder": "Aa"},
        {"css": "input[type='file']", "attached": True},
    ],
)
def test_parse_spec_accepts(raw):
    assert parse_spec(raw) == raw


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        ("Title", "JSON object"),
        ({}, "exactly one"),
        ({"label": "Title", "text": "Title"}, "exactly one"),
        ({"label": "Title", "name": "x"}, "unexpected keys"),
        ({"css": ""}, "non-empty"),
        ({"text": "x", "exact": "yes"}, "true or false"),
    ],
)
def test_parse_spec_rejects(raw, why):
    with pytest.raises(ValueError, match=why):
        parse_spec(raw)


def test_every_target_the_adapter_uses_is_defined(registry):
    source = Path(adapter_module.__file__).read_text()
    used = set(
        re.findall(
            r'"((?:login|page|create|dialog|edit|item|sold|delete|selling|inbox|thread)\.[a-z_]+)"',
            source,
        )
    )
    assert used and used <= set(registry.names())


def test_overrides_come_first(overrides, driver):
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text(json.dumps({"create.title": [{"css": "#other"}]}))
    registry = SelectorRegistry.load(overrides)
    driver.extra.append(El("other", css=frozenset({"#other"})))

    assert registry.specs("create.title")[0] == {"css": "#other"}
    assert Finder(driver, registry).find("create.title").key == "other"


def test_a_broken_overrides_file_falls_back_to_the_defaults(overrides, caplog):
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text("{not json")
    assert SelectorRegistry.load(overrides).specs("create.title")[0] == {
        "label": "Title",
        "exact": True,
    }

    overrides.write_text(
        json.dumps(
            {"nope.target": [{"text": "x"}], "create.title": [{"css": ""}, {"label": "Name"}]}
        )
    )
    registry = SelectorRegistry.load(overrides)
    assert registry.specs("create.title")[0] == {"label": "Name"}
    assert "nope.target" in caplog.text


def test_specs_fall_back_in_order(registry, driver):
    tried: list[dict] = []
    find = driver.find
    driver.find = lambda spec, within=None: tried.append(dict(spec)) or find(spec, within)
    first, second = registry.specs("create.title")[:2]
    driver.break_spec(first)

    assert Finder(driver, registry).find("create.title").key == "title"
    assert tried[:2] == [first, second]


def test_find_waits_for_a_slow_page(registry):
    driver = FakeDriver(FakeSite(), load_waits=2)
    driver.goto(fb.CREATE_URL)
    assert Finder(driver, registry).find("create.title").key == "title"
    assert "wait_for" in driver.kinds()


def test_a_missing_target_that_cannot_heal_says_run_doctor(registry, driver):
    with pytest.raises(PlatformError, match=r"login\.form.*agent doctor"):
        Finder(driver, registry).find("login.form")


def test_without_a_healer_a_missing_target_fails(registry, driver):
    driver.break_target(registry, "create.title")
    with pytest.raises(PlatformError, match=r"create\.title"):
        Finder(driver, registry).find("create.title")


# ---------------------------------------------------------------- healing


def test_heal_validates_persists_and_continues_publishing(
    settings, clock, engine, registry, overrides, tmp_path
):
    router, provider = _router(engine, clock, [{"css": "#title-v2", "name": None}])
    site = FakeSite()
    driver = FakeDriver(site)
    driver.break_target(registry, "create.title")
    adapter = FacebookAdapter(
        settings, clock, driver, router, registry=registry, find_timeout_s=1.0
    )
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"jpeg")
    payload = ListingPayload("Desk", "Solid.", 5000, [photo], "Bloor", "Furniture", "good")

    ref = adapter.publish(payload)

    assert site.listings[ref.remote_id].title == "Desk"
    [request] = provider.requests
    assert (request.purpose, request.capability) == ("heal", Capability.VISION)
    assert request.images and request.images[0].mime_type == "image/png"
    assert "create.title" in request.prompt and "the Title text input" in request.prompt
    assert json.loads(overrides.read_text())["create.title"][0] == {"css": "#title-v2"}

    adapter.publish(payload)  # the override now matches first: no second model call
    assert len(provider.requests) == 1


def test_healing_a_chat_target_sends_no_screenshot_and_no_address(
    settings, clock, engine, registry
):
    router, provider = _router(engine, clock, [{"placeholder": "Aa"}])
    site = FakeSite()
    chat = site.add_chat("Lucas", "Desk", ("Lucas", "Where?"))
    driver = FakeDriver(site)
    driver.snapshot_extra = f"You: it is at {settings.pickup_address}"
    driver.break_target(registry, "thread.input")
    adapter = FacebookAdapter(
        settings, clock, driver, router, registry=registry, find_timeout_s=1.0
    )

    adapter.send_message(chat.chat_id, "See you at 7")

    [request] = provider.requests
    assert request.images == []
    assert settings.pickup_address not in request.prompt
    assert "it is at" not in request.prompt


# What Playwright's aria_snapshot gives for Messenger with personal chats in the list
MESSENGER_SNAPSHOT = """\
- tablist "Folders":
  - tab "Marketplace 2 new"
- navigation "Chats sidebar":
  - grid "Chats":
    - 'row "Mom Mom: your biopsy results came back, call me · 1h"':
      - 'link "Mom Mom: your biopsy results came back, call me · 1h"':
        - /url: /messages/t/100/
    - row "Family Dad sent the wifi password hunter2 · 2h":
      - button "Family Dad sent the wifi password hunter2 · 2h"
    - "row \\"Dr. Patel: it's moved to 3pm\\"":
      - button "More options for Dr. Patel"
- main:
  - heading "Mom" [level=2]
  - text: your biopsy results came back, call me
  - textbox "Message": I'll call after the appointment
  - button "Press enter to send"
"""
PRIVATE = ["biopsy", "Mom", "Dr. Patel", "Patel", "hunter2", "wifi", "3pm", "appointment", "/t/100"]


@pytest.mark.parametrize(
    "target", ["inbox.marketplace_folder", "inbox.thread_list", "thread.input"]
)
def test_healing_on_messenger_never_sends_the_owners_chats(
    settings, clock, engine, registry, target
):
    router, provider = _router(engine, clock, [{"role": "tab", "name": "Marketplace 2 new"}])
    driver = FakeDriver(FakeSite())
    driver.goto(fb.INBOX_URL)
    driver.snapshot_extra = MESSENGER_SNAPSHOT
    healer = _healer(driver, registry, router, settings, clock)

    with pytest.raises(PlatformError):
        healer.heal(registry.target(target))  # the fake page has no such tab: nothing matches

    [request] = provider.requests
    assert request.images == []
    leaked = [word for word in PRIVATE if word in request.prompt]
    assert leaked == []
    outline = request.prompt.split("chat text is left out):\n", 1)[1]
    assert '- tab "Marketplace 2 new"' in outline and '- grid "Chats"' in outline
    assert '- textbox "Message"' in outline and '- button "Press enter to send"' in outline
    assert "    - row\n" in outline


def test_healing_on_any_messenger_page_leaves_the_chats_out(settings, clock, engine, registry):
    """A target that is not a chat target, healed while a Messenger page is open."""
    router, provider = _router(engine, clock, [{}])
    driver = FakeDriver(FakeSite())
    driver.goto(fb.INBOX_URL)
    driver.snapshot_extra = MESSENGER_SNAPSHOT

    with pytest.raises(PlatformError, match="could not find it"):
        _healer(driver, registry, router, settings, clock).heal(registry.target("create.title"))

    [request] = provider.requests
    assert request.images == [] and "biopsy" not in request.prompt


def test_heal_budget_is_per_local_day(settings, clock, engine, registry, driver):
    settings.facebook_heal_budget_per_day = 1
    router, provider = _router(engine, clock, lambda request: {"css": "#title-v2"})
    healer = _healer(driver, registry, router, settings, clock)
    driver.break_target(registry, "create.title")
    title = registry.target("create.title")

    assert healer.heal(title).key == "title"
    with pytest.raises(PlatformError, match=r"create\.title.*budget of 1.*agent doctor"):
        healer.heal(title)
    assert len(provider.requests) == 1
    assert json.loads((settings.data_dir / BUDGET_FILE).read_text())["used"] == 1

    clock.advance(days=1)
    assert healer.heal(title).key == "title"


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        ({"css": "#nothing"}, "matches 0 elements"),
        ({"role": "textbox"}, "matches 3 elements"),
        ({"name": None, "css": None}, "could not find it"),
        ({"role": "textbox", "label": "Title"}, "unusable"),
    ],
)
def test_heal_rejects_a_bad_answer_and_keeps_nothing(
    settings, clock, engine, registry, overrides, driver, answer, why
):
    router, _ = _router(engine, clock, [answer])
    driver.break_target(registry, "create.title")
    with pytest.raises(PlatformError, match=why):
        _healer(driver, registry, router, settings, clock).heal(registry.target("create.title"))
    assert not overrides.exists()


def test_heal_without_a_vision_model_spends_nothing(settings, clock, engine, registry, driver):
    router, provider = _router(engine, clock, [], capabilities=frozenset({Capability.REASONING}))
    healer = _healer(driver, registry, router, settings, clock)
    with pytest.raises(PlatformError, match="no vision model"):
        healer.heal(registry.target("create.title"))
    assert healer.budget.used_today() == 0 and provider.requests == []


def test_a_failed_repair_request_is_a_platform_error(settings, clock, engine, registry, driver):
    router, _ = _router(engine, clock, [ProviderError("upstream down")])
    healer = _healer(driver, registry, router, settings, clock)
    with pytest.raises(PlatformError, match="repair request failed"):
        healer.heal(registry.target("create.title"))
    assert healer.budget.used_today() == 1


# ---------------------------------------------------------------- pacing


def test_pacer_sleeps_a_random_time_in_range():
    slept: list[float] = []
    pacer = Pacer((1.5, 4.0), sleep=slept.append, rng=random.Random(7))
    pauses = [pacer.pause() for _ in range(50)]
    assert slept == pauses and all(1.5 <= s <= 4.0 for s in slept)
    assert len(set(slept)) > 40
    assert all(60 <= pacer.key_delay_ms() <= 180 for _ in range(50))


@pytest.mark.parametrize("delay", [(-1.0, 2.0), (3.0, 2.0)])
def test_pacer_rejects_a_bad_range(delay):
    with pytest.raises(ValueError):
        Pacer(delay)


def test_paced_driver_pauses_before_every_action_but_not_reads(tmp_path):
    slept: list[float] = []
    fake = FakeDriver(FakeSite())
    paced = PacedDriver(fake, Pacer((1.0, 2.0), sleep=slept.append, rng=random.Random(1)))

    paced.goto(fb.CREATE_URL)
    title = paced.find({"label": "Title"})[0]
    paced.text_of(title)
    paced.attribute(title, "aria-disabled")
    paced.current_url()
    paced.aria_snapshot()
    paced.has_cookie(fb.SESSION_COOKIE)
    assert len(slept) == 1

    paced.fill(title, "Desk")
    paced.upload(paced.find({"css": "input[type='file']", "attached": True})[0], [tmp_path])
    paced.select_option(paced.find({"label": "Condition"})[0], "New", exact=True)
    paced.click(paced.find({"role": "button", "name": "Next"})[0])
    paced.press("Escape")
    assert len(slept) == 6
    assert fake.kinds() == ["goto", "fill", "upload", "select", "click", "press"]
