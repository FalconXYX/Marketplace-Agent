import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr

from marketplace_agent.ai.base import (
    Capability,
    LLMRequest,
    LLMResponse,
    ProviderError,
    QuotaExhausted,
    QuotaState,
    RateLimited,
)
from marketplace_agent.ai.factory import build_router
from marketplace_agent.ai.providers.openai_compat import GROQ_BASE_URL, OpenAICompatProvider
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter, extract_json
from marketplace_agent.config import ProviderLimit, Settings

TEXT = frozenset({Capability.REASONING, Capability.CHEAP})
SCHEMA = {"type": "object"}


def rate_limited(retry_after: float | None = None) -> RateLimited:
    exc = RateLimited("429")
    exc.retry_after = retry_after  # type: ignore[attr-defined]
    return exc


def daily_limited() -> RateLimited:
    """Gemini's per-day 429 still carries a short RetryInfo delay."""
    exc = rate_limited(34)
    exc.daily = True  # type: ignore[attr-defined]
    return exc


def model_unavailable() -> ProviderError:
    exc = ProviderError("openrouter: HTTP 400: x:free is not a valid model ID")
    exc.model_unavailable = True  # type: ignore[attr-defined]
    return exc


class TwoModelProvider(ScriptedProvider):
    """Like Gemini: a cheap model apart from the reasoning one."""

    def model_for(self, capability: Capability) -> str:
        return "lite" if capability is Capability.CHEAP else "flash"


@pytest.fixture
def quota(engine, clock) -> QuotaTracker:
    limits = {name: ProviderLimit(rpm=100, rpd=100) for name in ("a", "b", "c")}
    return QuotaTracker(engine, clock, limits)


def router_of(quota: QuotaTracker, *providers: ScriptedProvider) -> ProviderRouter:
    return ProviderRouter(list(providers), quota)


# extract_json


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure, here it is:\n```\n{"a": {"b": [1, 2]}}\n```\nHope that helps {x}', {"a": {"b": [1, 2]}}),
        ('{"s": "brace } and { inside"}', {"s": "brace } and { inside"}),
        ('use {curly} braces: {"a": 1}', {"a": 1}),
        ('{"a": 1,} then {"b": 2}', {"b": 2}),
        ('{"first": 1} {"second": 2}', {"first": 1}),
        ('{"a": [{"tool": "x"}, {"tool": "y"', None),  # truncated: never return a nested object
        ("no json here", None),
        ("", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


# ScriptedProvider


def test_scripted_replays_entries_in_order():
    calls: list[str] = []

    def echo(request: LLMRequest) -> str:
        calls.append(request.prompt)
        return '{"echo": true}'

    provider = ScriptedProvider([{"a": 1}, "plain", "x ```{\"b\": 2}```", ValueError("boom"), echo])
    first = provider.generate(LLMRequest(system="s", prompt="1"))
    assert (first.data, first.text) == ({"a": 1}, '{"a": 1}')
    assert (first.provider, first.model) == ("scripted", "scripted-1")
    assert provider.generate(LLMRequest(system="s", prompt="2")).data is None
    assert provider.generate(LLMRequest(system="s", prompt="3", json_schema=SCHEMA)).data == {"b": 2}
    with pytest.raises(ValueError):
        provider.generate(LLMRequest(system="s", prompt="4"))
    assert provider.generate(LLMRequest(system="s", prompt="5", json_schema=SCHEMA)).data == {"echo": True}
    with pytest.raises(ProviderError, match="script exhausted"):
        provider.generate(LLMRequest(system="s", prompt="6"))
    assert [r.prompt for r in provider.requests] == ["1", "2", "3", "4", "5", "6"]
    assert calls == ["5"]


def test_scripted_callable_script_never_exhausts():
    provider = ScriptedProvider(lambda req: {"purpose": req.purpose}, name="sim", model="sim-2")
    for purpose in ("wake", "draft", "heal"):
        response = provider.generate(LLMRequest(system="s", prompt="p", purpose=purpose))
        assert response.data == {"purpose": purpose}
    assert provider.model_for(Capability.VISION) == "sim-2"
    assert len(provider.requests) == 3


# ProviderRouter


def test_falls_back_in_order_after_provider_error(quota):
    a = ScriptedProvider([ProviderError("500")], name="a", capabilities=TEXT)
    b = ScriptedProvider(["from b"], name="b", capabilities=TEXT)
    c = ScriptedProvider(["from c"], name="c", capabilities=TEXT)
    response = router_of(quota, a, b, c).generate(LLMRequest(system="s", prompt="p"))
    assert (response.text, response.provider) == ("from b", "b")
    assert (len(a.requests), len(b.requests), len(c.requests)) == (1, 1, 0)


def test_cheap_prefers_native_then_falls_back_to_reasoning(quota):
    reasoning = ScriptedProvider(["smart"], name="a", capabilities=frozenset({Capability.REASONING}))
    cheap = ScriptedProvider([ProviderError("down")], name="b", capabilities=frozenset({Capability.CHEAP}))
    router = router_of(quota, reasoning, cheap)

    response = router.generate(LLMRequest(system="s", prompt="p", capability=Capability.CHEAP))
    assert response.text == "smart"
    assert len(cheap.requests) == 1
    assert reasoning.requests[0].capability is Capability.REASONING


def test_capability_coverage(quota):
    router = router_of(quota, ScriptedProvider([], name="a", capabilities=frozenset({Capability.REASONING})))
    assert router.has_capability(Capability.CHEAP)
    assert not router.has_capability(Capability.VISION)
    assert router.state(Capability.VISION) is QuotaState.EXHAUSTED
    with pytest.raises(QuotaExhausted):
        router.generate(LLMRequest(system="s", prompt="p", capability=Capability.VISION))


def test_rate_limited_moves_on_and_cools_provider(quota, clock):
    a = ScriptedProvider([rate_limited(10), "a again"], name="a", capabilities=TEXT)
    b = ScriptedProvider(["b1", "b2"], name="b", capabilities=TEXT)
    router = router_of(quota, a, b)

    assert router.generate(LLMRequest(system="s", prompt="p")).text == "b1"
    assert not quota.can_request("a")
    assert router.generate(LLMRequest(system="s", prompt="p")).text == "b2"
    assert len(a.requests) == 1

    clock.advance(seconds=10)
    assert router.generate(LLMRequest(system="s", prompt="p")).text == "a again"


def test_rate_limited_without_retry_after_cools_for_a_minute(quota, clock):
    router = router_of(quota, ScriptedProvider([RateLimited("429")], name="a", capabilities=TEXT))
    with pytest.raises(QuotaExhausted):
        router.generate(LLMRequest(system="s", prompt="p"))
    clock.advance(timedelta(seconds=59))
    assert not quota.can_request("a")
    clock.advance(timedelta(seconds=1))
    assert quota.can_request("a")


def test_invalid_json_moves_on_and_still_counts_usage(quota):
    a = ScriptedProvider(["I cannot answer that"], name="a", capabilities=TEXT)
    b = ScriptedProvider([{"ok": True}], name="b", capabilities=TEXT)
    response = router_of(quota, a, b).generate(LLMRequest(system="s", prompt="p", json_schema=SCHEMA))
    assert (response.provider, response.data) == ("b", {"ok": True})
    assert quota.requests_today("a") == 1
    assert quota.requests_today("b") == 1


def test_invalid_json_everywhere_reraises_provider_error(quota):
    router = router_of(quota, ScriptedProvider(["nope"], name="a", capabilities=TEXT))
    with pytest.raises(ProviderError, match="invalid JSON") as info:
        router.generate(LLMRequest(system="s", prompt="p", json_schema=SCHEMA))
    assert not isinstance(info.value, QuotaExhausted)


class TextOnlyProvider:
    """Returns raw text and token counts, leaving JSON parsing to the router."""

    name = "a"
    capabilities = TEXT

    def model_for(self, capability: Capability) -> str:
        return "text-1"

    def generate(self, request: LLMRequest) -> LLMResponse:
        text = 'Plan:\n```json\n{"actions": []}\n```'
        return LLMResponse(text=text, provider=self.name, model="text-1", tokens_in=1200, tokens_out=80)


def test_router_parses_json_and_records_usage_once(quota):
    router = ProviderRouter([TextOnlyProvider()], quota)
    response = router.generate(LLMRequest(system="s", prompt="p", json_schema=SCHEMA))
    assert response.data == {"actions": []}

    (usage,) = quota.usage_today()
    assert (usage.provider, usage.model, usage.requests) == ("a", "text-1", 1)
    assert (usage.tokens_in, usage.tokens_out) == (1200, 80)


def test_failed_calls_are_not_recorded(quota):
    a = ScriptedProvider([ProviderError("500")], name="a", capabilities=TEXT)
    b = ScriptedProvider([rate_limited(5)], name="b", capabilities=TEXT)
    c = ScriptedProvider(["ok"], name="c", capabilities=TEXT)
    router_of(quota, a, b, c).generate(LLMRequest(system="s", prompt="p"))
    assert [(u.provider, u.requests) for u in quota.usage_today()] == [("c", 1)]


def test_all_skipped_raises_quota_exhausted(engine, clock):
    quota = QuotaTracker(engine, clock, {name: ProviderLimit(rpm=100, rpd=1) for name in ("a", "b")})
    quota.record("a", "m")
    quota.mark_rate_limited("b")
    a = ScriptedProvider(["unused"], name="a", capabilities=TEXT)
    b = ScriptedProvider(["unused"], name="b", capabilities=TEXT)
    with pytest.raises(QuotaExhausted):
        router_of(quota, a, b).generate(LLMRequest(system="s", prompt="p"))
    assert a.requests == b.requests == []


def test_all_rate_limited_raises_quota_exhausted(quota):
    a = ScriptedProvider([rate_limited()], name="a", capabilities=TEXT)
    b = ScriptedProvider([rate_limited(30)], name="b", capabilities=TEXT)
    with pytest.raises(QuotaExhausted):
        router_of(quota, a, b).generate(LLMRequest(system="s", prompt="p"))


def test_mixed_failure_reraises_last_provider_error(quota):
    error = ProviderError("bad gateway")
    a = ScriptedProvider([error], name="a", capabilities=TEXT)
    b = ScriptedProvider([rate_limited()], name="b", capabilities=TEXT)
    with pytest.raises(ProviderError) as info:
        router_of(quota, a, b).generate(LLMRequest(system="s", prompt="p"))
    assert info.value is error


def test_no_providers_raises_quota_exhausted(quota):
    router = ProviderRouter([], quota)
    assert router.state() is QuotaState.EXHAUSTED
    with pytest.raises(QuotaExhausted):
        router.generate(LLMRequest(system="s", prompt="p"))


def test_router_state_reflects_quota(engine, clock):
    quota = QuotaTracker(engine, clock, {"a": ProviderLimit(rpm=100, rpd=4)})
    router = router_of(quota, ScriptedProvider(lambda req: "ok", name="a", capabilities=TEXT))
    states = []
    for _ in range(4):
        states.append(router.state())
        router.generate(LLMRequest(system="s", prompt="p"))
    states.append(router.state(Capability.CHEAP))
    assert states == [QuotaState.HEALTHY] * 4 + [QuotaState.EXHAUSTED]
    assert quota.state(["a"]) is QuotaState.EXHAUSTED


def test_logs_metadata_but_never_prompt(quota, caplog):
    router = router_of(quota, ScriptedProvider(["ok"], name="a", capabilities=TEXT))
    with caplog.at_level(logging.INFO, logger="marketplace_agent.ai"):
        router.generate(LLMRequest(system="SECRET SYSTEM", prompt="SECRET PROMPT", purpose="wake"))
    assert "provider=a" in caplog.text and "purpose=wake" in caplog.text and "scripted-1" in caplog.text
    assert "SECRET" not in caplog.text


# build_router


def without_keys(settings):
    keys = ("gemini_api_key", "groq_api_key", "openrouter_api_key")
    return settings.model_copy(update=dict.fromkeys(keys))


def test_build_router_without_keys_has_no_providers(settings, engine, clock):
    router = build_router(without_keys(settings), engine, clock)
    assert router.providers == []
    assert router.state() is QuotaState.EXHAUSTED
    assert router.quota.limits == settings.provider_limits
    with pytest.raises(QuotaExhausted):
        router.generate(LLMRequest(system="s", prompt="p"))


def test_build_router_with_all_keys(settings, engine, clock):
    pytest.importorskip("marketplace_agent.ai.providers.gemini")
    pytest.importorskip("marketplace_agent.ai.providers.openai_compat")
    keyed = settings.model_copy(update={
        "gemini_api_key": SecretStr("g-key"),
        "groq_api_key": SecretStr("q-key"),
        "openrouter_api_key": SecretStr("o-key"),
    })
    router = build_router(keyed, engine, clock)
    gemini, groq, openrouter = router.providers
    assert [p.name for p in router.providers] == ["gemini", "groq", "openrouter"]
    assert gemini.capabilities == frozenset(Capability)
    assert gemini.model_for(Capability.VISION) == keyed.vision_model
    assert gemini.model_for(Capability.CHEAP) == keyed.cheap_model
    assert groq.capabilities == openrouter.capabilities == TEXT
    assert groq.model_for(Capability.REASONING) == keyed.groq_model
    assert openrouter.model_for(Capability.CHEAP) == keyed.openrouter_model
    assert router.state() is QuotaState.HEALTHY


def test_build_router_groq_only(settings, engine, clock):
    pytest.importorskip("marketplace_agent.ai.providers.openai_compat")
    keyed = without_keys(settings).model_copy(update={"groq_api_key": SecretStr("q-key")})
    router = build_router(keyed, engine, clock)
    assert [p.name for p in router.providers] == ["groq"]
    assert router.has_capability(Capability.CHEAP)
    assert not router.has_capability(Capability.VISION)


# daily quota and dead models


def test_daily_rate_limit_reads_exhausted_until_pacific_midnight(engine, clock):
    quota = QuotaTracker(engine, clock, {"gemini": ProviderLimit(rpm=100, rpd=250)})
    clock.set(datetime(2026, 10, 9, 18, 0, tzinfo=UTC))  # 11:00 PDT
    gemini = ScriptedProvider([daily_limited(), "after reset"], name="gemini", capabilities=TEXT)
    router = router_of(quota, gemini)

    with pytest.raises(QuotaExhausted):
        router.generate(LLMRequest(system="s", prompt="p"))
    for _ in range(3):
        clock.advance(minutes=30)  # far past the 34s RetryInfo
        assert router.state() is QuotaState.EXHAUSTED
        with pytest.raises(QuotaExhausted):
            router.generate(LLMRequest(system="s", prompt="p"))
    assert len(gemini.requests) == 1

    clock.set(datetime(2026, 10, 10, 7, 0, tzinfo=UTC))  # midnight PDT
    assert router.state() is QuotaState.HEALTHY
    assert router.generate(LLMRequest(system="s", prompt="p")).text == "after reset"


def test_daily_rate_limit_on_one_model_leaves_the_others(engine, clock):
    quota = QuotaTracker(engine, clock, {"gemini": ProviderLimit(rpm=100, rpd=250)})
    gemini = TwoModelProvider([daily_limited(), "cheap ok"], name="gemini")
    router = router_of(quota, gemini)

    with pytest.raises(QuotaExhausted):
        router.generate(LLMRequest(system="s", prompt="p", capability=Capability.REASONING))

    assert router.state(Capability.REASONING) is QuotaState.EXHAUSTED
    assert router.state(Capability.CHEAP) is QuotaState.HEALTHY
    cheap = router.generate(LLMRequest(system="s", prompt="p", capability=Capability.CHEAP))
    assert (cheap.text, cheap.model) == ("cheap ok", "lite")


def test_unavailable_model_is_not_retried_until_its_day_ends(engine, clock):
    limits = {"groq": ProviderLimit(rpm=30, rpd=1000), "openrouter": ProviderLimit(rpm=20, rpd=50)}
    quota = QuotaTracker(engine, clock, limits)
    groq = ScriptedProvider(lambda request: rate_limited(), name="groq", capabilities=TEXT)
    dead = ScriptedProvider([model_unavailable(), "fixed"], name="openrouter", capabilities=TEXT)
    router = router_of(quota, groq, dead)

    with pytest.raises(ProviderError, match="not a valid model"):
        router.generate(LLMRequest(system="s", prompt="p"))
    for _ in range(4):
        clock.advance(minutes=2)  # groq's cooldown has passed; it 429s again
        with pytest.raises(QuotaExhausted):
            router.generate(LLMRequest(system="s", prompt="p"))
        assert router.state() is QuotaState.EXHAUSTED
    assert (len(groq.requests), len(dead.requests)) == (5, 1)

    clock.set(datetime(2026, 10, 2, 0, 0, tzinfo=UTC))  # OpenRouter's day ends at midnight UTC
    assert router.generate(LLMRequest(system="s", prompt="p")).text == "fixed"


def test_other_provider_errors_do_not_suspend(quota):
    a = ScriptedProvider([ProviderError("500"), "a again"], name="a", capabilities=TEXT)
    b = ScriptedProvider(["from b"], name="b", capabilities=TEXT)
    router = router_of(quota, a, b)

    assert router.generate(LLMRequest(system="s", prompt="p")).text == "from b"
    assert router.generate(LLMRequest(system="s", prompt="p")).text == "a again"


def test_default_openrouter_model_is_not_the_retired_llama_free_model():
    """meta-llama/llama-3.3-70b-instruct:free has no endpoints on OpenRouter (2026-10-09).
    tests/test_live_providers.py checks the configured model against the live catalog."""
    default = Settings.model_fields["openrouter_model"].default

    assert default.endswith(":free")
    assert default != "meta-llama/llama-3.3-70b-instruct:free"


# Retry-After values a provider should never send


def groq_429(retry_after: str) -> OpenAICompatProvider:
    response = httpx.Response(429, headers={"retry-after": retry_after})
    transport = httpx.MockTransport(lambda request: response)
    return OpenAICompatProvider("groq", GROQ_BASE_URL, "key", "m", transport=transport)


@pytest.mark.parametrize("retry_after", ["nan", "inf", "-inf", "1e12", "1e400"])
def test_odd_retry_after_still_falls_back(engine, clock, retry_after):
    quota = QuotaTracker(engine, clock, {})
    backup = ScriptedProvider(["from backup"], name="backup", capabilities=TEXT)
    router = router_of(quota, groq_429(retry_after), backup)

    assert router.generate(LLMRequest(system="s", prompt="p")).text == "from backup"
    assert not quota.can_request("groq")


def test_retry_after_zero_does_not_cool_down(engine, clock):
    quota = QuotaTracker(engine, clock, {})
    backup = ScriptedProvider(["from backup"], name="backup", capabilities=TEXT)
    router = router_of(quota, groq_429("0"), backup)

    assert router.generate(LLMRequest(system="s", prompt="p")).text == "from backup"
    assert quota.can_request("groq")
