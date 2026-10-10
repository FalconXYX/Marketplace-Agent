import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from marketplace_agent.ai.base import (
    Capability,
    ImageInput,
    LLMProvider,
    LLMRequest,
    ProviderError,
    RateLimited,
)
from marketplace_agent.ai.providers.gemini import GeminiProvider
from marketplace_agent.ai.providers.openai_compat import GROQ_BASE_URL, OpenAICompatProvider

KEY = "sk-secret-123"
SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "brand": {"type": ["string", "null"]}},
    "required": ["title", "brand"],
}
MODELS = {
    Capability.VISION: "vision-m",
    Capability.REASONING: "reason-m",
    Capability.CHEAP: "cheap-m",
}


# ---------- Gemini ----------


@dataclass
class FakeUsage:
    prompt_token_count: int | None = 11
    candidates_token_count: int | None = 7
    thoughts_token_count: int | None = 3


@dataclass
class FakeResponse:
    text: str | None = '{"title": "Desk", "brand": null}'
    parsed: Any = None
    usage_metadata: FakeUsage | None = field(default_factory=FakeUsage)


class FakeModels:
    def __init__(self, outcomes: list[Any]):
        self.calls: list[dict[str, Any]] = []
        self._outcomes = outcomes

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes: Any):
        self.models = FakeModels(list(outcomes))


def gemini(*outcomes: Any, models: dict[Capability, str] = MODELS) -> tuple[GeminiProvider, FakeModels]:
    client = FakeClient(*outcomes)
    return GeminiProvider(KEY, models, client=client), client.models


def api_error(
    code: int, status: str, message: str, details: list[dict] | None = None
) -> genai_errors.APIError:
    body = {"error": {"code": code, "status": status, "message": message, "details": details or []}}
    error_class = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return error_class(code, body, None)


def test_gemini_satisfies_protocol():
    provider: LLMProvider = gemini()[0]
    assert provider.name == "gemini"
    assert provider.capabilities == frozenset(MODELS)


def test_gemini_default_client_builds_without_network():
    provider = GeminiProvider(KEY, {Capability.REASONING: "reason-m"})
    assert provider.capabilities == {Capability.REASONING}


def test_gemini_sends_system_schema_and_images():
    provider, models = gemini(FakeResponse(parsed={"title": "Desk", "brand": None}))
    request = LLMRequest(
        system="You write listings.",
        prompt="Draft it.",
        capability=Capability.VISION,
        images=[ImageInput(b"jpeg-bytes"), ImageInput(b"png-bytes", "image/png")],
        json_schema=SCHEMA,
        max_output_tokens=512,
        temperature=0.2,
    )

    response = provider.generate(request)

    call = models.calls[0]
    config: types.GenerateContentConfig = call["config"]
    assert call["model"] == "vision-m"
    assert config.system_instruction == "You write listings."
    assert config.response_json_schema == SCHEMA
    assert config.response_mime_type == "application/json"
    assert (config.temperature, config.max_output_tokens) == (0.2, 512)
    assert config.thinking_config is None
    parts = call["contents"]
    assert [p.inline_data.data for p in parts[:2]] == [b"jpeg-bytes", b"png-bytes"]
    assert [p.inline_data.mime_type for p in parts[:2]] == ["image/jpeg", "image/png"]
    assert parts[2].text == "Draft it."
    assert response.data == {"title": "Desk", "brand": None}
    assert (response.provider, response.model) == ("gemini", "vision-m")
    assert (response.tokens_in, response.tokens_out) == (11, 10)


def test_gemini_parses_text_when_parsed_is_not_a_dict():
    provider, _ = gemini(FakeResponse(text='{"title": "Lamp", "brand": "IKEA"}', parsed=None))
    response = provider.generate(LLMRequest(system="s", prompt="p", json_schema=SCHEMA))
    assert response.data == {"title": "Lamp", "brand": "IKEA"}


def test_gemini_invalid_json_leaves_data_none_for_router():
    provider, _ = gemini(FakeResponse(text="not json {", parsed=None))
    response = provider.generate(LLMRequest(system="s", prompt="p", json_schema=SCHEMA))
    assert response.data is None
    assert response.text == "not json {"


def test_gemini_plain_text_request_has_no_json_config_and_tolerates_missing_usage():
    provider, models = gemini(FakeResponse(text=None, usage_metadata=None))
    response = provider.generate(LLMRequest(system="", prompt="hi"))
    config = models.calls[0]["config"]
    assert config.response_mime_type is None and config.response_json_schema is None
    assert config.system_instruction is None
    assert (response.text, response.data, response.tokens_in, response.tokens_out) == ("", None, 0, 0)


def test_gemini_cheap_uses_cheap_model_with_minimal_thinking():
    provider, models = gemini(FakeResponse())
    provider.generate(LLMRequest(system="s", prompt="p", capability=Capability.CHEAP))
    call = models.calls[0]
    assert call["model"] == "cheap-m"
    assert call["config"].thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_gemini_cheap_falls_back_to_reasoning_model():
    provider, models = gemini(FakeResponse(), models={Capability.REASONING: "reason-m"})
    assert provider.model_for(Capability.CHEAP) == "reason-m"
    provider.generate(LLMRequest(system="s", prompt="p", capability=Capability.CHEAP))
    assert models.calls[0]["model"] == "reason-m"


def test_gemini_unmapped_capability_is_provider_error():
    provider, _ = gemini(models={Capability.REASONING: "reason-m"})
    with pytest.raises(ProviderError):
        provider.model_for(Capability.VISION)


def test_gemini_model_rejecting_minimal_thinking_is_retried_and_remembered():
    rejection = api_error(400, "INVALID_ARGUMENT", "Thinking level MINIMAL is not supported.")
    provider, models = gemini(rejection, FakeResponse(), FakeResponse())
    cheap = LLMRequest(system="s", prompt="p", capability=Capability.CHEAP)

    provider.generate(cheap)
    provider.generate(cheap)

    thinking = [call["config"].thinking_config for call in models.calls]
    assert thinking[0] is not None
    assert thinking[1:] == [None, None]


def test_gemini_other_bad_request_is_not_retried():
    provider, models = gemini(api_error(400, "INVALID_ARGUMENT", "schema too deep"))
    with pytest.raises(ProviderError, match="400"):
        provider.generate(LLMRequest(system="s", prompt="p", capability=Capability.CHEAP))
    assert len(models.calls) == 1


def test_gemini_429_is_rate_limited_with_retry_after_and_no_key():
    details = [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34.5s"},
    ]
    error = api_error(429, "RESOURCE_EXHAUSTED", f"quota exceeded for key {KEY}", details)
    provider, _ = gemini(error)

    with pytest.raises(RateLimited) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert info.value.retry_after == 34.5
    assert info.value.daily is False
    assert KEY not in str(info.value)
    assert info.value.__suppress_context__


def test_gemini_429_without_retry_info_has_none_retry_after():
    provider, _ = gemini(api_error(429, "RESOURCE_EXHAUSTED", "slow down"))
    with pytest.raises(RateLimited) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))
    assert info.value.retry_after is None


def test_gemini_server_error_is_provider_error_without_key():
    provider, _ = gemini(api_error(503, "UNAVAILABLE", f"overloaded {KEY}"))
    with pytest.raises(ProviderError) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))
    assert not isinstance(info.value, RateLimited)
    assert "503" in str(info.value) and KEY not in str(info.value)


def test_gemini_network_error_is_provider_error():
    provider, _ = gemini(httpx.ConnectError("connection refused"))
    with pytest.raises(ProviderError, match="ConnectError"):
        provider.generate(LLMRequest(system="s", prompt="p"))


# ---------- OpenAI-compatible ----------


def completion(content: str, prompt_tokens: int = 20, completion_tokens: int = 5) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def compat(handler) -> tuple[OpenAICompatProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    provider = OpenAICompatProvider(
        "groq", GROQ_BASE_URL, KEY, "llama-x", transport=httpx.MockTransport(record)
    )
    return provider, seen


def test_compat_satisfies_protocol():
    provider: LLMProvider = compat(lambda r: httpx.Response(200, json=completion("x")))[0]
    assert provider.capabilities == {Capability.REASONING, Capability.CHEAP}
    assert provider.model_for(Capability.CHEAP) == "llama-x"


def test_compat_json_request_shape_and_fenced_reply():
    reply = '```json\n{"title": "Desk", "brand": null}\n```'
    provider, seen = compat(lambda r: httpx.Response(200, json=completion(reply)))

    response = provider.generate(
        LLMRequest(system="Be brief.", prompt="Draft it.", json_schema=SCHEMA, max_output_tokens=300)
    )

    request = seen[0]
    body = json.loads(request.content)
    assert request.method == "POST"
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert body["model"] == "llama-x"
    assert body["max_tokens"] == 300 and body["temperature"] == 0.4
    assert body["response_format"] == {"type": "json_object"}
    system, user = body["messages"]
    assert system["role"] == "system" and system["content"].startswith("Be brief.")
    assert "Respond with a single JSON object matching this JSON Schema:" in system["content"]
    assert json.dumps(SCHEMA) in system["content"]
    assert user == {"role": "user", "content": "Draft it."}
    assert response.data == {"title": "Desk", "brand": None}
    assert (response.provider, response.model) == ("groq", "llama-x")
    assert (response.tokens_in, response.tokens_out) == (20, 5)


def test_compat_plain_text_request():
    provider, seen = compat(lambda r: httpx.Response(200, json=completion("Still available.")))
    response = provider.generate(LLMRequest(system="s", prompt="p"))
    body = json.loads(seen[0].content)
    assert "response_format" not in body
    assert body["messages"][0]["content"] == "s"
    assert (response.text, response.data) == ("Still available.", None)


def test_compat_unparseable_json_reply_leaves_data_none():
    provider, _ = compat(lambda r: httpx.Response(200, json=completion("no json here")))
    assert provider.generate(LLMRequest(system="s", prompt="p", json_schema=SCHEMA)).data is None


def test_compat_429_uses_retry_after_header():
    provider, _ = compat(lambda r: httpx.Response(429, headers={"Retry-After": "7"}, json={}))
    with pytest.raises(RateLimited) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))
    assert info.value.retry_after == 7.0


def test_compat_429_without_header_has_none_retry_after():
    provider, _ = compat(lambda r: httpx.Response(429, text="slow down"))
    with pytest.raises(RateLimited) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))
    assert info.value.retry_after is None


def test_compat_500_is_provider_error_with_excerpt_and_no_key():
    body = f'{{"error": "upstream exploded", "echo": "{KEY}"}}' + "x" * 5000
    provider, _ = compat(lambda r: httpx.Response(500, text=body))
    with pytest.raises(ProviderError) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))
    message = str(info.value)
    assert not isinstance(info.value, RateLimited)
    assert "500" in message and "upstream exploded" in message
    assert KEY not in message and "Bearer" not in message
    assert len(message) < 400


def test_compat_timeout_is_provider_error():
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider, _ = compat(timeout)
    with pytest.raises(ProviderError, match="timed out"):
        provider.generate(LLMRequest(system="s", prompt="p"))


def test_compat_network_error_is_provider_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider, _ = compat(refuse)
    with pytest.raises(ProviderError, match="ConnectError"):
        provider.generate(LLMRequest(system="s", prompt="p"))


def test_compat_error_body_with_200_is_provider_error():
    provider, _ = compat(lambda r: httpx.Response(200, json={"error": {"message": "no choices"}}))
    with pytest.raises(ProviderError, match="no choices"):
        provider.generate(LLMRequest(system="s", prompt="p"))


def test_compat_rejects_images_without_calling():
    provider, seen = compat(lambda r: httpx.Response(200, json=completion("x")))
    with pytest.raises(ProviderError, match="images not supported"):
        provider.generate(LLMRequest(system="s", prompt="p", images=[ImageInput(b"jpeg")]))
    assert seen == []


# ---------- daily quota, dead models, odd delays ----------


def quota_failure(quota_id: str, retry_delay: str = "34s") -> genai_errors.APIError:
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaMetric": "generate_content_free_tier_requests", "quotaId": quota_id}],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
    ]
    return api_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota", details)


@pytest.mark.parametrize(
    ("quota_id", "daily"),
    [
        ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", True),
        ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", False),
    ],
)
def test_gemini_429_marks_per_day_quota(quota_id, daily):
    provider, _ = gemini(quota_failure(quota_id))

    with pytest.raises(RateLimited) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert (info.value.daily, info.value.retry_after) == (daily, 34.0)


@pytest.mark.parametrize("delay", ["infs", "nans", "-infs"])
def test_gemini_non_finite_retry_delay_is_none(delay):
    provider, _ = gemini(quota_failure("PerMinute", retry_delay=delay))

    with pytest.raises(RateLimited) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert info.value.retry_after is None


@pytest.mark.parametrize(("code", "unavailable"), [(404, True), (400, False), (503, False)])
def test_gemini_unknown_model_is_marked_unavailable(code, unavailable):
    status = {404: "NOT_FOUND", 400: "INVALID_ARGUMENT", 503: "UNAVAILABLE"}[code]
    provider, _ = gemini(api_error(code, status, "models/gemini-9 is not found"))

    with pytest.raises(ProviderError) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert info.value.model_unavailable is unavailable


@pytest.mark.parametrize(
    ("status", "body", "unavailable"),
    [
        (400, '{"error": {"message": "x:free is not a valid model ID", "code": 400}}', True),
        (404, '{"error": {"message": "No endpoints found for x:free.", "code": 404}}', True),
        (
            404,
            '{"error": {"message": "The model `x` does not exist or you do not have access to it.",'
            ' "code": "model_not_found"}}',
            True,
        ),
        (400, '{"error": {"message": "max_tokens is too large"}}', False),
        (500, '{"error": {"message": "model not a valid model, upstream exploded"}}', False),
    ],
)
def test_compat_unknown_model_is_marked_unavailable(status, body, unavailable):
    provider, _ = compat(lambda r: httpx.Response(status, text=body))

    with pytest.raises(ProviderError) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert not isinstance(info.value, RateLimited)
    assert info.value.model_unavailable is unavailable


@pytest.mark.parametrize(
    ("header", "expected"), [("nan", None), ("inf", None), ("1e400", None), ("-3", 0.0)]
)
def test_compat_retry_after_must_be_finite(header, expected):
    provider, _ = compat(lambda r: httpx.Response(429, headers={"Retry-After": header}))

    with pytest.raises(RateLimited) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert info.value.retry_after == expected


# ---------- API keys with stray whitespace or control characters ----------


def h11_like(request: httpx.Request) -> httpx.Response:
    """Rejects header values the way h11 does, quoting the raw bytes in the error."""
    raw = next(value for name, value in request.headers.raw if name.lower() == b"authorization")
    if raw != raw.strip() or any(byte < 0x20 or byte == 0x7F for byte in raw):
        raise httpx.LocalProtocolError(f"Illegal header value {raw!r}")
    return httpx.Response(200, json=completion("ok"))


@pytest.mark.parametrize("suffix", ["\r", "\n", "\t", "\r\n", " "])
def test_compat_key_is_stripped_before_use(suffix):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return h11_like(request)

    provider = OpenAICompatProvider(
        "groq", GROQ_BASE_URL, KEY + suffix, "m", transport=httpx.MockTransport(record)
    )

    assert provider.generate(LLMRequest(system="s", prompt="p")).text == "ok"
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("key", [KEY[:5] + "\x00" + KEY[5:], KEY[:5] + "\r" + KEY[5:], "  "])
def test_compat_unusable_key_fails_without_sending_or_leaking(key):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return h11_like(request)

    provider = OpenAICompatProvider(
        "groq", GROQ_BASE_URL, key, "m", transport=httpx.MockTransport(record)
    )

    with pytest.raises(ProviderError, match="API key") as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert seen == []
    assert KEY[:5] not in str(info.value) and KEY[6:] not in str(info.value)


def test_gemini_key_is_stripped_so_redaction_finds_it():
    """The SDK strips the key it sends, so error text quotes the stripped key."""
    client = FakeClient(api_error(400, "INVALID_ARGUMENT", f"API key not valid: {KEY}"))
    provider = GeminiProvider(KEY + "\r\n", MODELS, client=client)

    with pytest.raises(ProviderError) as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert KEY not in str(info.value)


def test_gemini_unusable_key_fails_without_calling():
    client = FakeClient(FakeResponse())
    provider = GeminiProvider(KEY[:5] + "\x00" + KEY[5:], MODELS, client=client)

    with pytest.raises(ProviderError, match="API key") as info:
        provider.generate(LLMRequest(system="s", prompt="p"))

    assert client.models.calls == []
    assert KEY[:5] not in str(info.value)
