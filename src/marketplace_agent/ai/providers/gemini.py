"""Gemini through the google-genai SDK. The only provider with vision."""
import json
import math
from collections.abc import Mapping
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from marketplace_agent.ai.base import (
    Capability,
    LLMRequest,
    LLMResponse,
    ProviderError,
    RateLimited,
)
from marketplace_agent.ai.providers import is_header_safe

_TIMEOUT_MS = 60_000  # the SDK default is no timeout, which would stall the tick loop
_EXCERPT_CHARS = 300
_RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"
_QUOTA_FAILURE_TYPE = "type.googleapis.com/google.rpc.QuotaFailure"
_SDK_ERRORS = (genai_errors.APIError, genai_errors.UnknownApiResponseError, httpx.HTTPError)


class GeminiProvider:
    name = "gemini"

    def __init__(
        self, api_key: str, models: Mapping[Capability, str], *, client: Any | None = None
    ):
        # The SDK strips the key it sends, so redacting must look for the stripped key too.
        self._api_key = api_key.strip()
        self._key_ok = is_header_safe(self._api_key)
        self._models = dict(models)
        self.capabilities = frozenset(self._models)
        if client is None and self._key_ok:
            client = genai.Client(
                api_key=self._api_key, http_options=types.HttpOptions(timeout=_TIMEOUT_MS)
            )
        self._client = client
        self._rejects_minimal_thinking: set[str] = set()

    def model_for(self, capability: Capability) -> str:
        if capability in self._models:
            return self._models[capability]
        if capability is Capability.CHEAP and Capability.REASONING in self._models:
            return self._models[Capability.REASONING]
        raise ProviderError(f"gemini has no model for {capability}")

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._key_ok:
            raise ProviderError("gemini: API key is empty or has spaces or control characters")
        model = self.model_for(request.capability)
        try:
            response = self._send(model, request)
        except _SDK_ERRORS as exc:
            raise self._translate(exc) from None
        return _to_response(model, request, response)

    def _send(self, model: str, request: LLMRequest) -> Any:
        """CHEAP asks for minimal thinking. A model that rejects it is remembered and retried without."""
        minimize = (
            request.capability is Capability.CHEAP and model not in self._rejects_minimal_thinking
        )
        try:
            return self._client.models.generate_content(
                model=model, contents=_contents(request), config=_config(request, minimize)
            )
        except genai_errors.ClientError as exc:
            if not (minimize and _is_thinking_rejection(exc)):
                raise
            self._rejects_minimal_thinking.add(model)
            return self._send(model, request)

    def _translate(self, exc: Exception) -> ProviderError:
        if isinstance(exc, genai_errors.APIError):
            detail = self._redact(exc.message or "")
            if exc.code == 429 or exc.status == "RESOURCE_EXHAUSTED":
                return _rate_limited(
                    f"gemini rate limited: {detail}", _retry_after(exc), daily=_is_daily(exc)
                )
            error = ProviderError(f"gemini HTTP {exc.code} {exc.status}: {detail}")
            # generate_content answers 404 only for a model name it does not know.
            error.model_unavailable = exc.code == 404  # type: ignore[attr-defined]
            return error
        return ProviderError(f"gemini {type(exc).__name__}: {self._redact(str(exc))}")

    def _redact(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return text[:_EXCERPT_CHARS]


def _contents(request: LLMRequest) -> list[types.Part]:
    images = [types.Part.from_bytes(data=i.data, mime_type=i.mime_type) for i in request.images]
    return [*images, types.Part.from_text(text=request.prompt)]


def _config(request: LLMRequest, minimize_thinking: bool) -> types.GenerateContentConfig:
    wants_json = request.json_schema is not None
    thinking = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    return types.GenerateContentConfig(
        system_instruction=request.system or None,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        response_mime_type="application/json" if wants_json else None,
        response_json_schema=request.json_schema,
        thinking_config=thinking if minimize_thinking else None,
    )


def _to_response(model: str, request: LLMRequest, response: Any) -> LLMResponse:
    text = response.text or ""
    usage = response.usage_metadata
    return LLMResponse(
        text=text,
        provider=GeminiProvider.name,
        model=model,
        data=_parse_data(response.parsed, text) if request.json_schema is not None else None,
        tokens_in=_count(usage, "prompt_token_count"),
        tokens_out=_count(usage, "candidates_token_count") + _count(usage, "thoughts_token_count"),
    )


def _parse_data(parsed: Any, text: str) -> dict[str, Any] | None:
    if isinstance(parsed, dict):
        return parsed
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _count(usage: Any, field: str) -> int:
    return getattr(usage, field, None) or 0


def _is_thinking_rejection(exc: genai_errors.ClientError) -> bool:
    return exc.code == 400 and "thinking" in (exc.message or "").lower()


def _error_details(exc: genai_errors.APIError) -> list[dict[str, Any]]:
    body = exc.details if isinstance(exc.details, dict) else {}
    error = body.get("error", body)
    details = error.get("details", []) if isinstance(error, dict) else []
    return [d for d in details if isinstance(d, dict)] if isinstance(details, list) else []


def _retry_after(exc: genai_errors.APIError) -> float | None:
    """Seconds from the google.rpc.RetryInfo detail, e.g. {"retryDelay": "34s"}."""
    for detail in _error_details(exc):
        if detail.get("@type") == _RETRY_INFO_TYPE:
            return _duration_seconds(detail.get("retryDelay"))
    return None


def _is_daily(exc: genai_errors.APIError) -> bool:
    """A QuotaFailure on a per-day quota, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier.

    Its RetryInfo still says something like "34s", but the quota returns at midnight Pacific.
    """
    for detail in _error_details(exc):
        violations = detail.get("violations") if detail.get("@type") == _QUOTA_FAILURE_TYPE else []
        for violation in violations if isinstance(violations, list) else []:
            if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def _duration_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.endswith("s"):
        return None
    try:
        seconds = float(value[:-1])
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) else None


def _rate_limited(message: str, retry_after: float | None, *, daily: bool) -> RateLimited:
    exc = RateLimited(message)
    # base.RateLimited is a frozen contract without these fields; the router reads them via getattr.
    exc.retry_after = retry_after  # type: ignore[attr-defined]
    exc.daily = daily  # type: ignore[attr-defined]
    return exc
