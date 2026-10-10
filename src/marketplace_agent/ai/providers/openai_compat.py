"""OpenAI-compatible chat completions (Groq, OpenRouter). Text only."""
import json
import math
from typing import Any

import httpx

from marketplace_agent.ai.base import (
    Capability,
    LLMRequest,
    LLMResponse,
    ProviderError,
    RateLimited,
)
from marketplace_agent.ai.providers import is_header_safe
from marketplace_agent.ai.router import extract_json

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_JSON_INSTRUCTION = "Respond with a single JSON object matching this JSON Schema:"
_EXCERPT_CHARS = 300
# Groq: 404 "model `x` does not exist". OpenRouter: 400 "x is not a valid model ID",
# 404 "No endpoints found for x".
_UNKNOWN_MODEL_MARKERS = ("model_not_found", "not a valid model", "no endpoints found")


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        capabilities: frozenset[Capability] = frozenset({Capability.REASONING, Capability.CHEAP}),
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ):
        self.name = name
        self.model = model
        self.capabilities = capabilities
        self._api_key = api_key.strip()  # a stray "\r" from a CRLF .env would otherwise be sent
        self._key_ok = is_header_safe(self._api_key)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._key_ok else {}
        self._http = httpx.Client(
            base_url=base_url, headers=headers, transport=transport, timeout=timeout
        )

    def model_for(self, capability: Capability) -> str:
        return self.model

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._key_ok:
            raise ProviderError(
                f"{self.name}: API key is empty or has spaces or control characters"
            )
        if request.images:
            raise ProviderError(f"{self.name}: images not supported")
        body = self._post(self._payload(request))
        return self._to_response(request, body)

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        system = request.system
        payload: dict[str, Any] = {
            "model": self.model_for(request.capability),
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.json_schema is not None:
            schema = f"{_JSON_INSTRUCTION}\n{json.dumps(request.json_schema)}"
            system = f"{system}\n\n{schema}" if system else schema
            payload["response_format"] = {"type": "json_object"}
        payload["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": request.prompt},
        ]
        return payload

    def _post(self, payload: dict[str, Any]) -> Any:
        try:
            response = self._http.post("/chat/completions", json=payload)
        except httpx.TimeoutException:
            raise ProviderError(f"{self.name}: request timed out") from None
        except httpx.HTTPError as exc:
            detail = f"{type(exc).__name__}: {self._redact(str(exc))}"
            raise ProviderError(f"{self.name}: {detail}") from None
        if response.status_code == 429:
            raise _rate_limited(
                f"{self.name}: rate limited", _seconds(response.headers.get("retry-after"))
            )
        if not response.is_success:
            error = ProviderError(
                f"{self.name}: HTTP {response.status_code}: {self._redact(response.text)}"
            )
            # base.ProviderError is frozen; the router reads this via getattr.
            error.model_unavailable = _is_unknown_model(response)  # type: ignore[attr-defined]
            raise error
        try:
            return response.json()
        except ValueError:
            raise ProviderError(f"{self.name}: response is not JSON") from None

    def _to_response(self, request: LLMRequest, body: Any) -> LLMResponse:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError(
                f"{self.name}: unexpected response: {self._redact(json.dumps(body))}"
            ) from None
        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model_for(request.capability),
            data=extract_json(text) if request.json_schema is not None else None,
            tokens_in=usage.get("prompt_tokens") or 0,
            tokens_out=usage.get("completion_tokens") or 0,
        )

    def _redact(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return text[:_EXCERPT_CHARS]


def _seconds(retry_after: str | None) -> float | None:
    """Retry-After in delta-seconds form. The HTTP-date form, NaN and infinity yield None and the
    router backs off by its default."""
    if retry_after is None:
        return None
    try:
        seconds = float(retry_after)
    except ValueError:
        return None
    return max(seconds, 0.0) if math.isfinite(seconds) else None


def _is_unknown_model(response: httpx.Response) -> bool:
    if response.status_code not in (400, 404):
        return False
    text = response.text.lower()
    return any(marker in text for marker in _UNKNOWN_MODEL_MARKERS) or (
        "model" in text and "does not exist" in text
    )


def _rate_limited(message: str, retry_after: float | None) -> RateLimited:
    exc = RateLimited(message)
    # base.RateLimited is a frozen contract without this field; the router reads it via getattr.
    exc.retry_after = retry_after  # type: ignore[attr-defined]
    return exc
