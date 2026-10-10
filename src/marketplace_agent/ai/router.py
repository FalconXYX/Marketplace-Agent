"""Routes each request to a provider with quota left, falling back down the list.

Providers mark errors with attributes, since the frozen `ai.base` errors do not declare them:

| Attribute | Router |
|---|---|
| `RateLimited.retry_after` | Cools the provider down that many seconds |
| `RateLimited.daily` | Suspends the model until the provider's day ends |
| `ProviderError.model_unavailable` | Suspends the model until the provider's day ends |
"""
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from marketplace_agent.ai.base import (
    Capability,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    QuotaExhausted,
    QuotaState,
    RateLimited,
)
from marketplace_agent.ai.quota import QuotaTracker

logger = logging.getLogger("marketplace_agent.ai")
_decoder = json.JSONDecoder()


def extract_json(text: str) -> dict[str, Any] | None:
    """First top-level JSON object in text, ignoring code fences and surrounding prose.

    A brace span that fails to parse is skipped whole, so a truncated object never
    yields one of its nested objects.
    """
    start = text.find("{")
    while start != -1:
        try:
            value, _ = _decoder.raw_decode(text, start)
            return value
        except json.JSONDecodeError:
            end = _span_end(text, start)
            if end is None:
                return None
            start = text.find("{", end)
    return None


def _span_end(text: str, start: int) -> int | None:
    """Index just past the brace matching text[start], honoring JSON strings; None if unbalanced."""
    depth = 0
    in_string = escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


class ProviderRouter:
    def __init__(self, providers: Sequence[LLMProvider], quota: QuotaTracker):
        self.providers = list(providers)
        self.quota = quota

    @staticmethod
    def serves(provider: LLMProvider, capability: Capability) -> bool:
        return capability in provider.capabilities or (
            capability is Capability.CHEAP and Capability.REASONING in provider.capabilities
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        last_error: ProviderError | None = None
        for provider in self._candidates(request.capability):
            model = self._model(provider, request.capability)
            if not self.quota.can_request(provider.name, model):
                continue
            try:
                return self._try(provider, request)
            except RateLimited as exc:
                daily = getattr(exc, "daily", False)
                if daily:
                    self.quota.suspend(provider.name, model)
                else:
                    self.quota.mark_rate_limited(provider.name, getattr(exc, "retry_after", None))
                logger.warning(
                    "rate limited provider=%s model=%s daily=%s purpose=%s",
                    provider.name,
                    model,
                    daily,
                    request.purpose,
                )
            except ProviderError as exc:
                last_error = exc
                unavailable = getattr(exc, "model_unavailable", False)
                if unavailable:
                    self.quota.suspend(provider.name, model)
                logger.warning(
                    "provider error provider=%s model=%s unavailable=%s purpose=%s error=%s",
                    provider.name,
                    model,
                    unavailable,
                    request.purpose,
                    type(exc).__name__,
                )
        if last_error is None:
            raise QuotaExhausted(f"no provider can serve {request.capability} right now")
        raise last_error

    def state(self, capability: Capability = Capability.REASONING) -> QuotaState:
        """Suspended models do not count: their unused local budget is not really available."""
        return self.quota.state(
            p.name
            for p in self._candidates(capability)
            if not self.quota.is_suspended(p.name, self._model(p, capability))
        )

    def throttled(self, capability: Capability = Capability.REASONING) -> bool:
        """Every provider for capability is held back by a per-minute limit or a short cooldown,
        not by spent daily quota: retry in a few minutes rather than give up."""
        return self.quota.throttled(
            p.name
            for p in self._candidates(capability)
            if not self.quota.is_suspended(p.name, self._model(p, capability))
        )

    def has_capability(self, capability: Capability) -> bool:
        return any(self.serves(p, capability) for p in self.providers)

    def _candidates(self, capability: Capability) -> list[LLMProvider]:
        native = [p for p in self.providers if capability in p.capabilities]
        fallback = [
            p for p in self.providers if capability not in p.capabilities and self.serves(p, capability)
        ]
        return native + fallback

    @staticmethod
    def _routed(provider: LLMProvider, capability: Capability) -> Capability:
        """CHEAP goes to a provider's reasoning model when it has no cheap one."""
        return capability if capability in provider.capabilities else Capability.REASONING

    def _model(self, provider: LLMProvider, capability: Capability) -> str:
        return provider.model_for(self._routed(provider, capability))

    def _try(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        """One provider call. Usage is recorded before JSON validation: a bad answer still spent quota."""
        capability = self._routed(provider, request.capability)
        if capability is not request.capability:
            request = replace(request, capability=capability)
        response = provider.generate(request)
        self.quota.record(provider.name, response.model, response.tokens_in, response.tokens_out)
        logger.info(
            "llm call provider=%s model=%s purpose=%s tokens_in=%d tokens_out=%d",
            provider.name,
            response.model,
            request.purpose,
            response.tokens_in,
            response.tokens_out,
        )
        if request.json_schema is not None and response.data is None:
            response.data = extract_json(response.text)
            if response.data is None:
                raise ProviderError(f"{provider.name} returned invalid JSON")
        return response
