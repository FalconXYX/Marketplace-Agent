"""Model provider contract. The agent and drafting code talk only to a ProviderRouter."""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Capability(StrEnum):
    VISION = "vision"        # photos -> listing draft, selector healing
    REASONING = "reasoning"  # agent wakes
    CHEAP = "cheap"          # message classification, phrasing


class QuotaState(StrEnum):
    HEALTHY = "healthy"            # full reasoning
    CONSTRAINED = "constrained"    # reason only on commitment-relevant and novel triggers
    EXHAUSTED = "exhausted"        # no model calls; escalate to the human


@dataclass
class ImageInput:
    data: bytes
    mime_type: str = "image/jpeg"


@dataclass
class LLMRequest:
    system: str
    prompt: str
    capability: Capability = Capability.REASONING
    images: list[ImageInput] = field(default_factory=list)
    json_schema: dict[str, Any] | None = None  # when set, response.data holds parsed JSON
    max_output_tokens: int = 2048
    temperature: float = 0.4
    purpose: str = ""                          # log label: draft | wake | heal | ...


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    data: dict[str, Any] | None = None
    tokens_in: int = 0
    tokens_out: int = 0


class ProviderError(Exception):
    def __init__(self, message: str = "", *, model_unavailable: bool = False):
        super().__init__(message)
        self.model_unavailable = model_unavailable  # model id unknown or retired: suspend it for the day


class RateLimited(ProviderError):
    """Provider said slow down (HTTP 429 or equivalent). Router tries the next provider."""

    def __init__(self, message: str = "", retry_after: float | None = None, *, daily: bool = False):
        super().__init__(message)
        self.retry_after = retry_after  # seconds, when the provider said
        self.daily = daily              # per-day quota hit: suspend the model until the provider's day ends


class QuotaExhausted(ProviderError):
    """No provider can serve this request right now."""


class LLMProvider(Protocol):
    name: str
    capabilities: frozenset[Capability]

    def model_for(self, capability: Capability) -> str: ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...
