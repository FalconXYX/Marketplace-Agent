"""Builds the ProviderRouter from Settings. Provider modules are imported only when their key is set."""
from sqlalchemy import Engine

from marketplace_agent.ai.base import Capability, LLMProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import Clock


def build_router(settings: Settings, engine: Engine, clock: Clock) -> ProviderRouter:
    quota = QuotaTracker(engine, clock, settings.provider_limits, settings.quota_constrained_fraction)
    return ProviderRouter(_providers(settings), quota)


def _providers(settings: Settings) -> list[LLMProvider]:
    """Priority order: Gemini (only one with vision), then Groq, then OpenRouter."""
    providers: list[LLMProvider] = []
    if settings.gemini_api_key:
        from marketplace_agent.ai.providers.gemini import GeminiProvider

        models = {
            Capability.VISION: settings.vision_model,
            Capability.REASONING: settings.reasoning_model,
            Capability.CHEAP: settings.cheap_model,
        }
        providers.append(GeminiProvider(settings.gemini_api_key.get_secret_value(), models))
    if settings.groq_api_key:
        from marketplace_agent.ai.providers.openai_compat import GROQ_BASE_URL, OpenAICompatProvider

        key = settings.groq_api_key.get_secret_value()
        providers.append(OpenAICompatProvider("groq", GROQ_BASE_URL, key, settings.groq_model))
    if settings.openrouter_api_key:
        from marketplace_agent.ai.providers.openai_compat import (
            OPENROUTER_BASE_URL,
            OpenAICompatProvider,
        )

        key = settings.openrouter_api_key.get_secret_value()
        providers.append(
            OpenAICompatProvider("openrouter", OPENROUTER_BASE_URL, key, settings.openrouter_model)
        )
    return providers
