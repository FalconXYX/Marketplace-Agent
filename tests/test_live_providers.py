"""Real provider calls, one request each. Needs MA_LIVE_TESTS=1.

Gemini tests also need a Gemini key in Settings: GEMINI_API_KEY or MA_GEMINI_API_KEY, from the
environment or .env. Models come from Settings too, so an override is what gets tested.
"""
import io
import os

import httpx
import pytest
from PIL import Image, ImageDraw

from marketplace_agent.ai.base import Capability, LLMRequest
from marketplace_agent.ai.drafting import draft_item
from marketplace_agent.ai.providers.gemini import GeminiProvider
from marketplace_agent.config import Settings
from marketplace_agent.core.items import create_item
from marketplace_agent.intake.photos import ingest_photo

LIVE = os.environ.get("MA_LIVE_TESTS") == "1"
LIVE_SETTINGS = Settings() if LIVE else None
GEMINI_KEY = (
    LIVE_SETTINGS.gemini_api_key.get_secret_value()
    if LIVE_SETTINGS is not None and LIVE_SETTINGS.gemini_api_key
    else None
)

pytestmark = [pytest.mark.live, pytest.mark.skipif(not LIVE, reason="needs MA_LIVE_TESTS=1")]
needs_gemini = pytest.mark.skipif(
    GEMINI_KEY is None,
    reason="needs MA_LIVE_TESTS=1 and a Gemini key: GEMINI_API_KEY or MA_GEMINI_API_KEY",
)

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "brand": {"type": ["string", "null"]},
    },
    "required": ["category", "brand"],
}


def _desk_photo() -> bytes:
    """A plain drawing of a wooden desk: enough to describe, nothing to identify."""
    image = Image.new("RGB", (480, 360), (235, 235, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 120, 420, 150), fill=(140, 95, 55))  # top
    for x in (75, 390):
        draw.rectangle((x, 150, x + 15, 320), fill=(120, 80, 45))  # legs
    draw.rectangle((250, 150, 380, 200), fill=(150, 105, 62))  # drawer
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


@needs_gemini
@pytest.mark.parametrize("capability", [Capability.CHEAP, Capability.REASONING])
def test_gemini_returns_schema_json(capability):
    assert LIVE_SETTINGS is not None and GEMINI_KEY is not None
    models = {
        Capability.REASONING: LIVE_SETTINGS.reasoning_model,
        Capability.CHEAP: LIVE_SETTINGS.cheap_model,
    }
    response = GeminiProvider(GEMINI_KEY, models).generate(
        LLMRequest(
            system="You classify second-hand items for a marketplace listing.",
            prompt="Item: a plain wooden desk, no visible brand. Give its category.",
            capability=capability,
            json_schema=SCHEMA,
            max_output_tokens=1024,
            temperature=0.0,
            purpose="live-test",
        )
    )

    assert response.data is not None, response.text
    assert isinstance(response.data["category"], str)
    assert response.data["brand"] is None or isinstance(response.data["brand"], str)
    assert response.tokens_in > 0


@needs_gemini
def test_gemini_vision_model_drafts_with_the_wire_schema(session, settings, clock):
    """The real draft path: photo, DRAFT_WIRE_SCHEMA (nullable enum included), vision model.

    A schema Gemini rejects fails here with HTTP 400; a reply that does not parse fails with
    "invalid draft".
    """
    assert LIVE_SETTINGS is not None and GEMINI_KEY is not None
    item = create_item(session, clock, "wooden desk with a drawer $40")
    ingest_photo(session, settings, item.id, _desk_photo())
    provider = GeminiProvider(GEMINI_KEY, {Capability.VISION: LIVE_SETTINGS.vision_model})

    draft = draft_item(session, provider, settings, clock, item.id)

    assert draft.title and item.title == draft.title


def test_configured_openrouter_model_is_in_the_catalog():
    """A retired ":free" model answers every request with 4xx. The catalog needs no key."""
    assert LIVE_SETTINGS is not None
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()

    ids = {model["id"] for model in response.json()["data"]}
    assert LIVE_SETTINGS.openrouter_model in ids
