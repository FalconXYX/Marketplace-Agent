"""Builders shared by the web tests: a runtime over FakeMarketplace and ScriptedProvider, with a
TestClient on its app. No tests here."""

import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi.testclient import TestClient
from PIL import ExifTags, Image
from sqlmodel import Session, select

from marketplace_agent.ai.base import Capability, LLMRequest
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.db import make_engine
from marketplace_agent.core.models import Listing, Thread
from marketplace_agent.core.runtime import Runtime, build_runtime
from marketplace_agent.notify.base import RecordingNotifier
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.web.app import create_app

BASE = "http://127.0.0.1"
ADDRESS = "123 Example St"
QUIET = {"rationale": "Nothing to do yet.", "actions": []}
DRAFT = {
    "title": "IKEA Linnmon desk",
    "description": "White desk with a 120 x 60 cm top.",
    "category": "Furniture",
    "condition": "good",
    "attributes": [{"key": "Brand", "value": "IKEA"}],
    "suggested_price_low_dollars": 90,
    "suggested_price_high_dollars": 110,
    "price_reasoning": "Similar desks sell for about $100.",
    "uncertain": ["exact dimensions"],
}
CONFIRM = {
    "title": "IKEA Linnmon desk, white",
    "description": "White desk with a 120 x 60 cm top. Pickup only.",
    "category": "Furniture",
    "condition": "good",
    "attributes": "Brand: IKEA\nColour: white",
    "price": "120",
    "floor": "95",
    "target": "110",
    "firmness": "slight",
    "deadline_days": "14",
    "notes": "No holds.",
    "platforms": ["fake"],
}
HX = {"HX-Request": "true"}
_THREAD = re.compile(r"\[thread (\d+),")

Responder = Callable[[LLMRequest], dict[str, Any]]


def jpeg_with_gps(size: tuple[int, int] = (640, 480), color=(120, 80, 40)) -> bytes:
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Phone"
    exif[ExifTags.Base.GPSInfo] = {
        ExifTags.GPS.GPSLatitudeRef: "N",
        ExifTags.GPS.GPSLatitude: (43.0, 39.0, 11.5),
        ExifTags.GPS.GPSLongitudeRef: "W",
        ExifTags.GPS.GPSLongitude: (79.0, 22.0, 59.5),
    }
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


def has_gps(jpeg: bytes) -> bool:
    return bool(Image.open(io.BytesIO(jpeg)).getexif().get_ifd(ExifTags.IFD.GPSInfo))


def drafting(request: LLMRequest) -> dict[str, Any]:
    """Drafts the listing; every wake does nothing."""
    return DRAFT if request.purpose == "draft" else QUIET


def accepting(price_cents: int) -> Responder:
    """Drafts, then accepts the first offer it is woken for, once."""
    accepted: list[int] = []

    def respond(request: LLMRequest) -> dict[str, Any]:
        if request.purpose == "draft":
            return DRAFT
        found = _THREAD.search(request.prompt)
        if found is None or accepted:
            return QUIET
        accepted.append(int(found[1]))
        args = {"thread_id": int(found[1]), "price_cents": price_cents}
        return {
            "rationale": "The buyer offered above the floor, so take it.",
            "actions": [{"tool": "accept_offer", "args_json": json.dumps(args), "why": "Fair."}],
        }

    return respond


@dataclass
class WebWorld:
    runtime: Runtime
    client: TestClient
    market: FakeMarketplace
    clock: FakeClock
    notifier: RecordingNotifier

    def session(self) -> Session:
        return Session(self.runtime.engine, expire_on_commit=False)

    def upload(self, hint: str = "IKEA desk $120", photos: list[bytes] | None = None) -> int:
        chosen = photos or [jpeg_with_gps()]
        files = [("photos", (f"p{i}.jpg", data, "image/jpeg")) for i, data in enumerate(chosen)]
        response = self.client.post(
            "/items", data={"hint": hint}, files=files, follow_redirects=False
        )
        assert response.status_code == 303, response.text
        return int(response.headers["location"].rsplit("/", 1)[1])

    def htmx(self, url: str, data: dict[str, Any] | None = None) -> httpx.Response:
        return self.client.post(url, data=data or {}, headers=HX)

    def listed(self, hint: str = "IKEA desk $120", **fields: Any) -> int:
        item_id = self.upload(hint)
        response = self.htmx(f"/items/{item_id}/confirm", {**CONFIRM, **fields})
        assert response.status_code == 200, response.text
        return item_id

    def buyer_writes(self, item_id: int, text: str, persona: str = "scripted") -> Thread:
        """A buyer messages about the item's fake listing, and a runner tick ingests it."""
        with self.session() as session:
            listing = session.exec(select(Listing).where(Listing.item_id == item_id)).one()
        assert listing.remote_id is not None
        remote = self.market.add_buyer(persona, listing.remote_id)
        self.market.buyer_say(remote, text)
        self.runtime.runner.tick()
        with self.session() as session:
            return session.exec(select(Thread).where(Thread.remote_id == remote)).one()


def build_world(
    tmp_path,
    responder: Responder = drafting,
    *,
    capabilities: frozenset[Capability] = frozenset(Capability),
    base_url: str = BASE,
    **overrides: Any,
) -> WebWorld:
    fields: dict[str, Any] = {
        "_env_file": None,
        "data_dir": tmp_path / "data",
        "db_url": "sqlite://",
        "pickup_area": "Bloor St W & Ossington Ave",
        "pickup_address": ADDRESS,
        "enabled_platforms": ["fake"],
    }
    settings = Settings(**{**fields, **overrides})
    clock = FakeClock()
    engine = make_engine(settings.database_url)
    market = FakeMarketplace(clock)
    notifier = RecordingNotifier()
    provider = ScriptedProvider(responder, capabilities=capabilities)
    runtime = build_runtime(
        settings,
        clock=clock,
        notifier=notifier,
        platforms={"fake": market},
        router=ProviderRouter([provider], QuotaTracker(engine, clock, {})),
        engine=engine,
    )
    client = TestClient(create_app(runtime), base_url=base_url, headers={"Origin": base_url})
    return WebWorld(runtime, client, market, clock, notifier)
