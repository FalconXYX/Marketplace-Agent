"""The owner's Saturday, end to end: photo to sale with every part wired together.

The web app (TestClient over create_app), the runner, ControlService, and the Discord cards and
buttons (discord_views) share one runtime over FakeMarketplace. The model is HeuristicPolicy,
the simulation's offline seller, behind a ScriptedProvider that also writes the photo draft.
Nothing touches the network.

| Local time, Sat Oct 10 | What happens |
|---|---|
| 09:00 | Owner photographs the desk on the phone, drafts it, confirms at $120 (floor $95) |
| 09:05 | Aisha offers $110; the agent accepts and books her. Owner taps Undo, tells her (web) |
| 09:20 | A scammer writes; the chat is frozen. Owner resolves the alert |
| 09:55 | Noah offers $115; accepting needs approval now (the undo demoted it). Owner approves |
| 13:10 | Noah heads over; the agent sends him the address, the undo window long closed |
| 14:40 | Pickup check: owner taps Picked up; the item is sold and its listing taken down |
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from marketplace_agent.ai.base import LLMRequest
from marketplace_agent.ai.providers.scripted import ScriptedProvider
from marketplace_agent.ai.quota import QuotaTracker
from marketplace_agent.ai.router import ProviderRouter
from marketplace_agent.config import Settings
from marketplace_agent.core.db import make_engine
from marketplace_agent.core.models import (
    Commitment,
    Escalation,
    Item,
    ItemStatus,
    Listing,
    ListingStatus,
    Thread,
    TrustRecord,
    TrustTier,
)
from marketplace_agent.core.runtime import Runtime, build_runtime
from marketplace_agent.guard.ledger import active_commitment
from marketplace_agent.notify import discord_views as views
from marketplace_agent.notify.base import Notice, NoticeKind, RecordingNotifier
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.sim.audit import audit
from marketplace_agent.sim.harness import LedgerOracle, TickingClock
from marketplace_agent.sim.policy import HeuristicPolicy
from marketplace_agent.web.app import create_app
from test_web_support import CONFIRM, DRAFT, HX, jpeg_with_gps

OWNER, STRANGER = 4242, 777
ADDRESS = "123 Example St"
BASE = "http://127.0.0.1"
SATURDAY_9AM = datetime(2026, 10, 10, 13, 0, tzinfo=UTC)  # 09:00 in Toronto
MAX_TICKS = 80


@dataclass
class Seen:
    """One Discord post as the owner sees it: the notice and the card rendered for it."""

    notice: Notice
    card: views.Card

    def line(self) -> str:
        buttons = " ".join(f"[{b.label}]" for b in self.card.buttons)
        text = f"[{self.notice.kind}] {self.card.title}: {self.card.description}"
        return text.replace("\n", " / ") + (f"  {buttons}" if buttons else "")


@dataclass
class Saturday:
    runtime: Runtime
    client: TestClient
    market: FakeMarketplace
    clock: TickingClock
    notifier: RecordingNotifier
    feed: list[Seen] = field(default_factory=list)  # the owner's Discord channel
    said: list[str] = field(default_factory=list)  # what the bot answered button presses

    @property
    def settings(self) -> Settings:
        return self.runtime.settings

    def session(self) -> Session:
        return Session(self.runtime.engine, expire_on_commit=False)

    # ------------------------------------------------------------ Discord

    def read_discord(self) -> list[Seen]:
        """Render every notice posted since the last read, as the bot does."""
        new = []
        with self.session() as session:
            for notice in self.notifier.notices[len(self.feed) :]:
                assert views.notice_ready(session, notice), notice
                card = views.render_notice(session, self.clock, self.settings, notice)
                new.append(Seen(notice, card))
        self.feed += new
        return new

    def rerender(self, seen: Seen) -> views.Card:
        with self.session() as session:
            return views.render_notice(session, self.clock, self.settings, seen.notice)

    def press(
        self, seen: Seen, label: str, form: dict[str, str] | None = None, user: int = OWNER
    ) -> views.ActionResponse:
        [button] = [b for b in self.rerender(seen).buttons if b.label == label]
        assert not button.disabled, f"{label} is switched off on {seen.card.title}"
        modal = views.modal_for(button.custom_id)
        if modal is not None and user == OWNER:  # the bot opens the modal first
            assert form is not None and set(form) <= {f.key for f in modal.fields}
        response = views.handle_action(
            self.runtime.control, button.custom_id, user, self.settings.discord_owner_id, form
        )
        if user == OWNER:
            self.said.append(f"{label} -> {response.text}")
        return response

    def card(self, kind: NoticeKind, title: str) -> Seen:
        [seen] = [s for s in self.feed if s.notice.kind == kind and s.card.title.startswith(title)]
        return seen

    # ------------------------------------------------------------ time and buyers

    def tick(self, minutes: float = 5) -> list[Seen]:
        self.clock.advance(timedelta(minutes=minutes))
        self.market.step(self._oracle())
        report = self.runtime.runner.tick()
        assert report.errors == [], report.errors
        return self.read_discord()

    def tick_until(self, done, minutes: float = 5) -> list[Seen]:
        posted: list[Seen] = []
        for _ in range(MAX_TICKS):
            posted += self.tick(minutes)
            if done():
                return posted
        raise AssertionError(f"never happened by {self.clock.now()}")

    def buyer(self, persona: str, says: str | None = None, at: datetime | None = None) -> str:
        with self.session() as session:
            live = select(Listing).where(Listing.status == ListingStatus.LIVE)
            listing = session.exec(live).one()
        assert listing.remote_id is not None
        remote = self.market.add_buyer(persona, listing.remote_id, at)
        if says is not None:
            self.market.buyer_say(remote, says)
        return remote

    def name(self, remote: str) -> str:
        return self.market.threads[remote].buyer_name

    def sent_to(self, remote: str) -> list[str]:
        return [body for thread, body, _ in self.market.sent if thread == remote]

    def thread_id(self, remote: str) -> int:
        with self.session() as session:
            thread = session.exec(select(Thread).where(Thread.remote_id == remote)).one()
        assert thread.id is not None
        return thread.id

    def _oracle(self) -> LedgerOracle:
        """What buyers may know: their committed price and booked pickup (sim.harness)."""
        prices: dict[str, int] = {}
        pickups: dict[str, tuple[datetime, datetime]] = {}
        with self.session() as session:
            for item_id in session.exec(select(Commitment.item_id).distinct()).all():
                active = active_commitment(session, item_id)
                if active is None:
                    continue
                thread = session.get(Thread, active.accepted.thread_id)
                assert thread is not None
                prices[thread.remote_id] = active.accepted.price_cents or 0
                booked = active.pickup
                if booked and booked.pickup_start and booked.pickup_end:
                    pickups[thread.remote_id] = (booked.pickup_start, booked.pickup_end)
        return LedgerOracle(prices, pickups)


@pytest.fixture
def day(tmp_path) -> Saturday:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        db_url="sqlite://",
        pickup_area="Bloor St W & Ossington Ave",
        pickup_address=ADDRESS,
        enabled_platforms=["fake"],
        discord_owner_id=OWNER,
    )
    clock = TickingClock(SATURDAY_9AM)
    engine = make_engine(settings.database_url)
    market = FakeMarketplace(clock, seed=7)
    notifier = RecordingNotifier()
    policy = HeuristicPolicy(engine, settings, clock)

    def model(request: LLMRequest) -> dict[str, Any]:
        return DRAFT if request.purpose == "draft" else policy(request)

    quota = QuotaTracker(engine, clock, settings.provider_limits)
    runtime = build_runtime(
        settings,
        clock=clock,
        notifier=notifier,
        platforms={"fake": market},
        router=ProviderRouter([ScriptedProvider(model, name="heuristic")], quota),
        engine=engine,
    )
    client = TestClient(create_app(runtime), base_url=BASE, headers={"Origin": BASE})
    return Saturday(runtime, client, market, clock, notifier)


def test_a_saturday_from_photo_to_sold(day):
    # 09:00, in the garage: photo, draft, confirm, all on the phone
    assert day.client.get("/").status_code == 200
    assert 'capture="environment"' in day.client.get("/items/new").text
    photo = ("photos", ("desk.jpg", jpeg_with_gps(), "image/jpeg"))
    created = day.client.post(
        "/items", data={"hint": "IKEA desk $120"}, files=[photo], follow_redirects=False
    )
    assert created.status_code == 303
    item_url = created.headers["location"]
    item_id = int(item_url.rsplit("/", 1)[1])
    assert 'hx-trigger="load"' in day.client.get(item_url).text
    drafted = day.client.post(f"{item_url}/draft", headers=HX)
    assert 'value="IKEA Linnmon desk"' in drafted.text
    assert 'name="price" value="120"' in drafted.text  # from the hint
    confirmed = day.client.post(f"{item_url}/confirm", data=CONFIRM, headers=HX)
    assert "Confirmed for sale." in confirmed.text
    [posted] = day.market.listings.values()
    assert posted.payload.title == "IKEA Linnmon desk, white"
    assert posted.payload.price_cents == 12_000
    assert day.read_discord() == []  # confirming is the owner's own act: nothing to tell them

    # 09:05: buyers arrive; Aisha offers $110 and the agent closes with her on its own
    jordan = day.buyer("lowballer")
    day.buyer("scammer", at=day.clock.now() + timedelta(minutes=20))
    aisha = day.buyer("scripted", "Hi! I can do $110 cash, could I grab it this afternoon?")
    day.tick_until(lambda: any(s.notice.kind == NoticeKind.COMMITMENT for s in day.feed))
    first_deal = day.card(NoticeKind.COMMITMENT, "Deal: IKEA Linnmon desk, white")
    deal_line = f"Deal #{first_deal.notice.ref_id}: promised to {day.name(aisha)} for $110.00."
    assert deal_line in first_deal.card.description
    assert "Undo until Sat Oct 10 09:" in first_deal.card.description
    assert [b.label for b in first_deal.card.buttons] == ["Undo"]
    assert day.card(NoticeKind.INFO, "Pickup booked").card.buttons[0].label == "Pause item"
    assert day.sent_to(aisha)[-1].startswith("Deal, it's yours for $110.")
    assert day.sent_to(jordan), "the lowballer got a counter"

    # The owner changes their mind inside the undo window
    assert day.press(first_deal, "Undo", user=STRANGER).text == views.NOT_OWNER
    undone = day.press(first_deal, "Undo")
    assert undone.disable_buttons and not undone.ephemeral
    assert undone.text == (
        f"Undid the $110.00 deal with {day.name(aisha).rstrip('.')}. Item {item_id} is listed "
        "again; the agent will re-plan."
    )
    closed = day.rerender(first_deal)
    assert "This deal is closed." in closed.description and closed.buttons[0].disabled
    with day.session() as session:
        record = session.exec(
            select(TrustRecord).where(TrustRecord.action == "accept_offer")
        ).one()
    assert record.tier == TrustTier.ASK  # one correction demotes
    # The agent may not message Aisha twice in a row, so the owner tells her from the web page
    sorry = "Sorry Aisha, I have to cancel: it was already promised to a neighbour."
    replied = day.client.post(
        f"/threads/{day.thread_id(aisha)}/reply", data={"body": sorry}, headers=HX
    )
    assert replied.status_code == 200 and day.sent_to(aisha)[-1] == sorry

    # 09:20: the scammer is frozen out; the owner clears the alert
    day.tick_until(lambda: any(s.notice.kind == NoticeKind.ALERT for s in day.feed))
    scam = day.card(NoticeKind.ALERT, "Possible scam on fake")
    assert [b.label for b in scam.card.buttons] == ["Resolve"]
    resolved = day.press(scam, "Resolve", {"answer": "Ignore them."})
    assert resolved.text.startswith("Resolved escalation #") and resolved.disable_buttons

    # 09:55: Noah offers $115. After the undo, accepting needs the owner's approval
    day.tick(30)
    noah = day.buyer("scripted", "Would you take $115? I can pick up today.")
    day.tick_until(lambda: any(s.notice.kind == NoticeKind.APPROVAL for s in day.feed))
    ask = day.card(NoticeKind.APPROVAL, "Approve #")
    assert ask.card.title == f"Approve #{ask.notice.ref_id}? IKEA Linnmon desk, white"
    assert f"Accept $115.00 from {day.name(noah)}" in ask.card.description
    assert [b.label for b in ask.card.buttons] == ["Approve", "Reject"]
    assert not any("Deal" in body for body in day.sent_to(noah))  # nothing promised yet
    approved = day.press(ask, "Approve")
    assert approved.text.startswith(f"Approved #{ask.notice.ref_id}: Accepted $115.00 from")
    assert day.rerender(ask).buttons[0].disabled
    day.read_discord()
    second_deal = day.feed[-1]
    assert second_deal.notice.kind == NoticeKind.COMMITMENT
    deal_line = f"Deal #{second_deal.notice.ref_id}: promised to {day.name(noah)} for $115.00."
    assert deal_line in second_deal.card.description
    day.tick_until(lambda: any("Deal, it's yours for $115" in b for b in day.sent_to(noah)))
    with day.session() as session:
        deal = active_commitment(session, item_id)
    assert deal is not None and deal.pickup is not None and deal.pickup.pickup_start is not None
    start, end = deal.pickup.pickup_start, deal.pickup.pickup_end

    # 13:10: Noah heads over, the undo window long closed: the agent sends him the address
    assert day.clock.now() < start - timedelta(minutes=20)
    day.clock.set(start - timedelta(minutes=20))
    assert day.rerender(second_deal).buttons[0].disabled  # the undo window is over
    day.market.buyer_say(noah, "Leaving now, see you soon!")
    day.tick(1)
    assert ADDRESS in day.sent_to(noah)[-1]

    # After the pickup: the owner says Noah came
    day.tick_until(lambda: any(s.card.title.startswith("Did") for s in day.feed), minutes=30)
    assert day.clock.now() > end
    check = day.card(NoticeKind.ESCALATION, f"Did {day.name(noah)} pick up")
    assert [b.label for b in check.card.buttons] == ["Picked up", "No-show"]
    assert day.press(check, "Picked up", user=STRANGER).text == views.NOT_OWNER
    sold = day.press(check, "Picked up")
    assert sold.text == f"Marked item #{item_id} sold and took its listings down."
    assert all(b.disabled for b in day.rerender(check).buttons)
    assert posted.delisted_at is not None
    with day.session() as session:
        assert session.get(Item, item_id).status == ItemStatus.SOLD
        assert all(e.resolved_at for e in session.exec(select(Escalation)).all())
        digest = views.build_digest(session, day.clock, day.settings)
        violations = audit(session, day.settings, sent_log=day.market.sent)
    assert f"#{item_id} IKEA Linnmon desk, white: $115.00 to {day.name(noah)}" in digest
    assert violations == []

    # The owner's channel, in order, and the item's page at the end of the day
    assert [(s.notice.kind.value, s.card.title) for s in day.feed] == [
        ("commitment", "Deal: IKEA Linnmon desk, white"),
        ("info", "Pickup booked: IKEA Linnmon desk, white"),
        ("alert", "Possible scam on fake"),
        ("approval", f"Approve #{ask.notice.ref_id}? IKEA Linnmon desk, white"),
        ("commitment", "Deal: IKEA Linnmon desk, white"),
        ("info", "Pickup booked: IKEA Linnmon desk, white"),
        ("escalation", f"Did {day.name(noah)} pick up IKEA Linnmon desk, white?"),
    ]
    page = day.client.get(item_url).text
    assert re.search(r"\bsold\b", page, re.IGNORECASE)
    everything_shown = [day.client.get("/").text, page, digest, *day.said]
    everything_shown += [json.dumps(asdict(s.card)) for s in day.feed]
    assert not any(ADDRESS in text for text in everything_shown)
    print("\n".join(["OWNER'S DISCORD", *(s.line() for s in day.feed), "BUTTONS", *day.said]))
