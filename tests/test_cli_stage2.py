"""Stage 2 CLI commands: run, tick, publish, undo, approve, reject, pause, resume, sold, log,
trust, memory, and the owner controls (price, floor, firmness, goal on a listed item, withdraw,
gone, escalations, resolve, reply). Each invocation builds a fresh runtime over the same
database file, so the fake market starts empty each time."""

import io
import json
import sqlite3
import sys
import types
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, select
from typer.testing import CliRunner

from marketplace_agent.cli import app
from marketplace_agent.control.service import ControlService
from marketplace_agent.core.clock import SystemClock
from marketplace_agent.core.db import make_engine
from marketplace_agent.core.models import (
    AgentRun,
    ApprovalStatus,
    Buyer,
    Direction,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Memory,
    MemoryScope,
    Listing,
    ListingStatus,
    Message,
    PendingApproval,
    PriceChange,
    Thread,
    ThreadState,
    TrustRecord,
    TrustTier,
    WakeRequest,
)
from marketplace_agent.guard import ledger
from marketplace_agent.platforms.fake import FakeMarketplace
from marketplace_agent.runner.tick import Runner

runner = CliRunner()
KEY_VARS = (
    "MA_GEMINI_API_KEY",
    "GEMINI_API_KEY",
    "MA_GROQ_API_KEY",
    "GROQ_API_KEY",
    "MA_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "MA_ENABLED_PLATFORMS",
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep a developer's .env and toml out of the run
    monkeypatch.setenv("MA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MA_DB_URL", raising=False)
    for name in KEY_VARS:  # no real model can be reached
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "data"


@pytest.fixture
def db(data_dir):
    def open_session() -> Session:
        return Session(engine, expire_on_commit=False)

    data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(f"sqlite:///{data_dir / 'agent.db'}")
    yield open_session
    engine.dispose()


@pytest.fixture
def item(data_dir, tmp_path, db) -> int:
    """Item 1: photo, goal ($120 list, $95 floor), and a title, still a draft."""
    photo = tmp_path / "desk.jpg"
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (150, 100, 60)).save(buffer, "JPEG")
    photo.write_bytes(buffer.getvalue())
    _ok("add-item", "--hint", "IKEA desk $120", "--photo", str(photo))
    _ok("goal", "1", "--price", "120", "--floor", "95")
    with db() as session:
        row = session.get(Item, 1)
        row.title, row.description = "IKEA desk", "Solid desk."
        session.commit()
    return 1


@pytest.fixture
def listed(item) -> int:
    _ok("publish", "1")
    return item


def _invoke(*args: str):
    return runner.invoke(app, list(args))


def _ok(*args: str) -> str:
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    return result.output


def _fails(*args: str) -> str:
    result = _invoke(*args)
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    return result.output


def _deal(db, item_id: int) -> int:
    clock = SystemClock()
    with db() as session:
        buyer = Buyer(platform="fake", handle="maya", display_name="Maya")
        session.add(buyer)
        session.flush()
        thread = Thread(
            platform="fake",
            remote_id="t1",
            item_id=item_id,
            buyer_id=buyer.id,
            state=ThreadState.NEGOTIATING,
            created_at=clock.now(),
        )
        session.add(thread)
        session.flush()
        session.add(
            Message(
                thread_id=thread.id,
                direction=Direction.INBOUND,
                body="Would you take $110?",
                sent_at=clock.now(),
                via="buyer",
                handled=True,
            )
        )
        accepted = ledger.accept(
            session,
            clock,
            item_id=item_id,
            thread_id=thread.id,
            buyer_id=buyer.id,
            price_cents=11_000,
            undo_minutes=10,
        )
        session.commit()
        return accepted.id


def _approval(db, item_id: int, tool: str, args: dict) -> int:
    with db() as session:
        approval = PendingApproval(
            item_id=item_id,
            tool=tool,
            args_json=json.dumps(args),
            created_at=SystemClock().now(),
        )
        session.add(approval)
        session.commit()
        return approval.id


# ---------------------------------------------------------------- run and tick


def test_tick_prints_a_report(data_dir):
    output = _ok("tick")

    assert "New messages:  0" in output
    assert "Wakes:         0" in output
    assert "Quota:         exhausted" in output  # no model key configured
    assert "Errors:        none" in output
    assert "No model is configured (set GEMINI_API_KEY)" in output


def test_tick_with_an_unbuilt_platform_explains_what_to_change(data_dir, monkeypatch):
    package = "marketplace_agent.platforms.kijiji"  # built now; an empty module stands in
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setenv("MA_ENABLED_PLATFORMS", '["kijiji"]')

    output = _fails("tick")

    assert "The kijiji adapter is not built yet" in output


@pytest.mark.parametrize(
    ("name", "says"),
    [
        ("kijiji", "set MA_KIJIJI_EMAIL and MA_KIJIJI_PASSWORD"),
        ("facebook", "run `agent facebook login` once"),
    ],
)
def test_tick_on_a_platform_without_credentials_explains_what_to_change(
    data_dir, monkeypatch, name, says
):
    for var in ("MA_KIJIJI_EMAIL", "MA_KIJIJI_PASSWORD", "MA_FACEBOOK_PROFILE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MA_ENABLED_PLATFORMS", f'["{name}"]')

    output = _fails("tick")

    assert says in output and f"remove '{name}' from enabled_platforms" in output


def test_run_stops_cleanly_on_ctrl_c(data_dir, monkeypatch):
    def interrupted(self, stop=None, sleep=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(Runner, "run_forever", interrupted)

    output = _ok("run")

    assert "Running on fake" in output and "Stopped." in output


# ---------------------------------------------------------------- publish


def test_publish_lists_and_posts_the_item(item, db):
    output = _ok("publish", "1")

    assert "Published on fake at $120.00: https://fake.invalid/listings/" in output
    with db() as session:
        assert session.get(Item, 1).status == ItemStatus.LISTED


def test_publish_on_a_platform_that_is_not_enabled_fails(item):
    output = _fails("publish", "1", "--platform", "kijiji")

    assert "'kijiji' is not an enabled, connected platform" in output


def test_publish_with_no_enabled_platform_says_so(item, monkeypatch):
    monkeypatch.setenv("MA_ENABLED_PLATFORMS", "[]")

    assert "No platform is enabled" in _fails("publish", "1")


def test_publish_needs_a_title(item, db):
    with db() as session:
        session.get(Item, 1).title = ""
        session.commit()

    assert "has no title" in _fails("publish", "1")


@pytest.mark.parametrize(
    ("command", "error"),
    [
        (["publish", "9"], "There is no item with id 9."),
        (["undo", "9"], "There is no deal with id 9."),
        (["approve", "9"], "There is no approval with id 9."),
        (["reject", "9"], "There is no approval with id 9."),
        (["pause", "9"], "There is no item with id 9."),
        (["resume", "9"], "There is no item with id 9."),
        (["sold", "9"], "There is no item with id 9."),
        (["log", "9"], "Item 9 not found"),
        (["memory", "--item", "9"], "Item 9 not found"),
        (["price", "9", "100"], "There is no item with id 9."),
        (["floor", "9", "100"], "There is no item with id 9."),
        (["firmness", "9", "firm"], "There is no item with id 9."),
        (["withdraw", "9"], "There is no item with id 9."),
        (["gone", "9"], "There is no item with id 9."),
        (["resolve", "9", "ok"], "There is no escalation with id 9."),
        (["reply", "9", "hi"], "There is no conversation with id 9."),
    ],
)
def test_bad_ids_print_a_message_not_a_traceback(data_dir, command, error):
    assert error in _fails(*command)


@pytest.mark.parametrize(
    "command", ["undo", "approve", "reject", "pause", "sold", "log", "withdraw", "gone", "resolve"]
)
def test_ids_must_be_positive(data_dir, command):
    assert _invoke(command, "0").exit_code == 2


# ---------------------------------------------------------------- deals and approvals


def test_a_busy_database_asks_to_try_again_instead_of_a_traceback(listed, db, monkeypatch):
    """`agent run` can hold SQLite's write lock past busy_timeout while it acts."""
    deal = _deal(db, listed)

    def locked(self, commitment_id):
        raise OperationalError("UPDATE item", {}, sqlite3.OperationalError("database is locked"))

    monkeypatch.setattr(ControlService, "undo_commitment", locked)

    output = _fails("undo", str(deal))

    assert "The database is busy" in output and "Try again in a moment" in output


def test_undo_cancels_a_deal(listed, db):
    deal = _deal(db, listed)

    output = _ok("undo", str(deal))

    assert "Undid the $110.00 deal with Maya. Item 1 is listed again" in output
    assert "already released" in _fails("undo", str(deal))


def test_approve_runs_the_action(item, db):
    with db() as session:
        session.get(Item, item).status = ItemStatus.LISTED
        session.commit()
    approval = _approval(db, item, "publish", {"platform": "fake"})

    output = _ok("approve", str(approval))

    assert "Published on fake at $120.00" in output
    assert "already approved" in _fails("approve", str(approval))


def test_approve_that_cannot_run_exits_nonzero(listed, db):
    approval = _approval(db, listed, "update_price", {"price_cents": 5_000})

    assert "below the floor" in _fails("approve", str(approval))


def test_reject_with_a_note(listed, db):
    approval = _approval(db, listed, "update_price", {"price_cents": 10_000})

    output = _ok("reject", str(approval), "--note", "Too early to drop")

    assert f"Rejected approval {approval}." in output
    with db() as session:
        assert session.get(PendingApproval, approval).status == ApprovalStatus.REJECTED
        [memory] = session.exec(select(Memory)).all()
    assert memory.lesson.endswith("Too early to drop")


# ---------------------------------------------------------------- pause, resume, sold


def test_pause_and_resume(listed, db):
    assert "Paused item 1." in _ok("pause", "1")
    with db() as session:
        assert session.get(Item, 1).paused

    assert "Resumed item 1." in _ok("resume", "1")
    with db() as session:
        assert not session.get(Item, 1).paused


def test_sold_marks_the_item_and_keeps_the_price(listed, db):
    assert "Marked item 1 sold." in _ok("sold", "1", "--price", "105")

    with db() as session:
        assert session.get(Item, 1).status == ItemStatus.SOLD
        [memory] = session.exec(select(Memory)).all()
    assert memory.lesson.startswith("Sold [IKEA desk] for $105 on fake")
    assert "already sold" in _fails("sold", "1")


def test_noshow_releases_a_missed_pickup_on_the_owners_word(listed, db):
    deal = _deal(db, listed)
    clock = SystemClock()
    with db() as session:
        start = clock.now() - timedelta(hours=2)
        ledger.book_pickup(
            session, clock, accepted_id=deal, start=start, end=start + timedelta(hours=1)
        )
        session.commit()
    _ok("tick")  # the pickup is over: the owner is asked whether Maya came

    waiting = "did the buyer collect it? Run: agent sold 1, or agent noshow 1"
    assert waiting in _ok("log", "1")
    assert "Released the deal with Maya as a no-show" in _ok("noshow", "1")

    with db() as session:
        assert session.get(Item, 1).status == ItemStatus.LISTED
    assert waiting not in _ok("log", "1")
    assert "Item 1 has no active deal." in _fails("noshow", "1")


def test_sold_rejects_a_malformed_price(listed):
    assert _invoke("sold", "1", "--price", "a lot").exit_code == 2


# ---------------------------------------------------------------- log, trust, memory


def test_log_shows_runs_and_what_waits_on_the_owner(listed, db):
    deal = _deal(db, listed)
    approval = _approval(db, listed, "publish", {"platform": "facebook"})
    with db() as session:
        session.add(
            AgentRun(
                item_id=listed,
                started_at=datetime(2026, 10, 1, 22, 0, tzinfo=UTC),
                triggers_json=json.dumps(
                    [{"kind": "new_message", "detail": "1 new message(s) from Maya"}]
                ),
                rationale="Maya is close to the floor; counter at $110.",
                results_json=json.dumps(
                    [
                        {
                            "step": 1,
                            "tool": "send_message",
                            "args": {},
                            "status": "executed",
                            "detail": "Sent to Maya",
                        }
                    ]
                ),
                provider="gemini",
                requests_used=1,
            )
        )
        session.add(
            AgentRun(
                item_id=listed,
                started_at=datetime(2026, 10, 2, 22, 0, tzinfo=UTC),
                triggers_json="not json",
                error="step 1: ProviderError: down",
            )
        )
        session.commit()

    output = _ok("log", "1")

    lines = output.splitlines()
    assert lines[0] == "Run 1, 2026-10-01 18:00 EDT, quota healthy, gemini (1 request(s))"
    assert "  Woke for: new_message (1 new message(s) from Maya)" in lines
    assert "  Thinking: Maya is close to the floor; counter at $110." in lines
    assert "  - send_message executed: Sent to Maya" in lines
    assert "  Woke for: -" in lines and "  Error: step 1: ProviderError: down" in lines
    assert (
        f"Active deal {deal} at $110.00. Cancel it within the undo window: agent undo {deal}"
        in lines
    )
    assert f"agent approve {approval}, or agent reject {approval}" in output


def test_log_without_runs(listed):
    assert "The agent has not run on item 1 yet." in _ok("log", "1")


def test_trust_lists_records(data_dir, db):
    assert "No trust records yet" in _ok("trust")
    with db() as session:
        session.add(
            TrustRecord(
                platform="*",
                action="update_price",
                tier=TrustTier.ACT_AND_NOTIFY,
                executed=3,
                corrected=1,
                consecutive_clean=2,
                updated_at=SystemClock().now(),
            )
        )
        session.commit()

    output = _ok("trust")

    assert "PLATFORM  ACTION        TIER            EXECUTED  CORRECTED  CLEAN STREAK" in output
    assert "*         update_price  act and notify  3         1          2" in output


def test_memory_lists_lessons_and_filters_by_item(listed, db):
    assert "No lessons yet." in _ok("memory")
    now = SystemClock().now()
    with db() as session:
        session.add(Buyer(platform="fake", handle="omar"))
        session.add(
            Memory(
                scope=MemoryScope.ITEM, item_id=1, lesson="Desk buyers ask size.", created_at=now
            )
        )
        session.add(
            Memory(
                scope=MemoryScope.GLOBAL,
                lesson="Dimensions in titles help.",
                created_at=now,
                uses=4,
            )
        )
        session.add(
            Memory(
                scope=MemoryScope.BUYER,
                buyer_id=1,
                lesson="Haggles hard.",
                created_at=now,
                source="agent",
            )
        )
        session.commit()

    everything = _ok("memory")
    for_item = _ok("memory", "--item", "1")

    assert "1. [item 1] (agent, used 0x) Desk buyers ask size." in everything
    assert "2. [global] (agent, used 4x) Dimensions in titles help." in everything
    assert "3. [buyer 1] (agent, used 0x) Haggles hard." in everything
    assert for_item.strip() == "1. [item 1] (agent, used 0x) Desk buyers ask size."


# ---------------------------------------------------------------- price, floor, firmness, goal


def _goal_row(db) -> Goal:
    with db() as session:
        return session.exec(select(Goal)).one()


def _wakes(db) -> list[str]:
    with db() as session:
        return [w.reason for w in session.exec(select(WakeRequest).order_by(WakeRequest.id))]


def test_price_floor_and_firmness_go_through_the_owner_controls(listed, db):
    output = _ok("price", "1", "110")

    assert "Price changed from $120.00 to $110.00." in output
    with db() as session:
        [change] = session.exec(select(PriceChange)).all()
    assert (change.new_cents, change.actor) == (11_000, "human")
    assert _wakes(db)[-1] == "The owner set the asking price to $110.00."

    assert "below the floor of $95.00" in _fails("price", "1", "50")
    assert "Set the floor of item 1 to $100.00." in _ok("floor", "1", "100")
    assert "Set item 1 to firm." in _ok("firmness", "1", "FIRM")
    goal = _goal_row(db)
    assert (goal.list_price_cents, goal.floor_price_cents, goal.firmness) == (
        11_000,
        10_000,
        Firmness.FIRM,
    )
    assert _wakes(db)[-2:] == [
        "The owner set the floor to $100.00.",
        "The owner set the price firmness to firm.",
    ]
    assert "cannot be above the list price" in _fails("floor", "1", "500")
    assert _invoke("price", "1", "a lot").exit_code == 2
    assert _invoke("firmness", "1", "stubborn").exit_code == 2


def test_goal_on_a_draft_only_writes_the_goal(item, db):
    with db() as session:
        assert session.exec(select(PriceChange)).all() == []
    assert _wakes(db) == []


def test_goal_on_a_listed_item_reprices_through_the_tools(listed, db):
    output = _ok("goal", "1", "--price", "110", "--floor", "90", "--firmness", "flexible")

    assert "Goal:      list $110.00, target $110.00, floor $90.00" in output
    with db() as session:
        [change] = session.exec(select(PriceChange)).all()
    assert (change.old_cents, change.new_cents, change.actor) == (12_000, 11_000, "human")
    assert _wakes(db)[-1] == (
        "The owner set the goal: list $110.00, target $110.00, floor $90.00, flexible."
    )
    assert "floor price cannot be above the list price" in _fails(
        "goal", "1", "--price", "100", "--floor", "105"
    )
    assert _goal_row(db).floor_price_cents == 9_000


def test_withdraw_takes_the_item_off_the_market(listed, db):
    assert "Withdrew item 1." in _ok("withdraw", "1")
    with db() as session:
        assert session.get(Item, 1).status == ItemStatus.WITHDRAWN
    assert "already withdrawn" in _fails("withdraw", "1")


def test_gone_lets_the_agent_post_again(listed, db):
    assert "Marked the fake listing of item 1 as gone." in _ok("gone", "1")
    with db() as session:
        [listing] = session.exec(select(Listing)).all()
    assert listing.status == ListingStatus.DELISTED
    assert "Relist or publish on fake" in _wakes(db)[-1]
    assert "has no live listing" in _fails("gone", "1")


# ---------------------------------------------------------------- escalations and replies


def _buyer_thread(db, item_id: int) -> int:
    with db() as session:
        buyer = Buyer(platform="fake", handle="omar", display_name="Omar")
        session.add(buyer)
        session.flush()
        thread = Thread(
            platform="fake",
            remote_id="t-omar",
            item_id=item_id,
            buyer_id=buyer.id,
            created_at=SystemClock().now(),
        )
        session.add(thread)
        session.commit()
        return thread.id


def test_escalations_are_listed_shown_and_resolved(listed, db):
    thread_id = _buyer_thread(db, listed)
    with db() as session:
        session.add(
            Escalation(
                item_id=listed,
                thread_id=thread_id,
                reason="Omar asks for a hold until Friday.",
                created_at=SystemClock().now(),
            )
        )
        session.commit()

    listing = _ok("escalations")
    assert "Omar asks for a hold until Friday." in listing
    assert 'agent resolve ID "what the agent should do"' in listing
    log = _ok("log", "1")
    assert (
        'Waiting for you: escalation 1, Omar asks for a hold until Friday. Run: agent resolve 1 '
        f'"answer", or answer the buyer yourself: agent reply {thread_id} "message"'
    ) in log

    assert "Resolved escalation 1." in _ok("resolve", "1", "No holds, first come first served.")

    assert "No open escalations." in _ok("escalations")
    assert "escalation 1" not in _ok("log", "1")
    assert _wakes(db)[-1] == (
        "The owner answered escalation #1: No holds, first come first served."
    )
    assert "already resolved" in _fails("resolve", "1", "again")


def test_reply_sends_as_the_owner_through_the_guard(listed, db, monkeypatch):
    thread_id = _buyer_thread(db, listed)
    monkeypatch.setattr(FakeMarketplace, "send_message", lambda self, thread, body: "m-1")

    assert "phone number" in _fails("reply", str(thread_id), "Text me at 416-555-0199")
    _ok("reply", str(thread_id), "Yes, still available. Cash at pickup.")

    with db() as session:
        [sent] = session.exec(select(Message).where(Message.thread_id == thread_id)).all()
    assert (sent.body, sent.via, sent.direction) == (
        "Yes, still available. Cash at pickup.",
        "human",
        Direction.OUTBOUND,
    )
    assert _wakes(db)[-1] == f"The owner replied in thread {thread_id} themselves."
