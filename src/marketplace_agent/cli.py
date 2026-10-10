"""Command line entry point: `agent`."""

import json
import logging
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn
from zoneinfo import ZoneInfo

import typer
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, col, select

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import SystemClock
from marketplace_agent.core.db import init_db, make_engine, session_scope
from marketplace_agent.core.migrate import upgrade_schema
from marketplace_agent.core.items import (
    MAX_PRICE_CENTS,
    ItemNotFound,
    create_item,
    get_attributes,
    get_goal,
    get_item,
    item_photos,
    list_items,
    parse_hint_price,
    set_goal,
)
from marketplace_agent.core.models import (
    AgentRun,
    ApprovalStatus,
    Escalation,
    Firmness,
    Goal,
    Item,
    ItemStatus,
    Memory,
    PendingApproval,
    TrustTier,
)
from marketplace_agent.intake.photos import UnsupportedPhoto, ingest_photos

if TYPE_CHECKING:
    from marketplace_agent.ai.drafting import ListingDraft
    from marketplace_agent.core.runtime import Runtime
    from marketplace_agent.runner.tick import TickReport
    from marketplace_agent.sim.harness import SimReport

_DOLLARS = re.compile(r"\d+(?:\.\d{1,2})?")
_MAX_ID = 2**63 - 1  # SQLite INTEGER
_MAX_DAYS = 3650

ItemId = Annotated[int, typer.Argument(min=1, max=_MAX_ID, show_default=False)]
RecordId = Annotated[int, typer.Argument(min=1, max=_MAX_ID, show_default=False)]

# Locals in tracebacks would print Settings, which holds the pickup address.
app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Marketplace Agent: sells your stuff.",
)


@contextmanager
def _runtime() -> Iterator[tuple[Settings, Engine]]:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.database_url)
    try:
        init_db(engine)
        upgrade_schema(engine)
        yield settings, engine
    finally:
        engine.dispose()


def _fail(message: str) -> NoReturn:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)


def _cents(text: str) -> int:
    if not _DOLLARS.fullmatch(text):
        raise typer.BadParameter(f"expected dollars like 120 or 120.50, got {text!r}")
    cents = Decimal(text) * 100
    if cents > MAX_PRICE_CENTS:
        raise typer.BadParameter(f"at most {_money(MAX_PRICE_CENTS)}, got {text!r}")
    return int(cents)


def _money(cents: int) -> str:
    dollars, rest = divmod(cents, 100)
    return f"${dollars:,}.{rest:02d}"


def _local(when: datetime, settings: Settings) -> str:
    return when.astimezone(ZoneInfo(settings.timezone)).strftime("%Y-%m-%d %H:%M %Z")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [headers, *rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(headers))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip()
        for line in lines
    )


def _label(item: Item, limit: int = 50) -> str:
    text = item.title or item.hint
    return text if len(text) <= limit else text[: limit - 3] + "..."


@app.command()
def init() -> None:
    """Create the data directory and database."""
    with _runtime() as (settings, engine):
        typer.echo(f"Data dir: {settings.data_dir.resolve()}")
        typer.echo(f"Database: {engine.url.render_as_string(hide_password=True)}")


@app.command("add-item")
def add_item(
    hint: Annotated[str, typer.Option(help='What it is, e.g. "IKEA desk $120".')],
    photo: Annotated[
        list[Path],
        typer.Option(exists=True, dir_okay=False, readable=True, help="Photo file. Repeatable."),
    ],
) -> None:
    """Create an item from a hint and at least one photo."""
    with _runtime() as (settings, engine), session_scope(engine) as session:
        item = create_item(session, SystemClock(), hint)
        assert item.id is not None
        try:
            ingest_photos(session, settings, item.id, photo)
        except UnsupportedPhoto as exc:
            _fail(str(exc))
        stored = len(item_photos(session, item.id))
    typer.echo(f"Created item {item.id} with {stored} photo(s)")
    price = parse_hint_price(hint)
    if price is not None:
        typer.echo(f"Hint price: {_money(price)}")


@app.command("items")
def items_command(
    status: Annotated[ItemStatus | None, typer.Option(case_sensitive=False)] = None,
) -> None:
    """List items."""
    with _runtime() as (_, engine), Session(engine) as session:
        rows = [_item_row(session, item) for item in list_items(session, status)]
    typer.echo(_table(("ID", "STATUS", "ITEM", "LIST PRICE"), rows) if rows else "No items.")


def _item_row(session: Session, item: Item) -> tuple[str, str, str, str]:
    assert item.id is not None
    item_goal = get_goal(session, item.id)
    price = _money(item_goal.list_price_cents) if item_goal else "-"
    return str(item.id), item.status.value, _label(item), price


@app.command()
def show(item_id: ItemId) -> None:
    """Show an item with its goal and photos."""
    with _runtime() as (settings, engine), Session(engine) as session:
        try:
            item = get_item(session, item_id)
        except ItemNotFound as exc:
            _fail(str(exc))
        lines = [
            *_item_lines(item, settings),
            *_goal_lines(get_goal(session, item_id), settings),
            *_photo_lines(session, item_id),
        ]
    typer.echo("\n".join(lines))


def _item_lines(item: Item, settings: Settings) -> list[str]:
    fields = {
        "Item": str(item.id),
        "Status": item.status.value + (" (paused)" if item.paused else ""),
        "Hint": item.hint,
        "Title": item.title or "-",
        "Category": item.category or "-",
        "Condition": item.condition or "-",
        "Created": _local(item.created_at, settings),
        "Updated": _local(item.updated_at, settings),
    }
    lines = [f"{name + ':':<11}{value}" for name, value in fields.items()]
    lines += [f"  {key}: {value}" for key, value in get_attributes(item).items()]
    if item.description:
        lines += ["Description:", *(f"  {line}" for line in item.description.splitlines())]
    return lines


def _goal_lines(goal: Goal | None, settings: Settings) -> list[str]:
    if goal is None:
        return ["Goal:      none (set one with `agent goal`)"]
    prices = (
        f"list {_money(goal.list_price_cents)}, target {_money(goal.target_price_cents)}, "
        f"floor {_money(goal.floor_price_cents)}"
    )
    return [
        f"Goal:      {prices}",
        f"           {goal.firmness.value}, by {_local(goal.deadline, settings)}",
        *([f"           {goal.notes}"] if goal.notes else []),
    ]


def _photo_lines(session: Session, item_id: int) -> list[str]:
    photos = item_photos(session, item_id)
    return [f"Photos:    {len(photos)}"] + [
        f"  {p.position}. {p.path} ({p.width}x{p.height})" for p in photos
    ]


@app.command("goal")
def goal_command(
    item_id: ItemId,
    price: Annotated[int, typer.Option(parser=_cents, metavar="DOLLARS", help="List price.")],
    floor: Annotated[int, typer.Option(parser=_cents, metavar="DOLLARS", help="Hard minimum.")],
    target: Annotated[
        int | None,
        typer.Option(parser=_cents, metavar="DOLLARS", help="Hoped-for price. Default: list."),
    ] = None,
    days: Annotated[int, typer.Option(min=1, max=_MAX_DAYS, help="Days until the deadline.")] = 14,
    firmness: Annotated[Firmness, typer.Option(case_sensitive=False)] = Firmness.SLIGHT,
) -> None:
    """Set the selling goal. On an item for sale, a new price reaches every live listing."""
    with _runtime() as (settings, engine), session_scope(engine) as session:
        try:
            on_sale = get_item(session, item_id).status in (ItemStatus.LISTED, ItemStatus.COMMITTED)
            if not on_sale:
                saved = set_goal(
                    session,
                    SystemClock(),
                    item_id,
                    list_price_cents=price,
                    floor_price_cents=floor,
                    target_price_cents=target,
                    deadline_days=days,
                    firmness=firmness,
                )
        except (ItemNotFound, ValueError) as exc:
            _fail(str(exc))
    if on_sale:
        with _agent_runtime() as runtime:
            saved = _control(
                runtime.control.set_goal,
                item_id,
                list_price_cents=price,
                floor_price_cents=floor,
                target_price_cents=target,
                deadline_days=days,
                firmness=firmness,
            )
    typer.echo("\n".join(_goal_lines(saved, settings)))


@app.command()
def draft(item_id: ItemId) -> None:
    """Draft the listing from the item's photos with a vision model."""
    from marketplace_agent.ai.base import Capability, ProviderError
    from marketplace_agent.ai.drafting import draft_item
    from marketplace_agent.ai.factory import build_router
    from marketplace_agent.core.runtime import PlatformSetupError, build_platforms

    clock = SystemClock()
    with _runtime() as (settings, engine):
        router = build_router(settings, engine, clock)
        if not router.has_capability(Capability.VISION):
            _fail(
                "No vision model configured. Set GEMINI_API_KEY "
                "(free key at https://aistudio.google.com). Groq and OpenRouter are text only."
            )
        try:
            platforms = build_platforms(settings, clock)  # comps searches
        except PlatformSetupError:
            platforms = None  # drafting still works on cached comps
        with session_scope(engine) as session:
            try:
                listing = draft_item(session, router, settings, clock, item_id, platforms=platforms)
            except (ItemNotFound, ValueError, ProviderError) as exc:
                _fail(str(exc))
            item = get_item(session, item_id)
        typer.echo("\n".join([*_item_lines(item, settings), *_suggestion_lines(listing)]))


def _suggestion_lines(listing: "ListingDraft") -> list[str]:
    """What the draft returns but does not store on the Item."""
    bounds = (listing.suggested_price_low_cents, listing.suggested_price_high_cents)
    prices = sorted({cents for cents in bounds if cents is not None})
    lines = []
    if prices:
        lines.append(f"{'Suggested:':<11}{' to '.join(map(_money, prices))}")
        if listing.price_reasoning:
            lines.append(f"           {listing.price_reasoning}")
    if listing.uncertain:
        lines.append(f"{'Uncertain:':<11}{', '.join(listing.uncertain)}")
    return lines


@app.command()
def quota() -> None:
    """Show today's model requests and the router's quota state."""
    from marketplace_agent.ai.factory import build_router

    with _runtime() as (settings, engine):
        router = build_router(settings, engine, SystemClock())
        rows = [
            (
                usage.provider,
                usage.model,
                str(usage.requests),
                _limit(router.quota.remaining_today(usage.provider)),
                str(usage.tokens_in),
                str(usage.tokens_out),
            )
            for usage in router.quota.usage_today()
        ]
        headers = ("PROVIDER", "MODEL", "REQUESTS", "LEFT TODAY", "TOKENS IN", "TOKENS OUT")
        typer.echo(_table(headers, rows) if rows else "No requests today.")
        if not router.providers:
            typer.echo("State: exhausted (no model provider configured; set GEMINI_API_KEY)")
            return
        typer.echo(f"State: {router.state().value}")


def _limit(remaining: int | None) -> str:
    return "-" if remaining is None else str(remaining)


# ---------------------------------------------------------------- running the agent


DATABASE_BUSY = (
    "The database is busy: the running agent is in the middle of an action. Try again in a "
    "moment."
)


@contextmanager
def _agent_runtime() -> Iterator["Runtime"]:
    """The full runtime: settings, database, router, platforms, agent, runner, control."""
    from marketplace_agent.core.runtime import PlatformSetupError, build_runtime

    try:
        runtime = build_runtime(Settings())
    except PlatformSetupError as exc:
        _fail(str(exc))
    except OperationalError as exc:
        _fail_if_busy(exc)
    try:
        yield runtime
    finally:
        runtime.engine.dispose()


def _control[**P, T](action: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a ControlService call; a refusal ends the command with its message, no traceback.
    `agent run` may hold SQLite's write lock for a moment; past busy_timeout, the command
    stops with a "try again" instead of a traceback, having changed nothing."""
    from marketplace_agent.control.service import ControlError

    try:
        return action(*args, **kwargs)
    except ControlError as exc:
        _fail(str(exc))
    except OperationalError as exc:
        _fail_if_busy(exc)


def _fail_if_busy(exc: OperationalError) -> NoReturn:
    if "locked" in str(exc.orig) or "busy" in str(exc.orig):
        _fail(DATABASE_BUSY)
    raise exc


NO_MODEL_WARNING = (
    "No model is configured (set GEMINI_API_KEY): only availability questions get an answer, "
    "and other buyer messages are escalated to you."
)


def _warn_without_model(runtime: "Runtime") -> None:
    if not runtime.router.providers:
        typer.secho(NO_MODEL_WARNING, err=True, fg=typer.colors.YELLOW)


@app.command()
def run() -> None:
    """Run the agent until Ctrl+C: read messages, reply, wake the agent, repeat."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with _agent_runtime() as runtime:
        _warn_without_model(runtime)
        platforms = ", ".join(runtime.platforms) or "no platform"
        typer.echo(
            f"Running on {platforms}, a tick about every {runtime.settings.tick_seconds}s. "
            "Ctrl+C stops."
        )
        try:
            runtime.runner.run_forever()
        except KeyboardInterrupt:
            typer.echo("Stopped.")


@app.command()
def tick() -> None:
    """Run one tick: read messages, answer the simple ones, wake the agent where needed."""
    with _agent_runtime() as runtime:
        _warn_without_model(runtime)
        report = runtime.runner.tick()
        typer.echo("\n".join(_tick_lines(report, runtime.settings)))


def _tick_lines(report: "TickReport", settings: Settings) -> list[str]:
    fields = {
        "Tick": _local(report.started_at, settings),
        "New messages": str(report.ingested),
        "Cache replies": str(report.cache_replies),
        "Wakes": str(report.wakes),
        "Escalated": str(report.escalated),
        "Courtesy": str(report.courtesy_replies),
        "Taken down": str(report.takedowns),
        "Quota": report.quota_state,
        "Errors": str(len(report.errors)) if report.errors else "none",
    }
    lines = [f"{name + ':':<15}{value}" for name, value in fields.items()]
    return lines + [f"  {error}" for error in report.errors]


@app.command()
def publish(
    item_id: ItemId,
    platform: Annotated[
        list[str] | None,
        typer.Option(help="Platform to post on. Repeatable. Default: every enabled platform."),
    ] = None,
) -> None:
    """Confirm an item for sale and post it."""
    with _agent_runtime() as runtime:
        results = _control(runtime.control.confirm_and_publish, item_id, platform)
    if not results:
        _fail("No platform is enabled. Set enabled_platforms, or pass --platform.")
    for result in results:
        typer.echo(result.summary)
    if not any(result.ok for result in results):
        raise typer.Exit(1)


@app.command()
def undo(commitment_id: RecordId) -> None:
    """Undo a deal the agent made, within its undo window."""
    with _agent_runtime() as runtime:
        typer.echo(_control(runtime.control.undo_commitment, commitment_id))


@app.command()
def approve(approval_id: RecordId) -> None:
    """Approve an action the agent asked permission for, and run it."""
    with _agent_runtime() as runtime:
        result = _control(runtime.control.approve, approval_id)
    typer.echo(result.summary)
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def reject(
    approval_id: RecordId,
    note: Annotated[str, typer.Option(help="Why, so the agent learns.")] = "",
) -> None:
    """Reject an action the agent asked permission for."""
    with _agent_runtime() as runtime:
        _control(runtime.control.reject, approval_id, note)
    typer.echo(f"Rejected approval {approval_id}.")


@app.command()
def pause(item_id: ItemId) -> None:
    """Stop the agent acting on an item until you resume it."""
    with _agent_runtime() as runtime:
        _control(runtime.control.pause, item_id)
    typer.echo(f"Paused item {item_id}. The agent takes no action on it until you resume it.")


@app.command()
def resume(item_id: ItemId) -> None:
    """Let the agent act on a paused item again."""
    with _agent_runtime() as runtime:
        _control(runtime.control.resume, item_id)
    typer.echo(f"Resumed item {item_id}.")


@app.command()
def sold(
    item_id: ItemId,
    price: Annotated[
        int | None,
        typer.Option(
            parser=_cents,
            metavar="DOLLARS",
            help="What it sold for; for a bundle, what the whole bundle sold for.",
        ),
    ] = None,
) -> None:
    """Mark an item sold and take its listings down."""
    from marketplace_agent.control.service import takedown_note

    with _agent_runtime() as runtime:
        still_up = _control(runtime.control.mark_sold, item_id, price)
    typer.echo(f"Marked item {item_id} sold.{takedown_note(still_up)}")


@app.command()
def noshow(item_id: ItemId) -> None:
    """The buyer did not come to the booked pickup: release the deal as a no-show."""
    with _agent_runtime() as runtime:
        typer.echo(_control(runtime.control.mark_no_show, item_id))


@app.command()
def price(
    item_id: ItemId,
    dollars: Annotated[
        int, typer.Argument(parser=_cents, metavar="DOLLARS", show_default=False)
    ],
) -> None:
    """Change the asking price on the goal and on every live listing."""
    with _agent_runtime() as runtime:
        result = _control(runtime.control.set_price, item_id, dollars)
    typer.echo(result.summary)
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def floor(
    item_id: ItemId,
    dollars: Annotated[
        int, typer.Argument(parser=_cents, metavar="DOLLARS", show_default=False)
    ],
) -> None:
    """Change the lowest price the agent may agree to."""
    with _agent_runtime() as runtime:
        _control(runtime.control.set_floor, item_id, dollars)
    typer.echo(f"Set the floor of item {item_id} to {_money(dollars)}. The agent will re-plan.")


@app.command()
def firmness(
    item_id: ItemId,
    level: Annotated[Firmness, typer.Argument(case_sensitive=False, show_default=False)],
) -> None:
    """Change how far the agent may negotiate: firm, slight, or flexible."""
    with _agent_runtime() as runtime:
        _control(runtime.control.set_firmness, item_id, level)
    typer.echo(f"Set item {item_id} to {level.value}. The agent will re-plan.")


@app.command()
def withdraw(item_id: ItemId) -> None:
    """Take an item off the market: cancel any deal and take its listings down."""
    from marketplace_agent.control.service import takedown_note

    with _agent_runtime() as runtime:
        still_up = _control(runtime.control.withdraw, item_id)
    typer.echo(f"Withdrew item {item_id}.{takedown_note(still_up)}")


@app.command()
def gone(
    item_id: ItemId,
    platform: Annotated[
        str | None,
        typer.Option(help="Platform whose post is gone. Needed when it is live on several."),
    ] = None,
) -> None:
    """A post is no longer up (removed, expired, deleted by hand): let the agent post it again."""
    with _agent_runtime() as runtime:
        typer.echo(_control(runtime.control.mark_listing_gone, item_id, platform))


@app.command()
def escalations() -> None:
    """List what the agent is waiting on you for."""
    with _runtime() as (settings, engine), Session(engine) as session:
        query = (
            select(Escalation)
            .where(col(Escalation.resolved_at).is_(None))
            .order_by(col(Escalation.id))
        )
        rows = [
            (
                str(e.id),
                str(e.item_id or "-"),
                str(e.thread_id or "-"),
                _local(e.created_at, settings),
                e.reason,
            )
            for e in session.exec(query).all()
        ]
    if not rows:
        typer.echo("No open escalations.")
        return
    typer.echo(_table(("ID", "ITEM", "THREAD", "OPENED", "REASON"), rows))
    typer.echo('Answer one with: agent resolve ID "what the agent should do"')


@app.command()
def resolve(
    escalation_id: RecordId,
    answer: Annotated[
        str, typer.Argument(help="What the agent should know or do.", show_default=False)
    ] = "",
) -> None:
    """Answer an escalation; the agent re-plans with your answer."""
    with _agent_runtime() as runtime:
        _control(runtime.control.resolve_escalation, escalation_id, answer)
    typer.echo(f"Resolved escalation {escalation_id}.")


@app.command()
def reply(
    thread_id: RecordId,
    message: Annotated[str, typer.Argument(help="Text to send.", show_default=False)],
) -> None:
    """Message a buyer yourself. Write [PICKUP_ADDRESS] to share the address."""
    with _agent_runtime() as runtime:
        result = _control(runtime.control.reply_as_human, thread_id, message)
    typer.echo(result.summary)
    if not result.ok:
        raise typer.Exit(1)


@app.command("log")
def log_command(
    item_id: ItemId,
    limit: Annotated[int, typer.Option(min=1, max=1000, help="How many runs to show.")] = 10,
) -> None:
    """Show the agent's recent runs on an item: why it woke, what it thought, what it did."""
    with _runtime() as (settings, engine), Session(engine) as session:
        try:
            get_item(session, item_id)
        except ItemNotFound as exc:
            _fail(str(exc))
        query = (
            select(AgentRun)
            .where(AgentRun.item_id == item_id)
            .order_by(col(AgentRun.started_at).desc(), col(AgentRun.id).desc())
            .limit(limit)
        )
        runs = list(reversed(session.exec(query).all()))
        lines = [line for run in runs for line in _run_lines(run, settings)]
        lines += _waiting_lines(session, item_id)
    typer.echo("\n".join(lines) if lines else f"The agent has not run on item {item_id} yet.")


def _run_lines(run: AgentRun, settings: Settings) -> list[str]:
    head = f"Run {run.id}, {_local(run.started_at, settings)}, quota {run.quota_state}"
    if run.provider:
        head += f", {run.provider} ({run.requests_used} request(s))"
    woke = "; ".join(
        f"{t.get('kind')} ({t['detail']})" if t.get("detail") else str(t.get("kind"))
        for t in _json_list(run.triggers_json)
    )
    lines = [head, f"  Woke for: {woke or '-'}"]
    if run.rationale:
        lines.append(f"  Thinking: {run.rationale}")
    lines += [
        f"  - {o.get('tool')} {o.get('status')}: {o.get('detail')}"
        for o in _json_list(run.results_json)
    ]
    if run.error:
        lines.append(f"  Error: {run.error}")
    return lines


def _waiting_lines(session: Session, item_id: int) -> list[str]:
    """Ids the owner needs for undo, approve and reject."""
    from marketplace_agent.guard.handover import open_check
    from marketplace_agent.guard.ledger import active_commitment, deal_rows

    lines = []
    active = active_commitment(session, item_id)
    if active is not None:
        accepted = active.accepted
        rows = deal_rows(session, accepted)
        bundle = ""
        if len(rows) > 1:
            others = ", ".join(str(row.item_id) for row in rows if row.item_id != item_id)
            total = _money(sum(row.price_cents or 0 for row in rows))
            bundle = f" in a bundle with items {others}, {total} in total"
        lines.append(
            f"Active deal {accepted.id} at {_money(accepted.price_cents or 0)}{bundle}. "
            f"Cancel it within the undo window: agent undo {accepted.id}"
        )
        if accepted.id is not None and open_check(session, accepted.id) is not None:
            it = "them" if len(rows) > 1 else "it"
            lines.append(
                f"Waiting for you: did the buyer collect {it}? Run: agent sold {item_id}, or "
                f"agent noshow {item_id}"
            )
    pending = session.exec(
        select(PendingApproval)
        .where(
            PendingApproval.item_id == item_id,
            PendingApproval.status == ApprovalStatus.PENDING,
        )
        .order_by(col(PendingApproval.id))
    ).all()
    lines += [
        f"Waiting for you: approval {a.id}, {a.tool} {a.args_json}. "
        f"Run: agent approve {a.id}, or agent reject {a.id}"
        for a in pending
    ]
    open_escalations = session.exec(
        select(Escalation)
        .where(Escalation.item_id == item_id, col(Escalation.resolved_at).is_(None))
        .order_by(col(Escalation.id))
    ).all()
    for e in open_escalations:
        reason = e.reason.strip().rstrip(".")
        line = f'Waiting for you: escalation {e.id}, {reason}. Run: agent resolve {e.id} "answer"'
        if e.thread_id is not None:
            line += f', or answer the buyer yourself: agent reply {e.thread_id} "message"'
        lines.append(line)
    return lines


def _json_list(text: str) -> list[dict]:
    try:
        value = json.loads(text or "[]")
    except json.JSONDecodeError:
        return []
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


@app.command()
def trust() -> None:
    """Show how much the agent may do without asking, per platform and action."""
    from marketplace_agent.guard.trust import summary

    with _runtime() as (_, engine), Session(engine) as session:
        rows = [
            (
                record.platform,
                record.action,
                TrustTier(record.tier).name.lower().replace("_", " "),
                str(record.executed),
                str(record.corrected),
                str(record.consecutive_clean),
            )
            for record in summary(session)
        ]
    headers = ("PLATFORM", "ACTION", "TIER", "EXECUTED", "CORRECTED", "CLEAN STREAK")
    typer.echo(_table(headers, rows) if rows else "No trust records yet; defaults apply.")


@app.command("memory")
def memory_command(
    item: Annotated[
        int | None,
        typer.Option(min=1, max=_MAX_ID, help="Only lessons learned on this item."),
    ] = None,
) -> None:
    """Show what the agent has learned."""
    with _runtime() as (_, engine), Session(engine) as session:
        query = select(Memory).order_by(col(Memory.id))
        if item is not None:
            try:
                get_item(session, item)
            except ItemNotFound as exc:
                _fail(str(exc))
            query = query.where(Memory.item_id == item)
        lines = [_memory_line(memory) for memory in session.exec(query).all()]
    typer.echo("\n".join(lines) if lines else "No lessons yet.")


def _memory_line(memory: Memory) -> str:
    target = {"item": f" {memory.item_id}", "buyer": f" {memory.buyer_id}"}
    scope = memory.scope.value + target.get(memory.scope.value, "")
    return f"{memory.id}. [{scope}] ({memory.source}, used {memory.uses}x) {memory.lesson}"


@app.command()
def simulate(
    scenario: Annotated[
        str,
        typer.Option(
            help="desk, lowballers, scam_wave, no_show_recovery, injection, firm, or bundle.",
        ),
    ] = "desk",
    days: Annotated[
        int | None,
        typer.Option(min=1, max=60, help="Simulated days. Default: the scenario's (14)."),
    ] = None,
    seed: Annotated[int, typer.Option(min=0, help="The same seed replays the same run.")] = 0,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Decide with the real model router (spends quota) instead of the offline policy.",
        ),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Print only the report.")] = False,
) -> None:
    """Sell to scripted buyers on a fake marketplace, then audit every rule."""
    from marketplace_agent.sim.harness import LiveModelMissing, run_simulation, sim_settings
    from marketplace_agent.sim.scenarios import SCENARIOS

    chosen = SCENARIOS.get(scenario)
    if chosen is None:
        _fail(f"Unknown scenario {scenario!r}. Choose one of: {', '.join(SCENARIOS)}.")
    settings = sim_settings(live=live)
    try:
        report = run_simulation(chosen, seed=seed, days=days, settings=settings, live=live)
    except LiveModelMissing as exc:
        _fail(str(exc))
    if not quiet:
        typer.echo("\n".join([*report.transcript, ""]))
    typer.echo("\n".join(_sim_lines(report)))
    if report.violations:
        raise typer.Exit(1)


def _sim_lines(report: "SimReport") -> list[str]:
    sale = "no"
    if report.sold and report.sale_price_cents is not None:
        sale = f"yes, {_money(report.sale_price_cents)} after {report.days_to_sell} days"
    fields = {
        "Scenario": f"{report.scenario} (seed {report.seed}, {report.days_run} days simulated)",
        "Sold": sale,
        "Model requests": f"{report.model_requests} (at most {report.max_requests_per_day} a day)",
        "Cache replies": str(report.cache_replies),
        "Messages sent": str(report.outbound),
        "Escalations": str(report.escalations),
        "Errors": str(len(report.errors)) if report.errors else "none",
        "Violations": str(len(report.violations)) if report.violations else "none",
    }
    lines = [f"{name + ':':<16}{value}" for name, value in fields.items()]
    return lines + [f"  {line}" for line in report.errors + report.violations]


# ---------------------------------------------------------------- stages 3 to 6
# Each module imports its heavy dependencies (uvicorn, discord.py, Playwright) inside the
# command, so `agent --help` and the commands above never load them.


def _register_stage_commands() -> None:
    from marketplace_agent.notify.commands import register as register_serve
    from marketplace_agent.platforms.facebook.commands import register as register_facebook
    from marketplace_agent.platforms.kijiji.commands import register as register_kijiji
    from marketplace_agent.web.commands import register as register_web

    register_web(app)
    register_serve(app)
    register_kijiji(app)
    register_facebook(app)


_register_stage_commands()
