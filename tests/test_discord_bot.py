"""notify.discord_bot: the thread-safe notifier, cards as Discord objects, and the interaction
glue, driven with fake interactions and channels over a real runtime. Nothing talks to Discord."""

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import discord
import pytest
from discord.components import ActionRow
from sqlmodel import select

from marketplace_agent.core.items import set_goal
from marketplace_agent.core.models import (
    ApprovalStatus,
    Item,
    Memory,
    PendingApproval,
    Photo,
)
from marketplace_agent.core.runtime import build_runtime
from marketplace_agent.notify import discord_bot as bot_module
from marketplace_agent.notify import discord_views as views
from marketplace_agent.notify.base import Notice, NoticeKind, RecordingNotifier
from marketplace_agent.notify.discord_bot import (
    ActionButton,
    ActionModal,
    DiscordBot,
    DiscordNotifier,
    disabled_view,
)
from marketplace_agent.platforms.fake import FakeMarketplace

OWNER, STRANGER = 4242, 777
CHANNEL_ID, GUILD_ID = 1234, 55


def _notice(n: int, kind: NoticeKind = NoticeKind.INFO) -> Notice:
    return Notice(kind=kind, title=f"n{n}", body=f"body {n}")


# ---------------------------------------------------------------- DiscordNotifier


class FakeLoop:
    """Records call_soon_threadsafe calls; run() replays them as the loop thread would."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.closed = False
        self._lock = threading.Lock()

    def call_soon_threadsafe(self, callback, *args) -> None:
        if self.closed:
            raise RuntimeError("Event loop is closed")
        with self._lock:
            self.calls.append((callback, args))

    def run(self) -> None:
        for callback, args in self.calls:
            callback(*args)
        self.calls.clear()


def _drain(queue: asyncio.Queue) -> list[Notice]:
    found = []
    while not queue.empty():
        found.append(queue.get_nowait())
    return found


def test_notifier_buffers_until_the_bot_attaches():
    notifier = DiscordNotifier(fallback=RecordingNotifier())
    loop, queue = FakeLoop(), asyncio.Queue()
    notifier.notify(_notice(1))
    notifier.notify(_notice(2))

    notifier.attach(loop, queue)
    notifier.notify(_notice(3))

    assert [n.title for n in _drain(queue)] == ["n1", "n2"]
    assert len(loop.calls) == 1
    loop.run()
    assert [n.title for n in _drain(queue)] == ["n3"]


def test_notifier_is_thread_safe_and_keeps_each_threads_order():
    notifier = DiscordNotifier(fallback=RecordingNotifier())
    loop, queue = FakeLoop(), asyncio.Queue()
    threads, per_thread = 8, 300
    started = threading.Barrier(threads + 1)

    def produce(worker: int) -> None:
        started.wait()
        for n in range(per_thread):
            notifier.notify(Notice(kind=NoticeKind.INFO, title=f"{worker}", body=f"{n}"))

    workers = [threading.Thread(target=produce, args=(w,)) for w in range(threads)]
    for worker in workers:
        worker.start()
    started.wait()
    notifier.attach(loop, queue)  # while the producers run
    for worker in workers:
        worker.join()
    loop.run()

    got = _drain(queue)
    assert len(got) == threads * per_thread
    for worker in range(threads):
        bodies = [int(n.body) for n in got if n.title == f"{worker}"]
        assert bodies == list(range(per_thread))


async def test_notifier_hands_notices_to_a_real_loop_from_other_threads():
    notifier = DiscordNotifier(fallback=RecordingNotifier())
    queue: asyncio.Queue[Notice] = asyncio.Queue()
    notifier.attach(asyncio.get_running_loop(), queue)

    workers = [
        threading.Thread(target=notifier.notify, args=(_notice(n),)) for n in range(20)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    got = [await asyncio.wait_for(queue.get(), 1) for _ in range(20)]

    assert sorted(n.title for n in got) == sorted(f"n{n}" for n in range(20))


def test_closed_notifier_sends_everything_left_to_the_fallback():
    fallback = RecordingNotifier()
    notifier = DiscordNotifier(fallback=fallback)
    notifier.notify(_notice(1))

    notifier.close(unsent=[_notice(0)])
    notifier.notify(_notice(2))

    assert [n.title for n in fallback.notices] == ["n0", "n1", "n2"]


def test_notice_for_a_closed_loop_goes_to_the_fallback():
    fallback = RecordingNotifier()
    notifier = DiscordNotifier(fallback=fallback)
    loop = FakeLoop()
    notifier.attach(loop, asyncio.Queue())
    loop.closed = True

    notifier.notify(_notice(1))

    assert [n.title for n in fallback.notices] == ["n1"]


# ---------------------------------------------------------------- cards as Discord objects


CARD = views.Card(
    title="Deal: desk",
    description="Promised.",
    color=0x2ECC71,
    fields=(views.CardField("Pickup", "Not booked yet", inline=True),),
    buttons=(
        views.Button("Undo", "ma:undo:3", "danger"),
        views.Button("Pause item", "ma:pause:1", disabled=True),
    ),
)


def test_card_becomes_an_embed_and_persistent_buttons():
    embed = bot_module._embed(CARD)
    view = bot_module._view(CARD)

    assert (embed.title, embed.description, embed.color.value) == (
        "Deal: desk", "Promised.", 0x2ECC71
    )
    fields = [(f.name, f.value, f.inline) for f in embed.fields]
    assert fields == [("Pickup", "Not booked yet", True)]
    assert view.timeout is None and view.is_persistent()
    assert all(isinstance(item, ActionButton) for item in view.children)
    assert [
        (item.item.label, item.custom_id, item.item.style, item.item.disabled)
        for item in view.children
    ] == [
        ("Undo", "ma:undo:3", discord.ButtonStyle.danger, False),
        ("Pause item", "ma:pause:1", discord.ButtonStyle.secondary, True),
    ]


def test_a_card_without_buttons_has_no_view_and_empty_text_is_left_out():
    card = views.Card(title="Alert", description="", color=1)

    assert bot_module._view(card) is None
    assert bot_module._embed(card).description is None


def test_text_embed_takes_its_title_from_the_first_paragraph():
    embed = bot_module._text_embed("Digest for Thu Oct 1\n\nSold today\nnone")

    assert (embed.title, embed.description) == ("Digest for Thu Oct 1", "Sold today\nnone")
    assert bot_module._text_embed("a\n\nb", "Status").title == "Status"


def _message_with(*custom_ids: str) -> SimpleNamespace:
    row = ActionRow(
        {
            "type": 1,
            "components": [
                {"type": 2, "style": 4, "label": f"b{n}", "custom_id": cid}
                for n, cid in enumerate(custom_ids)
            ],
        }
    )
    return SimpleNamespace(id=1, components=[row])


def test_disabled_view_switches_off_every_agent_button():
    view = disabled_view(_message_with("ma:sold:2", "ma:noshow:2", "someone-else"))

    assert [(item.custom_id, item.item.disabled) for item in view.children] == [
        ("ma:sold:2", True),
        ("ma:noshow:2", True),
    ]


async def test_action_button_is_rebuilt_from_its_custom_id():
    button = discord.ui.Button(label="Undo", custom_id="ma:undo:12")
    match = ActionButton.__discord_ui_compiled_template__.fullmatch("ma:undo:12")

    rebuilt = await ActionButton.from_custom_id(None, button, match)

    assert isinstance(rebuilt, ActionButton) and rebuilt.custom_id == "ma:undo:12"
    assert ActionButton.__discord_ui_compiled_template__.fullmatch("other:undo:12") is None


def test_every_slash_command_from_the_spec_exists():
    names = {command.name for command in bot_module.COMMANDS}

    assert names == {
        "items", "item", "price", "floor", "firm", "pause", "resume", "sold", "undo", "approve",
        "status", "digest",
    }
    assert all(len(command.description) <= 100 for command in bot_module.COMMANDS)


# ---------------------------------------------------------------- interactions


class FakeResponse:
    def __init__(self, log: list) -> None:
        self.log = log

    async def defer(self, **kwargs) -> None:
        self.log.append(("defer", kwargs))

    async def send_message(self, content: str, **kwargs) -> None:
        self.log.append(("send_message", content, kwargs))

    async def send_modal(self, modal) -> None:
        self.log.append(("send_modal", modal))


class FakeFollowup:
    def __init__(self, log: list) -> None:
        self.log = log

    async def send(self, content: str | None = None, **kwargs) -> None:
        self.log.append(("followup", content, kwargs))


class FakeInteraction:
    def __init__(self, client, user_id: int, message=None) -> None:
        self.client = client
        self.user = SimpleNamespace(id=user_id)
        self.message = message
        self.log: list = []
        self.response = FakeResponse(self.log)
        self.followup = FakeFollowup(self.log)

    async def edit_original_response(self, **kwargs) -> None:
        self.log.append(("edit", kwargs))


@pytest.fixture
def fallback() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
def runtime(settings, engine, clock, fallback):
    settings.discord_owner_id = OWNER
    return build_runtime(
        settings,
        clock=clock,
        engine=engine,
        notifier=DiscordNotifier(fallback=fallback),
        platforms={"fake": FakeMarketplace(clock)},
    )


@pytest.fixture
def bot(runtime) -> DiscordBot:
    return DiscordBot(runtime, runtime.notifier)


@pytest.fixture
def item_id(runtime, session, clock) -> int:
    now = clock.now()
    item = Item(title="IKEA desk", description="Solid.", created_at=now, updated_at=now)
    session.add(item)
    session.flush()
    session.add(Photo(item_id=item.id, path="photos/1.jpg", width=8, height=8, sha256="a"))
    set_goal(session, clock, item.id, list_price_cents=12_000, floor_price_cents=9_500)
    session.commit()
    [result] = runtime.control.confirm_and_publish(item.id)
    assert result.ok
    return item.id


def _park(session, clock, item_id: int, price_cents: int = 10_500) -> int:
    approval = PendingApproval(
        item_id=item_id,
        tool="update_price",
        args_json=json.dumps({"price_cents": price_cents}),
        created_at=clock.now(),
    )
    session.add(approval)
    session.commit()
    return approval.id


async def test_a_stranger_gets_an_ephemeral_refusal_and_nothing_runs(bot, session, clock, item_id):
    approval_id = _park(session, clock, item_id)
    click = FakeInteraction(bot, STRANGER, _message_with(f"ma:approve:{approval_id}"))

    await bot.on_button(click, f"ma:approve:{approval_id}")

    assert click.log == [("send_message", views.NOT_OWNER, {"ephemeral": True})]
    assert session.get(PendingApproval, approval_id).status == ApprovalStatus.PENDING


async def test_approve_defers_then_runs_and_switches_the_card_off(bot, session, clock, item_id):
    approval_id = _park(session, clock, item_id)
    custom_id = f"ma:approve:{approval_id}"
    click = FakeInteraction(bot, OWNER, _message_with(custom_id, f"ma:reject:{approval_id}"))

    await bot.on_button(click, custom_id)

    assert click.log[0] == ("defer", {})
    [(_, edit)] = [entry for entry in click.log if entry[0] == "edit"]
    assert [item.item.disabled for item in edit["view"].children] == [True, True]
    kind, text, kwargs = click.log[-1]
    assert kind == "followup" and text.startswith(f"Approved #{approval_id}: ")
    assert kwargs == {"ephemeral": False}
    session.expire_all()
    assert session.get(PendingApproval, approval_id).status == ApprovalStatus.APPROVED


async def test_reject_opens_a_modal_and_its_note_reaches_the_agent(bot, session, clock, item_id):
    approval_id = _park(session, clock, item_id)
    custom_id = f"ma:reject:{approval_id}"
    click = FakeInteraction(bot, OWNER, _message_with(f"ma:approve:{approval_id}", custom_id))

    await bot.on_button(click, custom_id)

    [(kind, modal)] = click.log
    assert kind == "send_modal"
    assert isinstance(modal, ActionModal) and modal.title == f"Reject #{approval_id}"
    assert list(modal.inputs) == ["note"] and not modal.inputs["note"].required
    modal.inputs["note"]._value = "Too early to drop."  # what Discord fills in on submit
    submit = FakeInteraction(bot, OWNER, click.message)
    await modal.on_submit(submit)
    assert submit.log[0] == ("defer", {})
    assert submit.log[-1] == (
        "followup",
        f"Rejected #{approval_id}. Note: Too early to drop.",
        {"ephemeral": False},
    )
    session.expire_all()
    assert session.get(PendingApproval, approval_id).status == ApprovalStatus.REJECTED
    assert session.exec(select(Memory)).one().lesson.endswith("Too early to drop.")


async def test_pause_answers_without_closing_the_card(bot, session, item_id):
    click = FakeInteraction(bot, OWNER, _message_with(f"ma:pause:{item_id}"))

    await bot.on_button(click, f"ma:pause:{item_id}")

    assert [entry[0] for entry in click.log] == ["defer", "followup"]
    session.expire_all()
    assert session.get(Item, item_id).paused


async def test_a_bug_in_an_action_still_answers_the_click(bot, item_id, monkeypatch):
    def broken(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(views, "handle_action", broken)
    click = FakeInteraction(bot, OWNER, _message_with(f"ma:undo:{item_id}"))

    await bot.on_button(click, f"ma:undo:{item_id}")

    assert click.log[-1] == ("followup", bot_module.ACTION_FAILED, {"ephemeral": True})


async def test_slash_command_runs_the_same_action(bot, session, item_id):
    command = FakeInteraction(bot, OWNER)

    await bot_module.floor_command.callback(command, item_id, "90")

    assert command.log == [
        ("defer", {"ephemeral": True, "thinking": True}),
        (
            "followup",
            f"Set the floor of item #{item_id} to $90.00. The agent will re-plan.",
            {"ephemeral": False},
        ),
    ]
    refused = FakeInteraction(bot, STRANGER)
    await bot_module.pause_command.callback(refused, item_id)
    assert refused.log == [("send_message", views.NOT_OWNER, {"ephemeral": True})]


async def test_read_only_commands_reply_with_embeds(bot, item_id):
    listing = FakeInteraction(bot, OWNER)
    card = FakeInteraction(bot, OWNER)
    missing = FakeInteraction(bot, OWNER)

    await bot_module.items_command.callback(listing)
    await bot_module.item_command.callback(card, item_id)
    await bot_module.item_command.callback(missing, 999)

    _, _, sent = listing.log[-1]
    assert sent["embed"].title == "Items" and "IKEA desk" in sent["embed"].description
    _, _, shown = card.log[-1]
    assert shown["embed"].title == f"#{item_id} IKEA desk"
    assert [item.custom_id for item in shown["view"].children] == [f"ma:pause:{item_id}"]
    assert missing.log[-1] == ("followup", "Item 999 not found", {"ephemeral": True})


async def test_status_and_digest_commands(bot):
    status = FakeInteraction(bot, OWNER)
    digest = FakeInteraction(bot, OWNER)

    await bot_module.status_command.callback(status)
    await bot_module.digest_command.callback(digest)

    assert status.log[-1][2]["embed"].title == "Status"
    assert digest.log[-1][2]["embed"].title == "Digest for Thu Oct 1"


# ---------------------------------------------------------------- through discord.py itself


USER = {"id": str(STRANGER), "username": "x", "discriminator": "0", "avatar": None,
        "global_name": None}


def _interaction_payload(kind: int, data: dict, custom_id: str = "ma:undo:5") -> dict:
    """An INTERACTION_CREATE as the gateway sends it, for a card posted before a restart."""
    button = {"type": 2, "style": 4, "label": "Undo", "custom_id": custom_id}
    return {
        "id": "1", "application_id": "2", "type": kind, "token": "t", "version": 1,
        "channel_id": "10", "user": USER, "data": data, "locale": "en-US", "entitlements": [],
        "authorizing_integration_owners": {}, "app_permissions": "0",
        "attachment_size_limit": 8_000_000,
        "message": {
            "id": "20", "channel_id": "10", "author": {**USER, "id": "2", "bot": True},
            "content": "", "timestamp": "2026-10-01T16:00:00+00:00", "edited_timestamp": None,
            "tts": False, "mention_everyone": False, "mentions": [], "mention_roles": [],
            "attachments": [], "embeds": [], "pinned": False, "type": 0, "flags": 0,
            "components": [{"type": 1, "components": [button]}],
        },
    }


async def test_setup_registers_the_buttons_and_every_command(bot):
    async with bot:
        await bot.setup_hook()
        names = {command.name for command in bot.tree.get_commands()}
        templates = bot._connection._view_store._dynamic_items
        await bot.shutdown()

    assert names == {command.name for command in bot_module.COMMANDS}
    assert templates == {ActionButton.__discord_ui_compiled_template__: ActionButton}


async def test_a_click_on_a_card_from_before_a_restart_reaches_the_bot(bot, monkeypatch):
    clicks = []

    async def on_button(interaction, custom_id):
        clicks.append((interaction.user.id, custom_id))

    monkeypatch.setattr(bot, "on_button", on_button)
    async with bot:
        bot.add_dynamic_items(ActionButton)  # as setup_hook does; no view is stored for the card
        data = {"custom_id": "ma:undo:5", "component_type": 2}
        bot._connection.parse_interaction_create(_interaction_payload(3, data))
        await _until(lambda: clicks)

    assert clicks == [(STRANGER, "ma:undo:5")]


async def test_a_submitted_modal_reaches_the_action_with_its_text(bot, monkeypatch):
    acted, deferred = [], []

    async def act(interaction, custom_id, form=None):
        acted.append((custom_id, form))

    async def defer(self, **kwargs):
        deferred.append(kwargs)

    monkeypatch.setattr(bot, "act", act)
    monkeypatch.setattr(discord.InteractionResponse, "defer", defer)
    modal = ActionModal(views.modal_for("ma:reject:5"), "ma:reject:5")
    note = {"type": 4, "custom_id": modal.inputs["note"].custom_id, "value": "Too early"}
    data = {"custom_id": modal.custom_id, "components": [{"type": 18, "component": note}]}
    async with bot:
        bot._connection.store_view(modal)
        bot._connection.parse_interaction_create(_interaction_payload(5, data))
        await _until(lambda: acted)

    assert deferred == [{}]
    assert acted == [("ma:reject:5", {"note": "Too early"})]


# ---------------------------------------------------------------- posting notices


class FakeMessage:
    def __init__(self, view) -> None:
        self.id = 99
        self.view = view
        buttons = [] if view is None else [item.to_component_dict() for item in view.children]
        self.components = [ActionRow({"type": 1, "components": buttons})] if buttons else []
        self.edits: list[dict] = []

    async def edit(self, **kwargs) -> None:
        self.edits.append(kwargs)


class FakeChannel:
    def __init__(self, fail_titles: tuple[str, ...] = ()) -> None:
        self.sent: list[FakeMessage] = []
        self.embeds: list[discord.Embed] = []
        self.fail_titles = fail_titles

    async def send(self, *, embed: discord.Embed, view=None) -> FakeMessage:
        if embed.title in self.fail_titles:
            raise discord.HTTPException(SimpleNamespace(status=500, reason="boom"), "boom")
        message = FakeMessage(view)
        self.sent.append(message)
        self.embeds.append(embed)
        return message


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(bot_module, "READY_WAIT_S", 0.01)
    monkeypatch.setattr(bot_module, "MAX_SLEEP_S", 0.01)


async def _until(condition, timeout: float = 2.0) -> None:
    async def wait() -> None:
        while not condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout)


async def test_notices_are_posted_in_order_and_a_failure_goes_to_the_console(bot, fallback, fast):
    channel = FakeChannel(fail_titles=("n2",))
    bot.notifier.notify(_notice(1))  # before the bot is ready: buffered
    bot.notifier.notify(_notice(2))

    task = asyncio.create_task(bot._deliver(channel))
    await _until(lambda: bot._queue is not None)
    bot.notifier.notify(_notice(3))
    await _until(lambda: len(channel.embeds) == 2)
    task.cancel()

    assert [embed.title for embed in channel.embeds] == ["n1", "n3"]
    assert [n.title for n in fallback.notices] == ["n2"]


async def test_undo_switches_itself_off_at_the_deadline(bot, session, clock, item_id, fast):
    channel = FakeChannel()
    deadline = clock.now().replace(minute=10)
    notice = Notice(
        kind=NoticeKind.COMMITMENT, title="Deal", body="b", item_id=item_id, ref_id=77,
        expires_at=deadline,
    )

    await bot._post_notice(channel, notice)
    [message] = channel.sent
    await asyncio.sleep(0.05)
    assert message.edits == []
    clock.set(deadline)
    await _until(lambda: message.edits)

    [edit] = message.edits
    assert [(item.custom_id, item.item.disabled) for item in edit["view"].children] == [
        ("ma:undo:77", True)
    ]


async def test_a_card_waits_for_its_record_to_commit(bot, session, clock, item_id, fast):
    notice = Notice(kind=NoticeKind.APPROVAL, title="Approve update_price?", body="{}",
                    item_id=item_id, ref_id=1)

    rendering = asyncio.create_task(bot._card_for(notice))
    await asyncio.sleep(0.03)
    _park(session, clock, item_id)
    card = await asyncio.wait_for(rendering, 2)

    assert "Change the asking price from $120.00 to $105.00" in card.description


async def test_shutdown_prints_what_discord_did_not_take(bot, fallback, monkeypatch):
    monkeypatch.setattr(bot_module, "DRAIN_S", 0.01)
    bot._queue = asyncio.Queue()
    bot._queue.put_nowait(_notice(1))

    await bot.shutdown()
    bot.notifier.notify(_notice(2))

    assert [n.title for n in fallback.notices] == ["n1", "n2"]
    assert bot.is_closed()


# ---------------------------------------------------------------- logging in


@pytest.fixture
def with_token(runtime):
    from pydantic import SecretStr

    runtime.settings.discord_token = SecretStr("token")


async def test_login_is_retried_while_discord_is_unreachable(bot, with_token, monkeypatch):
    monkeypatch.setattr(bot_module, "LOGIN_RETRY_S", 0.01)
    attempts = []

    async def start(token):
        attempts.append(token)
        if len(attempts) < 3:
            raise OSError("network is unreachable")
        await bot.close()  # served, then shut down

    monkeypatch.setattr(bot, "start", start)

    await asyncio.wait_for(bot.run_until_closed(), 2)

    assert attempts == ["token"] * 3


async def test_a_rejected_token_is_a_setup_error(bot, with_token, monkeypatch):
    async def start(token):
        raise discord.LoginFailure("Improper token has been passed.")

    monkeypatch.setattr(bot, "start", start)

    with pytest.raises(bot_module.DiscordSetupError, match="rejected the bot token"):
        await bot.run_until_closed()
    assert bot.is_closed()


async def test_shutdown_ends_the_login_retries_at_once(bot, with_token, monkeypatch):
    monkeypatch.setattr(bot_module, "LOGIN_RETRY_S", 60.0)

    async def start(token):
        raise OSError("network is unreachable")

    monkeypatch.setattr(bot, "start", start)
    running = asyncio.create_task(bot.run_until_closed())
    await asyncio.sleep(0.05)

    await bot.shutdown()
    await asyncio.wait_for(running, 1)


# ---------------------------------------------------------------- startup


class GuildChannel(FakeChannel, discord.abc.Messageable):
    """A text channel, as fetch_channel returns it; guild is None for a DM."""

    def __init__(self, guild: discord.Object | None = None) -> None:
        super().__init__()
        self.guild = guild


def _http_error(kind: type[discord.HTTPException], status: int) -> discord.HTTPException:
    return kind(SimpleNamespace(status=status, reason="no"), "no")


@pytest.fixture
def discord_api(bot, runtime, monkeypatch):
    """Discord's side of startup, recorded: the channel lookup and the slash command sync.
    Set .channel to what fetch_channel returns, or to an exception it raises."""
    runtime.settings.discord_channel_id = CHANNEL_ID
    api = SimpleNamespace(
        channel=GuildChannel(discord.Object(id=GUILD_ID)), fetched=[], synced=[]
    )

    async def ready() -> None:
        return None

    async def fetch_channel(channel_id: int):
        api.fetched.append(channel_id)
        if isinstance(api.channel, Exception):
            raise api.channel
        return api.channel

    async def sync(*, guild=None) -> list:
        api.synced.append(guild)
        return []

    monkeypatch.setattr(bot, "wait_until_ready", ready)
    monkeypatch.setattr(bot, "fetch_channel", fetch_channel)
    monkeypatch.setattr(bot.tree, "sync", sync)
    return api


async def test_startup_finds_the_channel_syncs_commands_and_posts(bot, clock, discord_api, fast):
    channel = discord_api.channel
    bot.notifier.notify(_notice(1))  # before the bot is ready: buffered

    async with bot:
        await bot.setup_hook()
        await _until(lambda: channel.embeds)
        guild_commands = {c.name for c in bot.tree.get_commands(guild=channel.guild)}
        clock.set(views.next_digest_at(bot.runtime.settings, clock.now()))
        await _until(lambda: len(channel.embeds) == 2)
        await bot.shutdown()

    assert discord_api.fetched == [CHANNEL_ID]
    assert discord_api.synced == [channel.guild]  # server commands show up at once
    assert guild_commands == {command.name for command in bot_module.COMMANDS}
    assert [embed.title for embed in channel.embeds] == ["n1", "Digest for Thu Oct 1"]
    assert bot._failure is None


async def test_without_a_server_commands_sync_globally_and_a_failed_sync_is_not_fatal(
    bot, discord_api, monkeypatch, fast
):
    discord_api.channel = GuildChannel(guild=None)
    synced = []

    async def failing_sync(*, guild=None) -> list:
        synced.append(guild)
        raise _http_error(discord.HTTPException, 500)

    monkeypatch.setattr(bot.tree, "sync", failing_sync)
    bot.notifier.notify(_notice(1))

    bot._spawn(bot._start_posting())
    await _until(lambda: discord_api.channel.embeds)
    await bot.shutdown()

    assert synced == [None]
    assert [embed.title for embed in discord_api.channel.embeds] == ["n1"]
    assert bot._failure is None


@pytest.mark.parametrize(
    ("channel", "problem"),
    [
        (_http_error(discord.NotFound, 404), f"The bot cannot see channel {CHANNEL_ID}."),
        (_http_error(discord.Forbidden, 403), f"The bot cannot see channel {CHANNEL_ID}."),
        (discord.Object(id=CHANNEL_ID), f"Channel {CHANNEL_ID} is not a text channel."),
        (None, "Set MA_DISCORD_CHANNEL_ID to the channel the bot posts in."),
    ],
)
async def test_a_bad_channel_closes_the_bot_with_a_setup_error(
    bot, runtime, discord_api, fallback, channel, problem
):
    if channel is None:
        runtime.settings.discord_channel_id = None
    else:
        discord_api.channel = channel
    bot.notifier.notify(_notice(1))

    await asyncio.wait_for(bot._start_posting(), 1)

    assert isinstance(bot._failure, bot_module.DiscordSetupError)
    assert str(bot._failure).startswith(problem)
    assert bot.is_closed()
    assert discord_api.synced == [] and not bot._tasks  # no commands, no digest loop
    assert bot._queue is None and fallback.notices == []  # the notice waits in the buffer
    assert discord_api.fetched == ([] if channel is None else [CHANNEL_ID])


async def test_run_until_closed_raises_the_channel_problem(
    bot, with_token, discord_api, monkeypatch
):
    discord_api.channel = _http_error(discord.NotFound, 404)

    async def start(token: str) -> None:
        await bot._start_posting()  # what login, then setup_hook, leads to

    monkeypatch.setattr(bot, "start", start)

    with pytest.raises(bot_module.DiscordSetupError, match="check MA_DISCORD_CHANNEL_ID"):
        await asyncio.wait_for(bot.run_until_closed(), 2)
    assert bot.is_closed()


# ---------------------------------------------------------------- the daily digest


@pytest.mark.parametrize(
    ("hour", "due", "days"),
    [
        (20, datetime(2026, 10, 2, 0, tzinfo=UTC), ["Thu Oct 1", "Fri Oct 2"]),
        (8, datetime(2026, 10, 2, 12, tzinfo=UTC), ["Fri Oct 2", "Sat Oct 3"]),
    ],
)
async def test_the_digest_goes_out_once_a_day_at_the_local_digest_hour(
    bot, runtime, clock, fast, hour, due, days
):
    runtime.settings.discord_digest_hour = hour  # Toronto local; the clock starts at noon there
    channel = FakeChannel()
    bot._spawn(bot._post_digests(channel))

    clock.set(due - timedelta(minutes=1))
    await asyncio.sleep(0.05)
    assert channel.embeds == []
    clock.set(due)
    await _until(lambda: channel.embeds)
    clock.advance(hours=23, minutes=59)
    await asyncio.sleep(0.05)
    assert len(channel.embeds) == 1
    clock.advance(minutes=1)
    await _until(lambda: len(channel.embeds) == 2)
    await bot.shutdown()

    assert [embed.title for embed in channel.embeds] == [f"Digest for {day}" for day in days]
    for embed in channel.embeds:
        assert "Sold today" in embed.description and "Waiting for you" in embed.description


async def test_a_failed_digest_is_logged_and_the_next_day_still_goes_out(
    bot, clock, fast, caplog
):
    channel = FakeChannel(fail_titles=("Digest for Thu Oct 1",))
    bot._spawn(bot._post_digests(channel))
    await asyncio.sleep(0.02)  # the loop is waiting for 20:00 local, 00:00 UTC

    clock.set(datetime(2026, 10, 2, 0, tzinfo=UTC))
    await _until(lambda: "Could not post the daily digest" in caplog.text)
    clock.set(datetime(2026, 10, 3, 0, tzinfo=UTC))
    await _until(lambda: channel.embeds)
    await bot.shutdown()

    assert [embed.title for embed in channel.embeds] == ["Digest for Fri Oct 2"]
