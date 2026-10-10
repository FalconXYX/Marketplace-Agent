"""The Discord bot: notices as cards with buttons, slash commands, and the daily digest.

What it shows and does lives in notify/discord_views.py; this module only talks to Discord.

| Part | How |
|---|---|
| Notices | DiscordNotifier hands them from any thread to the bot loop; one task posts in order |
| Buttons | ActionButton, a DynamicItem on "ma:<action>:<id>": cards work after a restart |
| A click | Owner check, then a modal (reject, reply, resolve) or a defer; work runs in a thread |
| Undo | A COMMITMENT card's Undo switches itself off at the undo deadline |
| Commands | /items /item /price /floor /firm /pause /resume /sold /undo /approve /status /digest |
| Digest | Posted daily at discord_digest_hour, seller-local time |

Discord drops an interaction not answered within 3 seconds, so nothing touches the database before
the defer (or the modal, which is an answer too).
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Iterable
from datetime import datetime
from re import Match
from typing import Any, Literal

import aiohttp
import discord
from discord import app_commands
from sqlmodel import Session

from marketplace_agent.core.items import ItemNotFound
from marketplace_agent.core.runtime import Runtime
from marketplace_agent.core.serve import ServeError
from marketplace_agent.notify import discord_views as views
from marketplace_agent.notify.base import ConsoleNotifier, Notice, NoticeKind, Notifier

READY_TRIES = 10  # the record a notice points at commits a moment after the notice is sent
READY_WAIT_S = 0.5
MAX_SLEEP_S = 60.0  # wake up this often to check the wall clock, which keeps running on sleep
DRAIN_S = 10.0
LOGIN_RETRY_S = 5.0
MAX_LOGIN_RETRY_S = 300.0
MODAL_TIMEOUT_S = 900.0
MODAL_TITLE_MAX = 45
CONTENT_MAX = 2000
ACTION_FAILED = "That did not work because of a bug; the agent's log has the details."

Interaction = discord.Interaction[Any]

_STYLES: dict[views.ButtonStyle, discord.ButtonStyle] = {
    "primary": discord.ButtonStyle.primary,
    "secondary": discord.ButtonStyle.secondary,
    "success": discord.ButtonStyle.success,
    "danger": discord.ButtonStyle.danger,
}

logger = logging.getLogger("marketplace_agent.discord")


class DiscordSetupError(ServeError):
    """The bot cannot start: a bad token, or a channel it cannot post in."""


class DiscordNotifier:
    """Notifier that the runner, web, and control threads share with the bot. Notices wait in a
    buffer until the bot can post, then go onto its loop in the order they came. After close
    they go to the fallback, so none is dropped silently."""

    def __init__(self, fallback: Notifier | None = None) -> None:
        self.fallback = fallback or ConsoleNotifier()
        self._lock = threading.Lock()
        self._buffer: list[Notice] = []
        self._target: tuple[asyncio.AbstractEventLoop, asyncio.Queue[Notice]] | None = None
        self._closed = False

    def notify(self, notice: Notice) -> None:
        with self._lock:
            if self._closed:
                self.fallback.notify(notice)
            elif self._target is None:
                self._buffer.append(notice)
            else:
                self._hand_over(notice, *self._target)

    def attach(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[Notice]) -> None:
        """Called on the bot loop once it can post: buffered notices go first."""
        with self._lock:
            for notice in self._buffer:
                queue.put_nowait(notice)
            self._buffer.clear()
            self._target = (loop, queue)

    def close(self, unsent: Iterable[Notice] = ()) -> None:
        """Stop handing notices to the bot; unsent and buffered ones go to the fallback."""
        with self._lock:
            self._closed = True
            self._target = None
            for notice in [*unsent, *self._buffer]:
                self.fallback.notify(notice)
            self._buffer.clear()

    def _hand_over(
        self, notice: Notice, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[Notice]
    ) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, notice)
        except RuntimeError:  # the loop closed under us
            self.fallback.notify(notice)


class ActionButton(
    discord.ui.DynamicItem[discord.ui.Button[discord.ui.View]],
    template=views.CUSTOM_ID_TEMPLATE,
):
    """Any agent button, found again from its custom_id after a restart."""

    @classmethod
    async def from_custom_id(
        cls,
        interaction: Interaction,
        item: discord.ui.Item[Any],
        match: Match[str],
        /,
    ) -> "ActionButton":
        if not isinstance(item, discord.ui.Button):
            raise TypeError(f"{match.string} is not a button")
        return cls(item)

    async def callback(self, interaction: Interaction) -> None:
        await _bot(interaction).on_button(interaction, self.custom_id)


class ActionModal(discord.ui.Modal):
    """The text a button's action needs (a rejection note, a reply, an answer)."""

    def __init__(self, spec: views.ModalSpec, custom_id: str) -> None:
        super().__init__(title=_clip(spec.title, MODAL_TITLE_MAX), timeout=MODAL_TIMEOUT_S)
        self.action_id = custom_id
        self.inputs: dict[str, discord.ui.TextInput[ActionModal]] = {}
        for field in spec.fields:
            text_input: discord.ui.TextInput[ActionModal] = discord.ui.TextInput(
                style=discord.TextStyle.paragraph,
                required=field.required,
                placeholder=field.placeholder or None,
                max_length=field.max_length,
            )
            self.add_item(discord.ui.Label(text=field.label, component=text_input))
            self.inputs[field.key] = text_input

    async def on_submit(self, interaction: Interaction) -> None:
        await interaction.response.defer()
        form = {key: text_input.value for key, text_input in self.inputs.items()}
        await _bot(interaction).act(interaction, self.action_id, form)


class DiscordBot(discord.Client):
    def __init__(self, runtime: Runtime, notifier: DiscordNotifier) -> None:
        super().__init__(intents=discord.Intents.default())
        self.runtime = runtime
        self.notifier = notifier
        self.tree = app_commands.CommandTree(self)
        self._queue: asyncio.Queue[Notice] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._failure: Exception | None = None
        self._stopping = asyncio.Event()

    @property
    def owner_id(self) -> int | None:
        return self.runtime.settings.discord_owner_id

    async def run_until_closed(self) -> None:
        """Log in and serve until shutdown. While Discord cannot be reached, retry with backoff
        (notices wait in the buffer); a bad token or channel is raised once the bot has closed."""
        token = self.runtime.settings.discord_token
        if token is None:
            raise DiscordSetupError("Set MA_DISCORD_TOKEN to run the Discord bot.")
        delay = LOGIN_RETRY_S
        try:
            while not self.is_closed() and not self._stopping.is_set():
                try:
                    await self.start(token.get_secret_value())
                except discord.LoginFailure as exc:
                    raise DiscordSetupError(
                        "Discord rejected the bot token (MA_DISCORD_TOKEN); copy it again from "
                        "the developer portal."
                    ) from exc
                except (OSError, aiohttp.ClientError, discord.HTTPException) as exc:
                    logger.warning("Cannot reach Discord (%s); trying again in %.0fs", exc, delay)
                    await self.http.close()  # login opens a new HTTP session each try
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._stopping.wait(), delay)
                    delay = min(delay * 2, MAX_LOGIN_RETRY_S)
        finally:
            if not self.is_closed():
                await self.close()
        if self._failure is not None:
            raise self._failure

    async def shutdown(self) -> None:
        """Post what is still queued (for up to DRAIN_S), then disconnect."""
        self._stopping.set()
        unsent: list[Notice] = []
        if self._queue is not None:
            try:
                await asyncio.wait_for(self._queue.join(), DRAIN_S)
            except TimeoutError:
                logger.warning("Discord did not take every notice in time; printing the rest.")
            while not self._queue.empty():
                unsent.append(self._queue.get_nowait())
        self.notifier.close(unsent)
        for task in self._tasks:
            task.cancel()
        await self.close()

    async def setup_hook(self) -> None:
        self.add_dynamic_items(ActionButton)
        for command in COMMANDS:
            self.tree.add_command(command)
        self._spawn(self._start_posting())

    # ------------------------------------------------------------ posting

    async def _start_posting(self) -> None:
        await self.wait_until_ready()
        try:
            channel = await self._channel()
        except DiscordSetupError as exc:
            self._failure = exc
            await self.close()
            return
        await self._sync_commands(channel)
        self._spawn(self._post_digests(channel))
        logger.info("Discord bot ready; posting to channel %s", channel)
        await self._deliver(channel)

    async def _deliver(self, channel: discord.abc.Messageable) -> None:
        """Post notices in order, for good. Buffered notices go first."""
        queue: asyncio.Queue[Notice] = asyncio.Queue()
        self._queue = queue
        self.notifier.attach(asyncio.get_running_loop(), queue)
        while True:
            notice = await queue.get()
            try:
                await self._post_notice(channel, notice)
            except Exception:
                logger.exception("Could not post a %s notice to Discord", notice.kind.value)
                self.notifier.fallback.notify(notice)
            finally:
                queue.task_done()

    async def _channel(self) -> discord.abc.Messageable:
        channel_id = self.runtime.settings.discord_channel_id
        if channel_id is None:
            raise DiscordSetupError("Set MA_DISCORD_CHANNEL_ID to the channel the bot posts in.")
        try:
            channel = await self.fetch_channel(channel_id)
        except (discord.NotFound, discord.Forbidden) as exc:
            raise DiscordSetupError(
                f"The bot cannot see channel {channel_id}. Invite it to the server and check "
                "MA_DISCORD_CHANNEL_ID."
            ) from exc
        if not isinstance(channel, discord.abc.Messageable):
            raise DiscordSetupError(f"Channel {channel_id} is not a text channel.")
        return channel

    async def _sync_commands(self, channel: discord.abc.Messageable) -> None:
        """Guild commands show up at once; global ones can take an hour."""
        guild = getattr(channel, "guild", None)
        try:
            if guild is None:
                await self.tree.sync()
                return
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        except discord.HTTPException:
            logger.exception("Could not register the slash commands; buttons still work.")

    async def _post_notice(self, channel: discord.abc.Messageable, notice: Notice) -> None:
        card = await self._card_for(notice)
        message = await channel.send(embed=_embed(card), view=_view(card))
        undo = any(b.custom_id.startswith("ma:undo:") and not b.disabled for b in card.buttons)
        if notice.kind == NoticeKind.COMMITMENT and notice.expires_at is not None and undo:
            self._spawn(self._expire_undo(message, notice.expires_at))

    async def _card_for(self, notice: Notice) -> views.Card:
        for _ in range(READY_TRIES):
            if await asyncio.to_thread(self._read, lambda s: views.notice_ready(s, notice)):
                break
            await asyncio.sleep(READY_WAIT_S)
        runtime = self.runtime
        return await asyncio.to_thread(
            self._read, lambda s: views.render_notice(s, runtime.clock, runtime.settings, notice)
        )

    async def _expire_undo(self, message: discord.Message, deadline: datetime) -> None:
        await self._sleep_until(deadline)
        try:
            await message.edit(view=disabled_view(message))
        except discord.HTTPException:
            logger.warning("Could not switch off the Undo button on message %s", message.id)

    async def _post_digests(self, channel: discord.abc.Messageable) -> None:
        settings = self.runtime.settings
        while not self.is_closed():
            await self._sleep_until(views.next_digest_at(settings, self.runtime.clock.now()))
            try:
                await channel.send(embed=_text_embed(await self._digest()))
            except discord.HTTPException:
                logger.exception("Could not post the daily digest")

    async def _digest(self) -> str:
        runtime = self.runtime
        return await asyncio.to_thread(
            self._read, lambda s: views.build_digest(s, runtime.clock, runtime.settings)
        )

    async def _sleep_until(self, when: datetime) -> None:
        """Checks the wall clock every MAX_SLEEP_S: asyncio's clock stops while the Mac sleeps."""
        while (left := (when - self.runtime.clock.now()).total_seconds()) > 0:
            await asyncio.sleep(min(left, MAX_SLEEP_S))

    # ------------------------------------------------------------ interactions

    async def on_button(self, interaction: Interaction, custom_id: str) -> None:
        if await self._refused(interaction):
            return
        spec = views.modal_for(custom_id)
        if spec is not None:
            await interaction.response.send_modal(ActionModal(spec, custom_id))
            return
        await interaction.response.defer()
        await self.act(interaction, custom_id)

    async def act(
        self,
        interaction: Interaction,
        custom_id: str,
        form: dict[str, str] | None = None,
    ) -> None:
        """After the defer: run the action in a worker thread, switch the card's buttons off once
        the decision is made, and say what happened."""
        user = interaction.user.id
        control = self.runtime.control
        try:
            response = await asyncio.to_thread(
                views.handle_action, control, custom_id, user, self.owner_id, form
            )
        except Exception:
            logger.exception("Discord action %s failed", custom_id)
            response = views.ActionResponse(ACTION_FAILED)
        if response.disable_buttons and interaction.message is not None:
            try:
                await interaction.edit_original_response(view=disabled_view(interaction.message))
            except discord.HTTPException:
                logger.warning("Could not switch off the buttons for %s", custom_id)
        await interaction.followup.send(
            _clip(response.text, CONTENT_MAX), ephemeral=response.ephemeral
        )

    async def command(
        self,
        interaction: Interaction,
        action: str,
        ref: int,
        form: dict[str, str] | None = None,
    ) -> None:
        """A slash command that changes something: the same path as the button for it."""
        if await self._refused(interaction):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.act(interaction, views.custom_id(action, ref), form)

    async def show(
        self,
        interaction: Interaction,
        work: Callable[[Session], views.Card | str],
        title: str = "",
    ) -> None:
        """A slash command that only reads: a card, or text under title."""
        if await self._refused(interaction):
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            result = await asyncio.to_thread(self._read, work)
        except ItemNotFound as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        if isinstance(result, str):
            await interaction.followup.send(embed=_text_embed(result, title), ephemeral=True)
            return
        view = _view(result)
        extra = {"view": view} if view is not None else {}
        await interaction.followup.send(embed=_embed(result), ephemeral=True, **extra)

    async def _refused(self, interaction: Interaction) -> bool:
        refused = views.check_owner(interaction.user.id, self.owner_id)
        if refused is not None:
            await interaction.response.send_message(refused.text, ephemeral=True)
        return refused is not None

    # ------------------------------------------------------------ helpers

    def _read[T](self, work: Callable[[Session], T]) -> T:
        with Session(self.runtime.engine) as session:
            return work(session)

    def _spawn(self, coroutine: Any) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def disabled_view(message: discord.Message) -> discord.ui.View:
    """The message's agent buttons, all switched off."""
    view = discord.ui.View(timeout=None)
    for row in message.components:
        for component in getattr(row, "children", ()):
            custom_id = getattr(component, "custom_id", None)
            if not isinstance(component, discord.Button) or not custom_id:
                continue
            try:
                views.parse_custom_id(custom_id)
            except ValueError:
                continue
            button: discord.ui.Button[discord.ui.View] = discord.ui.Button(
                label=component.label, style=component.style, custom_id=custom_id, disabled=True
            )
            view.add_item(ActionButton(button))
    return view


def _embed(card: views.Card) -> discord.Embed:
    embed = discord.Embed(title=card.title, description=card.description or None, color=card.color)
    for field in card.fields:
        embed.add_field(name=field.name, value=field.value, inline=field.inline)
    return embed


def _text_embed(text: str, title: str = "") -> discord.Embed:
    """Text whose first paragraph is its title, unless a title is given."""
    if not title:
        title, _, text = text.partition("\n\n")
    color = views.COLORS[NoticeKind.DIGEST]
    return discord.Embed(
        title=_clip(title, views.TITLE_MAX),
        description=_clip(text, views.DESCRIPTION_MAX) or None,
        color=color,
    )


def _view(card: views.Card) -> discord.ui.View | None:
    if not card.buttons:
        return None
    view = discord.ui.View(timeout=None)
    for spec in card.buttons:
        button: discord.ui.Button[discord.ui.View] = discord.ui.Button(
            label=spec.label,
            style=_STYLES[spec.style],
            custom_id=spec.custom_id,
            disabled=spec.disabled,
        )
        view.add_item(ActionButton(button))
    return view


def _bot(interaction: Interaction) -> DiscordBot:
    client = interaction.client
    if not isinstance(client, DiscordBot):
        raise TypeError("This interaction did not come through the agent's bot.")
    return client


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------- slash commands

ItemNumber = app_commands.Range[int, 1]
RecordNumber = app_commands.Range[int, 1]
Level = Literal["firm", "slight", "flexible"]


@app_commands.command(name="items", description="Your items and what each is waiting on")
async def items_command(interaction: Interaction) -> None:
    bot = _bot(interaction)
    settings = bot.runtime.settings
    await bot.show(interaction, lambda s: views.items_overview(s, settings), "Items")


@app_commands.command(name="item", description="One item at a glance")
@app_commands.describe(item="Item number, from /items")
async def item_command(interaction: Interaction, item: ItemNumber) -> None:
    bot = _bot(interaction)
    clock, settings = bot.runtime.clock, bot.runtime.settings
    await bot.show(interaction, lambda s: views.item_card(s, clock, settings, item))


@app_commands.command(name="price", description="Change the asking price everywhere")
@app_commands.describe(item="Item number", dollars="New asking price, e.g. 120 or 119.99")
async def price_command(interaction: Interaction, item: ItemNumber, dollars: str) -> None:
    await _bot(interaction).command(interaction, "price", item, {"amount": dollars})


@app_commands.command(name="floor", description="Change the lowest price the agent may agree to")
@app_commands.describe(item="Item number", dollars="New floor, e.g. 90")
async def floor_command(interaction: Interaction, item: ItemNumber, dollars: str) -> None:
    await _bot(interaction).command(interaction, "floor", item, {"amount": dollars})


@app_commands.command(name="firm", description="Change how far the agent may negotiate")
@app_commands.describe(item="Item number", level="firm, slight, or flexible")
async def firm_command(interaction: Interaction, item: ItemNumber, level: Level) -> None:
    await _bot(interaction).command(interaction, "firm", item, {"level": level})


@app_commands.command(name="pause", description="Stop the agent acting on an item")
@app_commands.describe(item="Item number")
async def pause_command(interaction: Interaction, item: ItemNumber) -> None:
    await _bot(interaction).command(interaction, "pause", item)


@app_commands.command(name="resume", description="Let the agent act on a paused item again")
@app_commands.describe(item="Item number")
async def resume_command(interaction: Interaction, item: ItemNumber) -> None:
    await _bot(interaction).command(interaction, "resume", item)


@app_commands.command(name="sold", description="Mark an item sold and take its listings down")
@app_commands.describe(item="Item number", dollars="What it sold for (optional)")
async def sold_command(
    interaction: Interaction, item: ItemNumber, dollars: str | None = None
) -> None:
    form = {"price": dollars} if dollars else None
    await _bot(interaction).command(interaction, "sold", item, form)


@app_commands.command(name="undo", description="Undo a deal the agent made, within its window")
@app_commands.describe(deal="Deal number, from its card")
async def undo_command(interaction: Interaction, deal: RecordNumber) -> None:
    await _bot(interaction).command(interaction, "undo", deal)


@app_commands.command(name="approve", description="Approve an action the agent asked about")
@app_commands.describe(approval="Approval number, from its card")
async def approve_command(interaction: Interaction, approval: RecordNumber) -> None:
    await _bot(interaction).command(interaction, "approve", approval)


@app_commands.command(name="status", description="Model quota and how much the agent may do alone")
async def status_command(interaction: Interaction) -> None:
    bot = _bot(interaction)
    settings, router = bot.runtime.settings, bot.runtime.router
    await bot.show(interaction, lambda s: views.status_text(s, settings, router), "Status")


@app_commands.command(name="digest", description="Today so far: sales, talks, actions, waiting")
async def digest_command(interaction: Interaction) -> None:
    bot = _bot(interaction)
    clock, settings = bot.runtime.clock, bot.runtime.settings
    await bot.show(interaction, lambda s: views.build_digest(s, clock, settings))


COMMANDS = (
    items_command,
    item_command,
    price_command,
    floor_command,
    firm_command,
    pause_command,
    resume_command,
    sold_command,
    undo_command,
    approve_command,
    status_command,
    digest_command,
)
