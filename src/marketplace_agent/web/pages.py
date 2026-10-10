"""Pages and intake: the item list, a new item from photos, drafting, confirming, photos."""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Form, Path, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.exc import OperationalError
from sqlmodel import Session
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException

from marketplace_agent.ai.base import Capability, ProviderError
from marketplace_agent.ai.drafting import draft_item
from marketplace_agent.config import Settings
from marketplace_agent.core.db import session_scope
from marketplace_agent.core.items import (
    ItemNotFound,
    create_item,
    get_item,
    set_attributes,
    set_goal,
    touch,
)
from marketplace_agent.core.models import Goal, ItemStatus, Photo
from marketplace_agent.core.runtime import Runtime
from marketplace_agent.guard.invariants import check_listing_text
from marketplace_agent.intake.photos import UnsupportedPhoto, ingest_photos, photo_file
from marketplace_agent.web.forms import ConfirmedDraft, DraftFields, check_draft
from marketplace_agent.web.ui import DATABASE_BUSY, DraftState, Flash, Ui, busy, get_ui

RecordId = Annotated[int, Path(ge=1, le=2**63 - 1)]

MB = 1024 * 1024
FORM_OVERHEAD = MB  # multipart boundaries, headers, and the hint, on top of the photos
EMPTY_PICKERS = 2  # a file input left empty still sends one empty part
MAX_FIELDS = 8
MAX_HINT_CHARS = 500

NO_VISION = (
    "No vision model is set up, so the listing was not drafted. Set GEMINI_API_KEY (free at "
    "aistudio.google.com) and reload, or fill it in yourself."
)

router = APIRouter()


class UploadRejected(Exception):
    def __init__(self, status: int, message: str, hint: str = ""):
        super().__init__(message)
        self.status = status
        self.hint = hint


class NotADraft(Exception):
    pass


@router.get("/")
def index(request: Request) -> Response:
    return get_ui(request).home(request)


@router.get("/items/new")
def new_item(request: Request) -> Response:
    return get_ui(request).render(request, "new_item.html", {"error": None, "hint": ""})


@router.post("/items")
async def create(request: Request) -> Response:
    ui = get_ui(request)
    try:
        hint, photos = await _read_upload(request, ui.settings)
    except UploadRejected as exc:
        return _upload_error(ui, request, exc.status, str(exc), exc.hint)
    try:
        item_id = await run_in_threadpool(_store_item, ui.runtime, hint, photos)
    except UnsupportedPhoto as exc:
        return _upload_error(ui, request, 422, f"{exc}. Nothing was saved.", hint)
    except OperationalError as exc:
        if not busy(exc):
            raise
        return _upload_error(ui, request, 409, f"{DATABASE_BUSY} Nothing was saved.", hint)
    return RedirectResponse(f"/items/{item_id}", status_code=303)


@router.get("/items/{item_id}")
def show(request: Request, item_id: RecordId) -> Response:
    ui = get_ui(request)
    try:
        return ui.item(request, item_id)
    except ItemNotFound:
        return ui.message(request, 404, f"There is no item with id {item_id}.")


@router.post("/items/{item_id}/draft")
def draft(request: Request, item_id: RecordId) -> Response:
    """Draft the listing from the photos. The answer is the form either way, so the owner can
    always fill it in by hand."""
    ui = get_ui(request)
    try:
        try:
            state = _draft(ui.runtime, item_id)
        except OperationalError as exc:
            if not busy(exc):
                raise
            state = DraftState(note=f"{DATABASE_BUSY} The draft was not saved.", retry=True)
        return ui.draft(request, item_id, state)
    except ItemNotFound:
        return ui.message(request, 404, f"There is no item with id {item_id}.")
    except NotADraft as exc:
        return ui.message(request, 409, str(exc))


@router.post("/items/{item_id}/confirm")
def confirm(
    request: Request, item_id: RecordId, fields: Annotated[DraftFields, Form()]
) -> Response:
    """Save the edited draft and its goal, then publish through ControlService."""
    ui = get_ui(request)
    checked = check_draft(fields, ui.settings.enabled_platforms)
    if isinstance(checked, dict):
        errors = checked
    else:
        errors = _text_errors(ui.settings, checked, ui.runtime.clock.now())
    try:
        if errors or isinstance(checked, dict):
            state = DraftState(fields=fields, errors=errors)
            return ui.item(request, item_id, draft=state, status=422)
        _save_draft(ui.runtime, item_id, checked)
    except ItemNotFound:
        return ui.message(request, 404, f"There is no item with id {item_id}.")
    except NotADraft as exc:
        return ui.item(request, item_id, flash=Flash(str(exc), ok=False), status=409)
    except OperationalError as exc:
        if not busy(exc):
            raise
        state = DraftState(fields=fields, note=f"{DATABASE_BUSY} Nothing was saved yet.")
        return ui.item(request, item_id, draft=state, status=409)
    platforms = checked.platforms
    return ui.act(request, item_id, "top", lambda c: c.confirm_and_publish(item_id, platforms))


@router.get("/photos/{photo_id}")
def photo(request: Request, photo_id: RecordId) -> Response:
    """Found by id in the database only; a stored path outside the photos dir is never served."""
    ui = get_ui(request)
    settings = ui.settings
    with Session(ui.runtime.engine) as session:
        found = session.get(Photo, photo_id)
    path = photo_file(settings, found).resolve() if found else None
    if path is None or not path.is_relative_to(settings.photos_dir.resolve()) or not path.is_file():
        return ui.message(request, 404, "That photo is not here.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private"})


# ---------------------------------------------------------------- helpers


async def _read_upload(request: Request, settings: Settings) -> tuple[str, list[bytes]]:
    """The hint and every chosen photo, within the limits. Raises UploadRejected."""
    max_bytes = settings.web_max_photo_mb * MB
    limits = f"Up to {settings.web_max_photos} photos, each under {settings.web_max_photo_mb} MB."
    declared = request.headers.get("content-length", "")
    if not declared.isdigit():
        raise UploadRejected(411, "The upload did not say how big it is. Try again.")
    if int(declared) > settings.web_max_photos * max_bytes + FORM_OVERHEAD:
        raise UploadRejected(413, f"That is too much to upload at once. {limits}")
    max_files = settings.web_max_photos + EMPTY_PICKERS
    try:
        async with request.form(max_files=max_files, max_fields=MAX_FIELDS) as form:
            return await _photos_in(form, settings, max_bytes, limits)
    except HTTPException as exc:  # the multipart parser hit a limit
        raise UploadRejected(413, f"That upload was refused. {limits}") from exc


async def _photos_in(
    form: FormData, settings: Settings, max_bytes: int, limits: str
) -> tuple[str, list[bytes]]:
    hint = form.get("hint")
    hint = " ".join(hint.split()) if isinstance(hint, str) else ""
    if len(hint) > MAX_HINT_CHARS:
        raise UploadRejected(422, f"Keep the note under {MAX_HINT_CHARS} characters.", hint)
    files = [f for f in form.getlist("photos") if isinstance(f, UploadFile) and f.filename]
    if not files:
        raise UploadRejected(422, "Add at least one photo.", hint)
    if len(files) > settings.web_max_photos:
        raise UploadRejected(413, f"You picked {len(files)} photos. {limits}", hint)
    for upload in files:
        if (upload.size or 0) > max_bytes:
            raise UploadRejected(413, f"{upload.filename} is too big. {limits}", hint)
    return hint, [await upload.read() for upload in files]


def _store_item(runtime: Runtime, hint: str, photos: list[bytes]) -> int:
    """One transaction: a photo that cannot be read leaves no item behind."""
    with session_scope(runtime.engine) as session:
        item = create_item(session, runtime.clock, hint)
        assert item.id is not None
        ingest_photos(session, runtime.settings, item.id, photos)
        return item.id


def _upload_error(ui: Ui, request: Request, status: int, message: str, hint: str) -> Response:
    context = {"error": f"{message} Pick the photos again.", "hint": hint}
    return ui.render(request, "new_item.html", context, status)


def _draft(runtime: Runtime, item_id: int) -> DraftState:
    with Session(runtime.engine) as session:
        _require_draft(get_item(session, item_id).status, item_id)
    if not runtime.router.has_capability(Capability.VISION):
        return DraftState(note=NO_VISION)
    try:
        with session_scope(runtime.engine) as session:
            listing = draft_item(
                session,
                runtime.router,
                runtime.settings,
                runtime.clock,
                item_id,
                platforms=runtime.platforms,
            )
    except (ValueError, ProviderError) as exc:
        note = f"The draft did not work: {exc}. Try again, or fill it in yourself."
        return DraftState(note=note, retry=True)
    return DraftState(suggestion=listing)


def _text_errors(settings: Settings, draft: ConfirmedDraft, now: datetime) -> dict[str, str]:
    """The listing text rules publishing enforces, checked while the form can still be fixed:
    once confirmed, the item is for sale and this form is gone."""
    goal = Goal(
        item_id=0,
        list_price_cents=draft.list_price_cents,
        target_price_cents=draft.target_price_cents or draft.list_price_cents,
        floor_price_cents=draft.floor_price_cents,
        deadline=now + timedelta(days=draft.deadline_days),
        firmness=draft.firmness,
    )
    errors = {}
    for name, text in (("title", draft.title), ("description", draft.description)):
        reasons = check_listing_text(settings, goal=goal, text=text)
        if reasons:
            errors[name] = " ".join(reasons)
    return errors


def _save_draft(runtime: Runtime, item_id: int, draft: ConfirmedDraft) -> None:
    """Edits to a draft, before it is for sale. No agent acts on a draft, so unlike the
    ControlService overrides there is nothing to wake or to learn from."""
    with session_scope(runtime.engine) as session:
        item = get_item(session, item_id)
        _require_draft(item.status, item_id)
        item.title, item.description = draft.title, draft.description
        item.category, item.condition = draft.category, draft.condition
        set_attributes(item, draft.attributes)
        touch(item, runtime.clock)
        session.add(item)
        set_goal(
            session,
            runtime.clock,
            item_id,
            list_price_cents=draft.list_price_cents,
            floor_price_cents=draft.floor_price_cents,
            target_price_cents=draft.target_price_cents,
            deadline_days=draft.deadline_days,
            firmness=draft.firmness,
            notes=draft.notes,
        )


def _require_draft(status: ItemStatus, item_id: int) -> None:
    if status != ItemStatus.DRAFT:
        raise NotADraft(f"Item {item_id} is already {status.value}; only a draft is edited here.")
