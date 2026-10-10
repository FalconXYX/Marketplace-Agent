"""The web UI: photos in from a phone, a listing out, and every owner control.

| Route | Does |
|---|---|
| GET / | Items by status with prices, platforms, and what needs the owner |
| GET /items/new, POST /items | Photos and a hint in, a draft item out |
| GET /items/{id} | A draft's form, or a listed item's deal, conversations, runs, controls |
| POST /items/{id}/draft, /confirm | Draft from the photos; save and publish |
| POST controls | See web/controls.py: one ControlService call each |
| GET /photos/{id} | A stored photo, looked up by id |

Bind to settings.web_host (loopback by default); reach it from a phone with `tailscale serve`.
"""

from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from marketplace_agent.core.runtime import Runtime
from marketplace_agent.web import controls, pages
from marketplace_agent.web.auth import install_guard
from marketplace_agent.web.ui import Ui

STATIC_DIR = Path(__file__).parent / "static"
INVALID_REQUEST = "That request is not valid. Go back and try again."


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="Marketplace Agent", docs_url=None, redoc_url=None, openapi_url=None)
    ui = Ui(runtime)
    app.state.ui = ui

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Response:
        text = "There is nothing here." if exc.status_code == 404 else str(exc.detail)
        return ui.message(request, exc.status_code, text)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> Response:
        return ui.message(request, 422, INVALID_REQUEST)

    install_guard(app, runtime.settings, ui.message)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(pages.router)
    app.include_router(controls.router)
    return app
