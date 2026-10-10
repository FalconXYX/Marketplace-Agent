"""`agent web`: serve the web UI."""

import logging
import re

import typer

TOKEN_IN_URL = re.compile(r"([?&]token=)[^&\s]*")


def register(app: typer.Typer) -> None:
    app.command("web")(web)


def web() -> None:
    """Serve the phone-friendly web UI: photos in, listing out, every control."""
    import uvicorn

    from marketplace_agent.config import Settings
    from marketplace_agent.core.runtime import PlatformSetupError, build_runtime
    from marketplace_agent.web.app import create_app

    try:
        runtime = build_runtime(Settings())
    except PlatformSetupError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from None
    settings = runtime.settings
    typer.echo(f"Web UI on http://{settings.web_host}:{settings.web_port}/")
    if settings.web_token:
        typer.echo("Open it once with ?token=<MA_WEB_TOKEN> at the end; the browser remembers it.")
    if "facebook" in settings.enabled_platforms:
        typer.secho(
            "This serves the UI only. With Facebook on, stop this and run `agent serve` "
            "instead: the agent and the UI both need the Facebook browser profile, and only "
            "one process can use it.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo("This serves the UI only. Run `agent run` too, so buyers get answers.")
    logging.getLogger("uvicorn.access").addFilter(redact_token)
    try:
        uvicorn.run(create_app(runtime), host=settings.web_host, port=settings.web_port)
    finally:
        runtime.engine.dispose()


def redact_token(record: logging.LogRecord) -> bool:
    """Keep the access token out of the access log, which prints each request's query."""
    if isinstance(record.args, tuple):
        record.args = tuple(
            TOKEN_IN_URL.sub(r"\1***", arg) if isinstance(arg, str) else arg for arg in record.args
        )
    return True
