"""CLI: `agent facebook login` and `agent doctor`. The integrator wires register() into cli.py."""

from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from marketplace_agent.config import Settings
from marketplace_agent.doctor import format_checks, run_doctor
from marketplace_agent.platforms.base import PlatformError
from marketplace_agent.platforms.facebook import flow_notes as fb
from marketplace_agent.platforms.facebook.adapter import signed_in
from marketplace_agent.platforms.facebook.driver import BrowserDriver, PlaywrightDriver
from marketplace_agent.platforms.facebook.pacing import Pacer

LOGIN_POLL_S = 3.0


def register(app: typer.Typer) -> None:
    facebook = typer.Typer(no_args_is_help=True, help="Facebook Marketplace browser session.")
    facebook.command("login")(login)
    app.add_typer(facebook, name="facebook")
    app.command("doctor")(doctor)


def login(
    minutes: Annotated[
        int, typer.Option(min=1, max=120, help="How long to wait for you to sign in.")
    ] = 10,
) -> None:
    """Open the agent's browser profile so you can sign in to Facebook yourself, once."""
    settings = _settings()
    driver = open_login_browser(settings)
    try:
        typer.echo(
            f"Sign in to Facebook in the browser window. Waiting up to {minutes} minutes. "
            "Your password is never stored; the session stays in "
            f"{settings.fb_profile_dir}."
        )
        if not wait_for_login(driver, timeout_s=minutes * 60):
            _fail("Not signed in yet. Run `agent facebook login` again when you are ready.")
        driver.goto(fb.INBOX_URL)
        typer.prompt(
            "Signed in. Messenger is open: if it asks for your chat PIN, enter it there, since "
            "the agent reads Marketplace chats through Messenger. Press Enter here when done",
            default="",
            show_default=False,
        )
    except PlatformError as exc:
        _fail(str(exc))
    finally:
        driver.close()
    typer.echo("Facebook session saved. Run `agent doctor` to check it.")


def doctor() -> None:
    """Check config, data dir, database, models, platforms, and Discord. Changes nothing."""
    checks = run_doctor(_settings())
    typer.echo(format_checks(checks))
    if not all(check.ok for check in checks):
        raise typer.Exit(1)


def open_login_browser(settings: Settings) -> BrowserDriver:
    """Always visible, whatever facebook_headless says: a person signs in."""
    return PlaywrightDriver(
        settings.fb_profile_dir, headless=False, pacer=Pacer(settings.facebook_action_delay_s)
    )


def wait_for_login(driver: BrowserDriver, timeout_s: float, poll_s: float = LOGIN_POLL_S) -> bool:
    driver.goto(fb.HOME_URL)
    for _ in range(max(1, round(timeout_s / poll_s))):
        if signed_in(driver):
            return True
        driver.wait(poll_s)
    return signed_in(driver)


def _settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        _fail(f"Invalid settings: {problems}")


def _fail(message: str) -> NoReturn:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)
