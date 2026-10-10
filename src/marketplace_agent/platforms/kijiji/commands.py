"""`agent kijiji probe`: a live, read-only check of every Kijiji endpoint the adapter reads.

Nothing is posted, sent, edited, deleted, or marked read. The checks live in probe.py.
"""
from collections.abc import Callable

import typer

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import SystemClock
from marketplace_agent.platforms.base import PlatformAuthError
from marketplace_agent.platforms.kijiji import build_client
from marketplace_agent.platforms.kijiji.client import KijijiClient
from marketplace_agent.platforms.kijiji.probe import run_probe

ClientFactory = Callable[[Settings], KijijiClient]


def register(app: typer.Typer, *, client_factory: ClientFactory | None = None) -> None:
    kijiji = typer.Typer(no_args_is_help=True, help="Kijiji account tools.")

    @kijiji.command()
    def probe() -> None:
        """Sign in and read your ads, conversations, search, and the post form. Writes nothing."""
        settings = Settings()
        try:
            client = (client_factory or _system_client)(settings)
        except PlatformAuthError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from None
        checks = run_probe(client, settings.kijiji_location_id)
        for check in checks:
            typer.echo(check.line())
        failed = sum(check.ok is False for check in checks)
        passed = sum(check.ok is True for check in checks)
        skipped = len(checks) - failed - passed
        typer.echo(f"\n{passed} ok, {failed} failed, {skipped} skipped. Nothing was written.")
        raise typer.Exit(1 if failed else 0)

    app.add_typer(kijiji, name="kijiji")


def _system_client(settings: Settings) -> KijijiClient:
    return build_client(settings, SystemClock())
