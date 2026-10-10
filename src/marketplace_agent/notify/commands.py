"""CLI commands for Stage 5: `agent serve`. cli.py wires them in with register(app)."""

import logging

import typer

from marketplace_agent.config import Settings


def register(app: typer.Typer) -> None:
    app.command("serve")(serve_command)


def serve_command() -> None:
    """Run everything in one process: the agent, the web intake, and the Discord bot."""
    from marketplace_agent.core.runtime import PlatformSetupError
    from marketplace_agent.core.serve import ServeError, serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        serve(Settings())
    except (ServeError, PlatformSetupError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from None
    typer.echo("Stopped.")
