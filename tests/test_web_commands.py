"""web.commands: `agent web`, registered on a fresh Typer app."""

import logging

import pytest
import typer
import uvicorn
from fastapi import FastAPI
from typer.testing import CliRunner

from marketplace_agent.web.commands import redact_token, register

KEY_VARS = (
    "MA_GEMINI_API_KEY",
    "GEMINI_API_KEY",
    "MA_GROQ_API_KEY",
    "GROQ_API_KEY",
    "MA_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "MA_ENABLED_PLATFORMS",
    "MA_WEB_TOKEN",
)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep a developer's .env and toml out of the run
    monkeypatch.setenv("MA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MA_WEB_PORT", "9123")
    monkeypatch.delenv("MA_DB_URL", raising=False)
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    app = typer.Typer()
    register(app)

    @app.command()
    def other() -> None:  # a second command, so `web` is a subcommand as in the real CLI
        pass

    return app


@pytest.fixture
def served(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append({"app": app, **kwargs}))
    return calls


def test_web_serves_the_app_on_the_configured_host_and_port(cli, served):
    result = CliRunner().invoke(cli, ["web"])

    assert result.exit_code == 0, result.output
    [call] = served
    assert isinstance(call["app"], FastAPI)
    assert (call["host"], call["port"]) == ("127.0.0.1", 9123)
    assert "http://127.0.0.1:9123/" in result.output and "agent run" in result.output


def test_web_with_facebook_points_to_serve_not_a_second_process(
    cli, served, monkeypatch, tmp_path
):
    """`agent web` next to `agent run` would fight over the one Facebook browser profile."""
    profile = tmp_path / "facebook_profile"
    profile.mkdir()
    monkeypatch.setenv("MA_FACEBOOK_PROFILE_DIR", str(profile))
    monkeypatch.setenv("MA_ENABLED_PLATFORMS", '["facebook"]')

    result = CliRunner().invoke(cli, ["web"])

    assert result.exit_code == 0, result.output
    assert "run `agent serve` instead" in result.output
    assert "Run `agent run` too" not in result.output


def test_web_explains_the_token_without_printing_it(cli, served, monkeypatch):
    monkeypatch.setenv("MA_WEB_TOKEN", "s3cret")

    result = CliRunner().invoke(cli, ["web"])

    assert result.exit_code == 0 and "?token=" in result.output
    assert "s3cret" not in result.output


def test_a_platform_that_cannot_be_built_stops_with_its_message(cli, served, monkeypatch):
    monkeypatch.setenv("MA_ENABLED_PLATFORMS", '["myspace"]')

    result = CliRunner().invoke(cli, ["web"])

    assert result.exit_code == 1 and "Unknown platform 'myspace'" in result.output
    assert served == []


def test_the_access_log_never_shows_the_token():
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", "/items/1?a=2&token=s3cret&b=3", "1.1", 303),
        None,
    )

    assert redact_token(record)
    assert "s3cret" not in record.getMessage()
    assert "/items/1?a=2&token=***&b=3" in record.getMessage()
