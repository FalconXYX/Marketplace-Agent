"""`agent facebook login` and `agent doctor`, registered on a fresh Typer app."""

import pytest
import typer
from fixtures.facebook.fake_driver import FakeDriver, FakeSite
from typer.testing import CliRunner

from marketplace_agent.doctor import Check
from marketplace_agent.platforms.facebook import commands
from marketplace_agent.platforms.facebook import flow_notes as fb
from marketplace_agent.platforms.facebook.driver import DriverError

runner = CliRunner()


@pytest.fixture
def app(tmp_path, monkeypatch) -> typer.Typer:
    monkeypatch.chdir(tmp_path)  # keep a developer's .env and toml out of the run
    monkeypatch.setenv("MA_DATA_DIR", str(tmp_path / "data"))
    app = typer.Typer()
    commands.register(app)
    return app


@pytest.fixture
def driver(monkeypatch) -> FakeDriver:
    driver = FakeDriver(FakeSite(logged_in=False))
    monkeypatch.setattr(commands, "open_login_browser", lambda settings: driver)
    return driver


def test_login_waits_for_the_human_then_opens_messenger(app, driver):
    def sign_in_on_second_poll() -> None:
        if driver.kinds().count("wait") == 2:
            driver.site.logged_in = True
            driver.url = fb.HOME_URL

    driver.on_wait = sign_in_on_second_poll
    result = runner.invoke(app, ["facebook", "login"], input="\n")

    assert result.exit_code == 0, result.output
    assert "chat PIN" in result.output and "session saved" in result.output
    assert driver.actions[0] == ("goto", fb.HOME_URL)
    assert ("goto", fb.INBOX_URL) in driver.actions
    assert driver.closed


def test_login_gives_up_after_the_wait(app, driver):
    result = runner.invoke(app, ["facebook", "login", "--minutes", "1"])
    assert result.exit_code == 1
    assert "Not signed in yet" in result.output
    assert driver.kinds().count("wait") == 20  # one minute of 3 second polls
    assert driver.closed


def test_login_reports_a_closed_window(app, driver):
    def closed(url: str) -> None:
        raise DriverError("The browser window was closed; try again to reopen it.")

    driver.goto = closed
    result = runner.invoke(app, ["facebook", "login"])
    assert result.exit_code == 1
    assert "window was closed" in result.output
    assert driver.closed


def test_doctor_prints_a_table_and_fails_on_any_failed_check(app, monkeypatch):
    checks = [
        Check("config", True, "platforms: fake"),
        Check("discord", False, "token set, but ids missing"),
    ]
    monkeypatch.setattr(commands, "run_doctor", lambda settings: checks)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    lines = result.output.splitlines()
    assert lines[0].split() == ["CHECK", "RESULT", "DETAIL"]
    assert lines[2].startswith("discord") and "FAIL" in lines[2]


def test_doctor_passes_when_everything_does(app, monkeypatch):
    monkeypatch.setattr(commands, "run_doctor", lambda settings: [Check("config", True, "fine")])
    assert runner.invoke(app, ["doctor"]).exit_code == 0


def test_doctor_reports_invalid_settings_without_a_traceback(app, monkeypatch):
    monkeypatch.setenv("MA_TICK_SECONDS", "soon")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "Invalid settings: tick_seconds" in result.output
    assert "Traceback" not in result.output
