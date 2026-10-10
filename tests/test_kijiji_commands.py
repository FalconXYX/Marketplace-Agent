import httpx
import pytest
import typer
from typer.testing import CliRunner

from marketplace_agent.platforms.kijiji.commands import register
from test_kijiji_support import EMAIL, PASSWORD, FakeKijiji, make_client

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Settings() in the command reads only what each test sets."""
    monkeypatch.chdir(tmp_path)
    for name in ("MA_KIJIJI_EMAIL", "MA_KIJIJI_PASSWORD", "MA_KIJIJI_LOCATION_ID"):
        monkeypatch.delenv(name, raising=False)


def _app(fake: FakeKijiji | None = None) -> typer.Typer:
    app = typer.Typer()
    if fake is None:
        register(app)
    else:
        register(app, client_factory=lambda settings: make_client(fake)[0])
    return app


def test_probe_reads_everything_and_writes_nothing(monkeypatch):
    monkeypatch.setenv("MA_KIJIJI_LOCATION_ID", "1700273")
    fake = FakeKijiji()

    result = runner.invoke(_app(fake), ["kijiji", "probe"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "ok    search (comps): 4 results for 'chair', 2 with a fixed price" in lines
    assert "ok    location: 1700273 is City of Toronto" in lines
    assert "ok    default category: Buy & Sell > Other" in lines
    assert "ok    sign in: signed in as user 1043690001" in lines
    assert "ok    my ads: 2 active ads; the newest (1744700001) has 37 views" in lines
    assert "ok    conversations: 2 conversations as seller" in lines
    assert "ok    messages: 3 messages in the latest conversation, 2 from the buyer" in lines
    assert "ok    post form: 11/11 fields found" in lines
    assert "8 ok, 0 failed, 0 skipped. Nothing was written." in result.output
    assert fake.submitted == [] and fake.sent == [] and fake.deleted == [] and fake.uploads == []


def test_probe_reports_a_failed_sign_in_and_skips_account_checks():
    fake = FakeKijiji(login_outcome="bad_password")

    result = runner.invoke(_app(fake), ["kijiji", "probe"])

    assert result.exit_code == 1
    assert "FAIL  sign in: kijiji: Kijiji rejected the email or password" in result.output
    assert "skip  conversations: needs sign in" in result.output
    assert "skip  location: MA_KIJIJI_LOCATION_ID not set" in result.output
    assert PASSWORD not in result.output and EMAIL not in result.output


def test_probe_flags_a_post_form_without_required_fields():
    fake = FakeKijiji()
    title_only = "<form><input name='postAdForm.title'></form>"
    fake.queue("/p-post-ad.html", httpx.Response(200, html=title_only))

    result = runner.invoke(_app(fake), ["kijiji", "probe"])

    assert result.exit_code == 1
    expected = "FAIL  post form: post form lacks description, price; publishing will fail"
    assert expected in result.output


def test_probe_without_an_account_explains_what_to_set():
    result = runner.invoke(_app(), ["kijiji", "probe"])

    assert result.exit_code == 1
    assert "MA_KIJIJI_EMAIL and MA_KIJIJI_PASSWORD" in result.output
