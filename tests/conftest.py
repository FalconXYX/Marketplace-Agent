import pytest
from sqlmodel import Session

from marketplace_agent.config import Settings
from marketplace_agent.core.clock import FakeClock
from marketplace_agent.core.db import init_db, make_engine
from marketplace_agent.notify.base import RecordingNotifier


@pytest.fixture(scope="session")
def _absent_toml(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("settings") / "marketplace_agent.toml")


@pytest.fixture(autouse=True)
def _no_local_toml(monkeypatch, _absent_toml):
    """Settings read marketplace_agent.toml from the working directory, and the README
    quickstart makes one in the repo root: its hours or platforms must not change a test."""
    monkeypatch.setitem(Settings.model_config, "toml_file", _absent_toml)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        db_url="sqlite://",
        pickup_area="Bloor St W & Ossington Ave",
        pickup_address="123 Example St",
        enabled_platforms=["fake"],
    )


@pytest.fixture
def engine(settings):
    eng = make_engine(settings.database_url)
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    with Session(engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()
