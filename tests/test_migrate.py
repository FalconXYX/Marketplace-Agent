"""A database from before Stage 7 gains Commitment.bundle_key in place."""
import pytest
from sqlalchemy import inspect
from sqlmodel import Session, select

from marketplace_agent.core.db import init_db, make_engine
from marketplace_agent.core.migrate import SchemaError, upgrade_schema
from marketplace_agent.core.models import Commitment


@pytest.fixture
def old_db(tmp_path):
    """A current database with bundle_key dropped, as Stages 0 to 6 made it."""
    engine = make_engine(f"sqlite:///{tmp_path / 'agent.db'}")
    init_db(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_commitment_bundle_key")
        conn.exec_driver_sql("ALTER TABLE commitment DROP COLUMN bundle_key")
    yield engine
    engine.dispose()


def _columns(engine, table: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table)}


def test_an_old_database_gains_the_bundle_column_and_its_index(old_db):
    assert "bundle_key" not in _columns(old_db, "commitment")

    assert upgrade_schema(old_db) == ["commitment.bundle_key"]

    assert "bundle_key" in _columns(old_db, "commitment")
    indexes = {index["name"] for index in inspect(old_db).get_indexes("commitment")}
    assert "ix_commitment_bundle_key" in indexes
    with Session(old_db) as session:
        assert session.exec(select(Commitment)).all() == []


def test_upgrading_twice_or_a_current_database_changes_nothing(old_db, tmp_path):
    upgrade_schema(old_db)
    assert upgrade_schema(old_db) == []
    fresh = make_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    init_db(fresh)
    assert upgrade_schema(fresh) == []
    fresh.dispose()


def test_a_missing_required_column_is_refused_with_its_name(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'agent.db'}")
    init_db(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE commitment DROP COLUMN actor")

    with pytest.raises(SchemaError, match="commitment.actor"):
        upgrade_schema(engine)
    engine.dispose()
