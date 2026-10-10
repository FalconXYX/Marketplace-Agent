"""Bring a database made by an earlier version up to the current models.

init_db (core/db.py, frozen) runs create_all, which makes missing tables but never adds a
column to a table that already exists. Stage 7 added Commitment.bundle_key, so a database from
Stages 0 to 6 fails every ledger read until it is upgraded.

| Model change | Done here |
|---|---|
| New table | Nothing: init_db creates it |
| New nullable column | ALTER TABLE ADD COLUMN, then its indexes |
| Anything else (NOT NULL column, type change) | SchemaError naming the column |
"""

from sqlalchemy import Engine, inspect
from sqlalchemy.exc import OperationalError
from sqlmodel import SQLModel


class SchemaError(RuntimeError):
    pass


def upgrade_schema(engine: Engine) -> list[str]:
    """Add the nullable columns the models have and the database lacks. Run after init_db.
    Safe to repeat, and to race another process doing the same. Returns "table.column" added."""
    from marketplace_agent.core import models  # noqa: F401  registers tables

    inspector = inspect(engine)
    added = []
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            have = {column["name"] for column in inspector.get_columns(table.name)}
            missing = [column for column in table.columns if column.name not in have]
            for column in missing:
                if column.primary_key or not column.nullable:
                    raise SchemaError(
                        f"The database lacks {table.name}.{column.name}, which cannot be added "
                        "in place. Back up the data directory and start a new database."
                    )
                kind = column.type.compile(dialect=engine.dialect)
                try:
                    conn.exec_driver_sql(
                        f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {kind}'
                    )
                except OperationalError as exc:
                    if "duplicate column" not in str(exc):
                        raise
                added.append(f"{table.name}.{column.name}")
            if missing:
                for index in table.indexes:
                    index.create(conn, checkfirst=True)
    return added
