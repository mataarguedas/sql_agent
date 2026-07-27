"""Layer-2 safety test: the read-only role must reject writes and DDL.

Skipped automatically when ``DATABASE_READONLY_URL`` is unset or the DB isn't
reachable, so the suite still runs in environments without a live Postgres.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from db.connection import get_readonly_engine


def _skip_if_no_db() -> None:
    if not os.getenv("DATABASE_READONLY_URL"):
        pytest.skip("DATABASE_READONLY_URL not set")
    try:
        with get_readonly_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Read-only DB not reachable: {e}")


def test_readonly_role_can_select() -> None:
    _skip_if_no_db()
    with get_readonly_engine().connect() as conn:
        row = conn.execute(text("SELECT 1 AS one")).one()
    assert row.one == 1


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE _sql_agent_probe (id int)",
        "INSERT INTO customers (customer_id, company_name) VALUES ('ZZZZZ', 'probe')",
        "UPDATE customers SET company_name = 'probe' WHERE customer_id = 'ALFKI'",
        "DELETE FROM customers WHERE customer_id = 'ALFKI'",
        "DROP TABLE customers",
    ],
)
def test_readonly_role_rejects_writes(statement: str) -> None:
    """Even if Layer 1 (the code guard) were bypassed, the DB itself must refuse."""
    _skip_if_no_db()
    with get_readonly_engine().connect() as conn:
        with pytest.raises((ProgrammingError, DBAPIError)):
            conn.execute(text(statement))
