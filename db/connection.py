"""Database connection factories.

Runtime callers MUST use :func:`get_readonly_engine`. The admin engine is
reserved for ``db/init`` (DDL, seed, role creation) and must never be used
by graph nodes at runtime — see CLAUDE.md §7.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

load_dotenv()

_DEFAULT_STATEMENT_TIMEOUT_MS = 10_000


def _statement_timeout_ms() -> int:
    return int(os.getenv("QUERY_STATEMENT_TIMEOUT_MS", _DEFAULT_STATEMENT_TIMEOUT_MS))


def _require_url(env_var: str) -> str:
    url = os.getenv(env_var)
    if not url:
        raise RuntimeError(
            f"{env_var} is not set. Copy .env.example to .env and fill it in."
        )
    return url


def _attach_statement_timeout(engine: Engine, timeout_ms: int) -> None:
    """Enforce ``statement_timeout`` on every new connection from this engine."""

    @event.listens_for(engine, "connect")
    def _set_timeout(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        with dbapi_connection.cursor() as cur:
            cur.execute(f"SET statement_timeout = {timeout_ms}")


@lru_cache(maxsize=1)
def get_readonly_engine() -> Engine:
    """Return a SQLAlchemy engine bound to the read-only role.

    This is the connection every graph node must use at runtime. A statement
    timeout is enforced on every connection so that runaway queries cannot
    exceed the configured budget.
    """
    url = make_url(_require_url("DATABASE_READONLY_URL"))
    engine = create_engine(url, pool_pre_ping=True, future=True)
    _attach_statement_timeout(engine, _statement_timeout_ms())
    return engine


@lru_cache(maxsize=1)
def get_admin_engine() -> Engine:
    """Return a SQLAlchemy engine bound to the admin role.

    For ``db/init`` only — do not import from graph nodes.
    """
    url = make_url(_require_url("DATABASE_ADMIN_URL"))
    return create_engine(url, pool_pre_ping=True, future=True)


def check_readonly_connectivity() -> bool:
    """Return True if the read-only role can open a connection and run a trivial query.

    Used by ``GET /health`` to confirm DB connectivity without leaking
    connection details to the API layer.
    """
    try:
        with get_readonly_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
