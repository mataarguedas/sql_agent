"""Database initialization: load Northwind DDL/seed and create the read-only role.

Run as ``python -m db.init``. Uses the admin engine (``DATABASE_ADMIN_URL``);
never invoked at runtime by graph nodes.
"""

from __future__ import annotations

from pathlib import Path

from db.connection import get_admin_engine

_INIT_DIR = Path(__file__).parent
_NORTHWIND_SQL = _INIT_DIR / "northwind.sql"
_READONLY_ROLE_SQL = _INIT_DIR / "create_readonly_role.sql"


def _run_script(path: Path) -> None:
    """Execute a .sql file through the admin engine as a single script."""
    sql = path.read_text()
    engine = get_admin_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def main() -> None:
    """Load Northwind DDL + seed, then create/grant the read-only role."""
    print(f"Loading Northwind schema + seed from {_NORTHWIND_SQL.name}...")
    _run_script(_NORTHWIND_SQL)
    print(f"Creating read-only role from {_READONLY_ROLE_SQL.name}...")
    _run_script(_READONLY_ROLE_SQL)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
