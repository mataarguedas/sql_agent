"""Schema introspection and relevant-subset selection for prompt context.

For the sample Northwind DB the full schema comfortably fits in a prompt, so
:func:`get_schema_context` falls back to the full schema when no tables match
the question. For larger DBs the retrieval strategy should be replaced with a
more sophisticated ranking (see PRD §10).
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_community.utilities import SQLDatabase

from db.connection import get_readonly_engine

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@lru_cache(maxsize=1)
def get_sql_database() -> SQLDatabase:
    """Return a cached LangChain ``SQLDatabase`` over the read-only engine."""
    return SQLDatabase(engine=get_readonly_engine())


def _tokenize(question: str) -> set[str]:
    return {tok.lower() for tok in _WORD_RE.findall(question)}


def _table_matches(table: str, tokens: set[str]) -> bool:
    """A table is relevant if its name (or its singular form) appears in the question."""
    name = table.lower()
    if name in tokens:
        return True
    if name.endswith("s") and name[:-1] in tokens:
        return True
    return any(name in tok or tok in name for tok in tokens if len(tok) > 3)


def get_schema_context(question: str) -> str:
    """Return the schema subset relevant to ``question`` as a prompt-ready string.

    Selects tables whose names appear in the question (with light singular/plural
    handling). If nothing matches, returns the full schema — the safe fallback
    for a small database. The returned string is the LangChain
    ``SQLDatabase.get_table_info`` output, which includes columns, types, and a
    few sample rows per table.
    """
    db = get_sql_database()
    all_tables: list[str] = db.get_usable_table_names()
    tokens = _tokenize(question)

    relevant = [t for t in all_tables if _table_matches(t, tokens)]
    tables = relevant or all_tables
    return db.get_table_info(table_names=tables)


def get_full_schema_info() -> dict[str, str | list[str]]:
    """Return every usable table name plus the full schema text, for ``GET /schema``."""
    db = get_sql_database()
    tables = db.get_usable_table_names()
    return {"tables": tables, "schema": db.get_table_info(table_names=tables)}
