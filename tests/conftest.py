"""Shared pytest fixtures.

``db.connection`` calls ``load_dotenv()`` at import time, so a developer's real
``.env`` leaks into every test process. With tracing enabled there, simply
running the suite exports junk runs to the production LangSmith project —
noise in the exact dashboard the tracing feature exists to make readable.
Tests should never talk to LangSmith, so tracing is forced off for all of them.
"""

from __future__ import annotations

import pytest

_TRACING_ENV_VARS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",  # deprecated v1 flag; left set, it breaks langchain-core
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)


@pytest.fixture(autouse=True)
def _disable_langsmith_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every test to run untraced, whatever the ambient ``.env`` says."""
    for var in _TRACING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
