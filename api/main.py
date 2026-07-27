"""FastAPI app exposing the SQL agent.

The API layer is a thin adapter: it validates input, invokes the compiled
graph, and streams/serializes output. All logic lives in ``graph/`` and ``db/``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

app: FastAPI = FastAPI(title="SQL Analyst Agent")


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service status and read-only DB connectivity."""
    raise NotImplementedError


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Return the introspected table/column listing used for generation."""
    raise NotImplementedError


@app.post("/query")
async def query(payload: dict[str, Any]) -> Any:
    """Run a natural-language question through the agent.

    Streams per-node progress (SSE) when ``stream`` is true; otherwise returns
    the final JSON object with ``answer``, ``final_sql``, ``result_rows``,
    ``result_columns``, ``retry_count``, and ``status``.
    """
    raise NotImplementedError
