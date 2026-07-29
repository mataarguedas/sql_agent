"""FastAPI app exposing the SQL agent.

The API layer is a thin adapter: it validates input, invokes the compiled
graph, and streams/serializes output. All logic lives in ``graph/`` and ``db/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from api.serializers import (
    QueryRequest,
    QueryResponse,
    initial_state,
    response_from_state,
    sse_event,
)
from db.connection import check_readonly_connectivity
from db.schema import get_full_schema_info
from graph.build import build_graph
from graph.state import AgentState
from graph.tracing import traced_run

logger = logging.getLogger(__name__)

app: FastAPI = FastAPI(title="SQL Analyst Agent")

_graph = build_graph()
_STATIC_DIR = Path(__file__).parent / "static"


async def _stream_query(question: str) -> AsyncIterator[str]:
    """Yield one SSE ``progress`` event per completed node, then a final ``result`` event.

    Any unexpected failure is reported as a terminal ``error`` event. Without
    it an exception raised mid-generator just ends the response body, and the
    client cannot distinguish a crash from a completed stream.
    """
    state: AgentState = initial_state(question)
    try:
        with traced_run(question) as run:
            async for update in _graph.astream(state, stream_mode="updates"):
                for node_name, partial in update.items():
                    state.update(partial)
                    yield sse_event("progress", {"node": node_name, "status": "done"})
            run.finish(state)
        yield sse_event("result", response_from_state(state).model_dump())
    except Exception as exc:  # noqa: BLE001 - surfaced to the client, then logged
        logger.exception("Streaming query failed")
        yield sse_event("error", {"message": str(exc)})


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page chat UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return service status and read-only DB connectivity."""
    db_connected = check_readonly_connectivity()
    return {
        "status": "ok" if db_connected else "degraded",
        "database": "connected" if db_connected else "unreachable",
    }


@app.get("/schema")
async def schema() -> dict[str, Any]:
    """Return the introspected table/column listing used for generation."""
    return get_full_schema_info()


@app.post("/query", response_model=None)
async def query(payload: QueryRequest) -> QueryResponse | StreamingResponse:
    """Run a natural-language question through the agent.

    Streams per-node progress (SSE) when ``stream`` is true; otherwise returns
    the final JSON object with ``answer``, ``final_sql``, ``result_rows``,
    ``result_columns``, ``retry_count``, and ``status``.
    """
    if payload.stream:
        return StreamingResponse(_stream_query(payload.question), media_type="text/event-stream")

    with traced_run(payload.question) as run:
        final_state = await _graph.ainvoke(initial_state(payload.question))
        run.finish(final_state)
    return response_from_state(final_state)
