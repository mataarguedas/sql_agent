"""Request/response Pydantic models and state<->JSON marshalling for the FastAPI layer.

Thin marshalling only: shapes data between the compiled graph's ``AgentState``
and the wire format described in PRD §7. No LLM/DB calls happen here.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from graph.state import AgentState, run_status


class QueryRequest(BaseModel):
    """Incoming ``POST /query`` payload."""

    question: str
    stream: bool = True


class QueryResponse(BaseModel):
    """Final ``POST /query`` response (also emitted as the terminal SSE event)."""

    answer: str
    final_sql: str
    result_rows: list[dict[str, Any]] = Field(default_factory=list)
    result_columns: list[str] = Field(default_factory=list)
    retry_count: int = 0
    status: str  # "success" | "unsafe" | "retry_exhausted"


def initial_state(question: str) -> AgentState:
    """Build the ``AgentState`` the graph is invoked with for a fresh question."""
    return {
        "question": question,
        "sql_history": [],
        "retry_count": 0,
        "failed": False,
    }


def status_for(state: AgentState) -> str:
    """Map a terminal ``AgentState`` to the wire-level status (PRD §5/§7).

    Delegates to :func:`graph.state.run_status` so the HTTP response and the
    LangSmith status tag can never disagree.
    """
    return run_status(state)


def response_from_state(state: AgentState) -> QueryResponse:
    """Build the terminal ``QueryResponse`` from a final (or accumulated) ``AgentState``."""
    return QueryResponse(
        answer=state.get("answer", ""),
        final_sql=state.get("final_sql") or state.get("sql", ""),
        result_rows=state.get("result_rows") or [],
        result_columns=state.get("result_columns") or [],
        retry_count=state.get("retry_count", 0),
        status=status_for(state),
    )


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Format a single named Server-Sent Event.

    Encodes with FastAPI's ``jsonable_encoder`` rather than raw ``json.dumps``:
    result rows come straight from the driver and routinely contain ``date``,
    ``datetime`` and ``Decimal`` values, which plain ``json.dumps`` rejects.
    A raise here would abort the response body mid-stream, so the client would
    see progress events and then a silently truncated stream. Using the same
    encoder as the non-streaming path also keeps the two wire formats identical.
    """
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data))}\n\n"
