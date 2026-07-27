"""Request/response Pydantic models for the FastAPI layer (marshalling only)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
