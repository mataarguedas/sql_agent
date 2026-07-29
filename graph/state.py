"""`AgentState` — the single TypedDict threaded through every graph node.

Nodes read this state and return a *partial* dict of the fields they update;
LangGraph merges the partial back into the running state. No node mutates the
input in place, and no node holds internal state of its own.
"""

from __future__ import annotations

from typing import Literal, TypedDict

RunStatus = Literal["success", "unsafe", "retry_exhausted"]


class AgentState(TypedDict, total=False):
    """State passed between graph nodes. See PRD §6 for the field contract."""

    question: str
    normalized_question: str
    schema_context: str
    sql: str
    sql_history: list[str]
    safety_passed: bool
    safety_reason: str
    result_rows: list[dict]
    result_columns: list[str]
    last_error: str | None
    retry_count: int
    answer: str
    final_sql: str
    failed: bool


def run_status(state: AgentState) -> RunStatus:
    """Classify a terminal ``AgentState`` into the run status used everywhere.

    This is the single source of truth for the three statuses in PRD §5/§7 —
    the API response, the LangSmith run tag, and any future reporting all
    derive from it, so they can never disagree.
    """
    if not state.get("failed"):
        return "success"
    return "unsafe" if state.get("safety_passed") is False else "retry_exhausted"
