"""Pure graph nodes.

Every node has signature ``def node(state: AgentState) -> dict`` and returns
only the fields it updates. No side effects beyond the DB/LLM calls the node
owns. Do not mutate the input state.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import sqlglot
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlalchemy import text

from db.connection import get_readonly_engine
from db.schema import get_schema_context
from graph.prompts import (
    GENERATE_SQL_FEW_SHOTS,
    GENERATE_SQL_SYSTEM_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
)
from graph.state import AgentState

_FORBIDDEN_SQL_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,
)


class SQLQuery(BaseModel):
    """A single read-only SQL statement produced by the generation LLM call."""

    sql: str = Field(
        description=(
            "A single PostgreSQL SELECT (optionally WITH ... SELECT) statement "
            "that answers the question. No other statement type, no trailing "
            "commentary."
        )
    )


class Answer(BaseModel):
    """A natural-language answer produced by the synthesis LLM call."""

    answer: str = Field(
        description=(
            "A concise natural-language answer grounded strictly in the given "
            "rows. States plainly if there are no rows."
        )
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)


def _normalize_question(question: str) -> str:
    return " ".join(question.split())


def _format_few_shots() -> str:
    blocks = [f"Question: {ex['question']}\nSQL: {ex['sql']}" for ex in GENERATE_SQL_FEW_SHOTS]
    return "\n\n".join(blocks)


def understand(state: AgentState) -> dict:
    """Introspect schema and normalize the question.

    Produces ``schema_context`` (the relevant subset of tables/columns) and
    ``normalized_question`` (a disambiguated restatement).
    """
    question = state["question"]
    return {
        "schema_context": get_schema_context(question),
        "normalized_question": _normalize_question(question),
    }


def generate_sql(state: AgentState) -> dict:
    """Generate a single ``SELECT`` via structured LLM output.

    On retry, incorporates ``last_error`` and the prior ``sql`` so the model
    can self-correct. Returns updated ``sql`` and appends to ``sql_history``.
    """
    human_parts = [
        f"Examples:\n{_format_few_shots()}",
        f"Schema:\n{state['schema_context']}",
        f"Question: {state['normalized_question']}",
    ]
    last_error = state.get("last_error")
    if last_error:
        human_parts.append(
            "The previous attempt failed.\n"
            f"Previous SQL:\n{state.get('sql', '')}\n"
            f"Error:\n{last_error}\n"
            "Fix the query."
        )

    messages = [
        SystemMessage(content=GENERATE_SQL_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(human_parts)),
    ]
    structured_llm = _get_llm().with_structured_output(SQLQuery)
    result = structured_llm.invoke(messages)
    sql = result.sql.strip()

    return {"sql": sql, "sql_history": [*state.get("sql_history", []), sql]}


def safety_check(state: AgentState) -> dict:
    """Code-level safety guard (defense-in-depth, independent of DB role).

    Parses ``sql`` and rejects anything that is not a single read-only
    ``SELECT``/``WITH … SELECT``. Sets ``safety_passed`` and ``safety_reason``.
    Safety failures route to ``give_up`` and never retry.
    """
    sql = state.get("sql", "")
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except Exception as exc:  # sqlglot raises on unparseable SQL
        return {"safety_passed": False, "safety_reason": f"Could not parse SQL: {exc}"}

    if len(statements) != 1:
        return {
            "safety_passed": False,
            "safety_reason": f"Exactly one SQL statement is required, got {len(statements)}.",
        }

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        return {
            "safety_passed": False,
            "safety_reason": f"Only SELECT statements are allowed, got {type(statement).__name__}.",
        }

    for node in statement.walk():
        if isinstance(node, _FORBIDDEN_SQL_TYPES):
            return {
                "safety_passed": False,
                "safety_reason": f"Disallowed construct in query: {type(node).__name__}.",
            }

    return {"safety_passed": True, "safety_reason": ""}


def validate(state: AgentState) -> dict:
    """Run ``EXPLAIN <sql>`` to catch planner/syntax errors before execution.

    On failure, sets ``last_error`` so the retry loop can feed it back. Never
    runs against SQL that failed :func:`safety_check`.
    """
    if not state.get("safety_passed", False):
        return {"last_error": state.get("safety_reason") or "SQL failed the safety check."}

    try:
        with get_readonly_engine().connect() as conn:
            conn.execute(text(f"EXPLAIN {state['sql']}"))
    except Exception as exc:
        return {"last_error": str(exc), "retry_count": state.get("retry_count", 0) + 1}

    return {"last_error": None}


def execute(state: AgentState) -> dict:
    """Execute the query over the read-only connection.

    Applies a row limit and statement timeout. On success sets ``result_rows``
    and ``result_columns``; on failure sets ``last_error`` and increments
    ``retry_count``. Never runs SQL that failed ``safety_check`` or ``validate``.
    """
    if state.get("last_error") or not state.get("safety_passed", False):
        return {}

    row_limit = int(os.getenv("QUERY_ROW_LIMIT", "1000"))
    try:
        with get_readonly_engine().connect() as conn:
            result = conn.execute(text(state["sql"]))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(row_limit)]
    except Exception as exc:
        return {"last_error": str(exc), "retry_count": state.get("retry_count", 0) + 1}

    return {"result_rows": rows, "result_columns": columns}


def check_results(state: AgentState) -> Literal["synthesize", "generate_sql", "give_up"]:
    """Routing node.

    Returns:
        ``"synthesize"`` on success, ``"generate_sql"`` if a retry is allowed
        (``retry_count < 3``), else ``"give_up"`` once retries are exhausted.
        Safety failures never reach this node — they route straight to
        ``give_up`` from ``safety_check``.
    """
    if not state.get("last_error"):
        return "synthesize"
    if state.get("retry_count", 0) < 3:
        return "generate_sql"
    return "give_up"


def synthesize(state: AgentState) -> dict:
    """Compose the natural-language answer, grounded strictly in returned rows.

    If an upstream node failed (safety check, validation, or execution), this
    reports the failure honestly instead of fabricating an answer — the
    ``give_up`` node does not exist yet in this milestone.
    """
    last_error = state.get("last_error")
    if last_error:
        return {
            "answer": f"I couldn't answer that question: {last_error}",
            "final_sql": state.get("sql", ""),
        }

    messages = [
        SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {state['normalized_question']}\n"
                f"SQL: {state['sql']}\n"
                f"Columns: {state.get('result_columns', [])}\n"
                f"Rows: {state.get('result_rows', [])}"
            )
        ),
    ]
    structured_llm = _get_llm().with_structured_output(Answer)
    result = structured_llm.invoke(messages)

    return {"answer": result.answer, "final_sql": state["sql"]}


def give_up(state: AgentState) -> dict:
    """Terminal failure node.

    Produces an honest failure message (unsafe or retry-exhausted) without
    fabricating results. Sets ``failed = True``.
    """
    if state.get("safety_passed") is False:
        reason = state.get("safety_reason") or "the generated SQL failed the safety check."
        message = f"I can't answer that question because the query wasn't safe to run: {reason}"
    else:
        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error") or "an unknown error."
        message = (
            f"I couldn't answer that question after {retry_count} attempt"
            f"{'s' if retry_count != 1 else ''}. Last error: {last_error}"
        )

    return {"answer": message, "failed": True}
