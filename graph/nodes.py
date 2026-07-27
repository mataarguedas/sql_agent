"""Pure graph nodes.

Every node has signature ``def node(state: AgentState) -> dict`` and returns
only the fields it updates. No side effects beyond the DB/LLM calls the node
owns. Do not mutate the input state.
"""

from __future__ import annotations

from typing import Literal

from graph.state import AgentState


def understand(state: AgentState) -> dict:
    """Introspect schema and normalize the question.

    Produces ``schema_context`` (the relevant subset of tables/columns) and
    ``normalized_question`` (a disambiguated restatement).
    """
    raise NotImplementedError


def generate_sql(state: AgentState) -> dict:
    """Generate a single ``SELECT`` via structured LLM output.

    On retry, incorporates ``last_error`` and the prior ``sql`` so the model
    can self-correct. Returns updated ``sql`` and appends to ``sql_history``.
    """
    raise NotImplementedError


def safety_check(state: AgentState) -> dict:
    """Code-level safety guard (defense-in-depth, independent of DB role).

    Parses ``sql`` and rejects anything that is not a single read-only
    ``SELECT``/``WITH … SELECT``. Sets ``safety_passed`` and ``safety_reason``.
    Safety failures route to ``give_up`` and never retry.
    """
    raise NotImplementedError


def validate(state: AgentState) -> dict:
    """Run ``EXPLAIN <sql>`` to catch planner/syntax errors before execution.

    On failure, sets ``last_error`` so the retry loop can feed it back.
    """
    raise NotImplementedError


def execute(state: AgentState) -> dict:
    """Execute the query over the read-only connection.

    Applies a row limit and statement timeout. On success sets ``result_rows``
    and ``result_columns``; on failure sets ``last_error`` and increments
    ``retry_count``.
    """
    raise NotImplementedError


def check_results(state: AgentState) -> Literal["synthesize", "generate_sql", "give_up"]:
    """Routing node.

    Returns:
        ``"synthesize"`` on success, ``"generate_sql"`` if a retry is allowed,
        else ``"give_up"``.
    """
    raise NotImplementedError


def synthesize(state: AgentState) -> dict:
    """Compose the natural-language answer, grounded strictly in returned rows."""
    raise NotImplementedError


def give_up(state: AgentState) -> dict:
    """Terminal failure node.

    Produces an honest failure message (unsafe or retry-exhausted) without
    fabricating results. Sets ``failed = True``.
    """
    raise NotImplementedError
