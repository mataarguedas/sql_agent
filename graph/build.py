"""Assemble and compile the LangGraph ``StateGraph``.

Full topology per CLAUDE.md §2 / PRD §4, including the bounded retry loop:

    START → understand → generate_sql → safety_check → validate → execute → check_results → synthesize → END
                              ▲                │            │          │            │
                              │                └──> give_up ┘          │            │
                              └──────────── retry (last_error) ────────┴── check_results ──> give_up

``safety_check`` branches: on pass it proceeds to ``validate``, on failure it
routes straight to ``give_up`` and never retries (safety failures are
terminal, per CLAUDE.md §7). ``validate`` and ``execute`` both feed into
``check_results``, which decides: success → ``synthesize``; failure with
``retry_count < 3`` → back to ``generate_sql`` with ``last_error`` and the
prior ``sql`` fed into the prompt; retries exhausted → ``give_up``.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    check_results,
    execute,
    generate_sql,
    give_up,
    safety_check,
    synthesize,
    understand,
    validate,
)
from graph.state import AgentState
from graph.tracing import traced_run


def _route_after_safety_check(state: AgentState) -> Literal["validate", "give_up"]:
    """Proceed to ``validate`` if the safety guard passed, else to ``give_up``."""
    return "validate" if state.get("safety_passed") else "give_up"


def build_graph() -> Any:
    """Build and compile the SQL-agent state graph.

    Returns:
        The compiled LangGraph app, ready to ``invoke`` / ``stream``.
    """
    builder = StateGraph(AgentState)
    builder.add_node("understand", understand)
    builder.add_node("generate_sql", generate_sql)
    builder.add_node("safety_check", safety_check)
    builder.add_node("validate", validate)
    builder.add_node("execute", execute)
    builder.add_node("synthesize", synthesize)
    builder.add_node("give_up", give_up)

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "generate_sql")
    builder.add_edge("generate_sql", "safety_check")
    builder.add_conditional_edges(
        "safety_check",
        _route_after_safety_check,
        {"validate": "validate", "give_up": "give_up"},
    )
    builder.add_edge("validate", "execute")
    builder.add_conditional_edges(
        "execute",
        check_results,
        {
            "synthesize": "synthesize",
            "generate_sql": "generate_sql",
            "give_up": "give_up",
        },
    )
    builder.add_edge("synthesize", END)
    builder.add_edge("give_up", END)

    return builder.compile()


def run_once(question: str) -> AgentState:
    """Convenience: run the compiled graph on a single question and return final state."""
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "sql_history": [],
        "retry_count": 0,
        "failed": False,
    }
    with traced_run(question) as run:
        final_state: AgentState = app.invoke(initial_state)  # type: ignore[assignment]
        run.finish(final_state)
    return final_state


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the SQL agent on a single question.")
    parser.add_argument("--question", required=True, help="Natural-language question.")
    args = parser.parse_args()

    final_state = run_once(args.question)

    print(f"Question: {args.question}")
    print(f"SQL: {final_state.get('final_sql') or final_state.get('sql')}")
    print(f"Answer: {final_state.get('answer')}")
    if final_state.get("result_rows") is not None:
        print(f"Rows returned: {len(final_state['result_rows'])}")
