"""Assemble and compile the LangGraph ``StateGraph``.

Wires the node topology described in CLAUDE.md §2 and PRD §4, including the
conditional edge from ``check_results`` and the safety-failure short-circuit
from ``safety_check`` to ``give_up``.
"""

from __future__ import annotations

from typing import Any

from graph.state import AgentState


def build_graph() -> Any:
    """Build and compile the SQL-agent state graph.

    Returns:
        The compiled LangGraph app, ready to ``invoke`` / ``stream``.
    """
    raise NotImplementedError


def run_once(question: str) -> AgentState:
    """Convenience: run the compiled graph on a single question and return final state."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the SQL agent on a single question.")
    parser.add_argument("--question", required=True, help="Natural-language question.")
    args = parser.parse_args()
    raise SystemExit(
        "Not implemented yet — scaffold only. Populate graph.build.run_once first."
    )
