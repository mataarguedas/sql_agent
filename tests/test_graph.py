"""End-to-end graph tests.

The LLM and DB are faked so these tests are deterministic and require no
network access or live Postgres — see CLAUDE.md §8 ("deterministic where
possible"). ``safety_check`` runs for real (it's pure, in-process code), so
the safety-routing behavior under test is exercised exactly as it runs in
production.
"""

from __future__ import annotations

import pytest

import graph.nodes as nodes
from graph.build import build_graph
from graph.state import AgentState
from tests.fakes import FakeEngine, FakeLLM, FakeResult, refusing_engine


def _initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "sql_history": [],
        "retry_count": 0,
        "failed": False,
    }


@pytest.fixture(autouse=True)
def _fake_schema_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """``understand`` shouldn't need a real DB/SQLDatabase to introspect."""
    monkeypatch.setattr(nodes, "get_schema_context", lambda question: "FAKE SCHEMA CONTEXT")


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM(
        sql_queue=["SELECT customer_id FROM customers"],
        answer="There are two customers: ALFKI and ANATR.",
    )
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

    def _behavior(sql: str) -> FakeResult:
        if sql.strip().upper().startswith("EXPLAIN"):
            return FakeResult(columns=[], rows=[])
        return FakeResult(columns=["customer_id"], rows=[("ALFKI",), ("ANATR",)])

    engine = FakeEngine(_behavior)
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)

    app = build_graph()
    final_state = app.invoke(_initial_state("List all customers"))

    assert final_state["failed"] is False
    assert final_state["retry_count"] == 0
    assert final_state["safety_passed"] is True
    assert final_state["result_rows"] == [
        {"customer_id": "ALFKI"},
        {"customer_id": "ANATR"},
    ]
    assert final_state["answer"] == "There are two customers: ALFKI and ANATR."
    assert final_state["final_sql"] == "SELECT customer_id FROM customers"
    assert final_state["sql_history"] == ["SELECT customer_id FROM customers"]


def test_gives_up_after_exactly_three_attempts_on_unsatisfiable_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every attempt yields SQL that is safe but fails validation (e.g. a
    # column that doesn't exist) — the query is unsatisfiable, so generation
    # never succeeds no matter how many times it retries.
    fake_llm = FakeLLM(sql_queue=["SELECT nonexistent_column FROM customers"] * 3)
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

    def _behavior(sql: str) -> FakeResult:
        if sql.strip().upper().startswith("EXPLAIN"):
            raise RuntimeError('column "nonexistent_column" does not exist')
        raise AssertionError("execute should never run a query that failed validate")

    engine = FakeEngine(_behavior)
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)

    app = build_graph()
    final_state = app.invoke(_initial_state("Show me a column that doesn't exist"))

    assert final_state["failed"] is True
    assert final_state["retry_count"] == 3
    assert final_state["sql_history"] == ["SELECT nonexistent_column FROM customers"] * 3
    assert "nonexistent_column" in final_state["answer"]
    assert not fake_llm.sql_queue  # generate_sql was called exactly 3 times


def test_unsafe_input_gives_up_without_touching_the_db(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM(sql_queue=["DROP TABLE customers"])
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

    engine = refusing_engine()
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)

    app = build_graph()
    final_state = app.invoke(_initial_state("Delete all customers"))

    assert final_state["failed"] is True
    assert final_state["safety_passed"] is False
    assert final_state["retry_count"] == 0
    assert final_state["sql_history"] == ["DROP TABLE customers"]
    assert engine.connect_count == 0
    assert final_state["answer"]
