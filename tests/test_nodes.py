"""Unit tests for individual graph nodes (LLM/DB mocked).

Each node is called with a hand-built ``AgentState`` in isolation: the LLM is
faked to return fixed Pydantic objects (``tests.fakes.FakeLLM``) and the DB is
faked (``tests.fakes.FakeEngine``) rather than hitting a real connection. For
every node under test we assert both the returned partial-state dict *and*
that the input state was not mutated — nodes must return updates, never write
through their input (see CLAUDE.md §6: "Nodes are pure functions").

The full DDL/DML/stacked-statement safety matrix lives in ``test_safety.py``;
here we only check that ``safety_check`` behaves correctly as a node (accept
one, reject one) alongside its siblings.
"""

from __future__ import annotations

import copy
from typing import Callable

import pytest

import graph.nodes as nodes
from graph.state import AgentState
from tests.fakes import FakeEngine, FakeLLM, FakeResult, refusing_engine


def _call(node: Callable[[AgentState], dict], state: AgentState) -> dict:
    """Call ``node`` and assert it left ``state`` untouched."""
    before = copy.deepcopy(state)
    result = node(state)
    assert state == before, f"{node.__name__} must not mutate its input state"
    return result


# --------------------------------------------------------------------------- #
# understand
# --------------------------------------------------------------------------- #


def test_understand_sets_schema_context_and_normalized_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nodes, "get_schema_context", lambda question: f"SCHEMA FOR: {question}")
    raw_question = "  What   are   the top   products?  "
    state: AgentState = {"question": raw_question}

    result = _call(nodes.understand, state)

    assert result == {
        "schema_context": f"SCHEMA FOR: {raw_question}",
        "normalized_question": "What are the top products?",
    }


# --------------------------------------------------------------------------- #
# generate_sql
# --------------------------------------------------------------------------- #


def test_generate_sql_first_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM(sql_queue=["  SELECT 1  "])
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)
    state: AgentState = {
        "normalized_question": "How many rows?",
        "schema_context": "TABLE foo(id int)",
        "sql_history": [],
    }

    result = _call(nodes.generate_sql, state)

    assert result == {"sql": "SELECT 1", "sql_history": ["SELECT 1"]}
    assert result["sql_history"] is not state["sql_history"]
    human_content = fake_llm.last_messages[1].content
    assert "previous attempt failed" not in human_content.lower()


def test_generate_sql_retry_feeds_back_error_and_prior_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM(sql_queue=["SELECT 2"])
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)
    state: AgentState = {
        "normalized_question": "How many rows?",
        "schema_context": "TABLE foo(id int)",
        "sql": "SELECT bad",
        "sql_history": ["SELECT bad"],
        "last_error": 'column "bad" does not exist',
    }

    result = _call(nodes.generate_sql, state)

    assert result == {"sql": "SELECT 2", "sql_history": ["SELECT bad", "SELECT 2"]}
    human_content = fake_llm.last_messages[1].content
    assert "SELECT bad" in human_content
    assert 'column "bad" does not exist' in human_content


# --------------------------------------------------------------------------- #
# safety_check (full matrix in test_safety.py; smoke-tested here as a node)
# --------------------------------------------------------------------------- #


def test_safety_check_accepts_a_benign_select() -> None:
    state: AgentState = {"sql": "SELECT * FROM customers"}
    result = _call(nodes.safety_check, state)
    assert result == {"safety_passed": True, "safety_reason": ""}


def test_safety_check_rejects_a_drop() -> None:
    state: AgentState = {"sql": "DROP TABLE customers"}
    result = _call(nodes.safety_check, state)
    assert result["safety_passed"] is False
    assert result["safety_reason"]


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def test_validate_skips_db_when_safety_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = refusing_engine()
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {
        "safety_passed": False,
        "safety_reason": "not a SELECT",
        "sql": "DROP TABLE customers",
    }

    result = _call(nodes.validate, state)

    assert result == {"last_error": "not a SELECT"}
    assert engine.connect_count == 0


def test_validate_success_clears_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _behavior(sql: str) -> FakeResult:
        assert sql.upper().startswith("EXPLAIN")
        return FakeResult(columns=[], rows=[])

    engine = FakeEngine(_behavior)
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {"safety_passed": True, "sql": "SELECT 1"}

    result = _call(nodes.validate, state)

    assert result == {"last_error": None}
    assert engine.connect_count == 1


def test_validate_failure_sets_error_and_increments_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _behavior(sql: str) -> FakeResult:
        raise RuntimeError("syntax error at or near")

    engine = FakeEngine(_behavior)
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {"safety_passed": True, "sql": "SELECT bad(", "retry_count": 1}

    result = _call(nodes.validate, state)

    assert result == {"last_error": "syntax error at or near", "retry_count": 2}


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #


def test_execute_skips_db_when_last_error_present(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = refusing_engine()
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {"safety_passed": True, "last_error": "boom", "sql": "SELECT 1"}

    result = _call(nodes.execute, state)

    assert result == {}
    assert engine.connect_count == 0


def test_execute_skips_db_when_safety_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = refusing_engine()
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {"safety_passed": False, "sql": "DROP TABLE customers"}

    result = _call(nodes.execute, state)

    assert result == {}
    assert engine.connect_count == 0


def test_execute_success_returns_rows_and_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    def _behavior(sql: str) -> FakeResult:
        assert sql == "SELECT id FROM foo"
        return FakeResult(columns=["id"], rows=[(1,), (2,), (3,)])

    engine = FakeEngine(_behavior)
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {"safety_passed": True, "sql": "SELECT id FROM foo"}

    result = _call(nodes.execute, state)

    assert result == {
        "result_rows": [{"id": 1}, {"id": 2}, {"id": 3}],
        "result_columns": ["id"],
    }


def test_execute_respects_row_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUERY_ROW_LIMIT", "2")

    def _behavior(sql: str) -> FakeResult:
        return FakeResult(columns=["id"], rows=[(1,), (2,), (3,)])

    engine = FakeEngine(_behavior)
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {"safety_passed": True, "sql": "SELECT id FROM foo"}

    result = _call(nodes.execute, state)

    assert result["result_rows"] == [{"id": 1}, {"id": 2}]


def test_execute_failure_sets_error_and_increments_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _behavior(sql: str) -> FakeResult:
        raise RuntimeError("connection reset by peer")

    engine = FakeEngine(_behavior)
    monkeypatch.setattr(nodes, "get_readonly_engine", lambda: engine)
    state: AgentState = {"safety_passed": True, "sql": "SELECT 1", "retry_count": 2}

    result = _call(nodes.execute, state)

    assert result == {"last_error": "connection reset by peer", "retry_count": 3}


# --------------------------------------------------------------------------- #
# check_results
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"last_error": None}, "synthesize"),
        ({}, "synthesize"),
        ({"last_error": "boom", "retry_count": 0}, "generate_sql"),
        ({"last_error": "boom", "retry_count": 2}, "generate_sql"),
        ({"last_error": "boom", "retry_count": 3}, "give_up"),
        ({"last_error": "boom", "retry_count": 5}, "give_up"),
        ({"last_error": "boom"}, "generate_sql"),
    ],
    ids=[
        "no-error-succeeds",
        "no-error-key-succeeds",
        "error-retries-remaining",
        "error-last-retry-remaining",
        "error-retries-exhausted",
        "error-retries-way-exhausted",
        "error-missing-retry-count-defaults-to-zero",
    ],
)
def test_check_results_routing(state: AgentState, expected: str) -> None:
    result = _call(nodes.check_results, state)
    assert result == expected


# --------------------------------------------------------------------------- #
# synthesize
# --------------------------------------------------------------------------- #


def test_synthesize_short_circuits_on_last_error_without_calling_the_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_llm() -> FakeLLM:
        raise AssertionError("the LLM must not be called when last_error is set")

    monkeypatch.setattr(nodes, "_get_llm", _no_llm)
    state: AgentState = {"last_error": "boom", "sql": "SELECT 1", "normalized_question": "?"}

    result = _call(nodes.synthesize, state)

    assert result == {"answer": "I couldn't answer that question: boom", "final_sql": "SELECT 1"}


def test_synthesize_success_grounds_answer_in_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM(answer="There are 2 customers.")
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)
    state: AgentState = {
        "normalized_question": "How many customers?",
        "sql": "SELECT COUNT(*) FROM customers",
        "result_columns": ["count"],
        "result_rows": [{"count": 2}],
    }

    result = _call(nodes.synthesize, state)

    assert result == {
        "answer": "There are 2 customers.",
        "final_sql": "SELECT COUNT(*) FROM customers",
    }
    human_content = fake_llm.last_messages[1].content
    assert "How many customers?" in human_content
    assert "count" in human_content


# --------------------------------------------------------------------------- #
# give_up
# --------------------------------------------------------------------------- #


def test_give_up_on_safety_failure_never_mentions_retries() -> None:
    state: AgentState = {"safety_passed": False, "safety_reason": "DROP is not allowed"}

    result = _call(nodes.give_up, state)

    assert result["failed"] is True
    assert "DROP is not allowed" in result["answer"]
    assert "attempt" not in result["answer"].lower()


def test_give_up_on_retry_exhaustion_reports_error_and_attempts() -> None:
    state: AgentState = {"safety_passed": True, "retry_count": 3, "last_error": "still broken"}

    result = _call(nodes.give_up, state)

    assert result["failed"] is True
    assert "3 attempts" in result["answer"]
    assert "still broken" in result["answer"]
