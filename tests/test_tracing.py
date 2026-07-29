"""Tracing helpers: env-gating, no-op behavior, and status/metadata mapping.

These tests never talk to LangSmith. The point of most of them is the negative
case: with the env vars absent, no span is opened and nothing is sent.
"""

from __future__ import annotations

from typing import Any

import pytest

import graph.tracing as tracing
from graph.state import AgentState, run_status

# Every test starts from a known-untraced environment: the autouse fixture in
# tests/conftest.py strips the LangSmith vars that ``.env`` would otherwise set.


def _enable(monkeypatch: pytest.MonkeyPatch, flag: str = "LANGCHAIN_TRACING_V2") -> None:
    monkeypatch.setenv(flag, "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-fake-key-for-tests")


class FakeRunTree:
    """Stands in for a ``langsmith`` RunTree; records what was attached to it."""

    id = "fake-run-id"

    def __init__(self) -> None:
        self.tags: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.outputs: dict[str, Any] | None = None
        self.end_calls = 0

    def add_tags(self, tags: list[str]) -> None:
        self.tags.extend(tags)

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(metadata)

    def end(self, outputs: dict[str, Any] | None = None) -> None:
        self.outputs = outputs
        self.end_calls += 1


# --------------------------------------------------------------------------- #
# tracing_enabled — both the flag and a key are required
# --------------------------------------------------------------------------- #


def test_tracing_disabled_when_env_absent() -> None:
    assert tracing.tracing_enabled() is False


@pytest.mark.parametrize("flag", ["LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"])
def test_tracing_enabled_with_either_flag_name(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    _enable(monkeypatch, flag)
    assert tracing.tracing_enabled() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_tracing_flag_accepts_truthy_spellings(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", value)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-fake-key-for-tests")
    assert tracing.tracing_enabled() is True


@pytest.mark.parametrize("value", ["", "false", "0", "no", "off"])
def test_tracing_flag_rejects_falsy_spellings(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", value)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-fake-key-for-tests")
    assert tracing.tracing_enabled() is False


def test_tracing_disabled_when_flag_set_but_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key-less flag would make every run attempt (and fail) an export."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert tracing.tracing_enabled() is False


# --------------------------------------------------------------------------- #
# traced_run — the disabled path must open nothing at all
# --------------------------------------------------------------------------- #


def test_traced_run_opens_no_span_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import langsmith

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no LangSmith span may be opened when tracing is off")

    monkeypatch.setattr(langsmith, "trace", _boom)

    with tracing.traced_run("how many customers?") as run:
        assert run.run_id is None
        run.finish({"failed": False, "retry_count": 0})  # must be inert


def test_traced_run_opens_span_with_base_tag_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    import langsmith

    opened: dict[str, Any] = {}
    fake = FakeRunTree()

    class _Ctx:
        def __enter__(self) -> FakeRunTree:
            return fake

        def __exit__(self, *exc_info: object) -> bool:
            return False

    def _fake_trace(**kwargs: Any) -> _Ctx:
        opened.update(kwargs)
        return _Ctx()

    monkeypatch.setattr(langsmith, "trace", _fake_trace)

    with tracing.traced_run("how many customers?") as run:
        assert run.run_id == "fake-run-id"

    assert opened["name"] == "sql_agent"
    assert opened["run_type"] == "chain"
    assert opened["tags"] == ["sql-agent"]
    assert opened["inputs"] == {"question": "how many customers?"}
    # The project must come from the environment, not be hardcoded here.
    assert "project_name" not in opened


# --------------------------------------------------------------------------- #
# RunHandle.finish
# --------------------------------------------------------------------------- #


def test_finish_attaches_status_retry_count_and_outputs() -> None:
    fake = FakeRunTree()
    handle = tracing.RunHandle(fake)

    state: AgentState = {
        "question": "top customers?",
        "failed": False,
        "safety_passed": True,
        "retry_count": 1,
        "sql_history": ["SELECT bad", "SELECT 1"],
        "final_sql": "SELECT 1",
        "answer": "Two customers.",
        "result_rows": [{"a": 1}, {"a": 2}],
    }

    handle.finish(state)

    assert fake.tags == ["status:success"]
    assert fake.metadata["status"] == "success"
    assert fake.metadata["retry_count"] == 1
    assert fake.metadata["sql_attempts"] == 2
    assert fake.metadata["row_count"] == 2
    assert fake.metadata["final_sql"] == "SELECT 1"
    assert fake.outputs == {
        "answer": "Two customers.",
        "final_sql": "SELECT 1",
        "row_count": 2,
    }
    assert fake.end_calls == 1


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"failed": False, "safety_passed": True}, "success"),
        ({"failed": True, "safety_passed": False, "safety_reason": "DROP"}, "unsafe"),
        ({"failed": True, "safety_passed": True, "retry_count": 3}, "retry_exhausted"),
    ],
    ids=["success", "unsafe", "retry_exhausted"],
)
def test_finish_status_tag_matches_run_status(state: AgentState, expected: str) -> None:
    fake = FakeRunTree()
    tracing.RunHandle(fake).finish(state)

    assert fake.tags == [f"status:{expected}"]
    assert fake.metadata["status"] == expected
    # The HTTP response derives from the same helper, so they cannot disagree.
    assert run_status(state) == expected


def test_finish_is_idempotent() -> None:
    """A double finish must not close the span twice."""
    fake = FakeRunTree()
    handle = tracing.RunHandle(fake)

    handle.finish({"failed": False})
    handle.finish({"failed": True, "safety_passed": False})

    assert fake.end_calls == 1
    assert fake.tags == ["status:success"]


def test_finish_swallows_langsmith_errors() -> None:
    """A LangSmith outage must degrade tracing, never fail the user's query."""

    class ExplodingRunTree:
        def add_tags(self, tags: list[str]) -> None:
            raise RuntimeError("LangSmith is down")

    tracing.RunHandle(ExplodingRunTree()).finish({"failed": False})  # must not raise


def test_finish_on_disabled_handle_is_inert() -> None:
    tracing.RunHandle(None).finish({"failed": True, "safety_passed": False})
