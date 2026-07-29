"""Shared test doubles for the LLM and DB dependencies ``graph.nodes`` calls.

Named without a ``test_`` prefix so pytest does not try to collect it as a
test module. Shared between ``test_nodes.py`` (node isolation) and
``test_graph.py`` (end-to-end).
"""

from __future__ import annotations

from typing import Callable

from graph.nodes import Answer, SQLQuery


class _FakeBoundLLM:
    """Stand-in for ``ChatOpenAI(...).with_structured_output(Model)``."""

    def __init__(self, model_cls: type, fake_llm: "FakeLLM") -> None:
        self._model_cls = model_cls
        self._fake_llm = fake_llm

    def invoke(self, messages: list) -> object:
        self._fake_llm.last_messages = messages
        self._fake_llm.invoke_count += 1
        if self._model_cls is SQLQuery:
            if not self._fake_llm.sql_queue:
                raise AssertionError("generate_sql was called more times than expected")
            return SQLQuery(sql=self._fake_llm.sql_queue.pop(0))
        if self._model_cls is Answer:
            return Answer(answer=self._fake_llm.answer)
        raise AssertionError(f"unexpected structured-output model: {self._model_cls}")


class FakeLLM:
    """Fakes the one LLM call shape nodes.py uses: ``with_structured_output(...).invoke(...)``."""

    def __init__(self, sql_queue: list[str] | None = None, answer: str = "fake answer") -> None:
        self.sql_queue = list(sql_queue or [])
        self.answer = answer
        self.last_messages: list | None = None
        self.invoke_count = 0

    def with_structured_output(self, model_cls: type) -> _FakeBoundLLM:
        return _FakeBoundLLM(model_cls, self)


class FakeResult:
    def __init__(self, columns: list[str], rows: list[tuple]) -> None:
        self._columns = columns
        self._rows = rows

    def keys(self) -> list[str]:
        return self._columns

    def fetchmany(self, n: int) -> list[tuple]:
        return self._rows[:n]


class FakeConnection:
    def __init__(self, behavior: Callable[[str], FakeResult]) -> None:
        self._behavior = behavior

    def execute(self, statement: object) -> FakeResult:
        return self._behavior(str(statement))

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeEngine:
    """Fakes ``get_readonly_engine()``. Tracks whether ``.connect()`` was ever called."""

    def __init__(self, behavior: Callable[[str], FakeResult]) -> None:
        self._behavior = behavior
        self.connect_count = 0

    def connect(self) -> FakeConnection:
        self.connect_count += 1
        return FakeConnection(self._behavior)


def refusing_engine() -> FakeEngine:
    """An engine that fails the test loudly if the DB is touched at all."""

    def _refuse(sql: str) -> FakeResult:
        raise AssertionError(f"DB should never be touched, but got: {sql!r}")

    return FakeEngine(_refuse)
