"""Tests for the FastAPI adapter: SSE framing, encoding, and the served UI page.

The graph itself is stubbed here — node behavior is covered by ``test_nodes.py``
and ``test_graph.py``. What matters at this layer is that the wire format is
correct and that nothing can silently truncate the stream.
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.serializers import sse_event

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class FakeGraph:
    """Stand-in for the compiled graph, yielding canned per-node updates."""

    def __init__(
        self,
        updates: list[dict[str, dict[str, Any]]],
        raise_at: int | None = None,
    ) -> None:
        self._updates = updates
        self._raise_at = raise_at

    async def astream(
        self, state: dict[str, Any], stream_mode: str | None = None
    ) -> AsyncIterator[dict[str, dict[str, Any]]]:
        for index, update in enumerate(self._updates):
            if self._raise_at is not None and index == self._raise_at:
                raise RuntimeError("boom")
            yield update

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        merged = dict(state)
        for update in self._updates:
            for partial in update.values():
                merged.update(partial)
        return merged


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a raw SSE body into ``(event_name, data)`` pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in body.strip().split("\n\n"):
        if not frame.strip():
            continue
        name = "message"
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((name, json.loads("\n".join(data_lines))))
    return events


SUCCESS_UPDATES: list[dict[str, dict[str, Any]]] = [
    {"understand": {"schema_context": "customers(...)"}},
    {"generate_sql": {"sql": "SELECT 1"}},
    {"safety_check": {"safety_passed": True}},
    {"validate": {"last_error": None}},
    {"execute": {"result_rows": [{"n": 1}], "result_columns": ["n"]}},
    {"synthesize": {"answer": "One.", "final_sql": "SELECT 1", "failed": False}},
]


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


# --------------------------------------------------------------------------- #
# Encoding: the streamed body must survive real driver types
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        (dt.date(1997, 8, 25), "1997-08-25"),
        (dt.datetime(1997, 8, 25, 13, 30), "1997-08-25T13:30:00"),
        (Decimal("12.50"), 12.5),
    ],
)
def test_sse_event_encodes_types_json_dumps_rejects(value: Any, expected: Any) -> None:
    """Rows come straight from the driver; plain ``json.dumps`` raises on these.

    A raise inside the streaming generator ends the response body with no
    terminal event, so the client cannot tell a crash from a finished stream.
    """
    with pytest.raises(TypeError):
        json.dumps({"v": value})

    payload = json.loads(sse_event("result", {"v": value}).split("data: ", 1)[1])
    assert payload["v"] == expected


def test_stream_delivers_result_for_rows_with_dates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end guard for the same regression, through the real endpoint."""
    rows = [{"order_id": 10248, "order_date": dt.date(1996, 7, 4), "freight": Decimal("32.38")}]
    updates = [
        *SUCCESS_UPDATES[:4],
        {"execute": {"result_rows": rows, "result_columns": list(rows[0])}},
        {"synthesize": {"answer": "One order.", "final_sql": "SELECT ...", "failed": False}},
    ]
    monkeypatch.setattr(main, "_graph", FakeGraph(updates))

    events = parse_sse(client.post("/query", json={"question": "orders?"}).text)

    assert events[-1][0] == "result", "stream truncated before the terminal event"
    assert events[-1][1]["result_rows"] == [
        {"order_id": 10248, "order_date": "1996-07-04", "freight": 32.38}
    ]


# --------------------------------------------------------------------------- #
# Streaming contract
# --------------------------------------------------------------------------- #


def test_stream_emits_one_progress_event_per_node_then_a_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "_graph", FakeGraph(SUCCESS_UPDATES))

    response = client.post("/query", json={"question": "how many?", "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert [data["node"] for name, data in events if name == "progress"] == [
        "understand",
        "generate_sql",
        "safety_check",
        "validate",
        "execute",
        "synthesize",
    ]

    name, result = events[-1]
    assert name == "result"
    assert result["status"] == "success"
    assert result["answer"] == "One."
    assert result["final_sql"] == "SELECT 1"
    assert result["result_rows"] == [{"n": 1}]
    assert result["result_columns"] == ["n"]


def test_stream_repeats_a_node_on_retry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UI infers retries from a repeated node name, so the repeat must show."""
    updates = [
        {"understand": {}},
        {"generate_sql": {"sql": "SELECT bad"}},
        {"safety_check": {"safety_passed": True}},
        {"validate": {"last_error": "no such column", "retry_count": 1}},
        {"generate_sql": {"sql": "SELECT 1"}},
        {"safety_check": {"safety_passed": True}},
        {"validate": {"last_error": None}},
        {"execute": {"result_rows": [], "result_columns": []}},
        {"synthesize": {"answer": "Fixed.", "failed": False}},
    ]
    monkeypatch.setattr(main, "_graph", FakeGraph(updates))

    events = parse_sse(client.post("/query", json={"question": "q"}).text)
    nodes = [data["node"] for name, data in events if name == "progress"]

    assert nodes.count("generate_sql") == 2
    assert events[-1][1]["retry_count"] == 1


def test_stream_reports_a_terminal_error_event_when_the_graph_raises(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "_graph", FakeGraph(SUCCESS_UPDATES, raise_at=2))

    events = parse_sse(client.post("/query", json={"question": "q"}).text)

    assert [name for name, _ in events] == ["progress", "progress", "error"]
    assert "boom" in events[-1][1]["message"]


@pytest.mark.parametrize(
    "final_partial, expected_status",
    [
        ({"answer": "ok", "failed": False}, "success"),
        ({"answer": "no", "failed": True, "safety_passed": False}, "unsafe"),
        ({"answer": "no", "failed": True, "retry_count": 3}, "retry_exhausted"),
    ],
)
def test_stream_result_carries_the_run_status(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    final_partial: dict[str, Any],
    expected_status: str,
) -> None:
    monkeypatch.setattr(main, "_graph", FakeGraph([{"understand": {}}, {"done": final_partial}]))

    events = parse_sse(client.post("/query", json={"question": "q"}).text)
    assert events[-1][1]["status"] == expected_status


# --------------------------------------------------------------------------- #
# Non-streaming mode
# --------------------------------------------------------------------------- #


def test_non_streaming_returns_a_json_object(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "_graph", FakeGraph(SUCCESS_UPDATES))

    response = client.post("/query", json={"question": "how many?", "stream": False})

    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "answer": "One.",
        "final_sql": "SELECT 1",
        "result_rows": [{"n": 1}],
        "result_columns": ["n"],
        "retry_count": 0,
        "status": "success",
    }


# --------------------------------------------------------------------------- #
# The served page
# --------------------------------------------------------------------------- #


def test_index_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>SQL Analyst Agent</title>" in response.text


def test_page_targets_the_endpoints_and_events_the_api_actually_emits() -> None:
    """Catches a route or event rename that leaves the page silently broken."""
    html = (main._STATIC_DIR / "index.html").read_text()

    assert 'fetch("/query"' in html and 'fetch("/health")' in html
    for event_name in ("progress", "result", "error"):
        assert f'"{event_name}"' in html
    for node_name in ("understand", "generate_sql", "safety_check", "validate", "execute",
                      "synthesize", "give_up"):
        assert node_name in html, f"UI has no label for the {node_name} node"
