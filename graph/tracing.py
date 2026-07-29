"""LangSmith tracing helpers for graph runs (PRD §5, Observability).

Per-node inputs/outputs, token usage, and latency are captured *automatically*
by LangChain/LangGraph once the LangSmith env vars are set — every node is a
step in the compiled graph, and every ``ChatOpenAI`` call reports its own token
usage, which LangSmith rolls up to the enclosing run. This module adds the
run-level facts LangSmith cannot infer on its own: ``retry_count`` and the
final ``status`` (``success`` | ``unsafe`` | ``retry_exhausted``).

Those two are only known once the run has *finished*, which rules out passing
them in at invoke time. It also rules out patching them onto the run
afterwards: LangSmith closes a run with a single terminal update and rejects a
second one with ``409 Duplicate run update requests``. So instead we open our
own root span with :func:`langsmith.trace` and invoke the graph inside it — the
graph nests underneath, and because *we* own the span, we can attach the status
just before it closes, while it is still open.

Everything here is a **no-op when the env vars are absent**: no LangSmith span
is opened, no client is constructed, and :meth:`RunHandle.finish` returns
immediately. Tracing is observability, never a dependency — failures are
swallowed and logged so a LangSmith outage can never fail a user's query.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from graph.state import AgentState, run_status

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Legacy ``LANGCHAIN_*`` names and current ``LANGSMITH_*`` names are both
# honored, matching what langchain-core itself accepts.
_TRACING_FLAG_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
_API_KEY_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")

_RUN_NAME = "sql_agent"
_BASE_TAG = "sql-agent"


def tracing_enabled() -> bool:
    """True only when tracing is switched on *and* an API key is available.

    Both conditions matter: a truthy flag without a key would make
    langchain-core attempt (and fail) to export traces on every run.
    """
    flag_on = any(os.getenv(var, "").strip().lower() in _TRUTHY for var in _TRACING_FLAG_VARS)
    has_key = any(os.getenv(var) for var in _API_KEY_VARS)
    return flag_on and has_key


def _run_metadata(state: AgentState, status: str) -> dict[str, Any]:
    """The run-level facts worth filtering and grouping by in LangSmith."""
    return {
        "status": status,
        "retry_count": state.get("retry_count", 0),
        "sql_attempts": len(state.get("sql_history") or []),
        "safety_passed": state.get("safety_passed"),
        "safety_reason": state.get("safety_reason") or "",
        "failed": bool(state.get("failed")),
        "final_sql": state.get("final_sql") or state.get("sql", ""),
        "row_count": len(state.get("result_rows") or []),
        "last_error": state.get("last_error") or "",
    }


def _run_outputs(state: AgentState) -> dict[str, Any]:
    return {
        "answer": state.get("answer", ""),
        "final_sql": state.get("final_sql") or state.get("sql", ""),
        "row_count": len(state.get("result_rows") or []),
    }


class RunHandle:
    """Handle to the root span of an in-flight graph run.

    ``run_tree`` is ``None`` when tracing is disabled, which makes
    :meth:`finish` a no-op.
    """

    def __init__(self, run_tree: Any | None) -> None:
        self._run_tree = run_tree
        self._finished = False

    @property
    def run_id(self) -> Any | None:
        """The root run's id, or ``None`` when tracing is disabled."""
        return getattr(self._run_tree, "id", None)

    def finish(
        self,
        state: AgentState,
        extra_tags: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach the terminal status, retry count, and outputs to the root span.

        ``extra_tags`` / ``extra_metadata`` let a caller record its own context
        alongside the run — the eval harness uses them to mark which dataset
        case a run came from and whether it passed.

        Safe to call at most once; later calls are ignored. Never raises.
        """
        if self._run_tree is None or self._finished:
            return
        self._finished = True
        try:
            status = run_status(state)
            self._run_tree.add_tags([f"status:{status}", *(extra_tags or [])])
            self._run_tree.add_metadata({**_run_metadata(state, status), **(extra_metadata or {})})
            self._run_tree.end(outputs=_run_outputs(state))
        except Exception:  # pragma: no cover - observability must never break a run
            logger.warning("LangSmith run finalization failed", exc_info=True)


@contextmanager
def traced_run(question: str) -> Iterator[RunHandle]:
    """Open a root LangSmith span for one graph run.

    Invoke the graph inside the ``with`` block so it nests under this span,
    then call :meth:`RunHandle.finish` with the final state to record the
    status. When tracing is disabled this yields an inert handle and opens
    nothing.

    Example:
        >>> with traced_run(question) as run:
        ...     final_state = app.invoke(initial_state)
        ...     run.finish(final_state)
    """
    if not tracing_enabled():
        yield RunHandle(None)
        return

    try:
        import langsmith as ls
    except Exception:  # pragma: no cover - langsmith is a declared dependency
        logger.warning("LangSmith import failed; continuing untraced", exc_info=True)
        yield RunHandle(None)
        return

    # project_name is deliberately omitted so LANGSMITH_PROJECT /
    # LANGCHAIN_PROJECT stays the single place the project is configured.
    with ls.trace(
        name=_RUN_NAME,
        run_type="chain",
        inputs={"question": question},
        tags=[_BASE_TAG],
    ) as run_tree:
        yield RunHandle(run_tree)
