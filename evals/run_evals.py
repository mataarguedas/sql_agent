"""Accuracy harness using execution-result equivalence (see PRD §8).

Run as ``python -m evals.run_evals``.

**Metric.** For each case the agent's generated SQL and the dataset's expected
SQL are both executed, and their result sets compared *order-insensitively*.
This is the point of result-equivalence: SQL phrased differently but computing
the same thing must score as correct.

Comparison is deliberately lenient in three specific ways, because the
alternative is failing queries that are actually right:

- **Row order is ignored** — rows are compared as a multiset, unless the
  question pins an order (``ORDER BY`` + ``LIMIT``, e.g. "top 5"), in which
  case ordering is significant and the sequence is compared as-is.
- **Column *names* are ignored** — the model naming a count ``total`` instead
  of ``order_count`` is not a semantic error.
- **Extra columns are tolerated** — a result is equivalent if *some* projection
  of the generated columns reproduces the expected rows, so answering "which
  products…" with ``(product_id, product_name)`` still matches ``(product_name)``.

Floats are compared to 6 significant digits: Northwind stores prices as
single-precision ``real``, so two correct queries can differ in the last bits
purely from summation order.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from itertools import permutations
from math import floor, log10
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import text

from db.connection import get_readonly_engine
from graph.build import build_graph
from graph.state import AgentState, run_status
from graph.tracing import traced_run

_DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
_RESULTS_DIR = Path(__file__).parent / "results"

# Above this many generated columns, stop trying every projection; the search
# is only there to forgive a few extra columns, not to brute-force a match.
_MAX_PROJECTION_COLUMNS = 8
_FLOAT_SIG_DIGITS = 6


@dataclass(frozen=True)
class EvalCase:
    """One question/expected-SQL pair from ``dataset.jsonl``."""

    id: str
    category: str
    question: str
    expected_sql: str


@dataclass
class CaseResult:
    """Everything worth logging about one evaluated case (PRD §8)."""

    id: str
    category: str
    question: str
    expected_sql: str
    generated_sql: str
    expected_columns: list[str]
    expected_rows: list[list[Any]]
    actual_columns: list[str]
    actual_rows: list[list[Any]]
    retry_count: int
    status: str
    passed: bool
    failure_category: str | None
    answer: str
    error: str | None = None
    latency_s: float = 0.0
    sql_history: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


def load_dataset(path: Path = _DATASET_PATH) -> list[EvalCase]:
    """Read ``dataset.jsonl`` into :class:`EvalCase` objects."""
    cases: list[EvalCase] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        cases.append(
            EvalCase(
                id=row["id"],
                category=row.get("category", "uncategorized"),
                question=row["question"],
                expected_sql=row["expected_sql"],
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# value normalization + result equivalence
# --------------------------------------------------------------------------- #


def _round_significant(value: float, digits: int = _FLOAT_SIG_DIGITS) -> float:
    """Round to N significant digits so float noise doesn't fail a correct query."""
    if value == 0 or value != value or value in (float("inf"), float("-inf")):
        return value
    return round(value, -int(floor(log10(abs(value)))) + (digits - 1))


def normalize_value(value: Any) -> Any:
    """Canonicalize one cell so equivalent values from different SQL compare equal."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, (int, float)):
        value = _round_significant(float(value))
        # 1.0 and 1 are the same answer; COUNT may come back as either.
        return int(value) if float(value).is_integer() else value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _normalize_rows(rows: Iterable[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_value(v) for v in row) for row in rows]


def results_equivalent(
    expected_rows: Sequence[Sequence[Any]],
    actual_rows: Sequence[Sequence[Any]],
    order_matters: bool = False,
) -> bool:
    """True when ``actual`` reproduces ``expected`` under the metric described above."""
    exp = _normalize_rows(expected_rows)
    act = _normalize_rows(actual_rows)

    if len(exp) != len(act):
        return False
    if not exp:
        return True

    n_expected_cols = len(exp[0])
    n_actual_cols = len(act[0])
    if n_actual_cols < n_expected_cols:
        return False

    def matches(projection: tuple[int, ...]) -> bool:
        projected = [tuple(row[i] for i in projection) for row in act]
        if order_matters:
            return projected == exp
        return sorted(projected, key=repr) == sorted(exp, key=repr)

    # Fast path: the columns already line up.
    if matches(tuple(range(n_expected_cols))):
        return True
    if n_actual_cols > _MAX_PROJECTION_COLUMNS:
        return False
    return any(matches(p) for p in permutations(range(n_actual_cols), n_expected_cols))


def _order_matters(sql: str) -> bool:
    """Ordering is only significant when the expected SQL ranks and truncates."""
    upper = sql.upper()
    return "ORDER BY" in upper and "LIMIT" in upper


# --------------------------------------------------------------------------- #
# failure classification (PRD §8: tag failures by category)
# --------------------------------------------------------------------------- #


def classify_failure(
    result_status: str,
    state: AgentState,
    expected_rows: Sequence[Sequence[Any]],
    actual_rows: Sequence[Sequence[Any]],
) -> str:
    """Best-effort diagnosis of *why* a case failed, for failure-log triage."""
    if result_status == "unsafe":
        return "unsafe_sql"
    if result_status == "retry_exhausted":
        error = (state.get("last_error") or "").lower()
        if "does not exist" in error or "unknown column" in error:
            return "hallucinated_column"
        if "syntax" in error:
            return "syntax_error"
        return "retry_exhausted"

    if len(actual_rows) != len(expected_rows):
        if not expected_rows:
            return "expected_empty_got_rows"
        if not actual_rows:
            return "expected_rows_got_empty"
        # A join that should have been an anti-join / outer join is the classic
        # cause of "far too many rows".
        return "too_many_rows" if len(actual_rows) > len(expected_rows) else "too_few_rows"

    expected_width = len(expected_rows[0]) if expected_rows else 0
    actual_width = len(actual_rows[0]) if actual_rows else 0
    if actual_width < expected_width:
        return "missing_columns"
    return "wrong_values"


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def run_sql(sql: str) -> tuple[list[str], list[list[Any]]]:
    """Execute SQL on the read-only connection and return (columns, rows)."""
    with get_readonly_engine().connect() as conn:
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [list(row) for row in result.fetchall()]
    return columns, rows


def evaluate_case(case: EvalCase, app: Any, label: str | None = None) -> CaseResult:
    """Run the agent on one case and compare its results to the expected SQL."""
    expected_columns, expected_rows = run_sql(case.expected_sql)

    initial: AgentState = {
        "question": case.question,
        "sql_history": [],
        "retry_count": 0,
        "failed": False,
    }

    started = time.perf_counter()
    error: str | None = None
    state: AgentState = {}
    with traced_run(case.question) as run:
        try:
            state = app.invoke(initial)
        except Exception as exc:  # a crash is a failed case, not a failed suite
            error = f"{type(exc).__name__}: {exc}"
            state = {"failed": True, "answer": "", "last_error": error}
        latency = time.perf_counter() - started

        status = run_status(state)
        generated_sql = state.get("final_sql") or state.get("sql", "")
        actual_columns = list(state.get("result_columns") or [])
        actual_rows = [
            [row.get(col) for col in actual_columns] for row in (state.get("result_rows") or [])
        ]

        passed = (
            error is None
            and status == "success"
            and results_equivalent(expected_rows, actual_rows, _order_matters(case.expected_sql))
        )
        failure_category = (
            None if passed else classify_failure(status, state, expected_rows, actual_rows)
        )

        # Mirror the verdict into LangSmith so failures are filterable there too.
        run.finish(
            state,
            extra_tags=[
                "eval",
                f"eval_case:{case.id}",
                f"eval_result:{'pass' if passed else 'fail'}",
                *([f"eval_label:{label}"] if label else []),
            ],
            extra_metadata={
                "eval_case_id": case.id,
                "eval_category": case.category,
                "eval_passed": passed,
                "eval_failure_category": failure_category,
                "eval_expected_sql": case.expected_sql,
                "eval_expected_row_count": len(expected_rows),
                "eval_actual_row_count": len(actual_rows),
                **({"eval_label": label} if label else {}),
            },
        )

    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        expected_sql=case.expected_sql,
        generated_sql=generated_sql,
        expected_columns=expected_columns,
        expected_rows=expected_rows,
        actual_columns=actual_columns,
        actual_rows=actual_rows,
        retry_count=state.get("retry_count", 0),
        status=status,
        passed=passed,
        failure_category=failure_category,
        answer=state.get("answer", ""),
        error=error or state.get("last_error"),
        latency_s=round(latency, 3),
        sql_history=list(state.get("sql_history") or []),
    )


def _json_safe(value: Any) -> Any:
    """Make DB values (dates, Decimals) writable as JSON."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_evaluation(
    cases: Sequence[EvalCase],
    results_dir: Path = _RESULTS_DIR,
    label: str | None = None,
    write_logs: bool = True,
) -> tuple[list[CaseResult], Path | None]:
    """Evaluate every case, write the per-run log, and return the results."""
    app = build_graph()
    results: list[CaseResult] = []

    for index, case in enumerate(cases, start=1):
        result = evaluate_case(case, app, label=label)
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        detail = "" if result.passed else f"  [{result.failure_category}]"
        print(
            f"  [{index:2d}/{len(cases)}] {mark}  {case.id:42s} "
            f"retries={result.retry_count}  {result.latency_s:5.2f}s{detail}"
        )

    run_dir: Path | None = None
    if write_logs:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = results_dir / (f"{stamp}-{label}" if label else stamp)
        run_dir.mkdir(parents=True, exist_ok=True)

        with (run_dir / "results.jsonl").open("w") as handle:
            for result in results:
                handle.write(json.dumps(_json_safe(asdict(result))) + "\n")

        (run_dir / "summary.json").write_text(
            json.dumps(_json_safe(summarize(results, label)), indent=2) + "\n"
        )

    return results, run_dir


def summarize(results: Sequence[CaseResult], label: str | None = None) -> dict[str, Any]:
    """Aggregate accuracy overall, by question category, and by failure category."""
    passed = sum(r.passed for r in results)
    total = len(results)

    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(result.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)

    failures: dict[str, int] = {}
    for result in results:
        if not result.passed and result.failure_category:
            failures[result.failure_category] = failures.get(result.failure_category, 0) + 1

    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "total": total,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "by_category": by_category,
        "failure_categories": failures,
        "failed_cases": [
            {
                "id": r.id,
                "failure_category": r.failure_category,
                "expected_rows": len(r.expected_rows),
                "actual_rows": len(r.actual_rows),
                "generated_sql": r.generated_sql,
            }
            for r in results
            if not r.passed
        ],
        "total_retries": sum(r.retry_count for r in results),
    }


def print_summary(results: Sequence[CaseResult], run_dir: Path | None) -> None:
    """Print overall accuracy plus category and failure breakdowns."""
    summary = summarize(results)
    print("\n" + "=" * 68)
    print(f"ACCURACY: {summary['passed']}/{summary['total']} = {summary['accuracy']:.0%}")
    print("=" * 68)

    print("\nBy category:")
    for category, bucket in sorted(summary["by_category"].items()):
        print(f"  {category:20s} {bucket['passed']}/{bucket['total']}")

    if summary["failed_cases"]:
        print("\nFailures:")
        for failure in summary["failed_cases"]:
            print(
                f"  {failure['id']:42s} {failure['failure_category']:24s} "
                f"expected {failure['expected_rows']} rows, got {failure['actual_rows']}"
            )
        print("\nFailure categories:")
        for name, count in sorted(summary["failure_categories"].items()):
            print(f"  {name:28s} {count}")

    print(f"\nTotal retries across the suite: {summary['total_retries']}")
    if run_dir:
        print(f"Per-case log: {run_dir / 'results.jsonl'}")
        print(f"Summary:      {run_dir / 'summary.json'}")


def main() -> None:
    """Run the eval set, print accuracy, and write per-run logs to ``evals/results/``."""
    parser = argparse.ArgumentParser(description="Run the SQL-agent accuracy evals.")
    parser.add_argument("--dataset", type=Path, default=_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR)
    parser.add_argument(
        "--label", default=None, help="Names the results dir and tags LangSmith runs."
    )
    parser.add_argument("--only", nargs="*", default=None, help="Run only these case ids.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--no-logs", action="store_true", help="Skip writing evals/results/.")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        cases = cases[: args.limit]

    print(f"Running {len(cases)} eval cases" + (f" [{args.label}]" if args.label else "") + "\n")
    results, run_dir = run_evaluation(
        cases, results_dir=args.results_dir, label=args.label, write_logs=not args.no_logs
    )
    print_summary(results, run_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
