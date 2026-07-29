"""Tests for the eval harness's result-equivalence metric and failure tagging.

The metric decides every accuracy number the project reports, so its lenience
(and its limits) are pinned down here rather than discovered later.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from evals.run_evals import (
    _order_matters,
    classify_failure,
    load_dataset,
    normalize_value,
    results_equivalent,
    summarize,
)
from evals.run_evals import CaseResult


# --------------------------------------------------------------------------- #
# dataset integrity
# --------------------------------------------------------------------------- #


def test_dataset_has_twenty_unique_cases() -> None:
    cases = load_dataset()
    assert len(cases) == 20, "PRD §8 specifies a 20-pair eval set"
    assert len({c.id for c in cases}) == 20, "case ids must be unique"
    assert all(c.question.strip() and c.expected_sql.strip() for c in cases)


def test_dataset_covers_every_required_sql_feature() -> None:
    """PRD §8 lists the feature areas the set has to span."""
    categories = {c.category for c in load_dataset()}
    required = {
        "simple_filter",
        "aggregation",
        "join",
        "group_by",
        "group_by_having",
        "subquery",
        "date_logic",
        "anti_join",
    }
    assert required <= categories, f"missing: {required - categories}"


def test_dataset_includes_the_prd_anti_join_case() -> None:
    ids = {c.id for c in load_dataset()}
    assert "q16_antijoin_products_never_ordered" in ids


def test_every_expected_sql_is_a_single_select() -> None:
    """The harness executes these directly, so they must be read-only too."""
    for case in load_dataset():
        sql = case.expected_sql.strip().rstrip(";")
        assert sql.upper().startswith(("SELECT", "WITH")), case.id
        assert ";" not in sql, f"{case.id}: no stacked statements"


# --------------------------------------------------------------------------- #
# value normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        (Decimal("1996"), 1996),
        (1.0, 1),
        (1, 1),
        (28.833896092006, 28.8339),
        ("  Chai  ", "Chai"),
        (None, None),
        (True, True),
    ],
)
def test_normalize_value(raw: object, expected: object) -> None:
    assert normalize_value(raw) == expected


def test_float_noise_is_tolerated() -> None:
    """Northwind prices are single-precision; summation order shifts last bits."""
    assert results_equivalent([[12345.678]], [[12345.6781]])


def test_genuinely_different_numbers_still_fail() -> None:
    assert not results_equivalent([[100.0]], [[110.0]])


# --------------------------------------------------------------------------- #
# result equivalence
# --------------------------------------------------------------------------- #


def test_row_order_is_ignored_by_default() -> None:
    assert results_equivalent([["a"], ["b"]], [["b"], ["a"]])


def test_row_order_is_enforced_when_the_query_ranks_and_truncates() -> None:
    assert not results_equivalent([["a"], ["b"]], [["b"], ["a"]], order_matters=True)
    assert results_equivalent([["a"], ["b"]], [["a"], ["b"]], order_matters=True)


def test_extra_columns_are_tolerated() -> None:
    """Answering with (id, name) when (name) was expected is still correct."""
    assert results_equivalent([["Chai"], ["Chang"]], [[1, "Chai"], [2, "Chang"]])


def test_column_order_differences_are_tolerated() -> None:
    assert results_equivalent([["Berlin", "Alfreds"]], [["Alfreds", "Berlin"]])


def test_missing_columns_fail() -> None:
    assert not results_equivalent([["Alfreds", "Berlin"]], [["Alfreds"]])


def test_row_count_mismatch_fails() -> None:
    assert not results_equivalent([["a"]], [["a"], ["b"]])


def test_two_empty_result_sets_are_equivalent() -> None:
    assert results_equivalent([], [])


def test_empty_expected_versus_rows_fails() -> None:
    """The PRD anti-join case expects zero rows; the INNER JOIN bug returns many."""
    assert not results_equivalent([], [[1, "Chai"]])


def test_duplicate_rows_are_compared_as_a_multiset() -> None:
    assert results_equivalent([["a"], ["a"]], [["a"], ["a"]])
    assert not results_equivalent([["a"], ["a"]], [["a"], ["b"]])


def test_wide_results_skip_the_projection_search() -> None:
    """Guards against factorial blowup on very wide result sets."""
    wide = [[i for i in range(12)]]
    assert not results_equivalent([[11, 10]], wide)


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT a FROM t ORDER BY a LIMIT 5", True),
        ("SELECT a FROM t ORDER BY a", False),
        ("SELECT a FROM t", False),
    ],
)
def test_order_matters_detection(sql: str, expected: bool) -> None:
    assert _order_matters(sql) is expected


# --------------------------------------------------------------------------- #
# failure classification
# --------------------------------------------------------------------------- #


def test_classify_unsafe() -> None:
    assert classify_failure("unsafe", {}, [], []) == "unsafe_sql"


def test_classify_hallucinated_column() -> None:
    state = {"last_error": 'column "nope" does not exist'}
    assert classify_failure("retry_exhausted", state, [], []) == "hallucinated_column"


def test_classify_anti_join_symptom() -> None:
    """The documented PRD §8 bug: expected no rows, got the whole table."""
    assert classify_failure("success", {}, [], [[1], [2]]) == "expected_empty_got_rows"


def test_classify_too_many_rows() -> None:
    assert classify_failure("success", {}, [[1]], [[1], [2]]) == "too_many_rows"


def test_classify_wrong_values() -> None:
    assert classify_failure("success", {}, [[1]], [[2]]) == "wrong_values"


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #


def _case_result(case_id: str, category: str, passed: bool, failure: str | None) -> CaseResult:
    return CaseResult(
        id=case_id,
        category=category,
        question="q",
        expected_sql="SELECT 1",
        generated_sql="SELECT 1",
        expected_columns=[],
        expected_rows=[],
        actual_columns=[],
        actual_rows=[],
        retry_count=1,
        status="success" if passed else "retry_exhausted",
        passed=passed,
        failure_category=failure,
        answer="a",
    )


def test_summarize_computes_accuracy_and_breakdowns() -> None:
    results = [
        _case_result("a", "join", True, None),
        _case_result("b", "join", False, "too_many_rows"),
        _case_result("c", "anti_join", False, "too_many_rows"),
    ]
    summary = summarize(results, label="test")

    assert summary["passed"] == 1
    assert summary["total"] == 3
    assert summary["accuracy"] == pytest.approx(0.3333, abs=1e-4)
    assert summary["by_category"]["join"] == {"passed": 1, "total": 2}
    assert summary["failure_categories"] == {"too_many_rows": 2}
    assert {f["id"] for f in summary["failed_cases"]} == {"b", "c"}
    assert summary["total_retries"] == 3
