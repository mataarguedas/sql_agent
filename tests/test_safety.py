"""Safety-guard matrix: DML/DDL/stacked/comment-obfuscated variants must all fail."""

from __future__ import annotations

import pytest

from graph.nodes import safety_check

# (sql, label) — anything here must be rejected by safety_check.
REJECTED_STATEMENTS: list[tuple[str, str]] = [
    ("DROP TABLE customers", "ddl-drop"),
    ("CREATE TABLE evil (id int)", "ddl-create"),
    ("ALTER TABLE customers ADD COLUMN evil int", "ddl-alter"),
    ("TRUNCATE TABLE customers", "ddl-truncate"),
    ("INSERT INTO customers (customer_id) VALUES ('ZZZZZ')", "dml-insert"),
    ("UPDATE customers SET company_name = 'x'", "dml-update"),
    ("DELETE FROM customers", "dml-delete"),
    ("SELECT * FROM customers; DELETE FROM customers", "stacked-select-then-delete"),
    ("SELECT 1; SELECT 2", "stacked-two-selects"),
    ("SELECT 1; DROP TABLE customers", "stacked-select-then-drop"),
    ("/* harmless */ DELETE FROM customers", "comment-obfuscated-delete-leading"),
    ("DELETE /* trick */ FROM customers WHERE customer_id = 'ALFKI'", "comment-obfuscated-delete-inline"),
    ("SELECT 1 --\n; DROP TABLE customers", "comment-obfuscated-stacked-drop"),
    ("-- select the customers\nDELETE FROM customers", "comment-disguised-as-select"),
    ("EXPLAIN SELECT 1", "non-select-root-explain"),
    ("not valid sql at all !!!", "unparseable-garbage"),
]

# (sql, label) — anything here must be accepted by safety_check.
ACCEPTED_STATEMENTS: list[tuple[str, str]] = [
    ("SELECT 1", "trivial-literal-select"),
    ("SELECT * FROM customers", "select-star"),
    ("select * from customers", "lowercase-select"),
    ("SELECT customer_id, company_name FROM customers WHERE country = 'USA'", "filtered-select"),
    (
        "SELECT c.customer_id, COUNT(o.order_id) AS order_count "
        "FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
        "GROUP BY c.customer_id ORDER BY order_count DESC LIMIT 5",
        "join-aggregation-limit",
    ),
    (
        "WITH recent_orders AS (SELECT * FROM orders WHERE order_date > '1997-01-01') "
        "SELECT * FROM recent_orders",
        "cte-with-select",
    ),
    ("SELECT * FROM customers -- trailing comment", "benign-trailing-comment"),
    (
        "SELECT p.product_id, p.product_name "
        "FROM products p LEFT JOIN order_details od ON od.product_id = p.product_id "
        "WHERE od.order_id IS NULL",
        "left-join-anti-join",
    ),
]


@pytest.mark.parametrize("sql,label", REJECTED_STATEMENTS, ids=[label for _, label in REJECTED_STATEMENTS])
def test_safety_check_rejects(sql: str, label: str) -> None:
    result = safety_check({"sql": sql})
    assert result["safety_passed"] is False, f"expected rejection ({label}): {sql!r}"
    assert result["safety_reason"], f"expected a safety_reason for rejection ({label})"


@pytest.mark.parametrize("sql,label", ACCEPTED_STATEMENTS, ids=[label for _, label in ACCEPTED_STATEMENTS])
def test_safety_check_accepts(sql: str, label: str) -> None:
    result = safety_check({"sql": sql})
    assert result["safety_passed"] is True, (
        f"expected acceptance ({label}): {sql!r}, got reason: {result.get('safety_reason')!r}"
    )
    assert result["safety_reason"] == ""
