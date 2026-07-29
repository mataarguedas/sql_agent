"""Regression guards for prompt fixes that were paid for with eval failures.

Each rule pinned here was added in response to a measured, reproduced failure
(PRD §8). They are cheap to delete by accident during prompt tidying and
expensive to rediscover — the symptom is a plausible-looking but wrong number,
not an error — so the eval-backed behavior is locked down here.
"""

from __future__ import annotations

from graph.prompts import GENERATE_SQL_FEW_SHOTS, GENERATE_SQL_SYSTEM_PROMPT


def _all_few_shot_sql() -> str:
    return "\n".join(example["sql"] for example in GENERATE_SQL_FEW_SHOTS)


# --------------------------------------------------------------------------- #
# PRD §8 documented case: anti-join for "never ordered"
# --------------------------------------------------------------------------- #


def test_system_prompt_keeps_the_anti_join_rule() -> None:
    prompt = GENERATE_SQL_SYSTEM_PROMPT.upper()
    assert "LEFT JOIN" in prompt
    assert "IS NULL" in prompt
    assert "INNER JOIN" in prompt, "the rule should say why INNER JOIN is wrong here"


def test_few_shots_demonstrate_the_anti_join_pattern() -> None:
    sql = _all_few_shot_sql().upper()
    assert "LEFT JOIN" in sql and "IS NULL" in sql


# --------------------------------------------------------------------------- #
# Measured failure: money aggregated without the discount factor
# --------------------------------------------------------------------------- #


def test_system_prompt_keeps_the_discount_rule() -> None:
    """Without this, revenue/spend totals are silently overstated."""
    prompt = GENERATE_SQL_SYSTEM_PROMPT.lower()
    assert "discount" in prompt
    assert "1 - discount" in prompt or "(1 - discount)" in prompt


def test_system_prompt_keeps_the_averaging_grain_rule() -> None:
    """Without this, "average order value" averages order *lines*, not orders."""
    prompt = GENERATE_SQL_SYSTEM_PROMPT.lower()
    assert "average order value" in prompt
    assert "order_id" in prompt, "the rule must name the grain to aggregate to"


def test_few_shots_demonstrate_per_order_totals_with_discount() -> None:
    sql = _all_few_shot_sql().lower()
    assert "1 - od.discount" in sql or "1 - discount" in sql
    assert "group by od.order_id" in sql or "group by order_id" in sql
