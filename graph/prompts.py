"""Prompt templates and few-shot examples for ``generate_sql`` and ``synthesize``.

Kept out of ``nodes.py`` so prompt iteration doesn't touch node logic.
"""

from __future__ import annotations

GENERATE_SQL_SYSTEM_PROMPT: str = ""
"""System prompt for the SQL-generation LLM call."""

GENERATE_SQL_FEW_SHOTS: list[dict[str, str]] = []
"""Few-shot examples (question → SQL) injected into the generation prompt."""

SYNTHESIZE_SYSTEM_PROMPT: str = ""
"""System prompt for grounding the final natural-language answer in returned rows."""
