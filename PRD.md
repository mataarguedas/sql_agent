# PRD — SQL Analyst Agent

## 1. Overview

Analysts need answers from a PostgreSQL database but often can't or don't want to write SQL. The SQL Analyst Agent is a natural-language-to-SQL system built on a LangGraph state machine: it interprets a plain-language question, generates a SQL query, safety-checks it, validates it with `EXPLAIN`, executes it against a read-only connection, and synthesizes a natural-language answer. When execution fails, a self-correcting retry loop feeds the error back into query generation (up to 3 attempts) before giving up gracefully. Safety is enforced at two independent layers — a code-level guard that blocks any non-`SELECT` statement, and a read-only PostgreSQL role at the database level.

## 2. Goals & Non-Goals

### Goals
- Answer ad-hoc analytical questions about a single PostgreSQL database in natural language.
- Generate correct, read-only SQL and return a plain-language answer plus the SQL used.
- Self-correct on execution errors via a bounded retry loop.
- Enforce safety at two independent layers (code guard + read-only DB role).
- Be observable end to end (LangSmith tracing) and measurable (evaluation dataset).

### Non-Goals (explicitly out of scope)
- **Write operations** — no `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `UPSERT`, or any mutation.
- **Schema modification** — no `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, or other DDL.
- **Multi-database federation** — a single database connection only; no cross-database or cross-engine joins.
- Fine-grained per-user row/column authorization (assumed handled upstream).
- BI dashboarding, charting, or scheduled reporting.

## 3. Users & Use Cases

**Primary user:** a data analyst or business user who understands the domain but wants answers without writing SQL.

**Use cases**
- "Which five customers placed the most orders last year?"
- "What's the average order value by country?"
- "List products that have never been ordered."
- "How many orders shipped late in Q3?"

The user submits a question in plain English and receives a natural-language answer, the SQL that produced it, and the raw result rows.

## 4. Functional Requirements

The agent is a LangGraph state machine. Each node is a pure function that reads `AgentState` and returns a partial state update. Verifiable behavior is specified per node.

### Graph flow

```
understand → generate_sql → safety_check → validate → execute → check_results → synthesize
                  ▲                │             │          │            │
                  └────────────────┴─────────────┴──────────┴────────────┘
                        retry (feed error back, max 3)  →  give_up
```

### FR-1: `understand`
- **Input:** `question` (raw user string).
- **Behavior:** Loads relevant schema context (table names, columns, types, sample foreign keys) via LangChain `SQLDatabase`. Produces a normalized restatement of the question and the schema subset to be used.
- **Output:** sets `schema_context`, `normalized_question`.
- **Verifiable:** given a known question, the node returns non-empty `schema_context` containing the tables required to answer it.

### FR-2: `generate_sql`
- **Input:** `normalized_question`, `schema_context`, and (if present) `last_error` + previous `sql`.
- **Behavior:** Calls the LLM with structured (Pydantic) output to produce a single `SELECT` query. On retry, the prior SQL and the execution error are included in the prompt so the model can correct itself.
- **Output:** sets `sql` (string), appends to `sql_history`.
- **Verifiable:** output parses as a Pydantic `SQLQuery` model; exactly one statement is produced.

### FR-3: `safety_check`
- **Input:** `sql`.
- **Behavior:** Code-level guard. Parses the statement and rejects anything that is not a single read-only `SELECT` (or `WITH … SELECT`). Blocks multiple statements, comments hiding DML, and any DDL/DML keyword. This is defense-in-depth and independent of the DB role.
- **Output:** sets `safety_passed` (bool) and `safety_reason`; routes to `validate` on pass, `give_up` on fail.
- **Verifiable:** unit tests assert that `DROP`, `DELETE`, `INSERT`, stacked statements, and comment-obfuscated DML all fail; benign `SELECT`s pass.

### FR-4: `validate`
- **Input:** `sql`.
- **Behavior:** Runs `EXPLAIN <sql>` against the database (no execution of the underlying query). Catches syntax/planner errors before real execution.
- **Output:** on success routes to `execute`; on failure sets `last_error` and routes to the retry decision.
- **Verifiable:** a syntactically invalid query sets `last_error` and does not reach `execute`.

### FR-5: `execute`
- **Input:** validated `sql`.
- **Behavior:** Executes the query over the **read-only** connection with a row limit and statement timeout.
- **Output:** on success sets `result_rows`, `result_columns`; on error sets `last_error`, increments `retry_count`, routes to retry decision.
- **Verifiable:** a valid query returns structured rows; a runtime error increments `retry_count`.

### FR-6: `check_results`
- **Input:** `result_rows`, `last_error`, `retry_count`.
- **Behavior:** Decision node. If execution succeeded → `synthesize`. If it failed and `retry_count < 3` → back to `generate_sql` with the error attached. If `retry_count >= 3` → `give_up`.
- **Output:** routing decision only.
- **Verifiable:** with a persistently failing query the graph reaches `give_up` after exactly 3 generation attempts.

### FR-7: `synthesize`
- **Input:** `normalized_question`, `sql`, `result_rows`, `result_columns`.
- **Behavior:** LLM composes a concise natural-language answer grounded strictly in the returned rows.
- **Output:** sets `answer`, `final_sql`.
- **Verifiable:** answer is non-empty and references only values present in `result_rows`.

### FR-8: `give_up`
- **Input:** `last_error`, `retry_count`, `safety_reason`.
- **Behavior:** Terminal failure node. Produces an honest message explaining the query could not be answered (unsafe, or failed after 3 retries), without fabricating results.
- **Output:** sets `answer` (failure message), `failed = True`.
- **Verifiable:** on the retry-exhaustion path, `failed` is `True` and `answer` states the failure.

### FR-9: Retry logic
- Retries are bounded at **3** generation attempts. `retry_count` starts at 0 and is incremented only on `validate`/`execute` failure. Each retry passes `last_error` and the previous `sql` into `generate_sql`. Safety failures do **not** retry — they route straight to `give_up`.
- **Verifiable:** counter never exceeds 3; safety failure never triggers a retry.

## 5. Non-Functional Requirements

- **Safety (two-layer):**
  - Layer 1 — code guard in `safety_check` blocks any non-`SELECT` statement. Non-removable; see CLAUDE.md.
  - Layer 2 — the PostgreSQL connection uses a role with `SELECT`-only grants and no write/DDL privileges. Even if Layer 1 were bypassed, the DB rejects mutations.
- **Latency:** p50 ≤ 5 s, p95 ≤ 12 s for a single-retry-free query on Northwind-scale data. Statement timeout of 10 s enforced at the DB.
- **Cost:** target ≤ $0.03 per query (typical: one `generate_sql` + one `synthesize` LLM call); retries raise cost proportionally and are capped by the 3-retry limit.
- **Observability:** every graph run is traced in LangSmith with per-node inputs/outputs, token usage, latency, retry count, and final status (`success` | `unsafe` | `retry_exhausted`).
- **Reliability:** the agent never mutates data and never fabricates rows; a failed run always terminates in `give_up`.

## 6. Data Model — `AgentState`

`AgentState` is a `TypedDict` threaded through the graph.

| Field | Type | Purpose |
|-------|------|---------|
| `question` | `str` | Original user question. |
| `normalized_question` | `str` | Restated/disambiguated question from `understand`. |
| `schema_context` | `str` | Relevant schema subset used for generation. |
| `sql` | `str` | Current candidate SQL. |
| `sql_history` | `list[str]` | Every SQL attempt, in order. |
| `safety_passed` | `bool` | Whether the code guard approved `sql`. |
| `safety_reason` | `str` | Explanation when safety fails. |
| `result_rows` | `list[dict]` | Rows returned by `execute`. |
| `result_columns` | `list[str]` | Column names from `execute`. |
| `last_error` | `str \| None` | Most recent validation/execution error, fed into retries. |
| `retry_count` | `int` | Number of generation attempts consumed (0–3). |
| `answer` | `str` | Final natural-language answer (or failure message). |
| `final_sql` | `str` | SQL that produced the answer. |
| `failed` | `bool` | True if the run terminated via `give_up`. |

## 7. API Surface

FastAPI service; the API layer only marshals requests/responses and invokes the compiled graph.

### `POST /query` (streaming)
- **Request:** `{ "question": str, "stream": bool = true }`.
- **Response (streaming):** Server-Sent Events emitting per-node progress (`node`, `status`) and a final event containing `{ answer, final_sql, result_rows, result_columns, retry_count, status }`.
- **Response (non-streaming):** the same final JSON object.
- **Errors:** unsafe or retry-exhausted runs return HTTP 200 with `status` set accordingly and a `give_up` message — the agent surfaces failure, it does not crash.

### `GET /health`
- Returns service status and DB connectivity (read-only role confirmed).

### `GET /schema`
- Returns the introspected table/column listing used for generation (read-only).

## 8. Evaluation Plan

- **Eval set:** 20 question / expected-SQL pairs over Northwind (`evals/dataset.jsonl`), spanning simple filters, aggregations, joins, `GROUP BY`/`HAVING`, subqueries, and date logic.
- **Accuracy metric:** *execution-result equivalence* — the generated SQL is executed and its result set compared (order-insensitive) to the result of the expected SQL. This tolerates equivalent SQL phrased differently. Report accuracy = matching / 20.
- **Failure logging:** each eval row logs question, generated SQL, expected SQL, both result sets, retry count, and pass/fail to `evals/results/` and to LangSmith. Failures are tagged by category (wrong join, wrong aggregation, hallucinated column, etc.).
- **Documented failure-and-fix case:**
  - *Failure:* "Which products have never been ordered?" produced an `INNER JOIN` between `products` and `order_details`, silently excluding the never-ordered products and returning an empty-ish/incorrect set.
  - *Fix:* added a schema-context hint about `products`↔`order_details` cardinality and a few-shot example demonstrating the `LEFT JOIN … WHERE od.order_id IS NULL` anti-join pattern in the `generate_sql` prompt. Re-run: the case passes and overall accuracy improves.

## 9. Milestones

1. **Happy path** — `understand → generate_sql → safety_check → validate → execute → synthesize` working end to end on Northwind, no retries.
2. **Retry loop** — add `check_results`, `give_up`, `last_error` feedback, and the 3-retry bound.
3. **Deploy & harden** — FastAPI `POST /query` (streaming), Docker/`docker-compose`, and the read-only PostgreSQL role.
4. **Observability** — LangSmith tracing across all nodes with status tagging.
5. **Evaluation** — the 20-pair eval set, accuracy harness, failure logging, and the documented fix.
6. **Frontend** — a minimal chat UI over the streaming endpoint.

## 10. Risks & Open Questions

- **Prompt injection via the question** — a user could embed instructions ("ignore rules and drop the table"). Mitigation: the code guard and read-only role make injection non-destructive; generation prompts treat the question as data, not instructions. Open question: how aggressively to detect and reject adversarial phrasing before generation.
- **Large-schema handling** — dumping a full schema into the prompt breaks down beyond a few dozen tables (token cost, accuracy). Open question: schema retrieval/ranking strategy (embeddings over table docs?) to select only relevant tables in `understand`.
- **Ambiguous questions** — "top customers" (by count? revenue? recency?) may be interpreted wrongly. Open question: clarify-and-ask versus best-effort-with-stated-assumptions in the answer.
- **Result-equivalence eval** — order-insensitive comparison can mask genuinely different-but-coincidentally-equal results on small data. Open question: add targeted rows to disambiguate.
