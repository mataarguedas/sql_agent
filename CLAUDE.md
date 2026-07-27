# CLAUDE.md — SQL Analyst Agent

## 1. Project Summary

A natural-language-to-SQL agent that answers questions about a PostgreSQL database (sample: Northwind) via a LangGraph state machine. It generates a query, safety-checks it, validates it with `EXPLAIN`, executes it over a read-only connection, and synthesizes a plain-language answer, with a bounded self-correcting retry loop on execution errors.

## 2. Architecture

A LangGraph `StateGraph` over `AgentState`. State lives entirely in the `AgentState` TypedDict (see `graph/state.py`) and is threaded through every node; nodes hold no internal state.

**Topology**

```
START → understand → generate_sql → safety_check → validate → execute → check_results → synthesize → END
                          ▲                │            │          │            │
                          │                └──> give_up ┘          │            │
                          └──────────── retry (last_error) ────────┴── check_results ──> give_up (retries exhausted)
```

**Node responsibilities (one each)**
- `understand` — introspect schema, produce `schema_context` + `normalized_question`.
- `generate_sql` — LLM → structured `SELECT` (Pydantic); on retry, incorporate `last_error` + prior `sql`.
- `safety_check` — code-level guard; reject anything not a single read-only `SELECT`.
- `validate` — `EXPLAIN` the query; catch errors before execution.
- `execute` — run the query on the read-only connection with limit + timeout.
- `check_results` — route: success → `synthesize`; error & `retry_count < 3` → `generate_sql`; else → `give_up`.
- `synthesize` — LLM answer grounded strictly in returned rows.
- `give_up` — terminal honest-failure node; sets `failed = True`.

## 3. Tech Stack & Versions

Pin these in `pyproject.toml` / `requirements.txt`:

```
python           >=3.11,<3.13
langgraph        ~=0.2
langchain        ~=0.3
langchain-openai ~=0.2
langchain-community ~=0.3   # SQLDatabase utility
pydantic         ~=2.9
psycopg[binary]  ~=3.2
fastapi          ~=0.115
uvicorn[standard] ~=0.32
langsmith        ~=0.1
sqlglot          ~=25       # SQL parsing for the safety guard
pytest           ~=8.3
```

Model: `gpt-4o` (or `gpt-4o-mini` for cost-sensitive paths) via `langchain-openai`.

## 4. Repo Structure

```
.
├── graph/
│   ├── state.py          # AgentState TypedDict
│   ├── nodes.py          # pure node functions (one per responsibility)
│   ├── build.py          # StateGraph wiring, edges, conditional routing, compile()
│   └── prompts.py        # generation & synthesis prompt templates + few-shot examples
├── db/
│   ├── connection.py     # read-only engine/session factory
│   ├── schema.py         # SQLDatabase introspection + schema-context selection
│   └── init/             # Northwind DDL + seed, read-only role grant script
├── api/
│   ├── main.py           # FastAPI app; POST /query (streaming), /health, /schema
│   └── serializers.py    # request/response models (thin marshalling only)
├── evals/
│   ├── dataset.jsonl     # 20 question/expected-SQL pairs
│   ├── run_evals.py      # accuracy harness (result-equivalence)
│   └── results/          # per-run logs
├── tests/
│   ├── test_nodes.py     # each node in isolation
│   ├── test_safety.py    # guard rejection/acceptance matrix
│   └── test_graph.py     # end-to-end graph runs
├── docker-compose.yml    # app + postgres (with read-only role)
├── Dockerfile
├── pyproject.toml
├── PRD.md
└── CLAUDE.md
```

## 5. Commands

```bash
# Install deps
pip install -e ".[dev]"          # or: pip install -r requirements.txt

# Run the sample DB (Postgres + Northwind + read-only role)
docker compose up -d postgres
python -m db.init                # loads Northwind, creates read-only role

# Run the agent locally (single question, no server)
python -m graph.build --question "Top 5 customers by number of orders"

# Run the API
uvicorn api.main:app --reload    # POST http://localhost:8000/query

# Run evals
python -m evals.run_evals        # prints accuracy, writes evals/results/

# Run tests
pytest                           # all
pytest tests/test_safety.py -v   # safety guard only
```

## 6. Conventions

- **Type hints required** on every function signature and return.
- **Pydantic for all structured LLM output** — never parse free-form text; define a model (e.g. `SQLQuery`, `Answer`) and use structured output.
- **Nodes are pure functions** — signature `def node(state: AgentState) -> dict`, returning a *partial* state dict (only the keys it updates). No side effects beyond the DB/LLM calls the node owns; no mutation of the input state.
- **No business logic in the API layer** — `api/` only validates input, invokes the compiled graph, and streams/serializes output. All logic lives in `graph/` and `db/`.
- Prompts and few-shot examples live in `graph/prompts.py`, not inline in nodes.
- One responsibility per node; if a node does two things, split it.

## 7. Critical Safety Rules

**These are non-negotiable. Do not weaken, bypass, or "optimize away" any of them.**

1. **NEVER generate or permit write/DDL statements.** Only `SELECT` (and `WITH … SELECT`) is ever valid. No `INSERT`/`UPDATE`/`DELETE`/`MERGE`, no `CREATE`/`ALTER`/`DROP`/`TRUNCATE`, no stacked statements.
2. **The DB connection MUST use a read-only role** with `SELECT`-only grants and no write/DDL privileges. This is Layer 2 and must exist even in local/dev setups.
3. **The code-level guard is defense-in-depth and must never be removed.** `safety_check` stays even though the DB role also protects us — two independent layers. Removing either is a defect.
4. **All SQL generation flows through `safety_check`** before `validate`/`execute`. No node or code path may execute SQL that hasn't passed the guard. Safety failures route to `give_up` and never retry.
5. Enforce a statement timeout and row limit on execution.

## 8. Testing

- **Node unit tests (`test_nodes.py`)** — call each node with a hand-built `AgentState` and assert the returned partial dict. Mock the LLM (return fixed Pydantic objects) and mock/seed the DB so nodes are tested in isolation.
- **Safety matrix (`test_safety.py`)** — table-driven: assert `DROP`, `DELETE`, `INSERT`, `UPDATE`, stacked statements (`SELECT …; DELETE …`), and comment-obfuscated DML all fail; assert a range of legitimate `SELECT`/CTE queries pass.
- **End-to-end (`test_graph.py`)** — compile the graph and run real questions against the seeded read-only Northwind DB: assert happy-path answers, assert the retry loop reaches `give_up` after exactly 3 attempts on an unsatisfiable query, and assert an unsafe question terminates via `give_up` without touching the DB.
- Deterministic where possible: fixed seeds/temperature 0 for generation in tests.

## 9. Do / Don't

**Do**
- Use `EXPLAIN` in `validate` before ever executing a query.
- Select only the relevant subset of the schema into the prompt.
- Return partial state dicts from nodes; let LangGraph merge them.
- Feed `last_error` and the prior SQL back into `generate_sql` on retry.
- Ground synthesized answers strictly in returned rows.

**Don't**
- Don't dump the full schema into prompts for large DBs — retrieve/rank relevant tables instead.
- Don't rely on the prompt alone for safety — the guard and read-only role are the real controls.
- Don't put logic in `api/` — keep it a thin adapter over the graph.
- Don't remove or short-circuit `safety_check`, and don't retry on safety failures.
- Don't fabricate rows or values in `synthesize` or `give_up`.
