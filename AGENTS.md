# AGENTS.md — Master Plan for Knitting-CRP-Ortools

## Project Overview & Stack

**App:** Knitting-CRP-Ortools
**Overview:** A production-grade CP-SAT solver microservice that acts as the scheduling brain of a textile knitting factory's APS (Advanced Planning & Scheduling) system. It consumes structured JSON payloads from a Go backend, solves a constrained job-shop scheduling problem (Knitting → Linking DAG with machine affinity, shift capacity, and pinned tasks), and returns deterministic, explainable schedules with bottleneck diagnostics via HTTP webhook callback.
**Stack:** Python 3.9+, OR-Tools CP-SAT v9.8+ (pinned), FastAPI, Celery + Redis, Pydantic v2
**Critical Constraints:**
- OR-Tools version pinned at v9.8+ — do NOT upgrade without an explicit ADR
- All CP-SAT model variables must be integer Virtual Minutes — no floats
- `SolverPayload` and `CompletedJobResultPayload` schemas are contracts with Go — do NOT change field names or types without coordinating with the Go team
- Stateless workers: max 8 GB RAM per Celery worker, no DB access from Python
- Deterministic output: fixed `random_seed` + `num_search_workers` → identical assignments for identical payloads

## Setup & Commands

Execute these commands for standard development workflows. Do not invent new commands.

- **Setup:** `pip install -r requirements.txt`
- **Development (API):** `uvicorn app.main:app --host 0.0.0.0 --port 8083 --reload`
- **Development (Worker):** `celery -A app.core.celery_app worker --loglevel=info --concurrency=1`
- **Testing:** `pytest tests/ -v`
- **Testing (single file):** `pytest tests/test_determinism.py -v`
- **Linting:** `ruff check app/ tests/`
- **Type check:** `mypy app/`
- **Benchmark:** `python scripts/benchmark.py`
- **Docker:** `docker compose up --build`

## Protected Areas

Do NOT modify these areas without explicit human approval:

- **Pydantic schemas:** `app/schemas/request_schema.py` and `app/schemas/response_schema.py` — Go contract
- **OR-Tools version pin** in `requirements.txt`
- **Webhook callback mechanism** in `app/tasks/solver_task.py` — Go integration contract
- **Docker + Compose files** — Infrastructure configuration
- **`app/core/config.py`** — Env var names are shared with Go deployment

## Coding Conventions

- **Architecture:** `Engine` orchestrates; `TaskModelBuilder` owns all CP-SAT logic. No CP-SAT calls outside `builder.py`.
- **Type annotations:** All public methods and functions must be typed. `Dict[str, Any]` is acceptable at JSON boundaries only.
- **Integer math:** All CP-SAT variable bounds, penalties, and time values must be `int`. Never pass `float` to CP-SAT.
- **Naming:** `snake_case` for Python files and functions; `UPPER_SNAKE_CASE` for module-level penalty constants; `PascalCase` for classes.
- **Logging:** Use `logger = logging.getLogger(__name__)` per module. Structured log fields for solve events.
- **Testing:** All new builder methods must have unit tests. Root-cause branches must have parametrized fixture tests.
- **Error handling:** CP-SAT failures return structured dicts — never raise unhandled exceptions from `Engine.solve()`.

## Agent Behaviors

These rules apply to all AI coding assistants working in this repo:

1. **Plan Before Execution:** ALWAYS propose a brief step-by-step plan before changing more than one file. For `builder.py` changes, state which methods are affected.
2. **Refactor Over Rewrite:** Prefer adding/modifying methods on `TaskModelBuilder` incrementally. Do not rewrite the builder chain.
3. **Context Compaction:** Write solver state discoveries and architectural decisions to `MEMORY.md` rather than repeating them in conversation.
4. **Iterative Verification:** Run `pytest tests/` after each logical change. Fix failures before proceeding. See `REVIEW-CHECKLIST.md`.
5. **Solver Correctness First:** When a CP-SAT model change is proposed, verify it does not make the model infeasible on known fixtures before committing.

## Active Phase

**Phase 4 — Dynamic Material Constraints ✅ COMPLETE**

All prior phases (P1–P3) are complete. See `MEMORY.md`.

**Phase 4 additions (2026-04-15):**
1. ~~`SolverPayload.material_capacities` + `SolverTask.material_demands` schema fields~~ ✅
2. ~~`TaskModelBuilder.build_material_constraints()` — `AddCumulative` per material~~ ✅
3. ~~`Engine.solve()` wired: `build_resource_allocations → build_material_constraints → apply_routing_constraints`~~ ✅
4. ~~`tests/test_material_constraints.py` (6 tests)~~ ✅

## Key Files Reference

| File | Purpose |
|------|---------|
| `app/engine/builder.py` | All CP-SAT model logic — primary working file |
| `app/engine/model.py` | Engine orchestration + solver config |
| `app/schemas/request_schema.py` | Go → Python contract (DO NOT break aliases) |
| `app/schemas/response_schema.py` | Python → Go contract |
| `app/tasks/solver_task.py` | Celery task + webhook callback |
| `tests/test_material_constraints.py` | Material cumulative constraint tests |
| `docs/PRD-Knitting-CRP-Ortools-MVP.md` | What to build + acceptance criteria |
| `docs/TechDesign-Knitting-CRP-Ortools-MVP.md` | How to build it + implementation sketches |
| `agent_docs/code_patterns.md` | CP-SAT patterns specific to this project |
| `agent_docs/testing.md` | Test fixtures + coverage requirements |
| `MEMORY.md` | Running state: decisions, known issues, completed phases |
