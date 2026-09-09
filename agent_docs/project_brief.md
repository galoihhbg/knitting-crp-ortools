# Project Brief (Persistent)

- **Product vision:** A deterministic, production-grade CP-SAT solver microservice that converts JSON job payloads from a Go APS backend into conflict-free knitting factory schedules with bottleneck diagnostics.
- **Target Audience:** Internal — Go backend (primary caller) and factory planners (consume results via Go UI).

## Architecture Principles

- **`Engine` orchestrates; `TaskModelBuilder` builds.** All CP-SAT model logic lives in `builder.py`. `Engine` in `model.py` only parses the payload, chains the builder, and runs the solver.
- **Route handlers are thin.** `solver_route.py` only validates `SolverPayload` and calls `.delay()`. No business logic.
- **Stateless workers.** Each Celery worker handles one solve at a time (`--concurrency=1`). Scale by adding containers, not threads.
- **Go contracts are frozen.** `request_schema.py` and `response_schema.py` define the wire format. Field renames must be coordinated with Go team.

## Conventions

- **Naming:** `snake_case` files, `PascalCase` classes, `UPPER_SNAKE_CASE` penalty constants with inline ratio comments
- **Integer math:** All CP-SAT variable bounds and objective terms must be `int`. No `float` in the model.
- **Comments:** Domain-specific Vietnamese comments in `builder.py` are intentional — they explain yarn-setup business logic to the factory domain team. Do not remove or translate.
- **Logging:** Every builder method logs its key constraint applications at `INFO`. Warnings for skipped/unresolvable inputs. Never silently skip.

## Material Constraint Invariants

- `build_material_constraints()` is the **only** place `AddCumulative` is called for material limits — no per-task-pair BoolVars
- Interval for material cumulative is `NewIntervalVar(tv["start"], duration, tv["end"])` — strictly bound to task window; material auto-released on task end
- `material_capacities` absent from payload → method is a no-op (early return); backward compatible
- All demands and capacities passed to `AddCumulative` must be `int`
- Undeclared materials in `task.material_demands` (not in `material_capacities`) are warned and skipped — no crash

## Quality Gates

- `pytest tests/ -v` must pass before any commit
- Determinism test (5-run replay with `num_search_workers=1`) must pass for any `builder.py` change
- No new CP-SAT calls outside `builder.py`
- No `float` values in CP-SAT variable creation or objective terms
- New penalty constants documented with `# LATENESS : AFFINITY : ACTIVATION ≈ 1000 : 10 : 2` ratio comment

## Key Commands

```bash
# Run API server (development)
uvicorn app.main:app --host 0.0.0.0 --port 8083 --reload

# Run Celery worker (development)
celery -A app.core.celery_app worker --loglevel=info --concurrency=1

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing

# Run single test file
pytest tests/test_determinism.py -v

# Type check
mypy app/

# Lint
ruff check app/ tests/

# Docker (all services)
docker compose up --build

# Hit solver endpoint manually
curl -X POST http://localhost:8083/api/v1/solve \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/payload_10_tasks.json
```

## Update Cadence

- Update `MEMORY.md` after every phase step completion or architectural decision
- Update `AGENTS.md` active phase when a phase is finished
- Update `agent_docs/testing.md` when new fixture types are added
- Update this file when a new architectural invariant is established
