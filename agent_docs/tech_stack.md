# Tech Stack & Tools

## Runtime & Core

- **Language:** Python 3.9+
- **Solver:** `ortools` CP-SAT — pinned at `>=9.8` in `requirements.txt`. Do NOT upgrade without an ADR.
- **API Server:** FastAPI (latest) — async, Pydantic-native
- **Task Queue:** Celery (latest) — `app.core.celery_app`
- **Queue Broker:** Redis 7+ — `REDIS_URL` env var
- **Validation:** Pydantic v2 — `populate_by_name = True` pattern used for Go alias support
- **HTTP Client:** `requests` — used only in `solver_task.py` for webhook callback

## Module Structure

```
app/
├── main.py                   # FastAPI app + router registration
├── api/v1/
│   ├── solver_route.py       # Thin HTTP handlers — no logic
│   └── health.py             # GET /health
├── engine/
│   ├── model.py              # Engine: payload parsing + solver orchestration
│   ├── builder.py            # TaskModelBuilder: all CP-SAT model logic
│   └── utils.py              # filter_dummy_tasks, filter_dummy_overloads
├── tasks/
│   └── solver_task.py        # Celery task + webhook POST
├── schemas/
│   ├── request_schema.py     # SolverPayload (Go → Python contract)
│   └── response_schema.py    # SolverResponse (Python → Go contract)
└── core/
    ├── celery_app.py         # Celery + Redis configuration
    └── config.py             # REDIS_URL, WEBHOOK_URL env vars
```

## CP-SAT Patterns

### Creating Variables
```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Integer variable (free task)
start = model.NewIntVar(0, horizon, f"start_{task_id}")
end   = model.NewIntVar(0, horizon, f"end_{task_id}")

# Constant (pinned task — solver cannot move)
start = model.NewConstant(pinned_start_time)
end   = model.NewConstant(pinned_end_time)

# Boolean (task assigned to machine r_id?)
is_selected = model.NewBoolVar(f"{task_id}_on_{r_id}")

# Optional interval (used with AddNoOverlap on a machine)
opt_interval = model.NewOptionalIntervalVar(
    start, duration, end, is_selected, f"int_{task_id}_{r_id}"
)

# Fixed-size interval (unavailability window)
unavail = model.NewFixedSizeIntervalVar(w_start, w_end - w_start, name)
```

### Constraints
```python
# Exactly one machine per task
model.AddExactlyOne(literals)

# No two tasks overlap on the same machine (serial resource, capacity=1)
model.AddNoOverlap(intervals_on_machine)

# Batch machine: allow concurrent tasks up to capacity (capacity > 1)
demands = [1] * len(intervals)
model.AddCumulative(intervals, demands, capacity)

# Workforce cap (knitting + capacity_block demands)
model.AddCumulative(knitting_intervals, demands, MAX_FACTORY_MACHINES)

# Dependency: task B starts after task A ends
model.Add(task_b_start >= task_a_end)

# Pipeline offset: B starts after A has progressed by offset
model.Add(task_b_start >= task_a_start + offset)

# Conditional constraint (only enforced when boolean is True)
model.Add(start >= available_at).OnlyEnforceIf(is_selected)

# MaxEquality: Boolean is True iff any of the list is True
model.AddMaxEquality(any_bool, [bool_a, bool_b, bool_c])

# Implication: if A then B (used for symmetry breaking)
model.AddImplication(batch_active[k + 1], batch_active[k])
```

### Smart Batching Variables
| Variable | Type | Purpose |
|----------|------|---------|
| `x[t_id][k]` | `BoolVar` | Task `t_id` assigned to batch slot `k` |
| `batch_starts[k]` | `IntVar(0, horizon)` | Start time of batch slot `k` |
| `batch_active[k]` | `BoolVar` | Slot `k` has at least one task |
| `group_uses_slot[k]` | `BoolVar` | A compatibility group occupies slot `k` |

### Objective
```python
# Minimize weighted sum — all terms must be integers
model.Minimize(sum(objective_terms))

# Lateness penalty: higher priority = higher weight
weight = 10 ** (6 - priority)  # priority 1 → 100000, priority 5 → 1
lateness = model.NewIntVar(0, horizon, f"lat_{task_id}")
model.Add(lateness >= end_var - due_at)
objective_terms.append(lateness * weight * lateness_scale)
```

### Solver Configuration
```python
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = int(config.get("max_search_time", 300))
solver.parameters.relative_gap_limit = 0.01          # Stop at 1% gap
solver.parameters.num_search_workers = int(config.get("num_search_workers", 8))
solver.parameters.random_seed = int(config.get("random_seed", 42))
# NOTE: num_search_workers=1 required for byte-identical determinism
```

## Error Handling Pattern

```python
# Engine.solve() — always returns a dict, never raises
def solve(self) -> Dict[str, Any]:
    try:
        builder = (
            TaskModelBuilder(...)
            .build_time_variables()
            # ... chain
            .define_objective()
        )
        status = solver.Solve(builder.model)
        return builder.extract_results(solver, status)
    except Exception as exc:
        logger.error(f"Solver error: {exc}", exc_info=True)
        return {"status": "infeasible", "assignments": [], "overloads": []}

# extract_results() — handles OPTIMAL, FEASIBLE, or infeasible
if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    return {"status": "infeasible", "assignments": [], "overloads": []}
```

## Naming Conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase` (e.g., `TaskModelBuilder`, `Engine`)
- **Functions/methods:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `PENALTY_PER_ROLL_SWAP`, `MAX_GHOST_TASKS_BEFORE_WARN`)
- **CP-SAT variable names:** `f"{prefix}_{task_id}"` or `f"{task_id}_on_{r_id}"` — must be unique across the model
- **Task IDs:** assigned by Go backend; follow `K<n>-<order_id>` (Knitting) and `L<n>-<order_id>` (Linking) conventions

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker |
| `WEBHOOK_URL` | `http://backend:8082/api/webhook/solver` | Go callback endpoint |
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API port (override to 8083 in docker-compose) |
