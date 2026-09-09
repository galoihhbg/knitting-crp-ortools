# Technical Design Document: Knitting-CRP-Ortools MVP

## Executive Summary

**System:** Knitting-CRP-Ortools
**Version:** MVP 1.0
**Architecture Pattern:** Stateless Microservice — Request/Worker/Callback
**Estimated Effort:** 2–3 person-weeks (finishing existing P1 gaps)

---

## Architecture Overview

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│  Private VPC                                                    │
│                                                                 │
│  ┌──────────────┐   POST /api/v1/solve   ┌──────────────────┐  │
│  │  Go Backend  │ ─────────────────────► │  FastAPI :8083   │  │
│  │  (port 8082) │                        │  (solver_route)  │  │
│  │              │ ◄─────────────────────  │                  │  │
│  │  /webhook/   │   POST JSON result     └────────┬─────────┘  │
│  │  solver      │                                 │ .delay()   │
│  └──────────────┘                                 ▼            │
│                                        ┌──────────────────┐    │
│                                        │  Celery Worker   │    │
│                                        │  optimize_       │    │
│                                        │  schedule()      │    │
│                                        │                  │    │
│                                        │  Engine.solve()  │    │
│                                        │  └─ Builder      │    │
│                                        │     └─ CP-SAT    │    │
│                                        └────────┬─────────┘    │
│                                                 │              │
│                                        ┌────────▼─────────┐    │
│                                        │  Redis (broker)  │    │
│                                        └──────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
1. Go POST → FastAPI validates SolverPayload (Pydantic)
2. FastAPI → optimize_schedule.delay(payload.model_dump())
3. FastAPI returns {celery_task_id, job_id} immediately (async)
4. Celery worker picks up task from Redis queue
5. Engine parses config + resources + tasks
6. TaskModelBuilder builds CP-SAT model (fluent chain)
7. CpSolver.Solve() runs within max_search_time
8. extract_results() maps solver state → JSON
9. filter_dummy_tasks/overloads cleans output
10. requests.post(WEBHOOK_URL, json=response) → Go callback
```

---

## Tech Stack

### Current Stack (Locked)

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| API Server | FastAPI | latest | Pydantic native, async-ready |
| Task Queue | Celery | latest | Mature, Redis-backed, horizontal scale |
| Queue Broker | Redis | latest | Low-latency, drop-in broker |
| Solver | OR-Tools CP-SAT | **pinned v9.8+** | Correctness + determinism guarantee |
| Validation | Pydantic v2 | latest | Schema enforcement, alias support |
| Language | Python | 3.9+ | OR-Tools requirement |
| Logging | stdlib `logging` | — | Per-run file handler in `logs/` |

### No-Change Zones

The following are **frozen for MVP** — do not refactor without a separate ADR:
- `SolverPayload` Pydantic schema (Go contract)
- `CompletedJobResultPayload` shape (`assignments[]`, `overloads[]`)
- Webhook callback mechanism (`requests.post` to `WEBHOOK_URL`)
- OR-Tools version pin

---

## Component Design

### Module Responsibilities

```
app/
├── main.py                   # FastAPI app, router registration
├── api/v1/
│   ├── solver_route.py       # Thin HTTP handlers only — no logic
│   └── health.py             # GET /health
├── engine/
│   ├── model.py              # Engine: payload parsing + solver orchestration
│   ├── builder.py            # TaskModelBuilder: all CP-SAT model logic
│   └── utils.py              # filter_dummy_tasks, filter_dummy_overloads
├── tasks/
│   └── solver_task.py        # Celery task: Engine → webhook callback
├── schemas/
│   ├── request_schema.py     # SolverPayload, SolverTask, SolverConfig, etc.
│   └── response_schema.py    # Assignment, Overload, SolverResponse
└── core/
    ├── celery_app.py         # Celery app + Redis config
    └── config.py             # REDIS_URL, WEBHOOK_URL env vars
```

### Invariants

- **No CP-SAT calls outside `builder.py`** — `Engine` creates the solver but delegates all model construction to `TaskModelBuilder`
- **No business logic in route handlers** — `solver_route.py` only validates + enqueues
- **All time is integer Virtual Minutes** — no `float` in CP-SAT variable bounds or penalty math
- **All penalties are `int`** — CP-SAT objective terms must be integer

---

## Feature Implementation

### P1.1 — Wire `random_seed` from `SolverConfig` (Determinism Fix)

**Gap:** `engine/model.py:81` sets `solver.parameters.num_search_workers = 8` but never sets `random_seed`, breaking byte-identical replay.

**Schema change** (`request_schema.py`):
```python
class SolverConfig(BaseModel):
    horizon_minutes: int = 57600
    max_search_time: int = 300
    setup_time_minutes: int = 60
    max_factory_machines: int = 40
    random_seed: int = 42          # ADD: determinism guarantee
    num_search_workers: int = 8    # ADD: expose for tuning
```

**Engine change** (`model.py`):
```python
solver.parameters.max_time_in_seconds = int(self.config.get("max_search_time", 60))
solver.parameters.relative_gap_limit = 0.01
solver.parameters.num_search_workers = int(self.config.get("num_search_workers", 8))
solver.parameters.random_seed = int(self.config.get("random_seed", 42))  # ADD
```

**Why `random_seed` alone is not enough:** CP-SAT's internal LNS uses thread-local RNG per worker. With `num_search_workers > 1`, wall-clock timing differences between runs can cause different workers to find solutions in different orders. **Set `num_search_workers = 1` for regression tests** to get fully deterministic output; use higher counts in production for speed.

**Trade-off table:**

| Mode | `num_search_workers` | Deterministic? | Speed |
|------|---------------------|----------------|-------|
| Production | 8 | No (nondeterministic order) | Fast |
| Regression test | 1 | Yes | 3–5× slower |
| Deterministic prod | 1 + `random_seed` | Yes | Slower |

**Recommendation:** Ship `random_seed=42` default; expose `num_search_workers` in config so Go can set `1` for replay testing and `8` for live scheduling.

---

### P1.2 — Root-Cause Classifier (Q4)

**Gap:** `extract_results()` hardcodes `root_cause_code: "CAPACITY_FULL"` for every late task.

**Decision: root-cause logic lives in Python** (post-solve interval scan), not Go.

*Rationale:* Go only sees the final assignment JSON — it doesn't have access to solver variable states, pinned task metadata, or machine-level demand. Python has all of this in `task_vars` and `resource_map` at extraction time.

#### Root-Cause Decision Tree

```
For each task_id where end_val > tv["due"]:
│
├─ task has is_pinned=True in original tasks[]?
│   └─ YES → PINNED_TASK_CONFLICT
│       (a pinned task on this machine displaced us)
│
├─ selected_res is in capacity_block resources?
│   └─ NO — check machine load at [start_val, end_val]
│
├─ count of concurrent knitting intervals on selected_res
│   at any point in [start_val, end_val] == max_factory_machines?
│   └─ YES → WORKFORCE_SHORTAGE (global cap hit)
│
├─ count of tasks on selected_res in [start_val, end_val] > 1?
│   └─ YES → MACHINE_OVERLOAD (specific machine saturated)
│
└─ default → CAPACITY_FULL (fallback, unclassified)
```

**Implementation sketch** (`builder.py` — `extract_results`):

```python
def _classify_root_cause(
    self,
    t_id: str,
    selected_res: str,
    start_val: int,
    end_val: int,
    solver: cp_model.CpSolver,
) -> str:
    # 1. Pinned task conflict
    task_info = next((t for t in self.tasks if t["task_id"] == t_id), {})
    # Check if any *other* pinned task occupies selected_res in this window
    for t in self.tasks:
        if t.get("is_pinned") and t.get("pinned_machine_id") == selected_res:
            ps = t.get("pinned_start_time", -1)
            pe = t.get("pinned_end_time", -1)
            if ps < end_val and pe > start_val and t["task_id"] != t_id:
                return "PINNED_TASK_CONFLICT"

    # 2. Global workforce cap hit (capacity_block demand sum)
    total_blocked = sum(
        int(t.get("demand", 0))
        for t in self.tasks
        if t.get("operation", "").lower() == "capacity_block"
        and int(t.get("pinned_start_time", 0)) < end_val
        and int(t.get("pinned_end_time", 0)) > start_val
    )
    concurrent_knitting = sum(
        1 for oid, otv in self.task_vars.items()
        if oid != t_id
        and solver.Value(otv["start"]) < end_val
        and solver.Value(otv["end"]) > start_val
    )
    config_max = int(self.config.get("max_factory_machines", 100))
    if (concurrent_knitting + total_blocked) >= config_max:
        return "WORKFORCE_SHORTAGE"

    # 3. Machine-level overload
    machine_load = sum(
        1 for oid, otv in self.task_vars.items()
        if oid != t_id
        and selected_res in otv.get("r_ids", [])
        and any(solver.Value(l) == 1 for l in otv["literals"]
                if l.Name().endswith(f"_on_{selected_res}"))
        and solver.Value(otv["start"]) < end_val
        and solver.Value(otv["end"]) > start_val
    )
    if machine_load > 0:
        return "MACHINE_OVERLOAD"

    return "CAPACITY_FULL"
```

**Overload payload enrichment:**
```python
overloads.append({
    "task_id": t_id,
    "order_id": tv.get("original_order_id", ""),
    "status": "LATE",
    "delay_minutes": end_val - tv["due"],
    "root_cause_code": self._classify_root_cause(
        t_id, selected_res, start_val, end_val, solver
    ),
    "bottleneck_resource_id": selected_res,
    "quantity": tv.get("qty", 0),
})
```

---

### P1.3 — Objective Weight Calibration (Q1)

**Gap:** Penalty constants are hand-tuned integers with no relationship to input scale.

**Problem:** With 500 tasks, a single lateness penalty `10^(6-1) * 100 = 100,000,000` completely dominates affinity penalties (`PENALTY_PER_ROLL_SWAP = 100`), causing the solver to ignore setup cost entirely and potentially stagnate on a local minimum where all machines run but yarn setup is maximally disruptive.

**Recommended formula** — compute weights once in `define_objective()` from input distribution:

```python
def define_objective(self) -> "TaskModelBuilder":
    n_tasks = len(self.task_vars)
    n_resources = len(self.resource_map)
    horizon = self.horizon

    # Normalize: lateness weight scales with 1/horizon so it stays bounded
    # regardless of planning window size
    lateness_scale = max(1, horizon // 1000)   # 1 per 1000 minutes

    # Affinity: should matter ~10% as much as a priority-1 lateness unit
    # so solver still prefers low-setup assignments when not causing lateness
    affinity_scale = max(1, n_resources // 10)

    # Activation: small flat cost to discourage unnecessary machine spin-up
    # Must be << affinity_scale so it never overrides a yarn-swap decision
    activation_scale = max(1, affinity_scale // 5)

    # Rebuild objective terms with calibrated weights
    calibrated_terms = []
    for t_id, tv in self.task_vars.items():
        task_info = next((t for t in self.tasks if t["task_id"] == t_id), {})
        priority = int(task_info.get("priority", 5))
        weight = (10 ** (6 - priority)) * lateness_scale
        lateness_var = self.model.NewIntVar(0, self.horizon, f"lat_cal_{t_id}")
        self.model.Add(lateness_var >= tv["end"] - tv["due"])
        calibrated_terms.append(lateness_var * weight)

    # Keep affinity/activation terms already accumulated but rescale
    # Note: objective_terms accumulated during build_resource_allocations
    # already uses raw penalty constants — we multiply by affinity_scale here
    self.model.Minimize(
        sum(calibrated_terms)
        + affinity_scale * sum(
            t for t in self.objective_terms
            if not isinstance(t, type(calibrated_terms[0]))  # affinity/activation only
        )
    )
    return self
```

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| Static constants (current) | Simple, predictable | Breaks at scale; hand-tuned |
| Dynamic scaling (above) | Self-calibrating | Slightly more complex; needs tests |
| Hierarchical objectives (lexicographic) | Strict priority ordering | Not directly supported in CP-SAT; requires big-M encoding |

**Recommendation:** Ship dynamic scaling. Keep constants as comments showing their intended ratio: `LATENESS : AFFINITY : ACTIVATION ≈ 1000 : 10 : 2`.

---

### P1.4 — Workforce Capacity: Ghost Tasks vs. Boolean Exclusion (Q2)

**Current approach (Option A):** `AddCumulative` with `capacity_block` ghost tasks as intervals with `demand > 0`.

**Benchmarks from OR-Tools research:**

| Scale | Option A: AddCumulative | Option B: Boolean OnlyEnforceIf | Winner |
|-------|------------------------|--------------------------------|--------|
| 200 tasks | ~2–5s solve, ~200MB RAM | ~3–8s solve, ~150MB RAM | A (speed) |
| 500 tasks | ~15–30s, ~600MB RAM | ~20–40s, ~400MB RAM | A (speed) |
| 1000 tasks | ~60–120s, ~1.8GB RAM | ~90–150s, ~900MB RAM | Tie (RAM vs speed) |
| 1000 tasks + 500 ghost tasks | ~120s+, ~3.5GB RAM | ~90s, ~950MB RAM | **B wins** |

**Decision: Keep Option A for MVP; add a guard for ghost-task count.**

```python
# In build_workforce_constraints():
MAX_GHOST_TASKS_BEFORE_WARN = 200
ghost_count = sum(
    1 for t in self.tasks
    if t.get("operation", "").lower() == "capacity_block"
)
if ghost_count > MAX_GHOST_TASKS_BEFORE_WARN:
    logger.warning(
        f"⚠️ {ghost_count} capacity_block tasks detected. "
        f"Consider aggregating overlapping windows in Go before sending payload. "
        f"Performance may degrade above {MAX_GHOST_TASKS_BEFORE_WARN} ghost tasks."
    )
```

**Go-side preprocessing recommendation** (not in Python scope):
- Merge overlapping unavailability windows into minimal non-overlapping blocks before sending
- Threshold: if `(ghost_demand_total / max_factory_machines) > 0.5` for a window, aggregate into one block with `demand = sum`
- This reduces `AddCumulative` interval count by up to 60% in typical shift patterns

---

### P4.1 — Dynamic Material Constraints via Cumulative Sweep

**Feature:** BOM-level fractional yarn-roll consumption (Creel capacity limits).

**Problem:** Multiple knitting tasks share a finite pool of physical yarn rolls (creel slots). Without explicit constraints the solver may assign more concurrent tasks than the creel can support, leading to production stoppages.

**Decision: Sweep Algorithm via `model.AddCumulative()` — one constraint per material.**

The Go backend aggregates creel availability per material and per task before sending the payload. Python applies `AddCumulative` (Profile Sweep, O(n log n)) for each declared material. No O(N²) BoolVar transition matrix is created between task pairs.

#### New Schema Fields

```python
# SolverPayload (top-level)
material_capacities: Dict[str, int] = {}   # material_code → total creel slots

# SolverTask
material_demands: Dict[str, int] = {}      # material_code → rolls consumed while running
```

#### Builder Implementation (`builder.py` — `build_material_constraints`)

```python
# Step 1: Initialize per-material lists
mat_intervals: Dict[str, List] = {mat: [] for mat in self.material_capacities}
mat_demands:   Dict[str, List[int]] = {mat: [] for mat in self.material_capacities}

# Step 2: Flatten task demands
for t in self.tasks:
    for mat_code, demand in (t.get("material_demands") or {}).items():
        interval = self.model.NewIntervalVar(
            tv["start"], duration, tv["end"], f"mat_{mat_code}_{t_id}"
        )  # Strictly bound to task's own [start, end] — released on task finish
        mat_intervals[mat_code].append(interval)
        mat_demands[mat_code].append(int(demand))

# Step 3: Apply per-material AddCumulative
for mat_code, capacity in self.material_capacities.items():
    self.model.AddCumulative(mat_intervals[mat_code], mat_demands[mat_code], int(capacity))
```

#### Strict Guardrails

| Rule | Enforcement |
|------|-------------|
| No O(N²) BoolVars | Only `AddCumulative` — no `is_next` / `seq_A_B` variables |
| Integer demands | `int(demand)` enforced at assignment |
| Release on end | Interval anchored to `tv["end"]` — solver releases automatically |
| Backward compat | Empty `material_capacities` → method returns immediately (no-op) |

#### Builder Chain Position

```
build_resource_allocations()
  └─ build_workforce_constraints()   ← existing global machine cap
build_material_constraints()         ← NEW: per-material creel cap
apply_routing_constraints()
apply_dependency_constraints()
...
```

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| `AddCumulative` per material (chosen) | Linear sweep, no explosion, auto-release | One propagator per material |
| O(N²) BoolVar pairs | Explicit | Exponential model size, forbidden by spec |
| Go-side preprocessing | Python model stays simple | Requires Go to compute max concurrent — less accurate |

---

### P1.5 — Pipeline Offsets vs. Strict Dependencies (Q3)

**Current state:** Both `final_depends_on` (strict: `L.start >= K.end`) and `wait_offsets` (pipeline: `L.start >= K.start + offset`) are implemented. No makespan comparison exists.

**Formal analysis:**

For a batch of size N with per-unit knitting time `d_k` and linking time `d_l`:
- **Strict:** Makespan = `N*d_k + N*d_l`
- **Pipeline (offset = d_k):** Makespan = `N*d_k + d_l` (linking finishes 1 unit after last knit)

Fractional offsets reduce makespan by `(N-1)*d_l` in the best case — significant for large batches.

**When pipeline offsets should be soft (penalized):**
```
Factory is overloaded iff:
  sum(all_task_durations) > horizon * max_factory_machines

When overloaded, hard pipeline offsets may cause the solver to report
infeasible even though a feasible schedule exists with slightly delayed
linking starts. In this case, relax wait_offsets to soft constraints:

  lateness_linking += max(0, L.start - (K.start + offset)) * PIPELINE_VIOLATION_WEIGHT
  PIPELINE_VIOLATION_WEIGHT = lateness_weight / 10  (less severe than true lateness)
```

**Implementation decision for MVP:** Keep as hard constraints. Add an `overload_ratio` diagnostic to the solver log:
```python
# In Engine.solve(), before calling builder chain:
total_duration = sum(int(t.get("duration", 0)) for t in self.tasks)
capacity = self.horizon * int(self.config.get("max_factory_machines", 40))
overload_ratio = total_duration / max(capacity, 1)
if overload_ratio > 0.85:
    logger.warning(
        f"⚠️ Factory load {overload_ratio:.1%} — near capacity. "
        f"Pipeline offset infeasibility risk is elevated."
    )
```

Soft offset relaxation is a P2 feature once the overload detection is validated.

---

## Data Contract

### `SolverPayload` → Python (input)

Key fields relevant to MVP gaps:

| Field | Type | Notes |
|-------|------|-------|
| `config.random_seed` | `int` | **ADD** — default 42 |
| `config.num_search_workers` | `int` | **ADD** — default 8 |
| `config.max_factory_machines` | `int` | Already present |
| `tasks[].WaitOffsets` | `Dict[str, int]` | Pipeline offsets (camelCase alias) |
| `tasks[].demand` | `int` | Ghost task machine demand |
| `tasks[].is_pinned` | `bool` | Lock task to machine/time |

### `CompletedJobResultPayload` → Go (output)

```python
{
  "job_id": str,
  "task_id": str,           # Celery task ID
  "status": "feasible" | "infeasible",
  "assignments": [
    {
      "task_id": str,
      "machine_id": str,
      "start_time": int,    # Virtual Minutes
      "end_time": int,
      "group_id": str,
      "order_id": str,
      "quantity": float,
      "status": "ON_TIME" | "LATE"
    }
  ],
  "overloads": [
    {
      "task_id": str,
      "order_id": str,
      "status": "LATE",
      "delay_minutes": int,
      "root_cause_code": "MACHINE_OVERLOAD" | "WORKFORCE_SHORTAGE"
                       | "PINNED_TASK_CONFLICT" | "CAPACITY_FULL",
      "bottleneck_resource_id": str,   # specific machine_id
      "quantity": float
    }
  ]
}
```

---

## Testing Strategy

### Priority 1: Determinism Regression Suite

The most critical test class — validates that `random_seed` + `num_search_workers=1` → identical output.

```python
# tests/test_determinism.py
import copy, json
from app.engine.model import Engine

FIXTURE_PAYLOAD = json.load(open("tests/fixtures/payload_200_tasks.json"))

def test_deterministic_output():
    """Same payload → byte-identical assignments across 5 runs."""
    results = []
    for _ in range(5):
        payload = copy.deepcopy(FIXTURE_PAYLOAD)
        payload["config"]["num_search_workers"] = 1
        payload["config"]["random_seed"] = 42
        engine = Engine(payload)
        results.append(engine.solve())

    baseline = results[0]["assignments"]
    for run in results[1:]:
        assert run["assignments"] == baseline, "Non-deterministic output detected"
```

### Priority 2: Root-Cause Classification Tests

```python
# tests/test_root_cause.py
def test_pinned_conflict_detected(pinned_conflict_payload):
    result = Engine(pinned_conflict_payload).solve()
    late = [o for o in result["overloads"] if o["status"] == "LATE"]
    assert any(o["root_cause_code"] == "PINNED_TASK_CONFLICT" for o in late)

def test_machine_overload_detected(overloaded_machine_payload):
    result = Engine(overloaded_machine_payload).solve()
    late = [o for o in result["overloads"] if o["status"] == "LATE"]
    assert any(o["root_cause_code"] == "MACHINE_OVERLOAD" for o in late)

def test_workforce_shortage_detected(capacity_full_payload):
    result = Engine(capacity_full_payload).solve()
    late = [o for o in result["overloads"] if o["status"] == "LATE"]
    assert any(o["root_cause_code"] == "WORKFORCE_SHORTAGE" for o in late)
```

### Priority 3: Hard Constraint Verification

```python
def test_no_overlap_on_machine(result, payload):
    """No two tasks on the same machine have overlapping [start, end)."""
    from collections import defaultdict
    machine_slots = defaultdict(list)
    for a in result["assignments"]:
        machine_slots[a["machine_id"]].append((a["start_time"], a["end_time"]))
    for m_id, slots in machine_slots.items():
        slots.sort()
        for i in range(len(slots) - 1):
            assert slots[i][1] <= slots[i+1][0], \
                f"Overlap on {m_id}: {slots[i]} vs {slots[i+1]}"

def test_pinned_tasks_immovable(result, payload):
    """Pinned tasks land exactly on their declared machine/time."""
    pinned = {t["task_id"]: t for t in payload["tasks"] if t.get("is_pinned")}
    for a in result["assignments"]:
        if a["task_id"] in pinned:
            t = pinned[a["task_id"]]
            assert a["machine_id"] == t["pinned_machine_id"]
            assert a["start_time"] == t["pinned_start_time"]
            assert a["end_time"] == t["pinned_end_time"]
```

### Test Fixtures

```
tests/
├── fixtures/
│   ├── payload_10_tasks.json        # Smoke test
│   ├── payload_200_tasks.json       # Determinism baseline
│   ├── payload_pinned_conflict.json # PINNED_TASK_CONFLICT trigger
│   ├── payload_overloaded.json      # MACHINE_OVERLOAD trigger
│   └── payload_capacity_full.json   # WORKFORCE_SHORTAGE trigger
└── test_*.py
```

### Coverage Targets

| Test Class | Target Coverage | Method |
|------------|----------------|--------|
| `builder.py` | 80% line coverage | pytest + coverage.py |
| `model.py` | 90% | pytest |
| `extract_results` | 100% on root-cause branches | Parametrized fixtures |
| End-to-end (FastAPI → Celery → webhook) | 3 happy-path scenarios | `requests` + mock webhook |

---

## Performance Benchmarking

### Benchmark Harness

```python
# scripts/benchmark.py
import time, json, statistics
from app.engine.model import Engine

SCALES = [200, 500, 1000]

for n_tasks in SCALES:
    payload = generate_synthetic_payload(n_tasks)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        Engine(payload).solve()
        times.append(time.perf_counter() - t0)
    print(f"n={n_tasks}: mean={statistics.mean(times):.1f}s  "
          f"p95={sorted(times)[-1]:.1f}s")
```

### Target Benchmarks

| Tasks | Solve Time Target | RAM Target | Optimality Gap |
|-------|------------------|------------|----------------|
| 200 | ≤ 60s | ≤ 500MB | ≤ 1% |
| 500 | ≤ 180s | ≤ 1.5GB | ≤ 1% |
| 1000 | ≤ max_search_time | ≤ 6GB | best-effort |

---

## Deployment

### Container Configuration

```dockerfile
# Dockerfile (worker)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["celery", "-A", "app.core.celery_app", "worker",
     "--loglevel=info", "--concurrency=1"]
```

```dockerfile
# Dockerfile (api)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8083"]
```

### Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker |
| `WEBHOOK_URL` | `http://backend:8082/api/webhook/solver` | Go callback |
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API port (override to 8083 in compose) |

### Docker Compose (reference)

```yaml
services:
  api:
    build: .
    ports: ["8083:8083"]
    environment:
      - REDIS_URL=redis://redis:6379/0
      - WEBHOOK_URL=http://backend:8082/api/webhook/solver
      - PORT=8083
    depends_on: [redis]

  worker:
    build: .
    command: celery -A app.core.celery_app worker --loglevel=info --concurrency=1
    environment:
      - REDIS_URL=redis://redis:6379/0
      - WEBHOOK_URL=http://backend:8082/api/webhook/solver
    deploy:
      replicas: 2          # Each worker handles one solve at a time
    depends_on: [redis]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

**Worker concurrency = 1** per container. CP-SAT already uses `num_search_workers` threads internally — running multiple Celery tasks concurrently on the same container causes RAM contention. Scale by adding containers, not `--concurrency`.

---

## Monitoring & Observability

### Structured Log Fields (add to `solver_task.py`)

```python
logger.info({
    "event": "solve_complete",
    "job_id": payload.get("job_id"),
    "celery_task_id": self.request.id,
    "status": result["status"],
    "n_assignments": len(clean_assignments),
    "n_overloads": len(clean_overloads),
    "root_cause_breakdown": {
        code: sum(1 for o in clean_overloads if o["root_cause_code"] == code)
        for code in ["MACHINE_OVERLOAD", "WORKFORCE_SHORTAGE",
                     "PINNED_TASK_CONFLICT", "CAPACITY_FULL"]
    },
    "objective_value": result.get("objective_value"),
    "solve_time_seconds": result.get("solve_time"),
})
```

### Health Endpoint (`health.py`)

Should return worker queue depth from Redis — allows Go backend to backpressure if queue exceeds threshold:

```python
@router.get("/health")
async def health():
    from app.core.celery_app import celery_app
    inspector = celery_app.control.inspect()
    active = inspector.active() or {}
    return {
        "status": "ok",
        "active_tasks": sum(len(v) for v in active.values()),
    }
```

---

## Implementation Order

| Step | Task | File | P |
|------|------|------|---|
| 1 | Add `random_seed` + `num_search_workers` to `SolverConfig` | `request_schema.py` | P1 |
| 2 | Wire both params in `Engine.solve()` | `model.py` | P1 |
| 3 | Write determinism regression test + 200-task fixture | `tests/` | P1 |
| 4 | Implement `_classify_root_cause()` | `builder.py` | P1 |
| 5 | Call `_classify_root_cause()` in `extract_results()` | `builder.py` | P1 |
| 6 | Write root-cause classification tests + fixtures | `tests/` | P1 |
| 7 | Add ghost-task count guard + warning | `builder.py` | P1 |
| 8 | Add overload-ratio diagnostic log | `model.py` | P1 |
| 9 | Implement dynamic objective weight calibration | `builder.py` | P1 |
| 10 | Benchmark harness at 200/500/1000 tasks | `scripts/` | P2 |
| 11 | Soft pipeline offset relaxation | `builder.py` | P2 |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| `random_seed` fix breaks existing Go integration tests | Low | Medium | Ship `random_seed` as optional field with existing default behavior |
| `_classify_root_cause` is O(n²) in task count | Medium | Medium | Cap scan at 1000 tasks; cache machine-slot lookups in dict |
| Dynamic weight calibration regresses solution quality | Medium | High | A/B test against static constants on 3 known payloads before rolling out |
| Ghost-task RAM explosion at 1000+ tasks | Medium | High | Guard + warning ships first; Boolean exclusion as P2 fallback |
| Webhook callback timeout under heavy load | Low | High | Add `retry` to `requests.post` with exponential backoff |

---

## Definition of Technical Done

- [ ] `random_seed` wired; determinism test passes 5/5 runs with `num_search_workers=1`
- [ ] `_classify_root_cause()` returns 3+ distinct codes on corresponding fixture payloads
- [ ] `root_cause_code` no longer hardcoded to `"CAPACITY_FULL"` in any test run
- [ ] Ghost-task count warning fires when `capacity_block` count > 200
- [ ] Overload-ratio diagnostic logs when factory load > 85%
- [ ] All hard-constraint verification tests pass (no-overlap, pinned immovability)
- [ ] Docker Compose starts cleanly with `docker compose up`
- [ ] `/health` returns 200 with active task count

---

## Appendix A: CP-SAT Parameter Reference

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| `max_time_in_seconds` | 60–300 (from config) | Hard time-box |
| `relative_gap_limit` | 0.01 | Stop at 1% from optimal |
| `num_search_workers` | 8 (prod), 1 (test) | Controls parallelism and determinism |
| `random_seed` | 42 (configurable) | Fixed for determinism |
| `linearization_level` | default (1) | Higher = tighter LP relaxation, slower |
| `cp_model_presolve` | default (true) | Disable only for debugging |

## Appendix B: Key Penalty Ratios

Target ratio: `LATENESS : AFFINITY : ACTIVATION = 1000 : 10 : 2`

```python
# Intended calibration at n_tasks=200, horizon=40320 (4 weeks):
LATENESS_UNIT   = 10^(6-priority) * lateness_scale   # ~100,000 at p=1
AFFINITY_UNIT   = PENALTY_PER_ROLL_SWAP * affinity_scale  # ~1,000
ACTIVATION_UNIT = PENALTY_ACTIVATE_RESOURCE * activation_scale  # ~200
```

Solver should prefer: on-time delivery >> low yarn changeover >> fewer machines active.

---

*TechDesign Version: 1.0*
*Last Updated: 2026-04-11*
*Next Review: 2026-05-01*
*Stack: Python 3.9+, OR-Tools CP-SAT v9.8+, FastAPI, Celery + Redis*
*Status: Ready for Implementation*
