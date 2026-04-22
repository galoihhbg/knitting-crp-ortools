# Code Patterns

## Architecture Pattern

- **Primary pattern:** Service-oriented — `Engine` (orchestrator) + `TaskModelBuilder` (domain service)
- **Rule:** All CP-SAT model construction goes in `TaskModelBuilder` methods. `Engine.solve()` only parses, chains, and runs.
- **Rule:** Reuse existing builder methods before adding new ones. Add to `build_resource_allocations()` if it's resource-related; add a new method only for a new constraint category.
- **Rule:** Builder methods return `self` — preserve the fluent chain in `Engine.solve()`.

## Builder Method Pattern

Every new builder step follows this structure:

```python
def apply_my_constraint(self) -> "TaskModelBuilder":
    """
    Docstring: what constraint this applies and why.
    Reference to research question if applicable (e.g., "Q2 workforce capacity").
    """
    logger.info("\n🔧 APPLYING MY CONSTRAINT:")
    for t_id, tv in self.task_vars.items():
        # Guard: skip if this task type doesn't apply
        task_info = next((t for t in self.tasks if t["task_id"] == t_id), {})
        if task_info.get("operation", "").lower() != "relevant_operation":
            continue

        # Build the constraint
        self.model.Add(...)
        logger.info(f"   ✅ {t_id}: constraint applied")

    return self  # Always return self for chaining
```

## Penalty Constants Pattern

```python
# Module-level constants with ratio documentation
# Target: LATENESS : AFFINITY : ACTIVATION ≈ 1000 : 10 : 2
PENALTY_PER_ROLL_SWAP: int = 100   # Per yarn roll that must be swapped on a machine
PENALTY_COLD_START: int = 200      # Machine has no thread — full reel-threading needed
PENALTY_ACTIVATE_RESOURCE: int = 50  # Activating a previously idle machine
PENALTY_CHANGE_DESIGN: int = 10    # Load new design file (fast — USB only)
```

When adding a new penalty constant:
1. Add at module level with type annotation `int`
2. Add inline comment explaining what physical action it represents
3. Verify it fits the intended ratio relative to existing constants
4. Add to the objective via `self.objective_terms.append(is_selected * penalty)`

## Conditional CP-SAT Constraints Pattern

```python
# CORRECT: OnlyEnforceIf takes the literal itself (not negation)
self.model.Add(start_var >= available_at).OnlyEnforceIf(is_selected)

# CORRECT: Negated literal for "when NOT selected"
self.model.Add(something).OnlyEnforceIf(is_selected.Not())

# WRONG: Never pass Python bool — CP-SAT requires BoolVar
# self.model.Add(...).OnlyEnforceIf(True)  ← WRONG
```

## task_vars Dictionary Pattern

`self.task_vars[t_id]` is the single source of truth per task during model construction:

```python
{
    "start": IntVar,          # CP-SAT start variable (or NewConstant for pinned)
    "end": IntVar,            # CP-SAT end variable (or NewConstant for pinned)
    "literals": [BoolVar],    # One per compatible resource, in same order as r_ids
    "r_ids": [str],           # Compatible resource IDs (same index as literals)
    "due": int,               # due_at_min in Virtual Minutes
    "original_order_id": str,
    "group_id": str,
    "depends_on": [str],      # final_depends_on list
    "qty": float,
}
```

When reading a literal for a specific resource:
```python
lit = next(
    (l for l in tv["literals"] if l.Name().endswith(f"_on_{r_id}")),
    None
)
```

## Data Fetching / State

- **No database.** All state comes from the `SolverPayload` JSON. Parse it in `Engine.__init__()`.
- **`self.task_vars`** is populated by `build_time_variables()` and mutated by later builder steps.
- **`self.resource_map`** is populated in `__init__` and mutated (intervals added) by `build_resource_allocations()`.
- **`self.objective_terms`** is a list — append to it in any builder step that adds a penalty.

## Error Handling

```python
# At JSON boundaries — always return a dict, never raise
def solve(self) -> Dict[str, Any]:
    try:
        # ... builder chain + solver
        return builder.extract_results(solver, status)
    except Exception as exc:
        logger.error(f"Solver error: {exc}", exc_info=True)
        return {"status": "infeasible", "assignments": [], "overloads": []}

# For missing/invalid inputs — warn and skip, don't crash
if actual_id not in self.task_vars:
    logger.warning(f"⚠️ Task '{t_id}' depends on '{raw_id}' — not found!")
    continue  # Do not raise
```

## Validation Pattern

- Pydantic validates the incoming `SolverPayload` at the FastAPI layer.
- Inside `builder.py`, trust that `task["task_id"]` and `task["duration"]` exist (Pydantic guarantee).
- Guard only against *business logic* edge cases: no compatible resources, missing translation map entries, zero-duration tasks.

## Change Discipline

- One builder method per commit — do not modify multiple constraint methods in one change.
- Do not change `request_schema.py` field aliases without notifying the Go team first.
- Do not upgrade the OR-Tools pin without a benchmark showing no regression on `payload_200_tasks.json`.
- Penalty constant changes require a note in `MEMORY.md` explaining the calibration reasoning.

## Naming for CP-SAT Variables

CP-SAT variable names must be unique within a model. Use these patterns:

| Variable Type | Name Pattern | Example |
|--------------|-------------|---------|
| Task start | `start_{task_id}` | `start_K1-order_001` |
| Task end | `end_{task_id}` | `end_K1-order_001` |
| Lateness | `lat_{task_id}` | `lat_K1-order_001` |
| Assignment literal | `{task_id}_on_{r_id}` | `K1-order_001_on_M01` |
| Optional interval | `int_{task_id}_{r_id}` | `int_K1-order_001_M01` |
| Unavailability | `unavail_{r_id}` | `unavail_M01` (may need index for multiple windows) |
| PO bounding start | `po_{po_id}_{r_id}_start` | `po_order_001_M01_start` |
| Resource activated | `activated_{r_id}` | `activated_M01` |
| Global interval | `global_interval_{t_id}` | `global_interval_K1-order_001` |
| Batch assignment | `x_{task_id}_{k}` | `x_W1-order_001_2` |
| Batch slot start | `batch_start_{k}` | `batch_start_0` |
| Batch slot active | `batch_active_{k}` | `batch_active_0` |
| Group uses slot | `group_{color}_{substance}_{k}` | `group_red_cotton_0` |

## Smart Batching Pattern

Used for washing tasks: group tasks by `(color, substance)` compatibility and assign to shared batch slots.

### K Calculation (Qty-Based)
```python
# CORRECT: K based on total product quantity, not task count
total_qty = sum(task["qty"] for task in washing_tasks)
min_batches = math.ceil(total_qty / capacity)  # capacity in product units
K = min(n_tasks, max(min_batches * 3, 5))

# WRONG: K = math.ceil(n_tasks / capacity)  ← mixes units (tasks vs qty)
```

### BoolVar Assignment Matrix
```python
# x[t_id][k] = True iff task t_id is in batch slot k
x: Dict[str, List[BoolVar]] = {}
for t_id in washing_task_ids:
    x[t_id] = [model.NewBoolVar(f"x_{t_id}_{k}") for k in range(K)]
    model.AddExactlyOne(x[t_id])  # Every task → exactly one slot

# Capacity constraint (in product qty, not task count)
for k in range(K):
    model.Add(sum(x[t_id][k] * task_qtys[t_id] for t_id in washing_task_ids) <= capacity)
```

### Grouped Compatibility — O(nK) Not O(n²K)
```python
# CORRECT: group_uses_slot BoolVar per group per slot
task_groups: Dict[Tuple[str, str], List[str]] = {}
for t_id in washing_task_ids:
    task = task_by_id[t_id]
    key = (task.get("color", ""), task.get("substance", ""))
    task_groups.setdefault(key, []).append(t_id)

for k in range(K):
    group_uses_slot = []
    for group_key, group_task_ids in task_groups.items():
        uses = model.NewBoolVar(f"group_{group_key}_{k}")
        model.AddMaxEquality(uses, [x[t_id][k] for t_id in group_task_ids])
        group_uses_slot.append(uses)
    model.Add(sum(group_uses_slot) <= 1)  # Only one color/substance config per slot

# WRONG: per-pair binary constraint for every (i,j) pair — O(n²K) constraints
# for i in range(n):
#     for j in range(i+1, n):  ← 12,250 constraints for 50 tasks → infeasibility risk
```

### Start-Time Synchronization
```python
# CORRECT: == (exact sync)
for t_id in washing_task_ids:
    for k in range(K):
        model.Add(task_vars[t_id]["start"] == batch_starts[k]).OnlyEnforceIf(x[t_id][k])

# WRONG: >= (allows WA=60, WB=0 in same slot)
# model.Add(task_vars[t_id]["start"] >= batch_starts[k]).OnlyEnforceIf(x[t_id][k])
```

### Symmetry Breaking (Clustering Without Time Ordering)
```python
# CORRECT: clustering prevents K! permutations of empty slots
for k in range(K - 1):
    model.AddImplication(batch_active[k + 1], batch_active[k])

# Lock floating variables — prevents AI searching millions of useless values
for k in range(K):
    model.Add(batch_starts[k] == 0).OnlyEnforceIf(batch_active[k].Not())

# WRONG: time ordering — kills parallel machine scheduling
# model.Add(batch_starts[k] <= batch_starts[k + 1])  ← prevents concurrent washing
```

### Batch Machine Routing
When washing resources have `capacity > 1` (sent by Go), use `AddCumulative` not `AddNoOverlap`:
```python
# In apply_routing_constraints():
if cap > 1:
    demands = [1] * len(intervals)
    model.AddCumulative(intervals, demands, cap)  # allows concurrent tasks
else:
    model.AddNoOverlap(intervals)  # serial: one task at a time

# Go must send washing resources with capacity > 1 for machine sync to work:
# {"id": "W_WASHING_01", "type": "batch", "capacity": 200, "operation": "washing"}
```

### Batch Minimize Objective
```python
# Incentive to merge batches when deadlines allow
# Weight ratio: batch_penalty << lateness_penalty (e.g. 50 vs 50,000)
_batch_w: int = 50 * lateness_scale
objective_terms.append(batch_active[k] * _batch_w)
# Also add WIP bias (early-start penalty for idle batch starts)
objective_terms.append(batch_starts[k] * 1)
```

### Compatibility Fields
- `color` + `substance` on `SolverTask` determine washing compatibility (NOT `color_config`)
- `color_config` is for knitting machine thread/design — unrelated to washing batching
- Tasks with identical `(color, substance)` tuple can share a batch slot
