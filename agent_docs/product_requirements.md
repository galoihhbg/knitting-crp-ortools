# Product Requirements

> Source: `docs/PRD-Knitting-CRP-Ortools-MVP.md`

## Primary User Story

"As the Go backend, I want to POST a SolverPayload and receive a CompletedJobResultPayload via webhook, so that the planning UI can display a conflict-free production schedule."

**Acceptance Criteria:**
- Valid payload → feasible result within `max_search_time` seconds
- Result includes `assignments[]` with `task_id`, `machine_id`, `start_time`, `end_time`, `status`
- Result includes `overloads[]` with `root_cause_code` and `bottleneck_resource_id`
- Deterministic: same payload with same `random_seed` → byte-identical assignments

## Must-Have Features (P0)

| # | Feature | Status | Key AC |
|---|---------|--------|--------|
| 1 | `build_time_variables()` — pinned constants, lateness penalty | Complete | Pinned = NewConstant; `lateness` = `10^(6-priority)` scale |
| 2 | `build_resource_allocations()` — affinity + contiguous PO | Complete | Per-roll yarn penalty; bounding-box per PO per machine |
| 3 | `apply_routing_constraints()` — NoOverlap + unavailability | Complete | No task in declared unavailability window |
| 4 | `apply_dependency_constraints()` — DAG + inferred K→L | Complete | Unresolvable deps → warning, not crash |
| 5 | `apply_batch_offset_constraints()` — WaitOffsets pipeline | Complete | Multiple batch deps per task supported |
| 6 | `build_workforce_constraints()` — AddCumulative factory cap | Complete | capacity_block demand reduces headroom |
| 7 | Objective + Solver execution — random_seed, 1% gap limit | **GAP: random_seed not wired** | `random_seed` + `num_search_workers` from config |
| 8 | `extract_results()` — assignments + overload diagnostics | **GAP: root_cause hardcoded** | 3+ distinct `root_cause_code` values |

## Should Have (P1)

- `random_seed` wired from `SolverConfig` (closes determinism gap)
- Dynamic objective weight calibration (prevents stagnation at scale)
- Structured root-cause classifier (`MACHINE_OVERLOAD`, `WORKFORCE_SHORTAGE`, `PINNED_TASK_CONFLICT`, `CAPACITY_FULL`)
- Ghost-task count guard + warning at > 200 capacity_block tasks
- Overload-ratio diagnostic log at > 85% factory load
- **Smart Batching (COMPLETE v1.1):** Auto-merge washing tasks by `(color, substance)` into shared slots — saves water/soap/energy. Features: qty-based K, grouped O(nK) compatibility, exact start-time sync, symmetry breaking, batch machine routing via `AddCumulative`. Requires Go to send washing resources with `capacity > 1` for machine sync.
- **Shift Boundary Constraints (COMPLETE):** Washing tasks must not straddle virtual-time shift boundaries. Disjunctive constraint `end <= S OR start >= S` per task per boundary. Requires Go to send `shift_ends_min` in `SolverConfig`.

## Out of Scope (Won't Have for MVP)

- Alternative solvers (Gurobi, CPLEX)
- Real-time MES/IoT integration
- UI/UX rendering (owned by Go frontend)
- Database access from Python (stateless workers by design)
- Inventory / BOM management

## Success Metrics

| Category | Metric | Target |
|----------|--------|--------|
| Reliability | Feasible solve rate | > 95% of valid payloads |
| Performance | P95 solve time (200 tasks) | < 60s |
| Correctness | Hard-constraint violations | 0 |
| Determinism | Identical output on replay | 100% (with `num_search_workers=1`) |
| Diagnostics | Specific `root_cause_code` | > 80% specificity (not always CAPACITY_FULL) |

## Root Cause Codes

The `overloads[].root_cause_code` field must be one of:

| Code | Meaning | Trigger |
|------|---------|---------|
| `MACHINE_OVERLOAD` | Specific machine saturated | Other tasks on same machine in same window |
| `WORKFORCE_SHORTAGE` | Global factory cap hit | `concurrent_knitting + capacity_block_demand >= max_factory_machines` |
| `PINNED_TASK_CONFLICT` | Pinned task displaced this task | Another `is_pinned` task on same machine in same window |
| `CAPACITY_FULL` | Fallback / unclassified | None of the above triggered |

## Non-Functional Requirements

- **Solve Time:** ≤ `max_search_time` (config default 300s); target 60s at ≤ 200 tasks
- **RAM:** ≤ 8 GB per Celery worker
- **Optimality Gap:** ≤ 1% (`relative_gap_limit = 0.01`)
- **Determinism:** `random_seed` + `num_search_workers=1` → byte-identical output
- **No HTTP 500s:** All solver failures return `{"status": "infeasible", ...}` with HTTP 200
- **No Secrets in Payload:** `SolverPayload` contains no PII or credentials
