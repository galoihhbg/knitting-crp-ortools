# System Memory & Context 🧠
<!--
AGENTS: Update this file after every major milestone, structural change, or resolved bug.
DO NOT delete historical context if it is still relevant. Compress older completed items.
-->

## 🏗️ Active Phase & Goal

**Current Phase:** Phase 1 — Close P1 Gaps
**Current Phase:** Phase 3 — COMPLETE ✅
**All MVP phases done.** Codebase is production-ready.

## ✅ Phase 2 Completed Steps
1. ~~Benchmark harness `scripts/benchmark.py` at 200/500/1000 tasks~~ ✅ (200-task: 0.3s, 69MB)
2. ~~Soft pipeline offset relaxation in `apply_batch_offset_constraints()` (auto at load > 85%)~~ ✅
3. ~~Boolean exclusion workforce mode `use_boolean_exclusion` + slot-based NoOverlap~~ ✅ (auto at ghost_count > 200)

## ✅ Phase 1 Completed Steps
1. ~~Add `random_seed` + `num_search_workers` to `SolverConfig` and wire in `Engine.solve()`~~ ✅
2. ~~Write `tests/test_determinism.py` (5-run replay, smoke + 200-task, seed/worker wiring)~~ ✅
3. ~~Implement `_classify_root_cause()` + `tests/test_root_cause.py` (4 tests, all passing)~~ ✅
4. ~~Ghost-task count guard in `build_workforce_constraints()` (warn at > 200 capacity_blocks)~~ ✅
5. ~~Overload-ratio diagnostic in `model.py` (warn when factory load > 85%)~~ ✅
6. ~~Dynamic objective weight calibration: `LATENESS:AFFINITY:ACTIVATION ≈ 1000:10:2`, `lateness_scale = max(1, horizon // 1000)`~~ ✅

## ✅ Phase 4 Completed Steps

1. ~~Dynamic material cumulative constraints (`build_material_constraints()` in `builder.py`)~~ ✅
   - `SolverPayload.material_capacities: Dict[str, int]` — top-level creel capacity map
   - `SolverTask.material_demands: Dict[str, int]` — per-task yarn-roll consumption
   - `AddCumulative` (Profile Sweep) per material — O(n log n), no BoolVar explosion
   - 6 unit tests in `tests/test_material_constraints.py`

## 📂 Architectural Decisions

- **2026-04-15** — Dynamic material constraints: `build_material_constraints()` uses `AddCumulative` (not BoolVar pairs) per material. Interval anchored to task [start, end] so material is auto-released on task completion. Constraint is a no-op when `material_capacities` is absent — backward compatible.
- **2026-04-11** — Chose `AddCumulative` + ghost tasks (Option A) for workforce capacity. Keep for MVP. Add ghost-task count guard at 200. Switch to boolean exclusion only if RAM > 4GB at 500+ tasks.
- **2026-04-11** — `wait_offsets` constraints remain hard for MVP. Add `overload_ratio` warning log when factory load > 85%. Soft relaxation is Phase 2.
- **2026-04-11** — `root_cause_code` classification lives in Python (`builder.py`), not Go. Go only sees the final JSON — it lacks access to solver variable states and pinned task metadata.
- **2026-04-11** — Dynamic objective weight calibration: `LATENESS : AFFINITY : ACTIVATION ≈ 1000 : 10 : 2`. `lateness_scale = max(1, horizon // 1000)`. Ship in Phase 1 Step 7.
- **2026-04-11** — Determinism strategy: `random_seed=42` default + `num_search_workers=1` for regression tests. Production uses `num_search_workers=8` for speed; replay tests must override to 1.

## 🐛 Known Issues & Quirks

- `_classify_root_cause()` scan is O(n²) in task count — cap scan at 1000 tasks; acceptable for MVP scale.
- `end_var <= due_at` was a hard constraint that made unavoidably-late tasks infeasible — removed in Step 3. Lateness is now purely soft (penalty only).
- Duplicate `task_id` detection renames to `_dupN` suffix — this is intentional defensive behavior, not a bug.
- Vietnamese comments in `builder.py` — do not remove or translate; they explain business-domain setup logic.

## 📜 Completed Phases

- [x] Initial codebase scaffold (FastAPI + Celery + OR-Tools)
- [x] `build_time_variables()` — pinned tasks, lateness penalty
- [x] `build_resource_allocations()` — affinity penalties, contiguous PO bounding box
- [x] `build_workforce_constraints()` — AddCumulative + ghost tasks
- [x] `apply_routing_constraints()` — NoOverlap + unavailability windows
- [x] `apply_dependency_constraints()` — explicit + inferred K→L
- [x] `apply_batch_offset_constraints()` — WaitOffsets pipeline
- [x] `define_objective()` — weighted lateness + affinity (static constants)
- [x] `extract_results()` — assignments + overloads (root_cause hardcoded)
- [x] PRD written → `docs/PRD-Knitting-CRP-Ortools-MVP.md`
- [x] TechDesign written → `docs/TechDesign-Knitting-CRP-Ortools-MVP.md`
- [x] Phase 1: Wire random_seed, root-cause classifier, ghost-task guard, overload diagnostic, dynamic weights
- [x] Phase 2: Benchmark harness, soft pipeline offsets, boolean exclusion workforce mode
- [x] Phase 3: Webhook retry + structured logging, enhanced /health, E2E test suite (22 tests)
