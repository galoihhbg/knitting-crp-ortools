"""
Phase 3: Washing Batching — Group-Isolated CP-SAT solver.

Instead of one giant model for all washing tasks, tasks are grouped by
(color, substance) compatibility and each group gets its own small CP-SAT model.
This prevents combinatorial explosion: a group with 3 tasks × 5 slots = 15 BoolVars
vs. the monolithic approach where N tasks share a global K-slot matrix.

Within each group model:
  - Batch slot start IntVars  batch_start[k]
  - Assignment BoolVars       x[t][k]  (1 = task t goes to slot k)
  - Capacity per slot         sum(x[t][k] * qty[t]) <= batch_capacity
  - Synchronization           task_start == batch_start[k]  when x[t][k]=1
  - start_lb from Phase 2     batch_start[k] >= linking_end  when x[t][k]=1
  - Shift boundary            end <= S OR start >= S  per boundary point S
  - Machine assignment        build_resource_model (NoOverlap / AddCumulative)
  - Minimize active batches + lateness

Output: BatchInfo list (one entry per active slot across all groups) plus
end_times dict for Phase 4 downstream lower-bound calculation.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from app.engine.shared import (
    apply_soft_deadlines,
    build_resource_model,
    compute_horizon,
    extract_results,
    make_solver,
)

logger = logging.getLogger(__name__)

PHASE3_OPS = frozenset({"washing"})


@dataclass
class BatchInfo:
    batch_id: str
    task_ids: List[str]
    start_time: int
    end_time: int  # max task end time within this batch


@dataclass
class Phase3Result:
    status: str
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    batches: List[BatchInfo] = field(default_factory=list)
    end_times: Dict[str, int] = field(default_factory=dict)
    solve_time_seconds: float = 0.0


def solve_washing(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    p2_end_times: Dict[str, int],
    shift_ends: List[int],
    horizon: Optional[int] = None,
) -> Phase3Result:
    """
    Solve the washing phase using per-group model isolation.

    Args:
        tasks:        Washing tasks (operation == 'washing').
        resources:    Washing machines.
        config:       Solver config.
        p2_end_times: task_id → end minute from Phase 2.
        shift_ends:   Shift boundary points (virtual minutes).
    """
    washing_tasks = [t for t in tasks if t.get("operation", "").lower() in PHASE3_OPS]
    if not washing_tasks:
        logger.info("⚙️ Phase 3 (Washing): no tasks — skipped.")
        return Phase3Result(status="empty")

    if horizon is None:
        horizon = compute_horizon(washing_tasks, config)
    capacity = max(1, int(config.get("washing_batch_capacity", 10)))

    # Compute start lower-bounds from Phase 2
    start_lb = _compute_start_lb(washing_tasks, p2_end_times)

    # Group by (color, substance)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for t in washing_tasks:
        key = (t.get("color", ""), t.get("substance", ""))
        groups.setdefault(key, []).append(t)

    logger.info(f"⚙️ Phase 3 (Washing): {len(washing_tasks)} tasks → {len(groups)} groups")

    all_assignments: List[Dict[str, Any]] = []
    all_overloads: List[Dict[str, Any]] = []
    all_batches: List[BatchInfo] = []
    all_end_times: Dict[str, int] = {}
    total_solve_time = 0.0

    for group_key, group_tasks in groups.items():
        result = _solve_group(
            group_key=group_key,
            group_tasks=group_tasks,
            resources=resources,
            config=config,
            horizon=horizon,
            capacity=capacity,
            start_lb=start_lb,
            shift_ends=shift_ends,
        )
        all_assignments.extend(result["assignments"])
        all_overloads.extend(result["overloads"])
        all_batches.extend(result["batches"])
        all_end_times.update(result["end_times"])
        total_solve_time += result["solve_time"]

    return Phase3Result(
        status="feasible" if all_assignments or not washing_tasks else "infeasible",
        assignments=all_assignments,
        overloads=all_overloads,
        batches=all_batches,
        end_times=all_end_times,
        solve_time_seconds=total_solve_time,
    )


# ---------------------------------------------------------------------------
# Per-group solver
# ---------------------------------------------------------------------------

def _solve_group(
    group_key: Tuple[str, str],
    group_tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    horizon: int,
    capacity: int,
    start_lb: Dict[str, int],
    shift_ends: List[int],
) -> Dict[str, Any]:
    # Split pinned vs free tasks.
    # Pinned tasks have fixed start/end and must NOT consume batch slots or
    # receive shift-boundary constraints (they are already scheduled and immovable).
    # They still pass through build_resource_model so their machine intervals
    # block capacity for free tasks on the same machine.
    # K and all slot constraints are computed from free tasks only, so that
    # passing washing_num_slots sized for new orders never causes infeasibility
    # from pinned tasks consuming slots they don't need.
    pinned_tasks = [t for t in group_tasks if t.get("is_pinned")]
    free_tasks   = [t for t in group_tasks if not t.get("is_pinned")]
    n            = len(free_tasks)
    all_task_ids = [t["task_id"] for t in group_tasks]

    # Compute K from free tasks only
    _raw_qtys   = {t["task_id"]: max(0, int(t.get("qty") or 0)) for t in free_tasks}
    group_qty   = sum(_raw_qtys.values())
    min_batches = max(1, math.ceil(group_qty / capacity)) if group_qty > 0 else 1

    cfg_slots = config.get("washing_num_slots")
    if n == 0:
        K = 0  # all tasks pinned — no slot assignment needed
    elif cfg_slots is not None:
        K = max(1, min(n, int(cfg_slots)))
    else:
        K = max(1, min(n, min_batches + 2))
        max_k = config.get("max_washing_batches")
        if max_k is not None:
            K = min(K, int(max_k))
        K = max(1, K)

    if n * K > 50_000:
        logger.warning(
            f"   ⚠️  Group {group_key}: {n}×{K}={n*K:,} BoolVars — consider reducing K."
        )

    logger.info(
        f"   Group {group_key}: {len(group_tasks)} tasks "
        f"({len(pinned_tasks)} pinned, {n} free), qty={group_qty}, "
        f"K={K}, capacity={capacity}"
    )

    model = cp_model.CpModel()
    resource_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in resources}

    # Override machine capacity -> batch mode (AddCumulative per machine).
    # Pass ALL group tasks so pinned tasks register intervals and block capacity.
    washing_resource_ids = set(
        r_id
        for t in group_tasks
        for r_id in (t.get("compatible_resource_ids") or [])
    )
    batch_resource_map = {
        r_id: ({**r, "capacity": capacity} if r_id in washing_resource_ids else r)
        for r_id, r in resource_map.items()
    }

    group_start_lb = {t_id: start_lb.get(t_id, 0) for t_id in all_task_ids}
    task_vars, _, no_res = build_resource_model(
        model, group_tasks, batch_resource_map, horizon,
        start_lb=group_start_lb,
    )
    if no_res:
        logger.warning(f"   ⚠️ Group {group_key}: {len(no_res)} tasks unschedulable — no resources.")

    if not task_vars:
        return {"assignments": [], "overloads": [], "batches": [], "end_times": {}, "solve_time": 0.0}

    # Separate scheduled IDs by pinned vs free (dedup may have renamed some)
    free_scheduled_ids   = [t_id for t_id in task_vars if not task_vars[t_id].get("is_pinned")]
    pinned_scheduled_ids = [t_id for t_id in task_vars if     task_vars[t_id].get("is_pinned")]

    # task_qtys only for free tasks — slot capacity constraint is free-task only
    task_qtys: Dict[str, int] = {
        t_id: max(0, int(task_vars[t_id].get("qty") or 0))
        for t_id in free_scheduled_ids
    }
    actual_free_qty = sum(task_qtys.values())
    for t_id, q in task_qtys.items():
        if q > capacity:
            logger.warning(
                f"   ⚠️ Task {t_id}: qty={q} > batch_capacity={capacity} "
                f"→ cannot fit alone → group model may be INFEASIBLE"
            )

    # Early exit: nothing to slot-assign (all tasks pinned)
    if K == 0 or not free_scheduled_ids:
        logger.info(f"   ⏭️ Group {group_key}: all tasks pinned — skipping slot assignment")
        solver = make_solver(config)
        per_group_time = max(5, int(config.get("max_search_time", 60)) // max(1, 4))
        solver.parameters.max_time_in_seconds = per_group_time
        status_code = solver.Solve(model)
        status_str, assignments, overloads, _, end_times = extract_results(
            solver, status_code, task_vars, group_tasks, wash_x={}
        )
        if status_str != "feasible":
            end_times = _fallback_end_times(group_tasks, group_start_lb)
            assignments = []
        return {
            "assignments": assignments,
            "overloads": overloads,
            "batches": [],
            "end_times": end_times,
            "solve_time": solver.WallTime(),
        }

    # ── Batch slot variables ──────────────────────────────────────────────────────────────────────────
    batch_starts = [
        model.NewIntVar(0, horizon, f"wbatch_{group_key}_{k}_start")
        for k in range(K)
    ]
    batch_active = [
        model.NewBoolVar(f"wbatch_{group_key}_{k}_active")
        for k in range(K)
    ]

    # Assignment BoolVars for FREE tasks only
    x: Dict[str, List] = {
        t_id: [model.NewBoolVar(f"wx_{t_id}_{k}") for k in range(K)]
        for t_id in free_scheduled_ids
    }

    for t_id in free_scheduled_ids:
        model.AddExactlyOne(x[t_id])

    # Capacity per slot (free tasks only)
    for k in range(K):
        model.Add(
            sum(x[t_id][k] * task_qtys[t_id] for t_id in free_scheduled_ids) <= capacity
        )

    # Synchronization: task_start == batch_start when assigned
    for t_id in free_scheduled_ids:
        tv = task_vars[t_id]
        for k in range(K):
            model.Add(tv["start"] == batch_starts[k]).OnlyEnforceIf(x[t_id][k])
            lb = group_start_lb.get(t_id, 0)
            if lb > 0:
                model.Add(batch_starts[k] >= lb).OnlyEnforceIf(x[t_id][k])

    # Batch activation
    for k in range(K):
        model.AddMaxEquality(batch_active[k], [x[t_id][k] for t_id in free_scheduled_ids])

    # Symmetry breaking: active slots cluster at beginning
    for k in range(K - 1):
        model.AddImplication(batch_active[k + 1], batch_active[k])

    # Freeze inactive slot starts to 0
    for k in range(K):
        model.Add(batch_starts[k] == 0).OnlyEnforceIf(batch_active[k].Not())

    # Shift boundary constraints (FREE tasks only — pinned times are immutable)
    if shift_ends:
        for t_id in free_scheduled_ids:
            tv = task_vars[t_id]
            for S in shift_ends:
                b = model.NewBoolVar(f"before_shift_{t_id}_{S}")
                model.Add(tv["end"] <= S).OnlyEnforceIf(b)
                model.Add(tv["start"] >= S).OnlyEnforceIf(b.Not())

    # Co-location: if free-task qty fits in one slot, force all to same slot.
    # Guard: need at least as many machines as free tasks for concurrent execution.
    if actual_free_qty <= capacity and len(free_scheduled_ids) > 1:
        group_machine_ids: set = set()
        for t_id in free_scheduled_ids:
            group_machine_ids.update(task_vars[t_id].get("r_ids") or [])
        n_machines = len(group_machine_ids)

        if n_machines >= len(free_scheduled_ids):
            t0 = free_scheduled_ids[0]
            for t_other in free_scheduled_ids[1:]:
                for k in range(K):
                    model.Add(x[t0][k] == x[t_other][k])
            logger.info(
                f"   🔒 Group {group_key}: co-located "
                f"(free_qty={actual_free_qty}≤cap={capacity}, "
                f"{n_machines} machines ≥ {len(free_scheduled_ids)} tasks)"
            )
        else:
            logger.info(
                f"   ⚡ Group {group_key}: NOT co-located "
                f"(only {n_machines} machines < {len(free_scheduled_ids)} tasks — will serialize)"
            )

    # ── Objective ──────────────────────────────────────────────────────────────────────────────────
    lateness_scale = min(max(1, horizon // 1000), 50)
    obj_terms: List = []

    batch_w = 50 * lateness_scale
    for k in range(K):
        obj_terms.append(batch_active[k] * batch_w)

    for k in range(K):
        obj_terms.append(batch_starts[k] * 1)

    task_map = {t["task_id"]: t for t in group_tasks}
    obj_terms += apply_soft_deadlines(model, task_vars, task_map, horizon)

    model.Minimize(sum(obj_terms) if obj_terms else 0)

    # ── Solve ────────────────────────────────────────────────────────────────────────────────────────
    solver = make_solver(config)
    per_group_time = max(5, int(config.get("max_search_time", 60)) // max(1, 4))
    solver.parameters.max_time_in_seconds = per_group_time

    status_code = solver.Solve(model)
    logger.info(
        f"   Group {group_key}: status={solver.StatusName(status_code)}, "
        f"time={solver.WallTime():.1f}s"
    )

    status_str, assignments, overloads, start_times, end_times = extract_results(
        solver, status_code, task_vars, group_tasks, wash_x=x
    )

    # Fallback: if solver failed, provide conservative end_times so Phase 4
    # downstream tasks don't incorrectly start at T=0.
    if status_str != "feasible":
        end_times = _fallback_end_times(group_tasks, group_start_lb)
        logger.warning(
            f"   ⚠️ Group {group_key}: solver {status_str} — "
            f"using fallback end_times (start_lb + duration) for {len(end_times)} tasks"
        )

    # ── Extract batches ────────────────────────────────────────────────────────────────────────────
    batches: List[BatchInfo] = []
    if status_str == "feasible":
        slot_tasks: Dict[int, List[str]] = {}
        for t_id in free_scheduled_ids:
            for k in range(K):
                if solver.Value(x[t_id][k]) == 1:
                    slot_tasks.setdefault(k, []).append(t_id)
                    break

        for k, slot_task_ids in slot_tasks.items():
            b_start = solver.Value(batch_starts[k])
            b_end = max(
                (end_times.get(t_id, b_start) for t_id in slot_task_ids),
                default=b_start,
            )
            batches.append(BatchInfo(
                batch_id=f"wash_batch_{'_'.join(str(g) for g in group_key)}_{k}",
                task_ids=slot_task_ids,
                start_time=b_start,
                end_time=b_end,
            ))

    return {
        "assignments": assignments,
        "overloads": overloads,
        "batches": batches,
        "end_times": end_times,
        "solve_time": solver.WallTime(),
    }



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_end_times(
    tasks: List[Dict[str, Any]],
    start_lb: Dict[str, int],
) -> Dict[str, int]:
    """
    Conservative end_times when the solver is infeasible or times out.
    Computes end = max(start_lb, start_after_min) + duration for each task
    so Phase 4 downstream tasks don't start at T=0.
    Pinned tasks use their pinned_end_time directly.
    """
    result: Dict[str, int] = {}
    for t in tasks:
        t_id = t["task_id"]
        if t.get("is_pinned") and t.get("pinned_end_time") is not None:
            result[t_id] = int(t["pinned_end_time"])
        else:
            lb = max(start_lb.get(t_id, 0), int(t.get("start_after_min", 0)))
            duration = max(0, int(t.get("duration", 0)))
            result[t_id] = lb + duration
    return result


def _compute_start_lb(
    tasks: List[Dict[str, Any]],
    p2_end_times: Dict[str, int],
) -> Dict[str, int]:
    """
    Set start_lb[task_id] = max Phase 2 end time across its final_depends_on.
    """
    lb: Dict[str, int] = {}
    for t in tasks:
        t_id = t["task_id"]
        current_lb = 0
        for dep_id in (t.get("final_depends_on") or []):
            if dep_id in p2_end_times:
                current_lb = max(current_lb, p2_end_times[dep_id])
        if current_lb > 0:
            lb[t_id] = current_lb
    return lb
