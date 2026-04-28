"""
Phase 1: Knitting CP-SAT solver.

Handles:
  - KNITTING tasks — machine allocation + affinity penalties
  - CAPACITY_BLOCK tasks — workforce capacity via AddCumulative/gap-filler
  - PO bounding-box co-location (tasks from same PO contiguous on same machine)

Output: start_times + end_times for all knitting tasks, used by Phase 2 as
lower bounds via final_depends_on and wait_offsets resolution.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ortools.sat.python import cp_model

from app.engine.shared import (
    apply_soft_deadlines,
    build_resource_model,
    compute_horizon,
    extract_results,
    make_solver,
)

logger = logging.getLogger(__name__)

PHASE1_OPS = frozenset({"knitting", "capacity_block"})


def _log_task_diagnostics(tasks: List[Dict[str, Any]], horizon: int) -> None:
    """Log due_at_min, duration, start_after_min anomalies that can cause INFEASIBLE."""
    knitting = [t for t in tasks if t.get("operation", "").lower() == "knitting"]
    if not knitting:
        return

    issues: List[str] = []
    due_vals: List[int] = []

    for t in knitting:
        t_id = t["task_id"]
        due = t.get("due_at_min")
        due_int = int(due) if due is not None else horizon
        due_vals.append(due_int)

        duration = int(t.get("duration", 0))
        start_after = int(t.get("start_after_min", 0))
        pinned_start = t.get("pinned_start_time")
        pinned_end = t.get("pinned_end_time")
        is_pinned = t.get("is_pinned", False)

        if due is None:
            issues.append(f"  ⚠️ {t_id}: due_at_min=None (defaulting to horizon={horizon})")
        elif due_int <= 0:
            issues.append(f"  ⚠️ {t_id}: due_at_min={due_int} ≤ 0 (already overdue at t=0)")
        elif due_int > horizon:
            issues.append(f"  ℹ️ {t_id}: due_at_min={due_int} > horizon={horizon} (no lateness possible)")

        if not is_pinned:
            if duration > horizon:
                issues.append(f"  ❌ {t_id}: duration={duration} > horizon={horizon} → domain empty → INFEASIBLE")
            if start_after > horizon:
                issues.append(f"  ❌ {t_id}: start_after_min={start_after} > horizon={horizon} → domain empty → INFEASIBLE")
        else:
            if pinned_start is not None and int(pinned_start) < 0:
                issues.append(f"  ⚠️ {t_id}: pinned_start_time={pinned_start} < 0 (in-progress task)")
            if pinned_end is not None and pinned_start is not None:
                span = int(pinned_end) - int(pinned_start)
                if span != duration:
                    issues.append(
                        f"  ⚠️ {t_id}: pinned span={span} ≠ duration={duration} "
                        f"(in-progress? pinned=[{pinned_start},{pinned_end}])"
                    )

    min_due = min(due_vals) if due_vals else None
    max_due = max(due_vals) if due_vals else None
    neg_due = sum(1 for d in due_vals if d <= 0)
    over_horizon = sum(1 for d in due_vals if d > horizon)
    logger.info(
        f"📋 Phase 1 task diagnostics ({len(knitting)} knitting): "
        f"due_at_min min={min_due} max={max_due} "
        f"(≤0: {neg_due}, >horizon: {over_horizon})"
    )
    for msg in issues:
        logger.warning(msg) if "⚠️" in msg or "ℹ️" in msg else logger.error(msg)


@dataclass
class Phase1Result:
    status: str
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    start_times: Dict[str, int] = field(default_factory=dict)
    end_times: Dict[str, int] = field(default_factory=dict)
    solve_time_seconds: float = 0.0
    objective_value: Optional[float] = None


def solve_knitting(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    material_capacities: Optional[Dict[str, int]] = None,
    horizon: Optional[int] = None,
) -> Phase1Result:
    """
    Solve the knitting phase in isolation.

    Args:
        tasks:     All tasks for this phase (operation in PHASE1_OPS).
        resources: All resources (task.compatible_resource_ids controls assignment).
        config:    Solver config dict (horizon_minutes, max_search_time, etc.).
        horizon:   Global horizon (minutes). If None, computed from tasks in this phase.

    Returns:
        Phase1Result with start_times/end_times keyed by task_id.
    """
    knitting_tasks = [t for t in tasks if t.get("operation", "").lower() in PHASE1_OPS]
    if not knitting_tasks:
        logger.info("⚙️ Phase 1 (Knitting): no tasks — skipped.")
        return Phase1Result(status="empty")

    if horizon is None:
        horizon = compute_horizon(knitting_tasks, config, resources=resources)
    resource_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in resources}

    _log_task_diagnostics(knitting_tasks, horizon)

    model = cp_model.CpModel()

    task_vars, affinity_terms, no_resource_tasks = build_resource_model(
        model, knitting_tasks, resource_map, horizon, use_affinity=True
    )

    # Any real (non-dummy) task that has no compatible machine → infeasible
    real_no_res = [
        t for t in no_resource_tasks
        if t.get("operation", "").lower() not in ("capacity_block",)
    ]
    if real_no_res:
        ids = [t["task_id"] for t in real_no_res]
        logger.error(f"❌ Phase 1: {len(ids)} task(s) have no resources: {ids}")
        return Phase1Result(
            status="infeasible",
            overloads=[
                {
                    "task_id": t["task_id"],
                    "order_id": t.get("original_order_id", ""),
                    "status": "UNSCHEDULABLE",
                    "delay_minutes": 0,
                    "root_cause_code": "NO_COMPATIBLE_RESOURCE",
                    "bottleneck_resource_id": None,
                    "quantity": t.get("qty", 0),
                }
                for t in real_no_res
            ],
        )

    obj_terms = list(affinity_terms)
    task_map = {t["task_id"]: t for t in knitting_tasks}

    # Workforce constraints — capacity_block tasks constrain concurrent knitting
    _apply_workforce_constraints(model, task_vars, knitting_tasks, resource_map, config, horizon)

    # Per-material creel capacity (AddCumulative per material)
    if material_capacities:
        _apply_material_constraints(model, task_vars, knitting_tasks, material_capacities)

    # PO bounding-box: soft co-location for non-pinned tasks on each machine.
    # Pinned tasks are excluded: their fixed start times may be negative (in-progress)
    # which would force po_start < 0 while po_start ∈ [0, horizon] → instant INFEASIBLE.
    _apply_po_bounding_box(model, task_vars, knitting_tasks, resource_map, horizon)

    obj_terms += apply_soft_deadlines(model, task_vars, task_map, horizon)
    model.Minimize(sum(obj_terms) if obj_terms else 0)

    validation = model.Validate()
    if validation:
        logger.error(f"❌ Phase 1 MODEL_INVALID: {validation}")
        return Phase1Result(status="model_invalid")

    solver = make_solver(config)
    status_code = solver.Solve(model)

    logger.info(
        f"⚙️ Phase 1 (Knitting): {len(task_vars)} task vars, "
        f"status={solver.StatusName(status_code)}, "
        f"time={solver.WallTime():.1f}s"
    )

    status_str, assignments, overloads, start_times, end_times = extract_results(
        solver, status_code, task_vars, knitting_tasks, config=config
    )
    return Phase1Result(
        status=status_str,
        assignments=assignments,
        overloads=overloads,
        start_times=start_times,
        end_times=end_times,
        solve_time_seconds=solver.WallTime(),
        objective_value=solver.ObjectiveValue() if status_str == "feasible" else None,
    )


# ---------------------------------------------------------------------------
# Workforce constraints (AddCumulative with capacity_block + gap-filler)
# ---------------------------------------------------------------------------

def _apply_workforce_constraints(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    resource_map: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    horizon: int,
) -> None:
    MAX_MACHINES = int(config.get("max_factory_machines", 100))

    ghost_count = sum(
        1 for t in tasks if t.get("operation", "").lower() == "capacity_block"
    )
    if ghost_count > 200:
        logger.warning(
            f"⚠️ capacity_block count={ghost_count} > 200 — "
            "consider splitting shift windows."
        )

    use_bool = bool(config.get("use_boolean_exclusion", False)) or ghost_count > 200
    if use_bool:
        _workforce_boolean(model, task_vars, tasks, config, horizon, MAX_MACHINES)
    else:
        _workforce_cumulative(model, task_vars, tasks, config, horizon, MAX_MACHINES)


def _workforce_cumulative(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    horizon: int,
    MAX_MACHINES: int,
) -> None:
    task_map = {t["task_id"]: t for t in tasks}
    knitting_intervals: List = []
    demands: List[int] = []

    for t_id, tv in task_vars.items():
        t_info = task_map.get(t_id, {})
        op = t_info.get("operation", "").lower()

        ps = t_info.get("pinned_start_time")
        pe = t_info.get("pinned_end_time")
        is_fully_pinned = t_info.get("is_pinned") and ps is not None and pe is not None
        duration = int(pe) - int(ps) if is_fully_pinned else int(t_info.get("duration", 0))

        if op == "knitting":
            literals = tv.get("literals", [])
            if literals:
                any_assigned = model.NewBoolVar(f"cumul_active_{t_id}")
                model.AddMaxEquality(any_assigned, literals)
                iv = model.NewOptionalIntervalVar(
                    tv["start"], duration, tv["end"], any_assigned,
                    f"wf_iv_{t_id}",
                )
            else:
                iv = model.NewIntervalVar(tv["start"], duration, tv["end"], f"wf_iv_{t_id}")
            knitting_intervals.append(iv)
            demands.append(1)

        elif op == "capacity_block":
            blk_demand = int(t_info.get("demand", 0))
            if blk_demand <= 0:
                continue  # zero-demand block has no effect — skip

            if is_fully_pinned:
                ps_int, pe_int = int(ps), int(pe)
                duration = pe_int - ps_int
            else:
                duration = int(t_info.get("duration", 0))

            if duration <= 0:
                logger.warning(f"   ⚠️ capacity_block {t_id}: duration={duration} ≤ 0 — skipping")
                continue

            if blk_demand >= MAX_MACHINES:
                logger.error(
                    f"   ❌ capacity_block {t_id}: demand={blk_demand} >= MAX_MACHINES={MAX_MACHINES} "
                    f"→ zero slots for knitting during [{ps}, {pe}] → likely INFEASIBLE"
                )

            iv = model.NewIntervalVar(
                tv["start"], duration, tv["end"], f"wf_iv_{t_id}"
            )
            knitting_intervals.append(iv)
            demands.append(blk_demand)

    # NOTE: No gap-filler intervals are added between or after capacity_block windows.
    # capacity_block tasks already encode the exact workforce-unavailability windows
    # sent by the Go backend. Adding extra blocking intervals would incorrectly prevent
    # knitting machines (which run autonomously) from scheduling outside those explicit
    # windows, causing guaranteed INFEASIBLE when the last shift ends before horizon.

    if knitting_intervals:
        n_knitting = sum(1 for t in tasks if t.get("operation", "").lower() == "knitting")
        n_cap_block = len(knitting_intervals) - n_knitting
        cap_block_max_demand = max(
            (d for t, d in zip(knitting_intervals, demands) if True),
            default=0,
        ) if demands else 0
        cap_block_demands = [
            demands[i] for i, t in enumerate(tasks[:len(demands)])
            if t.get("operation", "").lower() == "capacity_block"
        ]
        if cap_block_demands:
            cap_block_max_demand = max(cap_block_demands)
        else:
            cap_block_max_demand = 0
        logger.info(
            f"   📊 Workforce AddCumulative: {len(knitting_intervals)} intervals "
            f"(knitting={n_knitting}, capacity_block={n_cap_block}), "
            f"capacity={MAX_MACHINES}, max_cap_block_demand={cap_block_max_demand}, "
            f"knitting_headroom={MAX_MACHINES - cap_block_max_demand}"
        )
        model.AddCumulative(knitting_intervals, demands, MAX_MACHINES)


def _workforce_boolean(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    horizon: int,
    MAX_MACHINES: int,
) -> None:
    task_map = {t["task_id"]: t for t in tasks}
    n_slots = max(1, MAX_MACHINES)
    slot_intervals: List[List] = [[] for _ in range(n_slots)]

    for t_id, tv in task_vars.items():
        t_info = task_map.get(t_id, {})
        op = t_info.get("operation", "").lower()

        ps = t_info.get("pinned_start_time")
        pe = t_info.get("pinned_end_time")
        is_fully_pinned = t_info.get("is_pinned") and ps is not None and pe is not None
        duration = int(pe) - int(ps) if is_fully_pinned else int(t_info.get("duration", 0))

        if op == "capacity_block":
            blocked = min(int(t_info.get("demand", 0)), n_slots)
            for s in range(blocked):
                iv = model.NewIntervalVar(
                    tv["start"], duration, tv["end"], f"bool_blk_{t_id}_s{s}"
                )
                slot_intervals[s].append(iv)

        elif op == "knitting":
            slot_bools = [model.NewBoolVar(f"{t_id}_vslot_{s}") for s in range(n_slots)]
            model.AddExactlyOne(slot_bools)
            for s, sb in enumerate(slot_bools):
                opt_iv = model.NewOptionalIntervalVar(
                    tv["start"], duration, tv["end"], sb, f"bool_k_{t_id}_s{s}"
                )
                slot_intervals[s].append(opt_iv)

    for s in range(n_slots):
        if slot_intervals[s]:
            model.AddNoOverlap(slot_intervals[s])

    logger.info(f"   📊 Workforce Boolean Exclusion: {n_slots} virtual slots")


# ---------------------------------------------------------------------------
# PO bounding-box (tasks from same purchase order stay contiguous on each machine)
# ---------------------------------------------------------------------------

def _apply_material_constraints(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    material_capacities: Dict[str, int],
) -> None:
    """
    AddCumulative per material: mirrors builder.py build_material_constraints().
    Intervals are strictly [task.start, task.end] so material is released on finish.
    """
    task_map = {t["task_id"]: t for t in tasks}
    mat_intervals: Dict[str, List] = {mat: [] for mat in material_capacities}
    mat_demands: Dict[str, List[int]] = {mat: [] for mat in material_capacities}

    for t_id, tv in task_vars.items():
        task = task_map.get(t_id, {})
        task_mat_demands: Dict[str, int] = task.get("material_demands") or {}
        if not task_mat_demands:
            continue

        ps = task.get("pinned_start_time")
        pe = task.get("pinned_end_time")
        is_fully_pinned = task.get("is_pinned") and ps is not None and pe is not None
        duration = int(pe) - int(ps) if is_fully_pinned else max(0, int(task.get("duration", 0)))
        if duration <= 0:
            continue

        for mat_code, demand in task_mat_demands.items():
            if mat_code not in mat_intervals:
                continue
            demand_int = int(demand)
            if demand_int <= 0:
                continue
            
            mat_cap = int(material_capacities.get(mat_code, 0))
            if demand_int > mat_cap > 0:
                logger.warning(
                    f"⚠️ Task {t_id} material '{mat_code}' demand ({demand_int}) exceeds factory capacity "
                    f"({mat_cap}). Capping constraint demand to {mat_cap} to prevent instant solver INFEASIBLE."
                )
                demand_int = mat_cap
                
            interval = model.NewIntervalVar(
                tv["start"], duration, tv["end"], f"mat_{mat_code}_{t_id}"
            )
            mat_intervals[mat_code].append(interval)
            mat_demands[mat_code].append(demand_int)

    for mat_code, capacity in material_capacities.items():
        intervals = mat_intervals.get(mat_code, [])
        demands = mat_demands.get(mat_code, [])
        if not intervals:
            continue
        model.AddCumulative(intervals, demands, int(capacity))
        logger.info(
            f"📦 Material '{mat_code}': AddCumulative "
            f"({len(intervals)} tasks, cap={capacity})"
        )


def _apply_po_bounding_box(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    resource_map: Dict[str, Dict[str, Any]],
    horizon: int,
) -> None:
    # Only group FREE (non-pinned) knitting tasks.
    # Pinned tasks may have pinned_start_time < 0 (in-progress), which would force
    # po_start ≤ negative value while po_start ∈ [0, horizon] → instant INFEASIBLE.
    po_groups: Dict[str, List[Dict]] = {}
    for t in tasks:
        if t.get("operation", "").lower() == "knitting" and not t.get("is_pinned", False):
            po_id = t.get("original_order_id")
            if po_id:
                po_groups.setdefault(po_id, []).append(t)

    for po_id, po_tasks in po_groups.items():
        if len(po_tasks) <= 1:
            continue

        for r_id in resource_map:
            lits, starts, ends = [], [], []
            for t in po_tasks:
                t_id = t["task_id"]
                if t_id not in task_vars:
                    continue
                tv = task_vars[t_id]
                lit = next(
                    (l for l in tv["literals"] if l.Name().endswith(f"_on_{r_id}")),
                    None,
                )
                if lit is not None:
                    lits.append(lit)
                    starts.append(tv["start"])
                    ends.append(tv["end"])

            if len(lits) <= 1:
                continue

            po_start = model.NewIntVar(0, horizon, f"po_{po_id}_{r_id}_start")
            po_end = model.NewIntVar(0, horizon, f"po_{po_id}_{r_id}_end")
            po_active = model.NewBoolVar(f"po_{po_id}_{r_id}_active")
            model.AddMaxEquality(po_active, lits)

            for lit, st, en in zip(lits, starts, ends):
                model.Add(st >= po_start).OnlyEnforceIf(lit)
                model.Add(en <= po_end).OnlyEnforceIf(lit)

            # Intentionally NOT adding po_end - po_start == total_dur_expr.
            # That equality forces zero gap between PO tasks, which is INFEASIBLE when
            # shift breaks separate tasks (span > sum-of-durations). Co-location is
            # already encouraged via machine affinity scoring in the objective.
