"""
Shared CP-SAT utilities used by all pipeline phases.

Provides:
  compute_horizon        — safe horizon with int64 overflow guard
  make_solver            — configured CpSolver factory
  compute_affinity_penalty — machine affinity scoring (knitting-specific)
  build_resource_model   — task vars + machine assignment + routing constraints
  apply_soft_deadlines   — lateness + early-start objective terms
  extract_results        — post-solve assignment/overload extraction
"""
import math
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)

INT64_MAX: int = 9_223_372_036_854_775_807
_PENALTY_CHANGE_DESIGN: int = 10
_PENALTY_COLD_START: int = 200
_PENALTY_PER_ROLL_SWAP: int = 100


# ---------------------------------------------------------------------------
# Horizon & solver
# ---------------------------------------------------------------------------

def compute_horizon(
    tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    resources: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Compute a safe scheduling horizon.

    Uses the minimum makespan lower-bound to decide whether config_horizon
    needs expanding.  The minimum makespan accounts for parallelism:

        min_makespan = max(
            max_single_task_duration,          # longest single task
            ceil(total_duration / n_machines)  # parallel throughput
        )

    If min_makespan ≤ config_horizon, the user's horizon is sufficient and is
    used as-is.  Keeping the horizon as small as possible keeps CP-SAT variable
    domains tight, which lets the solver prove optimality faster and preserves
    determinism under wall-clock time limits.

    If resources are not supplied (e.g. standalone test calls), falls back to
    the conservative total_duration expansion so feasibility is guaranteed.
    """
    config_horizon = int(config.get("horizon_minutes", 40320))

    real_tasks = [
        t for t in tasks
        if t.get("operation", "").lower() not in ("capacity_block",)
    ]
    max_single = max((int(t.get("duration", 0)) for t in real_tasks), default=0)
    total_duration = sum(int(t.get("duration", 0)) for t in real_tasks)

    if resources is not None:
        # Count only resources that at least one task can actually be assigned to.
        # Passing all resources but tasks only being compatible with a subset would
        # overcount machines and produce an underestimate of the true makespan.
        assignable_ids: Set[str] = set()
        for t in real_tasks:
            assignable_ids.update(t.get("compatible_resource_ids", []))
        resource_ids = {r.get("id") for r in resources if r.get("id")}
        effective_machines = max(1, len(assignable_ids & resource_ids))
        min_makespan = max(max_single, math.ceil(total_duration / effective_machines))
    else:
        # Conservative fallback: assume single machine (sequential worst-case)
        min_makespan = total_duration

    if min_makespan <= config_horizon:
        base_horizon = config_horizon
    else:
        base_horizon = min_makespan + 5000
        logger.warning(
            f"⚠️ Min makespan {min_makespan}min > horizon config {config_horizon}min "
            f"— expanded to {base_horizon}min"
        )

    # Int64 overflow guard: objective term = horizon * weight * 100 per task
    _n = max(len(real_tasks), 1)
    _safe = int(math.isqrt(INT64_MAX // (_n * 100_000 * 2)))
    _safe = max(_safe, config_horizon)
    return min(base_horizon, _safe)


def compute_global_horizon(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> int:
    """
    Compute the global horizon for a multi-operation payload.

    For each operation (knitting, linking, washing, …) the minimum makespan
    is estimated from that operation's total task duration and the number of
    machines that any task in that operation can actually use.  The global
    horizon is the maximum across all operations, ensuring every operation
    can physically complete.

    This replaces the naive sequential total_duration formula, which over-
    estimates the horizon for parallel-machine phases (e.g. 100 knitting tasks
    on 20 machines → 1 200 min, not 24 000 min).
    """
    config_horizon = int(config.get("horizon_minutes", 40320))
    resource_ids: Set[str] = {r.get("id", "") for r in resources}

    # Group by operation
    op_tasks: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        op = t.get("operation", "").lower()
        if op == "capacity_block":
            continue
        op_tasks.setdefault(op, []).append(t)

    max_op_makespan = 0
    for op, op_task_list in op_tasks.items():
        total_dur = sum(int(t.get("duration", 0)) for t in op_task_list)
        max_single = max((int(t.get("duration", 0)) for t in op_task_list), default=0)

        # Count only resources that at least one task in this operation can use
        assignable: Set[str] = set()
        for t in op_task_list:
            assignable.update(t.get("compatible_resource_ids", []))
        n_machines = max(1, len(assignable & resource_ids))

        makespan = max(max_single, math.ceil(total_dur / n_machines))
        max_op_makespan = max(max_op_makespan, makespan)

    if max_op_makespan <= config_horizon:
        base = config_horizon
    else:
        base = max_op_makespan + 5000
        logger.warning(
            f"⚠️ Min makespan {max_op_makespan}min > horizon config {config_horizon}min "
            f"— global horizon expanded to {base}min"
        )

    _n = max(len(tasks), 1)
    _safe = int(math.isqrt(INT64_MAX // (_n * 100_000 * 2)))
    _safe = max(_safe, config_horizon)
    return min(base, _safe)


def make_solver(config: Dict[str, Any]) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = int(config.get("max_search_time", 60))
    solver.parameters.relative_gap_limit = 0.01
    solver.parameters.num_search_workers = int(config.get("num_search_workers", 8))
    solver.parameters.random_seed = int(config.get("random_seed", 42))
    return solver


# ---------------------------------------------------------------------------
# Affinity
# ---------------------------------------------------------------------------

def compute_affinity_penalty(
    resource: Dict[str, Any], task_design: str, task_color_str: str
) -> int:
    curr_design = resource.get("design_item_id", "")
    curr_color_str = resource.get("color_config", "")
    penalty = 0

    if task_design and curr_design and curr_design != task_design:
        penalty += _PENALTY_CHANGE_DESIGN

    if not task_color_str:
        return penalty

    if curr_color_str == task_color_str:
        pass
    elif not curr_color_str:
        total_rolls = sum(
            int(p.split(":")[1]) for p in task_color_str.split("|") if ":" in p
        )
        penalty += _PENALTY_COLD_START + total_rolls * (_PENALTY_PER_ROLL_SWAP // 2)
    else:
        def _parse(s: str) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for item in s.split("|"):
                if ":" in item:
                    m, q = item.split(":", 1)
                    out[m] = int(q)
                else:
                    out[item] = 1
            return out

        curr_y = _parse(curr_color_str)
        task_y = _parse(task_color_str)
        swaps = sum(max(0, tq - curr_y.get(m, 0)) for m, tq in task_y.items())
        penalty += swaps * _PENALTY_PER_ROLL_SWAP

    return penalty


# ---------------------------------------------------------------------------
# Core model builder
# ---------------------------------------------------------------------------

def build_resource_model(
    model: cp_model.CpModel,
    tasks: List[Dict[str, Any]],
    resource_map: Dict[str, Dict[str, Any]],
    horizon: int,
    start_lb: Optional[Dict[str, int]] = None,
    use_affinity: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], List[Any], List[Dict[str, Any]]]:
    """
    Build start/end variables, machine assignment BoolVars, OptionalIntervalVars,
    and AddNoOverlap/AddCumulative routing constraints for a list of tasks.

    Args:
        model:        CpModel being built for this phase.
        tasks:        Tasks for this phase (already filtered by operation type).
        resource_map: {resource_id: resource_dict} for this phase's machines.
                      Modified in-place only when a pinned task references an
                      unregistered machine (auto-register guard).
        horizon:      Upper bound for all time variables.
        start_lb:     Optional {task_id: min_start} lower bounds from prior phases.
        use_affinity: When True, add machine affinity penalty terms to objective.

    Returns:
        task_vars:         {task_id: {start, end, literals, r_ids, due, ...}}
        obj_terms:         Affinity penalty objective terms (empty when use_affinity=False).
        no_resource_tasks: Tasks that could not be assigned to any machine.
    """
    start_lb = start_lb or {}
    task_vars: Dict[str, Dict[str, Any]] = {}
    no_resource_tasks: List[Dict[str, Any]] = []
    obj_terms: List[Any] = []
    # Per-resource list of (OptionalIntervalVar, demand) tuples for routing constraints.
    # demand = task qty; used by AddCumulative when resource capacity > 1 (batch mode).
    resource_intervals: Dict[str, List] = {r_id: [] for r_id in resource_map}

    lateness_scale = min(max(1, horizon // 1000), 50)

    # ── Dedup task_ids (Go may emit duplicates for split segments) ──────────
    seen_ids: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for t in tasks:
        orig = t["task_id"]
        if orig in seen_ids:
            ctr = 2
            cand = f"{orig}_dup{ctr}"
            while cand in seen_ids:
                ctr += 1
                cand = f"{orig}_dup{ctr}"
            t = dict(t)
            t["task_id"] = cand
            logger.warning(f"⚠️ Duplicate task_id '{orig}' → '{cand}'")
        seen_ids.add(t["task_id"])
        deduped.append(t)
    tasks = deduped

    # ── Step 1: create start/end IntVars ────────────────────────────────────
    for t in tasks:
        t_id = t["task_id"]
        compatible_ids = t.get("compatible_resource_ids", [])
        operation = t.get("operation", "").lower()

        if operation != "capacity_block" and not compatible_ids:
            logger.warning(f"⚠️ Task {t_id} has NO compatible resources — skipping.")
            no_resource_tasks.append(t)
            continue

        is_pinned = t.get("is_pinned", False)
        due_at = int(t.get("due_at_min", horizon))

        if (
            is_pinned
            and t.get("pinned_start_time") is not None
            and t.get("pinned_end_time") is not None
        ):
            start_var = model.NewConstant(int(t["pinned_start_time"]))
            end_var = model.NewConstant(int(t["pinned_end_time"]))
        else:
            lb = max(0, start_lb.get(t_id, 0))
            start_var = model.NewIntVar(lb, horizon, f"start_{t_id}")
            end_var = model.NewIntVar(lb, horizon, f"end_{t_id}")
            if operation != "capacity_block":
                duration_val = max(0, int(t.get("duration", 0)))
                model.Add(end_var == start_var + duration_val)
            start_after = int(t.get("start_after_min", 0))
            effective_lb = max(lb, start_after)
            if effective_lb > 0 and not is_pinned:
                model.Add(start_var >= effective_lb)

        task_vars[t_id] = {
            "start": start_var,
            "end": end_var,
            "literals": [],
            "r_ids": list(compatible_ids),
            "due": due_at,
            "original_order_id": t.get("original_order_id", ""),
            "group_id": t.get("group_id", ""),
            "qty": t.get("qty", 0),
            "is_pinned": is_pinned,
        }

    # ── Step 2: machine assignment BoolVars ─────────────────────────────────
    for t in tasks:
        t_id = t["task_id"]
        if t_id not in task_vars:
            continue
        operation = t.get("operation", "").lower()
        if operation == "capacity_block":
            continue

        tv = task_vars[t_id]
        is_pinned = t.get("is_pinned", False)

        if is_pinned:
            effective_id = t.get("pinned_machine_id") or (
                tv["r_ids"][0] if tv["r_ids"] else None
            )
            if not effective_id:
                no_resource_tasks.append(t)
                task_vars.pop(t_id, None)
                continue
            tv["r_ids"] = [effective_id]
            if effective_id not in resource_map:
                resource_map[effective_id] = {
                    "id": effective_id,
                    "type": "serial",
                    "capacity": 1,
                    "unavailability": [],
                    "available_at_min": 0,
                }
                resource_intervals[effective_id] = []

        pinned_start = t.get("pinned_start_time")
        pinned_end = t.get("pinned_end_time")
        is_fully_pinned = is_pinned and pinned_start is not None and pinned_end is not None
        actual_duration = (
            int(pinned_end) - int(pinned_start)
            if is_fully_pinned
            else max(0, int(t.get("duration", 0)))
        )

        # Fully pinned: machine and time are already decided — use a fixed interval
        # var instead of NewBoolVar + model.Add(is_selected==1) + NewOptionalIntervalVar
        # + AddExactlyOne.  This eliminates all solver branching overhead for pinned tasks.
        # extract_results falls back to tv["r_ids"][0] when literals is empty and
        # is_pinned is True, so assignments are still captured correctly.
        if is_fully_pinned:
            effective_id = tv["r_ids"][0]  # already narrowed above
            if effective_id in resource_map and actual_duration > 0:
                fixed_iv = model.NewFixedSizeIntervalVar(
                    int(pinned_start), actual_duration, f"int_fixed_{t_id}"
                )
                task_demand = max(1, int(t.get("qty") or 1))
                resource_intervals.setdefault(effective_id, []).append((fixed_iv, task_demand))
            tv["literals"] = []
            tv["r_ids"] = [effective_id]
            continue  # skip BoolVar / AddExactlyOne entirely

        literals: List[Any] = []
        actual_r_ids: List[str] = []

        for r_id in tv["r_ids"]:
            if r_id not in resource_map:
                continue
            is_selected = model.NewBoolVar(f"{t_id}_on_{r_id}")
            literals.append(is_selected)
            actual_r_ids.append(r_id)

            available_at = int(resource_map[r_id].get("available_at_min", 0))
            if available_at > 0:
                model.Add(tv["start"] >= available_at).OnlyEnforceIf(is_selected)

            if use_affinity:
                penalty = compute_affinity_penalty(
                    resource_map[r_id],
                    t.get("design_item_id", ""),
                    t.get("color_config", ""),
                )
                if penalty > 0:
                    obj_terms.append(is_selected * penalty * 10 * lateness_scale)

            opt_iv = model.NewOptionalIntervalVar(
                tv["start"], actual_duration, tv["end"],
                is_selected, f"int_{t_id}_{r_id}",
            )
            task_demand = max(1, int(t.get("qty") or 1))
            resource_intervals.setdefault(r_id, []).append((opt_iv, task_demand))

        if not literals:
            logger.warning(
                f"⚠️ Task {t_id}: none of {tv['r_ids']} found in resource_map — unschedulable."
            )
            no_resource_tasks.append(t)
            task_vars.pop(t_id, None)
            continue

        model.AddExactlyOne(literals)
        tv["literals"] = literals
        tv["r_ids"] = actual_r_ids

    # ── Step 3: routing constraints per resource ─────────────────────────────
    for r_id, iv_demand_pairs in sorted(resource_intervals.items()):
        res = resource_map.get(r_id, {})
        cap = int(res.get("capacity", 1))

        for window in res.get("unavailability", []):
            w_s, w_e = int(window["start"]), int(window["end"])
            if w_e > w_s:
                unavail = model.NewFixedSizeIntervalVar(
                    w_s, w_e - w_s, f"unavail_{r_id}"
                )
                iv_demand_pairs.append((unavail, cap))  # unavailability occupies full capacity

        if not iv_demand_pairs:
            continue

        ivs = [iv for iv, _ in iv_demand_pairs]
        if cap > 1:
            demands = [d for _, d in iv_demand_pairs]
            model.AddCumulative(ivs, demands, cap)
            logger.debug(f"   🔄 '{r_id}': AddCumulative (cap={cap}, {len(ivs)} ivs, demands={demands[:5]}...)")
        else:
            model.AddNoOverlap(ivs)

    return task_vars, obj_terms, no_resource_tasks


# ---------------------------------------------------------------------------
# Objective terms
# ---------------------------------------------------------------------------

def apply_soft_deadlines(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    task_map: Dict[str, Dict[str, Any]],
    horizon: int,
) -> List[Any]:
    """
    Create a lateness IntVar (= max(0, end - due_at)) per non-pinned, non-dummy task.
    Returns weighted objective terms to be summed into model.Minimize().
    """
    terms: List[Any] = []
    for t_id, tv in sorted(task_vars.items()):
        task = task_map.get(t_id, {})
        if task.get("is_pinned", False):
            continue
        if task.get("operation", "").lower() == "capacity_block":
            continue

        priority = int(task.get("priority", 5))
        due_at = int(task.get("due_at_min", horizon))
        weight = 10 ** (6 - priority)

        max_lateness = max(0, horizon - due_at)
        lateness = model.NewIntVar(0, max_lateness, f"lat_{t_id}")
        model.Add(lateness >= tv["end"] - due_at)
        terms.append(lateness * weight * 100)

        # Tie-breaker: prefer earlier starts (10 000× lighter than lateness)
        start_coeff = max(1, weight // 100)
        terms.append(tv["start"] * start_coeff)

    return terms


def apply_order_flow_objective(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    horizon: int,
) -> List[Any]:
    """
    Minimize the completion time (makespan) and span of each order.

    For non-slice groups this drives parallelism across machines (both group_end
    and span terms).  For groups where every task is is_slice=True, both terms
    are skipped — apply_slice_sync_objective handles coordination instead.
    Keeping span for slice groups causes the solver to bunch all slices of one
    order together before starting the next (span minimisation), delaying the
    downstream tasks that wait for slice_N of multiple POs.
    """
    terms: List[Any] = []
    lateness_scale = min(max(1, horizon // 1000), 50)
    task_map = {t["task_id"]: t for t in tasks}

    # Group tasks by group_id (the Production Order / Order ID)
    groups: Dict[str, List[str]] = {}
    for t in tasks:
        gid = t.get("group_id")
        t_id = t["task_id"]
        if gid and t_id in task_vars and t.get("operation", "").lower() != "capacity_block":
            groups.setdefault(gid, []).append(t_id)

    for gid, t_ids in sorted(groups.items()):
        if not t_ids:
            continue

        # If every task in this group is a slice, skip group_end + span entirely.
        # apply_slice_sync_objective owns cross-order slice coordination.
        if all(task_map.get(tid, {}).get("is_slice") for tid in t_ids):
            continue

        # Representative priority (use the highest priority in the group)
        max_priority = min((int(task_map[tid].get("priority", 5)) for tid in t_ids), default=3)
        weight = 10 ** (6 - max_priority)
        flow_w = (weight * lateness_scale) // 20

        if flow_w <= 0:
            continue

        # group_end = max(all task ends in group)
        group_end = model.NewIntVar(0, horizon, f"group_end_{gid}")
        # group_start = min(all task starts in group)
        group_start = model.NewIntVar(0, horizon, f"group_start_{gid}")

        for tid in t_ids:
            tv = task_vars[tid]
            model.Add(group_end >= tv["end"])
            model.Add(group_start <= tv["start"])

        # Penalty 1: Minimize completion time of the group
        terms.append(group_end * flow_w)

        # Penalty 2: Minimize span (elapsed time) to drive parallelism
        span = model.NewIntVar(0, horizon, f"group_span_{gid}")
        model.Add(span == group_end - group_start)
        terms.append(span * flow_w)

    return terms


def apply_slice_sync_objective(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    horizon: int,
) -> List[Any]:
    """
    For is_slice=True tasks, minimize the maximum end time across all tasks
    sharing the same slice_index (across all orders/groups).

    This drives interleaving: completing slice_1 of ALL orders before any order
    moves to slice_2, so downstream tasks that depend on slice_1 from multiple
    POs can start at the earliest possible time.

    Weight is higher for smaller slice_index because earlier syncs unblock more
    downstream work.  Uses the same flow_w scale as apply_order_flow_objective
    so the two objectives are commensurate.
    """
    terms: List[Any] = []
    task_map = {t["task_id"]: t for t in tasks}

    # Collect is_slice tasks grouped by slice_index
    slice_groups: Dict[int, List[str]] = {}
    for t in tasks:
        if not t.get("is_slice"):
            continue
        t_id = t["task_id"]
        if t_id not in task_vars:
            continue
        idx = int(t.get("slice_index", 0))
        slice_groups.setdefault(idx, []).append(t_id)

    if not slice_groups:
        return terms

    lateness_scale = min(max(1, horizon // 1000), 50)
    max_idx = max(slice_groups.keys())

    # Representative priority: best (lowest) across all slice tasks
    all_priorities = [
        int(task_map[tid].get("priority", 5))
        for t_ids in slice_groups.values()
        for tid in t_ids
        if tid in task_map
    ]
    best_priority = min(all_priorities) if all_priorities else 3
    weight = 10 ** (6 - best_priority)
    flow_w = (weight * lateness_scale) // 20

    for idx, t_ids in sorted(slice_groups.items()):
        if not t_ids:
            continue

        # Higher weight for smaller slice_index (slice_1 is most urgent)
        idx_weight = max(1, max_idx - idx + 1)

        slice_end = model.NewIntVar(0, horizon, f"slice_sync_end_{idx}")
        for t_id in t_ids:
            model.Add(slice_end >= task_vars[t_id]["end"])

        terms.append(slice_end * idx_weight * flow_w)

    return terms


# ---------------------------------------------------------------------------
# Infeasibility diagnosis
# ---------------------------------------------------------------------------

def diagnose_infeasibility(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    horizon: int,
    solver_status: str,
) -> List[Dict[str, Any]]:
    """
    Classify each task with a root-cause code when the solver cannot find a schedule.

    solver_status: "infeasible" | "timeout" | "model_invalid"
    Returns one overload dict per non-capacity_block task.
    """
    real_tasks = [t for t in tasks if t.get("operation", "").lower() != "capacity_block"]

    if solver_status == "timeout":
        return [
            {
                "task_id": t["task_id"],
                "order_id": t.get("original_order_id", ""),
                "status": "UNSCHEDULABLE",
                "delay_minutes": 0,
                "root_cause_code": "SOLVER_TIMEOUT",
                "bottleneck_resource_id": None,
                "quantity": t.get("qty", 0),
            }
            for t in real_tasks
        ]

    resource_ids: Set[str] = {r.get("id") for r in resources if r.get("id")}

    # Detect pinned-task conflicts: pairs on the same machine with overlapping intervals
    pinned_conflict_ids: Set[str] = set()
    pinned = [
        t for t in real_tasks
        if t.get("is_pinned")
        and t.get("pinned_start_time") is not None
        and t.get("pinned_end_time") is not None
        and t.get("pinned_machine_id")
    ]
    for i, ta in enumerate(pinned):
        for tb in pinned[i + 1:]:
            if ta.get("pinned_machine_id") != tb.get("pinned_machine_id"):
                continue
            if int(ta["pinned_start_time"]) < int(tb["pinned_end_time"]) and \
               int(tb["pinned_start_time"]) < int(ta["pinned_end_time"]):
                pinned_conflict_ids.add(ta["task_id"])
                pinned_conflict_ids.add(tb["task_id"])

    # Total load per resource across all tasks compatible with it
    resource_load: Dict[str, int] = {}
    for t in real_tasks:
        for r_id in t.get("compatible_resource_ids", []):
            if r_id in resource_ids:
                resource_load[r_id] = resource_load.get(r_id, 0) + int(t.get("duration", 0))
    overloaded: Set[str] = {r_id for r_id, load in resource_load.items() if load > horizon}

    overloads: List[Dict[str, Any]] = []
    for t in real_tasks:
        t_id = t["task_id"]
        compatible = [r for r in t.get("compatible_resource_ids", []) if r in resource_ids]
        duration = int(t.get("duration", 0))
        start_after = int(t.get("start_after_min", 0))

        if t_id in pinned_conflict_ids:
            code = "PINNED_TASK_CONFLICT"
            bottleneck = t.get("pinned_machine_id")
        elif not t.get("is_pinned") and not t.get("compatible_resource_ids"):
            code = "NO_COMPATIBLE_RESOURCE"
            bottleneck = None
        elif duration > horizon:
            code = "TASK_TOO_LONG"
            bottleneck = compatible[0] if compatible else None
        elif start_after > horizon:
            code = "START_AFTER_EXCEEDS_HORIZON"
            bottleneck = None
        elif compatible and all(r in overloaded for r in compatible):
            code = "MACHINE_OVERLOAD"
            bottleneck = compatible[0] if compatible else None
        else:
            code = "CAPACITY_FULL"
            bottleneck = compatible[0] if compatible else None

        overloads.append({
            "task_id": t_id,
            "order_id": t.get("original_order_id", ""),
            "status": "UNSCHEDULABLE",
            "delay_minutes": 0,
            "root_cause_code": code,
            "bottleneck_resource_id": bottleneck,
            "quantity": t.get("qty", 0),
        })

    return overloads


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _classify_root_cause(
    t_id: str,
    selected_res: str,
    start_val: int,
    solver: cp_model.CpSolver,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    """
    Heuristic post-solve root-cause classifier.

    Priority order (first match wins):
      1. PINNED_TASK_CONFLICT  — a locked task on the same machine displaced us
      2. WORKFORCE_SHORTAGE    — factory capacity was saturated at our desired start
      3. MACHINE_OVERLOAD      — another task on this machine ran before us
      4. CAPACITY_FULL         — fallback

    O(n) per late task.  Only called when solver has a feasible solution.
    """
    task_map = {t["task_id"]: t for t in tasks}
    task_info = task_map.get(t_id, {})
    desired_start = int(task_info.get("start_after_min", 0))

    # 1. PINNED_TASK_CONFLICT
    for t in tasks:
        if (
            t.get("is_pinned")
            and t.get("pinned_machine_id") == selected_res
            and t["task_id"] != t_id
        ):
            ps = t.get("pinned_start_time") or 0
            pe = t.get("pinned_end_time") or 0
            if ps < start_val and pe > desired_start:
                return "PINNED_TASK_CONFLICT"

    # 2. WORKFORCE_SHORTAGE
    config_max = int(config.get("max_factory_machines", 100))
    concurrent_knitting = 0
    blocked_demand = 0
    for other_id, other_tv in task_vars.items():
        if other_id == t_id:
            continue
        o_start = solver.Value(other_tv["start"])
        o_end = solver.Value(other_tv["end"])
        if o_start <= desired_start < o_end:
            other_info = task_map.get(other_id, {})
            op = other_info.get("operation", "").lower()
            if op == "knitting":
                concurrent_knitting += 1
            elif op == "capacity_block":
                blocked_demand += int(other_info.get("demand", 0))
    if (concurrent_knitting + blocked_demand) >= config_max:
        return "WORKFORCE_SHORTAGE"

    # 3. MACHINE_OVERLOAD
    for other_id, other_tv in task_vars.items():
        if other_id == t_id or not other_tv.get("literals"):
            continue
        on_res = any(
            solver.Value(lit) == 1
            for lit in other_tv["literals"]
            if lit.Name().endswith(f"_on_{selected_res}")
        )
        if on_res and solver.Value(other_tv["end"]) <= start_val:
            return "MACHINE_OVERLOAD"

    return "CAPACITY_FULL"


def extract_results(
    solver: cp_model.CpSolver,
    status: int,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    wash_x: Optional[Dict[str, List]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """
    Extract assignments from a solved model.

    Returns:
        status_str, assignments, overloads, start_times, end_times
    """
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status_str = "feasible"
    elif status == cp_model.UNKNOWN:
        logger.warning("⏱ Phase TIMEOUT — no feasible solution found within time limit.")
        return "timeout", [], [], {}, {}
    elif status == cp_model.MODEL_INVALID:
        logger.error("❌ Phase MODEL_INVALID — structural bug in builder.")
        return "model_invalid", [], [], {}, {}
    else:
        logger.warning("❌ Phase INFEASIBLE — no valid schedule exists.")
        return "infeasible", [], [], {}, {}

    task_map = {t["task_id"]: t for t in tasks}
    assignments: List[Dict[str, Any]] = []
    overloads: List[Dict[str, Any]] = []
    start_times: Dict[str, int] = {}
    end_times: Dict[str, int] = {}

    for t in tasks:
        t_id = t["task_id"]
        if t_id not in task_vars:
            continue
        tv = task_vars[t_id]
        start_val = solver.Value(tv["start"])
        end_val = solver.Value(tv["end"])
        start_times[t_id] = start_val
        end_times[t_id] = end_val

        task = task_map.get(t_id, {})
        if task.get("operation", "").lower() == "capacity_block":
            continue

        # Resolve assigned machine
        selected_res: Optional[str] = None
        for i, lit in enumerate(tv.get("literals", [])):
            if solver.Value(lit) == 1:
                selected_res = tv["r_ids"][i]
                break
        if selected_res is None and tv.get("is_pinned"):
            r_ids = tv.get("r_ids", [])
            selected_res = r_ids[0] if r_ids else None
        if selected_res is None:
            continue

        due = tv.get("due", end_val + 1)
        is_late = end_val > due

        # Resolve washing batch slot if available
        batch_slot_id = ""
        if wash_x and t_id in wash_x:
            for k, bv in enumerate(wash_x[t_id]):
                if solver.Value(bv) == 1:
                    batch_slot_id = f"wash_batch_{k}"
                    break

        assignments.append({
            "task_id": t_id,
            "machine_id": selected_res,
            "start_time": start_val,
            "end_time": end_val,
            "group_id": tv.get("group_id", ""),
            "order_id": tv.get("original_order_id", ""),
            "quantity": tv.get("qty", 0),
            "status": "LATE" if is_late else "ON_TIME",
            "batch_slot_id": batch_slot_id,
        })

        if is_late:
            if config is not None:
                root_cause = _classify_root_cause(
                    t_id, selected_res, start_val,
                    solver, task_vars, tasks, config,
                )
            else:
                root_cause = "CAPACITY_FULL"
            overloads.append({
                "task_id": t_id,
                "order_id": tv.get("original_order_id", ""),
                "status": "LATE",
                "delay_minutes": end_val - due,
                "root_cause_code": root_cause,
                "bottleneck_resource_id": selected_res,
                "quantity": tv.get("qty", 0),
            })

    return status_str, assignments, overloads, start_times, end_times
