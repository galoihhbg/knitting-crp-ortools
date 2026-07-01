"""
Phase 4: Downstream CP-SAT solver (Ironing, Packing, and any other ops).

The washing schedule from Phase 3 is treated as fixed input: each BatchInfo
provides a concrete end_time that becomes a hard start lower-bound for tasks
whose final_depends_on includes a washing task from that batch.

Pipelining constraints (all enforced as integer start_lb, not new CP-SAT vars):
  ironing_start >= washing_batch_end   (via Phase 3 end_times)
  packing_start >= ironing_end         (via Phase 4 end_times after linking)

Tasks are solved in a single CP-SAT model. If IRONING tasks depend on PACKING
tasks (unusual), the dependency is captured via final_depends_on lookup inside
Phase 4's own result — but since they're in the same model, start_lb handles it.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ortools.sat.python import cp_model

from app.engine.shared import (
    apply_order_flow_objective,
    apply_slice_sync_objective,
    apply_soft_deadlines,
    apply_stability_objective,
    build_resource_model,
    compute_horizon,
    extract_results,
    make_solver,
)

logger = logging.getLogger(__name__)

# All operations NOT handled by phases 1–3 land here
UPSTREAM_OPS = frozenset({"knitting", "capacity_block", "linking", "washing"})


@dataclass
class Phase4Result:
    status: str
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    end_times: Dict[str, int] = field(default_factory=dict)
    solve_time_seconds: float = 0.0
    objective_value: Optional[float] = None


def solve_downstream(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    p3_end_times: Dict[str, int],
    horizon: Optional[int] = None,
    reschedule_hint: Optional[Dict[str, Any]] = None,
    workload_shrank: bool = False,
) -> Phase4Result:
    """
    Solve all downstream operations (ironing, packing, or any future op).

    Args:
        tasks:         All remaining tasks not in phases 1–3.
        resources:     Resources compatible with downstream operations.
        config:        Solver config.
        p3_end_times:  task_id → end minute from Phase 3 (washing end times).
    """
    downstream_tasks = [
        t for t in tasks
        if t.get("operation", "").lower() not in UPSTREAM_OPS
    ]
    if not downstream_tasks:
        logger.info("⚙️ Phase 4 (Downstream): no tasks — skipped.")
        return Phase4Result(status="empty")

    if horizon is None:
        horizon = compute_horizon(downstream_tasks, config)

    # Compute start lower-bounds from Phase 3 end times via final_depends_on
    start_lb = _compute_start_lb(downstream_tasks, p3_end_times)

    resource_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in resources}
    model = cp_model.CpModel()

    task_vars, _, no_resource_tasks = build_resource_model(
        model, downstream_tasks, resource_map, horizon, start_lb=start_lb
    )
    if no_resource_tasks:
        ids = [t["task_id"] for t in no_resource_tasks]
        logger.error(f"❌ Phase 4: {len(ids)} task(s) have no resources: {ids}")
        return Phase4Result(
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
                for t in no_resource_tasks
            ],
        )

    task_map = {t["task_id"]: t for t in downstream_tasks}
    obj_terms = apply_soft_deadlines(model, task_vars, task_map, horizon)
    # Re-schedule: skip flow/sync (they outweigh stability pin) — see phase1.
    # EXCEPTION — workload shrank: re-enable so survivors re-pack (no gaps); the
    # soft anchor is one-sided (late_only) this run so it won't fight compaction.
    cold = not reschedule_hint or workload_shrank
    if cold:
        # NB: identical-task symmetry break is NOT applied here — packing's start_lb is
        # derived from washing, but packing actually depends on iron WITHIN this phase,
        # so a washing-based ordering can contradict the intra-phase iron→packing
        # constraints.  Only knitting (independent, first stage) is safe for it.
        obj_terms += apply_order_flow_objective(model, task_vars, downstream_tasks, horizon)
        # slice_sync coordinates cross-order slice TIMING for a DOWNSTREAM consumer
        # (its real job in linking).  Ironing/packing are terminal — nothing consumes
        # their slice ordering — so here slice_sync only adds objective noise that
        # misleads the FEASIBLE-stop: measured it pushed cold iron/packing 14 task-min
        # late, which the first reschedule (which omits flow/sync) then "fixed",
        # producing the run-1≠run-2 drift.  Default OFF on downstream; the reschedule
        # path already skips it.  Flag-gated for reversibility.
        if config.get("enable_downstream_slice_sync", False):
            obj_terms += apply_slice_sync_objective(model, task_vars, downstream_tasks, horizon)

    # ── Intra-phase dependency constraints ──────────────────────────────────
    # final_depends_on may reference tasks within the same Phase 4 model
    # (e.g. packing depends on ironing). start_lb cannot resolve these because
    # ironing has no end_time yet at lb-computation time → must add CP-SAT constraints.
    intra_dep_count = 0
    for t in downstream_tasks:
        t_id = t["task_id"]
        if t_id not in task_vars:
            continue
        for dep_id in (t.get("final_depends_on") or []):
            if dep_id in task_vars:  # dep resolved within this phase
                model.Add(task_vars[t_id]["start"] >= task_vars[dep_id]["end"])
                intra_dep_count += 1
    if intra_dep_count:
        logger.info(f"   🔗 Phase 4: {intra_dep_count} intra-phase dependency constraints added")

    stab_terms, stab_stats = apply_stability_objective(
        model, task_vars, downstream_tasks, reschedule_hint, horizon, start_lb=start_lb,
        time_penalty="late_only" if workload_shrank else "abs",
    )
    obj_terms += stab_terms
    if reschedule_hint:
        logger.info(
            f"   🎯 Phase4 stability_stats: total_previous={stab_stats.total_previous} "
            f"matched_exact={stab_stats.matched_exact} matched_via_order={stab_stats.matched_via_order} "
            f"n_hinted={stab_stats.n_hinted} time_terms={stab_stats.time_terms_added} "
            f"machine_terms={stab_stats.machine_terms_added}"
        )

    model.Minimize(sum(obj_terms) if obj_terms else 0)

    validation = model.Validate()
    if validation:
        logger.error(f"❌ Phase 4 MODEL_INVALID: {validation}")
        return Phase4Result(status="model_invalid")

    # Cold solve: tighten the gap to 0 so the solver pursues the true optimum and
    # balances load across interchangeable machines (independent packing/ironing
    # tasks otherwise serialise onto one machine — the <1% balance gain is swallowed
    # by the default 1% gap). On reschedule keep the 1% gap so the larger stability
    # anchors (machine-swap penalty) win and pinned tasks are not re-optimised away.
    solver = make_solver(
        config,
        has_hint=bool(reschedule_hint),
        relative_gap=0.0 if cold else None,
    )
    status_code = solver.Solve(model)

    logger.info(
        f"⚙️ Phase 4 (Downstream): {len(task_vars)} task vars, "
        f"status={solver.StatusName(status_code)}, "
        f"time={solver.WallTime():.1f}s"
    )

    status_str, assignments, overloads, _, end_times = extract_results(
        solver, status_code, task_vars, downstream_tasks, config=config
    )
    return Phase4Result(
        status=status_str,
        assignments=assignments,
        overloads=overloads,
        end_times=end_times,
        solve_time_seconds=solver.WallTime(),
        objective_value=solver.ObjectiveValue() if status_str == "feasible" else None,
    )


def _compute_start_lb(
    tasks: List[Dict[str, Any]],
    upstream_end_times: Dict[str, int],
) -> Dict[str, int]:
    """
    Derive start lower-bounds from upstream end times via final_depends_on.
    """
    lb: Dict[str, int] = {}
    for t in tasks:
        t_id = t["task_id"]
        current_lb = 0
        for dep_id in (t.get("final_depends_on") or []):
            if dep_id in upstream_end_times:
                current_lb = max(current_lb, upstream_end_times[dep_id])
        if current_lb > 0:
            lb[t_id] = current_lb
    return lb


def left_shift_cold_ironing(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    dep_ends: Dict[str, int],
) -> int:
    """COLD-only post-pass: pull each ironing task to its earliest feasible start on its
    OWN (serial) machine, preserving per-machine order.

    The downstream solver only weakly rewards early starts, so with loose due dates it
    stalls at FEASIBLE and staggers ironing a few minutes after its washing-ready time
    even though the (serial, capacity-1) iron machine is free — the tester observes iron
    starting 1–5 min after the wash finishes.  This pulls each iron task to
    max(release, prev_end) where release = max(start_after_min, latest washing-end it
    depends on via final_depends_on).

    Safety: every task only moves EARLIER (ns ≤ old start) ⇒ packing's release bound only
    relaxes ⇒ downstream stays valid and end-to-end lateness is monotone non-increasing.
    Pinned iron tasks are immovable anchors.  Deterministic O(n log n).  Returns #tasks
    moved.  NOT applied on re-schedule (iron is hard-kept there).
    """
    info = {t["task_id"]: t for t in all_tasks}
    iron_ids = {
        t["task_id"] for t in all_tasks
        if str(t.get("operation", "")).lower() in ("iron", "ironing")
    }
    iron_assigns = [a for a in assignments if a["task_id"] in iron_ids]
    if not iron_assigns:
        return 0

    def _release(t_id: str) -> int:
        t = info[t_id]
        rel = int(t.get("start_after_min", 0))
        for d in (t.get("final_depends_on") or []):
            if d in dep_ends:
                rel = max(rel, int(dep_ends[d]))
        return rel

    by_machine: Dict[str, List[Dict[str, Any]]] = {}
    for a in iron_assigns:
        by_machine.setdefault(a["machine_id"], []).append(a)

    new_start: Dict[str, int] = {}
    new_end: Dict[str, int] = {}
    for _m, items in by_machine.items():
        items.sort(key=lambda a: (a["start_time"], a["end_time"], a["task_id"]))
        prev_end = 0
        for a in items:
            t_id = a["task_id"]
            if info[t_id].get("is_pinned"):
                # in-progress / frozen — immovable anchor, keep solver position
                new_start[t_id] = a["start_time"]
                new_end[t_id] = a["end_time"]
                prev_end = a["end_time"]
                continue
            dur = int(a["end_time"]) - int(a["start_time"])
            ns = max(_release(t_id), prev_end)
            new_start[t_id] = ns
            new_end[t_id] = ns + dur
            prev_end = ns + dur

    moved = 0
    for a in iron_assigns:
        t_id = a["task_id"]
        if a["start_time"] != new_start[t_id]:
            moved += 1
        a["start_time"] = new_start[t_id]
        a["end_time"] = new_end[t_id]
        due = int(info[t_id].get("due_at_min", new_end[t_id] + 1))
        a["status"] = "LATE" if new_end[t_id] > due else "ON_TIME"

    if moved:
        logger.info(
            f"   ⬅️ Cold ironing left-shift: pulled {moved} iron task(s) to earliest "
            f"feasible start (serial-machine idle compaction, downstream untouched)."
        )
    return moved
