"""
Phase 2: Linking CP-SAT solver.

Receives knitting start/end times from Phase 1 and enforces:
  - start >= knitting_end   (via final_depends_on → Phase 1 end_times)
  - start >= k_start + offset  (via WaitOffsets → Phase 1 start_times)

The two dependency styles are unified into a single start_lb dict passed to
build_resource_model, keeping all CP-SAT calls in one place.
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

PHASE2_OPS = frozenset({"linking"})


@dataclass
class Phase2Result:
    status: str
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    start_times: Dict[str, int] = field(default_factory=dict)
    end_times: Dict[str, int] = field(default_factory=dict)
    solve_time_seconds: float = 0.0
    objective_value: Optional[float] = None


def solve_linking(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    p1_start_times: Dict[str, int],
    p1_end_times: Dict[str, int],
    translation_map: Dict[str, str],
    horizon: Optional[int] = None,
    reschedule_hint: Optional[Dict[str, Any]] = None,
) -> Phase2Result:
    """
    Solve the linking phase.

    Args:
        tasks:           Linking tasks (operation == 'linking').
        resources:       Linking machines.
        config:          Solver config.
        p1_start_times:  task_id -> start minute from Phase 1.
        p1_end_times:    task_id -> end minute from Phase 1.
        translation_map: sub-task / original-order ID → batch task ID.
    """
    linking_tasks = [t for t in tasks if t.get("operation", "").lower() in PHASE2_OPS]
    if not linking_tasks:
        logger.info("⚙️ Phase 2 (Linking): no tasks — skipped.")
        return Phase2Result(status="empty")

    if horizon is None:
        horizon = compute_horizon(linking_tasks, config, resources=resources)

    # Compute start lower-bounds from Phase 1
    start_lb = _compute_start_lb(linking_tasks, p1_start_times, p1_end_times, translation_map)

    resource_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in resources}
    model = cp_model.CpModel()

    task_vars, _, no_resource_tasks = build_resource_model(
        model, linking_tasks, resource_map, horizon, start_lb=start_lb
    )
    if no_resource_tasks:
        ids = [t["task_id"] for t in no_resource_tasks]
        logger.error(f"❌ Phase 2: {len(ids)} task(s) have no resources: {ids}")
        return Phase2Result(
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

    task_map = {t["task_id"]: t for t in linking_tasks}
    obj_terms = apply_soft_deadlines(model, task_vars, task_map, horizon)
    # Re-schedule: skip flow/sync (they outweigh stability pin) — see phase1.
    if not reschedule_hint:
        obj_terms += apply_order_flow_objective(model, task_vars, linking_tasks, horizon)
        obj_terms += apply_slice_sync_objective(model, task_vars, linking_tasks, horizon)

    stab_terms, stab_stats = apply_stability_objective(
        model, task_vars, linking_tasks, reschedule_hint, horizon, start_lb=start_lb,
    )
    obj_terms += stab_terms
    if reschedule_hint:
        logger.info(
            f"   🎯 Phase2 stability_stats: total_previous={stab_stats.total_previous} "
            f"matched_exact={stab_stats.matched_exact} matched_via_order={stab_stats.matched_via_order} "
            f"n_hinted={stab_stats.n_hinted} time_terms={stab_stats.time_terms_added} "
            f"machine_terms={stab_stats.machine_terms_added}"
        )

    model.Minimize(sum(obj_terms) if obj_terms else 0)

    validation = model.Validate()
    if validation:
        logger.error(f"❌ Phase 2 MODEL_INVALID: {validation}")
        return Phase2Result(status="model_invalid")

    solver = make_solver(config, has_hint=bool(reschedule_hint))
    status_code = solver.Solve(model)

    logger.info(
        f"⚙️ Phase 2 (Linking): {len(task_vars)} task vars, "
        f"status={solver.StatusName(status_code)}, "
        f"time={solver.WallTime():.1f}s"
    )

    status_str, assignments, overloads, start_times, end_times = extract_results(
        solver, status_code, task_vars, linking_tasks, config=config
    )
    return Phase2Result(
        status=status_str,
        assignments=assignments,
        overloads=overloads,
        start_times=start_times,
        end_times=end_times,
        solve_time_seconds=solver.WallTime(),
        objective_value=solver.ObjectiveValue() if status_str == "feasible" else None,
    )


def _compute_start_lb(
    tasks: List[Dict[str, Any]],
    p1_start_times: Dict[str, int],
    p1_end_times: Dict[str, int],
    translation_map: Dict[str, str],
) -> Dict[str, int]:
    """
    Derive integer start lower-bounds for each linking task from Phase 1 output.

    Two sources are merged (max wins):
      1. final_depends_on → lb = max(p1_end_times[dep] for dep in deps)
      2. WaitOffsets       → lb = max(p1_start_times[batch] + offset for batch, offset)
    """
    lb: Dict[str, int] = {}

    def _resolve(raw_id: str) -> Optional[str]:
        if raw_id in p1_end_times:
            return raw_id
        translated = translation_map.get(raw_id, raw_id)
        if translated in p1_end_times:
            return translated
        return None

    for t in tasks:
        t_id = t["task_id"]
        current_lb = 0

        # 1. final_depends_on
        for dep_id in (t.get("final_depends_on") or []):
            resolved = _resolve(dep_id)
            if resolved:
                current_lb = max(current_lb, p1_end_times[resolved])
            else:
                logger.debug(
                    f"   Phase 2: linking '{t_id}' depends on '{dep_id}' "
                    "— not found in Phase 1 end_times, skipping."
                )

        # 2. WaitOffsets → linking_start >= knitting_start + offset
        wait_offsets = t.get("wait_offsets") or t.get("WaitOffsets") or {}
        for raw_batch_id, offset in wait_offsets.items():
            resolved = _resolve(raw_batch_id)
            if resolved and resolved in p1_start_times:
                lb_val = p1_start_times[resolved] + int(offset)
                current_lb = max(current_lb, lb_val)
                logger.info(
                    f"   ⏱ Phase 2: '{t_id}' waits for '{resolved}' "
                    f"+{offset}min → lb={lb_val}"
                )
            else:
                logger.debug(
                    f"   Phase 2: wait_offset batch '{raw_batch_id}' not in Phase 1 — skipped."
                )

        if current_lb > 0:
            lb[t_id] = current_lb

    return lb
