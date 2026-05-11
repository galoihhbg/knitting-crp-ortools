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
    obj_terms += apply_order_flow_objective(model, task_vars, downstream_tasks, horizon)
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

    model.Minimize(sum(obj_terms) if obj_terms else 0)

    validation = model.Validate()
    if validation:
        logger.error(f"❌ Phase 4 MODEL_INVALID: {validation}")
        return Phase4Result(status="model_invalid")

    solver = make_solver(config)
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
