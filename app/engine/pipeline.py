"""
4-Phase Sequential Pipeline Orchestrator.

Replaces the monolithic TaskModelBuilder for large payloads by decomposing
the scheduling problem into four independent CP-SAT models:

  Phase 1 — Knitting:    machine allocation + workforce capacity
  Phase 2 — Linking:     waits for knitting via start_lb (no shared CP-SAT vars)
  Phase 3 — Washing:     group-isolated batching (one model per color+substance)
  Phase 4 — Downstream:  ironing, packing, any other ops; waits for washing

Each phase receives only the tasks and resources relevant to it. Dependencies
between phases are passed as plain integers (start/end times), not shared
CP-SAT variables — this is what eliminates combinatorial explosion.

Entry point: Pipeline(config, resources, tasks, material_capacities).run()
Returns the same dict format as the legacy Engine.solve():
  {"status", "assignments", "overloads", "objective_value", "solve_time_seconds"}
"""
import logging
from typing import Any, Dict, List, Optional

from .phases.phase1_knitting import PHASE1_OPS, Phase1Result, solve_knitting
from .shared import compute_global_horizon, diagnose_infeasibility
from .phases.phase2_linking import PHASE2_OPS, Phase2Result, solve_linking
from .phases.phase3_batching import PHASE3_OPS, Phase3Result, solve_washing
from .phases.phase4_downstream import UPSTREAM_OPS, Phase4Result, solve_downstream

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the 4-phase scheduling pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        resources: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        material_capacities: Optional[Dict[str, int]] = None,
    ) -> None:
        self.config = config
        self.resources = resources
        self.tasks = _sanitize_dummy_tasks(tasks)
        self.material_capacities = material_capacities or {}
        self.translation_map: Dict[str, str] = _build_translation_map(self.tasks)

    def run(self) -> Dict[str, Any]:
        if not self.tasks:
            return {"status": "feasible", "assignments": [], "overloads": [],
                    "objective_value": None, "solve_time_seconds": 0.0}

        # Each phase receives ALL resources — tasks' compatible_resource_ids is the
        # sole gating mechanism, not the resource's operation tag.  This mirrors the
        # old monolithic model where every resource was in one shared resource_map.
        all_resources = self.resources

        # Compute the global horizon from per-operation makespans so the
        # horizon is tight (avoids large lateness variable domains) while
        # remaining large enough for all tasks to complete.  A single consistent
        # horizon across all phases preserves CP-SAT determinism (fixed seed + 1 worker).
        global_horizon = compute_global_horizon(self.tasks, all_resources, self.config)
        logger.info(f"🌐 Global horizon: {global_horizon} min")

        # ── Phase 1: Knitting ─────────────────────────────────────────────
        p1_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE1_OPS]
        p1: Phase1Result = solve_knitting(
            p1_tasks, all_resources, self.config,
            material_capacities=self.material_capacities,
            horizon=global_horizon,
        )
        logger.info(f"✅ Phase 1 complete: {len(p1.assignments)} assignments, status={p1.status}")

        if p1.status not in ("feasible", "empty"):
            return _phase_failure_result(p1.status, p1_tasks, all_resources, self.config, global_horizon)

        # ── Phase 2: Linking ──────────────────────────────────────────────
        p2_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE2_OPS]
        p2: Phase2Result = solve_linking(
            p2_tasks, all_resources, self.config,
            p1_start_times=p1.start_times,
            p1_end_times=p1.end_times,
            translation_map=self.translation_map,
            horizon=global_horizon,
        )
        logger.info(f"✅ Phase 2 complete: {len(p2.assignments)} assignments, status={p2.status}")

        if p2.status not in ("feasible", "empty"):
            return _phase_failure_result(p2.status, p2_tasks, all_resources, self.config, global_horizon)

        # ── Phase 3: Washing ──────────────────────────────────────────────
        # Merge Phase 1+2 end times so washing tasks can depend on either
        combined_end_times = {**p1.end_times, **p2.end_times}

        p3_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE3_OPS]
        shift_ends: List[int] = [int(s) for s in self.config.get("shift_ends_min", [])]
        p3: Phase3Result = solve_washing(
            p3_tasks, all_resources, self.config,
            p2_end_times=combined_end_times,
            shift_ends=shift_ends,
            horizon=global_horizon,
        )
        logger.info(
            f"✅ Phase 3 complete: {len(p3.assignments)} assignments, "
            f"{len(p3.batches)} batches, status={p3.status}"
        )

        if p3.status not in ("feasible", "empty"):
            return _phase_failure_result(p3.status, p3_tasks, all_resources, self.config, global_horizon)

        # ── Phase 4: Downstream ───────────────────────────────────────────
        # All end times so downstream can depend on any upstream task
        all_end_times = {**combined_end_times, **p3.end_times}

        p4_tasks = [
            t for t in self.tasks
            if t.get("operation", "").lower() not in UPSTREAM_OPS
        ]
        p4: Phase4Result = solve_downstream(
            p4_tasks, all_resources, self.config,
            p3_end_times=all_end_times,
            horizon=global_horizon,
        )
        logger.info(f"✅ Phase 4 complete: {len(p4.assignments)} assignments, status={p4.status}")

        # ── Aggregate results ─────────────────────────────────────────────
        all_assignments = p1.assignments + p2.assignments + p3.assignments + p4.assignments
        all_overloads = p1.overloads + p2.overloads + p3.overloads + p4.overloads
        total_time = (
            p1.solve_time_seconds + p2.solve_time_seconds
            + p3.solve_time_seconds + p4.solve_time_seconds
        )

        # Objective: sum of per-phase objective values where available
        obj_vals = [
            v for v in [p1.objective_value, p2.objective_value, p4.objective_value]
            if v is not None
        ]
        combined_obj = sum(obj_vals) if obj_vals else None

        return {
            "status": "feasible",
            "assignments": all_assignments,
            "overloads": all_overloads,
            "objective_value": combined_obj,
            "solve_time_seconds": total_time,
        }


# ---------------------------------------------------------------------------
# Helpers (mirrors of builder.py preprocessing)
# ---------------------------------------------------------------------------

def _build_translation_map(tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map sub-task / original-order IDs → parent batch task ID."""
    translation: Dict[str, str] = {}
    for t in tasks:
        translation[t["task_id"]] = t["task_id"]
        if t.get("is_batch") and t.get("sub_tasks"):
            for sub in t["sub_tasks"]:
                translation[sub["task_id"]] = t["task_id"]
                if sub.get("original_order_id"):
                    translation[sub["original_order_id"]] = t["task_id"]
    return translation


def _sanitize_dummy_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge overlapping pinned dummy tasks (qty=0, is_pinned=True) on the same
    machine to prevent AddNoOverlap INFEASIBLE when Go maps multiple shift
    blockers to the same virtual machine slot.
    """
    from collections import defaultdict

    dummies = [t for t in tasks if t.get("is_pinned") and t.get("qty", 0) == 0]
    real_tasks = [t for t in tasks if not (t.get("is_pinned") and t.get("qty", 0) == 0)]

    if not dummies:
        return tasks

    grouped: Dict[str, List] = defaultdict(list)
    for t in dummies:
        r_ids = t.get("compatible_resource_ids") or []
        m_id = t.get("pinned_machine_id") or (r_ids[0] if r_ids else None)
        if not m_id:
            real_tasks.append(t)
            continue
        grouped[m_id].append(t)

    merged_dummies: List[Dict[str, Any]] = []
    for m_id, m_tasks in sorted(grouped.items()):
        intervals = [
            (int(t["pinned_start_time"]), int(t["pinned_end_time"]), t)
            for t in m_tasks
            if t.get("pinned_start_time") is not None and t.get("pinned_end_time") is not None
        ]
        if not intervals:
            continue

        intervals.sort(key=lambda x: x[0])
        merged = []
        cs, ce, bt = intervals[0]
        for s, e, t in intervals[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                merged.append((cs, ce, bt))
                cs, ce, bt = s, e, t
        merged.append((cs, ce, bt))

        for idx, (s, e, base_t) in enumerate(merged):
            new_dummy = dict(base_t)
            new_dummy["task_id"] = f"DUMMY_MERGED_{m_id}_{s}_{e}_{idx}"
            new_dummy["original_order_id"] = new_dummy["task_id"]
            new_dummy["pinned_start_time"] = s
            new_dummy["pinned_end_time"] = e
            new_dummy["duration"] = e - s
            merged_dummies.append(new_dummy)

        if len(intervals) > len(merged):
            logger.info(
                f"🧹 Merged {len(intervals)} dummy tasks → {len(merged)} for machine {m_id}"
            )

    return real_tasks + merged_dummies


def _phase_failure_result(
    status: str,
    tasks: List[Dict[str, Any]],
    resources: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a structured failure response when a phase cannot find a schedule."""
    if resources is not None and config is not None and horizon is not None:
        overloads = diagnose_infeasibility(tasks, resources, config, horizon, status)
    else:
        overloads = [
            {
                "task_id": t["task_id"],
                "order_id": t.get("original_order_id", ""),
                "status": "UNSCHEDULABLE",
                "delay_minutes": 0,
                "root_cause_code": status.upper(),
                "bottleneck_resource_id": None,
                "quantity": t.get("qty", 0),
            }
            for t in tasks
        ]
    return {
        "status": status,
        "assignments": [],
        "overloads": overloads,
        "objective_value": None,
        "solve_time_seconds": 0.0,
    }
