"""
4-Phase Sequential Pipeline Orchestrator.

Replaces the monolithic TaskModelBuilder for large payloads by decomposing
the scheduling problem into four independent CP-SAT models:

  Phase 1 — Knitting:    machine allocation + workforce capacity
  Phase 2 — Linking:     waits for knitting via start_lb (no shared CP-SAT vars)
  Phase 3 — Washing:     group-isolated batching (one model per color+substance)
  Phase 4 — Ironing:     waits for washing via start_lb
  Phase 5 — Packing:     waits for ironing; also any other downstream op

Each phase receives only the tasks and resources relevant to it. Dependencies
between phases are passed as plain integers (start/end times), not shared
CP-SAT variables — this is what eliminates combinatorial explosion.

Entry point: Pipeline(config, resources, tasks, material_capacities).run()
Returns the same dict format as the legacy Engine.solve():
  {"status", "assignments", "overloads", "objective_value", "solve_time_seconds"}
"""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

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
        reschedule_hint: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.resources = resources
        self.tasks = _sanitize_dummy_tasks(tasks)
        self.material_capacities = material_capacities or {}
        self.translation_map: Dict[str, str] = _build_translation_map(self.tasks)
        self.reschedule_hint = reschedule_hint
        # Detect BEFORE partitioning: partition_hint_for_pipeline drops previous
        # assignments whose task no longer exists, so the shrink signal must be
        # read from the raw hint vs the full current task list.
        self.workload_shrank: bool = _detect_workload_shrink(reschedule_hint, self.tasks)
        self.partitioned_hint: Dict[str, Any] = (
            partition_hint_for_pipeline(reschedule_hint, self.tasks)
            if reschedule_hint else {}
        )

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
        if self.reschedule_hint:
            logger.info(
                f"🔁 Re-schedule: workload_shrank={self.workload_shrank} "
                f"(orders removed since previous plan → relax to compaction)"
                if self.workload_shrank else
                f"🔁 Re-schedule: workload_shrank=False (same/grown set → keep layout)"
            )

        # ── Phase 1: Knitting ─────────────────────────────────────────────
        p1_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE1_OPS]
        p1_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("knitting"))
        p1: Phase1Result = solve_knitting(
            p1_tasks, all_resources, self.config,
            material_capacities=self.material_capacities,
            horizon=global_horizon,
            reschedule_hint=p1_hint,
            all_pipeline_tasks=self.tasks,
            workload_shrank=self.workload_shrank,
        )
        logger.info(f"✅ Phase 1 complete: {len(p1.assignments)} assignments, status={p1.status}")

        if p1.status not in ("feasible", "empty"):
            return _phase_failure_result(p1.status, p1_tasks, all_resources, self.config, global_horizon)

        # ── Phase 2: Linking ──────────────────────────────────────────────
        p2_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE2_OPS]
        p2_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("linking"))
        p2: Phase2Result = solve_linking(
            p2_tasks, all_resources, self.config,
            p1_start_times=p1.start_times,
            p1_end_times=p1.end_times,
            translation_map=self.translation_map,
            horizon=global_horizon,
            reschedule_hint=p2_hint,
            workload_shrank=self.workload_shrank,
        )
        logger.info(f"✅ Phase 2 complete: {len(p2.assignments)} assignments, status={p2.status}")

        if p2.status not in ("feasible", "empty"):
            return _phase_failure_result(p2.status, p2_tasks, all_resources, self.config, global_horizon)

        # ── Phase 3: Washing ──────────────────────────────────────────────
        # Merge Phase 1+2 end times so washing tasks can depend on either
        combined_end_times = {**p1.end_times, **p2.end_times}

        p3_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE3_OPS]
        shift_ends: List[int] = [int(s) for s in self.config.get("shift_ends_min", [])]
        # Phase 3 is group-isolated; pass the full hint (with washing-group partition
        # already computed) so each group filter its own previous_assignments.
        p3_hint = _hint_for_phase_washing(self.reschedule_hint, self.partitioned_hint.get("washing"))
        p3: Phase3Result = solve_washing(
            p3_tasks, all_resources, self.config,
            p2_end_times=combined_end_times,
            shift_ends=shift_ends,
            horizon=global_horizon,
            reschedule_hint=p3_hint,
            workload_shrank=self.workload_shrank,
        )
        logger.info(
            f"✅ Phase 3 complete: {len(p3.assignments)} assignments, "
            f"{len(p3.batches)} batches, status={p3.status}"
        )

        if p3.status not in ("feasible", "empty"):
            return _phase_failure_result(p3.status, p3_tasks, all_resources, self.config, global_horizon)

        # ── Phase 4: Ironing ──────────────────────────────────────────────
        # Downstream split into ironing → packing so each CP-SAT model stays
        # small (one combined "downstream" model solving ironing+packing together
        # was heavy and could time out / report infeasible).  Same cross-phase
        # handoff as knit→link→wash: integers (end_times), no shared CP-SAT vars.
        all_end_times = {**combined_end_times, **p3.end_times}

        p4_tasks = [t for t in self.tasks if t.get("operation", "").lower() in _PHASE4_OP_SET]
        p4_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("ironing"))
        p4: Phase4Result = solve_downstream(
            p4_tasks, all_resources, self.config,
            p3_end_times=all_end_times,
            horizon=global_horizon,
            reschedule_hint=p4_hint,
            workload_shrank=self.workload_shrank,
        )
        logger.info(f"✅ Phase 4 (Ironing) complete: {len(p4.assignments)} assignments, status={p4.status}")

        # ── Phase 5: Packing (+ any other downstream op) ──────────────────
        # Packing waits for ironing via end_times (start_lb).  Anything downstream
        # that is NOT ironing lands here, preserving the old "any other op" catch-all.
        all_end_times_iron = {**all_end_times, **p4.end_times}
        p5_tasks = [
            t for t in self.tasks
            if t.get("operation", "").lower() not in UPSTREAM_OPS
            and t.get("operation", "").lower() not in _PHASE4_OP_SET
        ]
        p5_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("downstream"))
        p5: Phase4Result = solve_downstream(
            p5_tasks, all_resources, self.config,
            p3_end_times=all_end_times_iron,
            horizon=global_horizon,
            reschedule_hint=p5_hint,
            workload_shrank=self.workload_shrank,
        )
        logger.info(f"✅ Phase 5 (Packing) complete: {len(p5.assignments)} assignments, status={p5.status}")

        # ── Aggregate results ─────────────────────────────────────────────
        all_assignments = (
            p1.assignments + p2.assignments + p3.assignments
            + p4.assignments + p5.assignments
        )
        all_overloads = (
            p1.overloads + p2.overloads + p3.overloads
            + p4.overloads + p5.overloads
        )
        total_time = (
            p1.solve_time_seconds + p2.solve_time_seconds + p3.solve_time_seconds
            + p4.solve_time_seconds + p5.solve_time_seconds
        )

        # Objective: sum of per-phase objective values where available
        obj_vals = [
            v for v in [p1.objective_value, p2.objective_value,
                        p4.objective_value, p5.objective_value]
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
# Reschedule-hint partitioning (B.3)
# ---------------------------------------------------------------------------

_PHASE1_OP_SET = {"knitting", "capacity_block"}
_PHASE2_OP_SET = {"linking"}
_PHASE3_OP_SET = {"washing"}
_PHASE4_OP_SET = {"iron", "ironing"}  # Phase 5 (packing + rest) takes everything else downstream
                                      # (payloads use op "Iron"→"iron"; accept "ironing" too)


def _detect_workload_shrink(
    reschedule_hint: Optional[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> bool:
    """True when orders present in the previous plan are GONE from the current
    task set (e.g. re-scheduling 10 orders down to 5).

    Detected at ORDER level so it is rename-safe: slicing changes task_ids but
    keeps the original_order_id, so a renamed task is NOT counted as removed.
    Falls back to task_id level only when the hint carries no order metadata.

    Why it matters: the previous absolute start times describe a denser plan.
    Keeping them — hard pin OR the symmetric |Δ| soft penalty — freezes the
    surviving tasks at their old clock positions and leaves GAPS where the
    removed work used to sit.  On a shrink every phase relaxes to a one-sided
    (late-only) anchor + cold-style compaction so the schedule re-packs forward.
    """
    if not reschedule_hint:
        return False
    previous = reschedule_hint.get("previous_assignments") or []
    if not previous:
        return False

    current_orders = {
        t.get("original_order_id", "") for t in tasks if t.get("original_order_id")
    }
    prev_orders = {
        p.get("original_order_id", "") for p in previous if p.get("original_order_id")
    }
    if prev_orders:
        return bool(prev_orders - current_orders)

    # No order metadata in the hint — fall back to task_id level (rename-unsafe,
    # but better than silently missing a genuine order removal).
    current_tids = {t["task_id"] for t in tasks}
    prev_tids = {p.get("task_id") for p in previous}
    return bool(prev_tids - current_tids)


def partition_hint_for_pipeline(
    reschedule_hint: Optional[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Split a global reschedule_hint into per-phase / per-group sub-hints.

    Output shape:
        {
          "knitting":   [PreviousAssignment dicts],
          "linking":    [...],
          "washing":    { (color, substance): [...], ... },
          "ironing":    [...],
          "downstream": [...],   # packing + any other downstream op
        }

    The previous_assignments are matched to the current task list by task_id.
    Tasks whose task_id is unknown go to whichever bucket their original_order_id
    maps to via existing tasks (best-effort); otherwise they are dropped so a
    later fallback in apply_stability_objective can still match them by order.

    Always returns the full skeleton so callers can `.get()` safely.
    """
    result: Dict[str, Any] = {
        "knitting": [],
        "linking": [],
        "washing": {},
        "ironing": [],
        "downstream": [],
    }
    if not reschedule_hint:
        return result

    previous = reschedule_hint.get("previous_assignments") or []
    if not previous:
        return result

    task_op: Dict[str, str] = {
        t["task_id"]: t.get("operation", "").lower() for t in tasks
    }
    task_wash_key: Dict[str, Tuple[str, str]] = {
        t["task_id"]: (t.get("color", ""), t.get("substance", ""))
        for t in tasks
        if t.get("operation", "").lower() == "washing"
    }
    # Order-keyed fallback for rename cases (slicing changed → task_id no longer
    # matches but original_order_id still does).  Map order → set of ops it
    # touches, plus per-op washing keys, so a renamed prev can still land in
    # the right partition bucket.
    order_ops: Dict[str, Set[str]] = {}
    order_wash_keys: Dict[str, Set[Tuple[str, str]]] = {}
    for t in tasks:
        oid = t.get("original_order_id", "")
        if not oid:
            continue
        op = t.get("operation", "").lower()
        order_ops.setdefault(oid, set()).add(op)
        if op == "washing":
            order_wash_keys.setdefault(oid, set()).add(
                (t.get("color", ""), t.get("substance", ""))
            )

    def _route(op: str, prev: Dict[str, Any], wash_key: Tuple[str, str]) -> None:
        if op in _PHASE1_OP_SET:
            result["knitting"].append(prev)
        elif op in _PHASE2_OP_SET:
            result["linking"].append(prev)
        elif op in _PHASE3_OP_SET:
            result["washing"].setdefault(wash_key, []).append(prev)
        elif op in _PHASE4_OP_SET:
            result["ironing"].append(prev)
        elif op:
            result["downstream"].append(prev)

    for prev in previous:
        op = task_op.get(prev["task_id"], "").lower()
        if op:
            wash_key = task_wash_key.get(prev["task_id"], ("", ""))
            _route(op, prev, wash_key)
            continue

        # Rename / slicing case — task_id unknown.  Fan out by order's ops so
        # apply_stability_objective's machine-compatibility fallback can pick
        # a sensible match within each phase.
        oid = prev.get("original_order_id", "")
        if not oid or oid not in order_ops:
            continue
        for cand_op in order_ops[oid]:
            if cand_op == "washing":
                for wk in order_wash_keys.get(oid, {("", "")}):
                    _route(cand_op, prev, wk)
            else:
                _route(cand_op, prev, ("", ""))

    return result


def _hint_for_phase(
    base_hint: Optional[Dict[str, Any]],
    subset_previous: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Build a per-phase hint dict carrying the same weights as the global one."""
    if not base_hint or not subset_previous:
        return None
    return {
        "previous_assignments": list(subset_previous),
        "stability_weight_time_per_min": base_hint.get("stability_weight_time_per_min", 500),
        "stability_weight_machine_swap": base_hint.get("stability_weight_machine_swap", 5000),
        "match_by_order_fallback": base_hint.get("match_by_order_fallback", True),
    }


def _hint_for_phase_washing(
    base_hint: Optional[Dict[str, Any]],
    washing_groups: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    """Phase 3 hint: keep the per-group breakdown so _solve_group can filter."""
    if not base_hint or not washing_groups:
        return None
    # Flatten across groups; solve_washing will filter per (color, substance)
    flat = [p for prevs in washing_groups.values() for p in prevs]
    if not flat:
        return None
    h = _hint_for_phase(base_hint, flat)
    if h is None:
        return None
    # Carry the per-group partition for the per-group solver.
    h["_washing_groups"] = {k: list(v) for k, v in washing_groups.items()}
    return h


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
