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


def _earliest_slot(busy: List[tuple], release: int, dur: int) -> int:
    """Earliest start ≥ release for a `dur`-long task on a serial machine whose occupied
    intervals are `busy` (sorted list of [s, e)).  Fits into the first gap large enough;
    falls off the end otherwise."""
    t = release
    for s, e in busy:
        if e <= t:
            continue          # interval entirely before our candidate
        if s >= t + dur:
            break             # gap [t, t+dur) fits before this interval
        t = max(t, e)         # overlaps → push past it
    return t


def left_shift_cold_ironing(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    dep_ends: Dict[str, int],
) -> int:
    """Post-pass: pull each ironing task to its earliest feasible start on the earliest-free
    COMPATIBLE iron machine (iron machines are homogeneous serial/capacity-1 units, so which
    one is interchangeable).

    The downstream solver only weakly rewards early starts, so with loose due dates it stalls
    at FEASIBLE: it staggers ironing minutes after the wash finishes AND piles several
    ready-together slices onto ONE iron machine (serialising them) while sibling machines sit
    idle — the tester sees "only 1 of the 4 iron tasks of this wash runs, the other 3 slip".
    This re-seats each iron task at max(release, earliest free slot on any compatible machine),
    release = max(start_after_min, latest washing-end via final_depends_on).

    Monotone guard: a task is moved ONLY when the new start is ≤ its solver start, so every
    task moves EARLIER (or stays) ⇒ packing's release bound only relaxes ⇒ downstream stays
    valid and end-to-end lateness is monotone non-increasing.  Pinned iron tasks are immovable
    anchors (their machine/time is pre-reserved).  Deterministic (release/start/id order, machine-id
    tie-break).  Runs on re-schedule too (the double-solve returns pass-2 to the UI, where iron is
    hard-kept to pass-1 while washing may have moved earlier); idempotent + pinned-anchored, so it
    re-glues iron to washing without harming stability.  Returns #tasks moved.
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

    # Interchangeable iron-machine pool = every compatible machine across the iron tasks
    # (NOT just the ones the solver used — idle-but-compatible machines are exactly where
    # ready-together slices should spread) ∪ any machine already assigned (safety).
    pool = {a["machine_id"] for a in iron_assigns}
    for a in iron_assigns:
        pool |= set(info[a["task_id"]].get("compatible_resource_ids") or [])

    # Seed the occupancy with EVERY iron task at its solver position, so `busy` is always a
    # valid non-overlapping schedule.  Each task is then re-seated only when a STRICTLY
    # earlier conflict-free slot exists (remove-then-place): guarantees no overlap and
    # monotone (earlier-or-equal) by construction — no separate guard needed.
    busy: Dict[str, List[tuple]] = {m: [] for m in pool}
    for a in iron_assigns:
        busy[a["machine_id"]].append((int(a["start_time"]), int(a["end_time"])))
    for m in busy:
        busy[m].sort()

    moved = 0
    order = sorted(
        (a for a in iron_assigns if not info[a["task_id"]].get("is_pinned")),
        key=lambda a: (a["start_time"], a["end_time"], a["task_id"]),
    )
    for a in order:
        t_id = a["task_id"]
        cur_m, cur_s = a["machine_id"], int(a["start_time"])
        dur = int(a["end_time"]) - cur_s
        rel = _release(t_id)
        # Temporarily lift this task so it can be re-seated (incl. back onto its own slot).
        busy[cur_m].remove((cur_s, int(a["end_time"])))
        compat = set(info[t_id].get("compatible_resource_ids") or [])
        cands = [m for m in sorted(pool) if not compat or m in compat] or [cur_m]
        # Earliest feasible slot over all compatible machines (deterministic id order).
        cand_s = {m: _earliest_slot(list(busy[m]), rel, dur) for m in cands}
        min_s = min(cand_s.values())
        if min_s < cur_s:
            # Strictly earlier available → move (lowest machine id achieving it).
            best_s = min_s
            best_m = min(m for m in cands if cand_s[m] == min_s)
        else:
            # No earlier slot anywhere → stay put (no needless machine churn).
            best_m, best_s = cur_m, cur_s
        if best_m != cur_m or best_s != cur_s:
            moved += 1
        a["machine_id"] = best_m
        a["start_time"] = best_s
        a["end_time"] = best_s + dur
        due = int(info[t_id].get("due_at_min", a["end_time"] + 1))
        a["status"] = "LATE" if a["end_time"] > due else "ON_TIME"
        busy[best_m].append((best_s, best_s + dur))
        busy[best_m].sort()

    if moved:
        logger.info(
            f"   ⬅️ Cold ironing left-shift: re-seated {moved} iron task(s) onto the earliest-free "
            f"compatible machine (spreads ready-together slices, downstream untouched)."
        )
    return moved
