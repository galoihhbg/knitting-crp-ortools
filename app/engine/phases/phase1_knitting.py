"""
Phase 1: Knitting CP-SAT solver.

Handles:
  - KNITTING tasks — machine allocation + affinity penalties
  - CAPACITY_BLOCK tasks — workforce capacity via AddCumulative/gap-filler
  - PO bounding-box co-location (tasks from same PO contiguous on same machine)

Output: start_times + end_times for all knitting tasks, used by Phase 2 as
lower bounds via final_depends_on and wait_offsets resolution.
"""
import bisect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from app.engine.shared import (
    apply_identical_task_symmetry,
    apply_order_cluster_objective,
    apply_order_flow_objective,
    apply_panel_sync_objective,
    apply_slice_sync_objective,
    apply_soft_deadlines,
    apply_stability_hints_only,
    apply_stability_objective,
    build_panel_map,
    build_resource_model,
    compute_horizon,
    extract_results,
    make_solver,
)
from .placement import (  # A1a shared helpers
    bump_earliest,
    earliest_candidates,
    earliest_sweep as _earliest_gap,
)

logger = logging.getLogger(__name__)

PHASE1_OPS = frozenset({"knitting", "capacity_block"})


def _compute_downstream_chain_min(
    all_tasks: List[Dict[str, Any]],
) -> Dict[str, int]:
    """For every task, compute the minimum number of minutes required from
    that task's END to the end of the longest dependent chain.

    A dependent of t is any task d with `t ∈ d.final_depends_on` or
    `t ∈ d.WaitOffsets.keys()`.  The contribution of d to t's chain is:

        offset_from_t_to_d  +  d.duration  +  chain_min(d)

    where offset is `d.WaitOffsets[t]` if present, else 0 (synchronous dep).

    Used to drop reified-keep on a knitting task whose `prev_start` would
    cascade past horizon, e.g. prev_end ≤ horizon but
    prev_end + chain > horizon → linking/washing/downstream INFEASIBLE.
    """
    task_by_id: Dict[str, Dict[str, Any]] = {t["task_id"]: t for t in all_tasks}
    dependents: Dict[str, List[tuple]] = {}  # t_id → [(d_id, offset)]
    for d in all_tasks:
        d_id = d["task_id"]
        wait = d.get("WaitOffsets") or {}
        deps_from_final = d.get("final_depends_on") or []
        # Union of both reference sets
        all_deps = set(deps_from_final) | set(wait.keys())
        for dep_id in all_deps:
            offset = int(wait.get(dep_id, 0))
            dependents.setdefault(dep_id, []).append((d_id, offset))

    cache: Dict[str, int] = {}

    def chain(t_id: str) -> int:
        if t_id in cache:
            return cache[t_id]
        best = 0
        for d_id, offset in dependents.get(t_id, []):
            d = task_by_id.get(d_id)
            if d is None:
                continue
            d_dur = max(0, int(d.get("duration", 0)))
            cand = offset + d_dur + chain(d_id)
            if cand > best:
                best = cand
        cache[t_id] = best
        return best

    for t in all_tasks:
        chain(t["task_id"])
    return cache


def apply_knitting_keep_lex(
    model: Any,
    task_vars: Dict[str, Dict[str, Any]],
    knitting_tasks: List[Dict[str, Any]],
    reschedule_hint: Optional[Dict[str, Any]],
    horizon: int,
    downstream_chain_min: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Reified-keep on knitting start times.

    For every knitting task that has an EXACT task_id match in
    `reschedule_hint.previous_assignments`, create a Bool `keep_t` with the
    reified constraint:

        model.Add(start == prev_start).OnlyEnforceIf(keep_t)

    Pass 2 (`solve_knitting`) constrains `sum(keep_lits) >= len(keep_lits) - D*`
    so the solver hard-pins at least `N - D*` previous knitting tasks while
    choosing optimally WHICH to break.

    Skipped (not added to keep_lits, logged for diagnostics):
      * Pinned tasks (already have NewConstant start, no choice variable)
      * Prevs whose `prev_start` is outside `[0, horizon - duration]` — likely a
        REBASE error.  A high drop rate signals stale `t=0` from the Go side.
      * Order-fallback matches — knitting freeze is strict-by-task_id only;
        slicing renames are intentionally not stabilised here (would risk
        pinning the WRONG task to a stale start).

    Machine is NOT constrained — design D1: knitting machine stays free,
    `apply_stability_hints_only` adds soft `AddHint(lit)` separately for warm-start.

    Returns
    -------
    dict with:
      keep_lits:        List[BoolVar]   (length N_eligible)
      eligible_ids:     List[str]
      n_prev_knitting:  int             (count of knitting prevs in hint)
      n_dropped_oob:    int             (prev_start outside domain)
      n_dropped_pinned: int             (skipped because is_pinned)
      n_dropped_other:  int             (no exact match in task_vars)
    """
    result = {
        "keep_lits": [],
        "eligible_ids": [],
        "n_prev_knitting": 0,
        "n_dropped_oob": 0,
        "n_dropped_pinned": 0,
        "n_dropped_other": 0,
        "n_dropped_downstream_overflow": 0,
    }
    if not reschedule_hint:
        return result

    previous = reschedule_hint.get("previous_assignments") or []
    if not previous:
        return result

    knitting_ids = {t["task_id"] for t in knitting_tasks
                    if t.get("operation", "").lower() == "knitting"}
    duration_by_id = {t["task_id"]: int(t.get("duration", 0)) for t in knitting_tasks}
    downstream_chain_min = downstream_chain_min or {}

    knitting_prevs = [p for p in previous if p["task_id"] in knitting_ids]
    result["n_prev_knitting"] = len(knitting_prevs)

    for prev in knitting_prevs:
        t_id = prev["task_id"]
        tv = task_vars.get(t_id)
        if tv is None:
            result["n_dropped_other"] += 1
            continue
        if tv.get("is_pinned"):
            result["n_dropped_pinned"] += 1
            continue

        prev_start = int(prev.get("start_time", -1))
        dur = duration_by_id.get(t_id, 0)
        if prev_start < 0 or prev_start + dur > horizon:
            result["n_dropped_oob"] += 1
            logger.warning(
                f"⚠️ Knitting keep DROPPED (OOB): task={t_id} prev_start={prev_start} "
                f"duration={dur} horizon={horizon} — possible stale rebase from Go."
            )
            continue

        # Downstream-chain feasibility check: if prev_end + downstream_chain > horizon,
        # honoring this keep would force a linking/washing/downstream task past horizon.
        # Drop the keep here so this knitting can move earlier and unblock the cascade.
        chain = int(downstream_chain_min.get(t_id, 0))
        if chain > 0 and prev_start + dur + chain > horizon:
            result["n_dropped_downstream_overflow"] += 1
            logger.warning(
                f"⚠️ Knitting keep DROPPED (downstream-overflow): task={t_id} "
                f"prev_start={prev_start} duration={dur} chain={chain} horizon={horizon} "
                f"(prev_end + chain = {prev_start + dur + chain} > horizon)"
            )
            continue

        keep_lit = model.NewBoolVar(f"keep_{t_id}")
        model.Add(tv["start"] == prev_start).OnlyEnforceIf(keep_lit)
        # Also pin the previous MACHINE when still compatible.  Pinning start ALONE
        # left a machine symmetry: interchangeable tasks (same group + same start,
        # e.g. two batches of one order) could swap machines at equal objective
        # cost, so the FIRST re-schedule flipped them vs the cold solve and stability
        # was only reached at run 3.  Pinning the machine too makes knitting a fixed
        # point in ONE re-schedule.  The two-pass max-keep drops this keep if pinning
        # the machine is infeasible, so it never forces INFEASIBLE.
        prev_machine = prev.get("machine_id", "")
        for lit, r_id in zip(tv.get("literals") or [], tv.get("r_ids") or []):
            if r_id == prev_machine:
                model.Add(lit == 1).OnlyEnforceIf(keep_lit)
                break
        result["keep_lits"].append(keep_lit)
        result["eligible_ids"].append(t_id)

    if result["n_prev_knitting"] > 0:
        drop_pct = 100.0 * result["n_dropped_oob"] / result["n_prev_knitting"]
        if drop_pct > 30.0:
            logger.warning(
                f"⚠️ Knitting keep: dropped {result['n_dropped_oob']}/{result['n_prev_knitting']} "
                f"prev_start as OOB ({drop_pct:.0f}%).  Check Go-side rebase logic."
            )

    return result


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


# Default rolling-wave size (free knitting tasks per wave) when config omits
# `knitting_chunk_size`.  NOT a hard 150: it is a starting point — calibrate per
# deployment so each wave reaches OPTIMAL (a wave that returns FEASIBLE is the
# stall pathology and is logged loudly).  Lower = more waves, each easier/OPTIMAL.
_DEFAULT_KNIT_CHUNK: int = 90


@dataclass
class Phase1Result:
    status: str
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    overloads: List[Dict[str, Any]] = field(default_factory=list)
    start_times: Dict[str, int] = field(default_factory=dict)
    end_times: Dict[str, int] = field(default_factory=dict)
    solve_time_seconds: float = 0.0
    objective_value: Optional[float] = None
    # Raw CP-SAT status name ("OPTIMAL"/"FEASIBLE"/…) of the final solve — used by
    # the rolling-wave driver to detect a wave that stalled at FEASIBLE (guardrail).
    solver_status_name: str = ""


def solve_knitting(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    material_capacities: Optional[Dict[str, int]] = None,
    horizon: Optional[int] = None,
    reschedule_hint: Optional[Dict[str, Any]] = None,
    all_pipeline_tasks: Optional[List[Dict[str, Any]]] = None,
    workload_shrank: bool = False,  # accepted for pipeline-call uniformity; knitting
                                    # intentionally ignores it (stability > compaction:
                                    # it stays pinned on shrink, see the keep block below)
    translation_map: Optional[Dict[str, str]] = None,  # sub-task/order → batch id;
                                    # used to resolve linking deps for panel co-completion
    _wave: bool = False,            # internal: True when called for one rolling wave
                                    # (prevents recursive re-chunking)
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

    # ── Rolling-wave (cuốn chiếu) for large payloads ─────────────────────────
    # Solving all knitting in ONE model stalls at FEASIBLE and burns unbounded
    # wall-time at scale (deterministic-time does not bound wall-time).  Split the
    # free tasks into EDD-sorted waves small enough to reach OPTIMAL single-worker
    # (still fully deterministic), pinning each finished wave as fixed intervals
    # for the next.  Opt-in by size so small payloads / tests are byte-unchanged.
    free_knitting = [t for t in knitting_tasks if not t.get("is_pinned")]
    chunk_size = int(config.get("knitting_chunk_size", _DEFAULT_KNIT_CHUNK))
    if not _wave and chunk_size > 0 and len(free_knitting) > chunk_size:
        return _solve_knitting_chunked(
            knitting_tasks, resources, config, horizon, reschedule_hint,
            all_pipeline_tasks, chunk_size, translation_map,
        )

    resource_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in resources}

    _log_task_diagnostics(knitting_tasks, horizon)

    model = cp_model.CpModel()

    task_vars, affinity_terms, no_resource_tasks = build_resource_model(
        model, knitting_tasks, resource_map, horizon, use_affinity=True
    )

    # Break permutation symmetry among identical knitting tasks to speed up the
    # cold solve.  DEFAULT OFF — and the reason is a hard lesson: it speeds phase 1
    # (and improves phase-1's own objective) but on a SEQUENTIAL pipeline it WRECKS
    # downstream.  Each "identical" knitting batch actually feeds a DISTINCT order's
    # linking→washing→iron→packing chain, so a canonical start-order — equivalent
    # for phase 1 — throws away the per-order sequencing that panel-co-completion /
    # EDD encode for downstream.  Measured on a 240-identical-order payload: iron/
    # packing tardiness 13_523 (off) → 79_107 (on), a ~6× regression, while wall
    # time only dropped 54→29 min.  Kept flag-gated for a future TRULY-independent
    # single-phase use case; never enable it where knitting feeds a sequential tail.
    if not reschedule_hint and config.get("enable_identical_symmetry_break", False):
        apply_identical_task_symmetry(model, task_vars, knitting_tasks)

    # Pipeline-wide downstream chain — used both to upper-bound knitting end
    # (so NEW tasks must finish early enough for downstream to fit) and to
    # decide which prev keeps to drop (those whose prev_end + chain overflows).
    downstream_chain: Dict[str, int] = (
        _compute_downstream_chain_min(all_pipeline_tasks)
        if all_pipeline_tasks else {}
    )

    # Hard upper bound: end[t] ≤ horizon − chain[t].  Forces NEW knitting to
    # be scheduled early enough that the entire downstream pipeline fits.
    # For PREV knitting kept at prev_start that violates this bound, pass 1
    # will minimise broken keeps and drop the offending ones.
    n_bounded = 0
    for t_id, tv in task_vars.items():
        if tv.get("is_pinned"):
            continue
        chain = int(downstream_chain.get(t_id, 0))
        if chain > 0:
            ub = horizon - chain
            if ub >= 0:
                model.Add(tv["end"] <= ub)
                n_bounded += 1
            else:
                logger.error(
                    f"❌ Phase 1: task {t_id} has downstream chain {chain} > horizon {horizon}; "
                    f"pipeline geometrically infeasible regardless of keep."
                )
    if n_bounded and reschedule_hint:
        logger.info(
            f"   ⛓  Phase1 downstream chain bound: {n_bounded} tasks constrained to "
            f"end ≤ horizon − chain"
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

    # Panel co-completion map (cold solve only): each panel (linking consumer) →
    # its component knitting batches, inverted from linking final_depends_on.
    # Feeds the panel-sync objective below.
    panel_meta: Dict[str, Dict[str, Any]] = {}
    if (
        not reschedule_hint
        and all_pipeline_tasks
        and config.get("enable_panel_sync_objective", True)
    ):
        knit_ids = {
            t["task_id"] for t in knitting_tasks
            if t.get("operation", "").lower() == "knitting"
        }
        _, panel_meta = build_panel_map(
            all_pipeline_tasks, knit_ids, translation_map
        )

    # Workforce constraints — capacity_block tasks constrain concurrent knitting
    _apply_workforce_constraints(model, task_vars, knitting_tasks, resource_map, config, horizon)

    # Per-material creel capacity (AddCumulative per material)
    # Temporarily disabled per user request
    # if material_capacities:
    #     _apply_material_constraints(model, task_vars, knitting_tasks, material_capacities)

    # PO bounding-box: soft co-location for non-pinned tasks on each machine.
    # Pinned tasks are excluded: their fixed start times may be negative (in-progress)
    # which would force po_start < 0 while po_start ∈ [0, horizon] → instant INFEASIBLE.
    # Soft per-(order,machine) contiguity penalty (cold only): finish each order
    # before starting another on a machine ("dứt điểm đơn đó") instead of interleaving.
    contiguity_w = 0
    if not reschedule_hint and config.get("enable_knitting_contiguity", True):
        lateness_scale = min(max(1, horizon // 1000), 50)
        contiguity_w = max(1, lateness_scale * int(config.get("knitting_contiguity_mult", 4)))
    _po_terms = _apply_po_bounding_box(
        model, task_vars, knitting_tasks, resource_map, horizon, contiguity_w=contiguity_w
    )
    if contiguity_w and _po_terms:
        logger.info(f"   🧱 Phase1 knitting contiguity: {len(_po_terms)} (order,machine) span penalties (w={contiguity_w}).")
        obj_terms += _po_terms

    obj_terms += apply_soft_deadlines(model, task_vars, task_map, horizon)
    # Re-schedule path: skip flow/sync objectives — they fight the reified-keep
    # constraint by trying to re-optimise group_end/span on already-pinned tasks.
    # Previous solve already optimised those.
    # NOTE: knitting is the user's STABILITY priority — it stays pinned on EVERY
    # re-schedule (including workload shrink); we do NOT release the keep or
    # re-enable flow/sync on shrink (that broke stability + speed).  The keep's
    # own downstream-overflow guard already drops any pin that would push a task
    # past the horizon, so stability never forces lateness.
    if not reschedule_hint:
        obj_terms += apply_order_flow_objective(model, task_vars, knitting_tasks, horizon)
        obj_terms += apply_slice_sync_objective(model, task_vars, knitting_tasks, horizon)
        # Whole-order temporal continuity: keep each sales order's components in one
        # window so an order isn't started, abandoned for shifts, then resumed.
        # Below lateness → clusters only when free (see apply_order_cluster_objective).
        if config.get("enable_order_cluster_objective", True):
            cluster_terms = apply_order_cluster_objective(model, task_vars, knitting_tasks, horizon)
            if cluster_terms:
                logger.info(
                    f"   🧷 Phase1 order-cluster objective: {len(cluster_terms)} sales-order "
                    f"span(s) penalised for WIP continuity."
                )
            obj_terms += cluster_terms
        # B1: pull each panel's component batches to finish together (the max-end
        # gates its linking slice).  Commensurate scale → nudge, not override.
        if config.get("enable_panel_sync_objective", True) and panel_meta:
            panel_terms = apply_panel_sync_objective(model, task_vars, panel_meta, horizon)
            if panel_terms:
                logger.info(
                    f"   🧩 Phase1 panel-sync objective: {len(panel_terms)} panel(s) "
                    f"co-completion penalised."
                )
            obj_terms += panel_terms
        # PO setup-change cost: penalise each extra PO per machine so machines stay
        # DEDICATED to a PO (no unmodeled setup ping-pong; same-panel POs parallelise
        # on disjoint machines).  Banded below lateness via lateness_scale.  Default
        # OFF — in-solver terms on the FEASIBLE-stuck knitting solve can be inert or
        # harmful, so it is measured before being trusted.
        if config.get("enable_knitting_setup_cost", False):
            setup_scale = min(max(1, horizon // 1000), 50)
            setup_w = max(1, setup_scale * int(config.get("knitting_setup_mult", 4)))
            setup_terms = _apply_po_setup_cost(
                model, task_vars, knitting_tasks, resource_map, setup_w
            )
            if setup_terms:
                logger.info(
                    f"   🔧 Phase1 PO setup-cost: {len(setup_terms)} machine(s) "
                    f"penalised for hosting extra POs (w={setup_w})."
                )
            obj_terms += setup_terms

    # Reified-keep + hints-only on knitting.  apply_stability_objective is
    # NOT called here (it would double-stabilize via soft time penalty +
    # machine-swap penalty on the same start var).  Hints-only adds AddHint()
    # for warm-start + machine AddHint(lit) without any objective contribution.
    keep_info: Dict[str, Any] = {"keep_lits": []}
    if reschedule_hint:
        _, hint_stats = apply_stability_hints_only(
            model, task_vars, knitting_tasks, reschedule_hint, horizon, start_lb=None,
        )
        # ALWAYS hard-keep knitting on re-schedule (stability priority).  Pinning
        # to the previous starts makes knitting reproducible regardless of solver
        # non-determinism (hash seed / budget) — stability by construction, not by
        # luck — and shrinks the search so it converges fast.  The keep's
        # downstream-overflow eligibility check drops any pin that would force a
        # downstream task past the horizon, so it never causes lateness.
        keep_info = apply_knitting_keep_lex(
            model, task_vars, knitting_tasks, reschedule_hint, horizon,
            downstream_chain_min=downstream_chain,
        )
        logger.info(
            f"   🎯 Phase1 hints: total_previous={hint_stats.total_previous} "
            f"matched_exact={hint_stats.matched_exact} matched_via_order={hint_stats.matched_via_order} "
            f"n_hinted={hint_stats.n_hinted}"
        )
        logger.info(
            f"   🔒 Phase1 keep: n_prev_knitting={keep_info['n_prev_knitting']} "
            f"eligible={len(keep_info['eligible_ids'])} "
            f"dropped_oob={keep_info['n_dropped_oob']} "
            f"dropped_downstream={keep_info.get('n_dropped_downstream_overflow', 0)} "
            f"dropped_pinned={keep_info['n_dropped_pinned']} "
            f"dropped_other={keep_info['n_dropped_other']}"
        )
    elif config.get("enable_edd_knitting_hint", True):
        # Cold solve: EDD warm-start (hints-only, zero objective/constraints).
        # Cold knitting hay dừng ở FEASIBLE với đảo-due trên máy; seed bằng
        # incumbent earliest-due-date để solver khởi đầu từ lịch đã đúng thứ tự
        # due rồi tự cải thiện — xem _edd_warm_start_assignments.
        edd_prev = _edd_warm_start_assignments(knitting_tasks, resource_map, config)
        if edd_prev:
            _, edd_stats = apply_stability_hints_only(
                model, task_vars, knitting_tasks,
                {"previous_assignments": edd_prev, "match_by_order_fallback": False},
                horizon,
            )
            logger.info(
                f"   🧭 Phase1 EDD warm-start: hinted {edd_stats.n_hinted}/{len(edd_prev)} cold tasks"
            )

    validation = model.Validate()
    if validation:
        logger.error(f"❌ Phase 1 MODEL_INVALID: {validation}")
        return Phase1Result(status="model_invalid")

    keep_lits = keep_info["keep_lits"]
    if keep_lits:
        # Pass 1: maximise kept (= minimise n_broken).
        n_broken = model.NewIntVar(0, len(keep_lits), "n_broken_keep")
        model.Add(n_broken == len(keep_lits) - sum(keep_lits))
        model.Minimize(n_broken)

        solver = make_solver(config, has_hint=True)
        pass1_status = solver.Solve(model)
        if pass1_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.error(
                f"❌ Phase 1 pass 1 (keep maximisation) failed: "
                f"status={solver.StatusName(pass1_status)} — re-schedule cannot proceed."
            )
            status_str, assignments, overloads, start_times, end_times = extract_results(
                solver, pass1_status, task_vars, knitting_tasks, config=config
            )
            return Phase1Result(
                status=status_str, assignments=assignments, overloads=overloads,
                start_times=start_times, end_times=end_times,
                solve_time_seconds=solver.WallTime(),
            )

        d_star = int(solver.Value(n_broken))
        logger.info(
            f"   🔒 Phase1 pass 1: D*={d_star} broken / {len(keep_lits)} keep_lits "
            f"(pass1 time={solver.WallTime():.1f}s, status={solver.StatusName(pass1_status)})"
        )
        if d_star > 0:
            broken_ids = [
                keep_info["eligible_ids"][i]
                for i, kl in enumerate(keep_lits)
                if solver.Value(kl) == 0
            ]
            logger.warning(
                f"⚠️ Phase1 lex pass 2 will allow ≤{d_star} broken keep(s); "
                f"sample broken ids: {broken_ids[:5]}"
            )

        # Lex constraint for pass 2.  Pass 2 may choose a DIFFERENT subset of
        # `d_star` tasks to break (whichever minimises pass-2 obj), so we use
        # an inequality not equality on n_broken.
        model.Add(n_broken <= d_star)
        model.ClearObjective()
        model.Minimize(sum(obj_terms) if obj_terms else 0)

        pass1_time = solver.WallTime()
        solver = make_solver(config, has_hint=True)
        status_code = solver.Solve(model)
        logger.info(
            f"⚙️ Phase 1 (Knitting) two-pass: {len(task_vars)} task vars, "
            f"pass1={pass1_time:.1f}s pass2={solver.WallTime():.1f}s "
            f"status={solver.StatusName(status_code)}"
        )
    else:
        # Cold path or no eligible keeps — single-pass with normal objective.
        model.Minimize(sum(obj_terms) if obj_terms else 0)
        solver = make_solver(config, has_hint=bool(reschedule_hint))
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
        solver_status_name=solver.StatusName(status_code),
    )


def _make_knitting_vp(asgn: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a solved wave assignment into a virtual-pinned knitting task.

    Registers a FIXED interval on the assigned machine so the next wave's
    build_resource_model (machine NoOverlap) and workforce cumulative both
    account for capacity already consumed.  operation='knitting' + is_pinned so
    _workforce_cumulative counts it (demand 1) exactly like a real knitting task.
    """
    s, e, m = asgn["start_time"], asgn["end_time"], asgn["machine_id"]
    return {
        "task_id":                 f"__vp_{asgn['task_id']}",
        "operation":               "knitting",
        "is_pinned":               True,
        "pinned_machine_id":       m,
        "pinned_start_time":       s,
        "pinned_end_time":         e,
        "compatible_resource_ids": [m],
        "duration":                max(0, e - s),
        "qty":                     asgn.get("quantity", 1),
        "due_at_min":              999_999,
        "start_after_min":         0,
        "final_depends_on":        [],
        "original_depends_on":     [],
        "original_order_id":       "",
        "group_id":                "",
        "design_item_id":          "",
        "color_config":            "",
        "color":                   "",
        "substance":               "",
        "is_slice":                False,
        "is_batch":                False,
        "sub_tasks":               None,
        "demand":                  1,
        "worker_req":              1,
        "material_demands":        {},
    }


def _solve_knitting_chunked(
    knitting_tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    horizon: int,
    reschedule_hint: Optional[Dict[str, Any]],
    all_pipeline_tasks: Optional[List[Dict[str, Any]]],
    chunk_size: int,
    translation_map: Optional[Dict[str, str]] = None,
) -> Phase1Result:
    """Rolling-wave knitting: EDD-sorted whole-order waves, each solved single-
    worker (deterministic) with previous waves pinned as fixed intervals.

    Mirrors washing's `_solve_group_chunked`.  Determinism holds (each wave is
    single-worker + fixed det budget + sorted; the wave sequence is deterministic;
    the virtual-pin handoff is deterministic).  Re-schedule stability holds because
    `apply_knitting_keep_lex` runs per wave (the hint is filtered to each wave's
    task_ids inside solve_knitting).
    """
    pinned = [t for t in knitting_tasks if t.get("is_pinned")]
    free = [t for t in knitting_tasks if not t.get("is_pinned")]

    # Group whole SALES ORDERS together (never split a customer order's items/slices
    # across waves) and order them EDD-first (earliest due date), then earliest
    # start_after, then highest priority, then order_id for a stable tie.
    # Key = ship_group_id (the customer/sales order shared by all its items) when Go
    # sends it, else fall back to original_order_id (single component ⇒ old behaviour).
    # This keeps every item of one shippable order in the SAME wave so they knit
    # contiguously and finish together — a customer order split across waves is exactly
    # what pins its early items on day 1 and strands the rest days later ("làm item 1,
    # mãi sau mới xong").  NOTE: ship_group_id (đơn khách, coarse) ≠ order_group_id
    # (item, khóa dyelot) — dùng nhầm order_group_id sẽ chỉ gom được 1 item, không gom
    # cả đơn.  Dyelot vẫn dùng order_group_id riêng, không đụng ở đây.
    by_order: Dict[str, List[Dict[str, Any]]] = {}
    for t in free:
        oid = t.get("ship_group_id") or t.get("original_order_id", "") or t["task_id"]
        by_order.setdefault(oid, []).append(t)

    def _order_key(oid: str):
        ts = by_order[oid]
        return (
            min(int(x.get("due_at_min") or horizon) for x in ts),
            min(int(x.get("start_after_min", 0)) for x in ts),
            -max(int(x.get("priority", 0)) for x in ts),
            oid,
        )

    waves: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for oid in sorted(by_order, key=_order_key):
        if cur and len(cur) + len(by_order[oid]) > chunk_size:
            waves.append(cur)
            cur = []
        cur.extend(by_order[oid])
    if cur:
        waves.append(cur)

    logger.info(
        f"🌀 Phase 1 (Knitting) rolling-wave: {len(free)} free + {len(pinned)} pinned "
        f"→ {len(waves)} waves (≤{chunk_size} tasks/wave, EDD-sorted)"
    )

    virtual_pinned: List[Dict[str, Any]] = []
    assignments: List[Dict[str, Any]] = []
    overloads: List[Dict[str, Any]] = []
    start_times: Dict[str, int] = {}
    end_times: Dict[str, int] = {}
    total_time = 0.0
    n_nonoptimal = 0
    pinned_ids = {t["task_id"] for t in pinned}
    pinned_collected = False

    for wi, wave in enumerate(waves):
        wave_tasks = pinned + virtual_pinned + wave
        res = solve_knitting(
            wave_tasks, resources, config,
            horizon=horizon,
            reschedule_hint=reschedule_hint,
            all_pipeline_tasks=all_pipeline_tasks,
            translation_map=translation_map,
            _wave=True,
        )
        if res.status not in ("feasible", "empty"):
            logger.error(
                f"❌ Phase 1 knitting wave {wi + 1}/{len(waves)} status={res.status} "
                f"— aborting rolling-wave"
            )
            return res

        # Guardrail (a): a wave that did NOT reach OPTIMAL is the stall pathology.
        if res.solver_status_name != "OPTIMAL":
            n_nonoptimal += 1
            logger.warning(
                f"⚠️ Phase 1 knitting wave {wi + 1}/{len(waves)} status="
                f"{res.solver_status_name} (NOT OPTIMAL, {len(wave)} tasks, "
                f"{res.solve_time_seconds:.1f}s) — lower config.knitting_chunk_size."
            )
        else:
            logger.info(
                f"   ✅ knitting wave {wi + 1}/{len(waves)} OPTIMAL "
                f"({len(wave)} tasks, {res.solve_time_seconds:.1f}s)"
            )

        wave_ids = {t["task_id"] for t in wave}
        for a in res.assignments:
            tid = a["task_id"]
            if tid.startswith("__vp_"):
                continue
            # this wave's free tasks, plus the originally-pinned tasks once
            if tid in wave_ids or (not pinned_collected and tid in pinned_ids):
                assignments.append(a)
                start_times[tid] = a["start_time"]
                end_times[tid] = a["end_time"]
        overloads.extend(o for o in res.overloads if o.get("task_id") in wave_ids)
        pinned_collected = True
        total_time += res.solve_time_seconds

        # Handoff: pin this wave's solved tasks as fixed intervals for later waves.
        for a in res.assignments:
            if a["task_id"] in wave_ids and not a["task_id"].startswith("__vp_"):
                virtual_pinned.append(_make_knitting_vp(a))

    if n_nonoptimal:
        logger.warning(
            f"⚠️ Phase 1 knitting rolling-wave: {n_nonoptimal}/{len(waves)} waves did "
            f"NOT reach OPTIMAL — reduce config.knitting_chunk_size for full "
            f"determinism + quality (waves must be OPTIMAL)."
        )
    return Phase1Result(
        status="feasible",
        assignments=assignments,
        overloads=overloads,
        start_times=start_times,
        end_times=end_times,
        solve_time_seconds=total_time,
        solver_status_name="OPTIMAL" if n_nonoptimal == 0 else "FEASIBLE",
    )


# ---------------------------------------------------------------------------
# Cold-only left-shift post-pass (idle compaction without re-solving)
# ---------------------------------------------------------------------------

def _edd_warm_start_assignments(
    knitting_tasks: List[Dict[str, Any]],
    resource_map: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Lịch EDD-greedy dùng làm AddHint warm-start cho cold solve (KHÔNG phải nghiệm).

    Vì sao: cold knitting thường dừng ở FEASIBLE với 6–15% cặp đảo-due trên cùng
    máy (đo trên payload thật); seed search bằng một incumbent earliest-due-date
    gỡ 79–85% tổng phút trễ đơn trong replay offline.  Hint đi qua
    apply_stability_hints_only → ZERO ràng buộc/objective mới: model vẫn enforce
    no-overlap/workforce/release, nên hint tồi chỉ làm chậm search chứ không thể
    làm hỏng lịch.

    Greedy: task theo (due_at_min, task_id); mỗi task chọn máy compatible có
    start khả thi sớm nhất (đuôi máy, available_at_min, unavailability, cửa sổ
    pinned), rồi đẩy phải tới khi occupancy toàn cục (knitting chạy đồng thời +
    demand capacity_block đang hoạt động) ≤ max_factory_machines — đúng bất biến
    của _knitting_workforce_ok.  Deterministic (sort theo khóa ổn định).
    """
    MAXM = int(config.get("max_factory_machines", 100))

    # Cửa sổ bận cố định per machine: unavailability + pinned knitting.
    fixed_busy: Dict[str, List[Any]] = {}
    for r_id, res in resource_map.items():
        wins = []
        for w in res.get("unavailability", []) or []:
            ws, we = int(w["start"]), int(w["end"])
            if we > ws:
                wins.append((ws, we))
        fixed_busy[r_id] = wins

    # Occupancy toàn cục: event (time, delta), giữ sorted để sweep sớm-thoát.
    occ: List[Any] = []

    def _occ_add(s: int, e: int, dem: int = 1) -> None:
        bisect.insort(occ, (s, dem))
        bisect.insort(occ, (e, -dem))

    def _occ_fits(s: int, e: int) -> bool:
        cur = 0
        for tm, dl in sorted(occ + [(s, 1), (e, -1)]):
            cur += dl
            if cur > MAXM:
                return False
            if tm > e:
                break
        return True

    def _next_release(t0: int) -> int:
        for tm, dl in occ:
            if dl < 0 and tm > t0:
                return tm
        return t0 + 1

    movable: List[Dict[str, Any]] = []
    for t in knitting_tasks:
        op = t.get("operation", "").lower()
        if op == "capacity_block":
            dem = int(t.get("demand", 0) or 0)
            pe = t.get("pinned_end_time")
            if dem > 0 and pe is not None:
                ps = t.get("pinned_start_time")
                ps = int(ps) if ps is not None else int(pe) - int(t.get("duration", 0) or 0)
                _occ_add(ps, int(pe), dem)
            continue
        if op != "knitting":
            continue
        if t.get("is_pinned"):
            ps, pe = t.get("pinned_start_time"), t.get("pinned_end_time")
            if ps is not None and pe is not None:
                _occ_add(int(ps), int(pe))
                m = t.get("pinned_machine_id")
                if m in fixed_busy:
                    fixed_busy[m].append((int(ps), int(pe)))
            continue
        movable.append(t)

    for wins in fixed_busy.values():
        wins.sort()

    machine_tail: Dict[str, int] = {
        r_id: int(res.get("available_at_min", 0) or 0)
        for r_id, res in resource_map.items()
    }

    out: List[Dict[str, Any]] = []
    movable.sort(key=lambda t: (int(t.get("due_at_min", 0) or 0) or 10**9, t["task_id"]))
    for t in movable:
        dur = max(0, int(t.get("duration", 0) or 0))
        release = int(t.get("start_after_min", 0) or 0)
        compatible = [m for m in (t.get("compatible_resource_ids") or []) if m in machine_tail]
        if not compatible:
            continue

        def _machine_start(m: str) -> int:
            st = max(machine_tail[m], release)
            moved = True
            while moved:
                moved = False
                for ws, we in fixed_busy[m]:
                    if st < we and st + dur > ws:
                        st = we
                        moved = True
            return st

        best_m = min(compatible, key=lambda m: (_machine_start(m), m))
        st = _machine_start(best_m)
        while dur > 0 and not _occ_fits(st, st + dur):
            st = _next_release(st)
            moved = True
            while moved:  # đẩy qua workforce có thể rơi vào cửa sổ bận của máy
                moved = False
                for ws, we in fixed_busy[best_m]:
                    if st < we and st + dur > ws:
                        st = we
                        moved = True
        if dur > 0:
            _occ_add(st, st + dur)
        machine_tail[best_m] = st + dur
        out.append({
            "task_id": t["task_id"],
            "machine_id": best_m,
            "start_time": st,
            "end_time": st + dur,
        })
    return out


def _knitting_workforce_ok(
    new_start: Dict[str, int],
    new_end: Dict[str, int],
    knitting_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> bool:
    """True iff the compacted schedule keeps (concurrent knitting + active
    capacity_block demand) ≤ max_factory_machines at every instant — the same
    AddCumulative invariant `_workforce_cumulative` enforces in the model.

    Half-open intervals [start, end): an interval ending at t does not overlap one
    starting at t, so ties process end-events (negative delta) before start-events.
    """
    MAXM = int(config.get("max_factory_machines", 100))
    events: List[Any] = []
    for t_id, s in new_start.items():
        events.append((s, 1))
        events.append((new_end[t_id], -1))
    for t in knitting_tasks:
        if t.get("operation", "").lower() != "capacity_block":
            continue
        demand = int(t.get("demand", 0))
        if demand <= 0:
            continue
        pe = t.get("pinned_end_time")
        if pe is None:
            continue
        pe = int(pe)
        ps = t.get("pinned_start_time")
        ps = int(ps) if ps is not None else pe - int(t.get("duration", 0))
        events.append((ps, demand))
        events.append((pe, -demand))
    events.sort()  # (time, delta): ends (negative) before starts at equal time
    cur = 0
    for _, d in events:
        cur += d
        if cur > MAXM:
            return False
    return True


def _reentries(orders: List[str]) -> int:
    """Number of orders that appear in ≥2 separate runs in this time-ordered order
    sequence — i.e. the count of A…B…A interleavings on a machine."""
    runs: List[str] = []
    for o in orders:
        if not runs or runs[-1] != o:
            runs.append(o)
    from collections import Counter
    return sum(v - 1 for v in Counter(runs).values() if v > 1)


def _earliest_nonfrag_start(
    placed: List[tuple], release: int, dur: int, order: str
) -> Optional[int]:
    """Earliest start ≥ release for a `dur`-long task of `order` on a machine whose
    placed intervals are `placed` (list of (s, e, order), sorted by s), such that the
    task (a) does not overlap any existing interval and (b) does NOT increase the
    machine's order-reentry (A…B…A) count — so spreading never fragments an order or
    splits another order's run.  Returns None if no such start exists.

    A1a: thân hàm delegate sang ``placement.earliest_candidates`` với hook
    ``accept`` = guard re-entry (tương đương fuzz-đối-chiếu trong
    tests/test_placement_helpers.py)."""
    base_re = _reentries([o for _, _, o in placed])

    def _no_new_reentry(cs: int, ce: int) -> bool:
        merged = sorted(placed + [(cs, ce, order)])
        return _reentries([o for _, _, o in merged]) <= base_re

    return earliest_candidates(
        [(s, e) for s, e, _ in placed], release, dur,
        accept=_no_new_reentry,
    )


def spread_cold_knitting(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> int:
    """COLD-only post-pass: pull each knitting task to its earliest feasible start
    across ALL its COMPATIBLE machines (not just its own).  Generalises
    `left_shift_cold_knitting` from same-machine compaction to cross-machine
    re-balancing, so a PO whose tail the solver serialised onto one machine
    (FEASIBLE-stall) gets spread over idle compatible machines — its components
    finish earlier and the linking panel (max of component ends) is ready sooner.

    Safety (no re-solve, downstream left byte-identical, like left_shift):
    EVERY knitting task is first seeded at its ORIGINAL (machine, start); a free task
    is then moved only into an EARLIER feasible gap.  Because a task is removed from
    its own slot only while it is being processed — and no other task can have taken
    that slot (a task can only occupy a gap, and this slot was occupied until now) —
    every task can always fall back to its original position, so ``new_start ≤
    original_start`` for all.  Tasks therefore only move EARLIER ⇒ every downstream
    release bound (D.start ≥ knit_end) only relaxes ⇒ end-to-end lateness is
    monotonically non-increasing and downstream assignments stay untouched.

    Only COMPATIBLE machines are used (affinity preserved); pinned/in-progress tasks
    are immovable anchors pre-occupying their machine.  Contiguity-aware: a task is
    only re-placed where it does NOT increase any machine's order-reentry (A…B…A)
    count, so spreading never re-fragments an order or adds a setup change — it merely
    parallelises onto idle compatible machines (append/extend, never split).  After
    placement the workforce cumulative cap is validated — if cross-machine parallelism
    would exceed it the whole spread is abandoned (return 0) and the caller falls back
    to the same-machine left-shift.  Deterministic: ties broken by machine_id.

    NOT applied on re-schedule (knitting is hard-kept there).  Returns #tasks moved.
    """
    info = {t["task_id"]: t for t in all_tasks}
    knit_assigns = [
        a for a in assignments
        if (info.get(a["task_id"]) or {}).get("operation", "").lower() == "knitting"
    ]
    if not knit_assigns:
        return 0

    orig_m = {a["task_id"]: a["machine_id"] for a in knit_assigns}
    orig_s = {a["task_id"]: int(a["start_time"]) for a in knit_assigns}
    dur_of = {a["task_id"]: int(a["end_time"]) - int(a["start_time"]) for a in knit_assigns}
    order_of = {
        a["task_id"]: (info[a["task_id"]].get("original_order_id") or a["task_id"])
        for a in knit_assigns
    }

    # Seed every knitting task (free AND pinned) at its original slot, so each task's
    # own position is always available as a fallback ⇒ no task can be pushed later.
    placed: Dict[str, List[tuple]] = {}
    for a in knit_assigns:
        placed.setdefault(a["machine_id"], []).append(
            (int(a["start_time"]), int(a["end_time"]), order_of[a["task_id"]])
        )
    for m in placed:
        placed[m].sort()

    cur_m: Dict[str, str] = dict(orig_m)
    cur_s: Dict[str, int] = dict(orig_s)

    free = sorted(
        (a for a in knit_assigns if not info[a["task_id"]].get("is_pinned")),
        key=lambda a: (orig_s[a["task_id"]], a["task_id"]),
    )
    for a in free:
        t_id = a["task_id"]
        dur = dur_of[t_id]
        order = order_of[t_id]
        release = int(info[t_id].get("start_after_min", 0) or 0)
        compat = list(info[t_id].get("compatible_resource_ids") or [])
        if orig_m[t_id] not in compat:
            compat.append(orig_m[t_id])  # original machine is always a candidate

        # Lift this task out of its current slot before searching for an earlier gap.
        m0, s0 = cur_m[t_id], cur_s[t_id]
        placed[m0].remove((s0, s0 + dur, order))

        best_m, best_s = orig_m[t_id], orig_s[t_id]  # guaranteed-available fallback
        for m in sorted(compat):
            s = _earliest_nonfrag_start(placed.get(m, []), release, dur, order)
            if s is not None and (s < best_s or (s == best_s and m < best_m)):
                best_m, best_s = m, s

        cur_m[t_id], cur_s[t_id] = best_m, best_s
        lst = placed.setdefault(best_m, [])
        lst.append((best_s, best_s + dur, order))
        lst.sort()

    new_start = cur_s
    new_end = {t_id: cur_s[t_id] + dur_of[t_id] for t_id in cur_s}
    new_machine = cur_m

    knitting_tasks = [t for t in all_tasks
                      if t.get("operation", "").lower() in ("knitting", "capacity_block")]
    if not _knitting_workforce_ok(new_start, new_end, knitting_tasks, config):
        logger.info(
            "🧵 Cold knitting spread SKIPPED: cross-machine parallelism would exceed "
            "the workforce cap — falling back to same-machine left-shift."
        )
        return 0

    moved = 0
    for a in assignments:
        t_id = a["task_id"]
        if t_id not in new_start:
            continue
        if a["machine_id"] != new_machine[t_id] or a["start_time"] != new_start[t_id]:
            moved += 1
        a["machine_id"] = new_machine[t_id]
        a["start_time"] = new_start[t_id]
        a["end_time"] = new_end[t_id]
        due = int(info[t_id].get("due_at_min", new_end[t_id] + 1) or (new_end[t_id] + 1))
        a["status"] = "LATE" if new_end[t_id] > due else "ON_TIME"

    if moved:
        logger.info(
            f"   🧵 Cold knitting spread: re-balanced {moved} task(s) onto earliest "
            f"feasible compatible machines (parallelised serial tails, downstream untouched)."
        )
    return moved


def balance_cold_knitting(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> int:
    """Final knitting makespan-balance: move a makespan-defining TAIL task onto the
    earliest-free COMPATIBLE machine whenever that strictly lowers its finish and the
    workforce cap still holds.

    Why: spread/left-shift run BEFORE the parallel-component-PO relayout, which then
    re-concentrates a PO's tail onto one machine — leaving 1–2 machines ending far later
    than the rest while compatible machines sit idle (user: "2 máy chạy lâu hẳn, cắt 2
    task cuối sang 2 máy xong sớm").  This peels those critical tails across, greedily,
    until no move lowers the makespan.

    Safety: a task is moved only to a start EARLIER than its current one AND only when the
    per-move workforce check passes (NOT all-or-nothing) ⇒ every task moves EARLIER, the
    makespan is monotone non-increasing, downstream release bounds only relax (lateness
    non-increasing).  Pinned tasks immovable.  Deterministic (critical tasks by id, target
    by earliest-slot then machine id).  Returns #moves.
    """
    info = {t["task_id"]: t for t in all_tasks}
    knit = [a for a in assignments
            if str(info.get(a["task_id"], {}).get("operation", "")).lower() == "knitting"]
    if len(knit) < 2:
        return 0
    knitting_tasks = [t for t in all_tasks
                      if str(t.get("operation", "")).lower() in ("knitting", "capacity_block")]

    pool = {a["machine_id"] for a in knit}
    for a in knit:
        pool |= set(info[a["task_id"]].get("compatible_resource_ids") or [])
    pinned = {a["task_id"] for a in knit if info[a["task_id"]].get("is_pinned")}
    a_by_id = {a["task_id"]: a for a in knit}
    busy: Dict[str, List[List[int]]] = {m: [] for m in pool}
    for a in knit:
        busy[a["machine_id"]].append([int(a["start_time"]), int(a["end_time"]), a["task_id"]])
    for m in busy:
        busy[m].sort()

    # Pre-fold pinned capacity_block demand once — it never moves, so workforce re-checks
    # only need to overlay the current knitting start/end each time.
    cb_start: Dict[str, int] = {}
    cb_end: Dict[str, int] = {}
    for t in knitting_tasks:
        if str(t.get("operation", "")).lower() == "capacity_block" and t.get("pinned_end_time") is not None:
            pe = int(t["pinned_end_time"])
            ps = int(t.get("pinned_start_time") if t.get("pinned_start_time") is not None
                     else pe - int(t.get("duration", 0)))
            cb_start[t["task_id"]] = ps
            cb_end[t["task_id"]] = pe

    def _workforce_ok(move_tid: str, new_s: int, dur: int) -> bool:
        ns = {a["task_id"]: a["start_time"] for a in knit}
        ne = {a["task_id"]: a["end_time"] for a in knit}
        ns[move_tid] = new_s
        ne[move_tid] = new_s + dur
        ns.update(cb_start)
        ne.update(cb_end)
        return _knitting_workforce_ok(ns, ne, knitting_tasks, config)

    moved = 0
    for _ in range(len(knit) + 1):
        makespan = max((iv[1] for m in busy for iv in busy[m]), default=0)
        crit = sorted(
            (iv for m in busy for iv in busy[m] if iv[1] == makespan and iv[2] not in pinned),
            key=lambda iv: iv[2],
        )
        applied = False
        for s0, e0, tid in crit:
            dur = e0 - s0
            rel = int(info[tid].get("start_after_min", 0) or 0)
            cur_m = a_by_id[tid]["machine_id"]
            compat = set(info[tid].get("compatible_resource_ids") or [])
            cands = [m for m in sorted(pool)
                     if (not compat or m in compat) and m != cur_m]
            best_m, best_s = None, None
            for m in cands:
                slot = _earliest_gap([(x, y) for x, y, _ in busy[m]], rel, dur)
                if best_s is None or slot < best_s:
                    best_m, best_s = m, slot
            # Only worthwhile if the tail lands strictly before the current makespan.
            if best_s is not None and best_s + dur < makespan and _workforce_ok(tid, best_s, dur):
                busy[cur_m] = [iv for iv in busy[cur_m] if iv[2] != tid]
                busy[best_m].append([best_s, best_s + dur, tid])
                busy[best_m].sort()
                a = a_by_id[tid]
                a["machine_id"] = best_m
                a["start_time"] = best_s
                a["end_time"] = best_s + dur
                due = int(info[tid].get("due_at_min", a["end_time"] + 1) or (a["end_time"] + 1))
                a["status"] = "LATE" if a["end_time"] > due else "ON_TIME"
                moved += 1
                applied = True
                break
        if not applied:
            break

    if moved:
        logger.info(
            f"   ⚖️ Cold knitting balance: moved {moved} tail task(s) to earlier compatible "
            f"machines (makespan lowered, workforce-validated, downstream untouched)."
        )
    return moved


def left_shift_cold_knitting(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> int:
    """COLD-only post-pass on the FINAL pipeline output: pull each knitting task to its
    earliest feasible start on its OWN machine, preserving per-machine order.

    Runs AFTER every phase (downstream is already scheduled), so it leaves all
    non-knitting assignments byte-identical and only rewrites knitting start/end in
    place.  Returns the number of tasks moved.

    Why this is safe — knitting stalls at FEASIBLE so the span objective leaves
    machines idle even when a task could run earlier on the same machine with NO
    constraint forcing the gap.  Knitting has no intra-phase precedence / wait_offsets
    (verified on real payloads), so the only binds are machine no-overlap, release
    (`start_after_min`) and the workforce cumulative cap:
      * no-overlap & release  — preserved by construction (start = max(release, prev_end));
      * workforce             — validated by `_knitting_workforce_ok`; if a shift would
                                ever exceed the cap the schedule is left UNCHANGED;
      * downstream precedence  — every knitting task only moves EARLIER, so any
                                downstream task D with D.start ≥ old_knit_end still
                                satisfies D.start ≥ new_knit_end (the bound only relaxes);
      * end-to-end lateness    — knitting ends move earlier (tardiness ≤) and every
                                downstream task is left untouched ⇒ total lateness is
                                monotonically non-increasing.

    Pinned (in-progress) tasks are immovable anchors.  Deterministic O(n log n).
    NOT applied on re-schedule (knitting is hard-kept there — would fight the keep).
    """
    info = {t["task_id"]: t for t in all_tasks}
    by_machine: Dict[str, List[Dict[str, Any]]] = {}
    for a in assignments:
        t = info.get(a["task_id"])
        if t is None or t.get("operation", "").lower() != "knitting":
            continue
        by_machine.setdefault(a["machine_id"], []).append(a)
    if not by_machine:
        return 0

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
            sa = int(info[t_id].get("start_after_min", 0))
            dur = int(a["end_time"]) - int(a["start_time"])
            ns = max(sa, prev_end)
            new_start[t_id] = ns
            new_end[t_id] = ns + dur
            prev_end = ns + dur

    knitting_tasks = [t for t in all_tasks
                      if t.get("operation", "").lower() in ("knitting", "capacity_block")]
    if not _knitting_workforce_ok(new_start, new_end, knitting_tasks, config):
        logger.warning(
            "⚠️ Cold knitting left-shift SKIPPED: compaction would exceed workforce cap "
            "— keeping solver schedule unchanged."
        )
        return 0

    moved = 0
    for a in assignments:
        t_id = a["task_id"]
        if t_id not in new_start:
            continue
        if a["start_time"] != new_start[t_id]:
            moved += 1
        a["start_time"] = new_start[t_id]
        a["end_time"] = new_end[t_id]
        due = int(info[t_id].get("due_at_min", new_end[t_id] + 1))
        a["status"] = "LATE" if new_end[t_id] > due else "ON_TIME"

    if moved:
        logger.info(
            f"   ⬅️ Cold knitting left-shift: pulled {moved} task(s) to earliest "
            f"feasible start (same-machine idle compaction, downstream untouched)."
        )
    return moved


def _count_fragmented_orders(
    by_machine: Dict[str, List[Dict[str, Any]]],
    info: Dict[str, Dict[str, Any]],
    start_of,
) -> int:
    """Number of (order, machine) pairs whose tasks are split into ≥2 time-runs.

    `start_of(task_id)` returns the start time to sort by.  An order interleaved by
    another order on a machine shows up as the same order appearing in >1 run.
    """
    frag = 0
    for items in by_machine.values():
        seq = sorted(items, key=lambda a: (start_of(a["task_id"]), a["task_id"]))
        runs: List[str] = []
        for a in seq:
            oid = info[a["task_id"]].get("original_order_id") or a["task_id"]
            if not runs or runs[-1] != oid:
                runs.append(oid)
        from collections import Counter
        c = Counter(runs)
        frag += sum(1 for v in c.values() if v > 1)
    return frag


def reorder_contiguous_knitting(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, int]]]:
    """COLD-only: re-sequence each machine so an order's knitting tasks run
    contiguously ("dứt điểm đơn đó") instead of interleaved with other orders.

    Returns ``{"start": {tid: s}, "end": {tid: e}}`` for the knitting tasks IFF the
    reorder (a) keeps the workforce cap, and (b) strictly reduces the number of
    fragmented (order, machine) pairs.  Returns ``None`` otherwise — the caller then
    keeps the solver schedule.

    This does NOT decide acceptance on its own: re-sequencing moves the YIELDING
    order's tasks LATER, so unlike `left_shift_cold_knitting` it is NOT
    downstream-safe by construction.  The pipeline runs it as a CANDIDATE, re-solves
    phases 2–5 on the new ends, and accepts only if total lateness does not increase
    (see `Pipeline._try_knitting_relayout`).

    Determinism: orders are sequenced by (earliest original start on the machine,
    order_id); tasks within an order by (original start, task_id).  Machines that
    carry a pinned/in-progress knitting task are left untouched (the pin is an
    immovable anchor and repacking around it is out of scope for the cold path).
    """
    info = {t["task_id"]: t for t in all_tasks}
    by_machine: Dict[str, List[Dict[str, Any]]] = {}
    for a in assignments:
        t = info.get(a["task_id"])
        if t is None or t.get("operation", "").lower() != "knitting":
            continue
        by_machine.setdefault(a["machine_id"], []).append(a)
    if not by_machine:
        return None

    orig_start = {a["task_id"]: a["start_time"] for items in by_machine.values() for a in items}
    base_end = {a["task_id"]: a["end_time"] for items in by_machine.values() for a in items}
    base_frag = _count_fragmented_orders(by_machine, info, lambda tid: orig_start[tid])

    # Per-task latest end that keeps the task's whole downstream chain inside the
    # order due (chain_min = Σ downstream durations, optimistic re: contention — the
    # pipeline-wide verify is the real gate).  A task already at/past this (tight or
    # late) is capped at its baseline end so the reorder never pushes it later.  This
    # makes the candidate contiguize only the SLACK machines; the rest stay verbatim.
    chain_min = _compute_downstream_chain_min(all_tasks)
    cap: Dict[str, int] = {}
    for items in by_machine.values():
        for a in items:
            tid = a["task_id"]
            due = int(info[tid].get("due_at_min", 0) or 0)
            safe = (due - int(chain_min.get(tid, 0))) if due else 10**15
            cap[tid] = max(base_end[tid], safe)

    new_start: Dict[str, int] = dict(orig_start)
    new_end: Dict[str, int] = dict(base_end)
    for _m, items in by_machine.items():
        if any(info[a["task_id"]].get("is_pinned") for a in items):
            # immovable anchor on this machine — keep solver positions verbatim
            continue
        # Group the machine's tasks by order; order the groups by DUE DATE (EDD),
        # then by the order's earliest original start as a tie-breaker.
        #
        # Why EDD and not earliest-start: when an URGENT order (near due) was
        # interleaved into the middle of a SLACK order (far due) to hit its
        # deadline, an earliest-start ordering would try to pull the slack order's
        # whole footprint in front of the urgent one — pushing the urgent order
        # past its `cap[tid]` (due − downstream chain), which aborts the repack and
        # leaves the machine fragmented (measured: 2 fragmented machines stayed
        # split for exactly this reason).  EDD instead keeps the urgent order in its
        # early slot and dovetails the slack order contiguously AFTER it: the slack
        # order has huge due slack so delaying it is free → full contiguity with
        # equal-or-lower lateness (measured 2→0 re-entries, knit tardiness 1755→1466
        # on a real cold payload).  The pipeline-wide lateness gate still verifies.
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for a in items:
            oid = info[a["task_id"]].get("original_order_id") or a["task_id"]
            groups.setdefault(oid, []).append(a)
        order_keys = sorted(
            groups,
            key=lambda o: (
                min(int(info[x["task_id"]].get("due_at_min", 0) or 0) for x in groups[o]),
                min(x["start_time"] for x in groups[o]),
                o,
            ),
        )
        # Repack from the machine's original earliest start (do not shift left of it,
        # which would only add early-window concurrency pressure).
        prev_end = min(a["start_time"] for a in items)
        cand_s: Dict[str, int] = {}
        cand_e: Dict[str, int] = {}
        safe = True
        for oid in order_keys:
            for a in sorted(groups[oid], key=lambda x: (x["start_time"], x["task_id"])):
                t_id = a["task_id"]
                sa = int(info[t_id].get("start_after_min", 0) or 0)
                dur = int(a["end_time"]) - int(a["start_time"])
                ns = max(sa, prev_end)
                if ns + dur > cap[t_id]:
                    safe = False
                    break
                cand_s[t_id] = ns
                cand_e[t_id] = ns + dur
                prev_end = ns + dur
            if not safe:
                break
        if safe:
            new_start.update(cand_s)
            new_end.update(cand_e)
        # else: leave this machine's tasks at their baseline (already in new_*)

    knitting_tasks = [t for t in all_tasks
                      if t.get("operation", "").lower() in ("knitting", "capacity_block")]
    if not _knitting_workforce_ok(new_start, new_end, knitting_tasks, config):
        logger.info(
            "🧱 Knitting contiguity reorder: candidate exceeds workforce cap — skipped."
        )
        return None

    new_frag = _count_fragmented_orders(by_machine, info, lambda tid: new_start[tid])
    if new_frag >= base_frag:
        logger.info(
            f"🧱 Knitting contiguity reorder: no fragmentation gain "
            f"(base={base_frag}, candidate={new_frag}) — skipped."
        )
        return None

    logger.info(
        f"🧱 Knitting contiguity reorder: candidate cuts fragmented (order,machine) "
        f"pairs {base_frag}→{new_frag} — verifying downstream…"
    )
    return {"start": new_start, "end": new_end}


def _yarn_key(task: Dict[str, Any]) -> tuple:
    """Setup-comparison key for the yarn/creel state a knitting task needs.

    `color_config` is the Go-side YarnConfig string `SỢI:SỐ_CUỘN` (multi-yarn
    `A:2|B:1`) — two tasks with the same string need the SAME creel setup, so
    running them back-to-back costs no changeover.  Go sometimes sends a lookup
    error sentence instead of data ("No yarn requirements found for this design
    and color." — observed on 205/312 tasks of real payloads), and the machine
    field is sometimes a bare colour name; NEVER compare those raw (all broken
    tasks would "match" each other across colours).  A real config always
    contains ':', so anything without one falls back to (color, substance) —
    the same key washing compatibility uses.
    """
    cc = str(task.get("color_config") or "").strip()
    if ":" in cc:
        return ("cfg", cc)
    return ("fallback", str(task.get("color") or ""), str(task.get("substance") or ""))


def _yarn_reentries(keys: List[tuple]) -> int:
    """Number of extra runs: a yarn config appearing in ≥2 separate runs of this
    time-ordered key sequence means the machine was set up for it, switched away,
    and had to be re-set — each such return is one avoidable double changeover."""
    runs: List[tuple] = []
    for k in keys:
        if not runs or runs[-1] != k:
            runs.append(k)
    return len(runs) - len(set(runs))


def repair_yarn_config_reentry(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """COLD-only candidate: kill A→B→A yarn-config re-entries on knitting machines.

    Rule (user 2026-07-09): interleaving orders on a machine is fine ONLY between
    tasks with the same yarn config (`color_config`, SỢI:SỐ_CUỘN) — returning to a
    config the machine already left means tearing the creel down and setting it up
    again.  The solver has no objective term for adjacent-task config, so slack-due
    filler slices land after a different-config block while same-config machines
    sit idle (measured CP_1783583062535099757: 6 machines went :2 → :5 → :2 though
    5+ machines still in :2 state had finished — 12 avoidable creel changes).

    For each re-entry run (a maximal same-config run whose config already appeared
    earlier on that machine), relocate the WHOLE run — atomically, else not at all —
    appending each task to a compatible machine whose current tail has the SAME
    yarn config (or which is empty).  Appending after a matching tail can never
    create a new re-entry on the target.

    Guards (per task, before any commit):
      * pinned/in-progress runs are immovable;
      * new_end ≤ max(baseline end, due − downstream_chain_min) — a tight or late
        task is never pushed past its baseline (same cap as the contiguity reorder);
      * new_end ≤ the layout's knitting makespan — never extends the tail;
      * whole-candidate `_knitting_workforce_ok` + strict global re-entry decrease.

    NOT monotone (moved tasks start later), so the caller runs it as a relayout
    candidate: phases 2–5 are re-solved and it is accepted only if total lateness
    does not rise (`Pipeline._try_knitting_relayout`).  The source-machine hole is
    compacted by the left-shift that already follows an accepted relayout.
    Deterministic: runs by (machine, start); targets by (earliest start, machine).
    Returns {"start", "end", "machine"} over ALL knitting tasks, or None.
    """
    info = {t["task_id"]: t for t in all_tasks}
    knit = [a for a in assignments
            if str(info.get(a["task_id"], {}).get("operation", "")).lower() == "knitting"]
    if len(knit) < 2:
        return None

    key_of = {a["task_id"]: _yarn_key(info[a["task_id"]]) for a in knit}
    busy: Dict[str, List[List[Any]]] = {}
    for a in knit:
        busy.setdefault(a["machine_id"], []).append(
            [int(a["start_time"]), int(a["end_time"]), a["task_id"]]
        )
    for m in busy:
        busy[m].sort()

    def _machine_keys(m: str) -> List[tuple]:
        return [key_of[tid] for _, _, tid in busy[m]]

    base_reent = sum(_yarn_reentries(_machine_keys(m)) for m in busy)
    if base_reent == 0:
        return None

    # Per-task latest safe end — identical cap to reorder_contiguous_knitting: a
    # task at/past (due − chain) keeps its baseline end as the bound, so a tight
    # or late task never moves later; the pipeline-wide verify is the real gate.
    chain_min = _compute_downstream_chain_min(all_tasks)
    cap: Dict[str, int] = {}
    for a in knit:
        tid = a["task_id"]
        due = int(info[tid].get("due_at_min", 0) or 0)
        safe = (due - int(chain_min.get(tid, 0))) if due else 10**15
        cap[tid] = max(int(a["end_time"]), safe)
    knit_makespan = max(int(a["end_time"]) for a in knit)

    # Offending runs: maximal same-key runs whose key already ran earlier on the
    # machine.  Deterministic order: (machine, run start).
    offending: List[tuple] = []
    for m in sorted(busy):
        seen: set = set()
        run: List[List[Any]] = []
        for iv in busy[m] + [[None, None, None]]:  # sentinel flushes last run
            if run and (iv[2] is None or key_of[iv[2]] != key_of[run[0][2]]):
                k = key_of[run[0][2]]
                if k in seen:
                    offending.append((m, list(run)))
                seen.add(k)
                run = []
            if iv[2] is not None:
                run.append(iv)

    new_pos: Dict[str, tuple] = {}  # tid → (machine, start, end)
    for m, run in offending:
        if any(info[tid].get("is_pinned") for _, _, tid in run):
            continue
        staged: List[tuple] = []
        ok = True
        for s0, e0, tid in run:
            dur = e0 - s0
            rel = int(info[tid].get("start_after_min", 0) or 0)
            k = key_of[tid]
            compat = set(info[tid].get("compatible_resource_ids") or [])
            best = None  # (start, machine)
            for m2 in sorted(busy):
                if m2 == m or (compat and m2 not in compat):
                    continue
                tail = busy[m2][-1] if busy[m2] else None
                if tail is not None and key_of[tail[2]] != k:
                    continue  # different config at the tail → would just move the churn
                st = max(tail[1] if tail else 0, rel)
                if st + dur > cap[tid] or st + dur > knit_makespan:
                    continue
                if best is None or (st, m2) < best:
                    best = (st, m2)
            if best is None:
                ok = False
                break
            st, m2 = best
            busy[m2].append([st, st + dur, tid])  # later tasks of the run stack after
            staged.append((tid, m2, st, st + dur))
        if not ok:
            for tid, m2, st, en in staged:
                busy[m2].remove([st, en, tid])
            continue
        for tid, m2, st, en in staged:
            busy[m] = [iv for iv in busy[m] if iv[2] != tid]
            new_pos[tid] = (m2, st, en)

    if not new_pos:
        return None

    cand_reent = sum(_yarn_reentries(_machine_keys(m)) for m in busy)
    if cand_reent >= base_reent:
        return None

    new_start = {a["task_id"]: int(a["start_time"]) for a in knit}
    new_end = {a["task_id"]: int(a["end_time"]) for a in knit}
    new_machine = {a["task_id"]: a["machine_id"] for a in knit}
    for tid, (m2, st, en) in new_pos.items():
        new_start[tid], new_end[tid], new_machine[tid] = st, en, m2

    knitting_tasks = [t for t in all_tasks
                      if str(t.get("operation", "")).lower() in ("knitting", "capacity_block")]
    if not _knitting_workforce_ok(new_start, new_end, knitting_tasks, config):
        logger.info("🧶 Yarn-config repair: candidate exceeds workforce cap — skipped.")
        return None

    logger.info(
        f"🧶 Yarn-config repair: candidate moves {len(new_pos)} task(s) to same-config "
        f"machines, config re-entries {base_reent}→{cand_reent} — verifying downstream…"
    )
    return {"start": new_start, "end": new_end, "machine": new_machine}


def _panel_index(task_id: str) -> int:
    """Trailing _<int> of a knitting batch id (BATCH_0-641_7 → 7); 0 if none."""
    tail = task_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def parallelize_component_pos(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, int]]]:
    """COLD-only: when a garment's component POs (e.g. front 0-641 + back 0-642) are
    knit SERIALLY — every batch of one PO before the other starts — the first complete
    PANEL (one batch from EACH PO at the same index) isn't ready until the 2nd PO's
    first batch finishes, so linking workers idle through the whole first PO (measured:
    job CP_…841621, order WE3EwiOKOy, first panel ready 2590 though 641 batches finished
    from 914 — ~2 shift-days of linking idle).

    This DEDICATES a disjoint subset of the garment's machines to each component PO
    (split ∝ workload to balance makespan) and re-places each PO's batches on its own
    machines in panel-index order, so the POs knit IN PARALLEL: the first panel — and
    every middle panel — becomes ready far sooner, feeding linking early.  Each machine
    still runs ONE PO contiguously (no PO-switch churn; honours the contiguity
    preference — cf. [[project_po_setup_cost_parallel_panels]]).

    Returns ``{"start": {tid: s}, "end": {tid: e}}`` over ALL knitting tasks (moved +
    untouched), or ``None`` (no serialized multi-component garment, or it would breach
    the workforce cap).  NOT monotone — early batches of the first PO move LATER to free
    machines for the second PO — so the pipeline re-solves phases 2–5 and accepts ONLY
    if total lateness does not increase (see ``Pipeline._try_knitting_relayout``).
    Garments touching a pinned/in-progress knitting task are left untouched.
    Deterministic: garments by (due, id); POs by (−work, id); machines by earliest free.
    """
    info = {t["task_id"]: t for t in all_tasks}
    asg = {a["task_id"]: a for a in assignments}

    # 1. component PO (knitting group_id) → garment (linking order) it feeds + due.
    garment_pos: Dict[str, set] = {}
    garment_due: Dict[str, int] = {}
    for t in all_tasks:
        if t.get("operation", "").lower() != "linking":
            continue
        g = t.get("original_order_id") or t.get("parent_task_id") or t["task_id"]
        for dep in (t.get("final_depends_on") or []):
            kt = info.get(dep)
            if kt and kt.get("operation", "").lower() == "knitting" and kt.get("group_id"):
                garment_pos.setdefault(g, set()).add(kt["group_id"])
        if g in garment_pos:
            garment_due[g] = int(t.get("due_at_min", 0) or 0)
    multi = {g: pos for g, pos in garment_pos.items() if len(pos) >= 2}
    if not multi:
        return None

    # knitting tasks per PO (from the current assignment).
    po_tasks: Dict[str, List[str]] = {}
    for a in assignments:
        t = info.get(a["task_id"])
        if t is not None and t.get("operation", "").lower() == "knitting" and t.get("group_id"):
            po_tasks.setdefault(t["group_id"], []).append(a["task_id"])

    # 2. pick SERIALIZED garments: current first-panel (max over POs of the PO's
    #    earliest-finished batch) is later than the parallel-achievable first-panel
    #    (group start + longest single first batch) by more than one batch of idle.
    pinned_machines = {a["machine_id"] for a in assignments
                       if info.get(a["task_id"], {}).get("is_pinned")}
    affected_pos: set = set()
    target_garments: List[str] = []
    for g in sorted(multi, key=lambda x: (garment_due.get(x, 0), x)):
        pos = [p for p in multi[g] if po_tasks.get(p)]
        if len(pos) < 2:
            continue
        tids = [tid for p in pos for tid in po_tasks[p]]
        if any(info[tid].get("is_pinned") or asg[tid]["machine_id"] in pinned_machines
               for tid in tids):
            continue
        t0 = min(asg[tid]["start_time"] for tid in tids)
        cur_first = max(min(asg[tid]["end_time"] for tid in po_tasks[p]) for p in pos)
        par_first = t0 + max(
            min(asg[tid]["end_time"] - asg[tid]["start_time"] for tid in po_tasks[p])
            for p in pos
        )
        if cur_first - par_first <= par_first - t0:
            continue  # already overlapping / barely serialized
        target_garments.append(g)
        affected_pos |= set(pos)
    if not target_garments:
        return None

    # 3. Đặt lại theo vòng RESTART: chạy thử placement cho các garment đang active;
    #    garment nào re-pack làm knit-end MUỘN hơn gốc (cửa sổ gốc bị island đơn khác
    #    ăn bớt chỗ → batch tràn ra sau rất xa — CP_1783308395880305537: Wle9h8tXRA
    #    912→6345, tự lật LATE → gate reject cả 4 garment) hoặc không đặt được thì
    #    LOẠI, rồi chạy lại TOÀN BỘ placement từ đầu với interval gốc của garment bị
    #    loại giữ nguyên làm vật cản.  Khôi-phục-tại-chỗ không an toàn vì cửa sổ gốc
    #    các garment ĐAN XEN nhau: garment đặt trước có thể đã back-fill vào đúng chỗ
    #    garment sau trả lại (651_7 ↮ 652_1).  Thuần Python, không re-solve; hội tụ
    #    ≤ số garment vòng.
    MAXM = int(config.get("max_factory_machines", 100))
    excluded: set = set()
    new_start: Dict[str, int] = {}
    new_end: Dict[str, int] = {}
    new_machine: Dict[str, str] = {}
    placed_ok = False
    for _round in range(len(target_garments)):
        active = [g for g in target_garments if g not in excluded]
        if not active:
            break
        affected_tids = {tid for g in active for p in multi[g]
                         for tid in po_tasks.get(p, [])
                         if not info[tid].get("is_pinned")}
        if not affected_tids:
            break

        # timeline state — fixed busy windows + global workforce occupancy from
        # everything we do NOT re-place (machine unavailability, untouched knitting,
        # pinned knitting, capacity_block, POs of excluded garments), then place the
        # affected POs around them.
        avail_at: Dict[str, int] = {}
        fixed_busy: Dict[str, List[Any]] = {}
        occ: List[Any] = []
        for r in resources:
            mid = r["id"]
            avail_at[mid] = int(r.get("available_at_min", 0) or 0)
            fixed_busy[mid] = [(int(w["start"]), int(w["end"]))
                               for w in (r.get("unavailability") or [])
                               if int(w["end"]) > int(w["start"])]
        for t in all_tasks:
            op = t.get("operation", "").lower()
            tid = t["task_id"]
            if op == "capacity_block":
                dem = int(t.get("demand", 0) or 0)
                pe = t.get("pinned_end_time")
                if dem > 0 and pe is not None:
                    ps = t.get("pinned_start_time")
                    ps = int(ps) if ps is not None else int(pe) - int(t.get("duration", 0) or 0)
                    occ.append((ps, dem)); occ.append((int(pe), -dem))
                continue
            if op != "knitting" or tid in affected_tids:
                continue
            a = asg.get(tid)
            if t.get("is_pinned"):
                ps, pe = t.get("pinned_start_time"), t.get("pinned_end_time")
                if (ps is None or pe is None) and a:
                    ps, pe = a["start_time"], a["end_time"]
                m = t.get("pinned_machine_id") or (a["machine_id"] if a else None)
            elif a:
                ps, pe, m = a["start_time"], a["end_time"], a["machine_id"]
            else:
                continue
            if ps is None or pe is None or m is None:
                continue
            fixed_busy.setdefault(m, []).append((int(ps), int(pe)))
            occ.append((int(ps), 1)); occ.append((int(pe), -1))
        for m in fixed_busy:
            fixed_busy[m].sort()
        machine_tail: Dict[str, int] = dict(avail_at)

        def _occ_fits(s: int, e: int) -> bool:
            cur = 0
            for tm, dl in sorted(occ + [(s, 1), (e, -1)]):
                cur += dl
                if cur > MAXM:
                    return False
                if tm > e:
                    break
            return True

        def _next_release(t0: int) -> int:
            nxt = None
            for tm, dl in occ:
                if dl < 0 and tm > t0 and (nxt is None or tm < nxt):
                    nxt = tm
            return nxt if nxt is not None else t0 + 1

        def _machine_earliest(m: str, release: int, dur: int) -> int:
            # Quét GAP từ release (KHÔNG xuất phát từ machine_tail): các garment được
            # xử lý theo thứ tự due nhưng cửa sổ thời gian chúng nhả ra nằm theo thứ
            # tự t0 — nếu xuất phát từ tail (con trỏ chỉ tiến), garment due-sớm-nhưng-
            # t0-muộn đặt trước sẽ đẩy tail lên và các garment t0-sớm hơn KHÔNG
            # back-fill được vào chính cửa sổ mình vừa nhả (CP_1783308395880305537:
            # 649–651 bị văng 152→9841, lật LATE → gate reject cả 4 garment).  Vòng
            # bump-past-overlap tự hội tụ về gap sớm nhất đủ dur, nên bỏ tail là đủ
            # để back-fill.  (A1a: vòng bump = placement.bump_earliest, nguyên văn.)
            return bump_earliest(
                fixed_busy.get(m, []), max(release, avail_at.get(m, 0)), dur,
            )

        new_start = {}
        new_end = {}
        new_machine = {}
        bad: set = set()

        # Đặt garment theo thứ tự t0 (KHÔNG phải due): cửa sổ trống được giải phóng
        # theo trục thời gian; đặt theo due (W8jjpJWSUc due sớm nhưng t0=2432 đặt
        # trước) làm garment t0-sớm đặt sau spill batch qua island F0 tới 6193+ dù
        # cửa sổ 912.. của nó còn nguyên (CP_1783308395880305537).  Với t0-order,
        # spill nhỏ chỉ trượt sang mép cửa sổ garment kế → vẫn ≤ knit-end gốc.
        # Windows vốn rời nhau (đến từ layout nối tiếp) nên due chỉ còn là tie-break.
        def _g_t0(g_: str) -> int:
            return min((asg[t]["start_time"] for p in multi[g_]
                        for t in po_tasks.get(p, []) if t in affected_tids),
                       default=1 << 60)

        for g in sorted(active, key=lambda g_: (_g_t0(g_), garment_due.get(g_, 0), g_)):
            pos = [p for p in sorted(multi[g]) if any(t in affected_tids for t in po_tasks.get(p, []))]
            work = {p: sum(int(asg[t]["end_time"] - asg[t]["start_time"])
                           for t in po_tasks[p] if t in affected_tids) for p in pos}
            pos = [p for p in pos if work[p] > 0]
            if len(pos) < 2:
                # tid của garment này đã bị loại khỏi fixed scan nhưng không được
                # đặt lại → vòng sau phải coi nó là vật cản cố định.
                bad.add(g)
                continue
            comp = sorted({m for p in pos for t in po_tasks[p] if t in affected_tids
                           for m in (info[t].get("compatible_resource_ids") or [])
                           if m in avail_at})
            if len(comp) < len(pos):
                bad.add(g)
                continue
            t0 = min(asg[t]["start_time"] for p in pos for t in po_tasks[p] if t in affected_tids)
            # partition machines ∝ work (≥1 each); machines dealt earliest-free first.
            total = sum(work[p] for p in pos)
            nm = {p: max(1, round(work[p] / total * len(comp))) for p in pos}
            order_w = sorted(pos, key=lambda p: (-work[p], p))
            diff = len(comp) - sum(nm.values())
            i = 0
            while diff != 0 and order_w:
                p = order_w[i % len(order_w)]
                if diff > 0:
                    nm[p] += 1; diff -= 1
                elif nm[p] > 1:
                    nm[p] -= 1; diff += 1
                i += 1
            comp_sorted = sorted(comp, key=lambda m: (_machine_earliest(m, t0, 0), m))
            po_machines: Dict[str, List[str]] = {p: [] for p in pos}
            mi = 0
            for p in order_w:
                for _ in range(nm[p]):
                    if mi < len(comp_sorted):
                        po_machines[p].append(comp_sorted[mi]); mi += 1
            while mi < len(comp_sorted):
                po_machines[order_w[0]].append(comp_sorted[mi]); mi += 1

            # snapshot for per-garment rollback on failure.
            snap = (len(occ), {m: list(v) for m, v in fixed_busy.items()}, dict(machine_tail))
            local_s: Dict[str, int] = {}
            local_e: Dict[str, int] = {}
            local_m: Dict[str, str] = {}
            ok = True
            for p in pos:
                mset = po_machines[p]
                batches = sorted((t for t in po_tasks[p] if t in affected_tids),
                                 key=lambda t: (_panel_index(t), t))
                for tid in batches:
                    dur = int(asg[tid]["end_time"] - asg[tid]["start_time"])
                    release = max(t0, int(info[tid].get("start_after_min", 0) or 0))
                    cands = [m for m in mset
                             if m in (info[tid].get("compatible_resource_ids") or [])]
                    if not cands:
                        ok = False; break
                    m = min(cands, key=lambda mm: (_machine_earliest(mm, release, dur), mm))
                    st = _machine_earliest(m, release, dur)
                    guard = 0
                    while dur > 0 and not _occ_fits(st, st + dur):
                        st = bump_earliest(
                            fixed_busy.get(m, []), _next_release(st), dur,
                        )
                        guard += 1
                        if guard > 100000:
                            ok = False; break
                    if not ok:
                        break
                    if dur > 0:
                        occ.append((st, 1)); occ.append((st + dur, -1))
                    fixed_busy.setdefault(m, []).append((st, st + dur))
                    fixed_busy[m].sort()
                    # tail giờ chỉ là khoá sort máy (earliest-free) — back-fill có
                    # thể đặt TRƯỚC tail nên giữ max để khoá không tụt.
                    machine_tail[m] = max(machine_tail.get(m, 0), st + dur)
                    local_s[tid] = st; local_e[tid] = st + dur; local_m[tid] = m
                if not ok:
                    break
            # Re-pack chỉ có lợi khi first-panel sớm hơn mà knit-end của garment
            # KHÔNG muộn đi — garment bị kéo dài giữ layout nối tiếp gốc (loại rồi
            # restart).
            orig_last = max(int(asg[t]["end_time"])
                            for p in pos for t in po_tasks[p] if t in affected_tids)
            if ok and local_e and max(local_e.values()) > orig_last:
                logger.info(
                    f"⏸ Knitting parallel-PO: garment {g} re-pack kéo dài knit-end "
                    f"{orig_last}→{max(local_e.values())} → giữ layout gốc cho garment này."
                )
                ok = False
            if ok:
                new_start.update(local_s); new_end.update(local_e); new_machine.update(local_m)
            else:  # rollback partial placements; loại garment → restart vòng ngoài
                del occ[snap[0]:]
                fixed_busy.clear(); fixed_busy.update(snap[1])
                machine_tail.clear(); machine_tail.update(snap[2])
                bad.add(g)
        if not bad:
            placed_ok = True
            break
        excluded |= bad

    if not placed_ok or not new_start:
        return None

    # full knitting layout (moved + untouched) for the workforce check + p1b build.
    # Includes machine_id — the dedication CHANGES which machine runs each PO, so the
    # caller must apply machine too (else old-machine + new-time = overlap).
    full_start: Dict[str, int] = {}
    full_end: Dict[str, int] = {}
    full_machine: Dict[str, str] = {}
    for a in assignments:
        t = info.get(a["task_id"])
        if t is None or t.get("operation", "").lower() != "knitting":
            continue
        tid = a["task_id"]
        if tid in new_start:
            full_start[tid] = new_start[tid]; full_end[tid] = new_end[tid]
            full_machine[tid] = new_machine[tid]
        else:
            full_start[tid] = a["start_time"]; full_end[tid] = a["end_time"]
            full_machine[tid] = a["machine_id"]

    knitting_tasks = [t for t in all_tasks
                      if t.get("operation", "").lower() in ("knitting", "capacity_block")]
    if not _knitting_workforce_ok(full_start, full_end, knitting_tasks, config):
        logger.info("⏸ Knitting parallel-PO: candidate exceeds workforce cap — skipped.")
        return None

    # Sanity: không task nào đè nhau trên cùng máy trong layout đầy đủ (moved +
    # untouched).  Back-fill + rollback-khôi-phục có thể va nhau ở ca hiếm (garment
    # đặt trước chiếm vào cửa sổ của garment sau rồi garment sau rollback); một
    # layout đè nhau gửi sang Go là hỏng lịch → thà giữ solver plan.
    _by_m: Dict[str, List[Tuple[int, int, str]]] = {}
    for tid, s in full_start.items():
        e = full_end[tid]
        if e > s:
            _by_m.setdefault(full_machine[tid], []).append((s, e, tid))
    for _m, _arr in _by_m.items():
        _arr.sort()
        for _i in range(1, len(_arr)):
            if _arr[_i][0] < _arr[_i - 1][1]:
                logger.info(
                    f"⏸ Knitting parallel-PO: candidate has overlap on {_m} "
                    f"({_arr[_i - 1][2]} {_arr[_i - 1][:2]} ↮ {_arr[_i][2]} {_arr[_i][:2]}) — skipped."
                )
                return None

    logger.info(
        f"⏸ Knitting parallel-PO: dedicated machines to component POs of "
        f"{len(target_garments) - len(excluded)}/{len(target_garments)} garment(s), "
        f"{len(new_start)} batch(es) re-placed in parallel — verifying downstream…"
    )
    return {"start": full_start, "end": full_end, "machine": full_machine}


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

    for mat_code, capacity in sorted(material_capacities.items()):
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
    contiguity_w: int = 0,
) -> List[Any]:
    """Per-(order, machine) bounding box.  When contiguity_w > 0, also returns a
    SOFT penalty on each order's per-machine footprint span (po_end − po_start) so
    the solver finishes one order before starting another on a machine ("dứt điểm
    đơn đó") instead of interleaving — bosses prefer whole-order completion even
    when nothing is late.  Soft (not the old hard span==Σdur equality), so a shift
    break that legitimately splits an order still yields rather than going INFEASIBLE.
    """
    terms: List[Any] = []
    # Only group FREE (non-pinned) knitting tasks.
    # Pinned tasks may have pinned_start_time < 0 (in-progress), which would force
    # po_start ≤ negative value while po_start ∈ [0, horizon] → instant INFEASIBLE.
    po_groups: Dict[str, List[Dict]] = {}
    for t in tasks:
        if t.get("operation", "").lower() == "knitting" and not t.get("is_pinned", False):
            po_id = t.get("original_order_id")
            if po_id:
                po_groups.setdefault(po_id, []).append(t)

    for po_id, po_tasks in sorted(po_groups.items()):
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
            #
            # SOFT contiguity: penalise this order's footprint span on this machine.
            # Interleaving another order in the middle stretches po_end−po_start, so
            # minimising it makes the solver finish the order contiguously (or move it
            # off this machine) — fragmentation is otherwise objective-NEUTRAL (the
            # solver picks an interleaved layout arbitrarily, measured: gap→0 unchanged).
            # When inactive on this machine po_start/po_end collapse equal → 0 penalty.
            if contiguity_w > 0:
                span = model.NewIntVar(0, horizon, f"po_{po_id}_{r_id}_span")
                model.Add(span == po_end - po_start)
                terms.append(span * contiguity_w)

    return terms


def _apply_po_setup_cost(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    resource_map: Dict[str, Dict[str, Any]],
    setup_w: int,
) -> List[Any]:
    """SOFT setup-change penalty: each EXTRA PO assigned to a machine costs `setup_w`.

    Switching a knitting machine between POs costs real (unmodeled) setup time at the
    factory, so a machine hosting many POs is expensive.  For every (PO, machine) we
    reify ``po_on_m`` = "this PO has at least one task on this machine" (OR of the
    task→machine literals); per machine we then penalise ``max(0, Σ_PO po_on_m − 1)``
    — i.e. the first PO is free, every additional PO on the machine pays one setup.
    Minimising this drives the solver to DEDICATE machines to POs, so same-panel
    component POs run on disjoint machines in parallel (panel ready sooner) with no
    ping-pong.  Banded below lateness (caller scales `setup_w`) so it never makes an
    order late.  Returns objective terms.  Free (non-pinned) knitting tasks only.
    """
    if setup_w <= 0:
        return []
    po_groups: Dict[str, List[Dict]] = {}
    for t in tasks:
        if t.get("operation", "").lower() == "knitting" and not t.get("is_pinned", False):
            po_id = t.get("original_order_id")
            if po_id:
                po_groups.setdefault(po_id, []).append(t)

    machine_pos: Dict[str, List[Any]] = {r_id: [] for r_id in resource_map}
    for po_id, po_tasks in sorted(po_groups.items()):
        for r_id in resource_map:
            lits = []
            for t in po_tasks:
                tv = task_vars.get(t["task_id"])
                if tv is None:
                    continue
                lit = next(
                    (l for l in tv["literals"] if l.Name().endswith(f"_on_{r_id}")),
                    None,
                )
                if lit is not None:
                    lits.append(lit)
            if not lits:
                continue
            po_on_m = model.NewBoolVar(f"setup_{po_id}_{r_id}")
            model.AddMaxEquality(po_on_m, lits)  # True iff this PO touches r_id
            machine_pos[r_id].append(po_on_m)

    terms: List[Any] = []
    for r_id, actives in sorted(machine_pos.items()):
        if len(actives) <= 1:
            continue  # ≤1 candidate PO → never an extra setup
        extra = model.NewIntVar(0, len(actives), f"setup_extra_{r_id}")
        model.Add(extra >= sum(actives) - 1)  # extra ≥ 0 by domain ⇒ = max(0, Σ−1)
        terms.append(extra * setup_w)
    return terms
