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
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)

# On a re-schedule after convert, machine_materials is still empty (yarn not
# physically loaded), so Go cannot populate resources[].color_config. The
# converted order's PINNED task carries the machine's committed yarn instead.
# When enabled, build_resource_model derives each knitting machine's current
# config from its last pinned task so affinity can pull same-yarn new orders
# onto it. Enabled by default; CP_DERIVE_MACHINE_CONFIG=0 disables.
_DERIVE_MACHINE_CONFIG = os.getenv("CP_DERIVE_MACHINE_CONFIG", "1") != "0"

# Default deterministic-time budget per CP-SAT solve when config omits
# `max_deterministic_time`.  Deterministic units, NOT wall seconds — on hard
# models det-time advances much slower than wall, so keep this modest.  This is
# the primary speed/quality knob: lower = faster (stops earlier at the same
# FEASIBLE, still byte-reproducible), higher = more optimisation.  Override per
# run via config `max_deterministic_time`.
#
# Lowered 30→12 (2026-06): the knitting model is disjunctive (no-overlap +
# optional machine-assignment + cumulative workforce) → its LP lower bound is
# frozen at the root (gap 40–84% even on a 16-task instance), so the solver
# NEVER reaches OPTIMAL no matter the budget — it just burns time chasing an
# unprovable bound.  Measured: incumbent quality hits the diminishing-returns
# knee (~5% of the final FEASIBLE objective) around det≈12 on a 60-task wave,
# while det=30 costs ~2.5× the wall time for a <5% incumbent gain.  12 is the
# speed/quality sweet spot; raise per-run via config if a payload needs it.
DEFAULT_MAX_DET_TIME: float = 12.0

CAPACITY_BLOCK_OPS = frozenset({"capacity_block", "capacity_block_linking"})


def is_capacity_block_op(operation: str) -> bool:
    return (operation or "").lower() in CAPACITY_BLOCK_OPS


# ---------------------------------------------------------------------------
# Re-schedule stability — apply_stability_objective + StabilityStats
# ---------------------------------------------------------------------------


@dataclass
class StabilityStats:
    """Per-phase reporting for stability-hint application (used by tests + logs).

    Fields are integer counters so assertions don't depend on float noise.
    """
    total_previous: int = 0
    matched_exact: int = 0
    matched_via_order: int = 0
    n_hinted: int = 0
    time_terms_added: int = 0
    machine_terms_added: int = 0


def apply_stability_objective(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    reschedule_hint: Optional[Dict[str, Any]],
    horizon: int,
    start_lb: Optional[Dict[str, int]] = None,
    time_penalty: str = "abs",
) -> Tuple[List[Any], StabilityStats]:
    """
    Add warm-start hints + minimum-perturbation penalty terms to the model.

    Semantics
    ---------
    For each non-pinned task that matches a previous assignment:
      * `AddHint(start, clipped_prev_start)` — clipped to [max(0, start_lb[t_id]), horizon].
      * `AddHint(lit, 1)` for the literal of the previous machine; `AddHint(lit, 0)` for others.
      * Penalty: `|start - prev_start| * w_time` (EXACT-matched tasks only; fallback-matched
        tasks SKIP this term — see FIX-3, B.2 ngữ nghĩa).
      * Penalty: `lit_machine_other * w_machine` for every literal on a machine ≠ prev_machine.

    Pinned tasks are skipped entirely (they have no choice variables).
    Hint is dropped silently when reschedule_hint is None or empty.

    Returns
    -------
    (penalty_terms, StabilityStats) — caller sums `penalty_terms` into the
    phase's `model.Minimize(...)`.
    """
    stats = StabilityStats()
    if not reschedule_hint:
        return [], stats

    previous = reschedule_hint.get("previous_assignments") or []
    if not previous:
        return [], stats

    stats.total_previous = len(previous)

    w_time = int(reschedule_hint.get("stability_weight_time_per_min", 500))
    w_machine = int(reschedule_hint.get("stability_weight_machine_swap", 5000))
    fallback_enabled = bool(reschedule_hint.get("match_by_order_fallback", True))
    start_lb = start_lb or {}

    # Index previous by exact task_id AND a LIST per original_order_id (for fallback).
    # An order spans multiple operations (knitting + linking), so per-task fallback
    # must pick the prev whose machine is compatible with the current task — not
    # just "first prev in this order".
    prev_by_taskid: Dict[str, Dict[str, Any]] = {p["task_id"]: p for p in previous}
    prev_by_order: Dict[str, List[Dict[str, Any]]] = {}
    for p in previous:
        oid = p.get("original_order_id", "")
        if oid:
            prev_by_order.setdefault(oid, []).append(p)

    terms: List[Any] = []

    # Build task_info index (we also accept callers passing already-filtered tasks)
    task_info_by_id: Dict[str, Dict[str, Any]] = {t["task_id"]: t for t in tasks}

    for t_id, tv in task_vars.items():
        if tv.get("is_pinned"):
            continue

        # Match: exact first, then fallback by order
        match_kind: Optional[str] = None
        prev: Optional[Dict[str, Any]] = None
        if t_id in prev_by_taskid:
            prev = prev_by_taskid[t_id]
            match_kind = "exact"
        elif fallback_enabled:
            order_id = (
                task_info_by_id.get(t_id, {}).get("original_order_id", "")
                or tv.get("original_order_id", "")
            )
            if order_id and order_id in prev_by_order:
                compatible = set(tv.get("r_ids") or [])
                for cand in prev_by_order[order_id]:
                    if cand.get("machine_id") in compatible:
                        prev = cand
                        match_kind = "order"
                        break

        if prev is None or match_kind is None:
            continue

        # ── Counters & hints ─────────────────────────────────────────────────
        if match_kind == "exact":
            stats.matched_exact += 1
        else:
            stats.matched_via_order += 1
        stats.n_hinted += 1

        # Clip prev_start into the task's feasible window.
        prev_start = int(prev.get("start_time", 0))
        lo = max(0, int(start_lb.get(t_id, 0)))
        prev_start_clipped = max(lo, min(horizon, prev_start))
        # On a workload shrink (late_only) warm-start at the EARLIEST feasible
        # time so the solver is nudged to compact instead of re-discovering the
        # old (now gapped) layout — important for large groups that stop at
        # FEASIBLE before they would otherwise improve past the stale hint.
        model.AddHint(
            tv["start"], lo if time_penalty == "late_only" else prev_start_clipped
        )

        # Machine hints + machine-swap penalty terms
        prev_machine = prev.get("machine_id", "")
        literals = tv.get("literals", []) or []
        r_ids = tv.get("r_ids", []) or []
        for lit, r_id in zip(literals, r_ids):
            if r_id == prev_machine:
                model.AddHint(lit, 1)
            else:
                model.AddHint(lit, 0)
                if w_machine > 0:
                    terms.append(lit * w_machine)
                    stats.machine_terms_added += 1

        # Time-deviation penalty — EXACT match only.
        # Fallback-match (slicing rename, qty change) hints machine but skips
        # the time term to avoid pulling N renamed slices toward a single
        # prev_start (would conflict with no-overlap, see FIX-3 in C5 revised).
        #
        # time_penalty mode:
        #   "abs"       — phạt |start - prev| đối xứng (mặc định, giữ ổn định 2 chiều).
        #   "late_only" — CHỈ phạt khi dời MUỘN hơn: max(0, start - prev).  Cho phép
        #                 dồn SỚM miễn phí.  Dùng khi workload co lại (bớt đơn) để
        #                 các task còn lại nén về đầu thay vì đóng băng → hở gap.
        #   "off"       — bỏ hẳn time term.
        if match_kind == "exact" and w_time > 0 and time_penalty != "off":
            if time_penalty == "late_only":
                late = model.NewIntVar(0, horizon, f"slate_{t_id}")
                model.AddMaxEquality(late, [tv["start"] - prev_start_clipped, 0])
                terms.append(late * w_time)
            else:
                dev = model.NewIntVar(0, horizon, f"sdev_{t_id}")
                model.AddAbsEquality(dev, tv["start"] - prev_start_clipped)
                terms.append(dev * w_time)
            stats.time_terms_added += 1

    return terms, stats


def apply_stability_hints_only(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    reschedule_hint: Optional[Dict[str, Any]],
    horizon: int,
    start_lb: Optional[Dict[str, int]] = None,
    prefer_earliest: bool = False,
) -> Tuple[List[Any], StabilityStats]:
    """Hints-ONLY variant of apply_stability_objective for the knitting phase.

    Adds the same `AddHint` warm-start calls as `apply_stability_objective`
    (start IntVar + machine literals) but contributes ZERO objective terms.
    Used by phase 1 when reified-keep is active, so the soft time-dev / machine-
    swap penalties do NOT double-stabilize alongside the hard keep constraint.

    Returns ([], StabilityStats) for API parity with apply_stability_objective.
    Callers MUST NOT sum the returned terms (empty by contract).
    """
    stats = StabilityStats()
    if not reschedule_hint:
        return [], stats

    previous = reschedule_hint.get("previous_assignments") or []
    if not previous:
        return [], stats

    stats.total_previous = len(previous)
    fallback_enabled = bool(reschedule_hint.get("match_by_order_fallback", True))
    start_lb = start_lb or {}

    prev_by_taskid: Dict[str, Dict[str, Any]] = {p["task_id"]: p for p in previous}
    prev_by_order: Dict[str, List[Dict[str, Any]]] = {}
    for p in previous:
        oid = p.get("original_order_id", "")
        if oid:
            prev_by_order.setdefault(oid, []).append(p)

    task_info_by_id: Dict[str, Dict[str, Any]] = {t["task_id"]: t for t in tasks}

    for t_id, tv in task_vars.items():
        if tv.get("is_pinned"):
            continue

        match_kind: Optional[str] = None
        prev: Optional[Dict[str, Any]] = None
        if t_id in prev_by_taskid:
            prev = prev_by_taskid[t_id]
            match_kind = "exact"
        elif fallback_enabled:
            order_id = (
                task_info_by_id.get(t_id, {}).get("original_order_id", "")
                or tv.get("original_order_id", "")
            )
            if order_id and order_id in prev_by_order:
                compatible = set(tv.get("r_ids") or [])
                for cand in prev_by_order[order_id]:
                    if cand.get("machine_id") in compatible:
                        prev = cand
                        match_kind = "order"
                        break

        if prev is None or match_kind is None:
            continue

        if match_kind == "exact":
            stats.matched_exact += 1
        else:
            stats.matched_via_order += 1
        stats.n_hinted += 1

        prev_start = int(prev.get("start_time", 0))
        lo = max(0, int(start_lb.get(t_id, 0)))
        prev_start_clipped = max(lo, min(horizon, prev_start))
        # prefer_earliest (workload shrank) → warm-start at earliest feasible so
        # knitting re-packs forward instead of re-anchoring to the stale start.
        model.AddHint(tv["start"], lo if prefer_earliest else prev_start_clipped)

        prev_machine = prev.get("machine_id", "")
        literals = tv.get("literals", []) or []
        r_ids = tv.get("r_ids", []) or []
        for lit, r_id in zip(literals, r_ids):
            model.AddHint(lit, 1 if r_id == prev_machine else 0)

    return [], stats


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
        if not is_capacity_block_op(t.get("operation", ""))
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
        if is_capacity_block_op(op):
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


def make_solver(
    config: Dict[str, Any],
    *,
    has_hint: bool = False,
    relative_gap: Optional[float] = None,
) -> cp_model.CpSolver:
    """
    Configured CpSolver factory.

    Determinism strategy (revised after empirical measurement — supersedes D6).
    Reproducible "same input → same output" requires TWO things together, because
    there are two independent sources of run-to-run drift:

      1. Single-worker search.  CP-SAT with `num_search_workers > 1` shares bounds
         and learned clauses between worker threads by WALL-CLOCK timing, so the
         result varies between runs even with a fixed seed + `max_deterministic_time`
         — measured directly on the 850-task payload (155 vs 129 late orders across
         runs at 8 workers; byte-identical at 1 worker).  So we FORCE 1 worker.
         `interleave_search=True` is deterministic but far too slow here (INFEASIBLE
         within budget), so it is not used.
      2. `PYTHONHASHSEED=0` (set in the Dockerfile) so set/dict iteration — and thus
         the order variables/constraints are added to the model — is identical across
         processes.  Without it, even 1 worker is non-deterministic because the MODEL
         itself differs run-to-run.  make_solver cannot fix this (it is a process-start
         env var); it is documented here because the two fixes are a matched pair.

    `max_deterministic_time` + fixed `random_seed` are the ONLY stop criterion for
    EVERY phase (cold /solve and re-schedule alike).  `max_time_in_seconds` is
    deliberately NOT set: a wall-clock stop is non-deterministic (it fires at a
    machine-speed-dependent search node), so even as a "safety cap" it would
    reintroduce run-to-run drift whenever it bound under load.  Deterministic time
    always advances, so it guarantees termination on its own — no wall cap needed.
    Each phase feeds its end_times into the next phase's start_lb, so one
    non-deterministic phase destabilises everything downstream — the deterministic-
    only stop must hold for all phases.

    `has_hint` no longer changes the stop criterion (kept for API compatibility and
    call-site readability).

    Tuning notes:
      * `max_deterministic_time` (deterministic units — NOT wall seconds; on hard
        models det-time can advance ~10-15× slower than wall, so a budget of 120
        units was observed to take ~1763s wall on a 386-var knitting phase) is the
        ONLY knob.  An explicit config value always wins.  When omitted it defaults
        to `DEFAULT_MAX_DET_TIME` (NOT `max_search_time` — that field is a Go-side
        wall hint we no longer use as the budget; deriving the det budget from it
        is what made runs balloon to ~30 min).
      * Determinism needs the budget to be the SAME every run, NOT large.  Lower
        `max_deterministic_time` for speed (it stops earlier at the same FEASIBLE,
        still byte-reproducible); raise it only if a phase needs more time to
        converge.  Tune via config `max_deterministic_time`.
      * Do NOT set `max_time_in_seconds` (stays at the CP-SAT default of +inf) — a
        finite wall cap is non-deterministic.
      * Do NOT raise `num_search_workers` (breaks determinism).
    """
    solver = cp_model.CpSolver()

    # Determinism requires single-worker search (see docstring §1).  We force it
    # regardless of the caller's num_search_workers so /solve and /re-schedule are
    # both reproducible.  Empirically only ~12% slower than 8 workers on the
    # production payload, with equal-or-better lateness.
    effective_workers = 1

    # max_deterministic_time is the ONLY stop criterion for ALL phases.  Honor an
    # explicit config value; otherwise use DEFAULT_MAX_DET_TIME — capped by any
    # smaller max_search_time the caller sent so tiny test payloads stay quick.
    # max_time_in_seconds is intentionally left unset (CP-SAT default +inf): a
    # wall-clock stop would be non-deterministic.
    det_budget = config.get("max_deterministic_time")
    if det_budget is None:
        wall_hint = config.get("max_search_time")
        det_budget = (
            min(float(wall_hint), DEFAULT_MAX_DET_TIME)
            if wall_hint is not None else DEFAULT_MAX_DET_TIME
        )
    solver.parameters.max_deterministic_time = float(det_budget)

    # Default 1% gap keeps the expensive knitting phase fast (it stops as soon as
    # it is provably within 1% of optimum).  Small/cheap phases that must close a
    # SUB-1% improvement — e.g. load-balancing interchangeable downstream machines,
    # where spreading saves <1% of an objective dominated by absolute start-times —
    # pass a tighter `relative_gap` so that gain is not swallowed by the tolerance.
    solver.parameters.relative_gap_limit = (
        0.01 if relative_gap is None else float(relative_gap)
    )
    solver.parameters.num_search_workers = effective_workers
    solver.parameters.random_seed = int(config.get("random_seed", 42))

    # NOTE: `repair_hint=True` triggers `Check failed: heuristics.fixed_search != nullptr`
    # SIGABRT on CP-SAT 9.8+ unless `search_branching=FIXED_SEARCH` is also set.
    # AddHint() warm-start + reified-keep on knitting + soft penalty on other phases
    # are sufficient; repair_hint is an optional accelerator that crashes here.
    return solver


# ---------------------------------------------------------------------------
# Affinity
# ---------------------------------------------------------------------------

def hydrate_machine_affinity_from_pins(
    tasks: List[Dict[str, Any]],
    resource_map: Dict[str, Dict[str, Any]],
) -> int:
    """Suy trạng thái sợi/thiết kế HIỆN TẠI của mỗi máy knit từ task PINNED cuối cùng.

    Trên reschedule, Go gửi `color_config` TRÊN pinned task nhưng để TRỐNG
    `resources[].color_config` — mà `compute_affinity_penalty` chỉ đọc chỗ sau.  Không
    có nó, affinity coi MỌI máy là "cold" như nhau (cùng cold-start penalty) → đơn nối
    tiếp xếp theo máy rảnh sớm nhất, đổi sợi thừa (đo thực tế: 3 đổi sợi lẽ ra 1).

    Lấy `color_config`/`design_item_id` của pinned task có `pinned_end_time` LỚN NHẤT
    trên mỗi máy làm trạng thái đi vào lúc nối đơn, ghi vào resource_map — CHỈ cho máy
    KNITTING và CHỈ khi resource chưa có config (không đè config thật Go gửi, vd máy
    đang chạy dở).  Nguồn là pinned task có color_config (chỉ knit mới mang sợi).  Trả
    về số máy đã hydrate.  Idempotent.
    """
    latest: Dict[str, Tuple[int, str, str]] = {}
    for t in tasks:
        if not t.get("is_pinned"):
            continue
        cfg = t.get("color_config") or ""
        m = t.get("pinned_machine_id")
        if not cfg or not m:
            continue
        pe = t.get("pinned_end_time")
        pe = int(pe) if pe is not None else 0
        if m not in latest or pe > latest[m][0]:
            latest[m] = (pe, cfg, t.get("design_item_id") or "")

    n = 0
    for m, (_, cfg, design) in latest.items():
        r = resource_map.get(m)
        if r is None or str(r.get("operation", "")).lower() != "knitting":
            continue
        if not r.get("color_config"):
            r["color_config"] = cfg
            if design and not r.get("design_item_id"):
                r["design_item_id"] = design
            n += 1
    return n


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


def seed_resource_config_from_pins(
    tasks: List[Dict[str, Any]], resource_map: Dict[str, Dict[str, Any]]
) -> int:
    """Fill each machine's current color_config / design_item_id from the LAST
    pinned task on it (max pinned_end_time), but only when the resource itself
    carries none.

    Why: on a re-schedule after convert, machine_materials is empty so
    resources[].color_config arrives blank for every machine. Without a current
    config, compute_affinity_penalty scores every machine as cold-start
    (identical penalty), so a new order is NOT pulled onto the machine whose
    converted (pinned) task already commits it to the same yarn. The pinned task
    now carries the real yarn config (Go side), so the machine's committed
    config is exactly its last pinned task.

    Only fills EMPTY resource configs → a real Go-sent config is never
    overwritten, and a cold solve (no pins) is a no-op. Returns the number of
    resources whose color_config was filled (for logging).
    """
    best: Dict[str, Tuple[int, str, str]] = {}
    for t in tasks:
        if not t.get("is_pinned"):
            continue
        machine = t.get("pinned_machine_id")
        if not machine:
            continue
        cc = str(t.get("color_config") or "").strip()
        dii = str(t.get("design_item_id") or "").strip()
        if not cc and not dii:
            continue
        end_raw = t.get("pinned_end_time")
        end = int(end_raw) if end_raw is not None else 0
        prev = best.get(machine)
        if prev is None or end >= prev[0]:
            best[machine] = (end, cc, dii)

    filled = 0
    for machine, (_, cc, dii) in best.items():
        res = resource_map.get(machine)
        if res is None:
            continue
        if cc and not str(res.get("color_config") or "").strip():
            res["color_config"] = cc
            filled += 1
        if dii and not str(res.get("design_item_id") or "").strip():
            res["design_item_id"] = dii
    return filled


# ---------------------------------------------------------------------------
# Pinned-window normalisation
# ---------------------------------------------------------------------------

def normalize_pinned_window(task: Dict[str, Any]) -> None:
    """Fill in a partially-specified pin so it can be fully anchored.

    An in-progress task may arrive pinned with only ONE endpoint set — in
    production the Go side sends `pinned_end_time` with `pinned_start_time=None`
    for work already running.  `is_fully_pinned` (build_resource_model) and the
    washing committed-window reservation (phase3) both require BOTH endpoints, so
    a one-sided pin is treated as time-free: the solver floats it off its real
    machine slot and a new task gets booked on top of it (overlap).

    Infer the missing endpoint from `duration` (the segment length — verified
    against fully-specified pins where pinned_end - pinned_start == duration), so
    the task becomes fully pinned at [origin, end] and its slot is reserved.
    Mutates the task in place; idempotent; a no-op for free tasks, fully-specified
    pins, or non-positive duration (cannot infer a real window).
    """
    if not task.get("is_pinned"):
        return
    ps = task.get("pinned_start_time")
    pe = task.get("pinned_end_time")
    if (ps is None) == (pe is None):
        return  # both set (already full) or both None (no anchor possible)
    dur = int(task.get("duration", 0) or 0)
    if dur <= 0:
        return  # cannot infer a window without a positive duration
    if ps is None:
        task["pinned_start_time"] = int(pe) - dur
    else:
        task["pinned_end_time"] = int(ps) + dur


# ---------------------------------------------------------------------------
# Monotone end caps (Pareto guard for two-pass refinement solves)
# ---------------------------------------------------------------------------

def apply_end_caps(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    end_caps: Dict[str, int],
) -> int:
    """Cap mỗi task: end ≤ end của pass trước (Pareto guard).

    Pass 2 của một solve tinh-chỉnh (vd. nới floor linking theo same-qty) chỉ
    được phép ra lịch điểm-theo-điểm KHÔNG MUỘN HƠN pass 1 — nhờ đó không task
    nào (và không đơn nào) có thể trễ hơn baseline, theo cấu trúc chứ không phải
    may rủi.  Nghiệm pass 1 luôn thỏa mọi cap nên model vẫn khả thi.

    Pinned tasks bị bỏ qua: end của chúng là hằng số giống hệt giữa hai pass;
    thêm cap chỉ thừa, và một pin lệch (dữ liệu xấu) sẽ làm cả model INFEASIBLE.

    Returns: số cap đã thêm.
    """
    n = 0
    for t_id, cap in end_caps.items():
        tv = task_vars.get(t_id)
        if tv is None or tv.get("is_pinned"):
            continue
        model.Add(tv["end"] <= int(cap))
        n += 1
    return n


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

    # ── Anchor partially-specified pins ─────────────────────────────────────
    # An in-progress task may be pinned with only one endpoint (e.g. Go sends
    # pinned_end_time with pinned_start_time=None).  Infer the missing endpoint
    # from duration so it is treated as fully pinned below and its machine slot
    # is reserved in NoOverlap/Cumulative — otherwise the solver floats it off
    # its real slot and a new task gets booked on top of it.
    for t in tasks:
        normalize_pinned_window(t)

    # ── Derive machine current config from pins (re-schedule affinity) ──────
    # Blank resources[].color_config on a post-convert re-schedule would make
    # every machine look cold-start to affinity; seed each machine's config
    # from its last pinned task so same-yarn new orders are pulled onto the
    # machine already committed to that yarn. Only when affinity is active
    # (knitting) and only fills empty configs.
    if use_affinity and _DERIVE_MACHINE_CONFIG:
        filled = seed_resource_config_from_pins(tasks, resource_map)
        if filled:
            logger.info(
                f"   🧶 Affinity: seeded current yarn config for {filled} machine(s) from pinned tasks"
            )

    # ── Step 1: create start/end IntVars ────────────────────────────────────
    for t in tasks:
        t_id = t["task_id"]
        compatible_ids = t.get("compatible_resource_ids", [])
        operation = t.get("operation", "").lower()

        if not is_capacity_block_op(operation) and not compatible_ids:
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
            if not is_capacity_block_op(operation):
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
        if is_capacity_block_op(operation):
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
    deadline_override: Optional[Dict[str, int]] = None,
) -> List[Any]:
    """
    Create a lateness IntVar (= max(0, end - due_at)) per non-pinned, non-dummy task.
    Returns weighted objective terms to be summed into model.Minimize().

    deadline_override: optional task_id → deadline (minutes).  When provided for a
    task, lateness is measured against this value instead of its raw due_at_min.
    Phase 3 passes the lead-adjusted washing deadline (due − downstream_lead) here,
    because washing is mid-pipeline: finishing washing at due_at_min already leaves
    no time for the ironing/packing that follow.  Other callers pass nothing, so
    their behaviour is unchanged.

    Objective hierarchy per task (descending magnitude):
      1. lateness × weight × 100   — minimise total tardiness (primary)
      2. is_late × weight × 10     — minimise NUMBER of late tasks at equal
                                     total tardiness (tie-breaker).  Without
                                     this, two solutions with the same total
                                     minutes-late but different distributions
                                     (e.g. 1 task × 100min vs 10 × 10min)
                                     are tied → multi-worker race picks
                                     different distributions across runs →
                                     "lúc trễ đơn, lúc không trễ" symptom.
      3. start × weight // 100     — prefer earlier starts (light tie-breaker)
    """
    terms: List[Any] = []
    for t_id, tv in sorted(task_vars.items()):
        task = task_map.get(t_id, {})
        if task.get("is_pinned", False):
            continue
        if is_capacity_block_op(task.get("operation", "")):
            continue

        priority = int(task.get("priority", 5))
        is_normal = bool(task.get("is_normal", False))
        due_at = int(task.get("due_at_min", horizon))
        if deadline_override and t_id in deadline_override:
            due_at = int(deadline_override[t_id])
        weight = 10 ** (6 - priority)

        max_lateness = max(0, horizon - due_at)
        lateness = model.NewIntVar(0, max_lateness, f"lat_{t_id}")
        model.Add(lateness >= tv["end"] - due_at)
        terms.append(lateness * weight * 100)

        # Late-count tie-breaker: BoolVar is_late ⇔ lateness > 0.
        # Weight chosen as `weight × 10` so:
        #   * 1 task flipping late→on-time saves   weight × 10
        #   * 1 minute of lateness reduction saves weight × 100 (10× more)
        #   → reducing total tardiness still dominates reducing count.
        # Skipped when max_lateness == 0 (due past horizon → cannot be late).
        # SKIPPED for normal orders (is_normal): the late-count term is what makes the
        # solver fight hard to flip a task on-time (and sacrifice slack from other
        # orders to do it).  Normal orders still carry the lateness-minutes term above
        # (×100) so they keep aiming at dueDate, but without the count pressure they no
        # longer compete aggressively for capacity — "tuân thủ dueDate, không khắt khe".
        if max_lateness > 0 and not is_normal:
            is_late = model.NewBoolVar(f"is_late_{t_id}")
            model.Add(lateness >= 1).OnlyEnforceIf(is_late)
            model.Add(lateness == 0).OnlyEnforceIf(is_late.Not())
            terms.append(is_late * weight * 10)

        # Earliest-start tie-breaker (light)
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
        if gid and t_id in task_vars and not is_capacity_block_op(t.get("operation", "")):
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


def apply_order_cluster_objective(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    horizon: int,
    weight_mult: int = 1,
) -> List[Any]:
    """Gom mỗi SALES-ORDER vào một cửa sổ thời gian liền mạch.

    `apply_order_flow_objective` đã nén span theo `group_id` = COMPONENT batch
    (mỗi panel-batch chặt), NHƯNG một sales-order gồm nhiều component (KCC + KHL):
    không gì kéo các component của cùng một đơn về gần nhau → đơn bị knit một phần,
    bỏ trống vài ca, rồi chạy tiếp ("làm tư hôm trước, mấy ngày sau mới xong").
    Công nhân theo dõi theo ĐƠN nên WIP dở dang nằm chờ là pain thật.

    Phạt span (max_end − min_start) theo `ship_group_id` (ĐƠN KHÁCH — coarser hơn
    order_group_id/item; xem note granularity dưới).  Trọng số nằm DƯỚI lateness
    (×100) nên solver chỉ gom khi KHÔNG làm trễ đơn: dư slack thì gom miễn phí,
    deadline chặt thì lateness vẫn thắng (đã chứng minh ép liền mạch lúc chặt sẽ trễ
    → khi đó số hạng này tự nhường).  Cold-only (caller gate).

    LƯU Ý granularity: khóa gom là `ship_group_id` (đơn khách), KHÔNG phải
    `order_group_id` (item, khóa dyelot).  Các item khác nhau của một đơn được phép
    khác dyelot, nhưng vẫn phải xong gần cùng lúc để ship → co-completion ở mức đơn,
    dyelot ở mức item, hai trường tách biệt.
    """
    terms: List[Any] = []
    lateness_scale = min(max(1, horizon // 1000), 50)
    task_map = {t["task_id"]: t for t in tasks}

    # Group by ship_group_id (the customer/sales order), knitting tasks only.
    groups: Dict[str, List[str]] = {}
    for t in tasks:
        og = t.get("ship_group_id")
        t_id = t["task_id"]
        if og and t_id in task_vars and t.get("operation", "").lower() == "knitting":
            groups.setdefault(og, []).append(t_id)

    for og, t_ids in sorted(groups.items()):
        # Need ≥2 tasks for a span to be meaningful; a 1-task order is already tight.
        if len(t_ids) <= 1:
            continue

        max_priority = min((int(task_map[tid].get("priority", 5)) for tid in t_ids), default=3)
        weight = 10 ** (6 - max_priority)
        # Below order_flow's component span (//20) — a gentle nudge that breaks the
        # solver's lateness-neutral ties toward whole-order continuity without
        # disturbing component packing or trading any tardiness.  `weight_mult`
        # (config order_cluster_mult, default 1) scales this up when the caller wants
        # order-blocks enforced harder in a SATURATED pool (all items late ⇒ the tie
        # the gentle nudge relies on disappears, so it needs more weight to still pull
        # each order into a contiguous block instead of interleaving).
        cluster_w = (weight * lateness_scale * max(1, int(weight_mult))) // 20

        if cluster_w <= 0:
            continue

        o_start = model.NewIntVar(0, horizon, f"ord_start_{og}")
        o_end = model.NewIntVar(0, horizon, f"ord_end_{og}")
        for tid in t_ids:
            tv = task_vars[tid]
            model.Add(o_end >= tv["end"])
            model.Add(o_start <= tv["start"])

        span = model.NewIntVar(0, horizon, f"ord_span_{og}")
        model.Add(span == o_end - o_start)
        terms.append(span * cluster_w)

        # Co-COMPLETE the order EARLY, not merely close together.  Span alone is
        # satisfied by clustering the items anywhere on the timeline (all late, but
        # tight); a customer order can only ship once its LAST item is done, so the
        # ship-ready time is o_end.  Penalise o_end (same band, below lateness) so the
        # solver, once an order's items sit in one wave, pulls the whole block as early
        # as its slack allows instead of finishing one item on day 1 and stranding the
        # rest.  Below lateness ⇒ never trades tardiness; only spends free slack.
        terms.append(o_end * cluster_w)

    return terms


def apply_order_start_sync_objective(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    horizon: int,
    weight_mult: int = 0,
) -> List[Any]:
    """Cho các item của MỘT đơn khách knit SONG SONG — khởi động cùng lúc, mỗi item
    một máy — thay vì item lớn bị dồn ra sau các knit ngắn của đơn khác.

    Bối cảnh: đơn normal mang due HẰNG SỐ (chỉ để objective có mục tiêu, xem note
    is_normal), nên `apply_soft_deadlines` thoái hóa thành Σ(end)×100 = SPT: việc
    ngắn chạy trước, item LỚN (critical-path của đơn) bị đẩy muộn → các item cùng đơn
    KHÔNG chạy cùng lúc, WIP nằm chờ.  `apply_order_cluster_objective` phạt SPAN nhưng
    nằm DƯỚI ×100 nên bị SPT át (đo: nâng order_cluster_mult gần như không đổi
    start-spread).  Hàm này phạt riêng START-spread (max_start − min_start) mỗi
    ship_group để kéo các item khởi động cùng lúc.

    ⚠️ Trọng số NẰM TRONG băng lateness (10^(6−priority) × weight_mult) — để giành máy
    sớm cho item lớn, nó PHẢI thắng lực SPT ×100, nên nó ĐÁNH ĐỔI average completion
    lấy co-start/co-completion.  Đây là trade CÓ CHỦ Ý, opt-in qua config
    start_sync_mult (default 0 = OFF, model byte-identical).  Trần vật lý: số item
    co-start đồng thời ≤ số máy → không phải đơn nào cũng co-start được cùng lúc, solver
    xếp theo đợt.  Cold-only (caller gate), knitting-only, bỏ pinned/capacity_block.
    """
    terms: List[Any] = []
    mult = max(0, int(weight_mult))
    if mult <= 0:
        return terms
    task_map = {t["task_id"]: t for t in tasks}

    groups: Dict[str, List[str]] = {}
    for t in tasks:
        og = t.get("ship_group_id")
        t_id = t["task_id"]
        if (
            og
            and t_id in task_vars
            and t.get("operation", "").lower() == "knitting"
            and not t.get("is_pinned", False)
        ):
            groups.setdefault(og, []).append(t_id)

    for og, t_ids in sorted(groups.items()):
        # Need ≥2 items for a start-spread to exist; a 1-item order is trivially synced.
        if len(t_ids) <= 1:
            continue

        max_priority = min((int(task_map[tid].get("priority", 5)) for tid in t_ids), default=3)
        weight = (10 ** (6 - max_priority)) * mult

        s_min = model.NewIntVar(0, horizon, f"ssync_min_{og}")
        s_max = model.NewIntVar(0, horizon, f"ssync_max_{og}")
        for tid in t_ids:
            model.Add(s_min <= task_vars[tid]["start"])
            model.Add(s_max >= task_vars[tid]["start"])

        spread = model.NewIntVar(0, horizon, f"ssync_spread_{og}")
        model.Add(spread == s_max - s_min)
        terms.append(spread * weight)

    return terms


def apply_identical_task_symmetry(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    start_lb: Optional[Dict[str, int]] = None,
) -> int:
    """Phá đối-xứng giữa các task GIỐNG HỆT nhau (cold solve, mọi công đoạn).

    Khi payload có nhiều đơn y hệt (cùng design_item_id, color_config, qty,
    duration, due, priority, start_after, CÙNG earliest-start, và CÙNG tập máy
    tương thích), các task đó hoàn-toàn thay-thế-được: mọi hoán vị thứ tự cho
    cùng một objective.  CP-SAT phải duyệt N! hoán vị tương đương → kẹt FEASIBLE,
    mỗi solve tốn hàng phút (đo: knitting wave 161s, downstream 678s với 240 đơn
    giống nhau).

    Cách phá: trong mỗi nhóm-giống-hệt, sắp task theo task_id rồi ÉP start không
    giảm dần `start[i] <= start[i+1]`.  Ràng buộc HỢP LỆ — chỉ chọn một đại-diện
    chuẩn-tắc cho mỗi lớp hoán vị, KHÔNG loại giá-trị-objective phân-biệt nào →
    nghiệm tối ưu không đổi, chỉ co không gian tìm kiếm.

    ⚠️ CẢNH BÁO — KHÔNG an-toàn end-to-end trên pipeline TUẦN TỰ.  Ràng buộc này
    chỉ bảo-toàn objective CỦA RIÊNG pha hiện tại.  Nhưng các pha giải tuần tự, và
    hai task "giống hệt" thường nuôi HAI đơn downstream KHÁC nhau → hai lịch hòa
    nhau ở pha này KHÔNG hòa nhau ở downstream.  Ép thứ tự chuẩn-tắc vứt đi đúng
    thông tin xếp-chuỗi mà panel-co-completion/EDD đã cài cho downstream.  Đo trên
    payload 240 đơn giống hệt: iron/packing trễ 13_523 → 79_107 (~6×) khi bật cho
    knitting.  ⇒ CHỈ dùng khi các task thật-sự thay-thế-được CẢ Ở DOWNSTREAM (ví dụ
    pha cuối, hoặc bài toán một-pha độc lập).  Caller hiện default OFF.

    `start_lb` (earliest-start theo upstream) vào chữ ký để hai task chỉ bị ép thứ
    tự khi đầu-vào sẵn sàng cùng lúc.  Trả về số ràng buộc đã thêm.  Bỏ qua
    capacity_block, task đã pin, và task is_slice (slice cần interleave, không
    fungible-theo-thứ-tự).
    """
    start_lb = start_lb or {}

    def _sig(t: Dict[str, Any], tv: Dict[str, Any]) -> tuple:
        t_id = t["task_id"]
        return (
            t.get("operation", "").lower(),
            t.get("design_item_id"),
            t.get("color_config"),
            int(t.get("qty", 0) or 0),
            int(t.get("duration", 0) or 0),
            int(t.get("due_at_min", 0) or 0),
            int(t.get("priority", 5)),
            int(t.get("start_after_min", 0) or 0),
            int(start_lb.get(t_id, 0)),
            frozenset(tv.get("r_ids") or []),
        )

    groups: Dict[tuple, List[str]] = {}
    for t in tasks:
        t_id = t["task_id"]
        tv = task_vars.get(t_id)
        if tv is None or tv.get("is_pinned") or t.get("is_pinned"):
            continue
        if is_capacity_block_op(t.get("operation", "")):
            continue
        # Slices are NOT fungible-for-ordering: the pipeline deliberately INTERLEAVES
        # slice_1 of every order (so a downstream consumer gets one slice from each
        # early) — forcing a canonical start order would bunch them (A_s1..A_sN then
        # B_s1..), the exact anti-pattern slice_sync/panel_sync exist to prevent.
        # Only non-slice identical tasks (e.g. whole BATCH lots) are reorder-safe.
        if t.get("is_slice"):
            continue
        groups.setdefault(_sig(t, tv), []).append(t_id)

    n_constraints = 0
    n_groups = 0
    for _sig_key, t_ids in groups.items():
        if len(t_ids) < 2:
            continue
        n_groups += 1
        t_ids.sort()
        for a, b in zip(t_ids, t_ids[1:]):
            model.Add(task_vars[a]["start"] <= task_vars[b]["start"])
            n_constraints += 1
    if n_constraints:
        logger.info(
            f"   🪞 Identical-task symmetry break: {n_groups} group(s), "
            f"{n_constraints} ordering constraint(s) added."
        )
    return n_constraints


def apply_earliness_objective(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    horizon: int,
) -> List[Any]:
    """Kéo MỌI task của một công đoạn KHÔNG-PHẢI-CUỐI về xong sớm nhất có thể.

    Bối cảnh: due_at_min của từng task được back-propagate từ due của ĐƠN với
    lead = đúng MỘT task-duration mỗi công đoạn (linking 1477, washing 1537,
    iron 1552, packing 1559) — bỏ qua việc 240 task xếp hàng trên 2–5 máy.  Vì
    vậy due trung gian quá lỏng: solver chỉ phạt trễ-so-với-due, không thưởng
    xong-sớm, nên linking/washing "rề" tới sát due lỏng đó rồi mới xong, nuốt hết
    thời gian mà iron/packing (bị giới hạn năng lực thật) cần → downstream trễ.

    Với group toàn-slice, `apply_order_flow_objective` BỎ QUA group_end (nhường
    cho slice_sync), nên linking hoàn toàn không có lực kéo-về-sớm tuyệt đối —
    chỉ còn start tie-breaker (weight//100) bị slice_sync (×~0.7·weight) lấn át.
    Hàm này thêm lực `end × earliness_w` cho từng task.

    Trọng số `weight` (= 1·min_weight) NẰM TRÊN order_flow/slice_sync (~0.7·weight)
    để thắng việc rải, NHƯNG DƯỚI lateness (×100) gấp 100 lần → không bao giờ ép
    một task xong sớm bằng cách làm task khác trễ.  Chỉ phá thế-hòa-lateness theo
    hướng "xong sớm, chừa thời gian cho downstream".  Cold-only (caller gate).
    Bỏ qua capacity_block và task đã pin.
    """
    terms: List[Any] = []
    for t_id, tv in sorted(task_vars.items()):
        task = next((t for t in tasks if t["task_id"] == t_id), None)
        if task is None:
            continue
        if task.get("is_pinned", False):
            continue
        if is_capacity_block_op(task.get("operation", "")):
            continue
        priority = int(task.get("priority", 5))
        weight = 10 ** (6 - priority)
        terms.append(tv["end"] * weight)
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
# Panel co-completion (knitting components of one garment finish together)
# ---------------------------------------------------------------------------

def build_panel_map(
    all_pipeline_tasks: List[Dict[str, Any]],
    knitting_task_ids: Set[str],
    translation_map: Optional[Dict[str, str]] = None,
) -> "tuple[Dict[str, str], Dict[str, Dict[str, Any]]]":
    """Invert linking → knitting deps into per-panel groupings.

    A linking SLICE_k consumes the knitting batch of EVERY component (front/back/
    sleeve …) at index k via `final_depends_on`.  That set IS the panel: linking
    cannot start until the LAST of it finishes.  Phase-1 never sees this structure
    (the edge lives on the linking side), so the knitting solver has no reason to
    finish a panel's components together — they drift apart and linking waits on
    the straggler.  This builds the mapping phase-1 needs.

    A knitting batch feeds exactly ONE linking slice (confirmed on real payloads:
    each component → one linking order), so panels are disjoint.

    Returns
    -------
    panel_of:   knit_task_id → panel_key (= the consuming linking task_id)
    panel_meta: panel_key → {due, priority, index, members:[knit_task_id,…]}

    Deps that don't resolve to a known knitting task (translation/non-knit) are
    ignored — we never group what we can't identify as a knitting panel piece.
    """
    translation_map = translation_map or {}

    def _resolve(raw_id: str) -> Optional[str]:
        if raw_id in knitting_task_ids:
            return raw_id
        t = translation_map.get(raw_id)
        if t in knitting_task_ids:
            return t
        return None

    panel_of: Dict[str, str] = {}
    panel_meta: Dict[str, Dict[str, Any]] = {}
    for t in all_pipeline_tasks:
        if t.get("operation", "").lower() != "linking":
            continue
        members = []
        for dep in (t.get("final_depends_on") or []):
            r = _resolve(dep)
            if r is not None:
                members.append(r)
        if len(members) < 2:
            # A 0/1-component "panel" has no spread to close — skip so it adds no
            # objective term and the warm-start treats it as an ordinary task.
            continue
        panel_key = t["task_id"]
        due = t.get("due_at_min")
        panel_meta[panel_key] = {
            "due": int(due) if due is not None else None,
            "priority": int(t.get("priority", 5)),
            "index": int(t.get("slice_index", 0)),
            "members": members,
        }
        for k in members:
            # First writer wins (a batch feeds one slice); ties resolved
            # deterministically by sorted iteration is not guaranteed here, so
            # keep the EARLIEST-due consumer if a batch ever appears twice.
            prev = panel_of.get(k)
            if prev is None:
                panel_of[k] = panel_key
            else:
                pd = panel_meta[prev]["due"]
                nd = panel_meta[panel_key]["due"]
                if nd is not None and (pd is None or nd < pd):
                    panel_of[k] = panel_key
    return panel_of, panel_meta


def apply_panel_sync_objective(
    model: cp_model.CpModel,
    task_vars: Dict[str, Dict[str, Any]],
    panel_meta: Dict[str, Dict[str, Any]],
    horizon: int,
) -> List[Any]:
    """B1: minimise each panel's max component-end (the BOM-ready time that gates
    its linking slice).

    For every panel with ≥2 component batches present in this model, create
    `panel_end ≥ end` of each member and penalise `panel_end`.  Pulling the
    straggler component earlier lets the linking slice start sooner WITHOUT
    needing extra machines — it is a sequencing nudge, commensurate with the
    flow/slice-sync scale so it never overrides true lateness.

    Members not in `task_vars` (e.g. in another rolling wave) are skipped; the
    term still synchronises whatever subset this model owns.
    """
    terms: List[Any] = []
    lateness_scale = min(max(1, horizon // 1000), 50)
    for panel_key, meta in sorted(panel_meta.items()):
        members = [m for m in meta["members"] if m in task_vars]
        if len(members) < 2:
            continue
        prio = int(meta.get("priority", 5))
        weight = 10 ** (6 - prio)
        w = (weight * lateness_scale) // 20
        if w <= 0:
            continue
        panel_end = model.NewIntVar(0, horizon, f"panel_end_{panel_key}")
        for m in members:
            model.Add(panel_end >= task_vars[m]["end"])
        terms.append(panel_end * w)
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
    real_tasks = [t for t in tasks if not is_capacity_block_op(t.get("operation", ""))]

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
            elif is_capacity_block_op(op):
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
        if is_capacity_block_op(task.get("operation", "")):
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

        # Normal orders: still scheduled toward dueDate, but never reported as an
        # overload even when they finish late (per user — "tuân thủ dueDate nhưng
        # không khắt khe trả overload").  The assignment status above still reflects
        # LATE/ON_TIME truthfully; only the overloads diagnostic list suppresses them.
        if is_late and not tv.get("is_pinned") and not task.get("is_normal", False):
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
