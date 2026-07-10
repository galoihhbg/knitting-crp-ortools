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
from .placement import (  # A1a shared helpers
    earliest_sweep as _earliest_slot,
    overlaps,
    relabel_balance,
    release_from_deps,
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
        return release_from_deps(info[t_id], dep_ends)

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


def left_shift_cold_packing(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    dep_ends: Dict[str, int],
) -> int:
    """Post-pass: pull each packing task to its earliest feasible start on the earliest-free
    COMPATIBLE packing machine (packing machines are homogeneous serial units, so which one is
    interchangeable).  Same FEASIBLE-stall trap as ironing/linking: packing is the LAST phase,
    its due dates are the loosest, so the solver has the least early-start incentive and staggers
    packing minutes past its ironing-ready time while a compatible packing machine sits idle
    (measured: 5/78 slices slipped 100–158 min with a free compatible machine available).

    release = max(start_after_min, latest ironing-end via final_depends_on).  Monotone guard by
    construction (remove-then-place: every task seeded at its solver slot, moved only into a
    STRICTLY earlier conflict-free slot) ⇒ packing is the terminal op so no downstream to disturb
    — end-to-end lateness is monotone non-increasing.  Pinned packing tasks are immovable anchors.
    Deterministic (release/start/id order, machine-id tie-break).  Runs on re-schedule too (the
    double-solve returns pass-2 to the UI, where packing is hard-kept to pass-1 while ironing may
    have moved earlier); idempotent + pinned-anchored.  Returns #tasks moved.
    """
    info = {t["task_id"]: t for t in all_tasks}
    pack_ids = {
        t["task_id"] for t in all_tasks
        if str(t.get("operation", "")).lower() in ("pack", "packing")
    }
    pack_assigns = [a for a in assignments if a["task_id"] in pack_ids]
    if not pack_assigns:
        return 0

    def _release(t_id: str) -> int:
        return release_from_deps(info[t_id], dep_ends)

    pool = {a["machine_id"] for a in pack_assigns}
    for a in pack_assigns:
        pool |= set(info[a["task_id"]].get("compatible_resource_ids") or [])

    busy: Dict[str, List[tuple]] = {m: [] for m in pool}
    for a in pack_assigns:
        busy[a["machine_id"]].append((int(a["start_time"]), int(a["end_time"])))
    for m in busy:
        busy[m].sort()

    moved = 0
    order = sorted(
        (a for a in pack_assigns if not info[a["task_id"]].get("is_pinned")),
        key=lambda a: (a["start_time"], a["end_time"], a["task_id"]),
    )
    for a in order:
        t_id = a["task_id"]
        cur_m, cur_s = a["machine_id"], int(a["start_time"])
        dur = int(a["end_time"]) - cur_s
        rel = _release(t_id)
        busy[cur_m].remove((cur_s, int(a["end_time"])))
        compat = set(info[t_id].get("compatible_resource_ids") or [])
        cands = [m for m in sorted(pool) if not compat or m in compat] or [cur_m]
        cand_s = {m: _earliest_slot(list(busy[m]), rel, dur) for m in cands}
        min_s = min(cand_s.values())
        if min_s < cur_s:
            best_s = min_s
            best_m = min(m for m in cands if cand_s[m] == min_s)
        else:
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
            f"   ⬅️ Cold packing left-shift: re-seated {moved} packing task(s) onto the earliest-free "
            f"compatible machine (spreads ready-together slices, terminal phase — nothing downstream)."
        )
    return moved


def balance_downstream_load(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    ops: frozenset,
    label: str,
) -> int:
    """Post-pass cân tải thợ iron/packing bằng cách ĐỔI NHÃN MÁY (mirror của
    balance_linking_load, tham số hoá theo `ops`).

    Bài toán thực địa: iron/packing KHÔNG ai phải chờ (left-shift đã dán slice vào
    máy rảnh sớm nhất) nhưng chia việc lệch nặng — đo trên payload thật: iron
    49/38/6/3/1 task, packing 48/39/6/3/1 (thợ 05 làm 1 task/7 phút cả kỳ).  Nguồn
    gốc: solver + left-shift tie-break "máy id nhỏ nhất" và không có động lực trải
    việc; các máy iron/packing HOÁN ĐỔI ĐƯỢC (mọi task compatible cả 5 máy) nên
    việc dồn về 01/02 tuỳ tiện — "một người làm nhiều task, người thì không làm gì".

    Cách gỡ an toàn tuyệt đối (như linking): GIỮ NGUYÊN [start, end] của mọi task,
    chỉ gán lại máy bằng greedy "tô màu interval" — duyệt theo start, đặt mỗi task
    lên máy hợp-lệ (compatible, rảnh trong [start,end], ngoài unavailability, sau
    available_at_min) có TẢI HIỆN TẠI THẤP NHẤT.  Vì thời gian không đổi:
      * downstream byte-identical (packing phụ thuộc iron end-time, không phụ thuộc máy);
      * lateness không đổi, KHÔNG đơn nào trễ hơn (zero regression theo cấu trúc);
      * no-overlap giữ nguyên (chỉ chọn máy không đè); luôn khả thi vì số máy ≥
        đỉnh đồng thời (gán cũ đã chứng minh ≤ số máy).

    Pinned tasks (đang chạy) là mỏ neo bất động — giữ nguyên máy.  Deterministic
    O(n log n).  Chạy trên cold + stabilize pass (UI nhận lịch cân); KHÔNG chạy
    khi re-schedule thật của Go (phân công thợ là một phần của stability ở đó).
    Mutates `assignments` in place (machine_id).  Returns số task được đổi máy.

    A1a: thân hàm delegate sang ``placement.relabel_balance`` (bản hợp nhất với
    ``balance_linking_load`` — tương đương fuzz-đối-chiếu trong
    tests/test_placement_helpers.py).
    """
    return relabel_balance(
        assignments, all_tasks, resources, config, ops=ops, label=label,
    )


def fifo_swap_ironing(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    dep_ends: Dict[str, int],
) -> int:
    """Post-pass sửa FIFO inversion trên máy iron: đứa READY-TRƯỚC bị chờ vì một đứa
    READY-SAU chiếm đúng cửa sổ máy — vi phạm luật xưởng "đến trước làm trước".

    Ca thực đo được (CP_1783300710469478376): mẻ giặt nhả 5 slice cùng lúc @518 nhưng
    máy 03 chỉ trống 60 phút vì I1-Wf90PwfsLf SLICE_3 (ready @578, SAU) đã chiếm
    578-680 → WNesoSo0vK SLICE_4 (ready @518, TRƯỚC) phải lùi 102 phút.  Hoán đổi
    (SLICE_4 vào 03 @518, SLICE_3 re-seat @620) cho đơn sớm 102 phút mà đơn kia
    KHÔNG muộn đi giây nào — nhưng left-shift bị luật monotone cấm (phải dời muộn
    blocker), còn solver kẹt FEASIBLE không tự thấy.

    Với iron, guard "đơn không muộn hơn" dùng PROXY max iron-end của đơn (packing
    chưa solve).  Gọi MỘT LẦN tại site phase-4, TRƯỚC khi packing solve (packing bám
    theo end mới); các site muộn hơn (re-glue/hole-closing) KHÔNG gọi vì packing đã
    chốt, dời muộn iron sẽ gãy chuỗi.  Cơ chế/guard chi tiết: _fifo_swap_downstream.
    """
    return _fifo_swap_downstream(
        assignments, all_tasks, config, dep_ends,
        ops=("iron", "ironing"),
        flag="enable_ironing_fifo_swap",
        label="Ironing",
        use_due_cap=False,
    )


def fifo_swap_packing(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    dep_ends: Dict[str, int],
) -> int:
    """FIFO-swap cho PACKING — cùng cái bẫy iron nhưng ở phase cuối.

    Ca thực đo được (CP_1783586686707912847): P1-WMMkf9wYUm SLICE_1 (51′) ready @643
    nhưng cả hai cửa sổ trống lúc đó (W_PACKING_01/05) bị các task 3-phút ready-SAU
    (@667/@649, due 1679 — slack >1000′) chiếm đúng giữa → phải lùi tới 670 → trễ đơn
    2′; P1-WG9IlBWKGw trễ 3′ cùng cơ chế.  Swap các blocker tí hon ra sau cứu cả hai
    đơn mà không ai muộn đi.

    AN TOÀN HƠN iron: packing là op CUỐI nên guard "đơn không muộn hơn" dùng đúng
    max packing-end của đơn (chính là end đơn), không phải proxy; không có hạ nguồn
    nào phải bám theo.  Gọi sau left_shift_cold_packing trong _solve_phases_4_5.
    """
    return _fifo_swap_downstream(
        assignments, all_tasks, config, dep_ends,
        ops=("pack", "packing"),
        flag="enable_packing_fifo_swap",
        label="Packing",
        use_due_cap=True,
    )


def _fifo_swap_downstream(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    dep_ends: Dict[str, int],
    *,
    ops: tuple,
    flag: str,
    label: str,
    use_due_cap: bool,
) -> int:
    """Core FIFO-swap dùng chung cho iron/packing (A1a-style: một bản, tham số hoá).

    Cơ chế: với mỗi task w đang CHỜ (start > release), thử đặt w tại release trên một
    máy tương thích; tập blocker B = các task chồng cửa sổ đó.  Chỉ nhận swap khi MỌI
    blocker t ∈ B:
      * không pinned và release_t > release_w (đúng nghĩa inversion — t đến sau);
      * re-seat được tại slot trống sớm nhất ≥ release_t sao cho end mới của t
        ≤ end CŨ của w (nhóm task liên quan không dài ra) VÀ:
          - use_due_cap=False (iron): new_e ≤ max end hiện tại của ĐƠN t trong phase
            (proxy — packing chưa solve, không so due được) và t không chuyển từ
            ON_TIME sang LATE;
          - use_due_cap=True (packing, op CUỐI): new_e ≤ DUE của t — end đơn = max
            packing-end nên tardiness đơn không tăng và không task nào lật LATE;
            proxy iron ở đây từ chối oan blocker mà đơn nó chỉ có một packing task
            (case CP_1783586686707912847).  Không có due → rơi về proxy như iron.
    → không đơn nào muộn đi theo cấu trúc; w chỉ sớm lên.
    Deterministic; mutates in place; trả về số cặp swap đã nhận.
    """
    if not config.get(flag, True):
        return 0
    info = {t["task_id"]: t for t in all_tasks}
    op_ids = {
        t["task_id"] for t in all_tasks
        if str(t.get("operation", "")).lower() in ops
    }
    iron_assigns = [a for a in assignments if a["task_id"] in op_ids]
    if len(iron_assigns) < 2:
        return 0

    def _release(t_id: str) -> int:
        return release_from_deps(info[t_id], dep_ends)

    def _due(t_id: str) -> Optional[int]:
        v = info.get(t_id, {}).get("due_at_min")
        return int(v) if v is not None else None

    pool = {a["machine_id"] for a in iron_assigns}
    for a in iron_assigns:
        pool |= set(info[a["task_id"]].get("compatible_resource_ids") or [])
    busy: Dict[str, List[Dict[str, Any]]] = {m: [] for m in pool}
    for a in iron_assigns:
        busy[a["machine_id"]].append(a)
    for m in busy:
        busy[m].sort(key=lambda x: (x["start_time"], x["task_id"]))

    # Max iron end per order — proxy "đơn không muộn hơn" (packing chưa solve).
    order_max_end: Dict[str, int] = {}
    for a in iron_assigns:
        o = a.get("order_id") or a["task_id"]
        order_max_end[o] = max(order_max_end.get(o, 0), int(a["end_time"]))

    swapped = 0
    waiting = sorted(
        (a for a in iron_assigns
         if not info[a["task_id"]].get("is_pinned")
         and int(a["start_time"]) > _release(a["task_id"])),
        key=lambda a: (_release(a["task_id"]), a["start_time"], a["task_id"]),
    )
    for w in waiting:
        w_rel = _release(w["task_id"])
        w_dur = int(w["end_time"]) - int(w["start_time"])
        w_old_end = int(w["end_time"])
        if int(w["start_time"]) <= w_rel:
            continue  # đã được pass trước kéo về rồi
        compat = set(info[w["task_id"]].get("compatible_resource_ids") or []) or pool
        done = False
        for m in sorted(compat & set(busy.keys())):
            if done:
                break
            win_s, win_e = w_rel, w_rel + w_dur
            blockers = [
                t for t in busy[m]
                if t is not w
                and overlaps(int(t["start_time"]), int(t["end_time"]), win_s, win_e)
            ]
            if not blockers:
                continue  # cửa sổ trống — việc của left-shift, không phải swap
            if any(
                info[t["task_id"]].get("is_pinned")
                or _release(t["task_id"]) <= w_rel
                for t in blockers
            ):
                continue  # chỉ swap khi MỌI blocker đến SAU w (inversion thật)
            # Thử re-seat từng blocker với w đã chiếm [win_s, win_e) trên m.
            trial: List[tuple] = []
            occupied = {
                mm: [(int(x["start_time"]), int(x["end_time"]))
                     for x in busy[mm] if x is not w and x not in blockers]
                for mm in busy
            }
            occupied[m].append((win_s, win_e))
            for mm in occupied:
                occupied[mm].sort()
            ok = True
            for t in sorted(blockers, key=lambda x: (_release(x["task_id"]), x["task_id"])):
                t_rel = _release(t["task_id"])
                t_dur = int(t["end_time"]) - int(t["start_time"])
                t_compat = set(info[t["task_id"]].get("compatible_resource_ids") or []) or pool
                best = None
                for mm in sorted(t_compat & set(occupied.keys())):
                    s = _earliest_slot(occupied[mm], t_rel, t_dur)
                    if best is None or (s, mm) < best:
                        best = (s, mm)
                if best is None:
                    ok = False
                    break
                new_s, new_m = best
                new_e = new_s + t_dur
                o = t.get("order_id") or t["task_id"]
                t_due = _due(t["task_id"])
                if use_due_cap:
                    bad = (
                        new_e > w_old_end
                        or (t_due is not None and new_e > t_due)
                        or (t_due is None and new_e > order_max_end.get(o, new_e))
                    )
                else:
                    bad = (
                        new_e > w_old_end
                        or new_e > order_max_end.get(o, new_e)
                        or (t_due is not None and int(t["end_time"]) <= t_due < new_e)
                    )
                if bad:
                    ok = False
                    break
                occupied[new_m].append((new_s, new_e))
                occupied[new_m].sort()
                trial.append((t, new_m, new_s, new_e))
            if not ok:
                continue
            # Nhận swap: áp dụng w + mọi blocker.
            busy[w["machine_id"]].remove(w)
            w["machine_id"], w["start_time"], w["end_time"] = m, win_s, win_e
            for t, new_m, new_s, new_e in trial:
                busy[t["machine_id"]].remove(t)
                t["machine_id"], t["start_time"], t["end_time"] = new_m, new_s, new_e
                t_due = _due(t["task_id"])
                if t_due is not None:
                    t["status"] = "LATE" if new_e > t_due else "ON_TIME"
                busy[new_m].append(t)
                busy[new_m].sort(key=lambda x: (x["start_time"], x["task_id"]))
            w_due = _due(w["task_id"])
            if w_due is not None:
                w["status"] = "LATE" if win_e > w_due else "ON_TIME"
            busy[m].append(w)
            busy[m].sort(key=lambda x: (x["start_time"], x["task_id"]))
            swapped += 1
            done = True

    if swapped:
        logger.info(
            f"   🔁 {label} FIFO-swap: {swapped} inversion(s) fixed — earlier-ready "
            f"slice takes the machine, later-ready blocker re-seated (no order later)."
        )
    return swapped
