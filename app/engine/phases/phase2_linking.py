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
    apply_earliness_objective,
    apply_end_caps,
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
    avail_at,
    earliest_sweep,
    relabel_balance,
    unavail_windows,
)

logger = logging.getLogger(__name__)

PHASE2_OPS = frozenset({"linking"})


def balance_linking_load(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> int:
    """COLD-only post-pass: cân tải nhân công linking bằng cách ĐỔI NHÃN MÁY.

    Bài toán thực địa: linking workers nghỉ quá nhiều và tải lệch nặng (một thợ
    ôm 3000+ phút, thợ khác 0) — đo trên payload thật: stdev tải máy ~965, 4/20
    máy không việc.  Nguồn gốc: solver chỉ tối ưu trễ-hạn + slice-sync, không có
    động lực trải việc; các máy linking HOÁN ĐỔI ĐƯỢC (design/color rỗng) nên nó
    dồn việc lên ít máy tuỳ tiện.

    Cách gỡ an toàn tuyệt đối: GIỮ NGUYÊN [start, end] của mọi task linking, chỉ
    gán lại máy bằng greedy "tô màu interval" — duyệt theo start, đặt mỗi task lên
    máy hợp-lệ (trong compatible_resource_ids, rảnh trong [start,end], ngoài cửa
    sổ unavailability, sau available_at_min) có TẢI HIỆN TẠI THẤP NHẤT.  Vì thời
    gian không đổi:
      * downstream byte-identical (chúng phụ thuộc end-time, không phụ thuộc máy);
      * lateness không đổi; KHÔNG đơn nào trễ hơn (zero regression theo cấu trúc);
      * no-overlap giữ nguyên (chỉ chọn máy không đè); luôn khả thi vì số máy ≥
        đỉnh đồng thời (gán cũ đã chứng minh ≤ số máy).
    Đo: stdev tải 965→40, 0 thợ ngồi không (mọi máy ~tải đều).

    Pinned linking (đang chạy) là mỏ neo bất động — giữ nguyên máy.  Deterministic
    O(n log n).  KHÔNG chạy khi re-schedule (máy là một phần của stability ở đó).
    Mutates `assignments` in place (machine_id).  Returns số task được đổi máy.

    A1a: thân hàm delegate sang ``placement.relabel_balance`` (bản hợp nhất với
    ``balance_downstream_load`` — tương đương fuzz-đối-chiếu trong
    tests/test_placement_helpers.py).
    """
    return relabel_balance(
        assignments, all_tasks, resources, config,
        ops=frozenset(PHASE2_OPS), label="Linking",
    )


def left_shift_cold_linking(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> int:
    """COLD-only post-pass: pull each linking task to its earliest feasible start
    across all compatible workers, so linking begins the moment its knitting panel is
    ready instead of being staggered late.

    The linking due dates are typically far off, so the solver has no lateness
    incentive to start early and stalls at a FEASIBLE solution that spreads slices
    arbitrarily across time even though every worker is idle and every panel is ready
    (measured: starts staggered 406→4024 / makespan 4225 while a simple earliest-free
    placement gives 1009).  Each task's earliest start is a FIFO-theo-PO floor: the k-th
    slice of an order waits for the k-th-FINISHED panel of each component-PO bucket
    ((group_id, qty)) — NOT the rigid index-paired panel — read from the CURRENT (post
    knitting left-shift/spread) assignments, so linking also inherits any earlier
    knitting AND a middle slice no longer waits on a late index-panel when a same-PO
    panel already finished.  Bijection slice↔panel ⇒ the last slice still waits for the
    last panel (order completion unchanged); only middle slices are freed earlier.

    Safety mirrors the knitting spread: every linking task is seeded at its ORIGINAL
    (worker, start) and may only move into an EARLIER feasible gap (its own slot is
    always a fallback ⇒ new_end ≤ old_end).  Washing/iron/packing depend on linking
    END (and start+offset) — both only DECREASE — so every downstream release bound
    only relaxes ⇒ downstream assignments stay byte-identical and lateness is monotone
    non-increasing.  Worker unavailability windows + available_at_min are respected;
    pinned linking tasks are immovable anchors.  Deterministic; ties by worker id.
    NOT applied on re-schedule.  Mutates `assignments` in place.  Returns #tasks moved.
    """
    info = {t["task_id"]: t for t in all_tasks}
    workers = [r["id"] for r in resources if r.get("operation", "").lower() in PHASE2_OPS]
    if not workers:
        return 0
    res_by_id = {r["id"]: r for r in resources}

    def _unavail(m_id: str):
        return unavail_windows(res_by_id.get(m_id, {}))

    def _avail_at(m_id: str) -> int:
        return avail_at(res_by_id.get(m_id, {}))

    # Current upstream knitting timing (post knitting left-shift/spread) for the lb.
    k_start = {a["task_id"]: int(a["start_time"]) for a in assignments}
    k_end = {a["task_id"]: int(a["end_time"]) for a in assignments}

    link_assigns = [
        a for a in assignments
        if info.get(a["task_id"], {}).get("operation", "").lower() in PHASE2_OPS
    ]
    if not link_assigns:
        return 0

    # FIFO-theo-PO release floor (thay cho floor index cứng).  Go ghép cứng linking
    # SLICE_k ↔ panel BATCH_<comp>_k theo index, nhưng các panel CÙNG (component PO,
    # qty) là THAY-THẾ-ĐƯỢC (chốt domain): slice thứ k chỉ cần panel-xong-thứ-k của
    # bucket, không cần đúng panel số-hiệu-k.  Khi thứ tự dệt-xong ≠ thứ tự index,
    # một slice giữa bị floor index ghìm trong khi panel cùng-PO đã xong nằm chờ.
    # Ở đây floor mỗi slice = end của panel-xong-thứ-k trong bucket (con trỏ `ptr`
    # tiêu thụ mỗi panel đúng MỘT lần ⇒ không đếm trùng), WaitOffsets đọc từ ĐÚNG
    # panel đó.  Bijection slice↔panel ⇒ slice cuối vẫn chờ panel cuối (đơn xong
    # KHÔNG đổi); chỉ slice GIỮA được nới sớm.  Vì left-shift chỉ kéo sớm + seed mọi
    # task ở slot gốc, floor cao bất ngờ cũng vô hại (giữ nguyên).  Dep không-knitting
    # / không-resolve → hành vi index.  Floor hợp lệ vì Go chạy theo start/end trả về.
    def _knit_bucket(dep_id: str):
        kt = info.get(dep_id)
        if not kt or kt.get("operation", "").lower() != "knitting":
            return None
        group = kt.get("group_id") or ""
        if not group:
            return None
        return (group, int(round(float(kt.get("qty", 0) or 0))))

    def _index_floor(t: Dict[str, Any]) -> int:
        ends = [k_end[d] for d in (t.get("final_depends_on") or []) if d in k_end]
        return max(ends) if ends else 0

    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for a in link_assigns:
        t = info[a["task_id"]]
        by_parent.setdefault(t.get("parent_task_id") or t["task_id"], []).append(t)

    release_of: Dict[str, int] = {}
    for parent, slices in sorted(by_parent.items()):
        bucket_panels: Dict[tuple, Dict[str, tuple]] = {}
        for t in slices:
            for dep in (t.get("final_depends_on") or []):
                if dep not in k_end:
                    continue
                key = _knit_bucket(dep)
                if key is not None:
                    bucket_panels.setdefault(key, {})[dep] = (k_start.get(dep), k_end[dep])
        bucket_sorted = {
            k: sorted(v.values(), key=lambda se: (se[1], se[0] if se[0] is not None else se[1]))
            for k, v in bucket_panels.items()
        }
        ptr: Dict[tuple, int] = {}
        for t in sorted(slices, key=lambda x: (_index_floor(x), x["task_id"])):
            wait_offsets = t.get("wait_offsets") or t.get("WaitOffsets") or {}
            consumed: set = set()
            c = 0
            for dep in (t.get("final_depends_on") or []):
                if dep not in k_end:
                    continue
                off = wait_offsets.get(dep)
                if dep in wait_offsets:
                    consumed.add(dep)
                key = _knit_bucket(dep)
                if key is None:
                    c = max(c, k_end[dep])
                    if off is not None and dep in k_start:
                        c = max(c, k_start[dep] + int(off))
                    continue
                panels = bucket_sorted[key]
                k = ptr.get(key, 0)
                ptr[key] = k + 1
                p_start, p_end = panels[min(k, len(panels) - 1)]
                c = max(c, p_end)
                if off is not None and p_start is not None:
                    c = max(c, p_start + int(off))
            for batch, off in wait_offsets.items():
                if batch in consumed:
                    continue
                if batch in k_start:
                    c = max(c, k_start[batch] + int(off))
            release_of[t["task_id"]] = c

    orig_m = {a["task_id"]: a["machine_id"] for a in link_assigns}
    orig_s = {a["task_id"]: int(a["start_time"]) for a in link_assigns}
    dur_of = {a["task_id"]: int(a["end_time"]) - int(a["start_time"]) for a in link_assigns}

    # Seed: worker unavailability windows + every linking task at its original slot.
    placed: Dict[str, List[tuple]] = {m: list(_unavail(m)) for m in workers}
    for a in link_assigns:
        placed.setdefault(a["machine_id"], []).append(
            (orig_s[a["task_id"]], orig_s[a["task_id"]] + dur_of[a["task_id"]])
        )
    for m in placed:
        placed[m].sort()

    cur_m = dict(orig_m)
    cur_s = dict(orig_s)
    free = sorted(
        (a for a in link_assigns if not info[a["task_id"]].get("is_pinned")),
        key=lambda a: (orig_s[a["task_id"]], a["task_id"]),
    )
    for a in free:
        t_id = a["task_id"]
        dur = dur_of[t_id]
        release = max(release_of.get(t_id, 0), 0)
        compat = [w for w in (info[t_id].get("compatible_resource_ids") or []) if w in placed]
        if orig_m[t_id] not in compat:
            compat.append(orig_m[t_id])

        m0, s0 = cur_m[t_id], cur_s[t_id]
        placed[m0].remove((s0, s0 + dur))

        best_m, best_s = orig_m[t_id], orig_s[t_id]  # guaranteed-available fallback
        for m in sorted(compat):
            s = earliest_sweep(placed.get(m, []), max(release, _avail_at(m)), dur)
            if s < best_s or (s == best_s and m < best_m):
                best_m, best_s = m, s

        cur_m[t_id], cur_s[t_id] = best_m, best_s
        placed.setdefault(best_m, []).append((best_s, best_s + dur))
        placed[best_m].sort()

    moved = 0
    for a in assignments:
        t_id = a["task_id"]
        if t_id not in cur_s:
            continue
        if a["machine_id"] != cur_m[t_id] or a["start_time"] != cur_s[t_id]:
            moved += 1
        a["machine_id"] = cur_m[t_id]
        a["start_time"] = cur_s[t_id]
        a["end_time"] = cur_s[t_id] + dur_of[t_id]
        due = int(info[t_id].get("due_at_min", a["end_time"] + 1) or (a["end_time"] + 1))
        a["status"] = "LATE" if a["end_time"] > due else "ON_TIME"

    if moved:
        logger.info(
            f"   ⬅️ Cold linking left-shift: pulled {moved} task(s) to earliest "
            f"feasible start on free workers (linking now tight to knitting; "
            f"downstream untouched)."
        )
    return moved


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
    workload_shrank: bool = False,
    start_lb_override: Optional[Dict[str, int]] = None,
    end_caps: Optional[Dict[str, int]] = None,
    all_pipeline_tasks: Optional[List[Dict[str, Any]]] = None,
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
        start_lb_override: dùng floor này thay vì floor mặc định (pass 2 của
            same-qty relink — xem pipeline).  None → floor mặc định.
        end_caps:        task_id → end tối đa (end của pass 1).  Pareto guard:
            pass 2 không được ra lịch muộn hơn pass 1 ở bất kỳ task nào.
        all_pipeline_tasks: toàn bộ task pipeline (gồm knitting) để tra group_id/qty
            cho FIFO-by-PO floor (enable_fifo_linking_floor).  None → floor index.
    """
    linking_tasks = [t for t in tasks if t.get("operation", "").lower() in PHASE2_OPS]
    if not linking_tasks:
        logger.info("⚙️ Phase 2 (Linking): no tasks — skipped.")
        return Phase2Result(status="empty")

    if horizon is None:
        horizon = compute_horizon(linking_tasks, config, resources=resources)

    # Compute start lower-bounds from Phase 1.  Default = FIFO-by-PO floor (panels of
    # the same (component, qty) bucket are interchangeable → SLICE thứ k chờ panel
    # XONG-thứ-k, không phải panel số-hiệu-k); falls back to index pairing when the flag
    # is off or knitting metadata is unavailable.  Index pairing alone leaves linking
    # waiting on a late index-panel while an interchangeable sibling sits ready.
    if start_lb_override is not None:
        start_lb = start_lb_override
    elif config.get("enable_fifo_linking_floor", True) and all_pipeline_tasks is not None:
        start_lb = compute_sameqty_start_lb(
            linking_tasks, p1_start_times, p1_end_times, translation_map, all_pipeline_tasks
        )
    else:
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

    if end_caps:
        n_caps = apply_end_caps(model, task_vars, end_caps)
        logger.info(f"   🔒 Phase 2: {n_caps} Pareto end-cap(s) applied (two-pass refinement).")

    task_map = {t["task_id"]: t for t in linking_tasks}
    obj_terms = apply_soft_deadlines(model, task_vars, task_map, horizon)
    # Re-schedule: skip flow/sync (they outweigh stability pin) — see phase1.
    # EXCEPTION — workload shrank: re-enable so survivors re-pack (no gaps); the
    # soft anchor is one-sided (late_only) this run so it won't fight compaction.
    if not reschedule_hint or workload_shrank:
        # NB: identical-task symmetry break is NOT applied here — linking slices carry
        # wait_offsets / index-paired panel deps, so forcing start[i] ≤ start[j] between
        # same-signature slices contradicts their timing constraints → INFEASIBLE
        # (measured: linking returned UNKNOWN, whole pipeline produced no schedule).
        # The technique is only valid on independent tasks (knitting).
        obj_terms += apply_order_flow_objective(model, task_vars, linking_tasks, horizon)
        obj_terms += apply_slice_sync_objective(model, task_vars, linking_tasks, horizon)
        # Linking is all-slices → order_flow skips its group_end, so nothing pulls
        # linking to finish early; it dawdles to its (too-loose) due, stealing the
        # lead time the capacity-bound iron/packing actually need.  Pull it left.
        if config.get("enable_earliness_pull", False):
            obj_terms += apply_earliness_objective(model, task_vars, linking_tasks, horizon)

    stab_terms, stab_stats = apply_stability_objective(
        model, task_vars, linking_tasks, reschedule_hint, horizon, start_lb=start_lb,
        time_penalty="late_only" if workload_shrank else "abs",
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

    # Cold solve: tighten the gap to 0 so the solver pulls every linking slice to
    # its earliest feasible start instead of parking it late.  Linking machines are
    # interchangeable and lightly loaded, so a slice whose panels finished early can
    # almost always run early — but moving it earlier saves <1% of the (large,
    # absolute) objective, which the default 1% gap swallows: measured a slice left
    # idle 3215 min past its ready time (material done, machine free) until gap→0
    # pulled it back.  On reschedule keep the 1% gap so the stability anchors win.
    solver = make_solver(
        config,
        has_hint=bool(reschedule_hint),
        relative_gap=0.0 if not reschedule_hint else None,
    )
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


def compute_sameqty_start_lb(
    tasks: List[Dict[str, Any]],
    p1_start_times: Dict[str, int],
    p1_end_times: Dict[str, int],
    translation_map: Dict[str, str],
    all_pipeline_tasks: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Floor same-qty: panel knitting cùng (component, qty) là thay-thế-được.

    Luật vật-lý (chốt với domain): hai panel chỉ đổi chỗ cho nhau khi cùng
    component (group_id) VÀ cùng quantity — tuyệt đối không mix khác-qty
    (một panel qty-20 KHÔNG phủ được hai slice cần panel qty-10).

    Go ghép cứng linking SLICE_k ↔ panel BATCH_<comp>_k theo index; khi thứ tự
    dệt xong không trùng thứ tự index, một slice phải chờ "đúng panel số k"
    trong khi một panel cùng-loại-cùng-qty đã xong nằm chờ.  Floor này gỡ đúng
    chỗ đó: trong mỗi bucket (component, qty) của một parent, slice được floor
    theo panel-xong-thứ-k (FIFO theo end time) thay vì panel-số-hiệu-k.

    Per-order completion không đổi (song ánh slice↔panel ⇒ slice cuối vẫn chờ
    panel cuối) nhưng các slice GIỮA được nới sớm hơn → máy linking rảnh sớm →
    đơn khác + hạ nguồn hưởng.  Dùng làm start_lb_override cho pass 2, LUÔN
    kèm end_caps (Pareto guard) — không bao giờ dùng trần.

    WaitOffsets được merge y hệt _compute_start_lb (max wins).
    Deps không phải knitting (hoặc không resolve được) đóng góp end trực tiếp
    như floor index — không bucket hóa thứ mình không hiểu.
    """
    task_meta = {t["task_id"]: t for t in all_pipeline_tasks}

    def _resolve(raw_id: str) -> Optional[str]:
        if raw_id in p1_end_times:
            return raw_id
        translated = translation_map.get(raw_id, raw_id)
        if translated in p1_end_times:
            return translated
        return None

    def _knit_bucket(raw_id: str, resolved: str):
        """(group_id, qty) nếu dep là knitting có metadata; None → đối xử như index."""
        kt = task_meta.get(resolved) or task_meta.get(raw_id)
        if not kt or kt.get("operation", "").lower() != "knitting":
            return None
        group = kt.get("group_id") or ""
        if not group:
            return None
        return (group, int(round(float(kt.get("qty", 0) or 0))))

    # Gom theo parent (một đơn linking); bucket chỉ có nghĩa trong một parent.
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        by_parent.setdefault(t.get("parent_task_id") or t["task_id"], []).append(t)

    lb: Dict[str, int] = {}
    for parent, slices in sorted(by_parent.items()):
        # Panel records per bucket: resolved_id → (start, end).  We assign a slice
        # a WHOLE panel (FIFO by end) and read BOTH its end (final_depends_on floor)
        # and its start+offset (WaitOffsets floor) from that SAME panel.  Reading
        # both from one panel is what removes the old re-pin: previously the bucket
        # relaxed only the end while WaitOffsets re-pinned start+offset to the
        # specific index panel, capping the floor back to the index value (no-op).
        bucket_panels: Dict[tuple, Dict[str, tuple]] = {}
        for t in slices:
            for dep_id in (t.get("final_depends_on") or []):
                resolved = _resolve(dep_id)
                if not resolved:
                    continue
                key = _knit_bucket(dep_id, resolved)
                if key is not None:
                    bucket_panels.setdefault(key, {})[resolved] = (
                        p1_start_times.get(resolved), p1_end_times[resolved]
                    )
        # Sort each bucket FIFO by (end, start, id) for a deterministic assignment.
        bucket_sorted: Dict[tuple, List[tuple]] = {
            k: sorted(v.values(), key=lambda se: (se[1], se[0] if se[0] is not None else se[1]))
            for k, v in bucket_panels.items()
        }
        ptr: Dict[tuple, int] = {}

        # FIFO: slice có floor-index sớm nhất nhận panel-xong sớm nhất của bucket.
        def _index_floor(t: Dict[str, Any]) -> int:
            ends = [
                p1_end_times[r]
                for r in (_resolve(d) for d in (t.get("final_depends_on") or []))
                if r is not None
            ]
            return max(ends) if ends else 0

        for t in sorted(slices, key=lambda x: (_index_floor(x), x["task_id"])):
            t_id = t["task_id"]
            current_lb = 0
            wait_offsets = t.get("wait_offsets") or t.get("WaitOffsets") or {}
            consumed_offsets: set = set()
            for dep_id in (t.get("final_depends_on") or []):
                resolved = _resolve(dep_id)
                if not resolved:
                    continue
                # Offset for THIS component (per-component lead time, uniform across
                # its panels) — reused when the slice is re-assigned a bucket sibling.
                off = wait_offsets.get(dep_id)
                if off is None:
                    off = wait_offsets.get(resolved)
                if dep_id in wait_offsets:
                    consumed_offsets.add(dep_id)
                if resolved in wait_offsets:
                    consumed_offsets.add(resolved)

                key = _knit_bucket(dep_id, resolved)
                if key is None:
                    # Not a same-qty-relaxable knitting dep — index behaviour.
                    current_lb = max(current_lb, p1_end_times[resolved])
                    if off is not None and resolved in p1_start_times:
                        current_lb = max(current_lb, p1_start_times[resolved] + int(off))
                    continue

                panels = bucket_sorted[key]
                k = ptr.get(key, 0)
                ptr[key] = k + 1
                p_start, p_end = panels[min(k, len(panels) - 1)]
                current_lb = max(current_lb, p_end)
                if off is not None and p_start is not None:
                    current_lb = max(current_lb, p_start + int(off))

            # Any WaitOffsets not paired to a final_depends_on entry → index floor.
            for raw_batch_id, offset in wait_offsets.items():
                if raw_batch_id in consumed_offsets:
                    continue
                resolved = _resolve(raw_batch_id)
                if resolved and resolved in p1_start_times:
                    current_lb = max(current_lb, p1_start_times[resolved] + int(offset))

            if current_lb > 0:
                lb[t_id] = current_lb

    return lb
