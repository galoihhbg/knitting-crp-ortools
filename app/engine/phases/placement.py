"""Primitives đặt-chỗ dùng chung cho các post-pass tất định (A1a).

Hợp nhất các bản trùng lặp giữa phase1_knitting / phase2_linking /
phase3_batching / phase4_downstream:

  * ``overlaps``            — 8 call site của predicate chồng-lấn interval
  * ``earliest_sweep``      — phase4 ``_earliest_slot`` ≡ phase1 ``_earliest_gap``
                              ≡ phase2 ``_earliest`` (tương đương chứng minh được)
  * ``earliest_candidates`` — phase3 ``_earliest`` (boundary-safe, limit) và
                              phase1 ``_earliest_nonfrag_start`` (guard re-entry)
                              qua tham số ``extra_candidates``/``limit``/``accept``
  * ``bump_earliest``       — vòng bump-past-overlap của phase1
                              ``_machine_earliest`` (nguyên văn)
  * ``release_from_deps``   — phase4 ``_release`` (3 bản) + ready-time washing
                              (``include_start_after=False``)
  * ``unavail_windows`` / ``avail_at`` — 4 bản đọc lịch nghỉ / mốc sẵn sàng máy
  * ``relabel_balance``     — hợp nhất ``balance_linking_load`` (phase2) và
                              ``balance_downstream_load`` (phase4), tham số hoá
                              theo ``ops``/``label``

KỶ LUẬT BYTE-IDENTICAL: mỗi hàm tái tạo NGUYÊN VĂN hành vi của các bản gốc mà
nó thay thế (kiểm chứng bằng golden corpus + fuzz đối chiếu trong
tests/test_placement_helpers.py).  Semantics đặc thù phase — workforce cap,
batch/cycle giặt, FIFO-theo-PO floor, re-entry contiguity, ranh giới ca — Ở LẠI
pass gọi và đi vào đây qua tham số/hook, không bị "chuẩn hoá" ngầm.

Module này KHÔNG tự đọc unavailability/available_at_min từ resources khi tìm
slot: busy-map do caller dựng (pass nào đang bỏ qua downtime của máy thì tiếp
tục bỏ qua — vá lỗ hổng đó là việc của A2, làm ở đây sẽ phá byte-identical).
"""
import logging
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple,
)

logger = logging.getLogger(__name__)


def overlaps(s: int, e: int, bs: int, be: int) -> bool:
    """[s, e) chồng lấn [bs, be) — predicate chung của mọi kiểm tra overlap."""
    return s < be and bs < e


def earliest_sweep(busy: List[Tuple[int, int]], release: int, dur: int) -> int:
    """Earliest start ≥ release cho task dài `dur` trên máy serial có các interval
    bận `busy` (đã sort theo start, dạng [s, e)).  Chui vào gap đầu tiên đủ chỗ;
    hết gap thì nối đuôi.

    Nguyên văn phase4 ``_earliest_slot`` / phase1 ``_earliest_gap``; phase2
    ``_earliest`` viết khác thứ tự điều kiện nhưng tương đương trên input sorted
    (kể cả interval chồng nhau và dur=0) — đối chiếu fuzz trong test.
    """
    t = release
    for s, e in busy:
        if e <= t:
            continue          # interval nằm trọn trước ứng viên hiện tại
        if s >= t + dur:
            break             # gap [t, t+dur) lọt trước interval này
        t = max(t, e)         # chồng lấn → đẩy qua
    return t


def earliest_candidates(
    busy: List[Tuple[int, int]],
    release: int,
    dur: int,
    *,
    extra_candidates: Iterable[int] = (),
    limit: Optional[int] = None,
    accept: Optional[Callable[[int, int], bool]] = None,
) -> Optional[int]:
    """Earliest start hợp lệ theo kiểu QUÉT-ỨNG-VIÊN: thử ``release``, các điểm
    kết thúc interval bận, và ``extra_candidates`` (vd. ranh giới ca) theo thứ
    tự tăng dần; nhận điểm đầu tiên (a) không chồng interval bận nào và (b) qua
    hook ``accept(s, e)`` nếu có.  Trả None khi không có điểm nào đạt.

    ``limit`` là chặn TRÊN nghiêm ngặt (chỉ xét ứng viên < limit) — dùng cho
    luật "phải sớm hơn strict vị trí hiện tại" của washing left-shift.

    Thay cho: phase3 ``_earliest`` (extra_candidates=shift_bounds, limit=start
    cycle hiện tại, accept=không-vắt-qua-ca) và phase1
    ``_earliest_nonfrag_start`` (accept=không tăng re-entry đơn).  Hai bản gốc
    sinh tập ứng viên bằng biểu thức khác nhau nhưng cùng tập kết quả sau lọc
    ``release ≤ c [< limit]``; thứ tự hai phép loại (overlap trước hay accept
    trước) không đổi ứng viên được trả.
    """
    cands = {release}
    for _bs, be in busy:
        cands.add(be)
    for c in extra_candidates:
        cands.add(c)
    for s in sorted(
        c for c in cands if c >= release and (limit is None or c < limit)
    ):
        e = s + dur
        if any(overlaps(s, e, bs, be) for bs, be in busy):
            continue
        if accept is not None and not accept(s, e):
            continue
        return s
    return None


def bump_earliest(busy: List[Tuple[int, int]], start: int, dur: int) -> int:
    """Vòng bump-past-overlap: đẩy ``start`` qua mọi interval bận đụng phải, lặp
    tới bất động điểm.  Chịu được ``busy`` CHƯA sort / chồng lẫn nhau, và giữ
    nguyên hành vi dur=0 của bản gốc (chỉ bump khi start nằm HẲN TRONG interval,
    không bump khi chạm mép) — bản gốc được dùng với dur=0 làm khoá sort máy.

    Nguyên văn lõi phase1 ``_machine_earliest``; caller tự seed
    ``start = max(release, avail_at)`` như bản gốc.
    """
    st = start
    moved = True
    while moved:
        moved = False
        for ws, we in busy:
            if st < we and st + dur > ws:
                st = we
                moved = True
    return st


def release_from_deps(
    task: Dict[str, Any],
    dep_ends: Dict[str, int],
    *,
    include_start_after: bool = True,
) -> int:
    """Release floor = max(start_after_min?, end các dependency đã biết).

    ``include_start_after=True``: nguyên văn phase4 ``_release`` (iron/pack/
    fifo_swap — 3 bản copy).  ``include_start_after=False``: nguyên văn cách
    tính ``ready_of`` của washing (flush + left-shift) — CHỦ Ý bỏ qua
    start_after_min như bản gốc (quirk giữ nguyên, xem danh sách candidate-fix
    A1a; sửa nó là đổi hành vi).
    """
    rel = int(task.get("start_after_min", 0) or 0) if include_start_after else 0
    for d in (task.get("final_depends_on") or []):
        if d in dep_ends:
            rel = max(rel, int(dep_ends[d]))
    return rel


def unavail_windows(resource: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Các cửa sổ nghỉ (start, end) hợp lệ của một resource — 4 bản gốc ở
    phase2/phase4 (``_unavail``)."""
    return [
        (int(w["start"]), int(w["end"]))
        for w in (resource.get("unavailability") or [])
        if int(w["end"]) > int(w["start"])
    ]


def avail_at(resource: Dict[str, Any]) -> int:
    """Mốc sẵn sàng của resource — 4 bản gốc ở phase2/phase4 (``_avail_at``)."""
    return int(resource.get("available_at_min", 0) or 0)


def relabel_balance(
    assignments: List[Dict[str, Any]],
    all_tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    ops: frozenset,
    label: str,
) -> int:
    """Cân tải bằng ĐỔI NHÃN MÁY, thời gian đóng băng — hợp nhất nguyên văn
    ``balance_linking_load`` (phase2, ops=linking, label="Linking") và
    ``balance_downstream_load`` (phase4, ops=iron/pack).

    GIỮ NGUYÊN [start, end] của mọi task, chỉ gán lại máy bằng greedy "tô màu
    interval": duyệt theo (start, end, task_id), đặt mỗi task lên máy hợp-lệ
    (compatible, rảnh trong [start, end], ngoài unavailability, sau
    available_at_min) có TẢI HIỆN TẠI THẤP NHẤT (tie-break machine id).  Vì
    thời gian không đổi: downstream byte-identical, lateness không đổi,
    no-overlap giữ nguyên.  Pinned tasks là mỏ neo bất động (giữ máy, chiếm
    chỗ trước).  Khi không máy nào eligible → GIỮ máy hiện tại không re-check
    (quirk bản gốc, giữ nguyên — xem danh sách candidate-fix A1a).
    Deterministic O(n log n).  Mutates ``assignments`` in place (machine_id).
    Returns số task được đổi máy.
    """
    info = {t["task_id"]: t for t in all_tasks}
    machine_ids = [
        r["id"] for r in resources if str(r.get("operation", "")).lower() in ops
    ]
    if len(machine_ids) < 2:
        return 0
    res_by_id = {r["id"]: r for r in resources}

    op_assigns = [
        a for a in assignments
        if str(info.get(a["task_id"], {}).get("operation", "")).lower() in ops
        and a.get("machine_id")
    ]
    if len(op_assigns) < 2:
        return 0

    busy: Dict[str, List[Any]] = {
        m: list(unavail_windows(res_by_id.get(m, {}))) for m in machine_ids
    }
    load: Dict[str, int] = {m: 0 for m in machine_ids}

    def _free(m_id: str, s: int, e: int) -> bool:
        if s < avail_at(res_by_id.get(m_id, {})):
            return False
        return not any(overlaps(s, e, bs, be) for bs, be in busy[m_id])

    # Pinned tasks are immovable: pre-place them on their current machine.
    movable: List[Dict[str, Any]] = []
    for a in op_assigns:
        t = info.get(a["task_id"], {})
        if t.get("is_pinned"):
            m = a["machine_id"]
            busy.setdefault(m, []).append((a["start_time"], a["end_time"]))
            load[m] = load.get(m, 0) + (a["end_time"] - a["start_time"])
        else:
            movable.append(a)

    changed = 0
    # Deterministic order: by start, then end, then task_id.
    for a in sorted(movable, key=lambda x: (x["start_time"], x["end_time"], x["task_id"])):
        s, e = a["start_time"], a["end_time"]
        t = info.get(a["task_id"], {})
        compat = set(t.get("compatible_resource_ids") or []) & set(machine_ids)
        cands = [m for m in machine_ids if m in compat] if compat else list(machine_ids)
        eligible = [m for m in cands if _free(m, s, e)]
        if not eligible:
            # Keep current machine (must remain feasible there — it was before).
            m = a["machine_id"]
        else:
            m = min(eligible, key=lambda x: (load[x], x))
        if m != a["machine_id"]:
            changed += 1
        a["machine_id"] = m
        busy.setdefault(m, []).append((s, e))
        load[m] = load.get(m, 0) + (e - s)

    if changed:
        used = sum(1 for m in machine_ids if load.get(m, 0) > 0)
        logger.info(
            f"⚖️ {label} load-balance: re-assigned {changed} task(s) → "
            f"{used}/{len(machine_ids)} machines used (timing unchanged)."
        )
    return changed
