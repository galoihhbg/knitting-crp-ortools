"""
BẰNG CHỨNG: Python solver có lập lịch overlap không?

Test này tái hiện CHÍNH XÁC scenario từ log của user:
  - Pinned tasks: PIN_2_BATCH_0-663_* trên DT4VNJL8BAoUnmn
  - Pinned tasks: PIN_4_BATCH_0-664_* trên DT4VNJL8BAoUnmn và DT7hPmDj15YcFQW
  - Free tasks: BATCH_0-665_1..7 (duration=255, 18 candidate machines gồm cả DT4VNJL8BAoUnmn)

Kết quả:
  - Nếu test PASS: Python solver KHÔNG tạo overlap → bug ở Go (cross-run orchestration)
  - Nếu test FAIL: Python solver TẠO RA overlap → bug ở Python
"""
from collections import defaultdict
from typing import Any, Dict, List

from app.engine.model import Engine


# ── Machine IDs từ log thực tế ────────────────────────────────────────────
SK46 = "DT4VNJL8BAoUnmn"
SK41 = "DT7hPmDj15YcFQW"

# 18 candidate machines của BATCH_0-665 (từ log)
BATCH_665_MACHINES = [
    "3suGBUk1pGJNyW", "3w6aVrrwPqLPVA", "3zNJZirUjTB83N", "3zPoNbbjovrcvC",
    "3zS1L1wQNdpo2E", "43dKV5N7jtv5gU", "43fXSTbH9YhgCC", "46qLwvrSVarxit",
    "46sYuMBdJZ4wj4", "46wFfeGYwjichv", "DCxaj83gMUDUiEi", "DCxcw5U222vSZhC",
    "DCxf92rEhgqTy2p", "DSqRArnNUjxzEaG", "DSqTNpAbe9cmWNC", "DT1J3ywzmwnPYV6",
    "DT4TsVRGR3ajyJU", SK46,  # SK46 = DT4VNJL8BAoUnmn nằm trong danh sách!
]


def _resource(r_id: str) -> Dict[str, Any]:
    return {
        "id": r_id, "type": "serial", "capacity": 1,
        "operation": "knitting", "unavailability": [],
        "design_item_id": "", "color_config": "", "available_at_min": 0,
    }


def _pinned(task_id: str, machine_id: str, start: int, end: int,
             order_id: str = "2") -> Dict[str, Any]:
    return {
        "task_id": task_id, "original_order_id": order_id,
        "group_id": "PINNED", "operation": "knitting",
        "qty": 0.0, "total_qty": 0.0, "priority": 5,
        "final_depends_on": [], "start_after_min": 0,
        "due_at_min": end, "duration": end - start,
        "is_batch": False, "sub_tasks": None,
        "design_item_id": "", "color_config": "",
        "compatible_resource_ids": [machine_id],
        "WaitOffsets": None, "is_pinned": True,
        "pinned_machine_id": machine_id,
        "pinned_start_time": start, "pinned_end_time": end,
        "demand": 0, "material_demands": {},
    }


def _free(task_id: str, machines: List[str], duration: int = 255,
           order_id: str = "BATCH_0-665", due: int = 5904) -> Dict[str, Any]:
    return {
        "task_id": task_id, "original_order_id": order_id,
        "group_id": "BATCH_665", "operation": "knitting",
        "qty": 1.0, "total_qty": 7.0, "priority": 3,
        "final_depends_on": [], "start_after_min": 0,
        "due_at_min": due, "duration": duration,
        "is_batch": False, "sub_tasks": None,
        "design_item_id": "", "color_config": "",
        "compatible_resource_ids": machines,
        "WaitOffsets": None, "is_pinned": False,
        "pinned_machine_id": None, "pinned_start_time": None, "pinned_end_time": None,
        "demand": 0, "material_demands": {},
    }


def _capa_block(task_id: str, start: int, end: int, demand: int = 114) -> Dict[str, Any]:
    return {
        "task_id": task_id, "original_order_id": task_id,
        "group_id": "CAPA", "operation": "capacity_block",
        "qty": 0.0, "total_qty": 0.0, "priority": 5,
        "final_depends_on": [], "start_after_min": 0,
        "due_at_min": end, "duration": end - start,
        "is_batch": False, "sub_tasks": None,
        "design_item_id": "", "color_config": "",
        "compatible_resource_ids": [],
        "WaitOffsets": None, "is_pinned": True,
        "pinned_machine_id": None,
        "pinned_start_time": start, "pinned_end_time": end,
        "demand": demand, "material_demands": {},
    }


def _check_no_overlap(assignments: List[Dict]) -> List[str]:
    """Kiểm tra overlap trong output. Trả về danh sách lỗi (rỗng = không có overlap)."""
    machine_windows: Dict[str, List] = defaultdict(list)
    for a in assignments:
        m = a.get("machine_id")
        s, e = a.get("start_time", 0), a.get("end_time", 0)
        if m and e > s:
            machine_windows[m].append((s, e, a["task_id"]))

    errors = []
    for m_id, windows in machine_windows.items():
        windows.sort()
        for i in range(len(windows) - 1):
            s1, e1, t1 = windows[i]
            s2, e2, t2 = windows[i + 1]
            if s2 < e1:
                errors.append(
                    f"OVERLAP trên machine '{m_id}': "
                    f"'{t1}'[{s1},{e1}) vs '{t2}'[{s2},{e2}) — overlap [{s2},{min(e1,e2)})"
                )
    return errors


def _solve(tasks, resources, max_machines=134):
    payload = {
        "job_id": "frozen_pin_overlap_proof",
        "config": {
            "horizon_minutes": 6500,
            "max_search_time": 60,
            # 134 = giá trị thực tế từ docker logs (capacity=134, headroom=20, demand=114)
            "max_factory_machines": max_machines,
            "random_seed": 42,
            "num_search_workers": 1,
        },
        "machines": [{"id": r["id"], "design_item_id": "", "color_config": ""} for r in resources],
        "resources": resources,
        "tasks": tasks,
        "material_capacities": {},
    }
    return Engine(payload).solve()


# ════════════════════════════════════════════════════════════════════════════
# TEST CHÍNH: Tái hiện FrozenPin job từ log thực tế
# ════════════════════════════════════════════════════════════════════════════

def test_frozen_pin_no_overlap_on_sk46():
    """
    BẰNG CHỨNG TRỰC TIẾP: Python solver có lập lịch BATCH_0-665 overlap
    với các pinned tasks trên SK46 (DT4VNJL8BAoUnmn) không?

    Pinned trên SK46 (từ log):
      [0, 160)  — PIN_2_BATCH_0-663_1_0
      [160, 252) — PIN_2_BATCH_0-663_1_1
      [252, 400) — PIN_4_BATCH_0-664_1_10
      [400, 507) — PIN_4_BATCH_0-664_1_11
      [1017, 1272) — PIN_2_BATCH_0-663_5_8+9
      [1272, 1527) — PIN_4_BATCH_0-664_4_16+17
      [1782, 2037) — PIN_4_BATCH_0-664_8_24+25
      [2037, 2292) — PIN_2_BATCH_0-663_4_6+7
      [2802, 3057) — PIN_4_BATCH_0-664_5_18+19

    BATCH_0-665_* (free, duration=255) có SK46 trong candidate list.
    Expected: KHÔNG CÓ overlap nào.

    Nếu test PASS → Python solver đúng → lỗi ở Go orchestration.
    Nếu test FAIL → Python solver sai → lỗi ở Python.
    """
    # Tất cả resources (18 machine candidates + SK41)
    all_machines = list(set(BATCH_665_MACHINES + [SK41]))
    resources = [_resource(m) for m in all_machines]

    # === PINNED TASKS trên SK46 ===
    pinned_sk46 = [
        _pinned("PIN_2_BATCH_0-663_1_0",  SK46,  0,    160,  "2"),
        _pinned("PIN_2_BATCH_0-663_1_1",  SK46,  160,  252,  "2"),
        _pinned("PIN_4_BATCH_0-664_1_10", SK46,  252,  400,  "4"),
        _pinned("PIN_4_BATCH_0-664_1_11", SK46,  400,  507,  "4"),
        _pinned("PIN_2_BATCH_0-663_5_8",  SK46,  1017, 1060, "2"),
        _pinned("PIN_2_BATCH_0-663_5_9",  SK46,  1060, 1272, "2"),
        _pinned("PIN_4_BATCH_0-664_4_16", SK46,  1272, 1300, "4"),
        _pinned("PIN_4_BATCH_0-664_4_17", SK46,  1300, 1527, "4"),
        _pinned("PIN_4_BATCH_0-664_8_24", SK46,  1782, 1960, "4"),
        _pinned("PIN_4_BATCH_0-664_8_25", SK46,  1960, 2037, "4"),
        _pinned("PIN_2_BATCH_0-663_4_6",  SK46,  2037, 2200, "2"),
        _pinned("PIN_2_BATCH_0-663_4_7",  SK46,  2200, 2292, "2"),
        _pinned("PIN_4_BATCH_0-664_5_18", SK46,  2802, 2860, "4"),
        _pinned("PIN_4_BATCH_0-664_5_19", SK46,  2860, 3057, "4"),
    ]

    # === PINNED TASKS trên SK41 ===
    pinned_sk41 = [
        _pinned("PIN_2_BATCH_0-663_2_2",  SK41,  0,    160,  "2"),
        _pinned("PIN_2_BATCH_0-663_2_3",  SK41,  160,  252,  "2"),
        _pinned("PIN_2_BATCH_0-663_3_4",  SK41,  252,  400,  "2"),
        _pinned("PIN_2_BATCH_0-663_3_5",  SK41,  400,  507,  "2"),
        _pinned("PIN_4_BATCH_0-664_3_14", SK41,  1017, 1060, "4"),
        _pinned("PIN_4_BATCH_0-664_3_15", SK41,  1060, 1272, "4"),
        _pinned("PIN_4_BATCH_0-664_7_22", SK41,  1272, 1300, "4"),
        _pinned("PIN_4_BATCH_0-664_7_23", SK41,  1300, 1527, "4"),
        _pinned("PIN_4_BATCH_0-664_2_12", SK41,  2292, 2440, "4"),
        _pinned("PIN_4_BATCH_0-664_2_13", SK41,  2440, 2547, "4"),
        _pinned("PIN_4_BATCH_0-664_6_20", SK41,  2802, 2860, "4"),
        _pinned("PIN_4_BATCH_0-664_6_21", SK41,  2860, 3057, "4"),
    ]

    # === CAPACITY BLOCKS (một số từ log, đại diện) ===
    capa_blocks = [
        _capa_block("CAPA_BLOCK_CA SANG_2026-05-04", 0,    160,  114),
        _capa_block("CAPA_BLOCK_CA TOI_2026-05-04",  160,  640,  114),
        _capa_block("CAPA_BLOCK_CA SANG_2026-05-05", 640,  1060, 114),
        _capa_block("CAPA_BLOCK_CA TOI_2026-05-05",  1060, 1540, 114),
        _capa_block("CAPA_BLOCK_CA SANG_2026-05-06", 1540, 1960, 114),
        _capa_block("CAPA_BLOCK_CA TOI_2026-05-06",  1960, 2440, 114),
        _capa_block("CAPA_BLOCK_CA SANG_2026-05-07", 2440, 2860, 114),
        _capa_block("CAPA_BLOCK_CA TOI_2026-05-07",  2860, 3340, 114),
        _capa_block("CAPA_BLOCK_CA SANG_2026-05-08", 3340, 3760, 114),
        _capa_block("CAPA_BLOCK_CA TOI_2026-05-08",  3760, 4240, 114),
        _capa_block("CAPA_BLOCK_CA SANG_2026-05-09", 4240, 4660, 114),
        _capa_block("CAPA_BLOCK_CA TOI_2026-05-09",  4660, 5140, 114),
        _capa_block("CAPA_BLOCK_CA SANG_2026-05-10", 5140, 5560, 114),
        _capa_block("CAPA_BLOCK_CA TOI_2026-05-10",  5560, 6040, 114),
    ]

    # === FREE TASKS: BATCH_0-665 (7 slices, duration=255 mỗi cái) ===
    free_tasks = [
        _free(f"BATCH_0-665_{i}", BATCH_665_MACHINES, duration=255, due=5904)
        for i in range(1, 8)
    ]

    all_tasks = pinned_sk46 + pinned_sk41 + capa_blocks + free_tasks

    print(f"\n📋 Tổng tasks: {len(all_tasks)} "
          f"({len(pinned_sk46)+len(pinned_sk41)} pinned knitting, "
          f"{len(capa_blocks)} capacity_block, {len(free_tasks)} free)")
    print(f"📌 Pinned trên SK46: {len(pinned_sk46)} segments "
          f"→ occupied: [0,507), [1017,1527), [1782,2292), [2802,3057)")
    print(f"📌 Gaps trên SK46: [507,1017)=510min, [1527,1782)=255min, "
          f"[2292,2802)=510min, [3057,…)")

    result = _solve(all_tasks, resources)

    print(f"\n🔍 Solver status: {result['status']}")
    assert result["status"] == "feasible", (
        f"Solver INFEASIBLE — kiểm tra capacity_block demand vs max_factory_machines. "
        f"Overloads: {result.get('overloads', [])[:3]}"
    )

    assignments = result["assignments"]
    assigns_by_id = {a["task_id"]: a for a in assignments}

    # In ra assignments của free tasks
    print("\n📊 Assignments BATCH_0-665_*:")
    for i in range(1, 8):
        t_id = f"BATCH_0-665_{i}"
        if t_id in assigns_by_id:
            a = assigns_by_id[t_id]
            on_sk46 = a["machine_id"] == SK46
            print(f"  {t_id}: machine={a['machine_id']} "
                  f"[{a['start_time']},{a['end_time']}) "
                  f"{'⚠️ TRÊN SK46!' if on_sk46 else ''}")
        else:
            print(f"  {t_id}: KHÔNG ĐƯỢC LẬP LỊCH!")

    # Kiểm tra tất cả free tasks được lên lịch
    missing = [f"BATCH_0-665_{i}" for i in range(1, 8) if f"BATCH_0-665_{i}" not in assigns_by_id]
    assert not missing, f"Các free tasks KHÔNG được lập lịch: {missing}"

    # === KIỂM TRA OVERLAP ===
    errors = _check_no_overlap(assignments)

    if errors:
        print("\n❌ OVERLAP PHÁT HIỆN TRONG OUTPUT PYTHON SOLVER:")
        for e in errors:
            print(f"  {e}")
        print("\n➡️  KẾT LUẬN: BUG NẰM Ở PYTHON SOLVER (OR-Tools model)")
    else:
        print("\n✅ KHÔNG CÓ OVERLAP trong output Python solver!")
        print("➡️  KẾT LUẬN: Python solver ĐÚNG.")
        print("    Nếu Go vẫn báo overlap → BUG NẰM Ở GO (cross-run orchestration):")
        print("    Results của run N chưa được pin vào run N+1 đúng cách.")

    assert not errors, (
        f"Python solver TẠO RA {len(errors)} overlap(s):\n" +
        "\n".join(errors)
    )


def test_frozen_pin_free_tasks_avoid_pinned_windows():
    """
    Kiểm tra chi tiết: các free task được gán cho SK46 PHẢI nằm trong gap,
    không được đè lên pinned windows.

    Gaps hợp lệ trên SK46:
      [507, 1017)   = 510 phút  → đủ chỗ cho 2 task × 255 phút
      [1527, 1782)  = 255 phút  → đủ chỗ cho 1 task × 255 phút
      [2292, 2802)  = 510 phút  → đủ chỗ cho 2 task × 255 phút
      [3057, ∞)                 → đủ chỗ cho nhiều task
    """
    resources = [_resource(m) for m in BATCH_665_MACHINES + [SK41]]

    # Pinned windows trên SK46 (gộp lại cho đơn giản)
    pinned_windows_sk46 = [
        (0,    507),   # 0-160-252-400-507
        (1017, 1527),  # 1017-1060-1272-1300-1527
        (1782, 2292),  # 1782-1960-2037-2200-2292
        (2802, 3057),  # 2802-2860-3057
    ]

    pinned_tasks = []
    for i, (s, e) in enumerate(pinned_windows_sk46):
        pinned_tasks.append(_pinned(f"BLOCK_SK46_{i}", SK46, s, e, "X"))
    for i, (s, e) in enumerate([(0, 507), (1017, 1527), (2292, 2547), (2802, 3057)]):
        pinned_tasks.append(_pinned(f"BLOCK_SK41_{i}", SK41, s, e, "X"))

    free_tasks = [
        _free(f"FREE_{i}", BATCH_665_MACHINES, duration=255, due=5904)
        for i in range(1, 8)
    ]

    result = _solve(pinned_tasks + free_tasks, resources)
    assert result["status"] == "feasible", f"Infeasible: {result.get('overloads', [])[:2]}"

    errors = _check_no_overlap(result["assignments"])

    sk46_assignments = [
        a for a in result["assignments"]
        if a.get("machine_id") == SK46 and not a["task_id"].startswith("BLOCK")
    ]

    print(f"\n📊 Free tasks được gán SK46: {len(sk46_assignments)}")
    for a in sk46_assignments:
        in_gap = True
        for ws, we in pinned_windows_sk46:
            if a["start_time"] < we and a["end_time"] > ws:
                in_gap = False
                print(f"  ❌ {a['task_id']} [{a['start_time']},{a['end_time']}) "
                      f"→ ĐÈ LÊN pinned [{ws},{we})!")
                break
        if in_gap:
            print(f"  ✅ {a['task_id']} [{a['start_time']},{a['end_time']}) → nằm trong gap")

    assert not errors, f"Python solver tạo overlap:\n" + "\n".join(errors)
