"""Đơn 'normal' (is_normal=True): vẫn nhắm dueDate nhưng KHÔNG bị check khắt khe.

Hai thay đổi được kiểm chứng ở đây, đều ở app/engine/shared.py:
  1. apply_soft_deadlines — đơn normal giữ phạt lateness-phút (×weight×100) để vẫn
     nhắm dueDate, NHƯNG bỏ term đếm số-đơn-trễ (is_late × weight × 10) → không tạo
     BoolVar `is_late_<task>` cho task normal.
  2. extract_results — task normal dù end > due KHÔNG được đẩy vào danh sách overloads
     (status assignment vẫn báo LATE trung thực).
"""
from typing import Any, Dict, List

from ortools.sat.python import cp_model

from app.engine.shared import apply_soft_deadlines, extract_results


def _task(task_id: str, *, due: int, is_normal: bool, priority: int = 3) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": f"ORD_{task_id}",
        "group_id": f"G_{task_id}",
        "operation": "knitting",
        "priority": priority,
        "due_at_min": due,
        "qty": 1,
        "is_normal": is_normal,
    }


def _build(tasks: List[Dict[str, Any]], starts: Dict[str, int], horizon: int):
    """Tiny fully-pinned model: each task fixed at a known [start, end] on one machine."""
    model = cp_model.CpModel()
    task_vars: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        tid = t["task_id"]
        s = starts[tid]
        e = s + 10
        start = model.NewIntVar(s, s, f"start_{tid}")
        end = model.NewIntVar(e, e, f"end_{tid}")
        lit = model.NewBoolVar(f"{tid}_on_M0")
        model.Add(lit == 1)
        task_vars[tid] = {
            "start": start, "end": end,
            "literals": [lit], "r_ids": ["M0"],
            "due": t["due_at_min"],
            "group_id": t["group_id"],
            "original_order_id": t["original_order_id"],
            "qty": 1, "is_pinned": False,
        }
    return model, task_vars


def test_normal_late_task_not_reported_as_overload():
    horizon = 10_000
    # Both finish at end=110 (start 100 + dur 10); both due at 50 → both LATE.
    tasks = [
        _task("URGENT", due=50, is_normal=False),
        _task("NORMAL", due=50, is_normal=True),
    ]
    starts = {"URGENT": 100, "NORMAL": 100}
    model, task_vars = _build(tasks, starts, horizon)
    task_map = {t["task_id"]: t for t in tasks}

    model.Minimize(sum(apply_soft_deadlines(model, task_vars, task_map, horizon)))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    status_str, assignments, overloads, _, _ = extract_results(
        solver, status, task_vars, tasks, config={"max_factory_machines": 10}
    )
    assert status_str == "feasible"

    over_ids = {o["task_id"] for o in overloads}
    assert "URGENT" in over_ids, "đơn gấp trễ phải vẫn bị báo overload"
    assert "NORMAL" not in over_ids, "đơn normal trễ KHÔNG được báo overload"

    # Assignment status vẫn trung thực: cả hai đều LATE.
    by_id = {a["task_id"]: a for a in assignments}
    assert by_id["URGENT"]["status"] == "LATE"
    assert by_id["NORMAL"]["status"] == "LATE"


def test_normal_task_has_no_is_late_count_var():
    horizon = 10_000
    tasks = [
        _task("URGENT", due=50, is_normal=False),
        _task("NORMAL", due=50, is_normal=True),
    ]
    starts = {"URGENT": 0, "NORMAL": 0}
    model, task_vars = _build(tasks, starts, horizon)
    task_map = {t["task_id"]: t for t in tasks}

    apply_soft_deadlines(model, task_vars, task_map, horizon)
    proto_names = {v.name for v in model.Proto().variables}

    assert "is_late_URGENT" in proto_names, "đơn gấp phải có term đếm is_late"
    assert "is_late_NORMAL" not in proto_names, "đơn normal KHÔNG được có term đếm is_late"
    # Cả hai vẫn có biến lateness-phút (vẫn nhắm dueDate).
    assert "lat_URGENT" in proto_names
    assert "lat_NORMAL" in proto_names
