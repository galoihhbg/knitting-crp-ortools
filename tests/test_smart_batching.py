"""
Smart batching tests — verify washing task batch assignment logic.
"""
import pytest
from typing import Dict, Any, List
from app.engine.model import Engine


def _make_machine(m_id: str, design: str = "D1", color: str = "Black") -> Dict[str, Any]:
    """Create a machine dict."""
    return {
        "id": m_id,
        "capacity": 1,
        "type": "serial",
        "worker_req": 1,
        "routing": [],
        "design_item_id": design,
        "color_config": color,
    }


def _make_resource(r_id: str, op: str = "washing") -> Dict[str, Any]:
    """Create a resource dict."""
    return {
        "id": r_id,
        "type": "serial",
        "capacity": 1,
        "operation": op,
        "unavailability": [],
        "design_item_id": "",
        "color_config": "",
        "available_at_min": 0,
    }


def _make_washing_task(
    task_id: str,
    order_id: str,
    duration: int,
    due: int,
    resource_ids: List[str],
    depends_on: List[str] = None,
) -> Dict[str, Any]:
    """Create a washing task dict."""
    return {
        "task_id": task_id,
        "original_order_id": order_id,
        "group_id": order_id,
        "operation": "washing",
        "qty": 1.0,
        "total_qty": 1.0,
        "priority": 3,
        "original_depends_on": [],
        "final_depends_on": depends_on or [],
        "start_after_min": 0,
        "due_at_min": due,
        "duration": duration,
        "is_slice": False,
        "parent_task_id": "",
        "internal_dep": "",
        "slice_index": 0,
        "is_batch": False,
        "sub_tasks": None,
        "design_item_id": "",
        "color_config": "",
        "compatible_resource_ids": resource_ids,
        "sub_task_completion_offsets": None,
        "WaitOffsets": None,
        "is_pinned": False,
        "pinned_machine_id": None,
        "pinned_start_time": None,
        "pinned_end_time": None,
        "demand": 1,
        "material_demands": {},
    }


def _make_config(**overrides) -> Dict[str, Any]:
    """Create a solver config dict."""
    cfg = {
        "horizon_minutes": 5000,
        "max_search_time": 20,
        "setup_time_minutes": 0,
        "max_factory_machines": 5,
        "random_seed": 42,
        "num_search_workers": 1,
        "washing_batch_capacity": 3,
    }
    cfg.update(overrides)
    return cfg


def test_batch_capacity_respected():
    """
    Test: 4 washing tasks, capacity=2.
    Expectation: No batch slot has more than 2 tasks.
    """
    payload = {
        "job_id": "test_batch_capacity",
        "config": _make_config(washing_batch_capacity=2),
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            _make_washing_task(f"W{i}", f"ORDER_{i}", 60, 2000, ["WM_00"])
            for i in range(4)
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 4

    # Count tasks per batch slot
    from collections import Counter
    slot_counts = Counter(
        a["batch_slot_id"] for a in result["assignments"]
        if a["batch_slot_id"]
    )
    for slot_id, count in slot_counts.items():
        assert count <= 2, (
            f"Slot {slot_id} has {count} tasks but capacity=2"
        )


def test_urgent_order_not_forced_into_batch():
    """
    Test: 3 washing tasks, capacity=3 (all could fit in one batch).
    Task W_urgent has a very tight deadline while W_late1/W_late2 are flexible.
    Expectation: W_urgent gets its own early batch slot rather than being forced
    to wait for others and missing its deadline.
    """
    payload = {
        "job_id": "test_urgent_alone",
        "config": _make_config(washing_batch_capacity=3, max_search_time=30),
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            _make_washing_task("W_urgent", "ORD_urgent", 60, 100, ["WM_00"]),
            _make_washing_task("W_late1", "ORD_late1", 60, 4000, ["WM_00"]),
            _make_washing_task("W_late2", "ORD_late2", 60, 4000, ["WM_00"]),
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")

    by_id = {a["task_id"]: a for a in result["assignments"]}
    urgent = by_id["W_urgent"]
    late1 = by_id["W_late1"]
    late2 = by_id["W_late2"]

    # W_urgent must meet its deadline or be grouped separately
    assert urgent["end_time"] <= 100 or urgent["batch_slot_id"] not in (
        late1["batch_slot_id"],
        late2["batch_slot_id"],
    ), "W_urgent was batched with late orders and missed its deadline"


def test_multiple_batches_formed():
    """
    Test: 6 washing tasks, capacity=2.
    Expectation: Must form at least 3 active batch slots.
    """
    payload = {
        "job_id": "test_multi_batch",
        "config": _make_config(washing_batch_capacity=2),
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            _make_washing_task(f"W{i}", f"ORD_{i}", 60, 3000, ["WM_00"])
            for i in range(6)
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 6

    used_slots = {a["batch_slot_id"] for a in result["assignments"] if a["batch_slot_id"]}
    assert len(used_slots) >= 3, (
        f"6 tasks with capacity=2 need at least 3 slots, got {len(used_slots)}: {used_slots}"
    )


def test_no_washing_tasks_is_noop():
    """
    Test: No washing tasks (only non-washing operation).
    Expectation: apply_smart_batching_constraints() is a no-op;
    batch_slot_id should be empty for non-washing tasks.
    """
    payload = {
        "job_id": "test_no_washing",
        "config": _make_config(),
        "machines": [_make_machine("KM_00")],
        "resources": [_make_resource("KM_00", op="knitting")],
        "tasks": [
            {
                **_make_washing_task("K1", "ORD_A", 100, 2000, ["KM_00"]),
                "operation": "knitting",
            }
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 1
    assert result["assignments"][0]["batch_slot_id"] == ""


def test_batched_tasks_share_start_time():
    """
    Test: 2 washing tasks, capacity=2 (may be grouped).
    Expectation: If they share a batch_slot_id, their start_time must be identical.
    """
    payload = {
        "job_id": "test_sync_start",
        "config": _make_config(washing_batch_capacity=2),
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            _make_washing_task("WA", "ORD_A", 60, 2000, ["WM_00"]),
            _make_washing_task("WB", "ORD_B", 60, 2000, ["WM_00"]),
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")

    by_id = {a["task_id"]: a for a in result["assignments"]}
    wa, wb = by_id["WA"], by_id["WB"]

    if wa["batch_slot_id"] and wa["batch_slot_id"] == wb["batch_slot_id"]:
        assert wa["start_time"] == wb["start_time"], (
            f"WA and WB share slot {wa['batch_slot_id']} "
            f"but start at different times: {wa['start_time']} vs {wb['start_time']}"
        )
