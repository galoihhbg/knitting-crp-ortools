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
        "color": "red",
        "substance": "cotton",
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


def test_washing_task_does_not_cross_shift_boundary():
    """
    Test: 1 washing task, duration=80, shift_ends_min=[100].
    Task starting at 0 ends at 80 ≤ 100 → fits in shift 1.
    If started late (e.g. start=50), end=130 > 100 → must be pushed to start ≥ 100.
    Expectation: end_time ≤ 100 OR start_time ≥ 100 (never straddling boundary).
    """
    payload = {
        "job_id": "test_shift_boundary_single",
        "config": _make_config(shift_ends_min=[100]),
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            _make_washing_task("W1", "ORD_1", 80, 2000, ["WM_00"]),
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    a = result["assignments"][0]
    assert a["end_time"] <= 100 or a["start_time"] >= 100, (
        f"Task spans shift boundary 100min: start={a['start_time']}, end={a['end_time']}"
    )


def test_washing_task_pushed_past_boundary_when_too_long_for_remaining_shift():
    """
    Test: 1 washing task, duration=150, shift_ends_min=[100].
    Duration > boundary → impossible to end before 100 → must start ≥ 100.
    Expectation: start_time ≥ 100.
    """
    payload = {
        "job_id": "test_shift_boundary_pushed",
        "config": _make_config(shift_ends_min=[100]),
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            _make_washing_task("W1", "ORD_1", 150, 2000, ["WM_00"]),
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    a = result["assignments"][0]
    assert a["start_time"] >= 100, (
        f"Task duration=150 cannot end before boundary 100 — expected start≥100, got {a['start_time']}"
    )


def test_no_shift_ends_is_noop():
    """
    Test: shift_ends_min not set (empty list).
    Expectation: apply_shift_boundary_constraints() is a no-op; schedule unchanged.
    """
    payload = {
        "job_id": "test_shift_boundary_noop",
        "config": _make_config(),  # no shift_ends_min
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            _make_washing_task("W1", "ORD_1", 60, 2000, ["WM_00"]),
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 1


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


def test_two_washing_machines_run_parallel():
    """
    Test: Option A — Go lists both WASH_A and WASH_B in compatible_resource_ids.
    Each task has qty=5, capacity=5 per machine → one task fills a machine completely.

    With 2 tasks (qty=5 each):
    - They CANNOT share the same slot (5+5=10 > capacity=5).
    - They CANNOT both start at T=0 on the same machine (AddCumulative demand=10 > 5).
    - With tight deadline=65 (duration=60), sequential on one machine → W2 ends at 120
      (misses deadline by 55 min). Parallel on two machines → both end at 60 ≤ 65.
    - Therefore: solver MUST assign them to different machines to meet the deadline.

    Without Option A (only one machine listed per task), this would be infeasible or
    the second task would be forced to wait and miss its deadline.
    """
    payload = {
        "job_id": "test_two_machines_parallel",
        "config": _make_config(washing_batch_capacity=5, max_search_time=30),
        "machines": [_make_machine("WASH_A"), _make_machine("WASH_B")],
        "resources": [_make_resource("WASH_A"), _make_resource("WASH_B")],
        "tasks": [
            {**_make_washing_task("W1", "ORD_1", 60, 65, ["WASH_A", "WASH_B"]), "qty": 5.0},
            {**_make_washing_task("W2", "ORD_2", 60, 65, ["WASH_A", "WASH_B"]), "qty": 5.0},
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")

    by_id = {a["task_id"]: a for a in result["assignments"]}
    w1, w2 = by_id["W1"], by_id["W2"]

    assert w1["machine_id"] != w2["machine_id"], (
        f"W1 ({w1['machine_id']}) and W2 ({w2['machine_id']}) are on the same machine. "
        "With qty=capacity=5, each task fills a machine — they must be on different machines."
    )
    assert w1["end_time"] <= 65, f"W1 missed deadline: end={w1['end_time']}"
    assert w2["end_time"] <= 65, f"W2 missed deadline: end={w2['end_time']}"


def test_pinned_task_does_not_consume_washing_slot():
    """
    Bug: Pinned washing tasks consumed batch slots → infeasible when
    washing_num_slots was sized only for free tasks.

    Setup: 1 pinned task (already scheduled) + 2 free tasks, washing_num_slots=2.
    Before fix: K=min(3,2)=2 slots; pinned task takes slot 0, only 1 slot left
                for 2 free tasks → infeasible.
    After fix:  K computed from free tasks only → K=2 slots for 2 free tasks → feasible.
    """
    pinned = {
        **_make_washing_task("W_pinned", "ORD_P", 60, 2000, ["WM_00"]),
        "is_pinned": True,
        "pinned_machine_id": "WM_00",
        "pinned_start_time": 0,
        "pinned_end_time": 60,
    }
    payload = {
        "job_id": "test_pinned_no_slot_consumption",
        "config": _make_config(washing_batch_capacity=5, washing_num_slots=2),
        "machines": [_make_machine("WM_00")],
        "resources": [_make_resource("WM_00")],
        "tasks": [
            pinned,
            _make_washing_task("W1", "ORD_1", 60, 2000, ["WM_00"]),
            _make_washing_task("W2", "ORD_2", 60, 2000, ["WM_00"]),
        ],
    }
    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal"), (
        "Model infeasible — pinned task is likely consuming a washing slot"
    )
    task_ids = {a["task_id"] for a in result["assignments"]}
    assert "W1" in task_ids, "W1 (free task) not scheduled"
    assert "W2" in task_ids, "W2 (free task) not scheduled"


def test_two_group_downstream_gets_correct_start_lb():
    """
    Bug: When 2 washing groups exist and one group has no schedulable tasks
    (e.g., all pinned), the other group's downstream tasks incorrectly
    started at T=0 because end_times were missing for the empty group.

    Setup:
    - Group A (red, cotton): 1 free washing task W_A (duration=60)
    - Group B (blue, poly):  1 pinned washing task W_B (start=0, end=60)
    - Downstream task D_B depends on W_B (final_depends_on=["W_B"])

    After fix: W_B's pinned end_time (60) is in end_times even though
               it bypasses slot assignment → D_B gets start_lb=60.
    """
    from app.engine.phases.phase3_batching import solve_washing

    w_a = {
        **_make_washing_task("W_A", "ORD_A", 60, 2000, ["WM_00"]),
        "color": "red",
        "substance": "cotton",
    }
    w_b = {
        **_make_washing_task("W_B", "ORD_B", 60, 2000, ["WM_00"]),
        "color": "blue",
        "substance": "poly",
        "is_pinned": True,
        "pinned_machine_id": "WM_00",
        "pinned_start_time": 0,
        "pinned_end_time": 60,
    }
    config = _make_config(washing_batch_capacity=5)
    resources = [_make_resource("WM_00")]

    result = solve_washing(
        tasks=[w_a, w_b],
        resources=resources,
        config=config,
        p2_end_times={},
        shift_ends=[],
        horizon=5000,
    )

    assert result.status in ("feasible", "empty")
    # W_B must be in end_times so downstream tasks can use it as start_lb
    assert "W_B" in result.end_times, (
        f"W_B (pinned) missing from end_times={result.end_times}. "
        "Downstream tasks depending on W_B would start at T=0."
    )
    assert result.end_times["W_B"] == 60, (
        f"Expected W_B end_time=60 (pinned_end_time), got {result.end_times['W_B']}"
    )
