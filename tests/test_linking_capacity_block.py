from app.engine.shared import is_capacity_block_op
from app.engine.phases.phase2_linking import left_shift_cold_linking, solve_linking


def test_is_capacity_block_op():
    assert is_capacity_block_op("capacity_block")
    assert is_capacity_block_op("capacity_block_linking")
    assert not is_capacity_block_op("linking")


def _linking_task(task_id: str, machine_ids: list[str]) -> dict:
    return {
        "task_id": task_id,
        "original_order_id": task_id,
        "group_id": task_id,
        "operation": "linking",
        "qty": 1,
        "priority": 1,
        "duration": 100,
        "due_at_min": 100,
        "compatible_resource_ids": machine_ids,
        "final_depends_on": [],
        "wait_offsets": {},
    }


def test_capacity_block_linking_limits_phase2_workforce():
    """A two-worker block leaves at most two of four linking workers available."""
    machine_ids = [f"LM{i}" for i in range(4)]
    resources = [
        {"id": machine_id, "operation": "linking", "type": "serial", "capacity": 1}
        for machine_id in machine_ids
    ]
    tasks = [_linking_task(f"L{i}", machine_ids) for i in range(3)]
    tasks.append(
        {
            "task_id": "BLOCK",
            "operation": "capacity_block_linking",
            "duration": 480,
            "demand": 2,
            "is_pinned": True,
            "pinned_start_time": 0,
            "pinned_end_time": 480,
            "compatible_resource_ids": [],
        }
    )

    result = solve_linking(
        tasks,
        resources,
        {"horizon_minutes": 1000, "max_search_time": 2},
        p1_start_times={},
        p1_end_times={},
        translation_map={},
        horizon=1000,
    )

    assert result.status == "feasible"
    event_points = {
        time
        for assignment in result.assignments
        for time in (assignment["start_time"], assignment["end_time"])
        if time < 480
    }
    peak = max(
        (
            sum(
                assignment["start_time"] <= time < assignment["end_time"]
                for assignment in result.assignments
            )
            for time in event_points
        ),
        default=0,
    )
    assert peak <= 2


def test_capacity_block_linking_not_in_phase1_ops():
    from app.engine.phases.phase1_knitting import PHASE1_OPS

    assert "capacity_block_linking" not in PHASE1_OPS


def test_linking_left_shift_preserves_workforce_block():
    machine_ids = [f"LM{i}" for i in range(4)]
    resources = [
        {"id": machine_id, "operation": "linking", "type": "serial", "capacity": 1}
        for machine_id in machine_ids
    ]
    linking_tasks = [_linking_task(f"L{i}", machine_ids) for i in range(3)]
    block = {
        "task_id": "BLOCK",
        "operation": "capacity_block_linking",
        "duration": 480,
        "demand": 2,
        "is_pinned": True,
        "pinned_start_time": 0,
        "pinned_end_time": 480,
    }
    assignments = [
        {
            "task_id": f"L{i}",
            "machine_id": machine_ids[i],
            "start_time": 500,
            "end_time": 600,
            "status": "ON_TIME",
        }
        for i in range(3)
    ]

    left_shift_cold_linking(
        assignments, [*linking_tasks, block], resources, {}
    )

    event_points = {
        time
        for assignment in assignments
        for time in (assignment["start_time"], assignment["end_time"])
        if time < 480
    }
    peak = max(
        sum(
            assignment["start_time"] <= time < assignment["end_time"]
            for assignment in assignments
        )
        for time in event_points
    )
    assert peak <= 2
