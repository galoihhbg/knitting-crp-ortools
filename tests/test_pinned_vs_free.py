"""
Diagnostic test: pinned tasks must NOT suppress free tasks.

Scenario: Mix of pinned (dummy/shift-block) tasks and free tasks.
Free tasks (is_pinned=False) MUST appear in the result assignments.
"""
from typing import Any, Dict, List

from app.engine.model import Engine

HORIZON = 2880


def _cfg(**overrides: Any) -> Dict[str, Any]:
    base = {
        "horizon_minutes": HORIZON,
        "max_search_time": 15,
        "setup_time_minutes": 0,
        "max_factory_machines": 40,
        "random_seed": 42,
        "num_search_workers": 1,
    }
    base.update(overrides)
    return base


def _resource(r_id: str, operation: str = "knitting") -> Dict[str, Any]:
    return {
        "id": r_id,
        "type": "serial",
        "capacity": 1,
        "operation": operation,
        "unavailability": [],
        "design_item_id": "",
        "color_config": "",
        "available_at_min": 0,
    }


def _machine(m_id: str) -> Dict[str, Any]:
    return {
        "id": m_id,
        "type": "serial",
        "capacity": 1,
        "worker_req": 1,
        "routing": [{"operation": "knitting", "design_item_id": "", "duration": 0.0, "setup_time": 0.0}],
        "design_item_id": "",
        "color_config": "",
    }


def _free_task(
    task_id: str,
    compatible_ids: List[str],
    duration: int = 200,
    operation: str = "knitting",
    due: int = HORIZON,
    priority: int = 3,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": task_id,
        "group_id": "G1",
        "operation": operation,
        "qty": 1.0,
        "total_qty": 1.0,
        "priority": priority,
        "original_depends_on": [],
        "final_depends_on": [],
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
        "compatible_resource_ids": compatible_ids,
        "sub_task_completion_offsets": None,
        "WaitOffsets": None,
        "is_pinned": False,
        "pinned_machine_id": None,
        "pinned_start_time": None,
        "pinned_end_time": None,
        "demand": 1,
        "material_demands": {},
    }


def _pinned_task(
    task_id: str,
    machine_id: str,
    start: int,
    end: int,
    operation: str = "knitting",
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": task_id,
        "group_id": "PINNED",
        "operation": operation,
        "qty": 0.0,
        "total_qty": 0.0,
        "priority": 5,
        "original_depends_on": [],
        "final_depends_on": [],
        "start_after_min": 0,
        "due_at_min": end,
        "duration": end - start,
        "is_slice": False,
        "parent_task_id": "",
        "internal_dep": "",
        "slice_index": 0,
        "is_batch": False,
        "sub_tasks": None,
        "design_item_id": "",
        "color_config": "",
        "compatible_resource_ids": [machine_id],
        "sub_task_completion_offsets": None,
        "WaitOffsets": None,
        "is_pinned": True,
        "pinned_machine_id": machine_id,
        "pinned_start_time": start,
        "pinned_end_time": end,
        "demand": 0,
        "material_demands": {},
    }


def _solve(resources, tasks, machines=None, **cfg_overrides):
    payload = {
        "job_id": "test_pinned_vs_free",
        "config": _cfg(**cfg_overrides),
        "machines": machines or [_machine(r["id"]) for r in resources],
        "resources": resources,
        "tasks": tasks,
        "material_capacities": {},
    }
    return Engine(payload).solve()


def _get_assignments_by_id(result):
    return {a["task_id"]: a for a in result.get("assignments", [])}


# ── Test 1: Free tasks alongside pinned tasks on same machine ──────────────

def test_free_task_scheduled_alongside_pinned():
    """
    2 machines (KM_00, KM_01).
    Pinned task on KM_00 at [0, 500).
    Free task (duration=200) compatible with BOTH machines.
    Expected: free task is scheduled (on KM_01 or after 500 on KM_00).
    """
    resources = [_resource("KM_00"), _resource("KM_01")]
    tasks = [
        _pinned_task("PINNED_A", "KM_00", 0, 500),
        _free_task("FREE_B", ["KM_00", "KM_01"], duration=200),
    ]
    result = _solve(resources, tasks)
    assert result["status"] == "feasible", f"Expected feasible, got {result['status']}"
    assigns = _get_assignments_by_id(result)
    assert "FREE_B" in assigns, (
        f"FREE_B (pinned=false) NOT in assignments! "
        f"Only got: {list(assigns.keys())}"
    )


# ── Test 2: Multiple free tasks + multiple pinned across machines ──────────

def test_multiple_free_with_multiple_pinned():
    """
    5 machines (KM_00..KM_04).
    Pinned tasks block KM_00 and KM_01 at [0, 1000).
    3 free tasks (duration=200 each) compatible with all 5 machines.
    Expected: all 3 free tasks scheduled.
    """
    machine_ids = [f"KM_{i:02d}" for i in range(5)]
    resources = [_resource(m) for m in machine_ids]
    tasks = [
        _pinned_task("PINNED_0", "KM_00", 0, 1000),
        _pinned_task("PINNED_1", "KM_01", 0, 1000),
        _free_task("FREE_A", machine_ids, duration=200),
        _free_task("FREE_B", machine_ids, duration=200),
        _free_task("FREE_C", machine_ids, duration=200),
    ]
    result = _solve(resources, tasks)
    assert result["status"] == "feasible"
    assigns = _get_assignments_by_id(result)
    for t_id in ["FREE_A", "FREE_B", "FREE_C"]:
        assert t_id in assigns, (
            f"{t_id} (pinned=false) NOT in assignments! "
            f"Only got: {list(assigns.keys())}"
        )


# ── Test 3: All machines pinned — free task pushed to later slot ───────────

def test_free_task_after_all_pinned_end():
    """
    2 machines. Both pinned at [0, 500).
    Free task (duration=100) compatible with both.
    Expected: free task starts at >= 500.
    """
    resources = [_resource("KM_00"), _resource("KM_01")]
    tasks = [
        _pinned_task("PINNED_A", "KM_00", 0, 500),
        _pinned_task("PINNED_B", "KM_01", 0, 500),
        _free_task("FREE_X", ["KM_00", "KM_01"], duration=100),
    ]
    result = _solve(resources, tasks)
    assert result["status"] == "feasible"
    assigns = _get_assignments_by_id(result)
    assert "FREE_X" in assigns, f"FREE_X not scheduled! Got: {list(assigns.keys())}"
    assert assigns["FREE_X"]["start_time"] >= 500, (
        f"FREE_X started at {assigns['FREE_X']['start_time']}, expected >= 500"
    )


# ── Test 4: Pinned with different ratio of machines ────────────────────────

def test_scaling_workers_5_to_10():
    """
    Reproduces the user's exact scenario:
    10 machines, 5 pinned at [0, 1440), 5 free tasks.
    With 5 workers it worked, with 10 it broke.
    """
    machine_ids = [f"KM_{i:02d}" for i in range(10)]
    resources = [_resource(m) for m in machine_ids]
    # Pin first 5 machines with shift blocks
    pinned = [
        _pinned_task(f"SHIFT_{i:02d}", f"KM_{i:02d}", 0, 1440)
        for i in range(5)
    ]
    # 5 free tasks compatible with ALL 10 machines (can use KM_05..KM_09)
    free = [
        _free_task(f"JOB_{i}", machine_ids, duration=200)
        for i in range(5)
    ]
    result = _solve(resources, pinned + free)
    assert result["status"] == "feasible"
    assigns = _get_assignments_by_id(result)
    for i in range(5):
        t_id = f"JOB_{i}"
        assert t_id in assigns, (
            f"{t_id} NOT scheduled! Only got: {list(assigns.keys())}"
        )



# ── Test 5: Diagnostic — print all internal state ─────────────────────────

def test_diagnostic_print_builder_state():
    """
    Prints detailed builder internals to help diagnose why free tasks
    might be missing from the output. This test always passes but prints
    diagnostic info to stdout.
    """
    from app.engine.builder import TaskModelBuilder
    from ortools.sat.python import cp_model

    resources = [_resource("KM_00"), _resource("KM_01")]
    machines = [_machine(r["id"]) for r in resources]
    tasks = [
        _pinned_task("PINNED_A", "KM_00", 0, 500),
        _free_task("FREE_B", ["KM_00", "KM_01"], duration=200),
    ]

    config = _cfg()
    machine_states = {m["id"]: {"current_design": "", "current_color": ""} for m in machines}

    builder = TaskModelBuilder(config, resources, tasks, machine_states)
    builder.build_time_variables()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: After build_time_variables")
    print("=" * 60)
    for t_id, tv in builder.task_vars.items():
        print(f"  {t_id}: r_ids={tv['r_ids']}, literals={tv['literals']}")
    print(f"  no_resource_tasks: {[t['task_id'] for t in builder.no_resource_tasks]}")

    builder.build_resource_allocations()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: After build_resource_allocations")
    print("=" * 60)
    for t_id, tv in builder.task_vars.items():
        lit_names = [l.Name() for l in tv["literals"]]
        print(f"  {t_id}: r_ids={tv['r_ids']}, literals={lit_names}")
    print(f"  no_resource_tasks: {[t['task_id'] for t in builder.no_resource_tasks]}")

    # Check the critical r_ids ↔ literals index mapping
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: r_ids ↔ literals mapping check")
    print("=" * 60)
    for t_id, tv in builder.task_vars.items():
        r_ids = tv["r_ids"]
        lits = tv["literals"]
        if len(r_ids) != len(lits):
            print(f"  ⚠️ {t_id}: MISMATCH len(r_ids)={len(r_ids)} != len(literals)={len(lits)}")
            print(f"     r_ids = {r_ids}")
            print(f"     lits  = {[l.Name() for l in lits]}")
        else:
            print(f"  ✅ {t_id}: OK len={len(r_ids)}")

    builder.apply_routing_constraints()
    builder.apply_dependency_constraints()
    builder.apply_batch_offset_constraints()
    builder.define_objective()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    status = solver.Solve(builder.model)

    print("\n" + "=" * 60)
    print(f"DIAGNOSTIC: Solve status = {solver.StatusName(status)}")
    print("=" * 60)

    result = builder.extract_results(solver, status)
    print(f"  Result status: {result['status']}")
    print(f"  Assignments: {len(result['assignments'])}")
    for a in result["assignments"]:
        print(f"    {a['task_id']}: machine={a['machine_id']}, "
              f"start={a['start_time']}, end={a['end_time']}")

    assert result["status"] == "feasible"
    assigns = {a["task_id"]: a for a in result["assignments"]}
    assert "FREE_B" in assigns, f"FREE_B not scheduled!"


# ── Test 6: Two pinned knitting tasks with mismatched "duration" field ─────

def test_two_pinned_knitting_tasks_duration_mismatch():
    """
    Regression for "2-order infeasible" bug.

    When Go sends pinned knitting task with:
      pinned_start_time = 0, pinned_end_time = 480
      but "duration" field = 0 (or any value != 480)

    The old code called:
      NewIntervalVar(NewConstant(0), 0, NewConstant(480))
    which forces 480 == 0 + 0 → HARD CONTRADICTION → INFEASIBLE.

    With 1 task the constraint might accidentally be satisfied (if duration
    matches), but with 2 tasks even one mismatch makes the whole model fail.
    """
    resources = [_resource("KM_00"), _resource("KM_01")]

    # Pinned start=0, end=480, but "duration" field = 0 (Go sends wrong value)
    pinned_wrong_duration = {
        **_pinned_task("PINNED_KM00", "KM_00", 0, 480),
        "duration": 0,   # ← This is what the old code used for NewIntervalVar size
    }
    pinned_correct = {
        **_pinned_task("PINNED_KM01", "KM_01", 0, 480),
        "duration": 0,   # Same issue on second machine
    }

    tasks = [
        pinned_wrong_duration,
        pinned_correct,
        _free_task("FREE_A", ["KM_00", "KM_01"], duration=200),
    ]

    result = _solve(resources, tasks)
    # Without fix: infeasible (NewIntervalVar forces 480 == 0 + 0)
    # With fix: feasible (uses pinned_end - pinned_start = 480)
    assert result["status"] == "feasible", (
        f"Expected feasible but got {result['status']}. "
        "This is the 2-order duration mismatch regression."
    )
    assigns = _get_assignments_by_id(result)
    assert "FREE_A" in assigns, f"FREE_A not scheduled!"

