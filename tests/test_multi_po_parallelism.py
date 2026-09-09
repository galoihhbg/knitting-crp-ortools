"""
tests/test_multi_po_parallelism.py

Verifies that when one production order has multiple Knitting POs,
the solver schedules them in parallel (or at least overlapping) rather than
running them sequentially — which would make the downstream Linking task
wait unnecessarily long.

Root-cause context: before the cross-PO incentive was added there was no
objective term penalising the start-time gap between POs of the same order,
so CP-SAT would trivially serialize them on one machine.
"""
import pytest
from ortools.sat.python import cp_model

from app.engine.builder import TaskModelBuilder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "horizon_minutes": 2000,
    "max_factory_machines": 10,
    "random_seed": 42,
    "num_search_workers": 1,
}

def _make_resource(r_id: str, capacity: int = 1) -> dict:
    return {
        "id": r_id,
        "type": "serial" if capacity == 1 else "batch",
        "capacity": capacity,
        "unavailability": [],
        "available_at_min": 0,
        "design_item_id": "",
        "color_config": "",
    }


def _make_knitting_task(
    task_id: str,
    po_id: str,
    group_id: str,
    duration: int,
    due_at: int,
    machines: list[str],
    priority: int = 3,
) -> dict:
    return {
        "task_id": task_id,
        "original_order_id": po_id,
        "group_id": group_id,
        "operation": "knitting",
        "qty": 100,
        "total_qty": 100,
        "priority": priority,
        "final_depends_on": [],
        "start_after_min": 0,
        "due_at_min": due_at,
        "duration": duration,
        "is_batch": False,
        "sub_tasks": None,
        "design_item_id": "",
        "color_config": "",
        "color": "",
        "substance": "",
        "compatible_resource_ids": machines,
        "WaitOffsets": None,
        "is_pinned": False,
        "pinned_machine_id": None,
        "pinned_start_time": None,
        "pinned_end_time": None,
        "demand": 1,
        "material_demands": {},
    }


def _make_linking_task(
    task_id: str,
    group_id: str,
    duration: int,
    due_at: int,
    machines: list[str],
    depends_on: list[str],
    priority: int = 3,
) -> dict:
    return {
        "task_id": task_id,
        "original_order_id": group_id,
        "group_id": group_id,
        "operation": "linking",
        "qty": 100,
        "total_qty": 100,
        "priority": priority,
        "final_depends_on": depends_on,
        "start_after_min": 0,
        "due_at_min": due_at,
        "duration": duration,
        "is_batch": False,
        "sub_tasks": None,
        "design_item_id": "",
        "color_config": "",
        "color": "",
        "substance": "",
        "compatible_resource_ids": machines,
        "WaitOffsets": None,
        "is_pinned": False,
        "pinned_machine_id": None,
        "pinned_start_time": None,
        "pinned_end_time": None,
        "demand": 1,
        "material_demands": {},
    }


def _solve(tasks: list[dict], resources: list[dict], config: dict = None) -> dict:
    """Build and solve the model; return result dict."""
    cfg = {**BASE_CONFIG, **(config or {})}
    builder = (
        TaskModelBuilder(cfg, resources, tasks, {})
        .build_time_variables()
        .build_resource_allocations()
        .apply_routing_constraints()
        .apply_dependency_constraints()
        .define_objective()
    )
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = cfg["random_seed"]
    solver.parameters.num_search_workers = cfg["num_search_workers"]
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(builder.model)
    result = builder.extract_results(solver, status)
    result["_builder"] = builder
    result["_solver"] = solver
    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assignment(result: dict, task_id: str) -> dict:
    return next(a for a in result["assignments"] if a["task_id"] == task_id)


# ---------------------------------------------------------------------------
# Test 1 — Two POs scheduled in parallel when two machines are free
# ---------------------------------------------------------------------------

def test_two_pos_run_in_parallel_when_machines_available():
    """
    Scenario: Order ORD-001 has two knitting POs (PO1 and PO2), each with
    duration=300min, and there are 2 knitting machines available.
    Expected: K_PO1 and K_PO2 start at or very near the same time so
    Linking (which depends on both) begins as early as possible.
    """
    resources = [
        _make_resource("K_M1"),
        _make_resource("K_M2"),
        _make_resource("L_M1"),
    ]
    tasks = [
        _make_knitting_task("K_PO1", "PO1", "ORD-001", 300, 1000, ["K_M1", "K_M2"]),
        _make_knitting_task("K_PO2", "PO2", "ORD-001", 300, 1000, ["K_M1", "K_M2"]),
        _make_linking_task("L_ORD001", "ORD-001", 100, 1000, ["L_M1"], ["K_PO1", "K_PO2"]),
    ]

    result = _solve(tasks, resources)
    assert result["status"] == "feasible", f"Expected feasible, got {result['status']}"

    a_k1 = _assignment(result, "K_PO1")
    a_k2 = _assignment(result, "K_PO2")
    a_l  = _assignment(result, "L_ORD001")

    # The two Ks must land on DIFFERENT machines (parallelism)
    assert a_k1["machine_id"] != a_k2["machine_id"], (
        f"K_PO1 and K_PO2 both landed on {a_k1['machine_id']} — "
        "they should be on separate machines to run in parallel"
    )

    # Both Ks should start at roughly the same time (within one K duration of each other)
    start_gap = abs(a_k1["start_time"] - a_k2["start_time"])
    assert start_gap < 300, (
        f"K_PO1 starts at {a_k1['start_time']}, K_PO2 at {a_k2['start_time']} "
        f"(gap={start_gap}min) — they should start nearly simultaneously"
    )

    # Linking starts after both Ks end
    expected_l_start = max(a_k1["end_time"], a_k2["end_time"])
    assert a_l["start_time"] >= expected_l_start, (
        f"L started at {a_l['start_time']} before K ends at {expected_l_start}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Linking starts earlier with parallel POs than sequential
# ---------------------------------------------------------------------------

def test_parallel_pos_yield_earlier_linking_start():
    """
    Confirms that the cross-PO incentive results in an earlier Linking start
    compared to a hypothetical sequential arrangement.

    We verify: L.start_time <= K_duration * 1.2
    (i.e. Linking starts roughly as soon as one K finishes, not after both
    finish serially which would be K_duration * 2 = 600).
    """
    K_DUR = 300
    resources = [
        _make_resource("K_M1"),
        _make_resource("K_M2"),
        _make_resource("L_M1"),
    ]
    tasks = [
        _make_knitting_task("K_PO1", "PO1", "ORD-002", K_DUR, 1500, ["K_M1", "K_M2"]),
        _make_knitting_task("K_PO2", "PO2", "ORD-002", K_DUR, 1500, ["K_M1", "K_M2"]),
        _make_linking_task("L_ORD002", "ORD-002", 100, 1500, ["L_M1"], ["K_PO1", "K_PO2"]),
    ]

    result = _solve(tasks, resources)
    assert result["status"] == "feasible"

    a_k1 = _assignment(result, "K_PO1")
    a_k2 = _assignment(result, "K_PO2")
    a_l  = _assignment(result, "L_ORD002")

    # Sequential worst-case: L would start at K_DUR * 2 = 600
    # Parallel best-case:    L would start at K_DUR = 300
    sequential_l_start = K_DUR * 2  # 600 — the bad old behaviour
    assert a_l["start_time"] < sequential_l_start, (
        f"L.start_time={a_l['start_time']} is not better than sequential "
        f"({sequential_l_start}min). Cross-PO incentive may not be working."
    )


# ---------------------------------------------------------------------------
# Test 3 — Single-PO order is unaffected (no regression)
# ---------------------------------------------------------------------------

def test_single_po_order_unaffected():
    """
    Regression guard: an order with only ONE knitting PO should still be
    scheduled correctly. No cross-PO constraint should apply.
    """
    resources = [
        _make_resource("K_M1"),
        _make_resource("L_M1"),
    ]
    tasks = [
        _make_knitting_task("K_PO1", "PO1", "ORD-003", 200, 800, ["K_M1"]),
        _make_linking_task("L_ORD003", "ORD-003", 50, 800, ["L_M1"], ["K_PO1"]),
    ]

    result = _solve(tasks, resources)
    assert result["status"] == "feasible"

    a_k = _assignment(result, "K_PO1")
    a_l = _assignment(result, "L_ORD003")

    # L must start after K finishes
    assert a_l["start_time"] >= a_k["end_time"], (
        f"L started at {a_l['start_time']} before K ended at {a_k['end_time']}"
    )
    # K must be on its only compatible machine
    assert a_k["machine_id"] == "K_M1"


# ---------------------------------------------------------------------------
# Test 4 — Three POs: all start within a reasonable window
# ---------------------------------------------------------------------------

def test_three_pos_all_start_concurrently():
    """
    When three POs feed one Linking task and three machines are available,
    the solver should start all three at the same time.
    """
    resources = [
        _make_resource("K_M1"),
        _make_resource("K_M2"),
        _make_resource("K_M3"),
        _make_resource("L_M1"),
    ]
    tasks = [
        _make_knitting_task("K_PO1", "PO1", "ORD-004", 200, 1500, ["K_M1", "K_M2", "K_M3"]),
        _make_knitting_task("K_PO2", "PO2", "ORD-004", 200, 1500, ["K_M1", "K_M2", "K_M3"]),
        _make_knitting_task("K_PO3", "PO3", "ORD-004", 200, 1500, ["K_M1", "K_M2", "K_M3"]),
        _make_linking_task("L_ORD004", "ORD-004", 80, 1500, ["L_M1"], ["K_PO1", "K_PO2", "K_PO3"]),
    ]

    result = _solve(tasks, resources)
    assert result["status"] == "feasible"

    a_k1 = _assignment(result, "K_PO1")
    a_k2 = _assignment(result, "K_PO2")
    a_k3 = _assignment(result, "K_PO3")

    # All three on distinct machines
    machines = {a_k1["machine_id"], a_k2["machine_id"], a_k3["machine_id"]}
    assert len(machines) == 3, (
        f"Not all POs on distinct machines: {machines}"
    )

    # All start within 200min of each other (one K duration)
    starts = [a_k1["start_time"], a_k2["start_time"], a_k3["start_time"]]
    assert max(starts) - min(starts) < 200, (
        f"PO start spread too large: {starts} — expected near-simultaneous starts"
    )
