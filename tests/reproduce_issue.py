
import pytest
from ortools.sat.python import cp_model
from app.engine.builder import TaskModelBuilder

BASE_CONFIG = {
    "horizon_minutes": 2000,
    "max_factory_machines": 10,
    "random_seed": 42,
    "num_search_workers": 1,
}

def _make_resource(r_id: str, design: str = "", color: str = "") -> dict:
    return {
        "id": r_id,
        "type": "serial",
        "capacity": 1,
        "unavailability": [],
        "available_at_min": 0,
        "design_item_id": design,
        "color_config": color,
    }

def _make_knitting_task(task_id: str, po_id: str, group_id: str, duration: int, machines: list[str], design: str = "", color: str = "") -> dict:
    return {
        "task_id": task_id,
        "original_order_id": po_id,
        "group_id": group_id,
        "operation": "knitting",
        "qty": 100,
        "priority": 3,
        "duration": duration,
        "design_item_id": design,
        "color_config": color,
        "compatible_resource_ids": machines,
    }

def _make_linking_task(task_id: str, group_id: str, duration: int, depends_on: list[str], machines: list[str]) -> dict:
    return {
        "task_id": task_id,
        "original_order_id": group_id,
        "group_id": group_id,
        "operation": "linking",
        "qty": 100,
        "priority": 3,
        "duration": duration,
        "final_depends_on": depends_on,
        "compatible_resource_ids": machines,
        "design_item_id": "",
        "color_config": "",
    }

def test_reproduce_sequential_knitting_due_to_affinity():
    """
    Scenario:
    2 Knitting tasks (PO1, PO2) for the same order (ORD-001).
    2 Machines available (M1, M2).
    M1 is already set up for PO1's design/color.
    M2 is NOT set up (Cold Start).
    
    Current behavior (expected to fail if fixed correctly):
    Solver chooses M1 for BOTH PO1 and PO2 sequentially to save setup on M2.
    
    Desired behavior:
    Solver splits them (PO1 on M1, PO2 on M2) to start Linking earlier,
    even if it costs one setup on M2.
    """
    resources = [
        _make_resource("M1", design="D1", color="C1"),
        _make_resource("M2", design="D_X", color="C_X"), # Different -> needs setup
        _make_resource("L1"),
    ]
    
    tasks = [
        _make_knitting_task("K1", "PO1", "ORD-001", 300, ["M1", "M2"], design="D1", color="C1"),
        _make_knitting_task("K2", "PO2", "ORD-001", 300, ["M1", "M2"], design="D1", color="C1"),
        _make_linking_task("L_ORD001", "ORD-001", 100, ["K1", "K2"], ["L1"]),
    ]

    builder = (
        TaskModelBuilder(BASE_CONFIG, resources, tasks, {})
        .build_time_variables()
        .build_resource_allocations()
        .apply_routing_constraints()
        .apply_dependency_constraints()
        .define_objective()
    )
    solver = cp_model.CpSolver()
    status = solver.Solve(builder.model)
    result = builder.extract_results(solver, status)

    a_k1 = next(a for a in result["assignments"] if a["task_id"] == "K1")
    a_k2 = next(a for a in result["assignments"] if a["task_id"] == "K2")
    
    print(f"\nK1: machine={a_k1['machine_id']}, start={a_k1['start_time']}, end={a_k1['end_time']}")
    print(f"K2: machine={a_k2['machine_id']}, start={a_k2['start_time']}, end={a_k2['end_time']}")
    
    # Check if they are parallel (on different machines)
    # If they are on the same machine, they must be sequential.
    assert a_k1["machine_id"] != a_k2["machine_id"], "REPRODUCTION SUCCESS: Tasks are sequential on the same machine!"
