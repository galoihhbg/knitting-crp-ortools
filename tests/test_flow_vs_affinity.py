
import pytest
from ortools.sat.python import cp_model
from app.engine.phases.phase1_knitting import solve_knitting

BASE_CONFIG = {
    "horizon_minutes": 2000,
    "max_factory_machines": 10,
    "random_seed": 42,
    "num_search_workers": 1,
}

def _make_resource(r_id: str, color: str = "") -> dict:
    return {
        "id": r_id,
        "type": "serial",
        "capacity": 1,
        "unavailability": [],
        "available_at_min": 0,
        "color_config": color,
    }

def _make_knitting_task(task_id: str, group_id: str, duration: int, machines: list[str], color: str = "") -> dict:
    return {
        "task_id": task_id,
        "group_id": group_id,
        "original_order_id": task_id,
        "operation": "knitting",
        "qty": 100,
        "priority": 3,
        "duration": duration,
        "color_config": color,
        "compatible_resource_ids": machines,
    }

def test_flow_overcomes_affinity():
    """
    Scenario:
    - 2 Tasks (K1, K2) for the same order (ORD-001).
    - 2 Machines (M1, M2).
    - M1 is set up for 10 yarns (Y1..Y10).
    - K1 and K2 both need those 10 yarns.
    - M2 is set up for different yarns (Cold Start + 10 swaps).
    
    If flow objective is working:
    Solver should prefer parallelizing (K1 on M1, K2 on M2) to finish at 300,
    even if it costs 10 roll swaps (1000 penalty) on M2.
    
    If flow objective is weak:
    Solver will run them sequentially on M1 to save 10 swaps.
    """
    # 10 yarns string format: "Y1:1|Y2:1|..."
    yarns = "|".join([f"Y{i}:1" for i in range(1, 11)])
    
    resources = [
        _make_resource("M1", color=yarns),
        _make_resource("M2", color="OTHER:1"), # Mismatch
    ]
    
    tasks = [
        _make_knitting_task("K1", "ORD-001", 300, ["M1", "M2"], color=yarns),
        _make_knitting_task("K2", "ORD-001", 300, ["M1", "M2"], color=yarns),
    ]

    # Use solve_knitting directly to test Phase 1 logic
    result = solve_knitting(tasks, resources, BASE_CONFIG, horizon=1000)
    
    a_k1 = next(a for a in result.assignments if a["task_id"] == "K1")
    a_k2 = next(a for a in result.assignments if a["task_id"] == "K2")
    
    print(f"\nK1: machine={a_k1['machine_id']}, start={a_k1['start_time']}")
    print(f"K2: machine={a_k2['machine_id']}, start={a_k2['start_time']}")
    
    # Assert parallelism
    assert a_k1["machine_id"] != a_k2["machine_id"], (
        f"Tasks are serialized on {a_k1['machine_id']}! "
        "Flow objective failed to overcome affinity penalty."
    )
    assert a_k1["start_time"] == 0 and a_k2["start_time"] == 0
