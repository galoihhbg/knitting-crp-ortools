"""
AddCircuit no-gap tests — zero-gap contiguous scheduling per serial machine.

Each test constructs a minimal CP-SAT model via build_resource_model() and verifies
that tasks assigned to the same serial machine are scheduled back-to-back with no
idle gap between consecutive tasks.

Fixture design rules:
  - basic_no_gap:          3 free tasks, 1 machine → all contiguous
  - pinned_free_no_gap:    1 pinned + 2 free → free tasks pack tight against pinned
  - pinned_pinned_gap:     2 pinned tasks with inevitable gap → circuit still feasible
  - multi_machine_circuit: 2 machines, 4 tasks → each machine's tasks are contiguous
  - circuit_disabled:      config flag → gaps allowed (no circuit)
  - circuit_with_start_lb: task with start_lb → gap allowed when dep forces it
  - circuit_cap_exceeded:  tasks > cap → circuit skipped, falls back to NoOverlap only
"""
from typing import Any, Dict, List, Optional

import pytest
from ortools.sat.python import cp_model

from app.engine.shared import build_resource_model, make_solver


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_resource(r_id: str, capacity: int = 1) -> Dict[str, Any]:
    return {
        "id": r_id,
        "type": "serial",
        "capacity": capacity,
        "operation": "knitting",
        "unavailability": [],
        "design_item_id": "",
        "color_config": "",
        "available_at_min": 0,
    }


def _make_task(
    task_id: str,
    duration: int,
    compatible_ids: List[str],
    due: int = 5000,
    is_pinned: bool = False,
    pinned_machine_id: Optional[str] = None,
    pinned_start: Optional[int] = None,
    pinned_end: Optional[int] = None,
    start_after: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "task_id": task_id,
        "original_order_id": f"ORDER_{task_id}",
        "group_id": f"ORDER_{task_id}",
        "operation": "knitting",
        "qty": 10,
        "total_qty": 100,
        "priority": 3,
        "original_depends_on": [],
        "final_depends_on": [],
        "start_after_min": start_after,
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
        "is_pinned": is_pinned,
        "pinned_machine_id": pinned_machine_id,
        "pinned_start_time": pinned_start,
        "pinned_end_time": pinned_end,
        "demand": 1,
        "material_demands": {},
    }
    base.update(kwargs)
    return base


def _config(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "horizon_minutes": 5000,
        "max_search_time": 10,
        "max_factory_machines": 10,
        "random_seed": 42,
        "num_search_workers": 1,
        "enable_circuit_no_gap": True,
    }
    cfg.update(overrides)
    return cfg


def _solve_model(
    tasks: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    config: Dict[str, Any],
    start_lb: Optional[Dict[str, int]] = None,
    horizon: int = 5000,
):
    """Build and solve a minimal CP-SAT model using build_resource_model."""
    model = cp_model.CpModel()
    resource_map = {r["id"]: r for r in resources}

    task_vars, obj_terms, no_resource = build_resource_model(
        model, tasks, resource_map, horizon,
        start_lb=start_lb, use_affinity=False,
        config=config,
    )

    # Simple objective: minimize sum of start times (prefer early starts)
    if task_vars:
        model.Minimize(sum(tv["start"] for tv in task_vars.values()))

    solver = make_solver(config)
    status = solver.Solve(model)

    results = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for t_id, tv in task_vars.items():
            s = solver.Value(tv["start"])
            e = solver.Value(tv["end"])
            # Find assigned machine
            machine = None
            for i, lit in enumerate(tv.get("literals", [])):
                if solver.Value(lit) == 1:
                    machine = tv["r_ids"][i]
                    break
            if machine is None and tv.get("is_pinned"):
                machine = tv["r_ids"][0] if tv.get("r_ids") else None
            results[t_id] = {"start": s, "end": e, "machine": machine}

    return status, results, solver


# ── Tests ─────────────────────────────────────────────────────────────────


def test_basic_no_gap():
    """3 free tasks on 1 machine → all back-to-back, zero gap."""
    resources = [_make_resource("M1")]
    tasks = [
        _make_task("T1", 100, ["M1"]),
        _make_task("T2", 150, ["M1"]),
        _make_task("T3", 200, ["M1"]),
    ]
    cfg = _config()
    status, results, _ = _solve_model(tasks, resources, cfg)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert len(results) == 3

    # All tasks should be on M1
    for t_id in ["T1", "T2", "T3"]:
        assert results[t_id]["machine"] == "M1"

    # Sort by start time
    ordered = sorted(results.values(), key=lambda r: r["start"])

    # Check zero gap between consecutive tasks
    for i in range(len(ordered) - 1):
        gap = ordered[i + 1]["start"] - ordered[i]["end"]
        assert gap == 0, (
            f"Gap of {gap} min between task ending at {ordered[i]['end']} "
            f"and task starting at {ordered[i + 1]['start']} — expected zero gap"
        )


def test_pinned_free_no_gap():
    """1 pinned task at [0, 100) + 2 free tasks → free tasks pack tight against pinned end."""
    resources = [_make_resource("M1")]
    tasks = [
        _make_task("T_pinned", 100, ["M1"],
                   is_pinned=True, pinned_machine_id="M1",
                   pinned_start=0, pinned_end=100),
        _make_task("T_free1", 80, ["M1"]),
        _make_task("T_free2", 120, ["M1"]),
    ]
    cfg = _config()
    status, results, _ = _solve_model(tasks, resources, cfg)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert len(results) == 3

    # Pinned task at [0, 100)
    assert results["T_pinned"]["start"] == 0
    assert results["T_pinned"]["end"] == 100

    # Free tasks must pack tight after pinned end (no gap)
    ordered = sorted(results.values(), key=lambda r: r["start"])
    for i in range(len(ordered) - 1):
        gap = ordered[i + 1]["start"] - ordered[i]["end"]
        assert gap == 0, (
            f"Gap of {gap} min between task ending at {ordered[i]['end']} "
            f"and task starting at {ordered[i + 1]['start']} — expected zero gap"
        )


def test_pinned_pinned_gap_allowed():
    """2 pinned tasks with a gap between them → circuit still feasible (gap preserved)."""
    resources = [_make_resource("M1")]
    tasks = [
        _make_task("T_pin1", 100, ["M1"],
                   is_pinned=True, pinned_machine_id="M1",
                   pinned_start=0, pinned_end=100),
        _make_task("T_pin2", 100, ["M1"],
                   is_pinned=True, pinned_machine_id="M1",
                   pinned_start=200, pinned_end=300),
    ]
    cfg = _config()
    status, results, _ = _solve_model(tasks, resources, cfg)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
        "Two pinned tasks with a gap should NOT make the model infeasible"
    )
    assert len(results) == 2
    assert results["T_pin1"]["start"] == 0
    assert results["T_pin1"]["end"] == 100
    assert results["T_pin2"]["start"] == 200
    assert results["T_pin2"]["end"] == 300


def test_multi_machine_circuit():
    """2 machines, 4 tasks (2 per machine) → each machine's tasks are contiguous."""
    resources = [_make_resource("M1"), _make_resource("M2")]
    tasks = [
        # These tasks can only go on M1
        _make_task("T1_M1", 100, ["M1"]),
        _make_task("T2_M1", 150, ["M1"]),
        # These tasks can only go on M2
        _make_task("T1_M2", 80, ["M2"]),
        _make_task("T2_M2", 120, ["M2"]),
    ]
    cfg = _config()
    status, results, _ = _solve_model(tasks, resources, cfg)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert len(results) == 4

    # Check M1 tasks are contiguous
    m1_tasks = sorted(
        [r for r in results.values() if r["machine"] == "M1"],
        key=lambda r: r["start"],
    )
    for i in range(len(m1_tasks) - 1):
        gap = m1_tasks[i + 1]["start"] - m1_tasks[i]["end"]
        assert gap == 0, f"Gap on M1: {gap} min"

    # Check M2 tasks are contiguous
    m2_tasks = sorted(
        [r for r in results.values() if r["machine"] == "M2"],
        key=lambda r: r["start"],
    )
    for i in range(len(m2_tasks) - 1):
        gap = m2_tasks[i + 1]["start"] - m2_tasks[i]["end"]
        assert gap == 0, f"Gap on M2: {gap} min"


def test_circuit_disabled_by_config():
    """enable_circuit_no_gap=false → gaps may exist (solver only has NoOverlap)."""
    resources = [_make_resource("M1")]
    tasks = [
        _make_task("T1", 100, ["M1"]),
        _make_task("T2", 150, ["M1"]),
    ]
    cfg = _config(enable_circuit_no_gap=False)
    status, results, _ = _solve_model(tasks, resources, cfg)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert len(results) == 2
    # We don't assert gap==0 here; the point is the model is feasible without circuit


def test_circuit_with_start_lb():
    """Task with start_lb (dependency release) → gap allowed when dep forces it.

    T1 on M1: duration 100, free to start at 0.
    T2 on M1: duration 100, start_lb = 300 (must wait for upstream phase).
    Circuit enforces ordering + no-gap, BUT the start_lb constraint
    (start >= 300) takes precedence, creating an inevitable gap [100, 300).
    The model must still be feasible.
    """
    resources = [_make_resource("M1")]
    tasks = [
        _make_task("T1", 100, ["M1"]),
        _make_task("T2", 100, ["M1"], start_after=300),
    ]
    cfg = _config()
    # start_lb passed as lower bound from a prior phase
    status, results, _ = _solve_model(tasks, resources, cfg, start_lb={"T2": 300})

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
        "Model should be feasible even when start_lb forces a gap"
    )
    assert len(results) == 2

    # T1 should start at 0 (or as early as possible)
    assert results["T1"]["end"] <= 300, "T1 should finish before T2's release"
    # T2 must start at or after 300
    assert results["T2"]["start"] >= 300, (
        f"T2 started at {results['T2']['start']}, expected >= 300 (start_lb)"
    )


def test_circuit_cap_exceeded():
    """More tasks than circuit_max_tasks_per_machine → circuit skipped, NoOverlap only."""
    resources = [_make_resource("M1")]
    # Create 5 tasks but set cap to 3
    tasks = [_make_task(f"T{i}", 50, ["M1"]) for i in range(5)]
    cfg = _config(circuit_max_tasks_per_machine=3)
    status, results, _ = _solve_model(tasks, resources, cfg)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert len(results) == 5
    # Verify no overlap (NoOverlap still holds), but gaps may exist
    ordered = sorted(results.values(), key=lambda r: r["start"])
    for i in range(len(ordered) - 1):
        # NoOverlap: end[i] <= start[i+1]
        assert ordered[i]["end"] <= ordered[i + 1]["start"], (
            f"Overlap: task ending at {ordered[i]['end']} "
            f"vs task starting at {ordered[i + 1]['start']}"
        )
