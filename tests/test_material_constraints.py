"""
Material constraint tests — build_material_constraints() / AddCumulative.

Each test constructs a minimal payload that exercises a specific aspect of the
per-material creel capacity constraint.  The solver is run with num_search_workers=1
and a fixed random_seed for determinism.

Fixture design rules:
  - serial_material : capacity=2, two tasks each demand=2 → must serialise.
  - parallel_material: capacity=2, two tasks each demand=1 → may run concurrently.
  - no_material_cap  : no material_capacities key → feature is a no-op; solve works.
  - multi_material   : two materials, combined demand exceeds both caps → must serialise.
"""
from typing import Any, Dict

import pytest

from app.engine.model import Engine

# Phase-1 knitting material/creel enforcement is DELIBERATELY DISABLED.
# `_apply_material_constraints` is commented out in phase1_knitting.py
# ("Temporarily disabled per user request", commit 0b9019c, debug-log 2026-05-08)
# so that panels sharing scarce yarn can knit IN PARALLEL — material serialisation
# was holding back the parallel-knitting / earlier-panel-completion goal, and a
# real material with capacity 0 (vs nonzero demand) makes the cumulative INFEASIBLE.
# The tests below assert ENFORCEMENT, which contradicts that decision, so they are
# skipped (not deleted) — the infrastructure remains and they document the intended
# behaviour if creel limits are ever re-enabled (would also need a capacity<=0 guard).
_MATERIAL_P1_DISABLED = pytest.mark.skip(
    reason="Phase-1 knitting material/creel enforcement deliberately disabled "
    "(parallel-knitting); see phase1_knitting.py:_apply_material_constraints."
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_machine(m_id: str) -> Dict[str, Any]:
    return {
        "id": m_id,
        "type": "serial",
        "capacity": 1,
        "worker_req": 1,
        "routing": [
            {"operation": "knitting", "design_item_id": "", "duration": 0.0, "setup_time": 0.0}
        ],
        "design_item_id": "",
        "color_config": "",
    }


def _make_resource(r_id: str) -> Dict[str, Any]:
    return {
        "id": r_id,
        "type": "serial",
        "capacity": 1,
        "operation": "knitting",
        "unavailability": [],
        "design_item_id": "",
        "color_config": "",
        "available_at_min": 0,
    }


def _make_task(
    task_id: str,
    duration: int,
    due: int,
    compatible_ids: list,
    material_demands: Dict[str, int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "task_id": task_id,
        "original_order_id": "ORDER_X",
        "group_id": "ORDER_X",
        "operation": "knitting",
        "qty": 10.0,
        "total_qty": 100.0,
        "priority": 3,
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
        "material_demands": material_demands or {},
    }
    base.update(kwargs)
    return base


def _base_config(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "horizon_minutes": 2000,
        "max_search_time": 15,
        "setup_time_minutes": 0,
        "max_factory_machines": 10,
        "random_seed": 42,
        "num_search_workers": 1,
    }
    cfg.update(overrides)
    return cfg


def _overlaps(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True iff [a.start, a.end) and [b.start, b.end) overlap."""
    return a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]


# ── Tests ─────────────────────────────────────────────────────────────────


@_MATERIAL_P1_DISABLED
def test_material_constraint_serialises_tasks():
    """
    Two tasks each demand 2 rolls of Yarn_Red. Capacity = 2.
    Combined demand = 4 > 2 → AddCumulative must prevent concurrent execution.
    Two machines are available so the only constraint forcing serialisation
    is the material limit.
    Expected: the two task intervals do NOT overlap in the result.
    """
    payload = {
        "job_id": "test_mat_serial",
        "config": _base_config(),
        "material_capacities": {"Yarn_Red": 2},
        "machines": [_make_machine("KM_00"), _make_machine("KM_01")],
        "resources": [_make_resource("KM_00"), _make_resource("KM_01")],
        "tasks": [
            _make_task("T1", 300, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 2}),
            _make_task("T2", 400, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 2}),
        ],
    }

    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 2

    by_id = {a["task_id"]: a for a in result["assignments"]}
    assert not _overlaps(by_id["T1"], by_id["T2"]), (
        f"T1=[{by_id['T1']['start_time']},{by_id['T1']['end_time']}) and "
        f"T2=[{by_id['T2']['start_time']},{by_id['T2']['end_time']}) overlap — "
        "material capacity=2 with each task demanding 2 should have prevented this"
    )


def test_material_constraint_allows_parallel():
    """
    Two tasks each demand 1 roll of Yarn_Red. Capacity = 2.
    Combined demand = 2 ≤ 2 → solver is free to run them simultaneously.
    Two machines are available; workforce cap is high.
    Expected: both tasks are scheduled (feasible) — solver may or may not overlap
    them, but no spurious infeasibility.
    """
    payload = {
        "job_id": "test_mat_parallel",
        "config": _base_config(),
        "material_capacities": {"Yarn_Red": 2},
        "machines": [_make_machine("KM_00"), _make_machine("KM_01")],
        "resources": [_make_resource("KM_00"), _make_resource("KM_01")],
        "tasks": [
            _make_task("T1", 300, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 1}),
            _make_task("T2", 300, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 1}),
        ],
    }

    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    task_ids = {a["task_id"] for a in result["assignments"]}
    assert "T1" in task_ids and "T2" in task_ids, (
        "Both tasks should be scheduled when combined demand ≤ capacity"
    )


def test_no_material_capacities_is_noop():
    """
    Payload without material_capacities — build_material_constraints() is a no-op.
    The solver must still produce a valid schedule.
    """
    payload = {
        "job_id": "test_no_materials",
        "config": _base_config(),
        # Intentionally omit material_capacities
        "machines": [_make_machine("KM_00")],
        "resources": [_make_resource("KM_00")],
        "tasks": [_make_task("T1", 100, 500, ["KM_00"])],
    }

    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 1


@_MATERIAL_P1_DISABLED
def test_multi_material_constraint_serialises_tasks():
    """
    Yarn_Red capacity=2, Yarn_Blue capacity=2.
    T1 demands Yarn_Red=2, Yarn_Blue=1.
    T2 demands Yarn_Red=1, Yarn_Blue=2.
    Combined: Red=3 > 2 AND Blue=3 > 2 — both materials forbid concurrent execution.
    Two machines available; only material constraints force serialisation.
    Expected: T1 and T2 do NOT overlap.
    """
    payload = {
        "job_id": "test_multi_material",
        "config": _base_config(),
        "material_capacities": {"Yarn_Red": 2, "Yarn_Blue": 2},
        "machines": [_make_machine("KM_00"), _make_machine("KM_01")],
        "resources": [_make_resource("KM_00"), _make_resource("KM_01")],
        "tasks": [
            _make_task("T1", 200, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 2, "Yarn_Blue": 1}),
            _make_task("T2", 300, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 1, "Yarn_Blue": 2}),
        ],
    }

    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 2

    by_id = {a["task_id"]: a for a in result["assignments"]}
    assert not _overlaps(by_id["T1"], by_id["T2"]), (
        f"T1=[{by_id['T1']['start_time']},{by_id['T1']['end_time']}) and "
        f"T2=[{by_id['T2']['start_time']},{by_id['T2']['end_time']}) overlap — "
        "combined demand exceeds all material caps"
    )


def test_task_without_material_demand_is_not_constrained():
    """
    T1 has material_demands; T2 does not.
    T2 should be unaffected by the material capacity — the constraint applies
    only to tasks that declare a demand.
    """
    payload = {
        "job_id": "test_partial_demand",
        "config": _base_config(),
        "material_capacities": {"Yarn_Red": 1},
        "machines": [_make_machine("KM_00"), _make_machine("KM_01")],
        "resources": [_make_resource("KM_00"), _make_resource("KM_01")],
        "tasks": [
            _make_task("T1", 200, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 1}),
            # T2 has no material_demands — should not consume any Yarn_Red
            _make_task("T2", 150, 2000, ["KM_00", "KM_01"]),
        ],
    }

    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    task_ids = {a["task_id"] for a in result["assignments"]}
    assert "T1" in task_ids and "T2" in task_ids, (
        "T2 (no material demand) should be schedulable regardless of material cap"
    )


@_MATERIAL_P1_DISABLED
def test_material_constraint_with_pinned_task():
    """
    Pinned task occupies Yarn_Red for its fixed window.
    Free task also needs Yarn_Red=1; capacity=1.
    They cannot overlap in material usage → free task must start after pinned task ends.
    """
    payload = {
        "job_id": "test_mat_pinned",
        "config": _base_config(),
        "material_capacities": {"Yarn_Red": 1},
        "machines": [_make_machine("KM_00"), _make_machine("KM_01")],
        "resources": [_make_resource("KM_00"), _make_resource("KM_01")],
        "tasks": [
            # Pinned task: [0, 300) on KM_00, demands 1 roll
            _make_task(
                "T_pinned", 300, 2000, ["KM_00"],
                {"Yarn_Red": 1},
                is_pinned=True,
                pinned_machine_id="KM_00",
                pinned_start_time=0,
                pinned_end_time=300,
            ),
            # Free task: also needs 1 roll — must wait until T_pinned finishes
            _make_task("T_free", 200, 2000, ["KM_00", "KM_01"], {"Yarn_Red": 1}),
        ],
    }

    result = Engine(payload).solve()
    assert result["status"] in ("feasible", "optimal")
    assert len(result["assignments"]) == 2

    by_id = {a["task_id"]: a for a in result["assignments"]}
    assert not _overlaps(by_id["T_pinned"], by_id["T_free"]), (
        "Pinned and free task overlap in material usage despite capacity=1"
    )
    # Free task must start at or after pinned task's end (300)
    assert by_id["T_free"]["start_time"] >= 300, (
        f"T_free starts at {by_id['T_free']['start_time']} before pinned T_pinned ends at 300"
    )
