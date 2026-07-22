"""
Linking workforce-cap tests.

Linking now schedules on physical machines (like knitting), but a per-shift
worker limit still caps how many machines run at once.  Go sends that limit as
capacity_block ghost tasks tagged block_operation="linking"; the phase-2 linking
solver applies them as a factory-wide AddCumulative over the linking machine pool
(cap = max_linking_machines).  A linking block must NOT touch knitting capacity,
and vice-versa (block_operation routing).
"""
from typing import Any, Dict, List

from app.engine.model import Engine

HORIZON = 2880


def _cfg(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "horizon_minutes": HORIZON,
        "max_search_time": 15,
        "setup_time_minutes": 0,
        "max_factory_machines": 40,
        "max_linking_machines": 0,
        "random_seed": 42,
        "num_search_workers": 1,
    }
    base.update(overrides)
    return base


def _machine(m_id: str, operation: str = "linking") -> Dict[str, Any]:
    return {
        "id": m_id,
        "type": "serial",
        "capacity": 1,
        "worker_req": 1,
        "routing": [{"operation": operation, "design_item_id": "", "duration": 0.0, "setup_time": 0.0}],
        "design_item_id": "",
        "color_config": "",
    }


def _resource(r_id: str, operation: str = "linking") -> Dict[str, Any]:
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


def _task(task_id: str, compatible_ids: List[str], duration: int = 200,
          operation: str = "linking") -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": task_id,
        "group_id": "G1",
        "operation": operation,
        "qty": 1.0,
        "total_qty": 1.0,
        "priority": 3,
        "original_depends_on": [],
        "final_depends_on": [],
        "start_after_min": 0,
        "due_at_min": HORIZON,
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


def _block(task_id: str, demand: int, block_operation: str,
           start: int = 0, end: int = HORIZON) -> Dict[str, Any]:
    """A capacity_block ghost that removes `demand` machines from a stage pool."""
    return {
        "task_id": task_id,
        "original_order_id": task_id,
        "group_id": "DUMMY",
        "operation": "capacity_block",
        "block_operation": block_operation,
        "qty": 1.0,
        "total_qty": 1.0,
        "priority": 0,
        "original_depends_on": [],
        "final_depends_on": [],
        "start_after_min": start,
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
        "compatible_resource_ids": [],
        "sub_task_completion_offsets": None,
        "WaitOffsets": None,
        "is_pinned": True,
        "pinned_machine_id": None,
        "pinned_start_time": start,
        "pinned_end_time": end,
        "demand": demand,
        "material_demands": {},
    }


def _solve(resources, tasks, machines=None, **cfg) -> Dict[str, Any]:
    payload = {
        "job_id": "test_link_wf",
        "config": _cfg(**cfg),
        "machines": machines or [_machine(r["id"], r.get("operation", "linking")) for r in resources],
        "resources": resources,
        "tasks": tasks,
        "material_capacities": {},
    }
    return Engine(payload).solve()


def _reals(result: Dict) -> Dict[str, Dict]:
    return {a["task_id"]: a for a in result.get("assignments", []) if a["task_id"].startswith("L")}


def _peak_concurrency(assigns: Dict[str, Dict]) -> int:
    """Max number of intervals overlapping at any point."""
    events = []
    for a in assigns.values():
        events.append((int(a["start_time"]), 1))
        events.append((int(a["end_time"]), -1))
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


class TestLinkingWorkforceCap:
    def test_no_cap_runs_fully_parallel(self):
        """max_linking_machines=0 → no throttle → all 3 tasks run at once."""
        res = [_resource(f"LK0{i}") for i in (1, 2, 3)]
        ids = [r["id"] for r in res]
        tasks = [_task(f"L{i}", ids) for i in (1, 2, 3)]
        result = _solve(res, tasks, max_linking_machines=0)
        assert result["status"] == "feasible"
        assigns = _reals(result)
        assert len(assigns) == 3
        assert _peak_concurrency(assigns) == 3

    def test_cp_cumulative_caps_concurrency(self):
        """The phase-2 CP cumulative alone caps concurrency at max_linking_machines,
        independent of the capacity_block ghosts. Left-shift is disabled here to
        isolate the CP constraint (in production the cap equals the machine count and
        the throttle is expressed via blocks — see test_block_serializes_linking)."""
        res = [_resource(f"LK0{i}") for i in (1, 2, 3)]
        ids = [r["id"] for r in res]
        tasks = [_task(f"L{i}", ids) for i in (1, 2, 3)]
        result = _solve(res, tasks, max_linking_machines=2, enable_linking_left_shift=False)
        assert result["status"] == "feasible"
        assigns = _reals(result)
        assert _peak_concurrency(assigns) <= 2

    def test_block_serializes_linking(self):
        """A demand-2 linking block over the full horizon leaves 1 usable machine
        (3 machines, cap 3, block 2) → the 3 tasks must serialize."""
        res = [_resource(f"LK0{i}") for i in (1, 2, 3)]
        ids = [r["id"] for r in res]
        tasks = [_task(f"L{i}", ids) for i in (1, 2, 3)]
        tasks.append(_block("CAPA_BLOCK_LINKING_S1", demand=2, block_operation="linking"))
        result = _solve(res, tasks, max_linking_machines=3)
        assert result["status"] == "feasible"
        assigns = _reals(result)
        assert len(assigns) == 3
        assert _peak_concurrency(assigns) == 1, "linking block did not throttle to 1 machine"

    def test_linking_block_does_not_throttle_knitting(self):
        """A linking-tagged block must be routed to phase 2 only — knitting stays
        at full concurrency even when a high-demand linking block is present."""
        kres = [_resource(f"SK0{i}", operation="knitting") for i in (1, 2, 3)]
        kids = [r["id"] for r in kres]
        tasks = [_task(f"K{i}", kids, operation="knitting") for i in (1, 2, 3)]
        # A big linking block; there are NO linking machines/tasks, so it must be
        # inert for knitting (else knitting would be forced to serialize).
        tasks.append(_block("CAPA_BLOCK_LINKING_S1", demand=99, block_operation="linking"))
        result = _solve(kres, tasks, max_factory_machines=3, max_linking_machines=5)
        assert result["status"] == "feasible"
        kassigns = {a["task_id"]: a for a in result.get("assignments", []) if a["task_id"].startswith("K")}
        assert len(kassigns) == 3
        assert _peak_concurrency(kassigns) == 3, "linking block wrongly reduced knitting capacity"
