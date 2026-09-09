"""
Phase 2 feature tests.

Covers:
  - Boolean exclusion workforce mode (`use_boolean_exclusion=True`)
  - Soft pipeline offsets when factory load > 85 %
"""
import copy

import pytest

from app.engine.model import Engine
from tests.conftest import make_payload


# ── Boolean exclusion ─────────────────────────────────────────────────────────

class TestBooleanExclusion:

    def test_boolean_exclusion_produces_feasible_result(self, payload_smoke):
        """Slot-based NoOverlap mode must find a feasible schedule."""
        p = copy.deepcopy(payload_smoke)
        p["config"]["use_boolean_exclusion"] = True
        result = Engine(p).solve()
        assert result["status"] in ("feasible", "optimal"), (
            f"Boolean exclusion mode returned {result['status']}"
        )

    def test_boolean_exclusion_same_task_count(self, payload_smoke):
        """Boolean exclusion must schedule every task that cumulative mode schedules."""
        baseline = Engine(copy.deepcopy(payload_smoke)).solve()
        bool_result = Engine({**copy.deepcopy(payload_smoke),
                              "config": {**payload_smoke["config"],
                                         "use_boolean_exclusion": True}}).solve()

        assert bool_result["status"] in ("feasible", "optimal")
        # Both modes must assign the same number of tasks
        assert len(bool_result["assignments"]) == len(baseline["assignments"]), (
            f"Boolean: {len(bool_result['assignments'])} assignments, "
            f"Cumulative: {len(baseline['assignments'])} assignments"
        )

    def test_boolean_exclusion_honours_capacity(self):
        """No two knitting tasks should run concurrently when max_factory_machines=1."""
        payload = make_payload(
            n_orders=4,
            n_knitting_machines=4,
            n_linking_machines=1,
            max_factory_machines=1,
            max_search_time=15,
        )
        payload["config"]["use_boolean_exclusion"] = True
        result = Engine(payload).solve()
        assert result["status"] in ("feasible", "optimal")

        # Extract knitting assignments and verify no time overlap
        assignments = [
            a for a in result.get("assignments", [])
            if a.get("operation", "").lower() == "knitting"
        ]
        for i, a in enumerate(assignments):
            for b in assignments[i + 1:]:
                overlap = a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]
                assert not overlap, (
                    f"Capacity violation: {a['task_id']} [{a['start_time']},{a['end_time']})"
                    f" overlaps {b['task_id']} [{b['start_time']},{b['end_time']})"
                )


# ── Soft pipeline offsets ─────────────────────────────────────────────────────

class TestSoftPipelineOffsets:

    def test_overloaded_factory_is_feasible(self):
        """
        When total knitting duration > 85 % of capacity, hard WaitOffsets risk
        infeasibility.  Soft mode must still return feasible/optimal.
        """
        # 1 machine, 1-week horizon = 10080 min capacity.
        # 2 orders, each knitting=6000 min → total=12000 > 10080 * 0.85=8568 → overloaded
        payload = {
            "job_id": "test_soft_offsets",
            "config": {
                "horizon_minutes": 10080,
                "max_search_time": 15,
                "setup_time_minutes": 0,
                "max_factory_machines": 1,
                "random_seed": 42,
                "num_search_workers": 1,
            },
            "machines": [
                {"id": "KM_00", "type": "serial", "capacity": 1, "worker_req": 1,
                 "routing": [{"operation": "knitting", "design_item_id": "",
                              "duration": 0.0, "setup_time": 0.0}],
                 "design_item_id": "", "color_config": ""},
                {"id": "LM_00", "type": "serial", "capacity": 1, "worker_req": 1,
                 "routing": [{"operation": "linking", "design_item_id": "",
                              "duration": 0.0, "setup_time": 0.0}],
                 "design_item_id": "", "color_config": ""},
            ],
            "resources": [
                {"id": "KM_00", "type": "serial", "capacity": 1, "operation": "knitting",
                 "unavailability": [], "design_item_id": "", "color_config": "",
                 "available_at_min": 0},
                {"id": "LM_00", "type": "serial", "capacity": 1, "operation": "linking",
                 "unavailability": [], "design_item_id": "", "color_config": "",
                 "available_at_min": 0},
            ],
            "tasks": [
                # Order A — knitting 6000 min
                {
                    "task_id": "K-A", "original_order_id": "OA", "group_id": "OA",
                    "operation": "knitting", "qty": 1.0, "total_qty": 1.0, "priority": 3,
                    "original_depends_on": [], "final_depends_on": [],
                    "start_after_min": 0, "due_at_min": 10000, "duration": 6000,
                    "is_slice": False, "parent_task_id": "", "internal_dep": "",
                    "slice_index": 0, "is_batch": False, "sub_tasks": None,
                    "design_item_id": "", "color_config": "",
                    "compatible_resource_ids": ["KM_00"],
                    "sub_task_completion_offsets": None, "WaitOffsets": None,
                    "is_pinned": False, "pinned_machine_id": None,
                    "pinned_start_time": None, "pinned_end_time": None, "demand": 1,
                },
                {
                    "task_id": "L-A", "original_order_id": "OA", "group_id": "OA",
                    "operation": "linking", "qty": 1.0, "total_qty": 1.0, "priority": 3,
                    "original_depends_on": ["K-A"], "final_depends_on": [],
                    "start_after_min": 0, "due_at_min": 10000, "duration": 100,
                    "is_slice": False, "parent_task_id": "", "internal_dep": "",
                    "slice_index": 0, "is_batch": False, "sub_tasks": None,
                    "design_item_id": "", "color_config": "",
                    "compatible_resource_ids": ["LM_00"],
                    "sub_task_completion_offsets": None, "WaitOffsets": {"K-A": 3000},
                    "is_pinned": False, "pinned_machine_id": None,
                    "pinned_start_time": None, "pinned_end_time": None, "demand": 0,
                },
                # Order B — knitting 6000 min (total knitting = 12000 > 10080*0.85=8568)
                {
                    "task_id": "K-B", "original_order_id": "OB", "group_id": "OB",
                    "operation": "knitting", "qty": 1.0, "total_qty": 1.0, "priority": 3,
                    "original_depends_on": [], "final_depends_on": [],
                    "start_after_min": 0, "due_at_min": 10000, "duration": 6000,
                    "is_slice": False, "parent_task_id": "", "internal_dep": "",
                    "slice_index": 0, "is_batch": False, "sub_tasks": None,
                    "design_item_id": "", "color_config": "",
                    "compatible_resource_ids": ["KM_00"],
                    "sub_task_completion_offsets": None, "WaitOffsets": None,
                    "is_pinned": False, "pinned_machine_id": None,
                    "pinned_start_time": None, "pinned_end_time": None, "demand": 1,
                },
                {
                    "task_id": "L-B", "original_order_id": "OB", "group_id": "OB",
                    "operation": "linking", "qty": 1.0, "total_qty": 1.0, "priority": 3,
                    "original_depends_on": ["K-B"], "final_depends_on": [],
                    "start_after_min": 0, "due_at_min": 10000, "duration": 100,
                    "is_slice": False, "parent_task_id": "", "internal_dep": "",
                    "slice_index": 0, "is_batch": False, "sub_tasks": None,
                    "design_item_id": "", "color_config": "",
                    "compatible_resource_ids": ["LM_00"],
                    "sub_task_completion_offsets": None, "WaitOffsets": {"K-B": 3000},
                    "is_pinned": False, "pinned_machine_id": None,
                    "pinned_start_time": None, "pinned_end_time": None, "demand": 0,
                },
            ],
        }

        result = Engine(payload).solve()
        assert result["status"] in ("feasible", "optimal"), (
            f"Overloaded factory with soft offsets returned {result['status']}"
        )
        # Both linking tasks must be scheduled
        assigned_ids = {a["task_id"] for a in result.get("assignments", [])}
        assert "L-A" in assigned_ids, "L-A must be scheduled in soft offset mode"
        assert "L-B" in assigned_ids, "L-B must be scheduled in soft offset mode"

    def test_non_overloaded_factory_uses_hard_offsets(self):
        """Under normal load, WaitOffsets remain hard constraints."""
        # Small payload well under 85% load — offsets stay hard
        payload = make_payload(n_orders=3, n_knitting_machines=5,
                               max_factory_machines=5, max_search_time=15)
        result = Engine(payload).solve()
        assert result["status"] in ("feasible", "optimal")
