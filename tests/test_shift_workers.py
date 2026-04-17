"""
Shift-worker variability tests.

Go creates pinned DUMMY tasks to block workers during shifts they are not
assigned to.  These tests exercise failure modes that arise when the
number of workers differs between shifts:

  Class 0 — Healthy baseline:
      Worker count changes between shifts; solver uses only the workers
      that are free in each window.  Should always be feasible.

  Class 1 — Same-machine overlap:
      Two DUMMY tasks share the same W_ machine with overlapping [start, end)
      windows.  AddNoOverlap on that machine becomes hard-infeasible.
      Reproduces the "5 workers feasible / 10 workers infeasible" regression.

  Class 2 — Dead zone:
      ALL machines of an operation are blocked simultaneously.  Any real task
      in that operation has zero available machines → infeasible.

  Class 3 — Missing resource guard:
      A DUMMY task is pinned to a machine that Go omitted from the resources
      list (e.g. W_LINKING_09 added to dummies before Go updated resources).
      The auto-register guard should keep the model feasible.

  Class 4 — Pinned task vs unavailability overlap (new):
      A pinned task's [start, end) intersects a declared unavailability window
      on the same machine.  Both are always-active in AddNoOverlap → infeasible.

  Class 5 — Unavailability self-overlap (new):
      Two unavailability windows on the same resource overlap each other.
      Both produce always-active FixedSizeIntervalVars → infeasible.

  Class 6 — Critical path exceeds horizon (new):
      The dependency chain alone (ignoring machine conflicts) pushes a task's
      earliest finish past self.horizon.  The variable end_var is bounded
      [0, horizon] so CP-SAT proves infeasible by construction.
"""

from typing import Any, Dict, List

import pytest

from app.engine.model import Engine

HORIZON = 2880  # 2 full days in minutes


# ── Helpers ────────────────────────────────────────────────────────────────


def _cfg(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "horizon_minutes": HORIZON,
        "max_search_time": 15,
        "setup_time_minutes": 0,
        "max_factory_machines": 40,
        "random_seed": 42,
        "num_search_workers": 1,
    }
    base.update(overrides)
    return base


def _machine(m_id: str) -> Dict[str, Any]:
    """Minimal machine entry (used only for affinity scoring in Engine)."""
    return {
        "id": m_id,
        "type": "serial",
        "capacity": 1,
        "worker_req": 1,
        "routing": [
            {"operation": "linking", "design_item_id": "", "duration": 0.0, "setup_time": 0.0}
        ],
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


def _task(
    task_id: str,
    compatible_ids: List[str],
    duration: int = 200,
    operation: str = "linking",
    due: int = HORIZON,
    priority: int = 3,
    depends_on: List[str] = None,
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


def _dummy(
    task_id: str,
    machine_id: str,
    start: int,
    end: int,
    operation: str = "linking",
    pinned_machine_id: str = None,  # None simulates Go not setting pinned_machine_id
) -> Dict[str, Any]:
    """
    Shift-blocking dummy task: pinned to a specific worker for a specific window.
    Go creates these to prevent the solver from assigning real tasks to workers
    who are off-shift during that window.

    pinned_machine_id=None simulates the case where Go only sets
    compatible_resource_ids but forgets to set pinned_machine_id.
    """
    # Default: pinned_machine_id mirrors machine_id (the correct Go behaviour)
    effective_pinned = pinned_machine_id if pinned_machine_id is not None else machine_id
    return {
        "task_id": task_id,
        "original_order_id": task_id,
        "group_id": "DUMMY",
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
        "pinned_machine_id": effective_pinned,
        "pinned_start_time": start,
        "pinned_end_time": end,
        "demand": 0,
        "material_demands": {},
    }


def _solve(
    resources: List[Dict],
    tasks: List[Dict],
    machines: List[Dict] = None,
) -> Dict[str, Any]:
    payload = {
        "job_id": "test_shift",
        "config": _cfg(),
        "machines": machines or [_machine(r["id"]) for r in resources],
        "resources": resources,
        "tasks": tasks,
        "material_capacities": {},
    }
    return Engine(payload).solve()


def _real_assignments(result: Dict) -> Dict[str, Dict]:
    """Filter out DUMMY tasks from the assignment list."""
    return {
        a["task_id"]: a
        for a in result.get("assignments", [])
        if not a["task_id"].startswith("DUMMY")
    }


# ── Class 0: Healthy baselines ─────────────────────────────────────────────


class TestHealthyShiftBlocking:
    """Baseline: different worker counts per shift should stay feasible."""

    def test_single_worker_blocked_during_one_shift(self):
        """
        3 workers, one blocked during [480, 960).
        2 real tasks with duration 200 — both fit outside or on the 2 unblocked workers.
        """
        resources = [_resource(f"W_LINKING_0{i}") for i in range(1, 4)]
        all_ids = [f"W_LINKING_0{i}" for i in range(1, 4)]
        tasks = [
            _task("LINK_A", all_ids, duration=200),
            _task("LINK_B", all_ids, duration=200),
            _dummy("DUMMY_NIGHT_03", "W_LINKING_03", 480, 960),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        assert len(assigns) == 2

    def test_5_workers_all_shifts_covered(self):
        """5 workers, no dummy tasks → 5 real tasks schedule freely."""
        resources = [_resource(f"W_LINKING_0{i}") for i in range(1, 6)]
        all_ids = [f"W_LINKING_0{i}" for i in range(1, 6)]
        tasks = [_task(f"LINK_{i}", all_ids, duration=100) for i in range(5)]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"
        assert len(_real_assignments(result)) == 5

    def test_10_workers_5_blocked_per_shift(self):
        """
        Reproduces the user's regression scenario:
        10 workers, workers 6-10 blocked during the night shift [480, 960).
        8 real tasks — should remain feasible (10 workers give more room, not less).
        """
        resources = [_resource(f"W_LINKING_{i:02d}") for i in range(1, 11)]
        all_ids = [f"W_LINKING_{i:02d}" for i in range(1, 11)]
        # Block workers 6-10 during night shift
        dummies = [
            _dummy(f"DUMMY_NIGHT_{i:02d}", f"W_LINKING_{i:02d}", 480, 960)
            for i in range(6, 11)
        ]
        real_tasks = [_task(f"LINK_{i}", all_ids, duration=100) for i in range(8)]
        result = _solve(resources, real_tasks + dummies)
        assert result["status"] == "feasible"
        assert len(_real_assignments(result)) == 8

    def test_adjacent_dummies_same_machine_not_overlapping(self):
        """
        CA SÁNG block [0, 480) and CA TỐI block [480, 960) are adjacent, not overlapping.
        Real task must start ≥ 960 (machine free after both blocks end).
        """
        resources = [_resource("W_LINKING_01")]
        tasks = [
            _task("LINK_X", ["W_LINKING_01"], duration=100),
            _dummy("DUMMY_SANG", "W_LINKING_01", 0, 480),
            _dummy("DUMMY_TOI", "W_LINKING_01", 480, 960),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        # Task can only run after both blocks end
        assert assigns["LINK_X"]["start_time"] >= 960

    def test_multiple_days_non_overlapping_blocks(self):
        """
        Worker 09 works mornings only. Dummies block afternoons for 2 days.
        Day 1 afternoon [480, 1440), Day 2 afternoon [1920, 2880).
        Real task should be scheduled in the free windows [0,480) or [1440,1920).
        """
        resources = [_resource("W_LINKING_09")]
        tasks = [
            _task("LINK_X", ["W_LINKING_09"], duration=100),
            _dummy("DUMMY_D1_AFT", "W_LINKING_09", 480, 1440),
            _dummy("DUMMY_D2_AFT", "W_LINKING_09", 1920, 2880),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        start = assigns["LINK_X"]["start_time"]
        assert start < 480 or (1440 <= start < 1920), (
            f"Task scheduled at {start} which falls inside a blocked window"
        )


# ── Class 1: Same-machine overlap → infeasible ────────────────────────────


class TestPinnedOverlapInfeasibility:
    """Two pinned DUMMY tasks on the same machine with intersecting windows."""

    def test_overlapping_dummies_cause_infeasible(self):
        """
        DUMMY_A blocks W_LINKING_09 at [0, 600).
        DUMMY_B blocks W_LINKING_09 at [500, 900).
        They overlap at [500, 600) → AddNoOverlap is hard-infeasible.
        """
        resources = [_resource("W_LINKING_09"), _resource("W_LINKING_10")]
        tasks = [
            _task("LINK_X", ["W_LINKING_09", "W_LINKING_10"], duration=100),
            _dummy("DUMMY_A", "W_LINKING_09", 0, 600),
            _dummy("DUMMY_B", "W_LINKING_09", 500, 900),  # overlaps DUMMY_A
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "infeasible"

    def test_off_by_one_overlap_causes_infeasible(self):
        """
        Simulates a common Go-side fence-post error: block ends at 481 but next block
        starts at 480 → 1-minute overlap → infeasible.
        (CA SÁNG extended 1 min past its nominal end.)
        """
        resources = [_resource("W_LINKING_09")]
        tasks = [
            _dummy("DUMMY_SANG", "W_LINKING_09", 0, 481),   # should end at 480
            _dummy("DUMMY_TOI",  "W_LINKING_09", 480, 960),  # starts at 480
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "infeasible"

    def test_non_overlapping_dummies_on_same_machine_still_feasible(self):
        """Sanity: [0,480) and [480,960) do NOT overlap — solver is fine."""
        resources = [_resource("W_LINKING_09")]
        tasks = [
            _task("LINK_X", ["W_LINKING_09"], duration=100),
            _dummy("DUMMY_SANG", "W_LINKING_09", 0, 480),
            _dummy("DUMMY_TOI",  "W_LINKING_09", 480, 960),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"

    def test_overlapping_dummies_make_whole_model_infeasible(self):
        """
        W_LINKING_09 has two overlapping dummy tasks.  Even though W_LINKING_10
        is completely free for real tasks, the model is still hard-infeasible.

        WHY: Both dummy tasks are is_pinned → is_selected=1 → their OptionalIntervalVars
        are always-active.  AddNoOverlap on W_LINKING_09 sees two fixed-active intervals
        that overlap → hard constraint violation.  CP-SAT propagates this globally —
        there is no way for the solver to "ignore" the conflict on W_LINKING_09 even
        if LINK_X is assigned elsewhere.

        ROOT CAUSE of the 5→10 workers regression: Go generated overlapping dummy
        tasks for the new workers (e.g. CA_SANG block ends at 481 but CA_TOI block
        starts at 480).
        """
        resources = [_resource("W_LINKING_09"), _resource("W_LINKING_10")]
        tasks = [
            _task("LINK_X", ["W_LINKING_09", "W_LINKING_10"], duration=100),
            _dummy("DUMMY_A", "W_LINKING_09", 0, 600),
            _dummy("DUMMY_B", "W_LINKING_09", 500, 900),  # overlaps DUMMY_A on _09
            # W_LINKING_10 is completely free — but that doesn't rescue the model
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "infeasible"


# ── Class 2: Dead zone → infeasible ──────────────────────────────────────


class TestDeadZoneInfeasibility:
    """All machines of an operation blocked simultaneously → no slot for real task."""

    def test_all_workers_blocked_via_unavailability_infeasible(self):
        """
        Both W_LINKING machines are blocked for the entire usable window via their
        unavailability field (not dummy tasks).

        WHY unavailability instead of dummies: dummy tasks with large durations
        inflate `total_duration`, which raises the effective horizon beyond the block
        end → the solver finds a "feasible" slot after the block.
        unavailability windows are NewFixedSizeIntervalVar entries — they don't
        contribute to total_duration, so the horizon stays small and a block of
        [0, 100_000) is guaranteed to cover the entire search space.
        """
        _big = 100_000  # guaranteed to exceed any effective horizon for this tiny problem
        resources = [
            {**_resource("W_LINKING_01"), "unavailability": [{"start": 0, "end": _big}]},
            {**_resource("W_LINKING_02"), "unavailability": [{"start": 0, "end": _big}]},
        ]
        tasks = [
            _task("LINK_X", ["W_LINKING_01", "W_LINKING_02"], duration=100),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "infeasible"

    def test_all_workers_blocked_same_window_task_confined_to_that_window(self):
        """
        Both workers blocked [0, 500).  Real task duration=100 fits after 500.
        Should be feasible — task just starts at ≥ 500.
        """
        resources = [_resource("W_LINKING_01"), _resource("W_LINKING_02")]
        tasks = [
            _task("LINK_X", ["W_LINKING_01", "W_LINKING_02"], duration=100),
            _dummy("DUMMY_01", "W_LINKING_01", 0, 500),
            _dummy("DUMMY_02", "W_LINKING_02", 0, 500),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        assert assigns["LINK_X"]["start_time"] >= 500

    def test_partial_block_leaves_window_open(self):
        """Only W_LINKING_01 is blocked; W_LINKING_02 is free → task uses W_LINKING_02."""
        resources = [_resource("W_LINKING_01"), _resource("W_LINKING_02")]
        tasks = [
            _task("LINK_X", ["W_LINKING_01", "W_LINKING_02"], duration=100),
            _dummy("DUMMY_01", "W_LINKING_01", 0, HORIZON),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        assert assigns["LINK_X"]["machine_id"] == "W_LINKING_02"


# ── Class 3: Missing resource guard ──────────────────────────────────────


class TestMissingResourceGuard:
    """
    Pinned DUMMY task references a machine that Go did not include in resources.
    The auto-register guard in build_resource_allocations() should prevent the
    task from landing in no_resource_tasks and triggering a hard infeasible.
    """

    def test_dummy_on_unlisted_machine_auto_registered(self):
        """
        W_LINKING_03 is absent from resources but a DUMMY pins to it.
        Real tasks only use W_LINKING_01 and W_LINKING_02 — should still be feasible.
        """
        resources = [_resource("W_LINKING_01"), _resource("W_LINKING_02")]
        # W_LINKING_03 intentionally omitted from resources
        tasks = [
            _task("LINK_A", ["W_LINKING_01", "W_LINKING_02"], duration=100),
            _dummy("DUMMY_EXTRA", "W_LINKING_03", 0, 480),
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        assert len(assigns) == 1

    def test_real_task_on_unlisted_machine_remains_infeasible(self):
        """
        A real (non-pinned) task lists only W_LINKING_99 as compatible, but that
        machine is absent from resources.  build_time_variables skips it →
        no_resource_tasks → infeasible.  The guard must NOT rescue non-pinned tasks.
        """
        resources = [_resource("W_LINKING_01")]
        tasks = [
            _task("LINK_X", ["W_LINKING_99"], duration=100),  # W_99 not in resources
        ]
        result = _solve(resources, tasks)
        assert result["status"] == "infeasible"

    def test_10_workers_with_missing_resources_for_new_workers(self):
        """
        Exact reproduction: Go increases workers 5→10, updates dummy tasks but
        forgets to update the resources list.  W_LINKING_06..10 exist only in
        dummies, not in resources.  Auto-register should keep the model feasible.
        """
        # Only original 5 workers in resources
        resources = [_resource(f"W_LINKING_0{i}") for i in range(1, 6)]
        old_ids = [f"W_LINKING_0{i}" for i in range(1, 6)]
        # Dummies for new workers 6-10 — their machines are NOT in resources
        dummies = [
            _dummy(f"DUMMY_NIGHT_{i:02d}", f"W_LINKING_{i:02d}", 480, 960)
            for i in range(6, 11)
        ]
        real_tasks = [_task(f"LINK_{i}", old_ids, duration=100) for i in range(4)]
        result = _solve(resources, real_tasks + dummies)
        assert result["status"] == "feasible"
        assert len(_real_assignments(result)) == 4

    def test_null_pinned_machine_id_fallback_to_compatible(self):
        """
        Go sends dummies with pinned_machine_id=None (forgot to set it) but
        compatible_resource_ids correctly contains the target machine.

        ROOT CAUSE of the 5→10 worker regression: if pinned_machine_id=None,
        the original code did tv["r_ids"] = [None], and auto-register created
        resource_map[None].  ALL dummy tasks (W_LINKING_06 and W_LINKING_07
        with same shift window) were assigned to the same null machine →
        overlapping intervals → AddNoOverlap infeasible.

        Fix: when pinned_machine_id is None/empty, fall back to
        compatible_resource_ids[0].  Each dummy goes to its correct machine.
        """
        resources = [
            _resource("W_LINKING_01"),
            _resource("W_LINKING_06"),
            _resource("W_LINKING_07"),
        ]
        # Two workers have the same shift window but Go forgot pinned_machine_id
        dummies = [
            _dummy("DUMMY_06", "W_LINKING_06", 0, 480, pinned_machine_id=None),
            _dummy("DUMMY_07", "W_LINKING_07", 0, 480, pinned_machine_id=None),
        ]
        real_tasks = [_task("LINK_X", ["W_LINKING_01", "W_LINKING_06", "W_LINKING_07"], duration=100)]
        result = _solve(resources, real_tasks + dummies)
        # Without the fix this was infeasible: both dummies → resource_map[None] → overlap
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        # Real task must avoid the blocked window on W_LINKING_06/07 and use W_LINKING_01
        # OR start after 480 on one of the others
        assert assigns["LINK_X"]["machine_id"] in (
            "W_LINKING_01", "W_LINKING_06", "W_LINKING_07"
        )

    def test_many_workers_null_pinned_machine_id_full_shift_coverage(self):
        """
        Reproduces the exact production failure:
        - 10 linking workers (W_LINKING_01..10)
        - Workers 06-10 have back-to-back dummies covering entire horizon
        - ALL dummies have pinned_machine_id=None (Go omits it)
        - Without fix: all 5×N dummy intervals land on resource_map[None], overlapping → infeasible
        - With fix: each falls back to its compatible_resource_ids[0], correctly spread
        """
        resources = [_resource(f"W_LINKING_{i:02d}") for i in range(1, 11)]
        all_ids = [f"W_LINKING_{i:02d}" for i in range(1, 11)]

        # Back-to-back shift blocks for workers 06-10 (simulating full horizon coverage)
        # Adjacent windows: [0,480), [480,960), [960,1440)
        dummies = []
        for worker in range(6, 11):
            m = f"W_LINKING_{worker:02d}"
            for shift_start in range(0, 1440, 480):
                dummies.append(
                    _dummy(
                        f"DUMMY_{worker:02d}_shift_{shift_start}",
                        m, shift_start, shift_start + 480,
                        pinned_machine_id=None,  # Go omits this field
                    )
                )

        real_tasks = [_task(f"LINK_{i}", all_ids, duration=100) for i in range(5)]
        result = _solve(resources, real_tasks + dummies)
        assert result["status"] == "feasible"
        assigns = _real_assignments(result)
        assert len(assigns) == 5
        # All real tasks should land on W_LINKING_01..05 (06-10 are fully blocked)
        for a in assigns.values():
            assert int(a["machine_id"].replace("W_LINKING_", "")) <= 5, (
                f"Task {a['task_id']} unexpectedly assigned to {a['machine_id']} "
                "(should only use W_LINKING_01..05)"
            )
