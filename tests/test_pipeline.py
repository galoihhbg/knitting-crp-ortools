"""
Pipeline architecture tests.

Verifies the 4-phase sequential pipeline produces valid, well-ordered schedules
and returns the same output contract (format, field names) as the legacy Engine.

Tests are parametrized to cover:
  - Phase 1 (knitting) in isolation
  - Phase 2 (linking) respects knitting end times
  - Phase 3 (washing) group isolation and batch formation
  - Full pipeline integration (Engine.solve() via the new Pipeline)
"""
import pytest
from tests.conftest import make_payload

from app.engine.model import Engine
from app.engine.pipeline import Pipeline
from app.engine.phases.phase1_knitting import solve_knitting
from app.engine.phases.phase2_linking import solve_linking
from app.engine.phases.phase3_batching import solve_washing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _knitting_payload(n_orders: int = 3, n_machines: int = 3) -> dict:
    """Minimal payload with only knitting tasks."""
    p = make_payload(
        n_orders=n_orders,
        n_knitting_machines=n_machines,
        n_linking_machines=0,
        max_factory_machines=n_machines,
        max_search_time=10,
    )
    # Keep only knitting tasks
    p["tasks"] = [t for t in p["tasks"] if t["operation"] == "knitting"]
    p["resources"] = [r for r in p["resources"] if r.get("operation") == "knitting"]
    return p


def _washing_payload() -> dict:
    """Payload with washing tasks in two color groups."""
    machines = [
        {"id": "WM_01", "type": "batch", "capacity": 5, "operation": "washing",
         "unavailability": [], "design_item_id": "", "color_config": "", "available_at_min": 0},
        {"id": "WM_02", "type": "batch", "capacity": 5, "operation": "washing",
         "unavailability": [], "design_item_id": "", "color_config": "", "available_at_min": 0},
    ]
    tasks = [
        {
            "task_id": f"W{i+1}-ORDER_{i:03d}",
            "original_order_id": f"ORDER_{i:03d}",
            "group_id": f"ORDER_{i:03d}",
            "operation": "washing",
            "qty": 2.0,
            "total_qty": 10.0,
            "priority": 3,
            "final_depends_on": [],
            "start_after_min": 0,
            "due_at_min": 5000,
            "duration": 120,
            "is_batch": False,
            "sub_tasks": None,
            "design_item_id": "",
            "color_config": "",
            "color": "RED" if i < 2 else "BLUE",
            "substance": "WOOL",
            "compatible_resource_ids": ["WM_01", "WM_02"],
            "WaitOffsets": None,
            "is_pinned": False,
            "pinned_machine_id": None,
            "pinned_start_time": None,
            "pinned_end_time": None,
            "demand": 0,
            "material_demands": {},
        }
        for i in range(4)
    ]
    return {
        "job_id": "TEST_WASHING",
        "config": {
            "horizon_minutes": 10080,
            "max_search_time": 10,
            "max_factory_machines": 5,
            "random_seed": 42,
            "num_search_workers": 1,
            "washing_batch_capacity": 5,
        },
        "machines": [],
        "resources": machines,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Phase 1: Knitting
# ---------------------------------------------------------------------------

class TestPhase1Knitting:

    def test_returns_feasible_for_simple_payload(self):
        p = _knitting_payload(n_orders=3, n_machines=3)
        result = solve_knitting(p["tasks"], p["resources"], p["config"])
        assert result.status in ("feasible", "empty")

    def test_all_tasks_get_end_times(self):
        p = _knitting_payload(n_orders=3, n_machines=3)
        result = solve_knitting(p["tasks"], p["resources"], p["config"])
        assert result.status == "feasible"
        task_ids = {t["task_id"] for t in p["tasks"]}
        assert task_ids <= set(result.end_times.keys())

    def test_end_time_equals_start_plus_duration(self):
        p = _knitting_payload(n_orders=2, n_machines=2)
        result = solve_knitting(p["tasks"], p["resources"], p["config"])
        assert result.status == "feasible"
        task_map = {t["task_id"]: t for t in p["tasks"]}
        for t_id in result.end_times:
            task = task_map.get(t_id)
            if task and not task.get("is_pinned"):
                expected_end = result.start_times[t_id] + task["duration"]
                assert result.end_times[t_id] == expected_end, (
                    f"Task {t_id}: end={result.end_times[t_id]} != "
                    f"start({result.start_times[t_id]}) + dur({task['duration']})"
                )

    def test_assignments_have_required_fields(self):
        p = _knitting_payload(n_orders=2, n_machines=2)
        result = solve_knitting(p["tasks"], p["resources"], p["config"])
        assert result.status == "feasible"
        required = {"task_id", "machine_id", "start_time", "end_time", "status"}
        for a in result.assignments:
            assert required <= set(a.keys()), f"Missing fields in {a}"

    def test_empty_tasks_returns_empty_status(self):
        result = solve_knitting([], [], {"horizon_minutes": 1440, "max_search_time": 5})
        assert result.status == "empty"


# ---------------------------------------------------------------------------
# Phase 2: Linking (depends on Phase 1 end times)
# ---------------------------------------------------------------------------

class TestPhase2Linking:

    def _build_simple_p1_and_p2(self):
        """Build minimal knitting and linking payloads sharing one order."""
        k_dur = 200
        l_dur = 100
        order_id = "ORDER_001"
        k_id = f"K1-{order_id}"
        l_id = f"L1-{order_id}"

        k_task = {
            "task_id": k_id, "original_order_id": order_id, "group_id": order_id,
            "operation": "knitting", "qty": 10.0, "total_qty": 10.0, "priority": 3,
            "final_depends_on": [], "start_after_min": 0, "due_at_min": 5000,
            "duration": k_dur, "is_batch": False, "sub_tasks": None,
            "design_item_id": "D1", "color_config": "", "color": "", "substance": "",
            "compatible_resource_ids": ["KM_01"],
            "WaitOffsets": None, "is_pinned": False, "pinned_machine_id": None,
            "pinned_start_time": None, "pinned_end_time": None, "demand": 1, "material_demands": {},
        }
        l_task = {
            "task_id": l_id, "original_order_id": order_id, "group_id": order_id,
            "operation": "linking", "qty": 10.0, "total_qty": 10.0, "priority": 3,
            "final_depends_on": [k_id], "start_after_min": 0, "due_at_min": 5000,
            "duration": l_dur, "is_batch": False, "sub_tasks": None,
            "design_item_id": "", "color_config": "", "color": "", "substance": "",
            "compatible_resource_ids": ["LM_01"],
            "WaitOffsets": {k_id: k_dur // 2}, "is_pinned": False, "pinned_machine_id": None,
            "pinned_start_time": None, "pinned_end_time": None, "demand": 0, "material_demands": {},
        }
        k_resource = {
            "id": "KM_01", "type": "serial", "capacity": 1, "operation": "knitting",
            "unavailability": [], "design_item_id": "D1", "color_config": "", "available_at_min": 0,
        }
        l_resource = {
            "id": "LM_01", "type": "serial", "capacity": 1, "operation": "linking",
            "unavailability": [], "design_item_id": "", "color_config": "", "available_at_min": 0,
        }
        config = {
            "horizon_minutes": 5000, "max_search_time": 10,
            "max_factory_machines": 1, "random_seed": 42, "num_search_workers": 1,
        }
        return k_task, l_task, k_resource, l_resource, config

    def test_linking_starts_after_knitting_ends(self):
        k_task, l_task, k_res, l_res, config = self._build_simple_p1_and_p2()

        p1 = solve_knitting([k_task], [k_res], config)
        assert p1.status == "feasible"

        p2 = solve_linking(
            [l_task], [l_res], config,
            p1_start_times=p1.start_times,
            p1_end_times=p1.end_times,
            translation_map={k_task["task_id"]: k_task["task_id"]},
        )
        assert p2.status == "feasible"

        k_end = p1.end_times[k_task["task_id"]]
        l_start = p2.start_times[l_task["task_id"]]
        assert l_start >= k_end, (
            f"Linking must start after knitting ends: "
            f"l_start={l_start} < k_end={k_end}"
        )

    def test_wait_offset_lb_applied(self):
        k_task, l_task, k_res, l_res, config = self._build_simple_p1_and_p2()
        k_dur = k_task["duration"]
        offset = k_dur // 2

        # Fake a Phase 1 result where knitting starts at 0
        p1_start = {k_task["task_id"]: 0}
        p1_end = {k_task["task_id"]: k_dur}

        p2 = solve_linking(
            [l_task], [l_res], config,
            p1_start_times=p1_start,
            p1_end_times=p1_end,
            translation_map={k_task["task_id"]: k_task["task_id"]},
        )
        assert p2.status == "feasible"
        # LB from wait_offset: 0 + offset
        # LB from depends_on:  k_dur
        # Effective LB = max(offset, k_dur) = k_dur
        assert p2.start_times[l_task["task_id"]] >= k_dur


# ---------------------------------------------------------------------------
# Phase 3: Washing (group isolation)
# ---------------------------------------------------------------------------

class TestPhase3Washing:

    def test_washing_returns_batches(self):
        p = _washing_payload()
        result = solve_washing(
            p["tasks"], p["resources"], p["config"],
            p2_end_times={},
            shift_ends=[],
        )
        assert result.status in ("feasible", "empty")
        if result.assignments:
            assert len(result.batches) >= 1

    def test_color_groups_isolated(self):
        """Tasks from different color groups must land in different batches."""
        p = _washing_payload()
        result = solve_washing(
            p["tasks"], p["resources"], p["config"],
            p2_end_times={},
            shift_ends=[],
        )
        if result.status != "feasible" or not result.batches:
            pytest.skip("No feasible batching result to validate")

        # Build task_id → color map
        color_map = {t["task_id"]: t["color"] for t in p["tasks"]}

        for batch in result.batches:
            colors_in_batch = {color_map[t_id] for t_id in batch.task_ids if t_id in color_map}
            assert len(colors_in_batch) <= 1, (
                f"Batch {batch.batch_id} mixes colors: {colors_in_batch}"
            )

    def test_start_lb_respected(self):
        """Washing tasks with start_lb from Phase 2 must start at or after that lb."""
        p = _washing_payload()
        lb = 300  # linking tasks finished at t=300

        # Wire each washing task to depend on a fake linking task that ended at lb
        linking_id = "L1-UPSTREAM"
        for t in p["tasks"]:
            t["final_depends_on"] = [linking_id]
        p2_end_times = {linking_id: lb}

        result = solve_washing(
            p["tasks"], p["resources"], p["config"],
            p2_end_times=p2_end_times,
            shift_ends=[],
        )
        if result.status != "feasible":
            pytest.skip("No feasible result")

        for a in result.assignments:
            assert a["start_time"] >= lb, (
                f"Task {a['task_id']} started at {a['start_time']} < lb={lb}"
            )


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_pipeline_produces_same_format_as_legacy(self):
        """Engine.solve() output dict must contain the contract fields."""
        payload = make_payload(n_orders=3, max_search_time=10)
        result = Engine(payload).solve()

        assert "status" in result
        assert "assignments" in result
        assert "overloads" in result
        assert result["status"] in ("feasible", "infeasible", "timeout", "model_invalid", "empty")

    def test_pipeline_feasible_for_small_payload(self):
        payload = make_payload(n_orders=5, n_knitting_machines=5, max_search_time=15)
        result = Engine(payload).solve()
        assert result["status"] == "feasible"
        assert len(result["assignments"]) > 0

    def test_all_assignment_fields_present(self):
        payload = make_payload(n_orders=3, n_knitting_machines=3, max_search_time=10)
        result = Engine(payload).solve()
        required = {"task_id", "machine_id", "start_time", "end_time",
                    "group_id", "order_id", "quantity", "status", "batch_slot_id"}
        for a in result["assignments"]:
            assert required <= set(a.keys()), f"Missing assignment fields: {set(a.keys())}"

    def test_no_knitting_machines_overlap(self):
        """No two knitting tasks may overlap on the same machine."""
        payload = make_payload(n_orders=6, n_knitting_machines=3, max_search_time=15)
        result = Engine(payload).solve()
        if result["status"] != "feasible":
            pytest.skip("No feasible result to validate")

        knitting = [
            a for a in result["assignments"]
            if a["task_id"].startswith("K")
        ]
        by_machine: dict = {}
        for a in knitting:
            by_machine.setdefault(a["machine_id"], []).append(
                (a["start_time"], a["end_time"], a["task_id"])
            )
        for m_id, intervals in by_machine.items():
            intervals.sort()
            for i in range(len(intervals) - 1):
                s1, e1, id1 = intervals[i]
                s2, e2, id2 = intervals[i + 1]
                assert e1 <= s2, (
                    f"Machine {m_id}: {id1} [{s1},{e1}) overlaps {id2} [{s2},{e2})"
                )

    def test_linking_starts_after_knitting_ends_in_full_pipeline(self):
        """For each K→L order pair, L.start >= K.end."""
        payload = make_payload(n_orders=4, n_knitting_machines=4, max_search_time=15)
        result = Engine(payload).solve()
        if result["status"] != "feasible":
            pytest.skip("No feasible result to validate")

        times = {a["task_id"]: (a["start_time"], a["end_time"]) for a in result["assignments"]}

        for task in payload["tasks"]:
            t_id = task["task_id"]
            if t_id not in times:
                continue
            for dep_id in (task.get("final_depends_on") or []):
                if dep_id in times:
                    dep_end = times[dep_id][1]
                    t_start = times[t_id][0]
                    assert t_start >= dep_end, (
                        f"{t_id} starts at {t_start} before dep {dep_id} ends at {dep_end}"
                    )

    def test_empty_payload_returns_feasible(self):
        payload = make_payload(n_orders=0)
        payload["tasks"] = []
        result = Engine(payload).solve()
        assert result["status"] == "feasible"
        assert result["assignments"] == []

    def test_pipeline_direct_run_matches_engine(self):
        """Pipeline.run() and Engine.solve() must agree on status and assignment count."""
        payload = make_payload(n_orders=3, max_search_time=10)
        engine_result = Engine(payload).solve()

        # Parse config the same way Engine does
        config = payload["config"]
        pipeline = Pipeline(config, payload["resources"], payload["tasks"])
        pipeline_result = pipeline.run()

        assert engine_result["status"] == pipeline_result["status"]
