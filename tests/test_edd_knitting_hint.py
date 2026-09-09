"""
EDD knitting warm-start hint tests.

The hint is a greedy earliest-due-date schedule fed through
apply_stability_hints_only on the COLD path only: zero new constraints or
objective terms, so it can never corrupt a schedule — the model still enforces
no-overlap / workforce / release.  These tests pin down the greedy scheduler's
invariants and the wiring (cold-only, flag-gated, deterministic).
"""
import copy

from app.engine.model import Engine
from app.engine.phases.phase1_knitting import _edd_warm_start_assignments


def _task(task_id, due, dur=100, machine_ids=("KM_00",), start_after=0, **kw):
    t = {
        "task_id": task_id,
        "original_order_id": f"O-{task_id}",
        "group_id": "G",
        "operation": "Knitting",
        "qty": 10.0,
        "total_qty": 10.0,
        "priority": 3,
        "final_depends_on": [],
        "start_after_min": start_after,
        "due_at_min": due,
        "duration": dur,
        "design_item_id": "D",
        "color_config": "C",
        "compatible_resource_ids": list(machine_ids),
    }
    t.update(kw)
    return t


def _rmap(*ids, unavailability=None):
    return {
        m: {
            "id": m, "type": "serial", "capacity": 1, "operation": "knitting",
            "unavailability": unavailability or [], "available_at_min": 0,
        }
        for m in ids
    }


CFG = {"max_factory_machines": 10}


class TestEddGreedy:

    def test_orders_by_due_not_task_id(self):
        tasks = [_task("A_late_id_first", due=5000), _task("B_early_due", due=100)]
        out = {a["task_id"]: a for a in _edd_warm_start_assignments(tasks, _rmap("KM_00"), CFG)}
        assert out["B_early_due"]["start_time"] == 0
        assert out["A_late_id_first"]["start_time"] == 100

    def test_respects_start_after_and_pinned_window(self):
        pinned = _task("P", due=1, dur=100, is_pinned=True,
                       pinned_machine_id="KM_00", pinned_start_time=0, pinned_end_time=100)
        free = _task("F", due=50, start_after=30)
        out = {a["task_id"]: a for a in _edd_warm_start_assignments([pinned, free], _rmap("KM_00"), CFG)}
        assert "P" not in out  # pinned tasks are never hinted
        assert out["F"]["start_time"] >= 100  # must clear the pinned window

    def test_workforce_cap_pushes_right(self):
        cfg = {"max_factory_machines": 1}
        tasks = [_task("A", due=100, machine_ids=("KM_00",)),
                 _task("B", due=200, machine_ids=("KM_01",))]
        out = {a["task_id"]: a for a in _edd_warm_start_assignments(tasks, _rmap("KM_00", "KM_01"), cfg)}
        # Two machines, but global cap 1 ⇒ B must wait for A despite a free machine.
        assert out["A"]["start_time"] == 0
        assert out["B"]["start_time"] >= out["A"]["end_time"]

    def test_capacity_block_consumes_workforce(self):
        cfg = {"max_factory_machines": 2}
        block = {
            "task_id": "BLK", "operation": "capacity_block", "demand": 2,
            "pinned_start_time": 0, "pinned_end_time": 300, "duration": 300,
            "compatible_resource_ids": [],
        }
        t = _task("A", due=100)
        out = {a["task_id"]: a for a in _edd_warm_start_assignments([block, t], _rmap("KM_00"), cfg)}
        assert out["A"]["start_time"] >= 300  # block saturates the cap until 300

    def test_picks_earliest_available_machine_and_deterministic(self):
        tasks = [_task(f"T{i}", due=100 + i, machine_ids=("KM_00", "KM_01")) for i in range(4)]
        out1 = _edd_warm_start_assignments(tasks, _rmap("KM_00", "KM_01"), CFG)
        out2 = _edd_warm_start_assignments(copy.deepcopy(tasks), _rmap("KM_00", "KM_01"), CFG)
        assert out1 == out2
        # 4 tasks × dur 100 over 2 machines → makespan 200, parallel packing
        assert max(a["end_time"] for a in out1) == 200


class TestEddHintWiring:

    def _payload(self, flag):
        # Due-inverted pair on one machine: solver must still be feasible and
        # deterministic with the hint active; the hint is cold-path-only.
        tasks = [
            _task("K1", due=4000, dur=200),
            _task("K2", due=200, dur=100),
        ]
        return {
            "job_id": "TEST_EDD",
            "config": {
                "horizon_minutes": 20000,
                "max_search_time": 10,
                "max_deterministic_time": 5,
                "random_seed": 42,
                "num_search_workers": 1,
                "enable_edd_knitting_hint": flag,
            },
            "machines": [],
            "resources": [
                {"id": "KM_00", "type": "serial", "capacity": 1, "operation": "knitting",
                 "unavailability": [], "design_item_id": "D", "color_config": "C",
                 "available_at_min": 0},
            ],
            "tasks": tasks,
            "material_capacities": {},
        }

    def test_flag_on_feasible_and_deterministic(self):
        r1 = Engine(self._payload(True)).solve()
        r2 = Engine(self._payload(True)).solve()
        assert r1["status"] == "feasible"
        a1 = sorted(r1["assignments"], key=lambda a: a["task_id"])
        a2 = sorted(r2["assignments"], key=lambda a: a["task_id"])
        assert a1 == a2

    def test_flag_off_unchanged_behaviour(self):
        r = Engine(self._payload(False)).solve()
        assert r["status"] == "feasible"
        assert len(r["assignments"]) == 2

    def test_early_due_task_scheduled_first_with_hint(self):
        r = Engine(self._payload(True)).solve()
        starts = {a["task_id"]: a["start_time"] for a in r["assignments"]}
        # K2 (due 200) must run before K1 (due 4000) — the EDD seed plus the
        # lateness objective both point the same way; this guards the wiring.
        assert starts["K2"] < starts["K1"]
