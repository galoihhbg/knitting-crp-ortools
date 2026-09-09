"""Yarn-config re-entry repair tests (repair_yarn_config_reentry).

Rule (user 2026-07-09): interleaving orders on a knitting machine is allowed only
between tasks with the same yarn config (`color_config`, SỢI:SỐ_CUỘN).  A machine
that returns to a config it already left (:2→:5→:2) pays a double creel change.
The solver has no adjacent-config objective term, so the deterministic candidate
pass relocates the offending run onto a compatible machine whose tail already
holds the matching config; the pipeline verifies it via the relayout lateness gate
(measured trigger: CP_1783583062535099757 — 6 machines went :2→:5→:2 while 5+
machines still in :2 state sat idle).
"""
from app.engine.phases.phase1_knitting import (
    _yarn_key,
    _yarn_reentries,
    repair_yarn_config_reentry,
)

BIG_DUE = 100_000


def _task(task_id, cc, machines, due=BIG_DUE, color="Green", substance="Cotton",
          pinned=False, order=None):
    return {
        "task_id": task_id,
        "original_order_id": order or f"O_{task_id}",
        "group_id": order or f"O_{task_id}",
        "operation": "Knitting",
        "qty": 10.0,
        "total_qty": 10.0,
        "priority": 3,
        "final_depends_on": [],
        "start_after_min": 0,
        "due_at_min": due,
        "duration": 100,
        "design_item_id": "D",
        "color": color,
        "substance": substance,
        "color_config": cc,
        "compatible_resource_ids": list(machines),
        "is_pinned": pinned,
    }


def _asg(task_id, machine, start, end):
    return {"task_id": task_id, "machine_id": machine,
            "start_time": start, "end_time": end}


CFG = {"max_factory_machines": 100}


class TestYarnKey:

    def test_real_configs_compare_by_string(self):
        assert _yarn_key(_task("a", "K-YRN-GRN:2", ["M"])) == \
               _yarn_key(_task("b", "K-YRN-GRN:2", ["M"]))
        assert _yarn_key(_task("a", "K-YRN-GRN:2", ["M"])) != \
               _yarn_key(_task("b", "K-YRN-GRN:5", ["M"]))

    def test_sentinel_falls_back_to_color_substance(self):
        # Go error string carries no ':' → two broken tasks match ONLY on
        # (color, substance); Green never matches White despite identical strings.
        err = "No yarn requirements found for this design and color."
        g1 = _task("a", err, ["M"], color="Green")
        g2 = _task("b", err, ["M"], color="Green")
        w = _task("c", err, ["M"], color="White")
        assert _yarn_key(g1) == _yarn_key(g2)
        assert _yarn_key(g1) != _yarn_key(w)
        # And a broken task never equals a real config.
        assert _yarn_key(g1) != _yarn_key(_task("d", "K-YRN-GRN:2", ["M"], color="Green"))

    def test_reentry_count(self):
        a, b = ("cfg", "X:2"), ("cfg", "X:5")
        assert _yarn_reentries([a, a, b, b]) == 0
        assert _yarn_reentries([a, b, a]) == 1
        assert _yarn_reentries([a, b, a, b]) == 2


class TestRepair:

    def _fixture(self, c_due=BIG_DUE, m2_tail_cc="X:2", c_machines=("M1", "M2")):
        """M1 runs :2,:5 then a :2 re-entry (task C); M2 finished a task earlier
        and its tail config is `m2_tail_cc`."""
        tasks = [
            _task("A", "X:2", ["M1"]),
            _task("B", "X:5", ["M1"]),
            _task("C", "X:2", list(c_machines), due=c_due),
            _task("D", m2_tail_cc, ["M2"]),
        ]
        assignments = [
            _asg("A", "M1", 0, 100),
            _asg("B", "M1", 100, 200),
            _asg("C", "M1", 200, 300),   # re-entry: :2 after :5
            _asg("D", "M2", 0, 100),
        ]
        return tasks, assignments

    def test_moves_reentry_task_to_matching_tail(self):
        tasks, assignments = self._fixture()
        res = repair_yarn_config_reentry(assignments, tasks, CFG)
        assert res is not None
        assert res["machine"]["C"] == "M2"
        assert res["start"]["C"] == 100  # appended right after M2's tail
        assert res["end"]["C"] == 200
        # Untouched tasks keep their baseline placement.
        assert res["machine"]["A"] == "M1" and res["start"]["A"] == 0

    def test_no_reentry_returns_none(self):
        tasks, assignments = self._fixture()
        assignments[2]["start_time"], assignments[2]["end_time"] = 100, 200  # C before B
        assignments[1]["start_time"], assignments[1]["end_time"] = 200, 300
        assert repair_yarn_config_reentry(assignments, tasks, CFG) is None

    def test_tail_config_mismatch_returns_none(self):
        # M2's tail is :5 — appending C (:2) there would just relocate the churn.
        tasks, assignments = self._fixture(m2_tail_cc="X:5")
        assert repair_yarn_config_reentry(assignments, tasks, CFG) is None

    def test_incompatible_machine_returns_none(self):
        tasks, assignments = self._fixture(c_machines=("M1",))
        assert repair_yarn_config_reentry(assignments, tasks, CFG) is None

    def test_tight_due_never_moves_later(self):
        # M2's slot starts after C's baseline and C's due leaves no slack past its
        # baseline end (due == end, downstream chain 0) → cap forbids the move.
        # A long task on M3 keeps the knitting makespan high so ONLY the due cap
        # is exercised (not the makespan guard).
        tasks, assignments = self._fixture(c_due=300)
        assignments[3]["start_time"], assignments[3]["end_time"] = 150, 250
        tasks.append(_task("E", "X:9", ["M3"]))
        assignments.append(_asg("E", "M3", 0, 500))
        # C would land 250-350 on M2: ≤ makespan 500 but > cap 300 → rejected.
        assert repair_yarn_config_reentry(assignments, tasks, CFG) is None
        # Sanity: with a loose due the same geometry IS accepted.
        tasks[2]["due_at_min"] = BIG_DUE
        res = repair_yarn_config_reentry(assignments, tasks, CFG)
        assert res is not None and res["start"]["C"] == 250

    def test_makespan_never_extended(self):
        # M2's tail ends at the knitting makespan → appending would extend it.
        tasks, assignments = self._fixture()
        assignments[3]["start_time"], assignments[3]["end_time"] = 200, 300
        res = repair_yarn_config_reentry(assignments, tasks, CFG)
        assert res is None

    def test_pinned_run_is_immovable(self):
        tasks, assignments = self._fixture()
        tasks[2]["is_pinned"] = True
        assert repair_yarn_config_reentry(assignments, tasks, CFG) is None

    def test_workforce_cap_blocks_candidate(self):
        # Moving C to M2@100-200 raises concurrency in [100,200) to 3 > cap 2
        # (B on M1 + E on M3 + C) while the baseline peaks at 2 → whole
        # candidate dropped.
        tasks, assignments = self._fixture()
        tasks.append(_task("E", "X:9", ["M3"]))
        assignments.append(_asg("E", "M3", 100, 200))
        assert repair_yarn_config_reentry(
            assignments, tasks, {"max_factory_machines": 2}) is None

    def test_run_moves_atomically(self):
        # The offending run has TWO tasks; only one fits before the makespan →
        # neither may move (a half-moved run keeps the re-entry AND delays a task).
        tasks = [
            _task("A", "X:2", ["M1"]),
            _task("B", "X:5", ["M1"]),
            _task("C1", "X:2", ["M1", "M2"], order="OC"),
            _task("C2", "X:2", ["M1", "M2"], order="OC"),
            _task("D", "X:2", ["M2"]),
        ]
        assignments = [
            _asg("A", "M1", 0, 100),
            _asg("B", "M1", 100, 200),
            _asg("C1", "M1", 200, 300),
            _asg("C2", "M1", 300, 400),  # knit makespan 400
            _asg("D", "M2", 100, 200),   # M2 tail :2 ends 200 → C1@200-300 fits,
        ]                                #   C2@300-400 hits the makespan cap edge
        res = repair_yarn_config_reentry(assignments, tasks, CFG)
        # C1 → 200-300 and C2 → 300-400 both fit within makespan 400: moves OK.
        assert res is not None
        assert res["machine"]["C1"] == "M2" and res["machine"]["C2"] == "M2"
        # Now shrink the room: makespan 350 via a shorter C2 baseline.
        assignments[3]["start_time"], assignments[3]["end_time"] = 300, 350
        tasks[3]["duration"] = 50
        # C2 would land 300-350 on M2 (fits) — but C1 landing 200-300 pushes C2 to
        # 300-350 which still fits… so instead force failure: C2 incompatible with M2.
        tasks[3]["compatible_resource_ids"] = ["M1"]
        res2 = repair_yarn_config_reentry(assignments, tasks, CFG)
        assert res2 is None  # C1 alone must NOT move — run is atomic

    def test_real_payload_shape_multiple_machines(self):
        # Mirror of the measured trigger: several machines each end with one
        # slack-due :2 slice after a :5 block, while pure-:2 machines idle.
        tasks, assignments = [], []
        for i, m in enumerate(("V1", "V2")):  # violating machines
            tasks += [
                _task(f"a{i}", "X:2", [m]),
                _task(f"b{i}", "X:5", [m]),
                _task(f"c{i}", "X:2", [m, "P1", "P2"]),
            ]
            assignments += [
                _asg(f"a{i}", m, 0, 100),
                _asg(f"b{i}", m, 100, 800),
                _asg(f"c{i}", m, 800, 900),
            ]
        for m in ("P1", "P2"):  # pure :2 machines, idle from 500
            tasks.append(_task(f"p_{m}", "X:2", [m]))
            assignments.append(_asg(f"p_{m}", m, 0, 500))
        res = repair_yarn_config_reentry(assignments, tasks, CFG)
        assert res is not None
        moved = {tid for tid in res["machine"]
                 if res["machine"][tid] != dict((a["task_id"], a["machine_id"])
                                                for a in assignments)[tid]}
        assert moved == {"c0", "c1"}
        # Deterministic spread: each lands right after an idle pure machine's tail.
        assert {res["machine"]["c0"], res["machine"]["c1"]} == {"P1", "P2"}
        assert res["start"]["c0"] == 500 and res["start"]["c1"] == 500
