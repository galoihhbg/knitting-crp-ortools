"""
Linking worker load-balance post-pass tests.

The post-pass re-assigns linking tasks across interchangeable linking machines to
even out per-worker load, keeping every task's [start, end] fixed.  Invariants:
  * timing unchanged  → downstream byte-identical, lateness unchanged (zero regression)
  * no machine overlap introduced
  * compatible_resource_ids, pinned tasks, unavailability, available_at_min respected
  * load imbalance (stdev / max) strictly reduced on a skewed input
"""
import collections
import statistics

from app.engine.phases.phase2_linking import balance_linking_load


def _res(rid, op="linking", unavailability=None, avail=0):
    return {"id": rid, "type": "serial", "capacity": 1, "operation": op,
            "unavailability": unavailability or [], "available_at_min": avail}


def _ltask(tid, machines=("L0", "L1", "L2", "L3"), **kw):
    t = {"task_id": tid, "operation": "Linking",
         "compatible_resource_ids": list(machines)}
    t.update(kw)
    return t


def _asg(tid, m, s, e):
    return {"task_id": tid, "machine_id": m, "start_time": s, "end_time": e}


def _loads(assigns, machines):
    load = collections.defaultdict(int)
    for a in assigns:
        load[a["machine_id"]] += a["end_time"] - a["start_time"]
    return [load.get(m, 0) for m in machines]


def _has_overlap(assigns):
    bym = collections.defaultdict(list)
    for a in assigns:
        bym[a["machine_id"]].append((a["start_time"], a["end_time"]))
    for iv in bym.values():
        iv.sort()
        for i in range(1, len(iv)):
            if iv[i][0] < iv[i - 1][1]:
                return True
    return False


class TestLinkingBalance:

    def test_spreads_skewed_load_and_keeps_timing(self):
        machines = ["L0", "L1", "L2", "L3"]
        resources = [_res(m) for m in machines]
        # 4 non-overlapping tasks all piled on L0 (could be spread to 4 machines).
        assigns = [_asg(f"T{i}", "L0", i * 100, i * 100 + 100) for i in range(4)]
        tasks = [_ltask(f"T{i}") for i in range(4)]
        before_times = {a["task_id"]: (a["start_time"], a["end_time"]) for a in assigns}

        changed = balance_linking_load(assigns, tasks, resources, {})

        assert changed > 0
        # timing byte-identical
        for a in assigns:
            assert (a["start_time"], a["end_time"]) == before_times[a["task_id"]]
        # load now balanced (stdev drops to ~0; every task could be its own machine)
        loads = _loads(assigns, machines)
        assert statistics.pstdev(loads) < 1.0
        assert not _has_overlap(assigns)

    def test_overlapping_tasks_cannot_share_machine(self):
        machines = ["L0", "L1"]
        resources = [_res(m) for m in machines]
        # Two overlapping tasks → must end on different machines.
        assigns = [_asg("A", "L0", 0, 200), _asg("B", "L0", 100, 300)]
        tasks = [_ltask("A", machines), _ltask("B", machines)]
        balance_linking_load(assigns, tasks, resources, {})
        assert assigns[0]["machine_id"] != assigns[1]["machine_id"]
        assert not _has_overlap(assigns)

    def test_respects_compatible_resource_ids(self):
        machines = ["L0", "L1", "L2"]
        resources = [_res(m) for m in machines]
        # T can only run on L2 — must stay there even though L0/L1 are emptier.
        assigns = [_asg("T", "L2", 0, 100)]
        tasks = [_ltask("T", machines=["L2"])]
        balance_linking_load(assigns, tasks, resources, {})
        assert assigns[0]["machine_id"] == "L2"

    def test_pinned_task_immovable(self):
        machines = ["L0", "L1"]
        resources = [_res(m) for m in machines]
        assigns = [_asg("P", "L1", 0, 100), _asg("F", "L1", 100, 200)]
        tasks = [
            _ltask("P", machines, is_pinned=True),
            _ltask("F", machines),
        ]
        balance_linking_load(assigns, tasks, resources, {})
        pinned = next(a for a in assigns if a["task_id"] == "P")
        assert pinned["machine_id"] == "L1"  # pinned stays

    def test_respects_unavailability_and_available_at(self):
        machines = ["L0", "L1"]
        # L0 unavailable [0,150), L1 available from 0.
        resources = [_res("L0", unavailability=[{"start": 0, "end": 150}]), _res("L1")]
        assigns = [_asg("T", "L1", 0, 100)]
        tasks = [_ltask("T", machines)]
        balance_linking_load(assigns, tasks, resources, {})
        assert assigns[0]["machine_id"] == "L1"  # cannot move into L0's unavailable window

    def test_deterministic(self):
        machines = ["L0", "L1", "L2", "L3"]
        resources = [_res(m) for m in machines]

        def fresh():
            a = [_asg(f"T{i}", "L0", i * 100, i * 100 + 100) for i in range(8)]
            t = [_ltask(f"T{i}") for i in range(8)]
            balance_linking_load(a, t, resources, {})
            return [(x["task_id"], x["machine_id"]) for x in a]

        assert fresh() == fresh()

    def test_single_machine_noop(self):
        resources = [_res("L0")]
        assigns = [_asg("T", "L0", 0, 100)]
        assert balance_linking_load(assigns, [_ltask("T", ["L0"])], resources, {}) == 0
