"""Cold-only linking left-shift post-pass (tighten linking to knitting).

Linking due dates are usually far off, so the phase-2 solver has no lateness incentive
to start linking early and stalls at a FEASIBLE solution that staggers slices late even
though every worker is idle and every knitting panel is ready.  `left_shift_cold_linking`
pulls each linking task to its knitting-derived earliest start on the earliest free
compatible worker.  Every task is seeded at its original (worker, start) and may only
move into an EARLIER gap, so it never ends later → washing/iron/packing release bounds
only relax → downstream byte-identical, lateness monotone non-increasing.
"""
from typing import Any, Dict, List, Optional

from app.engine.phases.phase2_linking import left_shift_cold_linking


def _knit_assign(task_id: str, machine: str, start: int, end: int) -> Dict[str, Any]:
    return {"task_id": task_id, "machine_id": machine, "start_time": start,
            "end_time": end, "status": "ON_TIME"}


def _link_task(task_id: str, *, duration: int, deps: List[str],
               workers: List[str], due: int = 999_999) -> Dict[str, Any]:
    return {
        "task_id": task_id, "operation": "linking", "duration": duration,
        "final_depends_on": deps, "wait_offsets": {}, "due_at_min": due,
        "compatible_resource_ids": workers, "is_pinned": False,
        "parent_task_id": "LK", "priority": 1,
    }


def _knit_task(task_id: str) -> Dict[str, Any]:
    return {"task_id": task_id, "operation": "knitting", "final_depends_on": []}


def _workers(ids: List[str], unavail: Optional[Dict[str, list]] = None) -> List[Dict[str, Any]]:
    unavail = unavail or {}
    return [{"id": w, "operation": "linking", "capacity": 1,
             "unavailability": unavail.get(w, []), "available_at_min": 0} for w in ids]


def _link_starts(asg):
    return [a["start_time"] for a in asg if a["task_id"].startswith("L")]


def _no_overlap(asg):
    by_w: Dict[str, List] = {}
    for a in asg:
        if not a["task_id"].startswith("L"):  # only the linking tasks are managed here
            continue
        by_w.setdefault(a["machine_id"], []).append((a["start_time"], a["end_time"]))
    for iv in by_w.values():
        iv.sort()
        for i in range(1, len(iv)):
            if iv[i][0] < iv[i - 1][1]:
                return False
    return True


def test_linking_left_shift_collapses_staggered_slices():
    """Three independent linking slices whose panels are all ready by t=100 but which
    the solver staggered to 1000/1200/1400 → pulled onto the 3 idle workers so all
    start at ~100 (makespan collapses)."""
    workers = ["W1", "W2", "W3"]
    knit = [_knit_task("K0"), _knit_task("K1"), _knit_task("K2")]
    knit_asg = [_knit_assign("K0", "M", 0, 100), _knit_assign("K1", "M2", 0, 100),
                _knit_assign("K2", "M3", 0, 100)]
    links = [_link_task(f"L{i}", duration=200, deps=[f"K{i}"], workers=workers) for i in range(3)]
    link_asg = [
        {"task_id": "L0", "machine_id": "W1", "start_time": 1000, "end_time": 1200, "status": "ON_TIME"},
        {"task_id": "L1", "machine_id": "W1", "start_time": 1200, "end_time": 1400, "status": "ON_TIME"},
        {"task_id": "L2", "machine_id": "W1", "start_time": 1400, "end_time": 1600, "status": "ON_TIME"},
    ]
    asg = knit_asg + link_asg

    moved = left_shift_cold_linking(asg, knit + links, _workers(workers), {})

    assert moved >= 2
    assert _no_overlap(asg)
    # all three can start at their panel-ready (100) on the three idle workers
    assert max(_link_starts(asg)) <= 100
    assert {a["machine_id"] for a in asg if a["task_id"].startswith("L")} == set(workers)


def test_linking_left_shift_respects_knitting_release():
    """A slice cannot start before its knitting dependency finishes."""
    workers = ["W1", "W2"]
    knit = [_knit_task("K0")]
    knit_asg = [_knit_assign("K0", "M", 0, 500)]  # panel ready only at 500
    links = [_link_task("L0", duration=100, deps=["K0"], workers=workers)]
    link_asg = [{"task_id": "L0", "machine_id": "W1", "start_time": 900,
                 "end_time": 1000, "status": "ON_TIME"}]
    asg = knit_asg + link_asg

    left_shift_cold_linking(asg, knit + links, _workers(workers), {})

    l0 = next(a for a in asg if a["task_id"] == "L0")
    assert l0["start_time"] == 500  # pulled to exactly panel-ready, not earlier


def test_linking_left_shift_is_monotone():
    """No linking task may end later than before (keeps downstream byte-identical)."""
    workers = ["W1", "W2"]
    knit = [_knit_task(f"K{i}") for i in range(4)]
    knit_asg = [_knit_assign(f"K{i}", "M", 0, 100) for i in range(4)]
    links = [_link_task(f"L{i}", duration=150, deps=[f"K{i}"], workers=workers) for i in range(4)]
    link_asg = [{"task_id": f"L{i}", "machine_id": "W1", "start_time": 100 + i * 150,
                 "end_time": 250 + i * 150, "status": "ON_TIME"} for i in range(4)]
    asg = knit_asg + link_asg
    before = {a["task_id"]: a["end_time"] for a in asg}

    left_shift_cold_linking(asg, knit + links, _workers(workers), {})

    assert _no_overlap(asg)
    for a in asg:
        assert a["end_time"] <= before[a["task_id"]]


def test_linking_left_shift_respects_worker_unavailability():
    """A worker unavailable early forces the slice onto a free worker or a later slot,
    never overlapping the unavailability window."""
    workers = ["W1", "W2"]
    knit = [_knit_task("K0")]
    knit_asg = [_knit_assign("K0", "M", 0, 100)]
    links = [_link_task("L0", duration=100, deps=["K0"], workers=workers)]
    link_asg = [{"task_id": "L0", "machine_id": "W1", "start_time": 800,
                 "end_time": 900, "status": "ON_TIME"}]
    asg = knit_asg + link_asg
    # W1 busy 0-500, W2 fully free
    res = _workers(workers, unavail={"W1": [{"start": 0, "end": 500}]})

    left_shift_cold_linking(asg, knit + links, res, {})

    l0 = next(a for a in asg if a["task_id"] == "L0")
    # earliest feasible: W2 at 100 (panel ready) — not inside W1's 0-500 window
    assert l0["start_time"] == 100 and l0["machine_id"] == "W2"
