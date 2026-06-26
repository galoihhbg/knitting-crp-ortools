"""Cold-only knitting cross-machine SPREAD post-pass.

The solver stalls at FEASIBLE and sometimes serialises a PO's tail onto ONE machine
while other compatible machines sit idle.  Since a linking panel can't start until the
LAST of its component POs is knit, that serial tail gates the panel late.
`spread_cold_knitting` re-balances each knitting task to the earliest feasible start
across ALL its compatible machines.  Every task is seeded at its original slot and may
only move into an EARLIER gap, so it never ends later → downstream stays byte-identical
and lateness is monotonically non-increasing.  Falls back (no-op) if the extra
parallelism would breach the workforce cap.  Not applied on re-schedule.
"""
from typing import Any, Dict, List, Optional

from app.engine.phases.phase1_knitting import spread_cold_knitting


def _knit_task(
    task_id: str,
    *,
    duration: int,
    start_after: int = 0,
    due: int = 999_999,
    resource_ids: Optional[List[str]] = None,
    is_pinned: bool = False,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": f"ORD_{task_id[:3]}",
        "operation": "knitting",
        "priority": 3,
        "final_depends_on": [], "start_after_min": start_after, "due_at_min": due,
        "duration": duration, "is_pinned": is_pinned,
        "compatible_resource_ids": resource_ids or ["KM_0", "KM_1", "KM_2"],
        "demand": 1, "material_demands": {},
    }


def _assign(task_id: str, machine: str, start: int, end: int) -> Dict[str, Any]:
    return {
        "task_id": task_id, "machine_id": machine,
        "start_time": start, "end_time": end, "status": "ON_TIME",
    }


def _config(**ov) -> Dict[str, Any]:
    cfg = {"max_factory_machines": 50}
    cfg.update(ov)
    return cfg


def _max_end(asg, prefix=""):
    return max(a["end_time"] for a in asg if a["task_id"].startswith(prefix))


def _no_overlap(asg):
    by_m: Dict[str, List] = {}
    for a in asg:
        by_m.setdefault(a["machine_id"], []).append((a["start_time"], a["end_time"]))
    for iv in by_m.values():
        iv.sort()
        for i in range(1, len(iv)):
            if iv[i][0] < iv[i - 1][1]:
                return False
    return True


def test_spread_parallelises_serial_tail():
    """4 same-PO tasks serialised on ONE machine while 2 compatible machines are idle
    → spread distributes them so the PO makespan collapses (4×100 serial = 400 → ~200
    over 3 machines)."""
    pool = ["KM_0", "KM_1", "KM_2"]
    tasks = [_knit_task(f"A{i}", duration=100, resource_ids=pool) for i in range(4)]
    # All four piled onto KM_0 back-to-back; KM_1/KM_2 empty.
    asg = [_assign(f"A{i}", "KM_0", i * 100, i * 100 + 100) for i in range(4)]
    assert _max_end(asg) == 400

    moved = spread_cold_knitting(asg, tasks, _config())

    assert moved >= 2
    assert _no_overlap(asg)
    # With 3 machines and 4 tasks of 100, the makespan drops to 200 (two rounds).
    assert _max_end(asg) <= 200
    # affinity preserved
    assert all(a["machine_id"] in pool for a in asg)


def test_spread_never_pushes_a_task_later():
    """Monotonicity: no task may end later than it did before (the safety invariant
    that keeps downstream byte-identical)."""
    pool = ["KM_0", "KM_1"]
    tasks = [_knit_task(f"A{i}", duration=100, resource_ids=pool) for i in range(3)]
    asg = [
        _assign("A0", "KM_0", 0, 100),
        _assign("A1", "KM_0", 100, 200),
        _assign("A2", "KM_0", 200, 300),
    ]
    before = {a["task_id"]: a["end_time"] for a in asg}

    spread_cold_knitting(asg, tasks, _config())

    assert _no_overlap(asg)
    for a in asg:
        assert a["end_time"] <= before[a["task_id"]], f"{a['task_id']} ended later"


def test_spread_respects_release_time():
    """A task cannot be pulled before its start_after_min even onto an idle machine."""
    pool = ["KM_0", "KM_1"]
    tasks = [
        _knit_task("A0", duration=100, resource_ids=pool),
        _knit_task("A1", duration=100, start_after=300, resource_ids=pool),
    ]
    asg = [_assign("A0", "KM_0", 0, 100), _assign("A1", "KM_0", 300, 400)]

    spread_cold_knitting(asg, tasks, _config())

    a1 = next(a for a in asg if a["task_id"] == "A1")
    assert a1["start_time"] >= 300


def test_spread_falls_back_when_workforce_cap_blocks():
    """If spreading would exceed the workforce cap (max_factory_machines=1 → only one
    knitting task may run at a time), the spread is abandoned and nothing moves."""
    pool = ["KM_0", "KM_1"]
    tasks = [_knit_task(f"A{i}", duration=100, resource_ids=pool) for i in range(2)]
    asg = [_assign("A0", "KM_0", 0, 100), _assign("A1", "KM_0", 100, 200)]

    moved = spread_cold_knitting(asg, tasks, _config(max_factory_machines=1))

    assert moved == 0
    # unchanged
    assert {a["task_id"]: (a["machine_id"], a["start_time"]) for a in asg} == {
        "A0": ("KM_0", 0), "A1": ("KM_0", 100),
    }
