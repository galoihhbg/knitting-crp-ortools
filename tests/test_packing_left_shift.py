"""
Cold packing left-shift post-pass (`left_shift_cold_packing`).

Packing is the LAST phase, so its due dates are the loosest and the downstream solver
has the least early-start incentive — it stalls at FEASIBLE and staggers packing minutes
past its ironing-ready time, piling ready-together slices onto ONE packing machine while
sibling machines sit idle.  This post-pass re-seats each packing task at max(release,
earliest free slot on any COMPATIBLE machine), moving it ONLY when a strictly earlier
conflict-free slot exists (remove-then-place) → no overlap, monotone (packing is terminal
so nothing downstream to disturb).  These tests call the pass directly on hand-built
assignments (no solver needed).
"""
from typing import Any, Dict, List

from app.engine.phases.phase4_downstream import left_shift_cold_packing


def _pack_task(tid, dep, rids, dur=60, due=9000, pinned=False):
    return {
        "task_id": tid, "operation": "Packing",
        "final_depends_on": [dep] if dep else [],
        "compatible_resource_ids": rids, "duration": dur,
        "due_at_min": due, "is_pinned": pinned,
    }


def _asg(tid, machine, s, e):
    return {"task_id": tid, "machine_id": machine, "start_time": s, "end_time": e,
            "status": "ON_TIME"}


def _no_overlap(assigns: List[Dict[str, Any]]) -> bool:
    by_m: Dict[str, list] = {}
    for a in assigns:
        by_m.setdefault(a["machine_id"], []).append((a["start_time"], a["end_time"]))
    for ivs in by_m.values():
        ordered = sorted(ivs)
        for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
            if s2 < e1:
                return False
    return True


def _monotone(assigns, before) -> bool:
    return all(a["end_time"] <= before[a["task_id"]] for a in assigns)


def test_reseats_onto_idle_compatible_machine():
    """Iron done at 500 but packing scheduled at 640 on PA while PB is idle → moved onto
    PB at its ready time (the FEASIBLE-stall case measured in the log)."""
    dep = {"task_id": "I1", "operation": "Iron", "final_depends_on": []}
    tasks = [dep, _pack_task("P1", "I1", ["PA", "PB"])]
    dep_ends = {"I1": 500}
    assigns = [_asg("P1", "PA", 640, 700)]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_packing(assigns, tasks, {}, dep_ends)

    p1 = next(a for a in assigns if a["task_id"] == "P1")
    assert moved == 1
    assert p1["start_time"] == 500 and p1["end_time"] == 560
    assert _monotone(assigns, before) and _no_overlap(assigns)


def test_spreads_ready_together_slices_across_machines():
    """Four slices ready together at 500, all stacked serially on PA (500,560,620,680)
    while PB/PC/PD idle → three of them spread onto the sibling machines, all starting 500."""
    dep = {"task_id": "I", "operation": "Iron", "final_depends_on": []}
    rids = ["PA", "PB", "PC", "PD"]
    tasks = [dep] + [_pack_task(f"P{i}", "I", rids) for i in range(4)]
    dep_ends = {"I": 500}
    assigns = [_asg(f"P{i}", "PA", 500 + 60 * i, 560 + 60 * i) for i in range(4)]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_packing(assigns, tasks, {}, dep_ends)

    packs = [a for a in assigns if a["task_id"].startswith("P")]
    assert moved == 3
    assert all(a["start_time"] == 500 for a in packs)          # each on its own machine now
    assert len({a["machine_id"] for a in packs}) == 4
    assert _monotone(assigns, before) and _no_overlap(assigns)


def test_respects_release_from_ironing():
    """Packing cannot start before its ironing dep ends even if a machine is free earlier."""
    dep = {"task_id": "I", "operation": "Iron", "final_depends_on": []}
    tasks = [dep, _pack_task("P", "I", ["PA", "PB"])]
    dep_ends = {"I": 800}
    assigns = [_asg("P", "PA", 900, 960)]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_packing(assigns, tasks, {}, dep_ends)
    p = next(a for a in assigns if a["task_id"] == "P")
    assert moved == 1 and p["start_time"] == 800        # pulled to release, not earlier
    assert _monotone(assigns, before)


def test_pinned_packing_is_immovable():
    dep = {"task_id": "I", "operation": "Iron", "final_depends_on": []}
    tasks = [dep, _pack_task("P", "I", ["PA", "PB"], pinned=True)]
    assigns = [_asg("P", "PA", 900, 960)]
    moved = left_shift_cold_packing(assigns, tasks, {}, {"I": 500})
    assert moved == 0
    assert next(a for a in assigns if a["task_id"] == "P")["start_time"] == 900


def test_no_move_when_already_earliest():
    dep = {"task_id": "I", "operation": "Iron", "final_depends_on": []}
    tasks = [dep, _pack_task("P", "I", ["PA"])]
    dep_ends = {"I": 440}
    assigns = [_asg("P", "PA", 440, 500)]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_packing(assigns, tasks, {}, dep_ends)
    assert moved == 0
    assert _monotone(assigns, before)


def test_no_packing_tasks_is_noop():
    tasks = [{"task_id": "I", "operation": "Iron", "final_depends_on": []}]
    assigns = [_asg("I", "PA", 100, 200)]
    assert left_shift_cold_packing(assigns, tasks, {}, {}) == 0
