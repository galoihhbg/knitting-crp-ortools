"""
End-of-shift washing flush — COLD-only post-pass (`flush_unwashed_end_of_shift`).

Pulls washing tasks that became ready before a shift boundary but spilled into a
later shift into a pre-break batch ending exactly at the boundary, IF a compatible
machine's window is free.  Moves washing EARLIER only → downstream untouched, no
added lateness, machine no-overlap preserved.  These tests call the post-pass
directly on hand-built assignments (no solver needed).
"""
from typing import Any, Dict, List

from app.engine.phases.phase3_batching import flush_unwashed_end_of_shift

SHIFTS = [480, 960, 1440, 1920]
CFG = {"washing_batch_capacity": 50}


def _wash_task(tid, dep, rids, qty=10, dur=60, due=5000, pinned=False,
               color="red", subs="cotton"):
    return {
        "task_id": tid, "operation": "Washing", "final_depends_on": [dep] if dep else [],
        "compatible_resource_ids": rids, "qty": float(qty), "duration": dur,
        "due_at_min": due, "is_pinned": pinned, "color": color, "substance": subs,
    }


def _dep_task(tid):
    return {"task_id": tid, "operation": "Linking", "final_depends_on": []}


def _asg(tid, machine, s, e, qty=10):
    return {"task_id": tid, "machine_id": machine, "start_time": s, "end_time": e,
            "quantity": float(qty), "status": "ON_TIME", "batch_slot_id": ""}


def _no_overlap(assigns: List[Dict[str, Any]]) -> bool:
    """No two DISTINCT cycles overlap on a machine.  Identical [s,e) intervals are the
    same batch cycle (co-located members) and are collapsed first."""
    by_m: Dict[str, set] = {}
    for a in assigns:
        by_m.setdefault(a["machine_id"], set()).add((a["start_time"], a["end_time"]))
    for ivs in by_m.values():
        ordered = sorted(ivs)
        for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
            if s2 < e1:  # distinct intervals partially overlap
                return False
    return True


def test_flush_pulls_spilled_task_to_pre_break_batch():
    """Ready at 100 (shift ends 480) but scheduled at 600 → pulled to end at 480."""
    tasks = [_dep_task("L1"), _wash_task("W1", "L1", ["WA", "WB"])]
    assigns = [_asg("L1", "WL", 40, 100), _asg("W1", "WA", 600, 660)]
    moved = flush_unwashed_end_of_shift(assigns, tasks, CFG, SHIFTS)
    w1 = next(a for a in assigns if a["task_id"] == "W1")
    assert moved == 1
    assert w1["end_time"] == 480 and w1["start_time"] == 420
    assert w1["machine_id"] in ("WA", "WB")
    assert w1["batch_slot_id"] == "flush_480"


def test_flush_never_straddles_break():
    """Ready at 450, dur 60: T−dur = 420 < ready → can't finish by 480 → NOT moved."""
    tasks = [_dep_task("L1"), _wash_task("W1", "L1", ["WA"])]
    assigns = [_asg("L1", "WL", 390, 450), _asg("W1", "WA", 600, 660)]
    moved = flush_unwashed_end_of_shift(assigns, tasks, CFG, SHIFTS)
    assert moved == 0
    assert next(a for a in assigns if a["task_id"] == "W1")["start_time"] == 600


def test_flush_respects_machine_occupancy():
    """The only window [420,480] is already busy on every compatible machine → skip."""
    tasks = [_dep_task("L1"), _wash_task("W1", "L1", ["WA"]),
             _dep_task("L2"), _wash_task("W2", "L2", ["WA"])]
    assigns = [
        _asg("L1", "WL", 40, 100), _asg("W1", "WA", 600, 660),
        _asg("L2", "WL", 40, 100), _asg("W2", "WA", 430, 480),  # occupies [430,480] on WA
    ]
    moved = flush_unwashed_end_of_shift(assigns, tasks, CFG, SHIFTS)
    assert moved == 0  # W1 can't fit [420,480] on WA (overlaps W2), no other machine
    assert _no_overlap(assigns)


def test_flush_skips_pinned_and_already_inshift():
    tasks = [_dep_task("L1"), _wash_task("W1", "L1", ["WA"], pinned=True),
             _dep_task("L2"), _wash_task("W2", "L2", ["WB"])]
    assigns = [
        _asg("L1", "WA", 40, 100), _asg("W1", "WA", 600, 660),   # pinned → keep
        _asg("L2", "WL", 40, 100), _asg("W2", "WB", 200, 260),   # already in-shift → keep
    ]
    moved = flush_unwashed_end_of_shift(assigns, tasks, CFG, SHIFTS)
    assert moved == 0
    assert next(a for a in assigns if a["task_id"] == "W1")["start_time"] == 600
    assert next(a for a in assigns if a["task_id"] == "W2")["start_time"] == 200


def test_flush_respects_capacity_and_only_moves_earlier():
    """6 spilled tasks (qty 10, cap 50) sharing 2 machines → batches ≤ cap, no overlap,
    every moved task ends ≤ its old start (monotone-earlier)."""
    tasks, assigns = [], []
    for i in range(6):
        tasks += [_dep_task(f"L{i}"), _wash_task(f"W{i}", f"L{i}", ["WA", "WB"])]
        assigns += [_asg(f"L{i}", "WL", 40, 100), _asg(f"W{i}", "WA", 1000 + i, 1060 + i)]
    moved = flush_unwashed_end_of_shift(assigns, tasks, CFG, SHIFTS)
    assert moved >= 1
    assert _no_overlap(assigns)
    # capacity per flush cycle (machine,start) ≤ cap
    cyc: Dict[tuple, float] = {}
    for a in assigns:
        if a["task_id"].startswith("W"):
            cyc[(a["machine_id"], a["start_time"])] = cyc.get((a["machine_id"], a["start_time"]), 0) + a["quantity"]
    assert all(q <= CFG["washing_batch_capacity"] for q in cyc.values())
    # only-earlier invariant
    for a in assigns:
        if a["task_id"].startswith("W") and a["batch_slot_id"].startswith("flush_"):
            assert a["end_time"] <= 1000 + 5 + 60  # ≤ any original end


def test_flush_noop_without_shift_ends():
    tasks = [_dep_task("L1"), _wash_task("W1", "L1", ["WA"])]
    assigns = [_asg("L1", "WL", 40, 100), _asg("W1", "WA", 600, 660)]
    assert flush_unwashed_end_of_shift(assigns, tasks, CFG, []) == 0
