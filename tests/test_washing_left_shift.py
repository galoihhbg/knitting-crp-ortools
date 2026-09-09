"""
Cold-only washing left-shift post-pass (`left_shift_cold_washing`).

The washing solver consolidates batches and only weakly rewards early starts, so it
can co-batch an early-ready slice with a much-later-ready slice of another order: the
batch waits on its latest member while the (often single) compatible machine sits idle.
The end-of-shift flush only fixes the slot ending exactly at a boundary; when that
pre-break window is busy it can't help even though the machine is free after the break.
This post-pass peels the early-ready members out and re-seats them in the earliest
boundary-safe free wash slot strictly earlier than the cycle's start.  Washing only
moves EARLIER → downstream untouched, no added lateness, machine no-overlap preserved.
These tests call the post-pass directly on hand-built assignments (no solver needed).
"""
from typing import Any, Dict, List

from app.engine.phases.phase3_batching import left_shift_cold_washing

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
    by_m: Dict[str, set] = {}
    for a in assigns:
        by_m.setdefault(a["machine_id"], set()).add((a["start_time"], a["end_time"]))
    for ivs in by_m.values():
        ordered = sorted(ivs)
        for (s1, e1), (s2, e2) in zip(ordered, ordered[1:]):
            if s2 < e1:
                return False
    return True


def _no_straddle(assigns) -> bool:
    # Only washing (task_id starts with "W") must not cross a break; linking deps may.
    return not any(
        a["start_time"] < b < a["end_time"]
        for a in assigns if a["task_id"].startswith("W")
        for b in SHIFTS
    )


def _monotone(assigns, before) -> bool:
    return all(a["end_time"] <= before[a["task_id"]] for a in assigns)


def test_pulls_early_member_out_of_late_batch():
    """Early slice (ready 500) bundled with a late slice (ready 900) on the single
    compatible machine → pulled out to its own earlier cycle; the late one stays."""
    tasks = [
        _dep_task("L_E"), _dep_task("L_L"),
        _wash_task("W_E", "L_E", ["WA"]),
        _wash_task("W_L", "L_L", ["WA"]),
    ]
    assigns = [
        _asg("L_E", "KL", 440, 500), _asg("L_L", "KL", 840, 900),
        _asg("W_E", "WA", 900, 960), _asg("W_L", "WA", 900, 960),
    ]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_washing(assigns, tasks, CFG, SHIFTS)

    w_e = next(a for a in assigns if a["task_id"] == "W_E")
    w_l = next(a for a in assigns if a["task_id"] == "W_L")
    assert moved == 1
    assert w_e["end_time"] < 960 and w_e["start_time"] >= 500       # pulled earlier, after ready
    assert w_l["start_time"] == 900                                  # late member untouched
    assert _monotone(assigns, before) and _no_overlap(assigns) and _no_straddle(assigns)


def test_pulls_into_next_shift_when_pre_break_busy():
    """Reproduces the WRcNIp1s3f case: the single machine is busy continuously from the
    ready time up to 640 (pre-break window occupied → flush can't help), the 640-660 gap
    is too short to wash, but the machine is free right after the 660 break → the ready
    batch is washed at the start of the next shift (660-720)."""
    shifts = [660]
    cfg = {"washing_batch_capacity": 50}
    tasks = [
        _dep_task("LX"),
        _wash_task("BP1", "", ["WA"], pinned=True),   # immovable occupant 460-520
        _wash_task("BP2", "", ["WA"], pinned=True),   # immovable occupant 520-580
        _wash_task("BP3", "", ["WA"], pinned=True),   # immovable occupant 580-640
        _wash_task("WX", "LX", ["WA"]),               # ready 457, stranded at 900
    ]
    assigns = [
        _asg("LX", "KL", 397, 457),
        _asg("BP1", "WA", 460, 520), _asg("BP2", "WA", 520, 580), _asg("BP3", "WA", 580, 640),
        _asg("WX", "WA", 900, 960),
    ]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_washing(assigns, tasks, cfg, shifts)

    wx = next(a for a in assigns if a["task_id"] == "WX")
    assert moved == 1
    assert wx["start_time"] == 660 and wx["end_time"] == 720       # earliest slot after the break
    assert _monotone(assigns, before) and _no_overlap(assigns)
    assert not any(a["start_time"] < 660 < a["end_time"] for a in assigns)


def test_merges_underfilled_trailing_cycle():
    """WzpPs3LY9b case: a 4-task batch + a lone 1-task batch on the same machine that
    together fit one wash load (cap=50) → the lone task is folded into the earlier
    batch, removing a whole wash run.  All ready well before the earlier batch starts."""
    tasks = [_dep_task("L0")] + [
        _wash_task(f"WA{i}", "L0", ["WA"], qty=10) for i in range(4)
    ] + [_wash_task("WLONE", "L0", ["WA"], qty=10)]
    assigns = [_asg("L0", "KL", 40, 100)] + [
        _asg(f"WA{i}", "WA", 400, 460, qty=10) for i in range(4)
    ] + [_asg("WLONE", "WA", 500, 560, qty=10)]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    left_shift_cold_washing(assigns, tasks, {"washing_batch_capacity": 50}, SHIFTS)

    # The lone trailing run is eliminated: all five now share ONE wash cycle (the lone
    # task folded in; the whole load may also slide earlier into idle time).
    wash = [a for a in assigns if a["task_id"].startswith("W")]
    cycles = {(a["machine_id"], a["start_time"], a["end_time"]) for a in wash}
    assert len(cycles) == 1
    assert all(a["start_time"] == next(iter(cycles))[1] for a in wash)
    assert _monotone(assigns, before) and _no_overlap(assigns)


def test_merge_respects_capacity():
    """The earlier batch is already full (cap=40, 4×10) → the lone task cannot merge.
    Ready (350) only just before the full batch and no idle gap exists before its slot,
    so it also cannot left-shift → it stays put."""
    tasks = [_dep_task("L0")] + [
        _wash_task(f"WA{i}", "L0", ["WA"], qty=10) for i in range(4)
    ] + [_wash_task("WLONE", "L0", ["WA"], qty=10)]
    assigns = [_asg("L0", "KL", 290, 350)] + [
        _asg(f"WA{i}", "WA", 400, 460, qty=10) for i in range(4)
    ] + [_asg("WLONE", "WA", 500, 560, qty=10)]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_washing(assigns, tasks, {"washing_batch_capacity": 40}, SHIFTS)
    lone = next(a for a in assigns if a["task_id"] == "WLONE")
    assert moved == 0 and lone["start_time"] == 500    # full batch → no merge, no idle gap
    assert _monotone(assigns, before)


def test_pinned_washing_is_immovable():
    tasks = [_dep_task("L1"), _wash_task("W1", "L1", ["WA"], pinned=True)]
    assigns = [_asg("L1", "KL", 40, 100), _asg("W1", "WA", 900, 960)]
    moved = left_shift_cold_washing(assigns, tasks, CFG, SHIFTS)
    assert moved == 0
    assert next(a for a in assigns if a["task_id"] == "W1")["start_time"] == 900


def test_no_move_when_already_earliest():
    """Ready 440, washed immediately at 440 with the machine otherwise busy → nothing
    earlier exists, so no move."""
    tasks = [_dep_task("L1"), _wash_task("W1", "L1", ["WA"])]
    assigns = [_asg("L1", "KL", 380, 440), _asg("W1", "WA", 440, 500)]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_washing(assigns, tasks, CFG, SHIFTS)
    assert moved == 0
    assert _monotone(assigns, before)


def test_capacity_splits_into_multiple_earlier_cycles():
    """cap=10 and three qty-10 early slices stranded together → each forms its own
    earlier cycle (a single wash load is one slice)."""
    cfg = {"washing_batch_capacity": 10}
    tasks = [_dep_task("L0")] + [
        _wash_task(f"W{i}", "L0", ["WA"], qty=10) for i in range(3)
    ]
    assigns = [_asg("L0", "KL", 40, 100)] + [
        _asg(f"W{i}", "WA", 900, 960, qty=10) for i in range(3)
    ]
    before = {a["task_id"]: a["end_time"] for a in assigns}
    moved = left_shift_cold_washing(assigns, tasks, cfg, SHIFTS)
    washed = [a for a in assigns if a["task_id"].startswith("W")]
    assert moved == 3
    assert all(a["end_time"] < 960 for a in washed)
    assert _monotone(assigns, before) and _no_overlap(assigns) and _no_straddle(assigns)
