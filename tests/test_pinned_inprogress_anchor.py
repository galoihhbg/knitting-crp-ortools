"""In-progress pins arriving with only ONE endpoint must still be anchored.

Root cause (CP_1781170838781103042): a pinned in-progress task arrived with
`pinned_end_time=180` but `pinned_start_time=None`.  `build_resource_model`'s
`is_fully_pinned` requires BOTH endpoints, so the task was treated as time-free,
the solver floated it away from its real [0,180] slot, and a new task (BATCH_0-638_1)
was scheduled at [0,152] on the SAME machine → physical overlap with the
in-progress task at the start.

Fix: `normalize_pinned_window` infers the missing endpoint from duration so the
task is fully anchored and its machine slot is reserved in NoOverlap/Cumulative.
This lives in the shared model builder, so it covers every phase (knitting,
washing, linking, downstream).
"""
from typing import Any, Dict, List, Optional

from app.engine.shared import normalize_pinned_window
from app.engine.phases.phase1_knitting import solve_knitting, left_shift_cold_knitting, Phase1Result


def _knit(task_id, *, duration, sa=0, due=999999, rids=("KM_0",),
          is_pinned=False, ps=None, pe=None, oid=None):
    return {
        "task_id": task_id, "original_order_id": oid or f"O_{task_id}",
        "group_id": oid or f"O_{task_id}", "operation": "knitting",
        "qty": 1.0, "total_qty": 1.0, "priority": 3, "final_depends_on": [],
        "start_after_min": sa, "due_at_min": due, "duration": duration,
        "is_slice": False, "slice_index": 0, "parent_task_id": "", "is_batch": False,
        "sub_tasks": None, "design_item_id": "", "color_config": "", "color": "w",
        "substance": "c", "compatible_resource_ids": list(rids), "wait_offsets": None,
        "is_pinned": is_pinned, "pinned_machine_id": ("KM_0" if is_pinned else None),
        "pinned_start_time": ps, "pinned_end_time": pe, "demand": 1, "material_demands": {},
    }


def _resources(ids):
    return [{"id": r, "type": "serial", "capacity": 1, "operation": "knitting",
             "unavailability": [], "design_item_id": "", "color_config": "",
             "available_at_min": 0} for r in ids]


def _cfg(**o):
    c = {"horizon_minutes": 5000, "max_search_time": 10, "max_factory_machines": 50,
         "random_seed": 42, "num_search_workers": 1, "knitting_chunk_size": 0}
    c.update(o)
    return c


def _by(res):
    return {a["task_id"]: a for a in res.assignments}


# ---------------------------------------------------------------------------
# Unit tests on the shared normalizer
# ---------------------------------------------------------------------------

def test_normalize_infers_start_from_end():
    t = _knit("P", duration=180, is_pinned=True, ps=None, pe=180)
    normalize_pinned_window(t)
    assert t["pinned_start_time"] == 0 and t["pinned_end_time"] == 180


def test_normalize_infers_end_from_start():
    t = _knit("P", duration=78, is_pinned=True, ps=180, pe=None)
    normalize_pinned_window(t)
    assert t["pinned_start_time"] == 180 and t["pinned_end_time"] == 258


def test_normalize_leaves_full_pin_untouched():
    t = _knit("P", duration=78, is_pinned=True, ps=180, pe=258)
    normalize_pinned_window(t)
    assert t["pinned_start_time"] == 180 and t["pinned_end_time"] == 258


def test_normalize_ignores_non_pinned_and_zero_dur():
    free = _knit("F", duration=100)
    normalize_pinned_window(free)
    assert free["pinned_start_time"] is None
    zero = _knit("Z", duration=0, is_pinned=True, ps=None, pe=180)
    normalize_pinned_window(zero)               # dur<=0 → cannot infer
    assert zero["pinned_start_time"] is None


# ---------------------------------------------------------------------------
# Integration: end-only pin is anchored, free cannot overlap its real slot
# ---------------------------------------------------------------------------

def test_end_only_pin_is_anchored_and_reserved():
    """RED before fix: P (ps=None,pe=180,dur=180) floats away from [0,180] and a
    free task overlaps it. GREEN after: P anchored at [0,180], free starts >=180."""
    pin = _knit("P", duration=180, is_pinned=True, ps=None, pe=180, oid="OP")
    free = _knit("F", duration=100, sa=0, oid="OF")
    out = solve_knitting([pin, free], _resources(["KM_0"]), _cfg(),
                         horizon=5000, reschedule_hint=None)
    assert out.status == "feasible"
    by = _by(out)
    # P must be anchored at its real in-progress slot [0,180]
    assert (by["P"]["start_time"], by["P"]["end_time"]) == (0, 180)
    # F must NOT overlap the reserved [0,180] window on KM_0
    assert by["F"]["start_time"] >= 180


def test_left_shift_respects_anchored_end_only_pin():
    """The cold left-shift must not pull a free task into an anchored (now-inferred)
    pin slot."""
    pin = _knit("P", duration=180, is_pinned=True, ps=None, pe=180, oid="OP")
    free = _knit("F", duration=100, sa=0, oid="OF")
    # simulate a solved schedule where the pin is anchored [0,180] and free is later
    res = Phase1Result(
        status="feasible",
        assignments=[
            {"task_id": "P", "machine_id": "KM_0", "start_time": 0, "end_time": 180,
             "group_id": "", "order_id": "", "quantity": 1.0, "status": "ON_TIME", "batch_slot_id": ""},
            {"task_id": "F", "machine_id": "KM_0", "start_time": 500, "end_time": 600,
             "group_id": "", "order_id": "", "quantity": 1.0, "status": "ON_TIME", "batch_slot_id": ""},
        ],
        start_times={"P": 0, "F": 500}, end_times={"P": 180, "F": 600},
    )
    # normalize so the post-pass sees the pin as fully anchored
    normalize_pinned_window(pin)
    left_shift_cold_knitting(res.assignments, [pin, free], _cfg())
    by = _by(res)
    assert (by["P"]["start_time"], by["P"]["end_time"]) == (0, 180)   # anchor untouched
    assert by["F"]["start_time"] >= 180                               # packs AFTER the pin, no overlap
