"""Cold-only knitting left-shift post-pass (idle compaction without re-solving).

Knitting stalls at FEASIBLE, so the span objective leaves a machine idle even when a
task could run earlier on the SAME machine with no constraint forcing the gap.
`left_shift_cold_knitting` runs on the FINAL pipeline output (after every phase):
it pulls each cold knitting task to its earliest feasible start on its own machine
(preserving order) and leaves every non-knitting assignment byte-identical.  Because
knitting only moves EARLIER, downstream precedence (start ≥ knit_end) only relaxes and
end-to-end lateness is monotonically non-increasing.  It is NOT applied on re-schedule
(knitting is hard-kept there).
"""
from typing import Any, Dict, List, Optional

from app.engine.model import Engine
from app.engine.phases.phase1_knitting import left_shift_cold_knitting


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _knit_task(
    task_id: str,
    *,
    duration: int,
    start_after: int = 0,
    due: int = 999_999,
    priority: int = 3,
    resource_ids: Optional[List[str]] = None,
    is_pinned: bool = False,
    pinned_start: Optional[int] = None,
    pinned_end: Optional[int] = None,
    order_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": order_id or f"ORD_{task_id}",
        "group_id": order_id or f"ORD_{task_id}",
        "operation": "knitting",
        "qty": 1.0, "total_qty": 1.0, "priority": priority,
        "final_depends_on": [], "start_after_min": start_after, "due_at_min": due,
        "duration": duration, "is_slice": False, "slice_index": 0,
        "parent_task_id": "", "is_batch": False, "sub_tasks": None,
        "design_item_id": "", "color_config": "", "color": "white", "substance": "cotton",
        "compatible_resource_ids": resource_ids or ["KM_0", "KM_1"],
        "wait_offsets": None, "is_pinned": is_pinned,
        "pinned_machine_id": None, "pinned_start_time": pinned_start,
        "pinned_end_time": pinned_end, "demand": 1, "material_demands": {},
    }


def _assign(task_id: str, machine: str, start: int, end: int) -> Dict[str, Any]:
    return {
        "task_id": task_id, "machine_id": machine,
        "start_time": start, "end_time": end,
        "group_id": "", "order_id": "", "quantity": 1.0,
        "status": "ON_TIME", "batch_slot_id": "",
    }


def _config(**overrides) -> Dict[str, Any]:
    cfg = {
        "horizon_minutes": 5000, "max_search_time": 10, "max_factory_machines": 50,
        "random_seed": 42, "num_search_workers": 1, "knitting_chunk_size": 0,
    }
    cfg.update(overrides)
    return cfg


def _resources(ids: List[str]) -> List[Dict[str, Any]]:
    return [{
        "id": r, "type": "serial", "capacity": 1, "operation": "knitting",
        "unavailability": [], "design_item_id": "", "color_config": "",
        "available_at_min": 0,
    } for r in ids]


def _internal_idle(assignments: List[Dict[str, Any]], op: str = "knitting",
                   info: Optional[Dict[str, Dict]] = None) -> int:
    by_m: Dict[str, List] = {}
    for a in assignments:
        if info is not None and (info.get(a["task_id"], {}).get("operation", "").lower() != op):
            continue
        by_m.setdefault(a["machine_id"], []).append((a["start_time"], a["end_time"]))
    total = 0
    for its in by_m.values():
        its.sort()
        pe = None
        for s, e in its:
            if pe is not None and s > pe:
                total += s - pe
            pe = e
    return total


# ---------------------------------------------------------------------------
# Unit tests on the post-pass (deterministic, no solver)
# ---------------------------------------------------------------------------

def test_left_shift_closes_same_machine_gap():
    """A free task scheduled late behind an idle slot on its own machine is pulled
    forward → the gap disappears.  Mutation guard: a no-op post-pass leaves idle=400."""
    tasks = [_knit_task("A", duration=100), _knit_task("B", duration=100)]
    asg = [_assign("A", "KM_0", 0, 100), _assign("B", "KM_0", 500, 600)]
    assert _internal_idle(asg) == 400  # pre-condition (mutation baseline)

    moved = left_shift_cold_knitting(asg, tasks, _config())

    assert moved == 1
    assert _internal_idle(asg) == 0
    b = {a["task_id"]: a for a in asg}["B"]
    assert (b["start_time"], b["end_time"]) == (100, 200)


def test_left_shift_respects_release_keeps_legit_gap():
    """A gap forced by start_after_min is LEGIT and must remain — never move a task
    before its release."""
    tasks = [_knit_task("A", duration=100), _knit_task("B", duration=100, start_after=300)]
    asg = [_assign("A", "KM_0", 0, 100), _assign("B", "KM_0", 500, 600)]

    left_shift_cold_knitting(asg, tasks, _config())

    b = {a["task_id"]: a for a in asg}["B"]
    assert b["start_time"] == 300            # pulled only to its release
    assert _internal_idle(asg) == 200        # 100..300 gap is legitimate


def test_left_shift_never_increases_lateness():
    """Tasks only move earlier → end ≤ old end ⇒ tardiness cannot increase; a late
    task can become on-time and its status is recomputed."""
    tasks = [_knit_task("A", duration=100), _knit_task("B", duration=100, due=250)]
    asg = [_assign("A", "KM_0", 0, 100), _assign("B", "KM_0", 500, 600)]
    asg[1]["status"] = "LATE"
    left_shift_cold_knitting(asg, tasks, _config())
    b = {a["task_id"]: a for a in asg}["B"]
    assert b["end_time"] == 200 and b["end_time"] <= 250
    assert b["status"] == "ON_TIME"


def test_left_shift_leaves_pinned_immovable():
    """In-progress (pinned) tasks are anchors — never moved; free tasks pack around
    them in order."""
    tasks = [
        _knit_task("P", duration=100, is_pinned=True, pinned_start=200, pinned_end=300),
        _knit_task("F", duration=50),
    ]
    asg = [_assign("P", "KM_0", 200, 300), _assign("F", "KM_0", 800, 850)]
    left_shift_cold_knitting(asg, tasks, _config())
    by = {a["task_id"]: a for a in asg}
    assert (by["P"]["start_time"], by["P"]["end_time"]) == (200, 300)  # untouched
    assert by["F"]["start_time"] == 300      # packs right after the anchor


def test_left_shift_leaves_non_knitting_assignments_untouched():
    """Only knitting assignments are rewritten — downstream rows are byte-identical."""
    tasks = [_knit_task("K", duration=100)]
    link = {"task_id": "L", "operation": "linking", "machine_id": "LM_0",
            "duration": 50, "start_after_min": 0, "due_at_min": 999}
    asg = [_assign("K", "KM_0", 500, 600),
           {"task_id": "L", "machine_id": "LM_0", "start_time": 700, "end_time": 750,
            "group_id": "", "order_id": "", "quantity": 1.0, "status": "ON_TIME",
            "batch_slot_id": ""}]
    left_shift_cold_knitting(asg, tasks + [link], _config())
    by = {a["task_id"]: a for a in asg}
    assert (by["K"]["start_time"], by["K"]["end_time"]) == (0, 100)        # knitting shifted
    assert (by["L"]["start_time"], by["L"]["end_time"]) == (700, 750)      # linking untouched


def test_left_shift_skipped_when_workforce_would_exceed_cap():
    """If compaction would push concurrent knitting + capacity_block demand over
    max_factory_machines, the schedule is kept UNCHANGED (never ship infeasible)."""
    block = {
        "task_id": "CB", "operation": "capacity_block", "group_id": "DUMMY",
        "duration": 1000, "demand": 1, "is_pinned": True,
        "pinned_start_time": 0, "pinned_end_time": 1000,
        "compatible_resource_ids": [], "start_after_min": 0, "due_at_min": 1000,
    }
    tasks = [block, _knit_task("A", duration=100, start_after=0)]
    asg = [_assign("A", "KM_0", 500, 600)]   # currently inside the block window
    moved = left_shift_cold_knitting(asg, tasks, _config(max_factory_machines=1))
    assert moved == 0
    assert asg[0]["start_time"] == 500       # NOT moved — cap would be exceeded


# ---------------------------------------------------------------------------
# Pipeline-level scope (cold compacts; re-schedule is hard-kept, not shifted)
# ---------------------------------------------------------------------------

def _payload(tasks, reschedule_hint=None):
    return {
        "config": _config(),
        "machines": [{"id": "KM_0", "design_item_id": "", "color_config": ""}],
        "resources": _resources(["KM_0"]),
        "tasks": tasks,
        "reschedule_hint": reschedule_hint,
    }


def test_pipeline_cold_has_no_spurious_internal_gaps():
    """End-to-end through Engine: a cold solve has zero same-machine knitting idle
    when nothing forces a gap (all start_after=0, single machine)."""
    tasks = [_knit_task(f"T{i}", duration=120, resource_ids=["KM_0"], order_id=f"O{i}")
             for i in range(5)]
    out = Engine(_payload(tasks)).solve()
    assert out["status"] == "feasible"
    info = {t["task_id"]: t for t in tasks}
    assert _internal_idle(out["assignments"], info=info) == 0


def test_pipeline_reschedule_not_left_shifted():
    """Re-schedule is the hard-keep regime — the cold post-pass must not run, so a
    task kept at its previous (late) start retains the gap before it."""
    tasks = [_knit_task("A", duration=100, resource_ids=["KM_0"], order_id="OA"),
             _knit_task("B", duration=100, resource_ids=["KM_0"], order_id="OB")]
    hint = {
        "previous_assignments": [
            {"task_id": "A", "machine_id": "KM_0", "start_time": 0,
             "end_time": 100, "original_order_id": "OA"},
            {"task_id": "B", "machine_id": "KM_0", "start_time": 500,
             "end_time": 600, "original_order_id": "OB"},
        ],
        "match_by_order_fallback": False,
    }
    out = Engine(_payload(tasks, reschedule_hint=hint)).solve()
    assert out["status"] == "feasible"
    info = {t["task_id"]: t for t in tasks}
    b = {a["task_id"]: a for a in out["assignments"]}["B"]
    assert b["start_time"] == 500            # kept late (hard-keep) → gap survives
    assert _internal_idle(out["assignments"], info=info) == 400
