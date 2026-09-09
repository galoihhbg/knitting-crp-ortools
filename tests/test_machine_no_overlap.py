"""
tests/test_machine_no_overlap.py — RED test for washing machine double-booking.

Hypothesis under investigation
--------------------------------
The NoOverlap at phase3_batching.py:537-538 covers ONLY the per-slot
OptionalIntervalVars built from free tasks (machine_batch_ivs).
A pinned task's fixed interval (shared.py:700-704) is added to
resource_intervals and routed through AddCumulative (because capacity is
overridden to batch_capacity > 1).  AddCumulative allows overlap as long as
total demand ≤ capacity, so the pinned task does NOT block a free batch slot
from running concurrently on the same machine.

When machine_w (the machine-consolidation objective term) pulls free tasks
toward the machine occupied by the pinned task, the resulting schedule can
place a free batch slot whose time window overlaps the pinned task's fixed
interval — double-booking that machine for two batch CYCLES simultaneously.

Test scenario
-------------
- 1 group: color="red", substance="cotton"
- 1 pinned task P: machine=WM_00, start=480, end=540 (60-minute cycle, qty=1)
- 3 free tasks F1/F2/F3: compatible with [WM_00, WM_01], qty=1 each
  capacity=3 → all three fit in ONE batch slot.
  Their start_lb=480 (depends on linking finishing at 480), so the solver is
  forced to place the free batch at exactly t=480 on whichever machine.
  machine_w objective pushes the solver to use WM_00 (already used by P)
  rather than spin up WM_01 — creating a free batch slot [480,540] on WM_00
  that overlaps the pinned interval [480,540].

Invariant asserted (no hard-coded times)
-----------------------------------------
The check is at BATCH-CYCLE granularity (not individual task).  Multiple tasks
in the same batch slot legitimately run together in one washing cycle — they
are NOT a double-booking.  A double-booking is two DISTINCT batch cycles whose
machine-time windows overlap:
  cycle A on machine M occupies [s_A, e_A)
  cycle B on machine M occupies [s_B, e_B)
  overlap iff s_A < e_B AND s_B < e_A.

Batch cycles are constructed from:
  * Free task batch slots  → result.batches (each BatchInfo = one cycle)
  * Pinned tasks           → each pinned task is its own single-task cycle
"""

from typing import Any, Dict, List, Optional, Tuple

from app.engine.phases.phase3_batching import solve_washing, BatchInfo


# ---------------------------------------------------------------------------
# Builders (same pattern as test_short_term_deadline.py)
# ---------------------------------------------------------------------------

def _resource(r_id: str) -> Dict[str, Any]:
    return {
        "id": r_id,
        "type": "serial",
        "capacity": 1,
        "operation": "washing",
        "unavailability": [],
        "design_item_id": "",
        "color_config": "",
        "available_at_min": 0,
    }


def _wash(
    task_id: str,
    *,
    due: int,
    duration: int = 60,
    qty: int = 1,
    priority: int = 3,
    resource_ids: List[str],
    is_pinned: bool = False,
    pinned_machine_id: Optional[str] = None,
    pinned_start_time: Optional[int] = None,
    pinned_end_time: Optional[int] = None,
    depends_on: Optional[List[str]] = None,
    color: str = "red",
    substance: str = "cotton",
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": f"ORD_{task_id}",
        "group_id": f"ORD_{task_id}",
        "operation": "washing",
        "qty": float(qty),
        "total_qty": float(qty),
        "priority": priority,
        "final_depends_on": depends_on or [],
        "start_after_min": 0,
        "due_at_min": due,
        "duration": duration,
        "is_slice": False,
        "parent_task_id": "",
        "slice_index": 0,
        "is_batch": False,
        "sub_tasks": None,
        "design_item_id": "",
        "color_config": "",
        "color": color,
        "substance": substance,
        "compatible_resource_ids": resource_ids,
        "WaitOffsets": None,
        "is_pinned": is_pinned,
        "pinned_machine_id": pinned_machine_id,
        "pinned_start_time": pinned_start_time,
        "pinned_end_time": pinned_end_time,
        "demand": 1,
        "material_demands": {},
    }


def _config(horizon: int = 5000, capacity: int = 5, **overrides) -> Dict[str, Any]:
    cfg = {
        "horizon_minutes": horizon,
        "max_search_time": 30,
        "max_factory_machines": 5,
        "random_seed": 42,
        "num_search_workers": 1,
        "washing_batch_capacity": capacity,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Batch-cycle overlap checker
# ---------------------------------------------------------------------------

# A batch cycle on a machine: (machine_id, start, end, label)
MachineCycle = Tuple[str, int, int, str]


def _extract_batch_cycles(
    result,
    pinned_tasks: List[Dict[str, Any]],
) -> List[MachineCycle]:
    """
    Build the list of (machine_id, start, end, label) at BATCH-CYCLE granularity.

    One entry per BatchInfo (free task batch slot) + one entry per pinned task.
    The machine_id for each free batch is derived from the assignments of the
    tasks in that batch slot (all tasks in the same slot share one machine via
    the co-location constraint, or we take the first assignment's machine).

    Pinned tasks are each their own cycle: machine = pinned_machine_id.
    """
    cycles: List[MachineCycle] = []

    # Map task_id → machine_id from assignments
    task_machine: Dict[str, str] = {
        a["task_id"]: a["machine_id"]
        for a in result.assignments
        if not a["task_id"].startswith("__")
    }

    # Free batch slots: each BatchInfo is one cycle; machine = machine of any task in it
    for b in result.batches:
        machine_id = None
        for tid in b.task_ids:
            if tid in task_machine:
                machine_id = task_machine[tid]
                break
        if machine_id is None:
            continue  # no assignment found (e.g. all tasks unschedulable)
        cycles.append((machine_id, b.start_time, b.end_time, b.batch_id))

    # Pinned tasks: committed batch-mates that washed together share ONE
    # (machine, start, end) window — that is a single washing cycle, not N
    # overlapping cycles.  Dedup by window so batch-mates aren't flagged against
    # each other (mirrors how free batch-mates collapse into one BatchInfo).
    seen_windows: Dict[Tuple[str, int, int], str] = {}
    for t in pinned_tasks:
        if not t.get("is_pinned"):
            continue
        m = t.get("pinned_machine_id")
        s = t.get("pinned_start_time")
        e = t.get("pinned_end_time")
        if m is None or s is None or e is None:
            continue
        key = (m, int(s), int(e))
        if key in seen_windows:
            continue
        seen_windows[key] = t["task_id"]
        cycles.append((m, int(s), int(e), f"pinned:{t['task_id']}"))

    return cycles


def _overlapping_cycle_pairs(
    cycles: List[MachineCycle],
) -> List[Tuple]:
    """
    Return all pairs of same-machine BATCH CYCLES that overlap.
    Overlap: A.start < B.end AND B.start < A.end.
    """
    overlaps = []
    for i, (m1, s1, e1, label1) in enumerate(cycles):
        for (m2, s2, e2, label2) in cycles[i + 1:]:
            if m1 != m2:
                continue
            if s1 < e2 and s2 < e1:
                overlaps.append((label1, m1, s1, e1, label2, m2, s2, e2))
    return overlaps


# ---------------------------------------------------------------------------
# Shared assert helper
# ---------------------------------------------------------------------------

def _assert_no_machine_double_booking(result, pinned_tasks, test_name: str = ""):
    """
    Assert the invariant at BATCH-CYCLE granularity: no two distinct batch cycles
    on the same machine may have overlapping time windows.

    Multiple tasks in the same batch slot legitimately run together (co-batched)
    and are represented as ONE cycle entry from result.batches — so they are not
    flagged.  The pinned task is its own single-task cycle.
    """
    cycles = _extract_batch_cycles(result, pinned_tasks)

    cycle_str = "\n".join(
        f"  {label}: machine={m}, [{s}, {e})"
        for m, s, e, label in cycles
    )

    overlaps = _overlapping_cycle_pairs(cycles)
    assert not overlaps, (
        f"DOUBLE-BOOKING DETECTED{' (' + test_name + ')' if test_name else ''}:\n"
        f"Batch cycles:\n{cycle_str}\n"
        f"Overlapping pairs: {overlaps}"
    )


# ---------------------------------------------------------------------------
# Core RED test
# ---------------------------------------------------------------------------

def test_pinned_task_no_overlap_with_free_batch():
    """
    RED: machine_w consolidates free tasks onto WM_00 (already used by the
    pinned task).  The free batch's start_lb is 480 (depends on linking ending
    at 480), so it must start at 480 or later.  machine_w then places it on
    WM_00 — double-booking WM_00 at [480, 540].

    Invariant: no two DISTINCT batch cycles on the same machine may overlap.

    EXPECT: FAIL (RED) — the solver places the free batch cycle at [480,540]
    on WM_00 alongside the pinned cycle [480,540].

    After fix (pinned task's fixed interval added to machine_batch_ivs for
    AddNoOverlap): the free cycle is pushed to [540, 600] or onto WM_01 → GREEN.
    """
    PINNED_START = 480
    PINNED_END = 540
    PINNED_MACHINE = "WM_00"

    pinned = _wash(
        "P",
        due=9999,
        duration=60,
        qty=1,
        resource_ids=["WM_00"],
        is_pinned=True,
        pinned_machine_id=PINNED_MACHINE,
        pinned_start_time=PINNED_START,
        pinned_end_time=PINNED_END,
    )
    tasks = [
        pinned,
        # Free tasks: start_lb = 480 (link finishes at 480) → forced to land at 480
        # Compatible with both WM_00 and WM_01; machine_w consolidates onto WM_00
        _wash("F1", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
        _wash("F2", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
        _wash("F3", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
    ]
    resources = [_resource("WM_00"), _resource("WM_01")]

    result = solve_washing(
        tasks,
        resources,
        _config(capacity=5),
        # p2 linking finishes at 480 → free tasks start_lb = 480
        p2_end_times={"LNK": 480},
        shift_ends=[],
    )
    assert result.status == "feasible", f"Solver did not find a feasible schedule: {result.status}"

    _assert_no_machine_double_booking(result, [pinned], "core RED")


# ---------------------------------------------------------------------------
# Variant 1: Two pinned tasks on the same machine, one free batch forced between
# ---------------------------------------------------------------------------

def test_two_pinned_no_overlap_with_free_batch():
    """
    Variant: two pinned tasks on WM_00 at [0,60] and [120,180], with a gap at
    [60,120].  A free batch has start_lb=60 (linking ends at 60) and duration=60
    — exactly fills the gap.  machine_w consolidates it onto WM_00.

    The NoOverlap at machine_batch_ivs covers free-vs-free only.  The pinned
    tasks' fixed intervals are in AddCumulative — they don't block the free
    batch slot from overlapping them if demand sums are within capacity.

    EXPECT: RED — free batch [60,120] on WM_00 alongside the two pinned cycles.

    Note: the actual bug here is the free batch may also land at [0,60] or
    [60,120] overlapping P1 or staying in the gap (correct).  What we verify is
    no overlap with either pinned cycle.
    """
    PINNED_MACHINE = "WM_00"

    pinned1 = _wash(
        "P1",
        due=9999,
        duration=60,
        qty=1,
        resource_ids=["WM_00"],
        is_pinned=True,
        pinned_machine_id=PINNED_MACHINE,
        pinned_start_time=0,
        pinned_end_time=60,
    )
    pinned2 = _wash(
        "P2",
        due=9999,
        duration=60,
        qty=1,
        resource_ids=["WM_00"],
        is_pinned=True,
        pinned_machine_id=PINNED_MACHINE,
        pinned_start_time=120,
        pinned_end_time=180,
    )
    tasks = [
        pinned1,
        pinned2,
        # Free tasks: start_lb = 60, should fit in the gap [60,120]
        # machine_w will try to use WM_00 (already occupied at [0,60] and [120,180])
        _wash("F1", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
        _wash("F2", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
    ]
    resources = [_resource("WM_00"), _resource("WM_01")]

    result = solve_washing(
        tasks,
        resources,
        _config(capacity=5),
        p2_end_times={"LNK": 60},
        shift_ends=[],
    )
    assert result.status == "feasible", f"Solver infeasible: {result.status}"

    _assert_no_machine_double_booking(result, [pinned1, pinned2], "two pinned + free")


# ---------------------------------------------------------------------------
# Variant 2: Pinned at t=0, free tasks want the earliest slot (also t=0)
# ---------------------------------------------------------------------------

def test_pinned_task_no_overlap_early_pin():
    """
    Variant: pinned task at [0, 60] on WM_00.  Free tasks have start_lb=0 (no
    linking dependency) so the early-start objective + machine_w pulls the free
    batch to t=0 on WM_00 — same cycle window as the pinned task.
    """
    PINNED_START = 0
    PINNED_END = 60
    PINNED_MACHINE = "WM_00"

    pinned = _wash(
        "P",
        due=9999,
        duration=60,
        qty=1,
        resource_ids=["WM_00"],
        is_pinned=True,
        pinned_machine_id=PINNED_MACHINE,
        pinned_start_time=PINNED_START,
        pinned_end_time=PINNED_END,
    )
    tasks = [
        pinned,
        _wash("F1", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"]),
        _wash("F2", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"]),
    ]
    resources = [_resource("WM_00"), _resource("WM_01")]

    result = solve_washing(
        tasks,
        resources,
        _config(capacity=5),
        p2_end_times={},
        shift_ends=[],
    )
    assert result.status == "feasible", f"Solver infeasible: {result.status}"

    _assert_no_machine_double_booking(result, [pinned], "early pin at t=0")


# ---------------------------------------------------------------------------
# Variant 3: reschedule_hint pins free tasks to WM_00 at the pinned time
# ---------------------------------------------------------------------------

def test_pinned_task_no_overlap_with_reschedule_hint():
    """
    Variant: the hard-keep logic from a reschedule hint pins F1 and F2 to
    WM_00 at t=480.  The pinned task P also holds WM_00 at [480,540].
    Together they create a direct double-booking via the keep constraint
    (no machine exclusion in the keep path against pinned intervals).
    """
    PINNED_START = 480
    PINNED_END = 540
    PINNED_MACHINE = "WM_00"

    pinned = _wash(
        "P",
        due=9999,
        duration=60,
        qty=1,
        resource_ids=["WM_00"],
        is_pinned=True,
        pinned_machine_id=PINNED_MACHINE,
        pinned_start_time=PINNED_START,
        pinned_end_time=PINNED_END,
    )
    tasks = [
        pinned,
        _wash("F1", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
        _wash("F2", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
    ]
    resources = [_resource("WM_00"), _resource("WM_01")]

    # Hint: previous schedule placed F1 and F2 on WM_00 at t=480
    hint = {
        "_washing_groups": {
            ("red", "cotton"): [
                {
                    "task_id": "F1",
                    "machine_id": "WM_00",
                    "start_time": 480,
                    "end_time": 540,
                    "original_order_id": "ORD_F1",
                },
                {
                    "task_id": "F2",
                    "machine_id": "WM_00",
                    "start_time": 480,
                    "end_time": 540,
                    "original_order_id": "ORD_F2",
                },
            ]
        },
        "stability_weight_time_per_min": 500,
        "stability_weight_machine_swap": 50_000,
        "match_by_order_fallback": True,
    }

    result = solve_washing(
        tasks,
        resources,
        _config(capacity=5),
        p2_end_times={"LNK": 480},
        shift_ends=[],
        reschedule_hint=hint,
    )
    assert result.status == "feasible", f"Solver infeasible: {result.status}"

    _assert_no_machine_double_booking(result, [pinned], "reschedule_hint")


# ---------------------------------------------------------------------------
# GUARD 1 (the critical per-task trap): a multi-task committed batch must stay
# feasible.  Four pinned tasks washed TOGETHER share one (machine, start, end)
# window.  The fix injects ONE interval per WINDOW into the NoOverlap set.  A
# wrong fix that injects one interval PER TASK would make these four batch-mates
# mutually overlap on the same machine → self-conflict → INFEASIBLE.  This test
# stands guard over exactly that mistake.
# ---------------------------------------------------------------------------

def test_committed_batch_mates_stay_feasible():
    """
    GUARD: 4 committed tasks share WM_00 at [480,540] (one washing cycle).  A
    free task is added so K>0 and the NoOverlap path actually runs (an all-pinned
    group early-exits before NoOverlap is built).

    EXPECT: feasible.  Per-WINDOW interval → the four batch-mates collapse to one
    interval → no self-conflict.

    Mutation check: switching the fix to one interval PER TASK turns this RED
    (4 mutually-overlapping intervals on WM_00 → infeasible).
    """
    PINNED_START, PINNED_END, PINNED_MACHINE = 480, 540, "WM_00"

    committed = [
        _wash(
            f"C{i}", due=9999, duration=60, qty=1, resource_ids=["WM_00"],
            is_pinned=True, pinned_machine_id=PINNED_MACHINE,
            pinned_start_time=PINNED_START, pinned_end_time=PINNED_END,
        )
        for i in range(4)
    ]
    tasks = committed + [
        # One free task so the solver builds the slot/NoOverlap model.
        _wash("F1", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              depends_on=["LNK"]),
    ]
    resources = [_resource("WM_00"), _resource("WM_01")]

    result = solve_washing(
        tasks,
        resources,
        _config(capacity=5),
        p2_end_times={"LNK": 480},
        shift_ends=[],
    )
    assert result.status == "feasible", (
        f"Committed batch-mates made the model INFEASIBLE ({result.status}) — "
        f"the fix likely injected one interval PER TASK instead of per WINDOW."
    )
    _assert_no_machine_double_booking(result, committed, "committed batch-mates")


# ---------------------------------------------------------------------------
# GUARD 2: consolidation around a committed batch.  Free tasks that can run only
# on the committed machine must be placed SEQUENTIALLY after the committed cycle
# (not on top of it), while still consolidating onto that single machine.
# ---------------------------------------------------------------------------

def test_free_batch_serialises_after_committed_same_machine():
    """
    GUARD: committed cycle on WM_00 at [480,540].  Three free same-colour tasks
    compatible ONLY with WM_00 (forced consolidation onto the committed machine),
    start_lb=480, capacity=5 → all three fit one batch.

    EXPECT: feasible, no overlap, free batch starts ≥ 540 (after the committed
    cycle), and all three free tasks land on WM_00 (consolidated, not overlapping).
    """
    PINNED_START, PINNED_END, PINNED_MACHINE = 480, 540, "WM_00"

    pinned = _wash(
        "P", due=9999, duration=60, qty=1, resource_ids=["WM_00"],
        is_pinned=True, pinned_machine_id=PINNED_MACHINE,
        pinned_start_time=PINNED_START, pinned_end_time=PINNED_END,
    )
    tasks = [
        pinned,
        _wash("F1", due=9999, duration=60, qty=1, resource_ids=["WM_00"], depends_on=["LNK"]),
        _wash("F2", due=9999, duration=60, qty=1, resource_ids=["WM_00"], depends_on=["LNK"]),
        _wash("F3", due=9999, duration=60, qty=1, resource_ids=["WM_00"], depends_on=["LNK"]),
    ]
    resources = [_resource("WM_00"), _resource("WM_01")]

    result = solve_washing(
        tasks,
        resources,
        _config(capacity=5),
        p2_end_times={"LNK": 480},
        shift_ends=[],
    )
    assert result.status == "feasible", f"Solver infeasible: {result.status}"

    _assert_no_machine_double_booking(result, [pinned], "serialise after committed")

    # Free tasks must be on WM_00 (only option) and start no earlier than 540.
    free_assign = {
        a["task_id"]: a
        for a in result.assignments
        if a["task_id"] in {"F1", "F2", "F3"}
    }
    assert set(free_assign) == {"F1", "F2", "F3"}, f"missing free assignments: {free_assign}"
    for tid, a in free_assign.items():
        assert a["machine_id"] == "WM_00", f"{tid} not consolidated onto WM_00: {a}"
        assert a["start_time"] >= 540, (
            f"{tid} starts at {a['start_time']} < 540 → overlaps committed [480,540]"
        )


# ---------------------------------------------------------------------------
# Cross-group double-booking (production bug: White/Cotton ⨉ Mocha/Cotton on the
# same washing machine).  Phase 3 solves each (color, substance) group in its own
# isolated model; the per-machine NoOverlap there sees only that group's intervals.
# Two groups could therefore place batches on the SAME machine at overlapping
# times.  Fixed by reserving each solved group's machine windows for later groups.
# ---------------------------------------------------------------------------

def test_cross_group_no_machine_double_booking():
    """
    Two groups that both prefer WM_00 (lowest id) at the same early window.

    Group "amocha"/cotton (solved first — sorts before "bwhite") takes WM_00 for
    its batch.  Group "bwhite"/cotton has 3 small same-window tasks that, without
    cross-group reservation, also land on WM_00 → double-booking.

    EXPECT (after fix): the second group is pushed onto WM_01 (or a later WM_00
    window) → no two distinct batch cycles overlap on any machine.

    Before the fix this is RED: both groups independently pick WM_00 at [0,60].
    """
    tasks = [
        # Group A (solved first): one 2-task batch, start_lb=0 → wants WM_00 [0,60]
        _wash("A1", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              color="amocha", substance="cotton"),
        _wash("A2", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              color="amocha", substance="cotton"),
        # Group B (solved second): 3 small orders, start_lb=0 → also wants WM_00 [0,60]
        _wash("B1", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              color="bwhite", substance="cotton"),
        _wash("B2", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              color="bwhite", substance="cotton"),
        _wash("B3", due=9999, duration=60, qty=1, resource_ids=["WM_00", "WM_01"],
              color="bwhite", substance="cotton"),
    ]
    resources = [_resource("WM_00"), _resource("WM_01")]

    result = solve_washing(
        tasks,
        resources,
        _config(capacity=5),
        p2_end_times={},
        shift_ends=[],
    )
    assert result.status == "feasible", f"Solver infeasible: {result.status}"

    # No pinned tasks — all batch cycles come from result.batches.
    _assert_no_machine_double_booking(result, [], "cross-group White ⨉ Mocha")
