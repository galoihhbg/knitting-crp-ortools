"""
Washing deadline / co-location tests.

Problem: an order due same-day, sharing a (color, substance) washing group with a
slower order, is "clamped" (kẹp) into the SAME late batch, because:

  1. HARD co-location — phase3_batching.py:337-408 forces every free task in a
     group whose total qty <= capacity into the SAME slot
     (`model.Add(x[t0][k] == x[t_other][k])`), and
  2. batch_start[k] >= max(start_lb of every assigned task)
     (phase3_batching.py:317-318).

  => if one member's start_lb is late (its linking finishes late), the others are
     dragged to that late start.  No objective weight can beat the HARD equality,
     so this is NOT a weight problem — it is a structural co-location problem.

The fix is purely DEADLINE-DRIVEN (no "short-term" category, no threshold, no
weight tuning): a free task may join the hard co-location set ONLY IF, when the
batch is forced to start at `max(group start_lb)`, it still meets its OWN deadline
  deadline = due_at_min − downstream_lead
(downstream_lead = longest-duration path over the reversed final_depends_on DAG;
washing is mid-pipeline, so "washing <= due" is already too late once iron+pack
follow).  A task that would be made late drops out → free to take an early slot.
Tasks that comfortably make the deadline still co-locate (consolidation intact).
Once co-location yields, the existing lateness penalty (≈100k) already dominates
batch_w (≈4k), so tight-deadline tasks take an early slot WITHOUT any added weight.

These tests pin down the contract:
  * a tight-deadline task must escape co-location and take an EARLY batch,
  * loose-deadline orders still consolidate (no-regression),
  * the relax never crashes / goes infeasible (all-soft),
  * priority alone (no floor) decides a genuinely-contended slot, deterministically,
  * a re-schedule keep must not pin a task past its deadline (keep-override),
  * downstream lead governs the deadline used everywhere above.

Each test states its RED reason and the mutation that flips it red after the fix.
"""
from typing import Any, Dict, List, Optional

from app.engine.phases.phase3_batching import solve_washing


# ---------------------------------------------------------------------------
# Builders
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
    resource_ids: Optional[List[str]] = None,
    depends_on: Optional[List[str]] = None,
    color: str = "red",
    substance: str = "cotton",
    order_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "original_order_id": order_id or f"ORD_{task_id}",
        "group_id": order_id or f"ORD_{task_id}",
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
        "compatible_resource_ids": resource_ids or ["WM_00"],
        "WaitOffsets": None,
        "is_pinned": False,
        "pinned_machine_id": None,
        "pinned_start_time": None,
        "pinned_end_time": None,
        "demand": 1,
        "material_demands": {},
    }


def _downstream(
    task_id: str,
    *,
    duration: int,
    depends_on: List[str],
    operation: str = "ironing",
    order_id: str = "",
) -> Dict[str, Any]:
    """A non-washing downstream task (ironing/packing) — only used to feed the
    lead computation; never solved by solve_washing itself."""
    t = _wash(task_id, due=999_999, duration=duration, depends_on=depends_on, order_id=order_id)
    t["operation"] = operation
    return t


def _config(horizon: int = 5000, capacity: int = 3, **overrides) -> Dict[str, Any]:
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


def _by_id(result) -> Dict[str, Dict[str, Any]]:
    return {a["task_id"]: a for a in result.assignments}


# ---------------------------------------------------------------------------
# 1. Core kẹp — short-term escapes the late batch (HARD co-location regime)
# ---------------------------------------------------------------------------

def test_short_term_escapes_late_batch():
    """
    W_short (due=200) and W_long (due=5000) share group ("red","cotton").
    capacity=3, both qty=1 → total 2 <= 3 → HARD co-location forces same slot.
    W_long depends on a linking task ending at 1000 → its start_lb=1000, so the
    shared batch_start >= 1000 → W_short would end at 1060 >> 200.

    EXPECT: W_short finishes on time (<=200) AND sits in a different batch slot
    than W_long (it broke out of the forced co-location).

    RED now: hard co-location pins W_short to slot starting 1000 → end 1060, LATE,
             same batch_slot_id as W_long.
    Mutation: if the co-location gate stops checking the deadline → W_short is
              co-located and late again → red.
    """
    tasks = [
        _wash("W_short", due=200, duration=60),
        _wash("W_long", due=5000, duration=60, depends_on=["LNK_long"]),
    ]
    r = solve_washing(
        tasks, [_resource("WM_00")], _config(capacity=3),
        p2_end_times={"LNK_long": 1000}, shift_ends=[],
    )
    assert r.status == "feasible"
    a = _by_id(r)
    assert a["W_short"]["end_time"] <= 200, (
        f"W_short clamped into late batch: end={a['W_short']['end_time']}"
    )
    assert a["W_short"]["batch_slot_id"] != a["W_long"]["batch_slot_id"], (
        "W_short was force-co-located with the late long order"
    )


# ---------------------------------------------------------------------------
# 2. NO-REGRESSION — long-term orders still consolidate into one batch
# ---------------------------------------------------------------------------

def test_long_term_consolidation_preserved():
    """
    Three long-term orders, same group, capacity=3, all qty=1, all due far out,
    all start_lb=0.  They must consolidate into ONE active batch slot.

    GUARD (green now, must stay green): proves the fix does not weaken the soft
    consolidation that long-term orders rely on.
    Mutation: if the fix excludes ALL tasks (not just urgent) from co-location →
              they spread across slots → > 1 slot → red.
    """
    tasks = [
        _wash("W1", due=5000, duration=60),
        _wash("W2", due=5000, duration=60),
        _wash("W3", due=5000, duration=60),
    ]
    r = solve_washing(
        tasks, [_resource("WM_00")], _config(capacity=3),
        p2_end_times={}, shift_ends=[],
    )
    assert r.status == "feasible"
    slots = {a["batch_slot_id"] for a in r.assignments}
    assert len(slots) == 1, f"long-term orders failed to consolidate, slots={slots}"


# ---------------------------------------------------------------------------
# 3. Mixed group — short on time, longs still consolidate among themselves
# ---------------------------------------------------------------------------

def test_mixed_short_on_time_longs_consolidate():
    """
    Same group: W_short (due=200, lb=0) + W_long1/W_long2 (due=5000, lb=1000 each).
    capacity=3, total qty 3 <= 3 → HARD co-location of all three currently.

    EXPECT: W_short on time in its own early slot; W_long1 and W_long2 share one
    slot (consolidated with each other — long orders may even be pulled earlier
    to fill W_short's batch, which is allowed; only the reverse is forbidden).

    RED now: all three forced into one slot starting 1000 → W_short late.
    Mutation: if longs stop consolidating (different slots) → red.
    """
    tasks = [
        _wash("W_short", due=200, duration=60),
        _wash("W_long1", due=5000, duration=60, depends_on=["LNK1"]),
        _wash("W_long2", due=5000, duration=60, depends_on=["LNK2"]),
    ]
    r = solve_washing(
        tasks, [_resource("WM_00")], _config(capacity=3),
        p2_end_times={"LNK1": 1000, "LNK2": 1000}, shift_ends=[],
    )
    assert r.status == "feasible"
    a = _by_id(r)
    assert a["W_short"]["end_time"] <= 200, f"W_short late: end={a['W_short']['end_time']}"
    assert a["W_short"]["batch_slot_id"] not in (
        a["W_long1"]["batch_slot_id"], a["W_long2"]["batch_slot_id"],
    ), "W_short clamped with longs"
    assert a["W_long1"]["batch_slot_id"] == a["W_long2"]["batch_slot_id"], (
        "long-term orders failed to consolidate with each other"
    )


# ---------------------------------------------------------------------------
# 4. Genuinely-infeasible short order — minimise lateness, never crash
# ---------------------------------------------------------------------------

def test_short_term_unmeetable_does_not_crash():
    """
    A single short order whose duration (100) already exceeds its slack to the
    deadline (due=50) → it CANNOT be on time no matter what.

    EXPECT: solver returns a feasible schedule (no exception, status != infeasible)
    that minimises lateness — W ends as early as physically possible (==100).

    GUARD for approach A's "no hard deadline" promise.
    Mutation: if the fix introduces a HARD washing-completion<=deadline constraint
              for urgent tasks → this group goes INFEASIBLE / raises → red.
    """
    tasks = [_wash("W", due=50, duration=100)]
    r = solve_washing(
        tasks, [_resource("WM_00")], _config(capacity=3),
        p2_end_times={}, shift_ends=[],
    )
    assert r.status == "feasible"
    a = _by_id(r)
    # "No-crash" must include "still present in the schedule" — not dropped/empty.
    assert "W" in a, "unmeetable order was dropped from the schedule entirely"
    assert a["W"]["end_time"] == 100, (
        f"unmeetable order not packed to its earliest finish: end={a['W']['end_time']}"
    )
    assert a["W"]["status"] == "LATE", "unmeetable order should be flagged LATE"


# ---------------------------------------------------------------------------
# 5. Joint contention — two urgent tasks, one early slot → priority decides
# ---------------------------------------------------------------------------

def test_two_short_term_contend_priority_wins_deterministic():
    """
    One machine, capacity=1 → two qty-1 urgent tasks CANNOT share a batch and must
    serialise (one at 0-100, the other at 100-200).  Both due=120, so only the
    first can be on time.

    EXPECT: the higher-priority task (priority=1) is on time (<=120); the
    lower-priority task (priority=5) is relaxed (late); result is feasible and
    BYTE-DETERMINISTIC across two solves.

    GUARD: the EXISTING lateness weights (no added floor) must break the tie
    deterministically.  Proves we did NOT introduce preemptive weight-tuning that
    could flatten/invert priority ordering.
    Mutation: if a flat per-task floor is added → priorities flatten → low-priority
              sometimes wins → non-deterministic / wrong winner → red.
    """
    def build():
        return [
            _wash("W_hi", due=120, duration=100, priority=1),
            _wash("W_lo", due=120, duration=100, priority=5),
        ]

    r1 = solve_washing(build(), [_resource("WM_00")], _config(capacity=1),
                       p2_end_times={}, shift_ends=[])
    r2 = solve_washing(build(), [_resource("WM_00")], _config(capacity=1),
                       p2_end_times={}, shift_ends=[])
    assert r1.status == "feasible"
    a = _by_id(r1)
    assert a["W_hi"]["end_time"] <= 120, f"high-priority urgent late: {a['W_hi']['end_time']}"
    assert a["W_lo"]["end_time"] > 120, "expected the low-priority task to be the relaxed one"

    # Determinism: same start times across runs.
    s1 = {x["task_id"]: x["start_time"] for x in r1.assignments}
    s2 = {x["task_id"]: x["start_time"] for x in r2.assignments}
    assert s1 == s2, f"non-deterministic contention outcome: {s1} != {s2}"


# ---------------------------------------------------------------------------
# 6. KEEP-OVERRIDE — re-schedule must not pin a short order late
# ---------------------------------------------------------------------------

def test_keep_override_releases_late_short_term():
    """
    Re-schedule: the previous plan placed W_short at start=1000 (end=1060) but its
    deadline is 200.  The current keep mechanism hard-pins eligible tasks to their
    previous (start, slot, machine), which would clamp W_short late again.

    EXPECT: the deadline WINS the keep — W_short is released from the late keep and
    finishes on time (<=200).

    RED now: keep two-pass hard-pins W_short.start==1000 → end 1060, LATE.
    Mutation: drop the keep-override (let urgent tasks stay keep-eligible at a late
              position) → W_short pinned late → red.
    """
    tasks = [_wash("W_short", due=200, duration=60, order_id="ORD_s")]
    hint = {
        "_washing_groups": {
            ("red", "cotton"): [
                {
                    "task_id": "W_short",
                    "machine_id": "WM_00",
                    "start_time": 1000,
                    "end_time": 1060,
                    "original_order_id": "ORD_s",
                }
            ]
        },
        "stability_weight_time_per_min": 500,
        "stability_weight_machine_swap": 50_000,
        "match_by_order_fallback": True,
    }
    r = solve_washing(
        tasks, [_resource("WM_00")], _config(capacity=3),
        p2_end_times={}, shift_ends=[], reschedule_hint=hint,
    )
    assert r.status == "feasible"
    a = _by_id(r)
    assert a["W_short"]["end_time"] <= 200, (
        f"short order kept late by re-schedule keep: end={a['W_short']['end_time']}"
    )


# ---------------------------------------------------------------------------
# 7. Downstream lead — washing is mid-pipeline, deadline = due - lead
# ---------------------------------------------------------------------------

def test_downstream_lead_tightens_washing_deadline():
    """
    Washing is mid-pipeline: after it come ironing+packing.  A short order due=300
    with 200 minutes of downstream work (iron 120 + pack 80) must finish WASHING by
    300-200=100, otherwise the ORDER misses the day even though washing "<= due".

    The short order shares group with a long order whose linking ends at 1000, so
    HARD co-location would drag washing to >=1000.

    EXPECT: W_short washing end <= 100 (= due - lead), proving the lead-adjusted
    deadline (not the raw due=300) is what the co-location gate + soft penalty use.

    RED now: solve_washing has no downstream-lead awareness — W_short clamped to
    the late batch (and the new `all_pipeline_tasks` kwarg does not yet exist).
    Mutation: if lead is computed as 0 (ignored) → deadline stays 300, and a batch
              ending at e.g. 150 would wrongly pass → assert end<=100 catches it.
    """
    washing = [
        _wash("W_short", due=300, duration=60, order_id="ORD_s"),
        _wash("W_long", due=5000, duration=60, depends_on=["LNK_long"], order_id="ORD_l"),
    ]
    downstream = [
        _downstream("IRON_s", duration=120, depends_on=["W_short"], operation="ironing", order_id="ORD_s"),
        _downstream("PACK_s", duration=80, depends_on=["IRON_s"], operation="packing", order_id="ORD_s"),
    ]
    r = solve_washing(
        washing, [_resource("WM_00")], _config(capacity=3),
        p2_end_times={"LNK_long": 1000}, shift_ends=[],
        all_pipeline_tasks=washing + downstream,   # GREEN signature
    )
    assert r.status == "feasible"
    a = _by_id(r)
    # Fixture sanity: earliest-feasible washing end (lb=0 + dur=60 = 60) <= 100,
    # so failing end<=100 means lead is wrong, NOT that the deadline is unmeetable.
    assert a["W_short"]["end_time"] <= 100, (
        f"washing ignored downstream lead: end={a['W_short']['end_time']} (need <=100)"
    )


# ---------------------------------------------------------------------------
# 8. Lead helper — longest path, not sum, not single-hop (unit test)
# ---------------------------------------------------------------------------

def test_compute_downstream_lead_longest_path():
    """
    Unit-test the lead helper directly so a WRONG hop-count is caught (the lead=0
    mutation in test 7 only proves lead is *used*, not *correct*).

    Graph from W:  W → IRON(120) → PACK(80)   [chain = 200]
                   W → QC(50)                  [branch = 50]
    EXPECT lead[W] == 200  (longest path; NOT 250=sum, NOT 50=short branch,
    NOT 120=single hop).  Leaf tasks have lead 0.
    """
    from app.engine.phases.phase3_batching import _compute_downstream_lead
    tasks = [
        _wash("W", due=300, duration=60),
        _downstream("IRON", duration=120, depends_on=["W"], operation="ironing"),
        _downstream("PACK", duration=80, depends_on=["IRON"], operation="packing"),
        _downstream("QC", duration=50, depends_on=["W"], operation="qc"),
    ]
    lead = _compute_downstream_lead(tasks)
    assert lead["W"] == 200, f"longest downstream path wrong: lead[W]={lead.get('W')}"
    assert lead["IRON"] == 80, f"lead[IRON]={lead.get('IRON')} (PACK only)"
    assert lead["PACK"] == 0, "leaf task must have lead 0"
    assert lead["QC"] == 0, "leaf task must have lead 0"


# ---------------------------------------------------------------------------
# 9. Allowed direction — a long order ready EARLY may fill the short's batch
# ---------------------------------------------------------------------------

def test_long_term_ready_early_joins_short_batch():
    """
    Asymmetry guard: forbidding "short dragged into late batch" must NOT forbid the
    GOOD direction — a long order that is itself ready early may consolidate into
    the short order's early batch (free extra packing, no deadline harm).

    W_short due=200, W_long due=5000, BOTH start_lb=0.  At batch_start=0 both meet
    their deadlines (60 <= 200 and 60 <= 5000) → they SHOULD share one early slot.

    GUARD: proves the co-location gate does not over-split when consolidation is
    harmless.
    Mutation: if the gate excludes a task merely for having a tight deadline (even
              when it still makes it) → they split → > 1 slot → red.
    """
    tasks = [
        _wash("W_short", due=200, duration=60),
        _wash("W_long", due=5000, duration=60),
    ]
    r = solve_washing(
        tasks, [_resource("WM_00")], _config(capacity=3),
        p2_end_times={}, shift_ends=[],
    )
    assert r.status == "feasible"
    a = _by_id(r)
    assert a["W_short"]["end_time"] <= 200
    assert a["W_short"]["batch_slot_id"] == a["W_long"]["batch_slot_id"], (
        "harmless consolidation was over-split — long order excluded from short's batch"
    )


# ---------------------------------------------------------------------------
# 10. Corner case — the max-start_lb member is itself dropped; rest re-qualify
# ---------------------------------------------------------------------------

def test_dropped_dragger_lets_remaining_consolidate():
    """
    The "max-start_lb member is itself dropped" corner case (the one flagged at the
    gate).  Group of 3, capacity=3 (qty 1 each):
      D_drag : start_lb=3000 (late linking), due=3000  → late even at its own start
               (3000+60 > 3000): it is the MAX-lb member AND drag-late.
      T1, T2 : start_lb=0, due=3050.  At the group max (3000) they look drag-late
               (3060 > 3050), but once D_drag is dropped the max falls to 0 and they
               comfortably make 3050 → they must end up consolidated on time.

    EXPECT: T1 and T2 share ONE early slot (end<=3050); D_drag is on its own
    (released, genuinely late).

    NOTE on (a) vs (b): the iterative removal (design b) keeps {T1,T2} in the HARD
    co-location set.  Single-pass removal (design a) would drop all three from the
    HARD set — but the SOFT batch_active incentive then still regroups T1,T2 into
    one early slot, so the OUTCOME is the same.  This test therefore guards the
    user-visible contract — "no silent over-split to LATE after the dragger is
    released" — which holds under both; it does NOT by itself prove (b) over (a).
    (b) is chosen as the strictly-safer hard-grouping; see _coloc_eligible.

    RED before fix: hard co-location forced all three together at 3000 → T1/T2
    end 3060, LATE.
    Mutation: drop the deadline check in the co-location gate (revert the fix) →
              all three forced to 3000 → T1/T2 late → red.
    """
    tasks = [
        _wash("D_drag", due=3000, duration=60, depends_on=["LNK_d"]),
        _wash("T1", due=3050, duration=60),
        _wash("T2", due=3050, duration=60),
    ]
    r = solve_washing(
        tasks, [_resource("WM_00")], _config(horizon=8000, capacity=3),
        p2_end_times={"LNK_d": 3000}, shift_ends=[],
    )
    assert r.status == "feasible"
    a = _by_id(r)
    assert a["T1"]["end_time"] <= 3050 and a["T2"]["end_time"] <= 3050, (
        f"T1/T2 dragged late by removed dragger: "
        f"T1={a['T1']['end_time']} T2={a['T2']['end_time']}"
    )
    assert a["T1"]["batch_slot_id"] == a["T2"]["batch_slot_id"], (
        "over-split: T1/T2 failed to re-consolidate after the dragger was dropped"
    )
