"""PHASE P — dyelot allocation post-pass (app/engine/dyelot_allocator.py).

Verifies the CP-SAT allocator: 1 dyelot per (order, VI), approximate capacity,
flush-to-split over-grouped cohorts, lexicographic objective (feasibility ▸
min-flush ▸ small-lots-first), is_main filtering, shortage reporting without
crashing, and 3-leg determinism (1 worker + seed + stable id sort).

Synthetic fixtures give full control over the flush geometry (the real payload's
main-yarn cohorts happen to be smaller than a single order's cross-machine demand,
so capacity-forced flush cannot be reproduced on it); the real newest solver_input
is used for the abundant + determinism-on-real-data checks.
"""
import glob
import json
import os

import pytest

from app.engine.dyelot_allocator import allocate_dyelots

CFG = {"random_seed": 42, "max_deterministic_time": 5.0}


# ---------------------------------------------------------------------------
# Synthetic builders — call allocate_dyelots directly (no scheduler needed)
# ---------------------------------------------------------------------------

def _knit_task(task_id, order, vi_kg, is_main=None, slots=None):
    """vi_kg = {vi: kg}; is_main = optional {vi: bool}; slots = optional {vi: int}
    creel-position count (Go MinSlots) for the creel-up gross charge."""
    myc = []
    for vi, kg in vi_kg.items():
        e = {"vi": vi, "kg": kg}
        if is_main is not None and vi in is_main:
            e["is_main"] = is_main[vi]
        if slots is not None and vi in slots:
            e["slots"] = slots[vi]
        myc.append(e)
    return {"task_id": task_id, "original_order_id": order,
            "operation": "knitting", "main_yarn_consumption": myc}


def _assign(task_id, machine, start):
    return {"task_id": task_id, "machine_id": machine, "start_time": start}


# ---------------------------------------------------------------------------
# Real newest payload helper (for abundant + real determinism)
# ---------------------------------------------------------------------------

def _newest_real():
    """Newest solver_input with dyelot_stock + a paired output. Skip if none."""
    for inp in sorted(glob.glob("logs/solver_input_*.json"),
                      key=os.path.getmtime, reverse=True):
        out = inp.replace("solver_input", "solver_output")
        if not os.path.exists(out):
            continue
        p = json.load(open(inp))
        if p.get("dyelot_stock"):
            return p, json.load(open(out))
    pytest.skip("no real payload with dyelot_stock + paired output on disk")


def _main_vis(payload):
    """VIs consumed as MAIN (is_main True, default True) by knitting tasks."""
    vis = set()
    for t in payload["tasks"]:
        if str(t.get("operation", "")).lower() != "knitting":
            continue
        for c in t.get("main_yarn_consumption") or []:
            if c.get("is_main", True):
                vis.add(c["vi"])
    return vis


# ---------------------------------------------------------------------------
# 1. Abundant stock (real payload) — all assigned, no flush, deterministic
# ---------------------------------------------------------------------------

def test_abundant_all_assigned_no_flush():
    payload, output = _newest_real()
    main_vis = _main_vis(payload)
    if not main_vis:
        pytest.skip("payload has no main yarns")
    # Override stock: one huge lot per main VI → every order fits the same lot,
    # so the never-flush greedy is already feasible (matches greedy: 0 cuts).
    stock = [{"vi": vi, "dyelot": f"BIG_{vi}", "remaining_kg": 1e9,
              "packing_size": 1.0} for vi in sorted(main_vis)]

    res = allocate_dyelots(payload["tasks"], output["assignments"], stock, CFG)

    assigned_orders = {(a["order"], a["vi"]) for a in res["order_dyelot_assignment"]}
    # Every (order, main-VI) pair must be assigned; none unassigned; no shortage.
    assert res["dyelot_unassigned"] == []
    assert res["dyelot_shortage"] == []
    assert assigned_orders, "expected assignments for main yarns"
    # All orders share their VI's single huge lot → zero flush.
    assert res["dyelot_flush_points"] == []
    # Each (order, VI) assigned exactly once (1 dyelot per order per VI).
    keys = [(a["order"], a["vi"]) for a in res["order_dyelot_assignment"]]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# 2. Tight stock (CORE VALUE) — solver flushes to split an over-grouped cohort
# ---------------------------------------------------------------------------

def test_tight_stock_flush_splits_cohort():
    VI = "V"
    # 4 orders × 30 kg, all sequenced on ONE machine → cohort = 120 kg.
    # Plus order A also runs a 2nd unit on a 2nd machine (cross-machine
    # consistency: both units of A must end on the same dyelot).
    tasks = [
        _knit_task("u_A1", "A", {VI: 15.0}),
        _knit_task("u_A2", "A", {VI: 15.0}),   # A total = 30, split over 2 machines
        _knit_task("u_B", "B", {VI: 30.0}),
        _knit_task("u_C", "C", {VI: 30.0}),
        _knit_task("u_D", "D", {VI: 30.0}),
    ]
    assigns = [
        _assign("u_A1", "M1", 0),
        _assign("u_B", "M1", 100),
        _assign("u_C", "M1", 200),
        _assign("u_D", "M1", 300),
        _assign("u_A2", "M2", 0),
    ]
    # Two lots of 60 kg: total 120 ≥ demand 120, max lot 60 ≥ max order 30,
    # but largest cohort (120) > 60 → never-flush (all into one lot) is INFEASIBLE.
    stock = [
        {"vi": VI, "dyelot": "L1", "remaining_kg": 60.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "L2", "remaining_kg": 60.0, "packing_size": 1.0},
    ]
    cohort_kg = 120.0
    assert max(l["remaining_kg"] for l in stock) < cohort_kg  # greedy can't fit

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    # All four orders assigned (solver flushed to split the cohort across lots).
    assert res["dyelot_unassigned"] == []
    assert res["dyelot_shortage"] == []
    by_order = {a["order"]: a["dyelot"] for a in res["order_dyelot_assignment"]}
    assert set(by_order) == {"A", "B", "C", "D"}
    # Order consistency: A appears once → its two units (M1, M2) share one dyelot.
    a_entries = [a for a in res["order_dyelot_assignment"] if a["order"] == "A"]
    assert len(a_entries) == 1
    # Capacity respected per lot.
    load = {"L1": 0.0, "L2": 0.0}
    demand = {"A": 30.0, "B": 30.0, "C": 30.0, "D": 30.0}
    for o, d in by_order.items():
        load[d] += demand[o]
    assert all(load[d] <= 60.0 for d in load)
    # Splitting the cohort REQUIRES at least one flush on M1.
    assert len(res["dyelot_flush_points"]) >= 1
    assert all(fp["machine"] == "M1" for fp in res["dyelot_flush_points"])

    # Determinism: byte-identical on a re-run.
    res2 = allocate_dyelots(tasks, assigns, stock, CFG)
    assert json.dumps(res, sort_keys=True) == json.dumps(res2, sort_keys=True)


# ---------------------------------------------------------------------------
# 3. Real shortage (total demand > stock) — report, do not crash
# ---------------------------------------------------------------------------

def test_shortage_reported_no_crash():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 50.0}),
        _knit_task("u2", "B", {VI: 50.0}),
        _knit_task("u3", "C", {VI: 50.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100), _assign("u3", "M1", 200)]
    # Only 100 kg of stock for 150 kg of demand → 1 order cannot be placed.
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 100.0, "packing_size": 1.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert len(res["dyelot_unassigned"]) >= 1
    assert all(u["vi"] == VI for u in res["dyelot_unassigned"])
    assert any(s["vi"] == VI for s in res["dyelot_shortage"])
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == VI)
    assert sh["demand_kg"] == 150.0 and sh["stock_kg"] == 100.0
    # The two placeable orders are still assigned (≤100 kg), nothing crashed.
    assert len(res["order_dyelot_assignment"]) == 2


def test_no_stock_vi_reported_as_shortage():
    """VI consumed as main but with ZERO dyelot_stock → shortage, not crash."""
    tasks = [_knit_task("u1", "A", {"VNOSTOCK": 10.0})]
    assigns = [_assign("u1", "M1", 0)]
    stock = [{"vi": "OTHER", "dyelot": "L1", "remaining_kg": 100.0, "packing_size": 1.0}]
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    assert res["order_dyelot_assignment"] == []
    assert {"order": "A", "vi": "VNOSTOCK", "reason": "no_dyelot_stock"} in res["dyelot_unassigned"]
    assert any(s["vi"] == "VNOSTOCK" and s["stock_kg"] == 0.0 for s in res["dyelot_shortage"])


# ---------------------------------------------------------------------------
# 4. Small-lots-first — consume the nearly-empty lot before the large one
# ---------------------------------------------------------------------------

def test_small_lots_first():
    VI = "V"
    tasks = [_knit_task("u1", "A", {VI: 10.0})]
    assigns = [_assign("u1", "M1", 0)]
    # Order (10 kg) fits either lot; small (12) and large (500). Tier-3 objective
    # (prefer small lots / less fragmentation) must pick the small lot.
    stock = [
        {"vi": VI, "dyelot": "SMALL", "remaining_kg": 12.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "LARGE", "remaining_kg": 500.0, "packing_size": 1.0},
    ]
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    assert res["order_dyelot_assignment"] == [{"order": "A", "vi": VI, "dyelot": "SMALL"}]


# ---------------------------------------------------------------------------
# 5. Determinism — two runs byte-identical (real payload)
# ---------------------------------------------------------------------------

def test_determinism_real_payload():
    payload, output = _newest_real()
    r1 = allocate_dyelots(payload["tasks"], output["assignments"],
                          payload.get("dyelot_stock"), CFG)
    r2 = allocate_dyelots(payload["tasks"], output["assignments"],
                          payload.get("dyelot_stock"), CFG)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# ---------------------------------------------------------------------------
# 6. is_main filtering — secondary yarns are NOT allocated
# ---------------------------------------------------------------------------

def test_is_main_filter_skips_secondary():
    tasks = [_knit_task("u1", "A", {"MAIN": 10.0, "SEC": 5.0},
                        is_main={"MAIN": True, "SEC": False})]
    assigns = [_assign("u1", "M1", 0)]
    stock = [
        {"vi": "MAIN", "dyelot": "ML", "remaining_kg": 100.0, "packing_size": 1.0},
        {"vi": "SEC", "dyelot": "SL", "remaining_kg": 100.0, "packing_size": 1.0},
    ]
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    vis = {a["vi"] for a in res["order_dyelot_assignment"]}
    assert vis == {"MAIN"}  # SEC (is_main=False) excluded from allocation
    assert all(u["vi"] != "SEC" for u in res["dyelot_unassigned"])


# ===========================================================================
# Over-concentration & gross-capacity regression suite (RC1 + RC2)
# ===========================================================================

# ---------------------------------------------------------------------------
# 7. Spread-not-cram (RC1) — must not strand orders on the small lot while the
#    big lot sits idle.  small=50, big=200, packing=10; three 40 kg orders in
#    sequence (total 120 ≤ 250, each fits a single lot) → ALL must be assigned.
# ---------------------------------------------------------------------------

def test_spread_not_cram_no_false_unassigned():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 40.0}),
        _knit_task("u2", "B", {VI: 40.0}),
        _knit_task("u3", "C", {VI: 40.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100),
               _assign("u3", "M1", 200)]
    stock = [
        {"vi": VI, "dyelot": "SMALL", "remaining_kg": 50.0, "packing_size": 10.0},
        {"vi": VI, "dyelot": "BIG", "remaining_kg": 200.0, "packing_size": 10.0},
    ]
    res = allocate_dyelots(tasks, assigns, stock, CFG)

    # The over-concentration bug stranded two orders on the 50 kg lot; the fix
    # must place all three (the 200 kg lot alone holds 120 kg).
    assert res["dyelot_unassigned"] == []
    assert res["dyelot_shortage"] == []
    by_order = {a["order"]: a["dyelot"] for a in res["order_dyelot_assignment"]}
    assert set(by_order) == {"A", "B", "C"}
    # Per-lot GROSS load must respect physical capacity (40 kg = 4 whole rolls).
    load = {"SMALL": 0.0, "BIG": 0.0}
    for o, d in by_order.items():
        load[d] += 40.0
    assert load["SMALL"] <= 50.0 and load["BIG"] <= 200.0


# ---------------------------------------------------------------------------
# 8. Roll-rounding prevents over-subscription (RC2) — net would fit both,
#    gross does not.  lot=50, packing=10; order1 net=41 (→50 gross),
#    order2 net=5 (→10 gross): 50+10=60 > 50, so they CANNOT both share it.
#    Under the old NET capacity (41+5=46 ≤ 50) both were assigned and Go later
#    found the lot physically short.
# ---------------------------------------------------------------------------

def test_roll_rounding_blocks_oversubscription():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 41.0}),
        _knit_task("u2", "B", {VI: 5.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    # Net (46) would pack both; gross (50+10=60) cannot → exactly one placed.
    assert len(res["order_dyelot_assignment"]) == 1
    assert len(res["dyelot_unassigned"]) == 1


# ---------------------------------------------------------------------------
# 9. Genuine shortage preserved — no order fits the single lot after rounding;
#    both correctly unassigned + a shortage row reported (not a regression).
# ---------------------------------------------------------------------------

def test_genuine_shortage_preserved():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 55.0}),
        _knit_task("u2", "B", {VI: 55.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    # Single 50 kg lot; each order net=55 (→60 gross) exceeds it alone → genuine
    # shortage, both unassignable.  This is correct, not the over-concentration bug.
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert res["order_dyelot_assignment"] == []
    assert len(res["dyelot_unassigned"]) == 2
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == VI)
    # demand_kg is GROSS now: each order 55 net → 60 gross (whole 10 kg rolls).
    assert sh["demand_kg"] == 120.0 and sh["stock_kg"] == 50.0


# ---------------------------------------------------------------------------
# 12. Gross-overflow shortage IS reported even when NET demand fits stock
#     exactly (the BATCH_0-665 / VI-3039 production case: net 403 == stock 403
#     but gross 413 > 403, so one order is dropped — must NOT be a silent
#     unassigned with no shortage row).
# ---------------------------------------------------------------------------

def test_gross_overflow_emits_shortage_when_net_fits():
    VI = "V"
    # Two orders, net 8 + 2 == stock 10 exactly; packing 10 → each rounds up to a
    # whole 10 kg roll → gross 10 + 10 = 20 > 10, so exactly one is unplaceable.
    tasks = [
        _knit_task("u1", "A", {VI: 8.0}),
        _knit_task("u2", "B", {VI: 2.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 10.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert len(res["order_dyelot_assignment"]) == 1
    assert len(res["dyelot_unassigned"]) == 1
    # The fix: a shortage row appears although net (10) == stock (10).
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == VI)
    assert sh["demand_kg"] == 20.0 and sh["stock_kg"] == 10.0


# ---------------------------------------------------------------------------
# 13. order_group_id groups batches of one ĐƠN onto a SINGLE dyelot
#     (the BATCH_0-665 + BATCH_0-666 = đơn W9xTMuuLxR case: rolling-wave split
#     one order across two batches; both must share a dyelot to avoid a colour
#     mismatch on the assembled product).
# ---------------------------------------------------------------------------

def test_order_group_id_co_locates_batches_on_one_dyelot():
    VI = "V"
    # Two batch tasks, DIFFERENT original_order_id, SAME order_group_id (đơn DON1),
    # running on two different machines.
    b1 = _knit_task("BATCH_665", "BATCH_665", {VI: 20.0})
    b2 = _knit_task("BATCH_666", "BATCH_666", {VI: 20.0})
    b1["order_group_id"] = "DON1"
    b2["order_group_id"] = "DON1"
    assigns = [_assign("BATCH_665", "M1", 0), _assign("BATCH_666", "M2", 0)]
    stock = [
        {"vi": VI, "dyelot": "L1", "remaining_kg": 60.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "L2", "remaining_kg": 60.0, "packing_size": 1.0},
    ]
    res = allocate_dyelots([b1, b2], assigns, stock, CFG)

    # One ĐƠN → exactly ONE assignment row (both batches share that dyelot).
    assert len(res["order_dyelot_assignment"]) == 1
    a = res["order_dyelot_assignment"][0]
    assert a["order"] == "DON1" and a["vi"] == VI
    assert res["dyelot_unassigned"] == []


def test_no_order_group_id_falls_back_to_per_batch():
    """Legacy payload (no order_group_id) → grouping by original_order_id, unchanged."""
    VI = "V"
    b1 = _knit_task("BATCH_665", "BATCH_665", {VI: 20.0})
    b2 = _knit_task("BATCH_666", "BATCH_666", {VI: 20.0})
    assigns = [_assign("BATCH_665", "M1", 0), _assign("BATCH_666", "M2", 0)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 60.0, "packing_size": 1.0},
             {"vi": VI, "dyelot": "L2", "remaining_kg": 60.0, "packing_size": 1.0}]
    res = allocate_dyelots([b1, b2], assigns, stock, CFG)
    # Two independent orders → two assignment rows.
    orders = {a["order"] for a in res["order_dyelot_assignment"]}
    assert orders == {"BATCH_665", "BATCH_666"}


# ---------------------------------------------------------------------------
# 10. Flush still minimized (tier 2) — when spreading is shortage-free, the
#     chosen 2-lot split over a 4-unit cohort makes exactly ONE cut (the minimum;
#     0 is impossible since 120 kg > any single 60 kg lot).
# ---------------------------------------------------------------------------

def test_flush_minimized_when_spreading():
    VI = "V"
    tasks = [_knit_task(f"u{i}", o, {VI: 30.0})
             for i, o in enumerate(["A", "B", "C", "D"])]
    assigns = [_assign(f"u{i}", "M1", i * 100) for i in range(4)]
    stock = [
        {"vi": VI, "dyelot": "L1", "remaining_kg": 60.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "L2", "remaining_kg": 60.0, "packing_size": 1.0},
    ]
    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert res["dyelot_unassigned"] == []
    # Contiguous {A,B}|{C,D} split → 1 cut; any interleaved split costs more.
    assert len(res["dyelot_flush_points"]) == 1


# ---------------------------------------------------------------------------
# 11. Determinism (synthetic) — same input → byte-identical output over N runs.
# ---------------------------------------------------------------------------

def test_determinism_synthetic_spread():
    VI = "V"
    tasks = [_knit_task(f"u{i}", o, {VI: 30.0})
             for i, o in enumerate(["A", "B", "C", "D"])]
    assigns = [_assign(f"u{i}", "M1", i * 100) for i in range(4)]
    stock = [
        {"vi": VI, "dyelot": "L1", "remaining_kg": 60.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "L2", "remaining_kg": 60.0, "packing_size": 1.0},
    ]
    runs = [allocate_dyelots(tasks, assigns, stock, CFG) for _ in range(4)]
    first = json.dumps(runs[0], sort_keys=True)
    assert all(json.dumps(r, sort_keys=True) == first for r in runs[1:])


# ---------------------------------------------------------------------------
# 12. Creel-up (slots) blocks over-subscription (RC2, creel floor) — net fits,
#     gross with the slots floor does not. lot=50, packing=10; A net=20 slots=5
#     (→ max(2,5)·10 = 50 gross), B net=10 slots=2 (→ max(1,2)·10 = 20 gross):
#     50+20=70 > 50, so they CANNOT share it. Under plain net (20+10=30 ≤ 50)
#     both packed and Go later found the lot physically short — exactly the
#     dyelot-at-net-edge bug.
# ---------------------------------------------------------------------------

def test_creel_up_slots_blocks_oversubscription():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 20.0}, slots={VI: 5}),
        _knit_task("u2", "B", {VI: 10.0}, slots={VI: 2}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert len(res["order_dyelot_assignment"]) == 1
    assert len(res["dyelot_unassigned"]) == 1


# ---------------------------------------------------------------------------
# 13. Creel-up sums per machine run — an order knitting the VI on TWO machines
#     pays the slots floor on EACH. A: M1 net=5 slots=4, M2 net=5 slots=4 →
#     gross = max(1,4)·10 + max(1,4)·10 = 80 > 50 lot → unassigned, even though
#     net (10) and a single-run creel-up (40) would both fit.
# ---------------------------------------------------------------------------

def test_creel_up_sums_across_machines():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 5.0}, slots={VI: 4}),
        _knit_task("u2", "A", {VI: 5.0}, slots={VI: 4}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M2", 0)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert res["order_dyelot_assignment"] == []
    assert len(res["dyelot_unassigned"]) == 1


# ---------------------------------------------------------------------------
# 14. slots absent → no creel-up (back-compat). Same as #13's geometry but with
#     no slots: net 10 ≤ 50 → A is assigned (legacy payloads keep working).
# ---------------------------------------------------------------------------

def test_no_slots_is_backward_compatible():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 5.0}),
        _knit_task("u2", "A", {VI: 5.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M2", 0)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert len(res["order_dyelot_assignment"]) == 1
    assert res["dyelot_unassigned"] == []
