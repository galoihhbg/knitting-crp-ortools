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


def _assign(task_id, machine, start, end=None):
    """end = optional end_time. Omitted (legacy assignments / most fixtures) → the
    allocator cannot know when the creel is released, so it keeps the time-blind
    capacity bound; supplied → the creel-release model applies."""
    a = {"task_id": task_id, "machine_id": machine, "start_time": start}
    if end is not None:
        a["end_time"] = end
    return a


# ---------------------------------------------------------------------------
# Real newest payload helper (for abundant + real determinism)
# ---------------------------------------------------------------------------

def _newest_real():
    """Newest solver_input with dyelot_stock + a paired FEASIBLE output. Skip if none.

    Outputs with no assignments (e.g. an infeasible solve logged by the worker) are
    not usable fixtures — the allocator would have nothing to assign and every
    assertion would fail for reasons unrelated to the allocator.
    """
    for inp in sorted(glob.glob("logs/solver_input_*.json"),
                      key=os.path.getmtime, reverse=True):
        out = inp.replace("solver_input", "solver_output")
        if not os.path.exists(out):
            continue
        p = json.load(open(inp))
        if p.get("dyelot_stock"):
            o = json.load(open(out))
            if not o.get("assignments"):
                continue
            return p, o
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
    # net_demand_kg is the consumed-into-product demand (Go classifies on this).
    assert sh["net_demand_kg"] == 150.0
    # Minimal extra capacity to place all 3 orders (one lot each, sharing L1):
    # 150 demand − 100 stock = 50 kg.
    assert sh["single_lot_deficit_kg"] == 50.0
    # The two placeable orders are still assigned (≤100 kg), nothing crashed.
    assert len(res["order_dyelot_assignment"]) == 2


def test_single_lot_deficit_fragmentation():
    """Total stock ≥ demand but no single lot can host an order → capacity_shortage
    with a SMALL deficit (the honest 'add this much to clear'), not the whole
    stranded order's demand."""
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 5.0}),
        _knit_task("u2", "B", {VI: 5.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    # Two lots 6 + 4 = 10 kg total == 10 kg demand, but {5,5} cannot be packed
    # one-lot-each: one order fits the 6-lot, the other (5) exceeds the 4-lot.
    stock = [
        {"vi": VI, "dyelot": "L6", "remaining_kg": 6.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "L4", "remaining_kg": 4.0, "packing_size": 1.0},
    ]
    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert len(res["dyelot_unassigned"]) == 1
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == VI)
    assert sh["stock_kg"] == 10.0
    # Cheapest single-lot top-up = +1 kg (grow L4 4→5 to host the stranded order).
    assert sh["single_lot_deficit_kg"] == 1.0
    # topups are ALTERNATIVES (replenish ONE lot, never a spread): either grow L4
    # by 1 kg, OR grow L6 by 4 kg (both 5-kg orders pile on L6's shared creel).
    tu = {t["dyelot"]: t["add_kg"] for t in sh["topups"]}
    assert tu == {"L4": 1.0, "L6": 4.0}
    # OR import a fresh 5 kg dyelot to host one stranded order.
    assert sh["new_lot_kg"] == 5.0


def test_no_stock_vi_reported_as_shortage():
    """VI consumed as main but with ZERO dyelot_stock → shortage, not crash."""
    tasks = [_knit_task("u1", "A", {"VNOSTOCK": 10.0})]
    assigns = [_assign("u1", "M1", 0)]
    stock = [{"vi": "OTHER", "dyelot": "L1", "remaining_kg": 100.0, "packing_size": 1.0}]
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    assert res["order_dyelot_assignment"] == []
    assert {"order": "A", "vi": "VNOSTOCK", "reason": "no_dyelot_stock"} in res["dyelot_unassigned"]
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == "VNOSTOCK")
    assert sh["stock_kg"] == 0.0
    # Zero stock + no vi_packing → default 1 kg roll; gross of one 10 kg order on a
    # single machine with no creel slots = 10 kg (whole 1 kg rolls, no creel-up).
    assert sh["single_lot_deficit_kg"] == 10.0
    assert sh["topups"] == []
    assert sh["new_lot_kg"] == 10.0


def test_no_stock_new_lot_is_gross_when_packing_known():
    """Zero-stock vi WITH a known default packing → fresh-lot size is the GROSS
    (whole-roll + per-machine creel-up), not the net floor. One order knit on 2
    machines IN PARALLEL, each mounting slots=2 cones at packing 10: each machine
    pulls its OWN max(ceil(net_m/10), 2)·10 = 20 kg (parallel machines can't share a
    physical cone), so gross = 20 + 20 = 40 kg, NOT the pooled 20 kg."""
    tasks = [
        _knit_task("u1", "A", {"V": 7.0}, slots={"V": 2}),
        _knit_task("u2", "A", {"V": 6.0}, slots={"V": 2}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M2", 0)]
    stock = [{"vi": "OTHER", "dyelot": "L1", "remaining_kg": 100.0, "packing_size": 1.0}]
    res = allocate_dyelots(tasks, assigns, stock, CFG, vi_packing={"V": 10.0})
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == "V")
    assert sh["net_demand_kg"] == 13.0
    # Per-machine: M1 max(ceil(7/10),2)·10 + M2 max(ceil(6/10),2)·10 = 20 + 20 = 40 kg.
    assert sh["new_lot_kg"] == 40.0
    assert sh["demand_kg"] == 40.0
    assert sh["topups"] == []


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


def test_buffer_lot_preferred_over_smaller_raw():
    """A lot carrying Buffer-bin yarn (tier 4) wins over an equal-lot-count, equal-
    flush smaller raw lot that drain-small (tier 5) would otherwise pick — so staged
    buffer is drained before a fresh main-warehouse pull. Mirrors CP_..040829709:
    orders fit either a small raw lot or a larger buffer lot; must choose buffer."""
    VI = "V"
    tasks = [_knit_task("u1", "A", {VI: 10.0})]
    assigns = [_assign("u1", "M1", 0)]
    # SMALL (12, all raw) would win on drain-small; BUFFER (500, half staged in the
    # Buffer bin) must win because tier 4 prefers draining buffer.
    stock = [
        {"vi": VI, "dyelot": "SMALL", "remaining_kg": 12.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "BUFFER", "remaining_kg": 500.0, "packing_size": 1.0,
         "buffer_kg": 250.0},
    ]
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    assert res["order_dyelot_assignment"] == [{"order": "A", "vi": VI, "dyelot": "BUFFER"}]


def test_buffer_never_opens_extra_lot():
    """Buffer preference sits BELOW min-lots: it must not fragment. Two orders that
    both fit one raw lot stay consolidated even though splitting one onto a buffer
    lot would drain buffer — opening a second lot costs a full tier-3 unit."""
    VI = "V"
    tasks = [_knit_task("u1", "A", {VI: 10.0}), _knit_task("u2", "B", {VI: 10.0})]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [
        {"vi": VI, "dyelot": "RAW", "remaining_kg": 100.0, "packing_size": 1.0},
        {"vi": VI, "dyelot": "BUF", "remaining_kg": 100.0, "packing_size": 1.0,
         "buffer_kg": 100.0},
    ]
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    lots = {a["dyelot"] for a in res["order_dyelot_assignment"]}
    assert len(lots) == 1, f"expected one lot, got {lots}"


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
# 8. Orders sharing a machine SHARE the creel (reuse, C). Sequential orders on
#    one machine draw from the same mounted cones, so the rolls cover their
#    SUMMED net, not each rounded up independently. lot=50, pk=10; A net=41,
#    B net=5 on M1 sequentially: Σnet=46 → 5 rolls = 50 ≤ 50 → BOTH fit (A leaves
#    9 kg residual on the creel, B draws 5 of it). The old per-order gross
#    (50+10=60) wrongly blocked this — exactly the phantom over-count.
# ---------------------------------------------------------------------------

def test_machine_shared_creel_packs_both():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 41.0}),
        _knit_task("u2", "B", {VI: 5.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    # Reuse: 5 rolls (50 kg) cover Σnet 46 → both placed, lot exactly full.
    assert len(res["order_dyelot_assignment"]) == 2
    assert res["dyelot_unassigned"] == []


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
    # demand_kg is reuse-aware GROSS: both orders share M1's creel → Σnet 110 →
    # 11 rolls = 110 kg (not 60+60=120 per-order).
    assert sh["demand_kg"] == 110.0 and sh["stock_kg"] == 50.0


# ---------------------------------------------------------------------------
# 12. Reuse packs orders even when net == stock exactly. Sequential orders on one
#     machine share the mounted roll (the first order's residual feeds the next),
#     so there is NO phantom gross overflow. A net 8 + B net 2 on M1, lot 10,
#     pk 10 → 1 roll (10 kg) covers both → both placed, NO shortage. (This is the
#     VI-3039 / W9xTMuuLxR production case: with reuse the demand fits the stock.)
# ---------------------------------------------------------------------------

def test_reuse_packs_when_net_equals_stock():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 8.0}),
        _knit_task("u2", "B", {VI: 2.0}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 10.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert len(res["order_dyelot_assignment"]) == 2
    assert res["dyelot_unassigned"] == []
    assert all(s["vi"] != VI for s in res["dyelot_shortage"])


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
# 12b. Orders sharing a machine share the creel INCLUDING its width (max slots):
#     the cones mounted for the widest order are reused by the rest, charged once.
#     A net=20 slots=5, B net=10 slots=2 on M1 sequentially, lot=50 pk=10 →
#     max(ceil(30/10), 5) = 5 rolls = 50 ≤ 50 → BOTH fit (not 50+20=70 per-order).
# ---------------------------------------------------------------------------

def test_machine_shared_creel_with_slots():
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 20.0}, slots={VI: 5}),
        _knit_task("u2", "B", {VI: 10.0}, slots={VI: 2}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    assert len(res["order_dyelot_assignment"]) == 2
    assert res["dyelot_unassigned"] == []


# ---------------------------------------------------------------------------
# 13. Residual POOLS across machines (Go free-pool) — an order knitting the VI on
#     two machines does NOT pay a creel-up roll on each. A: M1 net=5, M2 net=5 →
#     pooled ceil(10/10)=1 roll = 10 kg ≤ 50, so A is assigned. (The old per-machine
#     model charged 2 creel rolls/machine = 80 kg and wrongly dropped it.)
# ---------------------------------------------------------------------------

def test_parallel_machines_each_charge_own_creel():
    """Một đơn knit trên 2 máy SONG SONG, mỗi máy mount slots=4 cone (packing 10) →
    mỗi máy giữ cone RIÊNG (không chia chung được), nên charge 4·10 = 40 kg/máy = 80
    kg > stock 50 → KHÔNG đủ, đơn bị unassigned + báo shortage. Đây là model đúng:
    số máy song song nhân lên lượng cuộn phải giữ."""
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 5.0}, slots={VI: 4}),
        _knit_task("u2", "A", {VI: 5.0}, slots={VI: 4}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M2", 0)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    res = allocate_dyelots(tasks, assigns, stock, CFG)

    # Per-machine creel: 2 máy × max(ceil(5/10),4)·10 = 2 × 40 = 80 kg > 50 → shortage.
    assert res["order_dyelot_assignment"] == []
    assert res["dyelot_unassigned"] == [{"order": "A", "vi": VI, "reason": "capacity_shortage"}]
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == VI)
    assert sh["demand_kg"] == 80.0
    assert sh["net_demand_kg"] == 10.0


def test_legacy_pooled_creel_flag_assigns():
    """Tắt per-machine (enable_dyelot_per_machine_creel=False) → quay về model pooled
    cũ: gross = ceil((5+5)/10)·10 = 10 kg ≤ 50 → đơn được assign. Giữ coverage nhánh
    legacy để A/B."""
    VI = "V"
    tasks = [
        _knit_task("u1", "A", {VI: 5.0}, slots={VI: 4}),
        _knit_task("u2", "A", {VI: 5.0}, slots={VI: 4}),
    ]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M2", 0)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]

    cfg = {**CFG, "enable_dyelot_per_machine_creel": False}
    res = allocate_dyelots(tasks, assigns, stock, cfg)

    assert len(res["order_dyelot_assignment"]) == 1
    assert res["dyelot_unassigned"] == []


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


# ---------------------------------------------------------------------------
# in_production signal — pin converted orders + co-lot a sharing new order
# ---------------------------------------------------------------------------

def test_in_production_pins_and_co_lots_new_order():
    """A NEW order sharing a machine + vi with an in-production order lands on the
    in-production lot (min-flush), not the smaller lot drain-small would pick."""
    tasks = [_knit_task("t_new", "NEW", {"3039": 4}, slots={"3039": 1})]
    assigns = [_assign("t_new", "SK1", 100)]
    dyelot_stock = [
        {"vi": "3039", "dyelot": "dyelot03", "remaining_kg": 100, "packing_size": 10},
        {"vi": "3039", "dyelot": "dyelot99", "remaining_kg": 30, "packing_size": 10},
    ]
    vi_packing = {"3039": 10}
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelot03",
                "machine_id": "SK1", "start_time": 0, "net_kg": 20, "slots": 1,
                "committed_kg": 20}]

    # Baseline (no signal): drain-small tie-break prefers the SMALLER lot.
    base = allocate_dyelots(tasks, assigns, dyelot_stock, CFG, vi_packing=vi_packing)
    base_new = next(a["dyelot"] for a in base["order_dyelot_assignment"]
                    if a["order"] == "NEW")
    assert base_new == "dyelot99"

    # With the signal: INPROD pinned; NEW co-lotted onto dyelot03 (no flush).
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing=vi_packing, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    assert amap["INPROD"] == "dyelot03"
    assert amap["NEW"] == "dyelot03"


def test_in_production_lot_absent_from_stock_is_still_pinned():
    """A committed lot with no free dyelot_stock row (fully committed) still pins
    via the synthetic lot row sized to committed_kg."""
    tasks = [_knit_task("t_new", "NEW", {"3039": 4}, slots={"3039": 1})]
    assigns = [_assign("t_new", "SK1", 100)]
    dyelot_stock = [{"vi": "3039", "dyelot": "dyelotFREE", "remaining_kg": 50, "packing_size": 10}]
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelotGONE",
                "machine_id": "SK1", "start_time": 0, "net_kg": 10, "slots": 1,
                "committed_kg": 10}]
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing={"3039": 10}, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    assert amap["INPROD"] == "dyelotGONE"


def test_in_production_absent_signal_is_noop():
    """No in_production (or None) → unchanged legacy behaviour."""
    tasks = [_knit_task("t_new", "NEW", {"3039": 4}, slots={"3039": 1})]
    assigns = [_assign("t_new", "SK1", 100)]
    stock = [{"vi": "3039", "dyelot": "dyelotA", "remaining_kg": 50, "packing_size": 10}]
    a = allocate_dyelots(tasks, assigns, stock, CFG, vi_packing={"3039": 10})
    b = allocate_dyelots(tasks, assigns, stock, CFG, vi_packing={"3039": 10}, in_production=None)
    assert a["order_dyelot_assignment"] == b["order_dyelot_assignment"]


def test_in_production_picked_creel_restores_capacity_for_co_lot():
    """Case 2/3: after picking + receipt, the committed lot's FREE stock dropped
    below the cohort need, but the picked creel (committed_kg, no longer in
    dyelot_stock) added back to capacity makes co-lot feasible again — the new
    order stays on the in-production lot instead of pulling a fresh lot."""
    tasks = [_knit_task("t_new", "NEW", {"3039": 4}, slots={"3039": 1})]  # gross 10
    assigns = [_assign("t_new", "SK1", 100)]
    # dyelot03 free is only 5kg after picking; dyelot02 (fresh) has plenty.
    dyelot_stock = [
        {"vi": "3039", "dyelot": "dyelot03", "remaining_kg": 5, "packing_size": 10},
        {"vi": "3039", "dyelot": "dyelot02", "remaining_kg": 500, "packing_size": 10},
    ]
    # 30kg of dyelot03 picked creel sits on SK1 for the in-production order.
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelot03",
                "machine_id": "SK1", "start_time": 0, "net_kg": 10, "slots": 1,
                "committed_kg": 30}]
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing={"3039": 10}, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    assert amap["INPROD"] == "dyelot03"
    assert amap["NEW"] == "dyelot03"   # co-lotted thanks to the re-counted creel


def test_in_production_many_machines_does_not_inflate_gross():
    """An in-production order mounted on MANY machines (tiny remaining net each,
    slots>0) must not blow its gross up per-machine and make the forced pin
    infeasible (which dropped the whole VI → empty allocation). Pooled charge keeps
    it feasible so both it and a sharing new order land on the committed lot."""
    machines = [f"M{i}" for i in range(20)]
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelot03",
                "machine_id": m, "start_time": 0, "net_kg": 0.1, "slots": 2,
                "committed_kg": 15} for m in machines]
    tasks = [_knit_task("t_new", "NEW", {"3039": 4}, slots={"3039": 1})]
    assigns = [_assign("t_new", "M0", 500)]  # shares M0 with INPROD → flush pair
    dyelot_stock = [
        {"vi": "3039", "dyelot": "dyelot03", "remaining_kg": 20, "packing_size": 10},
        {"vi": "3039", "dyelot": "dyelot02", "remaining_kg": 1000, "packing_size": 10},
    ]
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing={"3039": 10}, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    assert amap.get("INPROD") == "dyelot03"   # pin feasible (gross not inflated)
    assert amap.get("NEW") == "dyelot03"       # co-lotted


def test_in_production_shared_machines_creel_reuse_enables_colot():
    """Reported case (CP_1784687188624193131): in-prod + new order run on the SAME
    19 machines, both slots=2. Without creel reuse the new order's per-machine
    creel-up (19×2×10=380kg) overflows dyelot03 and it drifts to dyelot02. With the
    floor dropped on the mounted machines it co-lots onto dyelot03."""
    machines = [f"M{i}" for i in range(19)]
    # In-production WawskpLK96: dyelot03, picked (committed 319 ≈ 16.8/machine), slots 2.
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelot03",
                "machine_id": m, "start_time": 0, "net_kg": 69.0/19, "slots": 2,
                "committed_kg": 319.0/19} for m in machines]
    # New WntA49Mfml: net 65 spread over the same 19 machines, slots 2.
    tasks, assigns = [], []
    for i, m in enumerate(machines):
        tid = f"t_new_{i}"
        tasks.append(_knit_task(tid, "NEW", {"3039": 65.0/19}, slots={"3039": 2}))
        assigns.append(_assign(tid, m, 1000))
    dyelot_stock = [
        {"vi": "3039", "dyelot": "dyelot03", "remaining_kg": 120, "packing_size": 10},
        {"vi": "3039", "dyelot": "dyelot02", "remaining_kg": 1000, "packing_size": 10},
    ]
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing={"3039": 10}, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    assert amap.get("INPROD") == "dyelot03"
    assert amap.get("NEW") == "dyelot03", f"new order should co-lot, got {amap.get('NEW')}"


def test_in_production_not_picked_keeps_floor():
    """If the in-production creel is NOT picked (committed_kg=0 → cones not mounted),
    the floor must NOT drop — the new order still needs its own cones. Here dyelot03
    free (10kg) can't hold the new order's real creel-up, so it does NOT co-lot."""
    machines = [f"M{i}" for i in range(19)]
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelot03",
                "machine_id": m, "start_time": 0, "net_kg": 1.0, "slots": 2,
                "committed_kg": 0.0} for m in machines]   # NOT picked
    tasks, assigns = [], []
    for i, m in enumerate(machines):
        tid = f"t_new_{i}"
        tasks.append(_knit_task(tid, "NEW", {"3039": 3.0}, slots={"3039": 2}))
        assigns.append(_assign(tid, m, 1000))
    dyelot_stock = [
        {"vi": "3039", "dyelot": "dyelot03", "remaining_kg": 120, "packing_size": 10},
        {"vi": "3039", "dyelot": "dyelot02", "remaining_kg": 1000, "packing_size": 10},
    ]
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing={"3039": 10}, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    assert amap.get("NEW") == "dyelot02", f"unpicked → floor kept → not co-lot; got {amap.get('NEW')}"


def test_in_production_pool_colots_all_when_creel_pools(monkeypatch=None):
    """Multiple NEW orders co-lotting an in-production lot on the SAME machines all
    land on it when their combined NET fits (warehouse + mounted creel). The mounted
    creel pools across the shared machines, so the per-machine whole-roll rounding
    that would otherwise spill co-lottable orders to a 2nd lot is not charged.

    Mirrors the real case CP_1784694487626128369: in-prod + 3 new on 19 machines,
    net 69+170+150(+~28)=~417 ≤ dyelot03 supply 120 free + 319 creel = 439."""
    machines = [f"M{i}" for i in range(19)]
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelot03",
                "machine_id": m, "start_time": 0, "net_kg": 69.0/19, "slots": 2,
                "committed_kg": 319.0/19} for m in machines]
    tasks, assigns = [], []
    for name, tot in (("N1", 170.0), ("N2", 150.0), ("N3", 28.0)):
        for i, m in enumerate(machines):
            tid = f"t_{name}_{i}"
            tasks.append(_knit_task(tid, name, {"3039": tot/19}, slots={"3039": 2}))
            assigns.append(_assign(tid, m, 1000))
    dyelot_stock = [
        {"vi": "3039", "dyelot": "dyelot03", "remaining_kg": 120, "packing_size": 10},
        {"vi": "3039", "dyelot": "dyelot02", "remaining_kg": 1000, "packing_size": 10},
    ]
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing={"3039": 10}, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    for o in ("INPROD", "N1", "N2", "N3"):
        assert amap.get(o) == "dyelot03", f"{o} should co-lot dyelot03, got {amap.get(o)}"


def test_in_production_pool_still_respects_capacity():
    """SAFETY: the pooled-net path must NOT over-promise. When the NEW orders' net
    exceeds the in-production lot's real supply (warehouse + mounted creel), the
    solver still caps what lands on it and spills the rest to another lot — it never
    crams net beyond physical yarn (which Go commit would then find short)."""
    machines = [f"M{i}" for i in range(5)]
    # dyelot03: warehouse 20 + mounted creel 30 = 50 kg real supply. slots 2 mounted.
    in_prod = [{"order": "INPROD", "vi": "3039", "dyelot": "dyelot03",
                "machine_id": m, "start_time": 0, "net_kg": 5.0/5, "slots": 2,
                "committed_kg": 30.0/5} for m in machines]
    # Three NEW orders, net 40 each (120 total) — only ~1 can fit on dyelot03's 50.
    tasks, assigns = [], []
    for name in ("N1", "N2", "N3"):
        for i, m in enumerate(machines):
            tid = f"t_{name}_{i}"
            tasks.append(_knit_task(tid, name, {"3039": 40.0/5}, slots={"3039": 2}))
            assigns.append(_assign(tid, m, 1000))
    dyelot_stock = [
        {"vi": "3039", "dyelot": "dyelot03", "remaining_kg": 20, "packing_size": 10},
        {"vi": "3039", "dyelot": "dyelot02", "remaining_kg": 1000, "packing_size": 10},
    ]
    out = allocate_dyelots(tasks, assigns, dyelot_stock, CFG,
                           vi_packing={"3039": 10}, in_production=in_prod)
    amap = {a["order"]: a["dyelot"] for a in out["order_dyelot_assignment"]}
    net = {"INPROD": 5.0, "N1": 40.0, "N2": 40.0, "N3": 40.0}
    on03 = sum(net[o] for o, lot in amap.items() if lot == "dyelot03")
    # Placed net on dyelot03 must not exceed real supply (50) + at most one roll of
    # whole-roll slack (10). Cramming all 4 (125) would blow past physical yarn.
    assert on03 <= 60, f"dyelot03 over-promised: {on03}kg placed on 50kg supply"
    assert amap.get("INPROD") == "dyelot03", "in-production order stays pinned"


# ---------------------------------------------------------------------------
# 15. Remedy models must mirror the MAIN solve's capacity semantics.
#     Bug (CP_1785481968066285380 / AY02-DKG350, 2026-07-31): _topup_one_lot_g and
#     _new_lot_g were built WITHOUT committed_g / mounted / pins, so they saw the
#     lot's cap INCLUDING 33.97 kg of already-picked creel (55.971) instead of the
#     22 kg free warehouse the main solve is bound by → both priced the fix at
#     0.00 kg while 2 orders were genuinely stranded ("Top-up: add 0.00 kg").
# ---------------------------------------------------------------------------

def _picked_creel_shortage_fixture():
    """1 lot: 22 kg warehouse + 34 kg PICKED creel (cap 56). Five NEW orders each on
    their OWN non-mounted machine, slots=5, pk=1 → 5 rolls each = 25 rolls > 22 kg
    warehouse → exactly one order must be dropped (4 × 5 = 20 ≤ 22)."""
    VI = "V"
    tasks, assigns = [], []
    for i, name in enumerate(("A", "B", "C", "D", "E")):
        tid = f"t_{name}"
        tasks.append(_knit_task(tid, name, {VI: 0.2}, slots={VI: 5}))
        assigns.append(_assign(tid, f"M{i}", 100))
    stock = [{"vi": VI, "dyelot": "L", "remaining_kg": 22.0, "packing_size": 1.0}]
    in_prod = [{"order": "INPROD", "vi": VI, "dyelot": "L", "machine_id": "MOUNT",
                "start_time": 0, "net_kg": 0.05, "slots": 5, "committed_kg": 34.0}]
    return VI, tasks, assigns, stock, in_prod


def test_remedy_prices_the_free_warehouse_not_the_picked_creel():
    VI, tasks, assigns, stock, in_prod = _picked_creel_shortage_fixture()

    res = allocate_dyelots(tasks, assigns, stock, CFG, in_production=in_prod)

    assert len(res["dyelot_unassigned"]) == 1
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == VI)
    # The picked creel is reported, but the BINDING capacity is the free warehouse.
    assert sh["stock_kg"] == 56.0
    assert sh["committed_kg"] == 34.0
    assert sh["warehouse_kg"] == 22.0
    # Honest remedy: 25 rolls needed − 22 kg warehouse = 3 kg (3 rolls). NOT 0.
    assert sh["single_lot_deficit_kg"] == 3.0
    assert sh["topups"] == [{"dyelot": "L", "add_kg": 3.0}]
    # Or host the stranded order's machine creel on ONE fresh lot: 5 rolls = 5 kg.
    assert sh["new_lot_kg"] == 5.0


def test_shortage_kind_single_lot_is_never_fragmented():
    """A VI with ONE lot cannot be 'not gathered onto a single dyelot' — there is
    nothing to gather onto. The label must name the real binder (creel/roll gross)."""
    VI, tasks, assigns, stock, in_prod = _picked_creel_shortage_fixture()
    res = allocate_dyelots(tasks, assigns, stock, CFG, in_production=in_prod)
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == VI)
    assert sh["n_lots"] == 1
    # net (1.05 kg) fits the 22 kg warehouse; the GROSS creel-up does not.
    assert sh["net_demand_kg"] < sh["warehouse_kg"] < sh["demand_kg"]
    assert sh["shortage_kind"] == "CREEL_ROLL_SHORT"


def test_shortage_kind_material_short_and_fragmented():
    VI = "V"
    # (a) net demand alone exceeds stock → a real buy, whatever the lot layout.
    tasks = [_knit_task(f"u{i}", o, {VI: 50.0}) for i, o in enumerate("ABC")]
    assigns = [_assign(f"u{i}", "M1", i * 100) for i in range(3)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 100.0, "packing_size": 1.0}]
    sh = next(s for s in allocate_dyelots(tasks, assigns, stock, CFG)["dyelot_shortage"]
              if s["vi"] == VI)
    assert sh["shortage_kind"] == "MATERIAL_SHORT"
    assert sh["warehouse_kg"] == 100.0 and sh["committed_kg"] == 0.0

    # (b) 6+4 = 10 kg total covers the 10 kg demand, but {5,5} needs 2 lots and one
    #     order fits neither alone → genuine fragmentation (≥2 lots).
    tasks = [_knit_task("u1", "A", {VI: 5.0}), _knit_task("u2", "B", {VI: 5.0})]
    assigns = [_assign("u1", "M1", 0), _assign("u2", "M1", 100)]
    stock = [{"vi": VI, "dyelot": "L6", "remaining_kg": 6.0, "packing_size": 1.0},
             {"vi": VI, "dyelot": "L4", "remaining_kg": 4.0, "packing_size": 1.0}]
    sh = next(s for s in allocate_dyelots(tasks, assigns, stock, CFG)["dyelot_shortage"]
              if s["vi"] == VI)
    assert sh["shortage_kind"] == "FRAGMENTED" and sh["n_lots"] == 2


def test_no_stock_shortage_kind():
    tasks = [_knit_task("u1", "A", {"VNOSTOCK": 10.0})]
    res = allocate_dyelots(tasks, [_assign("u1", "M1", 0)],
                           [{"vi": "OTHER", "dyelot": "L1", "remaining_kg": 100.0,
                             "packing_size": 1.0}], CFG)
    sh = next(s for s in res["dyelot_shortage"] if s["vi"] == "VNOSTOCK")
    assert sh["shortage_kind"] == "NO_STOCK" and sh["n_lots"] == 0


# ---------------------------------------------------------------------------
# 16. Creel-release model — a machine holds its cones only while it runs the VI;
#     Go returns the leftover to the free pool at end-of-machine-VI (RELEASE_POOL)
#     and a later machine re-consumes it (SUCCESSOR_POOL).  So machines whose knit
#     windows are DISJOINT reuse the same rolls: the bound is the peak concurrent
#     reservation, not the sum over every machine.
# ---------------------------------------------------------------------------

def _two_machine_creel(end1, end2, start2):
    """One order on 2 machines, slots=4 at pk=10 → 40 kg of cones per machine,
    lot = 50 kg. Concurrent → 80 kg needed (shortage); disjoint → 40 kg (fits)."""
    VI = "V"
    tasks = [_knit_task("u1", "A", {VI: 5.0}, slots={VI: 4}),
             _knit_task("u2", "A", {VI: 5.0}, slots={VI: 4})]
    assigns = [_assign("u1", "M1", 0, end1), _assign("u2", "M2", start2, end2)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]
    return VI, tasks, assigns, stock


def test_creel_release_disjoint_machines_reuse_rolls():
    VI, tasks, assigns, stock = _two_machine_creel(end1=100, end2=200, start2=100)
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    # M1 releases its cones at t=100, M2 mounts them at t=100 → peak 40 ≤ 50.
    assert res["dyelot_unassigned"] == []
    assert [a["dyelot"] for a in res["order_dyelot_assignment"]] == ["L1"]
    assert res["dyelot_shortage"] == []


def test_creel_release_overlapping_machines_still_short():
    VI, tasks, assigns, stock = _two_machine_creel(end1=100, end2=150, start2=50)
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    # Windows overlap on [50,100) → both creels mounted at once = 80 kg > 50.
    assert res["dyelot_unassigned"] == [{"order": "A", "vi": VI,
                                        "reason": "capacity_shortage"}]


def test_creel_release_flag_off_is_time_blind():
    VI, tasks, assigns, stock = _two_machine_creel(end1=100, end2=200, start2=100)
    cfg = {**CFG, "enable_dyelot_creel_release": False}
    res = allocate_dyelots(tasks, assigns, stock, cfg)
    # Time-blind: 2 × 40 = 80 kg > 50 even though the windows never overlap.
    assert res["dyelot_unassigned"] == [{"order": "A", "vi": VI,
                                        "reason": "capacity_shortage"}]


def test_creel_release_needs_end_time_on_every_unit():
    """A single unit with no end_time → the release moment is unknown for that VI, so
    the conservative time-blind bound is kept (legacy payloads unchanged)."""
    VI, tasks, assigns, stock = _two_machine_creel(end1=None, end2=200, start2=100)
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    assert res["dyelot_unassigned"] == [{"order": "A", "vi": VI,
                                        "reason": "capacity_shortage"}]


def test_creel_release_never_over_promises_net():
    """SAFETY: releasing cones recycles the UNUSED residual only — consumed yarn never
    comes back. Two disjoint machines each CONSUMING 40 kg net cannot both be served
    by a 50 kg lot, even though the peak reservation (40) would fit."""
    VI = "V"
    tasks = [_knit_task("u1", "A", {VI: 40.0}, slots={VI: 4}),
             _knit_task("u2", "B", {VI: 40.0}, slots={VI: 4})]
    assigns = [_assign("u1", "M1", 0, 100), _assign("u2", "M2", 100, 200)]
    stock = [{"vi": VI, "dyelot": "L1", "remaining_kg": 50.0, "packing_size": 10.0}]
    res = allocate_dyelots(tasks, assigns, stock, CFG)
    assert len(res["dyelot_unassigned"]) == 1     # 80 kg of net on a 50 kg lot


def test_classify_shortage_all_kinds():
    """Direct coverage of the label cascade — REMEDY_UNKNOWN cannot be staged
    end-to-end (a top-up solve only fails to answer on a budget/pin corner), so the
    classifier is pinned down here."""
    from app.engine.dyelot_allocator import _classify_shortage as C

    assert C(0, 10.0, 10.0, 0.0, 10.0) == "NO_STOCK"
    # net alone over the warehouse → a real buy.
    assert C(1, 150.0, 150.0, 100.0, 50.0) == "MATERIAL_SHORT"
    # net fits, gross (whole-roll + creel-up) does not → cones, not lot layout.
    assert C(1, 3.9, 55.0, 22.0, 4.0) == "CREEL_ROLL_SHORT"
    # a single lot is NEVER fragmented, whatever the deficit.
    assert C(1, 5.0, 10.0, 10.0, 1.0) == "UNPROVEN"
    # ≥2 lots, everything fits in total, but not on one lot.
    assert C(2, 10.0, 10.0, 10.0, 1.0) == "FRAGMENTED"
    # deficit 0 with 2 lots → the main solve just didn't prove its optimum.
    assert C(2, 10.0, 10.0, 10.0, 0.0) == "UNPROVEN"
    # no top-up solve answered → the priced remedy is not trustworthy.
    assert C(2, 10.0, 10.0, 10.0, 0.0, topup_possible=False) == "REMEDY_UNKNOWN"
