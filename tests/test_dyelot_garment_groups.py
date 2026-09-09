"""GĐ3 dyelot relaxation — garment-level lot purity across piece-kinds.

A garment assembles one piece of each kind × mult (front body + back body +
2 sleeves…). Each kind knits as its own PO, possibly on its own machine family,
and the pieces meet again at linking — so every kind of an order must split its
garments across lots IDENTICALLY, or linking cannot pair single-lot garments.

These tests pin:
  * the headline case: 3 kinds on 3 machines split as the SAME 6/4 garments,
    and `pieces` reports GARMENTS (not the raw piece sum across kinds);
  * kinds sharing one machine still align;
  * explicit task.garment_qty resolves an all-kinds-share-a-factor structure
    that the gcd fallback cannot;
  * an unresolvable multi-kind order (a kind with no piece count) collapses to
    ONE lot for the whole order — visible shortage over unsewable panels;
  * alignment is honest about capacity: when tying makes the layout infeasible
    the order is unassigned and the remedies price the aligned gap;
  * determinism.
"""
from app.engine.dyelot_allocator import allocate_dyelots

CFG_MIX = {"random_seed": 42, "max_deterministic_time": 5.0,
           "dyelot_allow_mixing": True}

VI = "vi-1"


def _task(task_id, order, kind, kg, qty, slots=0, vi=VI, garment_qty=0):
    e = {"vi": vi, "kg": kg}
    if slots:
        e["slots"] = slots
    t = {"task_id": task_id, "original_order_id": order, "qty": qty,
         "design_item_id": kind, "operation": "knitting",
         "main_yarn_consumption": [e]}
    if garment_qty:
        t["garment_qty"] = garment_qty
    return t


def _assign(task_id, machine, start=0):
    return {"task_id": task_id, "machine_id": machine, "start_time": start}


def _lot(dyelot, kg, pk=1.0, vi=VI):
    return {"vi": vi, "dyelot": dyelot, "remaining_kg": kg, "packing_size": pk}


def _rows(res, order):
    return [a for a in res["order_dyelot_assignment"] if a["order"] == order]


# ---------------------------------------------------------------------------
# 1. Headline: 10 áo = front + back + 2 sleeves, three kinds on three machines
#    (three POs). Lots force 6 áo lot A + 4 áo lot B — and every kind must
#    follow that same garment split.
# ---------------------------------------------------------------------------

def test_three_kinds_on_three_machines_split_identically():
    tasks = [
        _task("T-front",  "O1", "front",  kg=10.0, qty=10),
        _task("T-back",   "O1", "back",   kg=10.0, qty=10),
        _task("T-sleeve", "O1", "sleeve", kg=10.0, qty=20),   # 2 sleeves / áo
    ]
    assigns = [_assign("T-front", "M1"), _assign("T-back", "M2"),
               _assign("T-sleeve", "M3")]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 18.0), _lot("B", 12.0)],
                           CFG_MIX)

    assert res["dyelot_unassigned"] == []
    assert res["dyelot_shortage"] == []
    rows = _rows(res, "O1")
    assert len(rows) == 2, f"expected a 2-lot split, got {rows}"
    by_lot = {r["dyelot"]: r for r in rows}
    # pieces = whole GARMENTS on the lot (6 áo A + 4 áo B), NOT the raw piece
    # sum across kinds (which would read 24 / 16).
    assert by_lot["A"]["pieces"] == 6 and by_lot["B"]["pieces"] == 4
    assert abs(by_lot["A"]["kg"] - 18.0) < 0.01
    assert abs(by_lot["B"]["kg"] - 12.0) < 0.01


# ---------------------------------------------------------------------------
# 2. Two kinds sharing ONE machine (same yarn, same creel) still align, and
#    the shared machine does not double-charge the creel floor.
# ---------------------------------------------------------------------------

def test_kinds_sharing_a_machine_still_align():
    tasks = [
        _task("T-front",  "O1", "front",  kg=10.0, qty=10, slots=1),
        _task("T-back",   "O1", "back",   kg=10.0, qty=10, slots=1),
        _task("T-sleeve", "O1", "sleeve", kg=10.0, qty=20, slots=1),
    ]
    assigns = [_assign("T-front", "M1"), _assign("T-back", "M1"),
               _assign("T-sleeve", "M2")]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 18.0), _lot("B", 12.0)],
                           CFG_MIX)

    assert res["dyelot_unassigned"] == []
    by_lot = {r["dyelot"]: r for r in _rows(res, "O1")}
    assert by_lot["A"]["pieces"] == 6 and by_lot["B"]["pieces"] == 4


# ---------------------------------------------------------------------------
# 3. Every kind is 2-per-garment: the gcd fallback alone cannot see the true
#    garment count, Go's explicit garment_qty pins it — pieces come back as
#    whole garments and the split lands on even piece counts per kind.
# ---------------------------------------------------------------------------

def test_explicit_garment_qty_resolves_shared_factor():
    tasks = [
        _task("T-l", "O1", "left",  kg=10.0, qty=20, garment_qty=10),
        _task("T-r", "O1", "right", kg=10.0, qty=20, garment_qty=10),
    ]
    assigns = [_assign("T-l", "M1"), _assign("T-r", "M2")]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 12.0), _lot("B", 8.0)],
                           CFG_MIX)

    assert res["dyelot_unassigned"] == []
    by_lot = {r["dyelot"]: r for r in _rows(res, "O1")}
    # 10 real garments, 2 kg each: 6 áo on A (12 kg), 4 áo on B (8 kg).
    assert by_lot["A"]["pieces"] == 6 and by_lot["B"]["pieces"] == 4


# ---------------------------------------------------------------------------
# 4. A kind with no piece count → the garment structure is unresolvable, so
#    the WHOLE order stays on one lot. Here no single lot can hold it: the
#    order is a visible shortage, never a free per-kind split.
# ---------------------------------------------------------------------------

def test_unresolvable_kind_collapses_to_one_lot():
    tasks = [
        _task("T-front", "O1", "front", kg=15.0, qty=10),
        _task("T-back",  "O1", "back",  kg=15.0, qty=0),    # unknown pieces
    ]
    assigns = [_assign("T-front", "M1"), _assign("T-back", "M2")]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 18.0), _lot("B", 12.0)],
                           CFG_MIX)

    assert [u["order"] for u in res["dyelot_unassigned"]] == ["O1"], \
        "an unresolvable multi-kind order must not be split across lots"
    assert _rows(res, "O1") == []
    assert len(res["dyelot_shortage"]) == 1


# ---------------------------------------------------------------------------
# 5. Tying is honest about capacity: whole-garment alignment can be infeasible
#    where a free per-kind split was not — the order goes unassigned and the
#    remedies price the ALIGNED gap (1 kg top-up / a 2 kg fresh lot).
# ---------------------------------------------------------------------------

def test_alignment_infeasible_prices_aligned_gap():
    tasks = [
        _task("T-front", "O1", "front", kg=10.0, qty=10),
        _task("T-back",  "O1", "back",  kg=10.0, qty=10),
    ]
    assigns = [_assign("T-front", "M1"), _assign("T-back", "M2")]
    # 10 garments × 2 kg. Lots 11 + 9: any garment split needs 2n_A ≤ 11 and
    # 2n_B ≤ 9 → at most 5 + 4 = 9 garments — infeasible, though a free
    # per-kind split would happily plate 11 + 9 kg.
    res = allocate_dyelots(tasks, assigns, [_lot("A", 11.0), _lot("B", 9.0)],
                           CFG_MIX)

    assert [u["order"] for u in res["dyelot_unassigned"]] == ["O1"]
    sh = res["dyelot_shortage"][0]
    assert abs(sh["single_lot_deficit_kg"] - 1.0) < 0.01, sh
    assert abs(sh["new_lot_kg"] - 2.0) < 0.01, sh


# ---------------------------------------------------------------------------
# 6. Determinism: same input twice → identical output.
# ---------------------------------------------------------------------------

def test_garment_groups_deterministic():
    tasks = [
        _task("T-front",  "O1", "front",  kg=10.0, qty=10),
        _task("T-back",   "O1", "back",   kg=10.0, qty=10),
        _task("T-sleeve", "O1", "sleeve", kg=10.0, qty=20),
        _task("T2",       "O2", "front",  kg=5.0,  qty=5),
    ]
    assigns = [_assign("T-front", "M1"), _assign("T-back", "M2"),
               _assign("T-sleeve", "M3"), _assign("T2", "M4", start=100)]
    lots = [_lot("A", 20.0), _lot("B", 16.0)]
    r1 = allocate_dyelots(tasks, assigns, lots, dict(CFG_MIX))
    r2 = allocate_dyelots(tasks, assigns, lots, dict(CFG_MIX))
    assert r1 == r2
