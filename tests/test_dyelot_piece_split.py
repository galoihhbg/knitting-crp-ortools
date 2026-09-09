"""GĐ2 dyelot relaxation — piece-split mode (dyelot_allow_mixing=True).

The business rule moved from "1 item = 1 dyelot" to "1 áo (garment) = 1 dyelot":
an item may split its integer garment count across lots (6 áo lot A + 4 áo lot B),
same-lot stays preferred. These tests pin:

  * an item splits across lots when no single lot fits — whole garments only;
  * a fitting single lot is still preferred (no gratuitous mixing);
  * the flag OFF keeps the legacy one-hot shape (no kg/pieces keys);
  * in-production pins never split;
  * remedies price the ANY-layout gap (mixing exhausted every lot first);
  * fractional kg-per-garment still splits on integral garments;
  * determinism (two runs, same result).
"""
from app.engine.dyelot_allocator import allocate_dyelots

CFG_MIX = {"random_seed": 42, "max_deterministic_time": 5.0,
           "dyelot_allow_mixing": True}
CFG_LEGACY = {"random_seed": 42, "max_deterministic_time": 5.0}

VI = "vi-1"


def _task(task_id, order, kg, qty, slots=0, vi=VI):
    e = {"vi": vi, "kg": kg}
    if slots:
        e["slots"] = slots
    return {"task_id": task_id, "original_order_id": order, "qty": qty,
            "operation": "knitting", "main_yarn_consumption": [e]}


def _assign(task_id, machine, start=0):
    return {"task_id": task_id, "machine_id": machine, "start_time": start}


def _lot(dyelot, kg, pk=1.0, vi=VI):
    return {"vi": vi, "dyelot": dyelot, "remaining_kg": kg, "packing_size": pk}


def _rows(res, order):
    return [a for a in res["order_dyelot_assignment"] if a["order"] == order]


# ---------------------------------------------------------------------------
# 1. The headline case: 10 áo, lot A 6 kg + lot B 4 kg → 6 áo A + 4 áo B.
# ---------------------------------------------------------------------------

def test_item_splits_whole_garments_across_lots():
    tasks = [_task("T1", "O1", kg=10.0, qty=10, slots=2)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 6.0), _lot("B", 4.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == []
    assert res["dyelot_shortage"] == []
    rows = _rows(res, "O1")
    assert len(rows) == 2, f"expected a 2-lot split, got {rows}"
    assert sum(r["pieces"] for r in rows) == 10
    assert all(r["pieces"] >= 1 for r in rows)
    # Primary (largest kg) row first; exact split is forced by the lot sizes.
    assert rows[0]["dyelot"] == "A" and rows[0]["pieces"] == 6
    assert rows[1]["dyelot"] == "B" and rows[1]["pieces"] == 4
    assert abs(sum(r["kg"] for r in rows) - 10.0) < 0.01


# ---------------------------------------------------------------------------
# 2. Same-lot still preferred: a lot that fits the whole item wins outright.
# ---------------------------------------------------------------------------

def test_single_fitting_lot_is_not_split():
    tasks = [_task("T1", "O1", kg=10.0, qty=10, slots=2)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 4.0), _lot("B", 20.0)], CFG_MIX)

    rows = _rows(res, "O1")
    assert len(rows) == 1, f"item mixed gratuitously: {rows}"
    assert rows[0]["dyelot"] == "B"
    # The one-hot probe served everyone, so the result is returned verbatim in
    # the legacy single-row shape (no kg/pieces keys) — the cheap, stable path.
    if "pieces" in rows[0]:
        assert rows[0]["pieces"] == 10


# ---------------------------------------------------------------------------
# 3. Flag OFF → legacy one-hot: same fixture is a FRAGMENTED shortage and the
#    assignment rows carry no kg/pieces keys.
# ---------------------------------------------------------------------------

def test_flag_off_keeps_legacy_shape_and_fragmented():
    tasks = [_task("T1", "O1", kg=10.0, qty=10, slots=2)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 6.0), _lot("B", 4.0)], CFG_LEGACY)

    assert [u["order"] for u in res["dyelot_unassigned"]] == ["O1"]
    assert len(res["dyelot_shortage"]) == 1
    assert res["dyelot_shortage"][0]["shortage_kind"] == "FRAGMENTED"
    for a in res["order_dyelot_assignment"]:
        assert "pieces" not in a and "kg" not in a


# ---------------------------------------------------------------------------
# 4. An in-production pin never splits — the order stays whole on its lot.
# ---------------------------------------------------------------------------

def test_pinned_in_production_order_never_splits():
    tasks = [_task("T1", "O1", kg=6.0, qty=6, slots=1)]
    in_prod = [{"order": "IP1", "vi": VI, "dyelot": "A", "machine_id": "M9",
                "start_time": 0, "net_kg": 3.0, "slots": 2, "committed_kg": 3.0}]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 3.0), _lot("B", 6.0)], CFG_MIX,
                           in_production=in_prod)

    ip_rows = _rows(res, "IP1")
    assert len(ip_rows) == 1 and ip_rows[0]["dyelot"] == "A", \
        f"pinned order must stay whole on its committed lot: {ip_rows}"
    assert res["dyelot_unassigned"] == []


# ---------------------------------------------------------------------------
# 5. Remedies price the ANY-layout gap: mixing already exhausted every lot, so
#    new_lot_kg is the missing remainder, not "one lot big enough for the item".
# ---------------------------------------------------------------------------

def test_remedy_prices_any_lot_gap_not_single_lot():
    tasks = [_task("T1", "O1", kg=10.0, qty=10)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 4.0)], CFG_MIX)

    assert [u["order"] for u in res["dyelot_unassigned"]] == ["O1"]
    sh = res["dyelot_shortage"][0]
    assert sh["shortage_kind"] == "MATERIAL_SHORT"
    # 10 kg gross demand vs 4 kg on lot A → 6 kg missing whichever way it lands.
    assert abs(sh["single_lot_deficit_kg"] - 6.0) < 0.01
    # A fresh lot only needs the REMAINDER (the split covers 4 kg from A) —
    # legacy one-hot would demand a 10 kg fresh lot here.
    assert abs(sh["new_lot_kg"] - 6.0) < 0.01


def test_legacy_remedy_still_prices_single_lot():
    tasks = [_task("T1", "O1", kg=10.0, qty=10)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 4.0)], CFG_LEGACY)
    sh = res["dyelot_shortage"][0]
    assert abs(sh["new_lot_kg"] - 10.0) < 0.01, \
        "legacy mode must still size the fresh lot for the WHOLE item"


# ---------------------------------------------------------------------------
# 6. Fractional kg-per-garment: the split still lands on whole garments.
# ---------------------------------------------------------------------------

def test_fractional_kg_per_piece_splits_on_whole_garments():
    tasks = [_task("T1", "O1", kg=10.0, qty=3)]   # ≈3.333 kg per garment
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 7.0), _lot("B", 7.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == []
    rows = _rows(res, "O1")
    assert len(rows) == 2
    assert sorted(r["pieces"] for r in rows) == [1, 2]
    assert sum(r["pieces"] for r in rows) == 3


# ---------------------------------------------------------------------------
# 7. Unknown qty (0) never fakes a split: the order falls back to one-hot and,
#    when no single lot fits, surfaces as a shortage exactly like legacy.
# ---------------------------------------------------------------------------

def test_zero_qty_falls_back_to_one_hot():
    tasks = [_task("T1", "O1", kg=10.0, qty=0)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 6.0), _lot("B", 4.0)], CFG_MIX)

    assert [u["order"] for u in res["dyelot_unassigned"]] == ["O1"], \
        "an order with unknown garment count must not be split"


# ---------------------------------------------------------------------------
# 8. Determinism: same input twice → identical output.
# ---------------------------------------------------------------------------

def test_piece_split_deterministic():
    tasks = [_task("T1", "O1", kg=10.0, qty=10, slots=2),
             _task("T2", "O2", kg=5.0, qty=5, slots=1)]
    assigns = [_assign("T1", "M1"), _assign("T2", "M2", start=100)]
    lots = [_lot("A", 8.0), _lot("B", 7.0)]
    r1 = allocate_dyelots(tasks, assigns, lots, dict(CFG_MIX))
    r2 = allocate_dyelots(tasks, assigns, lots, dict(CFG_MIX))
    assert r1 == r2
