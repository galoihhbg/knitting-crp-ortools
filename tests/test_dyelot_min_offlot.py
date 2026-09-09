"""Keeping a split LOPSIDED: as few garments as possible on the minority lot.

Every tier of the dyelot objective above the bottom counts how many extra LOTS
an item touches, not how much of the item went there. So once a split is
unavoidable, 19 áo lô A + 1 áo lô B and 11 áo lô A + 9 áo lô B cost exactly the
same, and which one comes back is whatever the search happened to reach first —
on real plans that was ~2:1 every time, with far more lopsided splits available.

The floor pays for that difference in garments it has to keep apart, so the
lowest tier now maximises the pieces on each order's primary lot. It is a
tie-break and nothing more: it must never buy a smaller minority share by
opening another lot, dropping an order, or serving a bigger order first.
"""
from app.engine.dyelot_allocator import allocate_dyelots

CFG_MIX = {"random_seed": 42, "max_deterministic_time": 5.0,
           "dyelot_allow_mixing": True}

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
    rows = [a for a in res["order_dyelot_assignment"] if a["order"] == order]
    return sorted(rows, key=lambda r: -r.get("kg", 0))


def _off_lot_pieces(res, order):
    """Garments left on anything but the order's primary lot."""
    rows = _rows(res, order)
    return sum(r.get("pieces", 0) for r in rows[1:])


# ---------------------------------------------------------------------------
# 1. 20 áo, and neither lot can take them all: the overflow must be the SMALLEST
#    the lots allow, not a comfortable half-and-half. Cones mount in sets of the
#    creel width (2 feeders here), so 19 cones of a lot are 18 usable cones and
#    the smallest possible overflow is 2 áo, not 1.
# ---------------------------------------------------------------------------

def test_overflow_is_as_small_as_the_lots_allow():
    tasks = [_task("T1", "O1", kg=20.0, qty=20, slots=2)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 19.0), _lot("B", 19.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == []
    rows = _rows(res, "O1")
    assert len(rows) == 2, f"expected a 2-lot split, got {rows}"
    assert sum(r["pieces"] for r in rows) == 20
    # The lots are interchangeable, so either may be primary — what is pinned is
    # that only one SET's worth of garments is left behind (2 feeders → 2 áo).
    assert _off_lot_pieces(res, "O1") == 2, rows


# ---------------------------------------------------------------------------
# 2. The tie-break never buys a smaller overflow with an extra lot. Three lots
#    are on offer; two of them can hold the item between them, so the third must
#    stay shut even though spreading further could even the shares out.
# ---------------------------------------------------------------------------

def test_never_opens_another_lot_to_shrink_the_overflow():
    tasks = [_task("T1", "O1", kg=20.0, qty=20, slots=2)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 18.0), _lot("B", 6.0), _lot("C", 6.0)],
                           CFG_MIX)

    rows = _rows(res, "O1")
    assert len(rows) == 2, f"expected exactly 2 lots opened, got {rows}"
    assert sum(r["pieces"] for r in rows) == 20
    assert _off_lot_pieces(res, "O1") == 2, rows


# ---------------------------------------------------------------------------
# 3. Feasibility still outranks it: two orders, and the only layout that places
#    BOTH is one lot each. The tie-break must not strand an order or start
#    splitting them to even shares out.
# ---------------------------------------------------------------------------

def test_serving_every_order_outranks_a_lopsided_split():
    tasks = [_task("T1", "O1", kg=10.0, qty=10, slots=2),
             _task("T2", "O2", kg=10.0, qty=10, slots=2)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1"), _assign("T2", "M2")],
                           [_lot("A", 10.0), _lot("B", 10.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == [], res["dyelot_unassigned"]
    lots = set()
    for order in ("O1", "O2"):
        rows = _rows(res, order)
        assert len(rows) == 1, f"{order} was split with a whole lot free: {rows}"
        assert _off_lot_pieces(res, order) == 0
        lots.add(rows[0]["dyelot"])
    assert lots == {"A", "B"}, f"both orders went to the same lot: {lots}"


# ---------------------------------------------------------------------------
# 4. A lot that fits the whole item is still taken whole: the tie-break must not
#    invent a split to have a primary share to maximise.
# ---------------------------------------------------------------------------

def test_no_gratuitous_split_when_one_lot_fits():
    tasks = [_task("T1", "O1", kg=10.0, qty=10, slots=2)]
    res = allocate_dyelots(tasks, [_assign("T1", "M1")],
                           [_lot("A", 50.0), _lot("B", 50.0)], CFG_MIX)

    rows = _rows(res, "O1")
    assert len(rows) == 1, f"expected one lot, got {rows}"


# ---------------------------------------------------------------------------
# 5. Deterministic: the extra tier must not make the answer depend on the run.
# ---------------------------------------------------------------------------

def test_deterministic():
    tasks = [_task("T1", "O1", kg=20.0, qty=20, slots=2)]
    lots = [_lot("A", 19.0), _lot("B", 19.0)]
    a = allocate_dyelots(tasks, [_assign("T1", "M1")], lots, CFG_MIX)
    b = allocate_dyelots(tasks, [_assign("T1", "M1")], lots, CFG_MIX)
    assert a["order_dyelot_assignment"] == b["order_dyelot_assignment"]
