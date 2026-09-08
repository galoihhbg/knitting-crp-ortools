"""GĐ4 — the dyelot split is handed to Go PER MACHINE RUN.

The allocator already decides lots per (order, machine, piece-kind) chunk and
prices every extra lot a run touches, so a machine that fits its whole run in
one lot stays on one lot. But the result used to be summed to the order level
("16.8 kg on A, 11.2 kg on B"), and Go re-spread that total chronologically
across the machines — putting the lot boundary in the middle of a run the
model had kept whole (SK16 on job CP_1788834138366583801: 0.4 kg A then 1.0 kg
B on a machine still holding 1.6 kg of A).

Each per-lot row now carries `runs`: the (machine, kind) chunks on that lot with
their kg, raw pieces and task ids. These tests pin:
  * runs partition the row: Σ kg / pieces over runs == the row's kg / pieces;
  * every task appears in exactly one (lot, run) — no task is split or lost;
  * a run the model kept on one lot shows up under one lot only;
  * two kinds sharing a machine are two runs, each naming its own tasks;
  * legacy one-hot rows stay byte-identical (no `runs` key).
"""
from app.engine.dyelot_allocator import allocate_dyelots

CFG_MIX = {"random_seed": 42, "max_deterministic_time": 5.0,
           "dyelot_allow_mixing": True}
CFG_LEGACY = {"random_seed": 42, "max_deterministic_time": 5.0}
VI = "vi-1"


def _task(task_id, order, kg, qty, slots=3, kind="body", vi=VI):
    return {"task_id": task_id, "original_order_id": order, "qty": qty,
            "design_item_id": kind, "operation": "knitting",
            "main_yarn_consumption": [{"vi": vi, "kg": kg, "slots": slots}]}


def _assign(task_id, machine, start=0, end=155):
    return {"task_id": task_id, "machine_id": machine,
            "start_time": start, "end_time": end}


def _lot(dyelot, kg, pk=1.0, vi=VI):
    return {"vi": vi, "dyelot": dyelot, "remaining_kg": kg, "packing_size": pk}


def _rows(res, order):
    return [a for a in res["order_dyelot_assignment"] if a["order"] == order]


def _check_partition(rows, all_tasks, pieces_per_garment=1):
    """runs partition each row; every task is named by its (machine, kind) run.
    A run split across lots is listed under each lot it touches (same task
    ids each time) — that is one run on two lots, not a task in two runs.
    Returns tid → set of (lot, machine, kind)."""
    seen = {}
    for r in rows:
        runs = r.get("runs") or []
        assert runs, f"split row without runs: {r}"
        assert abs(sum(x["kg"] for x in runs) - r["kg"]) < 0.01, (r, runs)
        # Row pieces are GARMENTS on a tied order; run pieces are raw pieces.
        assert sum(x["pieces"] for x in runs) == r["pieces"] * pieces_per_garment, (r, runs)
        seen_runs = set()
        for x in runs:
            assert x["tasks"], f"run without tasks: {x}"
            key = (x["machine"], x["kind"])
            assert key not in seen_runs, f"run {key} listed twice under lot {r['dyelot']}"
            seen_runs.add(key)
            for tid in x["tasks"]:
                seen.setdefault(tid, set()).add((r["dyelot"],) + key)
    assert set(seen) == set(all_tasks), (set(seen) ^ set(all_tasks))
    return seen


# Four machines, 3-slot creels, 1 kg cones. M1 and M2 run two tasks each
# (2.8 kg → fits one 3-cone creel), M3 and M4 one task each. All on one lot
# would need 4 × 3 = 12 kg gross; lot A holds 6 kg and lot B 7 kg, so neither
# lot alone fits and the order must split — two whole creels on A, two on B.
def test_runs_partition_the_split_and_no_machine_straddles():
    tasks = [_task("t1", "O1", 1.4, 10), _task("t2", "O1", 1.4, 10),
             _task("t3", "O1", 1.4, 10), _task("t4", "O1", 1.4, 10),
             _task("t5", "O1", 1.4, 10), _task("t6", "O1", 1.4, 10)]
    assigns = [_assign("t1", "M1"), _assign("t2", "M1", 155, 310),
               _assign("t3", "M2"), _assign("t4", "M2", 155, 310),
               _assign("t5", "M3"), _assign("t6", "M4")]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 6.0), _lot("B", 7.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == []
    rows = _rows(res, "O1")
    assert len(rows) == 2, rows
    seen = _check_partition(rows, [t["task_id"] for t in tasks])

    # A machine never appears under two lots: the model kept each run whole.
    assert all(len(v) == 1 for v in seen.values()), seen
    lots_of_machine = {}
    for tid, runs in seen.items():
        for lot, m, _k in runs:
            lots_of_machine.setdefault(m, set()).add(lot)
    assert all(len(v) == 1 for v in lots_of_machine.values()), lots_of_machine
    # Both tasks of a two-task machine travel together.
    assert seen["t1"] == seen["t2"] and seen["t3"] == seen["t4"], seen
    # Lot A carries exactly two whole creels' worth (6 kg gross → 2 machines).
    on_a = {m for m, lots in lots_of_machine.items() if lots == {"A"}}
    assert len(on_a) == 2, lots_of_machine


def test_two_kinds_on_one_machine_are_two_runs_with_their_own_tasks():
    tasks = [_task("f1", "O1", 1.0, 10, kind="front"),
             _task("b1", "O1", 1.0, 10, kind="back"),
             _task("f2", "O1", 1.0, 10, kind="front"),
             _task("b2", "O1", 1.0, 10, kind="back")]
    # M1 knits both kinds, M2 knits both kinds; each lot fits one creel only.
    assigns = [_assign("f1", "M1"), _assign("b1", "M1", 155, 310),
               _assign("f2", "M2"), _assign("b2", "M2", 155, 310)]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 3.0), _lot("B", 4.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == []
    rows = _rows(res, "O1")
    assert len(rows) == 2, rows
    seen = _check_partition(rows, [t["task_id"] for t in tasks], pieces_per_garment=2)
    for r in rows:
        for x in r["runs"]:
            assert len(x["tasks"]) == 1, x            # one task per (machine, kind) here
            assert x["kind"] in ("front", "back"), x
    # The garment tie holds per machine too: a machine's front and back share a lot.
    lot = lambda tid: {x[0] for x in seen[tid]}
    assert lot("f1") == lot("b1") and lot("f2") == lot("b2") and all(len(lot(t)) == 1 for t in seen), seen


def test_legacy_one_hot_rows_have_no_runs():
    tasks = [_task("t1", "O1", 1.4, 10), _task("t2", "O1", 1.4, 10)]
    assigns = [_assign("t1", "M1"), _assign("t2", "M2")]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 25.0)], CFG_LEGACY)
    rows = _rows(res, "O1")
    assert rows == [{"order": "O1", "vi": VI, "dyelot": "A"}], rows


def test_single_lot_split_mode_row_still_names_its_runs():
    # Enough A for everything → one row, but Go still gets the per-machine map
    # (it only budgets split pairs, so this is informational).
    tasks = [_task("t1", "O1", 1.4, 10), _task("t2", "O1", 1.4, 10)]
    assigns = [_assign("t1", "M1"), _assign("t2", "M2")]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 25.0)], CFG_MIX)
    rows = _rows(res, "O1")
    assert len(rows) == 1 and rows[0]["dyelot"] == "A"
    if "runs" in rows[0]:
        assert {tuple(x["tasks"]) for x in rows[0]["runs"]} == {("t1",), ("t2",)}


# Cones mount in SETS of the creel width. 25 one-kg cones of lot A on a
# 3-feeder machine are 24 usable cones (8 sets): the 25th cannot feed three
# feeders alone. So 200 garments at 0.14001 kg split 171 on A (24 kg buys
# 24000 / 140.01 = 171 garments) and 29 on B (2 sets = 6 cones, the 6 in stock),
# and nobody is asked to knit off a lone cone.
def test_cones_are_charged_in_creel_width_sets():
    tasks = [_task(f"t{i}", "O1", 1.4001, 10) for i in range(1, 21)]
    assigns = [_assign(f"t{i}", "M1", (i - 1) * 155, i * 155) for i in range(1, 21)]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 25.0), _lot("B", 6.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == [], res["dyelot_unassigned"]
    rows = {r["dyelot"]: r for r in _rows(res, "O1")}
    assert set(rows) == {"A", "B"}, rows
    assert rows["A"]["pieces"] == 171 and rows["B"]["pieces"] == 29, rows
    # One run, two lots: the same 20 task ids are listed under A and under B.
    seen = _check_partition(list(rows.values()), [t["task_id"] for t in tasks])
    assert all(len(v) == 2 for v in seen.values()), seen


def test_a_lone_cone_cannot_host_a_run():
    # 2 cones of A on a 3-feeder machine: A is unusable, everything goes to B.
    tasks = [_task("t1", "O1", 1.4, 10), _task("t2", "O1", 1.4, 10)]
    assigns = [_assign("t1", "M1"), _assign("t2", "M1", 155, 310)]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 2.0), _lot("B", 25.0)], CFG_MIX)
    rows = _rows(res, "O1")
    assert [r["dyelot"] for r in rows] == ["B"], rows


# Go budgets its walk from the reported kg and charges each task pieces × kg per
# garment against it, so a figure rounded to NEAREST gram sends the walk short:
# 30 garments of 0.14001 kg demand 4.2003 and were handed a budget of 4.2. The
# walker then spent it, read the last 0.3 g as a shortage, and its reconciliation
# had the floor carry three near-empty cones of the OTHER lot across the shop to
# weave that (SK16 task _9, CP_1788851414200481354). Reporting UP by a gram is
# free; reporting down is not.
def test_reported_kg_never_falls_under_the_pieces_it_covers():
    tasks = [_task(f"t{i}", "O1", 1.4001, 10) for i in range(1, 21)]
    assigns = [_assign(f"t{i}", "M1", (i - 1) * 155, i * 155) for i in range(1, 21)]
    res = allocate_dyelots(tasks, assigns, [_lot("A", 25.0), _lot("B", 6.0)], CFG_MIX)

    kg_per_garment = 1.4001 / 10
    rows = {r["dyelot"]: r for r in _rows(res, "O1")}
    assert set(rows) == {"A", "B"}, rows
    for lot, r in rows.items():
        assert r["kg"] >= r["pieces"] * kg_per_garment - 1e-9, (lot, r)
        # …and not by more than the gram the rounding is allowed to add.
        assert r["kg"] <= r["pieces"] * kg_per_garment + 0.001 + 1e-9, (lot, r)
        for run in r["runs"]:
            assert run["kg"] >= run["pieces"] * kg_per_garment - 1e-9, (lot, run)


# Grams are charged to a lot ONCE, not once per garment. CP-SAT works in whole
# grams, and the per-garment figure used to be ceiled there: 0.14001 kg became
# 141 g, so a 27 kg lot "bought" 27000/141 = 191 garments instead of 192. The
# plan then handed the last garment of the run to the next lot while 0.398 kg of
# the first was standing on the machine — enough for two more (SK16 task _9,
# CP_1788854762359001264) — and every plan carried 0.7% of phantom demand.
def test_a_lot_is_charged_the_grams_it_uses_not_a_gram_per_garment():
    tasks = [_task(f"t{i}", "O1", 1.4001, 10) for i in range(1, 21)]
    assigns = [_assign(f"t{i}", "M1", (i - 1) * 155, i * 155) for i in range(1, 21)]
    # A: 27 cones = 9 whole sets, all usable → 27000 / 140.01 = 192 garments.
    res = allocate_dyelots(tasks, assigns, [_lot("A", 27.0), _lot("B", 12.0)], CFG_MIX)

    assert res["dyelot_unassigned"] == [], res["dyelot_unassigned"]
    rows = {r["dyelot"]: r for r in _rows(res, "O1")}
    assert rows["A"]["pieces"] == 192 and rows["B"]["pieces"] == 8, rows
    # The kg reported for A stays inside the lot's 27 kg…
    assert rows["A"]["kg"] <= 27.0, rows["A"]
    # …and covers its garments (the walker charges pieces × 0.14001).
    assert rows["A"]["kg"] >= 192 * 0.14001 - 1e-9, rows["A"]
