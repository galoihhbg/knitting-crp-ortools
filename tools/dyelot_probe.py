#!/usr/bin/env python
"""PHASE V — dyelot allocation problem PROBE (READ-ONLY; builds no model).

Measures the real dyelot-allocation problem on a solved knitting schedule so we
can size the upcoming CP-SAT post-pass *before* writing it.  Touches no solver /
builder / CP-SAT code: it only reads a payload, (optionally) runs a cold knitting
solve to obtain the schedule, and prints a markdown report.

Domain model encoded here (the spec the post-pass must satisfy)
---------------------------------------------------------------
* VI            = one main-yarn type.  A knitting task lists its draw as
                  main_yarn_consumption = [{vi, kg}, ...].
* Dyelot        = a dye batch of a VI.  Stock: dyelot_stock =
                  [{vi, dyelot, remaining_kg, packing_size}, ...].
* 1-dyelot rule = each ORDER must use exactly one dyelot per VI (mixing two
                  dyelots of the same VI streaks the colour).
* Creel carry   = on a knitting machine, leftover yarn ("residual") on the creel
                  is inherited by the NEXT scheduled unit on that machine.  So
                  consecutive units on a machine are chained → forced same dyelot
                  per shared VI unless the chain is FLUSHED (residual discarded =
                  waste).  Order is by schedule SEQUENCE on the machine, NOT by
                  absolute time — a long idle gap does not break the carry.
* Cohort        = a set of units pinned to one dyelot (per VI) by un-flushed
                  carry.  The current Go builder is greedy never-flush → each
                  machine's whole chain collapses into ONE cohort.  That is the
                  over-grouping this probe quantifies (the thing to beat).
* Residual ≤ packing_size — the leftover handed to the next unit is bounded by a
                  pack/cone size.  Whether that bound is large vs a lot
                  (remaining_kg) decides exact-flow vs approximate capacity model.

Usage:
    python tools/dyelot_probe.py PAYLOAD.json [--output OUTPUT.json] [--md REPORT.md]

PAYLOAD.json  = the dict Engine(...).solve() receives (config/machines/resources/
                tasks/material_capacities/dyelot_stock).  Tasks must carry
                main_yarn_consumption and the payload dyelot_stock for a full run.
--output      = a solver_output_*.json to PAIR with (task_id→machine/start_time),
                skipping the cold solve.  Omit to run a cold knitting solve.
--md          = write the report to a file as well as stdout.
"""
import sys
import os
import json
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Schedule acquisition
# ---------------------------------------------------------------------------

def _knitting_ids(payload):
    return {
        t["task_id"] for t in payload.get("tasks", [])
        if str(t.get("operation", "")).lower() == "knitting"
    }


def _schedule_from_output(payload, output, knit_ids):
    """Pair task_id → (machine_id, start_time) from an existing solver output."""
    sched = {}
    for a in output.get("assignments", []):
        if a["task_id"] in knit_ids:
            sched[a["task_id"]] = (a.get("machine_id"), a.get("start_time"))
    return sched


def _schedule_cold(payload, knit_ids):
    """Run a cold solve (Engine) and pair knitting task_id → (machine, start)."""
    import logging
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, os.getcwd())
    from app.engine.model import Engine
    result = Engine(payload).solve()
    sched = {}
    for a in result.get("assignments", []):
        if a["task_id"] in knit_ids:
            sched[a["task_id"]] = (a.get("machine_id"), a.get("start_time"))
    return sched, result.get("status")


# ---------------------------------------------------------------------------
# Granularity / blocker check
# ---------------------------------------------------------------------------

def check_granularity(payload, knit_ids):
    """Is the scheduling unit a whole task or a chunk/slice, and does
    main_yarn_consumption sit at that granularity?  Returns a dict of findings.

    Python-side rolling-wave never splits a task (it groups whole orders), so the
    scheduling unit == the SolverTask as received from Go (which MAY already be a
    Go-side slice).  main_yarn_consumption is a per-SolverTask field, so it rides
    at unit granularity *iff* Go populated it on each unit (incl. each slice).
    """
    tasks = [t for t in payload.get("tasks", []) if t["task_id"] in knit_ids]
    n = len(tasks)
    n_slice = sum(1 for t in tasks if t.get("is_slice"))
    n_parent = sum(1 for t in tasks if t.get("parent_task_id"))
    n_with_myc = sum(1 for t in tasks if t.get("main_yarn_consumption"))
    n_slice_with_myc = sum(
        1 for t in tasks if t.get("is_slice") and t.get("main_yarn_consumption")
    )
    blocker = False
    notes = []
    if n_with_myc == 0:
        blocker = True
        notes.append(
            "NO knitting unit carries main_yarn_consumption — field absent from "
            "this payload (Go has not emitted it yet)."
        )
    if n_slice > 0 and n_slice_with_myc < n_slice:
        blocker = True
        notes.append(
            f"{n_slice - n_slice_with_myc}/{n_slice} Go-side SLICES lack "
            "main_yarn_consumption → per-handoff residual kg would be computed at "
            "the wrong granularity. CHẶN — Go must emit per-slice consumption."
        )
    return {
        "n_units": n, "n_slice": n_slice, "n_parent": n_parent,
        "n_with_myc": n_with_myc, "n_slice_with_myc": n_slice_with_myc,
        "blocker": blocker, "notes": notes,
    }


# ---------------------------------------------------------------------------
# Per-VI aggregation
# ---------------------------------------------------------------------------

def _unit_consumption(task):
    """task → {vi: kg} for the MAIN yarn only (aggregating duplicate vi).

    main_yarn_consumption entries may carry `is_main` (true = main yarn, false =
    secondary yarn that does NOT get a dyelot allocated).  We count an entry iff
    `is_main` is True, defaulting to True when the flag is absent (legacy payloads
    predate the main/secondary split — every entry was a main yarn then).
    """
    out = defaultdict(float)
    for c in task.get("main_yarn_consumption") or []:
        if c.get("is_main", True):
            out[c["vi"]] += float(c.get("kg", 0))
    return dict(out)


def _order_of(task):
    return task.get("original_order_id") or task.get("group_id") or task["task_id"]


def build_machine_cohorts(payload, knit_ids, sched):
    """Per machine: units in SCHEDULE-SEQUENCE order.  Under never-flush greedy the
    whole chain is ONE cohort.  Returns {machine_id: [task, ...] in seq order}."""
    tasks = {t["task_id"]: t for t in payload.get("tasks", []) if t["task_id"] in knit_ids}
    by_machine = defaultdict(list)
    for tid, (m, start) in sched.items():
        if tid in tasks and m is not None:
            by_machine[m].append((start if start is not None else 0, tid))
    cohorts = {}
    for m, rows in by_machine.items():
        rows.sort(key=lambda r: (r[0], r[1]))  # sequence by start, tie-break id
        cohorts[m] = [tasks[tid] for _, tid in rows]
    return cohorts


def per_vi_stats(payload, knit_ids, cohorts):
    """Aggregate demand/stock per VI plus counts.  Returns dict keyed by vi."""
    tasks = [t for t in payload.get("tasks", []) if t["task_id"] in knit_ids]

    # Stock per VI
    lots_by_vi = defaultdict(list)
    for d in payload.get("dyelot_stock") or []:
        lots_by_vi[d["vi"]].append(d)

    vi = defaultdict(lambda: {
        "demand_kg": 0.0, "orders": set(), "units": 0, "machines": set(),
        "lots": [], "total_stock_kg": 0.0, "packing_sizes": [],
    })
    # Demand
    machine_of = {}
    for m, chain in cohorts.items():
        for t in chain:
            machine_of[t["task_id"]] = m
    for t in tasks:
        cons = _unit_consumption(t)
        for v, kg in cons.items():
            rec = vi[v]
            rec["demand_kg"] += kg
            rec["orders"].add(_order_of(t))
            rec["units"] += 1
            if t["task_id"] in machine_of:
                rec["machines"].add(machine_of[t["task_id"]])
    # Stock
    for v, lots in lots_by_vi.items():
        rec = vi[v]
        rec["lots"] = sorted(lots, key=lambda d: -float(d.get("remaining_kg", 0)))
        rec["total_stock_kg"] = sum(float(d.get("remaining_kg", 0)) for d in lots)
        rec["packing_sizes"] = [float(d.get("packing_size", 0)) for d in lots]
    return vi


# ---------------------------------------------------------------------------
# Baseline over-group + feasibility + residual pivotalness
# ---------------------------------------------------------------------------

def over_group(cohorts, vi_stats):
    """Per (machine-cohort, VI): Σ kg forced onto one dyelot, and whether it fits
    that VI's largest lot.  Returns the worst (largest) binding."""
    worst = None  # (kg, n_orders, machine, vi, max_lot, fits)
    rows = []
    for m, chain in cohorts.items():
        per_vi_kg = defaultdict(float)
        per_vi_orders = defaultdict(set)
        for t in chain:
            for v, kg in _unit_consumption(t).items():
                per_vi_kg[v] += kg
                per_vi_orders[v].add(_order_of(t))
        for v, kg in per_vi_kg.items():
            max_lot = max((float(l.get("remaining_kg", 0)) for l in vi_stats[v]["lots"]),
                          default=0.0)
            fits = kg <= max_lot
            row = (kg, len(per_vi_orders[v]), m, v, max_lot, fits)
            rows.append(row)
            if worst is None or kg > worst[0]:
                worst = row
    rows.sort(key=lambda r: -r[0])
    return worst, rows


def feasibility(payload, knit_ids, vi_stats):
    """Orders whose per-VI demand alone exceeds the largest lot (unassignable even
    un-grouped) + VIs whose total demand exceeds total stock (real shortage)."""
    tasks = [t for t in payload.get("tasks", []) if t["task_id"] in knit_ids]
    # VIs that are consumed (as main) but have ZERO dyelot lots — reported once,
    # not as per-order spam.  These are either secondary yarns mis-flagged as main
    # (fix: is_main=false) or a genuine stock gap.
    no_stock_vis = sorted(
        v for v, rec in vi_stats.items()
        if rec["demand_kg"] > 0 and not rec["lots"]
    )
    order_vi_kg = defaultdict(float)  # (order, vi) -> kg
    for t in tasks:
        o = _order_of(t)
        for v, kg in _unit_consumption(t).items():
            order_vi_kg[(o, v)] += kg
    unassignable = []
    for (o, v), kg in order_vi_kg.items():
        if v in no_stock_vis:
            continue  # covered by no_stock_vis, don't double-report
        max_lot = max((float(l.get("remaining_kg", 0)) for l in vi_stats[v]["lots"]),
                      default=0.0)
        if kg > max_lot:
            unassignable.append((o, v, kg, max_lot))
    shortages = [
        (v, rec["demand_kg"], rec["total_stock_kg"])
        for v, rec in vi_stats.items()
        if v not in no_stock_vis and rec["demand_kg"] > rec["total_stock_kg"]
    ]
    return (sorted(unassignable, key=lambda r: -r[2]),
            sorted(shortages, key=lambda r: r[1]-r[2], reverse=True),
            no_stock_vis)


def residual_pivotalness(vi_stats):
    """For each VI: packing_size vs (largest lot remaining_kg) and vs mean unit
    consumption.  Small ratio → residual is second-order → approximate capacity OK."""
    rows = []
    for v, rec in vi_stats.items():
        max_lot = max((float(l.get("remaining_kg", 0)) for l in rec["lots"]), default=0.0)
        mean_unit = (rec["demand_kg"] / rec["units"]) if rec["units"] else 0.0
        pk = max(rec["packing_sizes"], default=0.0)
        rows.append((v, pk, max_lot, mean_unit,
                     (pk / max_lot) if max_lot else None,
                     (pk / mean_unit) if mean_unit else None))
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def detect_is_main_mode(payload, knit_ids):
    """Return ('flagged'|'legacy', n_secondary_dropped)."""
    flagged = 0
    secondary = 0
    for t in payload.get("tasks", []):
        if t["task_id"] not in knit_ids:
            continue
        for c in t.get("main_yarn_consumption") or []:
            if "is_main" in c:
                flagged += 1
                if not c["is_main"]:
                    secondary += 1
    return ("flagged" if flagged else "legacy", secondary)


def render(payload_path, gran, vi_stats, worst, og_rows, unassignable, shortages,
           no_stock_vis, resid, sched_status, ismain_mode):
    L = []
    w = L.append
    w(f"# PHASE V — Dyelot allocation problem measurement\n")
    w(f"Payload: `{payload_path}`  | knitting schedule status: `{sched_status}`")
    mode, n_sec = ismain_mode
    if mode == "flagged":
        w(f"main_yarn mode: **flagged** (is_main present; {n_sec} secondary-yarn entries excluded)\n")
    else:
        w(f"main_yarn mode: **legacy** (no is_main flag → every consumption entry treated as main)\n")

    # Granularity / blocker
    w("## 1. Chunk-granularity check (blocker gate)\n")
    w(f"- knitting scheduling units: **{gran['n_units']}**  "
      f"(Go-side slices: {gran['n_slice']}, with parent_task_id: {gran['n_parent']})")
    w(f"- units carrying `main_yarn_consumption`: **{gran['n_with_myc']}/{gran['n_units']}**")
    if gran['n_slice']:
        w(f"- slices carrying consumption: **{gran['n_slice_with_myc']}/{gran['n_slice']}**")
    w(f"- **BLOCKER: {'YES' if gran['blocker'] else 'NO'}**")
    for nt in gran["notes"]:
        w(f"  - ⚠️ {nt}")
    w("")

    # Per-VI table
    w("## 2. Per-VI summary\n")
    w("| VI | orders | dyelots | units | machines | demand kg | stock kg | demand/stock | max lot kg | packing_size |")
    w("|----|-------:|--------:|------:|---------:|----------:|---------:|-------------:|-----------:|-------------:|")
    for v in sorted(vi_stats):
        r = vi_stats[v]
        max_lot = max((float(l.get("remaining_kg", 0)) for l in r["lots"]), default=0.0)
        pk = max(r["packing_sizes"], default=0.0)
        ratio = (r["demand_kg"]/r["total_stock_kg"]) if r["total_stock_kg"] else float('inf')
        w(f"| {v} | {len(r['orders'])} | {len(r['lots'])} | {r['units']} | "
          f"{len(r['machines'])} | {r['demand_kg']:.1f} | {r['total_stock_kg']:.1f} | "
          f"{ratio:.2f} | {max_lot:.1f} | {pk:.1f} |")
    w("")

    # Over-group baseline
    w("## 3. Never-flush over-group baseline (the thing to beat)\n")
    if worst:
        kg, no, m, v, max_lot, fits = worst
        w(f"- Largest forced cohort: machine `{m}`, VI `{v}` → **{no} orders, {kg:.1f} kg** "
          f"onto ONE dyelot; largest lot = {max_lot:.1f} kg → "
          f"**{'FITS' if fits else 'DOES NOT FIT'}**.")
        n_overflow = sum(1 for r in og_rows if not r[5])
        w(f"- (machine-cohort, VI) bindings that overflow their largest lot: "
          f"**{n_overflow}/{len(og_rows)}**.")
        w("\nTop 8 cohort bindings by kg:\n")
        w("| machine | VI | orders | cohort kg | max lot kg | fits |")
        w("|---------|----|-------:|----------:|-----------:|:----:|")
        for kg, no, m, v, max_lot, fits in og_rows[:8]:
            w(f"| {m} | {v} | {no} | {kg:.1f} | {max_lot:.1f} | {'✅' if fits else '❌'} |")
    else:
        w("- (no consumption data — cannot compute)")
    w("")

    # Residual pivotalness
    w("## 4. Residual / packing pivotalness\n")
    w("| VI | packing_size | max lot kg | mean unit kg | pk/lot | pk/unit |")
    w("|----|-------------:|-----------:|-------------:|-------:|--------:|")
    for v, pk, max_lot, mu, r_lot, r_unit in sorted(resid):
        w(f"| {v} | {pk:.1f} | {max_lot:.1f} | {mu:.1f} | "
          f"{('%.3f'%r_lot) if r_lot is not None else '—'} | "
          f"{('%.2f'%r_unit) if r_unit is not None else '—'} |")
    w("")

    # Feasibility
    w("## 5. Feasibility sanity\n")
    if no_stock_vis:
        w(f"- ⚠️ **{len(no_stock_vis)} VI(s) consumed as MAIN but have ZERO dyelot_stock**: "
          f"{', '.join('`%s`' % v for v in no_stock_vis)}. "
          f"Either secondary yarns mis-flagged as main (fix: `is_main=false`) or a real "
          f"stock gap — must resolve before allocation (no lot can be assigned).")
    if unassignable:
        w(f"- **{len(unassignable)} (order, VI) pairs are UNASSIGNABLE** "
          f"(demand > largest lot, even un-grouped):")
        for o, v, kg, max_lot in unassignable[:10]:
            w(f"  - order `{o}` VI `{v}`: {kg:.1f} kg > {max_lot:.1f} kg")
    else:
        w("- No single (order, VI) exceeds its largest lot. ✅")
    if shortages:
        w(f"- **{len(shortages)} VIs have total demand > total stock** (real yarn shortage):")
        for v, dem, stock in shortages:
            w(f"  - VI `{v}`: demand {dem:.1f} kg > stock {stock:.1f} kg")
    else:
        w("- No VI is short on total stock. ✅")
    w("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payload")
    ap.add_argument("--output", default=None, help="solver_output_*.json to pair with")
    ap.add_argument("--md", default=None, help="also write report to this path")
    args = ap.parse_args()

    with open(args.payload) as fh:
        payload = json.load(fh)
    knit_ids = _knitting_ids(payload)

    sched_status = "paired-from-output"
    if args.output:
        with open(args.output) as fh:
            output = json.load(fh)
        sched = _schedule_from_output(payload, output, knit_ids)
    else:
        sched, sched_status = _schedule_cold(payload, knit_ids)

    gran = check_granularity(payload, knit_ids)
    cohorts = build_machine_cohorts(payload, knit_ids, sched)
    vi_stats = per_vi_stats(payload, knit_ids, cohorts)
    worst, og_rows = over_group(cohorts, vi_stats)
    unassignable, shortages, no_stock_vis = feasibility(payload, knit_ids, vi_stats)
    resid = residual_pivotalness(vi_stats)
    ismain_mode = detect_is_main_mode(payload, knit_ids)

    report = render(args.payload, gran, vi_stats, worst, og_rows,
                    unassignable, shortages, no_stock_vis, resid,
                    sched_status, ismain_mode)
    print(report)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(report)
        print(f"\n[written to {args.md}]", file=sys.stderr)


if __name__ == "__main__":
    main()
