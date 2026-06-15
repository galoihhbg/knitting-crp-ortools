"""Dyelot allocation post-pass (CP-SAT) — runs AFTER knitting is scheduled.

Replaces the Go-side greedy "never-flush" cohort builder with a flush-optimized
allocation: it decides which dyelot each order uses for each main yarn (VI),
cutting (flushing) creel chains where needed so an over-grouped cohort can be
split across several lots instead of being forced onto one lot that may not fit.

Domain (measured in PHASE V — see agent_docs/phase5_dyelot_measurement.md):
  * Each order uses exactly ONE dyelot per VI (mixing dyelots of one VI streaks
    the colour).  This is an order-level decision, free to be consistent across
    machines.
  * On a machine, consecutive units that consume a VI share the creel residual →
    they inherit the previous unit's dyelot unless the chain is FLUSHED.  A flush
    discards the residual (≤ packing_size kg) = waste.
  * Capacity is GROSS, not net.  Go commits stock by pulling WHOLE rolls
    (packing_size kg each), so an order's physical charge against a lot is its
    net demand rounded UP to a whole number of that lot's rolls:
    ceil(demand_kg / packing_size) · packing_size.  The lot capacity constraint
    uses this gross charge so the model never over-subscribes a lot that Go would
    later find physically short.  Residual creel-flow is still ignored (PHASE V:
    packing_size/lot ≈ 0.002, third-order) — no per-machine flow network.

The problem is SEPARABLE PER VI (flush is per-cone; lots of one VI never interact
with another VI), so we solve one tiny independent CP-SAT model per VI.

Architecture: pure orchestration.  Reads the knitting machine-sequence from the
scheduler's assignments + each task's main_yarn_consumption + dyelot_stock.
Touches NO scheduling CP-SAT / builder.py.  Output is attached to the result dict
returned to Go.

Entry point: allocate_dyelots(tasks, knitting_assignments, dyelot_stock, config).
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from .shared import make_solver

logger = logging.getLogger(__name__)

_KG_SCALE = 1000  # kg → integer grams (CP-SAT bounds/penalties must be int)


def _g(kg: float) -> int:
    """kg (float) → integer grams.  All CP-SAT quantities must be int."""
    return int(round(float(kg) * _KG_SCALE))


def _main_consumption(task: Dict[str, Any]) -> Dict[str, Tuple[float, int]]:
    """task → {vi: (kg, slots)} for MAIN yarn only.

    Honors is_main when present (True → main, gets a dyelot; False → secondary,
    skipped); defaults to True when the flag is absent (legacy payloads predate
    the main/secondary split — every entry was a main yarn then).

    slots = number of creel positions (cones) the task mounts for the vi (Go's
    MinSlots). Drives the creel-up gross charge; 0 when absent (legacy payloads
    → no creel-up, capacity falls back to whole-roll net rounding).
    """
    out: Dict[str, Tuple[float, int]] = {}
    for c in task.get("main_yarn_consumption") or []:
        if c.get("is_main", True):
            vi = c["vi"]
            kg, slots = out.get(vi, (0.0, 0))
            out[vi] = (kg + float(c.get("kg", 0)),
                       max(slots, int(c.get("slots", 0) or 0)))
    return out


def allocate_dyelots(
    tasks: List[Dict[str, Any]],
    knitting_assignments: List[Dict[str, Any]],
    dyelot_stock: Optional[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Allocate one dyelot per (order, VI), flush-optimized, per VI independently.

    Returns a dict to merge into the solver result:
      order_dyelot_assignment: [{order, vi, dyelot}]
      dyelot_flush_points:     [{machine, vi, after_task, before_task,
                                 order_after, order_before}]
      dyelot_unassigned:       [{order, vi, reason}]
      dyelot_shortage:         [{vi, demand_kg, stock_kg}]

    Never raises on an infeasible VI — an order that cannot be placed is reported
    in dyelot_unassigned (a procurement signal), not crashed.
    """
    empty = {
        "order_dyelot_assignment": [],
        "dyelot_flush_points": [],
        "dyelot_unassigned": [],
        "dyelot_shortage": [],
    }
    if not dyelot_stock:
        logger.info("🎨 Dyelot allocator: no dyelot_stock in payload — skipped.")
        return empty

    # task_id → (machine, start) from the scheduler's knitting assignments.
    knit_ids = {
        t["task_id"] for t in tasks
        if str(t.get("operation", "")).lower() == "knitting"
    }
    sched: Dict[str, Tuple[Any, int]] = {}
    for a in knitting_assignments:
        tid = a.get("task_id")
        if tid in knit_ids:
            sched[tid] = (a.get("machine_id"), int(a.get("start_time", 0)))
    task_by_id = {t["task_id"]: t for t in tasks if t["task_id"] in knit_ids}

    def _order_of(t: Dict[str, Any]) -> str:
        """Dyelot-grouping identity = the ĐƠN (sales order), so every batch/panel
        of one order shares a dyelot (no colour mismatch on an assembled product).

        Prefers the explicit order_group_id Go sends; rolling-wave batching can
        split one đơn across several batch tasks (e.g. BATCH_0-665 + BATCH_0-666
        are both đơn 'W9xTMuuLxR-1-200-200'), and only this field links them.
        Falls back to original_order_id / group_id / task_id when absent (legacy
        payload → per-batch grouping, unchanged)."""
        return (t.get("order_group_id")
                or t.get("original_order_id") or t.get("group_id") or t["task_id"])

    # ── Per-VI demand + machine sequences ────────────────────────────────────
    # vi → order → kg ; vi → machine → [ (start, task_id, order) ... ]
    # vi_mo: vi → machine → order → [net_kg, slots] — per-(machine, order) run
    # used to charge the creel-up gross (a machine run must mount `slots` cones).
    vi_order_kg: Dict[str, Dict[str, float]] = {}
    vi_machine_units: Dict[str, Dict[Any, List[Tuple[int, str, str]]]] = {}
    vi_mo: Dict[str, Dict[Any, Dict[str, List[float]]]] = {}
    for tid, t in task_by_id.items():
        if tid not in sched:
            continue
        machine, start = sched[tid]
        order = _order_of(t)
        for vi, (kg, slots) in _main_consumption(t).items():
            vi_order_kg.setdefault(vi, {}).setdefault(order, 0.0)
            vi_order_kg[vi][order] += kg
            vi_machine_units.setdefault(vi, {}).setdefault(machine, []).append(
                (start, tid, order)
            )
            run = vi_mo.setdefault(vi, {}).setdefault(machine, {}).setdefault(order, [0.0, 0])
            run[0] += kg
            run[1] = max(run[1], slots)

    # vi → sorted list of lots
    lots_by_vi: Dict[str, List[Dict[str, Any]]] = {}
    for d in dyelot_stock:
        lots_by_vi.setdefault(d["vi"], []).append(d)
    for vi in lots_by_vi:
        lots_by_vi[vi].sort(key=lambda d: str(d.get("dyelot", "")))

    assignments: List[Dict[str, Any]] = []
    flush_points: List[Dict[str, Any]] = []
    unassigned: List[Dict[str, Any]] = []
    shortage: List[Dict[str, Any]] = []

    # Determinism leg 3: iterate VIs in stable id order (pairs with 1 worker + seed).
    for vi in sorted(vi_order_kg):
        order_kg = vi_order_kg[vi]
        lots = lots_by_vi.get(vi, [])
        demand_kg = sum(order_kg.values())

        if not lots:
            # VI consumed as main but no stock at all → every order unassigned.
            for order in sorted(order_kg):
                unassigned.append({"order": order, "vi": vi,
                                   "reason": "no_dyelot_stock"})
            shortage.append({"vi": vi, "demand_kg": round(demand_kg, 3),
                             "stock_kg": 0.0})
            logger.warning(f"🎨 VI {vi}: {len(order_kg)} order(s) consume it but "
                           f"dyelot_stock is empty → shortage.")
            continue

        res = _solve_vi(vi, order_kg, lots, vi_machine_units.get(vi, {}),
                        vi_mo.get(vi, {}), config)
        assignments.extend(res["assignments"])
        flush_points.extend(res["flush_points"])
        unassigned.extend(res["unassigned"])
        if res["unassigned"]:
            # An order was dropped for capacity → ALWAYS surface a shortage row.
            # demand_kg is GROSS (physically-pullable: whole-roll + creel-up), so a
            # VI whose net demand exactly fits stock but overflows on roll/creel-up
            # waste (e.g. 3039: net 403/403 but gross 413) is reported, not hidden.
            stock_kg = sum(float(l.get("remaining_kg", 0)) for l in lots)
            gross_demand_kg = res["gross_demand_g"] / _KG_SCALE
            shortage.append({"vi": vi,
                             "demand_kg": round(gross_demand_kg, 3),
                             "stock_kg": round(stock_kg, 3)})

    logger.info(
        f"🎨 Dyelot allocator: {len(assignments)} (order,VI) assigned, "
        f"{len(flush_points)} flush point(s), {len(unassigned)} unassigned, "
        f"{len(shortage)} shortage VI(s)."
    )
    return {
        "order_dyelot_assignment": assignments,
        "dyelot_flush_points": flush_points,
        "dyelot_unassigned": unassigned,
        "dyelot_shortage": shortage,
    }


def _solve_vi(
    vi: str,
    order_kg: Dict[str, float],
    lots: List[Dict[str, Any]],
    machine_units: Dict[Any, List[Tuple[int, str, str]]],
    machine_order: Dict[Any, Dict[str, List[float]]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """One independent CP-SAT model for a single VI.

    Variables (all int, per the project invariant):
      x[o][d]      one-hot: order o uses dyelot d
      unassigned[o]
      exactly_one: Σ_d x[o][d] + unassigned[o] == 1     (1 dyelot per order per VI)
    Capacity (GROSS — whole rolls + creel-up): Σ_o gross[o][d]·x[o][d] ≤ cap[d].
      gross[o][d] sums, over each machine the order runs the vi on,
        max(ceil(net_run / packing[d]), slots_run) · packing[d]
      i.e. whole-roll rounding PLUS the creel-up floor — a machine run must mount
      `slots` cones (Go's MinSlots), so even a tiny net run pulls slots·packing kg.
      This mirrors Go's loaded = max(ceil(deficit/packing), MinSlots)·packing and
      stops the model packing a lot to its net edge (e.g. 219.75/220) that Go then
      finds physically short on commit. With slots absent (legacy payload) it
      degrades to plain whole-roll net rounding. Residual creel-flow between runs
      is still ignored; the per-(order,machine) charge can over-count two orders
      that share one creel on a machine without a flush — a deliberate safe bias
      (over-reserve, never under).
    Flush: per machine, walk the VI's units in schedule order; between adjacent
      units of DIFFERENT orders a flush is needed iff they pick different dyelots.
      cut = both_assigned − same_lot  (linear; same_lot=1 only when both assigned).
    Objective (lexicographic via data-bounded weights, Maximize):
      (1) feasibility       — maximize assigned orders
      (2) min flush         — fewest cohort cuts (Σ cut)
      (3) min lots opened   — Σ used[d]; consolidates a VI onto as few lots as
                              possible WITHOUT preferring small lots, so multi-order
                              demand lands on the lot that holds the most instead of
                              stranding on a nearly-full small lot while a big lot
                              sits idle (the production over-concentration bug).
      (4) drain-small       — Σ used[d]·remaining_g[d] as a pure TIE-BREAK below the
                              lot-count tier: among equal-lot-count, equal-flush
                              solutions, consume the nearly-empty lot first.  Demoted
                              from tier 3 (where it caused the over-concentration) so
                              it can no longer drive cramming.
    """
    orders = sorted(order_kg)
    n_lots = len(lots)
    demand_g = {o: _g(order_kg[o]) for o in orders}
    cap_g = [_g(l.get("remaining_kg", 0)) for l in lots]
    remaining_g = list(cap_g)
    # Per-lot roll size in grams (≥1 so the ceil is well-defined even if
    # packing_size is missing / rounds to 0).
    pk_g = [max(_g(l.get("packing_size", 0)), 1) for l in lots]

    def _gross_g(o: str, d: int) -> int:
        """Physically-pullable grams for order o on lot d: per machine run, the
        net rounded UP to whole rolls of lot d, floored at slots·packing (the
        creel-up — a run must mount `slots` cones). Falls back to whole-order
        roll rounding when the order has no per-machine breakdown."""
        p = pk_g[d]
        total = 0
        seen = False
        for mach in sorted(machine_order, key=str):
            run = machine_order[mach].get(o)
            if run is None:
                continue
            seen = True
            net_g_run = _g(run[0])
            slots = int(run[1])
            rolls = (net_g_run + p - 1) // p   # whole rolls to cover the net
            if slots > rolls:
                rolls = slots                  # creel-up: mount all slots' cones
            total += rolls * p
        if not seen:
            return ((demand_g[o] + p - 1) // p) * p
        return total

    # Lower-bound gross demand for the VI: each order charged on its CHEAPEST lot
    # (least roll/creel-up waste).  Reported in dyelot_shortage so Go sees the
    # physically-pullable kg, not the net kg — an order can be dropped for capacity
    # even when net demand == net stock once whole-roll/creel-up waste is counted.
    gross_demand_g = sum(min(_gross_g(o, d) for d in range(n_lots)) for o in orders)

    model = cp_model.CpModel()

    x = {o: [model.NewBoolVar(f"x_{vi}_{o}_{d}") for d in range(n_lots)] for o in orders}
    una = {o: model.NewBoolVar(f"una_{vi}_{o}") for o in orders}
    assigned = {o: model.NewBoolVar(f"asg_{vi}_{o}") for o in orders}
    for o in orders:
        model.Add(sum(x[o]) + una[o] == 1)        # 1 dyelot per order per VI
        model.Add(assigned[o] == 1 - una[o])

    # Capacity — GROSS (whole-roll) charge, residual creel-flow still ignored.
    for d in range(n_lots):
        model.Add(sum(_gross_g(o, d) * x[o][d] for o in orders) <= cap_g[d])

    # used[d]: 1 iff lot d carries any order.  Drives tier-3 (min lots) + tier-4.
    used = [model.NewBoolVar(f"used_{vi}_{d}") for d in range(n_lots)]
    for d in range(n_lots):
        for o in orders:
            model.Add(used[d] >= x[o][d])

    # Flush points: per machine, adjacent VI-units of different orders.
    flush_terms = []          # (cut_var, machine, taskA, taskB, orderA, orderB)
    for machine in sorted(machine_units, key=str):
        seq = sorted(machine_units[machine], key=lambda u: (u[0], u[1]))
        for (_, ta, oa), (_, tb, ob) in zip(seq, seq[1:]):
            if oa == ob:
                continue  # same order → same dyelot, residual carries, no cut
            # same_lot = Σ_d (x[oa][d] AND x[ob][d]); ∈{0,1} (one-hot).
            zsum = []
            for d in range(n_lots):
                z = model.NewBoolVar(f"z_{vi}_{machine}_{ta}_{tb}_{d}")
                model.Add(z <= x[oa][d])
                model.Add(z <= x[ob][d])
                model.Add(z >= x[oa][d] + x[ob][d] - 1)
                zsum.append(z)
            both = model.NewBoolVar(f"both_{vi}_{machine}_{ta}_{tb}")
            model.Add(both <= assigned[oa])
            model.Add(both <= assigned[ob])
            model.Add(both >= assigned[oa] + assigned[ob] - 1)
            cut = model.NewBoolVar(f"cut_{vi}_{machine}_{ta}_{tb}")
            # cut = both − same_lot  (same_lot ≤ both, so cut ∈ {0,1}).
            model.Add(cut == both - sum(zsum))
            flush_terms.append((cut, machine, ta, tb, oa, ob))

    # ── Lexicographic objective via data-bounded weights ─────────────────────
    # Tiers (high→low), folded into one Maximize:
    #   (1) +assigned  (2) −flush  (3) −lots opened  (4) −drain-small tie-break
    # Each tier's weight must dominate the MAX possible value of every lower tier
    # summed, so the optimum is exactly lexicographic.
    assigned_sum = sum(assigned[o] for o in orders)
    flush_sum = sum(cut for (cut, *_rest) in flush_terms)   # number of cohort cuts
    lots_sum = sum(used)                                    # distinct lots opened
    frag_sum = sum(used[d] * remaining_g[d] for d in range(n_lots))  # drain-small

    n_pairs = len(flush_terms)                              # max value of flush_sum
    max_frag = sum(remaining_g)                             # max value of frag_sum
    k_frag = 1
    # Opening one fewer lot must always beat any drain-small gain.
    k_lots = k_frag * max_frag + 1
    # One fewer flush must beat the worst (lots + drain-small) combined.
    k_flush = k_lots * n_lots + k_frag * max_frag + 1
    # Assigning one more order must beat the worst (flush + lots + drain-small)
    # combined — so feasibility is never traded for a lower tier (this is also why
    # a VI with no flush pairs still assigns: k_assign carries the lots+frag bound).
    k_assign = k_flush * n_pairs + k_lots * n_lots + k_frag * max_frag + 1
    model.Maximize(
        k_assign * assigned_sum
        - k_flush * flush_sum
        - k_lots * lots_sum
        - k_frag * frag_sum
    )

    solver = make_solver(config)
    status = solver.Solve(model)

    out: Dict[str, Any] = {"assignments": [], "flush_points": [],
                           "unassigned": [], "gross_demand_g": gross_demand_g}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Should not happen (unassigned makes it always feasible) — defensive.
        logger.warning(f"🎨 VI {vi}: solver status {solver.StatusName(status)} "
                       f"— reporting all orders unassigned.")
        for o in orders:
            out["unassigned"].append({"order": o, "vi": vi,
                                      "reason": f"solver_{solver.StatusName(status)}"})
        return out

    for o in orders:
        if solver.Value(una[o]):
            out["unassigned"].append({"order": o, "vi": vi,
                                      "reason": "capacity_shortage"})
            continue
        for d in range(n_lots):
            if solver.Value(x[o][d]):
                out["assignments"].append(
                    {"order": o, "vi": vi, "dyelot": lots[d].get("dyelot")}
                )
                break
    for cut, machine, ta, tb, oa, ob in flush_terms:
        if solver.Value(cut):
            out["flush_points"].append({
                "machine": machine, "vi": vi,
                "after_task": ta, "before_task": tb,
                "order_after": oa, "order_before": ob,
            })
    return out
