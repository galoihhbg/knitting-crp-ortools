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
      dyelot_shortage:         [{vi, demand_kg (gross), net_demand_kg, stock_kg,
                                 single_lot_deficit_kg, topups:[{dyelot, add_kg}],
                                 new_lot_kg}]
                                topups[] = ALTERNATIVES — "add add_kg to THIS dyelot
                                alone" (dyelots are never merged, so a real top-up
                                replenishes exactly one lot); single_lot_deficit_kg is
                                the cheapest of them. new_lot_kg = instead import ONE
                                fresh dyelot of this size.

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
            # No stock at all → the whole demand must be procured as a fresh lot;
            # no existing dyelot to top up.
            shortage.append({"vi": vi, "demand_kg": round(demand_kg, 3),
                             "net_demand_kg": round(demand_kg, 3),
                             "stock_kg": 0.0,
                             "single_lot_deficit_kg": round(demand_kg, 3),
                             "topups": [],
                             "new_lot_kg": round(demand_kg, 3)})
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
            # single_lot_deficit_kg = minimal extra capacity to assign EVERY order
            # to one dyelot (bin-packing aware). For a fragmentation-only shortage
            # (total stock ≥ gross demand) this is small/zero even though a large
            # order was stranded; it is the honest "add this much to clear".
            deficit_kg = res.get("deficit_g", 0) / _KG_SCALE
            # net_demand_kg is the actual consumed-into-product demand (no
            # whole-roll / creel-up inflation). The Go side classifies
            # material-shortage vs dyelot-fragmentation on NET vs stock — gross
            # (demand_kg) over-counts and would mislabel a fragmentation case as
            # a material buy near the boundary (e.g. 3039 net 403 ≤ stock 423 but
            # gross 440 > 423).
            # Procurement options to clear it (pick ONE downstream — dyelots are
            # never merged):
            #   topups[]    — ALTERNATIVES: "add add_kg to THIS dyelot alone".
            #                 single_lot_deficit_kg = the cheapest of these.
            #   new_lot_kg  — instead, import ONE fresh dyelot of this size.
            topups_g = res.get("topups_g") or []
            topups = [{"dyelot": lots[d].get("dyelot"),
                       "add_kg": round(topups_g[d] / _KG_SCALE, 3)}
                      for d in range(len(lots))
                      if d < len(topups_g) and topups_g[d] is not None
                      and topups_g[d] > 0]
            new_lot_kg = res.get("new_lot_g", 0) / _KG_SCALE
            shortage.append({"vi": vi,
                             "demand_kg": round(gross_demand_kg, 3),
                             "net_demand_kg": round(demand_kg, 3),
                             "stock_kg": round(stock_kg, 3),
                             "single_lot_deficit_kg": round(deficit_kg, 3),
                             "topups": topups,
                             "new_lot_kg": round(new_lot_kg, 3)})

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


def _vi_gross_demand_g(orders, demand_g, pk_g, machine_order) -> int:
    """Reuse-aware gross (grams) the VI physically pulls, lot-blind. On each
    machine the orders sharing it draw from ONE creel, so that machine pulls
    max(ceil(Σnet_m / pk), max_slots_m) rolls; summed across machines (each mounts
    its own cones). Orders with no machine breakdown fall back to whole-order
    rolls. Used for the dyelot_shortage report (vs net stock)."""
    pk = min(pk_g)
    machines = sorted(machine_order, key=str)
    with_machine = set()
    g = 0
    for m in machines:
        mo = machine_order[m]
        oz = [o for o in orders if o in mo]
        if not oz:
            continue
        with_machine.update(oz)
        net_m = sum(_g(mo[o][0]) for o in oz)
        max_slots = max((int(mo[o][1]) for o in oz), default=0)
        g += max((net_m + pk - 1) // pk, max_slots) * pk
    missing = [o for o in orders if o not in with_machine]
    if missing:
        net_miss = sum(demand_g[o] for o in missing)
        g += ((net_miss + pk - 1) // pk) * pk
    return g


def _cap_slack_g(demand_g, pk_g, machine_order) -> int:
    """A guaranteed-sufficient upper bound (grams) on the extra capacity any single
    lot could ever need / be charged: net total + one roll per (machine,order) run
    (whole-roll waste) + the creel-up of every run. Always ≥ the true gross of
    placing EVERYTHING on one lot, so it safely bounds expandable-bin Vars and the
    per-(machine,lot) roll Vars (the earlier Σnet+max_pk bound under-counted the
    gross and made the new-lot solve spuriously infeasible)."""
    runs = 0
    tslots = 0
    for m in machine_order:
        for _o, run in machine_order[m].items():
            runs += 1
            tslots += int(run[1])
    return sum(demand_g.values()) + (runs + tslots) * max(pk_g)


def _add_roll_capacity(model, x, orders, n_lots, demand_g, cap_g, pk_g,
                       machine_order, extra=None) -> None:
    """Add reuse-aware capacity: per (machine, lot) the orders of a lot SHARE the
    creel on that machine (a cone's residual feeds the next order — Go's
    free-pool), so the rolls mounted on machine m for lot d cover the SUM of those
    orders' net AND the widest creel (max slots). Rolls are summed ACROSS machines
    (each machine mounts its own cones from the lot) and bounded by the lot stock
    (cap_g[d], or cap_g[d]+extra[d] when expandable). This removes the per-ORDER
    creel-up over-count (orders sharing a machine no longer each pay the slots
    floor) while still charging an order that knits on several machines its creel
    on each."""
    machines = sorted(machine_order, key=str)
    with_machine = {o for m in machines for o in machine_order[m] if o in demand_g}
    missing = [o for o in orders if o not in with_machine]
    # When lots are expandable, a lot may grow far past Σnet (gross > net), so the
    # per-(machine,lot) roll Var ub must cover the worst-case slack, not just Σnet.
    bump = _cap_slack_g(demand_g, pk_g, machine_order) if extra is not None else 0
    for d in range(n_lots):
        ub = (cap_g[d] + bump) // pk_g[d] + 1
        lot_rolls = []
        for m in machines:
            mo = machine_order[m]
            oz = [o for o in orders if o in mo]
            if not oz:
                continue
            r = model.NewIntVar(0, ub, f"r_{d}_{m}")
            model.Add(r * pk_g[d] >= sum(_g(mo[o][0]) * x[o][d] for o in oz))
            for o in oz:
                s = int(mo[o][1])
                if s > 0:
                    model.Add(r * pk_g[d] >= s * pk_g[d] * x[o][d])  # creel floor
            lot_rolls.append(r)
        if missing:
            r = model.NewIntVar(0, ub, f"rm_{d}")
            model.Add(r * pk_g[d] >= sum(demand_g[o] * x[o][d] for o in missing))
            lot_rolls.append(r)
        if lot_rolls:
            cap = cap_g[d] + (extra[d] if extra is not None else 0)
            model.Add(sum(lot_rolls) * pk_g[d] <= cap)


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

    # Reuse-aware GROSS (C) — see _add_roll_capacity / _vi_gross_demand_g.
    gross_demand_g = _vi_gross_demand_g(orders, demand_g, pk_g, machine_order)

    model = cp_model.CpModel()

    x = {o: [model.NewBoolVar(f"x_{vi}_{o}_{d}") for d in range(n_lots)] for o in orders}
    una = {o: model.NewBoolVar(f"una_{vi}_{o}") for o in orders}
    assigned = {o: model.NewBoolVar(f"asg_{vi}_{o}") for o in orders}
    for o in orders:
        model.Add(sum(x[o]) + una[o] == 1)        # 1 dyelot per order per VI
        model.Add(assigned[o] == 1 - una[o])

    # Capacity — reuse-aware per-(machine, lot) roll model (C).
    _add_roll_capacity(model, x, orders, n_lots, demand_g, cap_g, pk_g, machine_order)

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

    any_unassigned = False
    for o in orders:
        if solver.Value(una[o]):
            out["unassigned"].append({"order": o, "vi": vi,
                                      "reason": "capacity_shortage"})
            any_unassigned = True
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

    # When an order was dropped, compute the MINIMAL extra single-lot capacity
    # that would let EVERY order land on one dyelot (expandable-bin). This is the
    # honest "how much to add to clear the error": for a tight bin-packing it is
    # tiny (e.g. ~0.5kg → 1 roll) even though a whole 130kg order was stranded,
    # and it is 0 when a feasible single-lot packing exists but the heuristic
    # over-concentrated. Go surfaces it as the procurement top-up instead of the
    # inflated cohort estimate. Only run the extra solve when needed.
    if any_unassigned:
        # Procurement options to clear the shortage (all honest minima under the
        # SAME reuse-aware capacity as the main solve). Dyelots are never merged,
        # so each option replenishes exactly ONE lot:
        #   topups_g[d] — extra kg to add to dyelot d ALONE (a standalone choice);
        #                 None when that lot can't be the single sink.
        #   deficit_g   — the CHEAPEST single-lot top-up (min over topups_g).
        #   new_lot_g   — size of ONE fresh dyelot to import instead.
        topups_g = [_topup_one_lot_g(d, orders, n_lots, demand_g, cap_g, pk_g,
                                     machine_order, config)
                    for d in range(n_lots)]
        out["topups_g"] = topups_g
        feasible = [g for g in topups_g if g is not None]
        out["deficit_g"] = min(feasible) if feasible else 0
        out["new_lot_g"] = _new_lot_g(
            orders, n_lots, demand_g, cap_g, pk_g, machine_order, config)
    return out


def _topup_one_lot_g(d_fill, orders, n_lots, demand_g, cap_g, pk_g,
                     machine_order, config) -> Optional[int]:
    """Minimal extra capacity (grams) added to dyelot `d_fill` ALONE (all other
    lots fixed at their current stock) so that a one-dyelot-per-order assignment of
    ALL orders becomes feasible. This is a STANDALONE option — dyelots are never
    merged, so a real top-up replenishes exactly one lot — not a spread across
    several. Returns None if even an unbounded single-lot top-up can't host the
    orders (should not happen: lot d_fill can absorb everything)."""
    m = cp_model.CpModel()
    x = {o: [m.NewBoolVar(f"tx_{o}_{d}") for d in range(n_lots)] for o in orders}
    ub = _cap_slack_g(demand_g, pk_g, machine_order)
    fill = m.NewIntVar(0, ub, "fill")
    extra = [fill if d == d_fill else 0 for d in range(n_lots)]  # only d_fill grows
    for o in orders:
        m.Add(sum(x[o]) == 1)  # every order must land on exactly one lot
    _add_roll_capacity(m, x, orders, n_lots, demand_g, cap_g, pk_g,
                       machine_order, extra=extra)
    m.Minimize(fill)
    s = make_solver(config)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return int(s.Value(fill))


def _new_lot_g(orders, n_lots, demand_g, cap_g, pk_g,
               machine_order, config) -> int:
    """Minimal size (grams) of ONE fresh dyelot which, added alongside the
    existing lots, makes a one-dyelot-per-order assignment of ALL orders feasible.
    Same expandable-bin model but the free capacity lives ONLY on a new
    (n_lots+1)-th lot — the 'import a new dyelot of this size' answer. The new lot
    inherits the smallest existing roll size (whole-roll charge stays consistent)."""
    nl = n_lots + 1
    cap2 = cap_g + [0]
    pk2 = pk_g + [min(pk_g)]
    m = cp_model.CpModel()
    x = {o: [m.NewBoolVar(f"nx_{o}_{d}") for d in range(nl)] for o in orders}
    ub = _cap_slack_g(demand_g, pk2, machine_order)  # safe bound (gross ≥ Σnet)
    new = m.NewIntVar(0, ub, "new_lot")
    extra = [0] * n_lots + [new]          # only the fresh lot is expandable
    for o in orders:
        m.Add(sum(x[o]) == 1)
    _add_roll_capacity(m, x, orders, nl, demand_g, cap2, pk2,
                       machine_order, extra=extra)
    m.Minimize(new)
    s = make_solver(config)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 0
    return int(s.Value(new))
