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
    vi_packing: Optional[Dict[str, float]] = None,
    in_production: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Allocate one dyelot per (order, VI), flush-optimized, per VI independently.

    Returns a dict to merge into the solver result:
      order_dyelot_assignment: [{order, vi, dyelot}]
      dyelot_flush_points:     [{machine, vi, after_task, before_task,
                                 order_after, order_before}]
      dyelot_unassigned:       [{order, vi, reason}]
      dyelot_shortage:         [{vi, demand_kg (gross), net_demand_kg, stock_kg,
                                 warehouse_kg, committed_kg, n_lots, shortage_kind,
                                 single_lot_deficit_kg, topups:[{dyelot, add_kg}],
                                 new_lot_kg}]
                                topups[] = ALTERNATIVES — "add add_kg to THIS dyelot
                                alone" (dyelots are never merged, so a real top-up
                                replenishes exactly one lot); single_lot_deficit_kg is
                                the cheapest of them. new_lot_kg = instead import ONE
                                fresh dyelot of this size.
                                stock_kg = warehouse_kg + committed_kg (in-production
                                creel already PICKED onto a machine — not purchasable);
                                the capacity that binds is warehouse_kg.
                                shortage_kind names WHY (NO_STOCK / MATERIAL_SHORT /
                                CREEL_ROLL_SHORT / FRAGMENTED / REMEDY_UNKNOWN /
                                UNPROVEN) so the consumer never re-derives it from
                                net-vs-stock — see _classify_shortage.

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

    # Dyelot solves get their OWN deterministic budget, independent of the
    # pipeline's max_deterministic_time: a VI with many orders (e.g. 134) needs
    # more search to PROVE the all-orders-assigned optimum than knitting/washing
    # do, and the main per-VI solve is what decides feasibility.  Falls back to
    # the shared pipeline budget when `dyelot_max_deterministic_time` is unset
    # (unchanged behaviour).  Paired with relative_gap=0.0 on the dyelot solves
    # below — see _solve_vi: a 1% gap on a 134-order VI swallows a single dropped
    # order (1/134 ≈ 0.75% < 1%), surfacing a SPURIOUS shortage Go then errors on.
    dyelot_det = config.get("dyelot_max_deterministic_time")
    solver_config = ({**config, "max_deterministic_time": dyelot_det}
                     if dyelot_det is not None else config)
    # When Go's commit walker gates cross-order seed inheritance
    # (restrictSeedInheritToOwnerEnabled), a NEW co-lotting order mounts its OWN
    # cones rather than reusing an in-production order's mounted creel — so its
    # creel-up floor must NOT drop for those cones. Go sends this False to keep the
    # two in lockstep; default True preserves the legacy reuse behaviour for any
    # other caller / older payload. The in-production order's OWN accounting
    # (committed_by capacity credit, its slots=0 pooled bucket) is unaffected.
    reuse_mounted_cones = bool(config.get("dyelot_new_order_reuses_mounted_cones", True))

    # task_id → (machine, start) from the scheduler's knitting assignments.
    knit_ids = {
        t["task_id"] for t in tasks
        if str(t.get("operation", "")).lower() == "knitting"
    }
    # tid → (machine, start, end|None). `end` drives the creel-release model below;
    # legacy assignments with no end_time keep the time-blind capacity bound.
    sched: Dict[str, Tuple[Any, int, Optional[int]]] = {}
    for a in knitting_assignments:
        tid = a.get("task_id")
        if tid in knit_ids:
            end = a.get("end_time")
            sched[tid] = (a.get("machine_id"), int(a.get("start_time", 0)),
                          int(end) if end is not None else None)
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
    # vi → machine → [span_start, span_end]: the window the machine holds this VI's
    # creel (first run start → last run end).  Go releases the leftover to the free
    # pool at the end of that window, so machines with DISJOINT windows can reuse the
    # same physical rolls — see _add_roll_capacity(machine_span=...).
    vi_machine_span: Dict[str, Dict[Any, List[int]]] = {}
    vi_span_unknown: set = set()      # a unit with no end_time → no release model
    for tid, t in task_by_id.items():
        if tid not in sched:
            continue
        machine, start, end = sched[tid]
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
            if end is None:
                vi_span_unknown.add(vi)
            else:
                sp = vi_machine_span.setdefault(vi, {}).setdefault(machine, [start, end])
                sp[0] = min(sp[0], start)
                sp[1] = max(sp[1], end)

    # In-production (converted/pinned) orders: inject their committed knit units so
    # they participate in the per-VI solve. Otherwise they are invisible (excluded
    # from config.Orders; their PIN_ tasks carry no main_yarn_consumption), so no
    # flush relationship exists for a new order to co-lot against, and the order
    # already being knitted would be free to drift onto a different lot.
    # pinned_lot[vi][order] = the lot to fix. committed_by[vi][dyelot] = the PICKED
    # creel kg to re-count into that lot's capacity (see below).
    pinned_lot: Dict[str, Dict[str, str]] = {}
    committed_by: Dict[str, Dict[str, float]] = {}
    # inprod_mounted[vi][machine][dyelot] = mounted cone count (slots) — ONLY on
    # machines where the in-production order's creel is physically PICKED
    # (committed_kg>0). A co-lotting new order reuses these cones, so its creel-up
    # floor on such (machine, lot) drops to max(0, its_slots − mounted). Machines
    # NOT yet picked keep the full floor (cones not there → new order must mount).
    # Left EMPTY when reuse_mounted_cones is off (Go's owner gate blocks the reuse),
    # so no floor is dropped and every co-lotting order mounts its own cones.
    inprod_mounted: Dict[str, Dict[Any, Dict[str, int]]] = {}
    _committed_seen = set()   # (vi, order, machine, dyelot) — committed_kg repeats per machine row
    _inprod_pool_net: Dict[Tuple[str, str], float] = {}  # (vi, order) → pooled remaining net
    # (vi, order) → creel width to charge on the pooled bucket. 0 for a PINNED order
    # (its cones are already mounted and paid for via committed_kg); the order's real
    # slots for an UNRESERVED one, which still has to mount cones from somewhere.
    _inprod_pool_slots: Dict[Tuple[str, str], int] = {}
    for s in (in_production or []):
        vi = str(s.get("vi", ""))
        order = str(s.get("order", ""))
        # An EMPTY dyelot is meaningful, not junk: Go sends it for an in-production
        # order that was converted while the VI was out of stock, so it never got a
        # reservation to read a committed lot from. It has no lot to pin but it DOES
        # still have knitting to run, so its remaining net must be charged here —
        # dropping the row (the old `not dyelot` filter) deleted that demand from the
        # model outright and under-reported the procurement shortage by exactly those
        # orders' kg. Such an order stays unpinned: the solver assigns it like any
        # free order.
        dyelot = s.get("dyelot")
        dyelot = str(dyelot) if dyelot else ""
        machine = s.get("machine_id")
        if not vi or not order or machine is None:
            continue
        start = int(s.get("start_time", 0) or 0)
        kg = float(s.get("net_kg", 0) or 0)          # per-machine (already split by Go)
        slots = int(s.get("slots", 0) or 0)
        committed_kg = float(s.get("committed_kg", 0) or 0)  # per-machine picked creel
        if not dyelot and kg <= 0:
            continue          # nothing to pin, nothing to knit
        tid = f"INPROD_{order}_{machine}_{start}"
        vi_order_kg.setdefault(vi, {}).setdefault(order, 0.0)
        vi_order_kg[vi][order] += kg
        # Real machine placement — for FLUSH adjacency with new orders only.
        vi_machine_units.setdefault(vi, {}).setdefault(machine, []).append((start, tid, order))
        # CAPACITY: pool the in-production order's remaining net into ONE bucket
        # (not per-machine). Its creel is already mounted (re-counted via
        # committed_kg into the lot cap below), so charging per-machine whole-roll +
        # creel-up for an order spread across many machines would balloon its gross
        # far past the lot and make the forced pin infeasible (→ whole VI dropped).
        _inprod_pool_net[(vi, order)] = _inprod_pool_net.get((vi, order), 0.0) + kg
        if not dyelot:
            # Unreserved: no cones mounted anywhere, so keep a creel-up floor of one
            # creel width. Charged ONCE on the pooled bucket rather than per machine —
            # same reasoning as the pooling above, and it keeps the floor from
            # exploding to (machines × creel) for an order that only needs grams.
            if slots > _inprod_pool_slots.get((vi, order), 0):
                _inprod_pool_slots[(vi, order)] = slots
            continue
        pinned_lot.setdefault(vi, {})[order] = dyelot
        # Cap: re-count picked creel ONCE per (vi, order, machine, dyelot); summed
        # across machines = the order's total leftover creel on this lot.
        key = (vi, order, machine, dyelot)
        if key not in _committed_seen:
            _committed_seen.add(key)
            committed_by.setdefault(vi, {}).setdefault(dyelot, 0.0)
            committed_by[vi][dyelot] += committed_kg
        # Mounted cones — only where physically picked (committed_kg>0). Skipped
        # when reuse is off (Go's owner gate): the new order cannot reuse these
        # cones, so its creel-up floor stays and it mounts fresh — leaving
        # inprod_mounted empty makes the floor-drop below a no-op.
        if reuse_mounted_cones and committed_kg > 0 and slots > 0:
            md = inprod_mounted.setdefault(vi, {}).setdefault(machine, {})
            if slots > md.get(dyelot, 0):
                md[dyelot] = slots

    # One pooled capacity bucket per in-production (vi, order): whole-roll on the
    # POOLED remaining net, slots 0 for a pinned order (no creel-up floor — the cones
    # are already mounted and paid for via committed_kg) and its real creel width for
    # an unreserved one. A unique per-order machine key keeps it out of any real
    # machine's flush sequence.
    for (vi, order), net in _inprod_pool_net.items():
        vi_mo.setdefault(vi, {}).setdefault(f"__INPROD_POOL_{order}", {})[order] = [
            net, _inprod_pool_slots.get((vi, order), 0)
        ]

    # vi → sorted list of lots (copies — we mutate remaining_kg below).
    lots_by_vi: Dict[str, List[Dict[str, Any]]] = {}
    for d in dyelot_stock:
        lots_by_vi.setdefault(d["vi"], []).append(dict(d))
    # Re-count the in-production PICKED creel into its lot's capacity. Picked yarn
    # has left the warehouse (no longer in dyelot_stock.remaining_kg) but is on the
    # machine, and a co-lotted new order inherits it. Adding committed_by back
    # makes free-post-pick + creel ≈ free-pre-pick, so a re-schedule after picking
    # co-lots the same as before picking. A fully-committed lot with no free row
    # gets a synthetic one (0 free, then the creel is added). Any pinned lot with
    # no creel is still ensured present so the pin stays representable.
    def _ensure_lot(vi, dyelot):
        lots = lots_by_vi.setdefault(vi, [])
        for l in lots:
            if str(l.get("dyelot", "")) == dyelot:
                return l
        l = {"vi": vi, "dyelot": dyelot, "remaining_kg": 0.0,
             "packing_size": float((vi_packing or {}).get(vi, 0) or 0) or 1.0}
        lots.append(l)
        return l
    for vi, by_dyelot in committed_by.items():
        for dyelot, creel_kg in by_dyelot.items():
            lot = _ensure_lot(vi, dyelot)
            lot["remaining_kg"] = float(lot.get("remaining_kg", 0) or 0) + creel_kg
    for vi, by_order in pinned_lot.items():
        for dyelot in set(by_order.values()):
            _ensure_lot(vi, dyelot)
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
            # no existing dyelot to top up. Size the fresh lot by the reuse-aware
            # GROSS (whole-roll + creel-up), using the vi's default packing size
            # from Go; fall back to the net floor when no packing is known.
            order_g = {o: _g(order_kg[o]) for o in order_kg}
            # Roll size from Go; default 1 kg when a vi has no configured packing
            # (matches Go's stock-lot fallback) so we still report GROSS, not net.
            pk = float((vi_packing or {}).get(vi, 0) or 0)
            if pk <= 0:
                pk = 1.0
            gross_g = _vi_gross_demand_g(
                sorted(order_kg), order_g, [max(_g(pk), 1)], vi_mo.get(vi, {}),
                bool(config.get("enable_dyelot_per_machine_creel", True)))
            new_lot_kg = gross_g / _KG_SCALE
            shortage.append({"vi": vi, "demand_kg": round(new_lot_kg, 3),
                             "net_demand_kg": round(demand_kg, 3),
                             "stock_kg": 0.0,
                             "warehouse_kg": 0.0,
                             "committed_kg": 0.0,
                             "n_lots": 0,
                             "shortage_kind": "NO_STOCK",
                             "single_lot_deficit_kg": round(new_lot_kg, 3),
                             "topups": [],
                             "new_lot_kg": round(new_lot_kg, 3),
                             "new_lot_possible": True})   # a fresh lot IS the answer here
            logger.warning(f"🎨 VI {vi}: {len(order_kg)} order(s) consume it but "
                           f"dyelot_stock is empty → shortage.")
            continue

        # Creel-release model (default ON): only usable when EVERY unit of this VI
        # carries an end_time — a missing end means an unknown release moment, so we
        # fall back to the time-blind bound rather than guess (legacy payloads and the
        # synthetic fixtures land here → byte-identical behaviour).
        span_vi = (vi_machine_span.get(vi)
                   if (config.get("enable_dyelot_creel_release", True)
                       and vi not in vi_span_unknown)
                   else None)
        res = _solve_vi(vi, order_kg, lots, vi_machine_units.get(vi, {}),
                        vi_mo.get(vi, {}), solver_config,
                        pinned_lot_vi=pinned_lot.get(vi),
                        mounted_vi=inprod_mounted.get(vi),
                        committed_vi=committed_by.get(vi),
                        machine_span=span_vi)
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
            # whole-roll / creel-up inflation) — kept because a classifier run on
            # gross alone would call a fragmentation case a material buy near the
            # boundary (e.g. 3039 net 403 ≤ stock 423 but gross 440 > 423). The
            # net-vs-stock comparison is now made HERE (shortage_kind) instead of
            # downstream, which had no way to see warehouse vs picked creel.
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
            # stock_kg includes the in-production PICKED creel re-counted into the lot
            # (see _ensure_lot) — that yarn is on a machine, NOT purchasable stock, and
            # the capacity that actually binds is the free WAREHOUSE (cap − committed).
            # Classifying "enough material" on stock_kg alone reads a creel/roll
            # shortage as dyelot fragmentation, so the split is reported explicitly.
            committed_kg = sum((committed_by.get(vi) or {}).values())
            warehouse_kg = max(stock_kg - committed_kg, 0.0)
            kind = _classify_shortage(len(lots), demand_kg, gross_demand_kg,
                                      warehouse_kg, deficit_kg,
                                      topup_possible=res.get("topup_possible", True))
            shortage.append({"vi": vi,
                             "demand_kg": round(gross_demand_kg, 3),
                             "net_demand_kg": round(demand_kg, 3),
                             "stock_kg": round(stock_kg, 3),
                             "warehouse_kg": round(warehouse_kg, 3),
                             "committed_kg": round(committed_kg, 3),
                             "n_lots": len(lots),
                             "shortage_kind": kind,
                             "single_lot_deficit_kg": round(deficit_kg, 3),
                             "topups": topups,
                             "new_lot_kg": round(new_lot_kg, 3),
                             # False ⇒ new_lot_kg 0 means "no fresh-lot option exists"
                             # (a pin holds an order on an overflowing lot), NOT
                             # "importing 0 kg clears it".
                             "new_lot_possible": bool(res.get("new_lot_possible", True))})
            logger.warning(
                f"🎨 VI {vi} shortage [{kind}]: {len(res['unassigned'])} order(s) "
                f"unassigned; net {demand_kg:.3f} / gross {gross_demand_kg:.3f} kg vs "
                f"warehouse {warehouse_kg:.3f} kg (+{committed_kg:.3f} kg picked creel) "
                f"on {len(lots)} lot(s) → top-up {deficit_kg:.3f} kg "
                f"or new lot {new_lot_kg:.3f} kg."
            )
            if deficit_kg <= 0 and new_lot_kg <= 0:
                logger.warning(
                    f"🎨 VI {vi}: orders dropped but BOTH remedies price at 0 kg — "
                    + ("no top-up solve returned a solution (budget, or a pin holds an "
                       "order on a lot that cannot host it)."
                       if kind == "REMEDY_UNKNOWN" else
                       "the main solve did not prove its optimum (raise "
                       "dyelot_max_deterministic_time).")
                    + " NOT a procurement signal."
                )

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


def _classify_shortage(n_lots, net_kg, gross_kg, warehouse_kg, deficit_kg,
                       topup_possible=True) -> str:
    """WHY this VI has unassignable orders — computed here, where the capacity model
    lives, so the consumer never has to re-derive it from net-vs-stock:

      NO_STOCK         — the VI has no dyelot row at all.
      MATERIAL_SHORT   — net (consumed-into-product) demand alone exceeds the free
                         warehouse: a real buy, whatever the lot layout.
      CREEL_ROLL_SHORT — net fits but GROSS (whole-roll + creel-up MinSlots) does not.
                         Nothing to "gather": the yarn is short because every parallel
                         machine must mount its own cones.
      FRAGMENTED       — gross fits the warehouse in TOTAL but not on ONE lot (≥2 lots
                         needed).  Only reachable with ≥2 lots — with a single lot there
                         is nothing to consolidate onto, so this is never emitted there.
      REMEDY_UNKNOWN   — no top-up solve returned a solution at all (its budget ran out,
                         or an in-production pin holds an order on a lot that cannot host
                         it), so the priced remedy is not trustworthy — not a buy signal.
      UNPROVEN         — none of the above bind, i.e. the main solve stranded orders it
                         could not prove unplaceable (budget), not a procurement signal.
    """
    eps = 1e-9
    if n_lots == 0:
        return "NO_STOCK"
    if net_kg > warehouse_kg + eps:
        return "MATERIAL_SHORT"
    if gross_kg > warehouse_kg + eps:
        return "CREEL_ROLL_SHORT"
    if n_lots >= 2 and deficit_kg > eps:
        return "FRAGMENTED"
    if not topup_possible:
        return "REMEDY_UNKNOWN"
    return "UNPROVEN"


def _vi_gross_demand_g(orders, demand_g, pk_g, machine_order, per_machine=True) -> int:
    """GROSS (grams) the VI physically pulls, for the dyelot_shortage report.

    per_machine=True (default — CORRECT physical model): máy chạy SONG SONG mỗi
    máy mount cone RIÊNG, không chia chung được. On each machine the orders sharing
    it draw from ONE creel, so that machine pulls max(ceil(Σnet_m / pk), max_slots_m)
    rolls (the creel-up MinSlots floor); summed across machines (each mounts its own
    cones). Orders with no machine breakdown fall back to whole-order rolls.

    per_machine=False (legacy pooled model — kept for A/B via
    enable_dyelot_per_machine_creel): net POOLED across all machines, ceil(Σnet/pk)·pk.
    Assumes residual cones return to a shared pool — UNDER-reserves when machines run
    concurrently (each truly needs its own slots cones), reporting false "enough"."""
    pk = min(pk_g)
    if not per_machine:
        net = sum(demand_g[o] for o in orders)
        return ((net + pk - 1) // pk) * pk
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


def _cap_slack_g(demand_g, pk_g, machine_order, per_machine=True) -> int:
    """Guaranteed-sufficient upper bound (grams) on the extra capacity any single
    lot could ever need / be charged — safely bounds expandable-bin Vars and the
    per-(machine,lot) roll Vars.

    per_machine=True: net total + one roll per (machine,order) run + the creel-up of
    every run (the per-machine roll Vars can each climb to its own slots floor).
    per_machine=False (pooled): Σnet + one roll suffices."""
    if not per_machine:
        return sum(demand_g.values()) + max(pk_g)
    runs = 0
    tslots = 0
    for m in machine_order:
        for _o, run in machine_order[m].items():
            runs += 1
            tslots += int(run[1])
    return sum(demand_g.values()) + (runs + tslots) * max(pk_g)


def _add_roll_capacity(model, x, orders, n_lots, demand_g, cap_g, pk_g,
                       machine_order, extra=None, per_machine=True,
                       mounted=None, lot_names=None,
                       committed_g=None, inprod_pool=True,
                       machine_span=None) -> None:
    """Whole-roll capacity per dyelot.

    per_machine=True (default — CORRECT): per (machine, lot) the orders of a lot
    SHARE the creel on that machine (residual feeds the next order), so the rolls
    mounted on machine m for lot d cover the SUM of those orders' net AND the widest
    creel (max slots). Rolls are summed ACROSS machines (each machine mounts its own
    cones from the lot) and bounded by the lot stock. So a VI knit on N machines in
    parallel charges ≈ N × its creel — the physically-real reservation.

    machine_span={machine: (start, end)} (optional) turns on the CREEL-RELEASE model:
    a machine holds its cones only while it runs the VI, and Go returns the leftover
    to the free pool when the machine finishes it (chrono_commit: RELEASE_POOL at
    is_last_for_machine_vi, re-consumed by a later machine as SUCCESSOR_POOL).  So the
    stock bound is the PEAK CONCURRENT reservation (checked at every span start), not
    the time-blind sum over all machines, plus net conservation (consumed yarn never
    comes back).  Without it, a VI knit on N machines that never overlap still charges
    N × creel and manufactures a shortage on a VI whose net demand is grams.
    Machines with no span (in-production pooled buckets, legacy payloads with no
    end_time) are charged at EVERY time point — conservative, unchanged behaviour.

    per_machine=False (legacy pooled): one rolls Var per lot, ceil(Σnet/pk[d]),
    NO per-machine rounding and NO creel-up floor (assumes residual pools)."""
    bump = (_cap_slack_g(demand_g, pk_g, machine_order, per_machine)
            if extra is not None else 0)
    if not per_machine:
        for d in range(n_lots):
            cap = cap_g[d] + (extra[d] if extra is not None else 0)
            ub_rolls = (cap_g[d] + bump) // pk_g[d] + 1
            rolls = model.NewIntVar(0, ub_rolls, f"rolls_{d}")
            # Enough whole rolls to cover the net pooled onto this lot …
            model.Add(rolls * pk_g[d] >= sum(demand_g[o] * x[o][d] for o in orders))
            # … and those rolls must fit the lot's (expandable) stock.
            model.Add(rolls * pk_g[d] <= cap)
        return
    machines = sorted(machine_order, key=str)
    with_machine = {o for m in machines for o in machine_order[m] if o in demand_g}
    missing = [o for o in orders if o not in with_machine]
    for d in range(n_lots):
        ub = (cap_g[d] + bump) // pk_g[d] + 1
        # An IN-PRODUCTION lot (its picked creel re-counted into cap via
        # committed_g[d] > 0) has that creel physically MOUNTED and pooling across
        # the machines it runs on: the residual of one run feeds the next, so a
        # co-lotting order draws from it instead of mounting fresh cones. Charging
        # each machine its own whole-roll rounding of NET then over-reserves by
        # ~(machines × ½ roll) and wrongly spills co-lottable orders to a second
        # lot. So for such a lot the NET is pooled (one lot-level whole-roll bound,
        # credited by the mounted creel), while the per-machine creel-up floor is
        # KEPT — a run needing WIDER creel than mounted still charges the extra
        # cones — and fresh rolls are bounded by WAREHOUSE (cap − committed) so the
        # mounted creel is never double-counted. Non-in-production lots keep the
        # strict per-machine model (parallel machines each mount their own cones).
        cmt = (committed_g[d] if committed_g else 0)
        pool_net = bool(inprod_pool and cmt > 0)
        lot_rolls = []
        pooled_net_terms = []
        all_net_terms = []        # every machine's net — conservation in release mode
        timed_rolls = []          # (rolls Var, (start, end)) for span-known machines
        untimed_rolls = []        # charged at EVERY time point (no span known)
        for m in machines:
            mo = machine_order[m]
            oz = [o for o in orders if o in mo]
            if not oz:
                continue
            r = model.NewIntVar(0, ub, f"r_{d}_{m}")
            net_m = sum(_g(mo[o][0]) * x[o][d] for o in oz)
            all_net_terms.append(net_m)
            span = (machine_span or {}).get(m)
            if span is not None:
                timed_rolls.append((r, span))
            else:
                untimed_rolls.append(r)
            if pool_net:
                pooled_net_terms.append(net_m)   # net pools at lot level (below)
            else:
                model.Add(r * pk_g[d] >= net_m)   # per-machine whole-roll of net
            # Creel-up floor per order. If an in-production order already mounts
            # this lot's cones on machine m (mounted[m][lot_name]), a co-lotting
            # order reuses them → its floor drops by the mounted count (kept even in
            # the pooled-net path: a wider-creel run must still mount extra cones).
            mnt = 0
            if mounted and lot_names is not None:
                mnt = mounted.get(m, {}).get(lot_names[d], 0)
            for o in oz:
                s = int(mo[o][1]) - mnt
                if s > 0:
                    model.Add(r * pk_g[d] >= s * pk_g[d] * x[o][d])  # creel floor
            lot_rolls.append(r)
        if missing:
            r = model.NewIntVar(0, ub, f"rm_{d}")
            miss_net = sum(demand_g[o] * x[o][d] for o in missing)
            all_net_terms.append(miss_net)
            untimed_rolls.append(r)      # no machine → no span → always charged
            if pool_net:
                pooled_net_terms.append(miss_net)
            else:
                model.Add(r * pk_g[d] >= miss_net)
            lot_rolls.append(r)
        if lot_rolls:
            extra_d = (extra[d] if extra is not None else 0)
            if pool_net:
                # Mounted creel (committed) covers net poolingly; fresh rolls cover
                # only the remainder and come from WAREHOUSE (cap − committed).
                model.Add(sum(lot_rolls) * pk_g[d] + cmt >= sum(pooled_net_terms))
                # NOTE: `extra_d` may be a CP-SAT Var (expandable-bin remedy models),
                # so the free-warehouse floor must be clamped on the CONSTANT part —
                # `max(cap-cmt, 0) + extra_d` — never `if warehouse > 0` (evaluating a
                # BoundedLinearExpression as a bool raises NotImplementedError).
                bound = max(cap_g[d] - cmt, 0) + extra_d
            else:
                bound = cap_g[d] + extra_d
            if timed_rolls:
                # CREEL-RELEASE model: bound the PEAK concurrent reservation. Checking
                # every span start is exact for interval overlap (a maximal overlap set
                # always begins at some span's start).
                base = sum(untimed_rolls)
                for t in sorted({s for _r, (s, _e) in timed_rolls}):
                    act = [r for r, (s, e) in timed_rolls if s <= t < max(e, s + 1)]
                    model.Add((sum(act) + base) * pk_g[d] <= bound)
                # Consumed yarn never returns to the pool: total net drawn from this lot
                # must still fit its TOTAL stock (warehouse + already-mounted creel).
                model.Add(sum(all_net_terms) <= cap_g[d] + extra_d)
            else:
                model.Add(sum(lot_rolls) * pk_g[d] <= bound)


def _solve_vi(
    vi: str,
    order_kg: Dict[str, float],
    lots: List[Dict[str, Any]],
    machine_units: Dict[Any, List[Tuple[int, str, str]]],
    machine_order: Dict[Any, Dict[str, List[float]]],
    config: Dict[str, Any],
    pinned_lot_vi: Optional[Dict[str, str]] = None,
    mounted_vi: Optional[Dict[Any, Dict[str, int]]] = None,
    committed_vi: Optional[Dict[str, float]] = None,
    machine_span: Optional[Dict[Any, Tuple[int, int]]] = None,
    forced_assigned: Optional[set] = None,
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
    STRICT small-first (enable_dyelot_small_order_priority, default ON) sits ABOVE the
    whole objective: when the solve below strands anything, the served set is recomputed
    by walking the orders smallest→largest (_lex_small_first_forced) and the model is
    re-solved with that set pinned.  The business rule is "a small order is completed
    whenever it CAN be, big orders take what is left", which the count tier alone does
    not give: maximising the COUNT still sacrifices one small order to serve two larger
    ones.  The two agree whenever orders mount comparable creels (measured on 134 orders
    of 0.5…67 kg: both serve the same 15 smallest), and diverge when a small order has a
    disproportionate creel footprint.

    Objective (lexicographic via data-bounded weights, Maximize):
      (1) feasibility       — maximize assigned orders.  COUNT-based: every order weighs
                              the same regardless of kg, so when a lot cannot host all of
                              them the solver drops the FEWEST orders — i.e. the big one
                              goes before several small ones.  Bounded above by the strict
                              small-first pass, which may deliberately serve FEWER.
      (2) small-first       — +Σ assigned[o]·small_w[o], small_w = size RANK (smallest
                              order gets the largest weight).  Tier 1 only fixes HOW MANY
                              orders are served; among the equal-count solutions the rest
                              of the objective was blind to order size, so which order got
                              stranded was decided by the order CODE (measured: renaming
                              the 40 kg order flipped the victim from it to a 0.5 kg order
                              in 3 of 5 namings).  This makes "small orders get stock
                              first" explicit instead of accidental.  Rank, not grams, so
                              the weight cascade below cannot overflow on a 5-tonne VI.
                              Flag enable_dyelot_small_order_first (default ON); OFF
                              restores the old byte-identical objective.
      (3) min flush         — fewest cohort cuts (Σ cut)
      (4) min lots opened   — Σ used[d]; consolidates a VI onto as few lots as
                              possible WITHOUT preferring small lots, so multi-order
                              demand lands on the lot that holds the most instead of
                              stranding on a nearly-full small lot while a big lot
                              sits idle (the production over-concentration bug).
      (5) prefer-buffer     — +Σ used[d]·buffer_g[d]; among equal-lot-count, equal-
                              flush solutions, prefer a lot that carries Buffer-bin
                              yarn. Staged buffer consumed here issues no fresh main-
                              warehouse pull, so drain it before raw — but BELOW min-
                              lots, so buffer never justifies opening an extra lot.
      (6) drain-small       — Σ used[d]·remaining_g[d] as a pure TIE-BREAK below the
                              buffer tier: among equal-lot-count, equal-flush, equal-
                              buffer solutions, consume the nearly-empty lot first.
                              Demoted from tier 3 (where it caused over-concentration)
                              so it can no longer drive cramming.
    """
    orders = sorted(order_kg)
    n_lots = len(lots)
    demand_g = {o: _g(order_kg[o]) for o in orders}
    cap_g = [_g(l.get("remaining_kg", 0)) for l in lots]
    remaining_g = list(cap_g)
    # Per-lot buffer stock (grams) — the portion of remaining_kg staged in the
    # Buffer bin. Consuming it issues no fresh main-warehouse pull, so we bias lot
    # selection toward lots that carry buffer (tier 4, below min-lots). Missing on
    # synthetic/committed lots → 0.
    buffer_g = [_g(l.get("buffer_kg", 0)) for l in lots]
    # Per-lot roll size in grams (≥1 so the ceil is well-defined even if
    # packing_size is missing / rounds to 0).
    pk_g = [max(_g(l.get("packing_size", 0)), 1) for l in lots]

    # Per-machine creel reservation (default) vs legacy pooled — see _add_roll_capacity.
    per_machine = bool(config.get("enable_dyelot_per_machine_creel", True))

    # Reuse-aware GROSS (C) — see _add_roll_capacity / _vi_gross_demand_g.
    gross_demand_g = _vi_gross_demand_g(orders, demand_g, pk_g, machine_order, per_machine)

    model = cp_model.CpModel()

    x = {o: [model.NewBoolVar(f"x_{vi}_{o}_{d}") for d in range(n_lots)] for o in orders}
    una = {o: model.NewBoolVar(f"una_{vi}_{o}") for o in orders}
    assigned = {o: model.NewBoolVar(f"asg_{vi}_{o}") for o in orders}
    for o in orders:
        model.Add(sum(x[o]) + una[o] == 1)        # 1 dyelot per order per VI
        model.Add(assigned[o] == 1 - una[o])

    # Pin in-production (converted/pinned) orders to their committed dye lot — the
    # solver must not move an order already being knitted onto a different lot.
    # Forcing x[o][d]=1 also forces una[o]=0/assigned[o]=1 (via exactly_one). The
    # min-flush objective then pulls a sharing new order onto the same lot when its
    # gross fits the lot's free capacity. The lot is guaranteed present in `lots`
    # (allocate_dyelots adds a synthetic row when fully committed), so d is found.
    if pinned_lot_vi:
        lot_index = {str(lots[d].get("dyelot", "")): d for d in range(n_lots)}
        for o in orders:
            want = pinned_lot_vi.get(o)
            if want is None:
                continue
            d = lot_index.get(str(want))
            if d is not None:
                model.Add(x[o][d] == 1)

    # STRICT small-first re-solve: the must-serve set computed by the lexicographic
    # walk below is imposed as HARD constraints, so the objective can no longer trade a
    # small order away for a larger count. Every other tier still optimises inside it.
    if forced_assigned:
        for o in orders:
            if o in forced_assigned:
                model.Add(una[o] == 0)

    # Capacity — reuse-aware per-(machine, lot) roll model (C). mounted lets a
    # co-lotting order reuse an in-production order's already-mounted cones
    # (creel-up floor drops on those machines).
    lot_names = [str(lots[d].get("dyelot", "")) for d in range(n_lots)]
    # Per-lot picked-creel kg (grams) re-counted into cap — drives the pooled-net
    # capacity for in-production lots (creel physically pools across their machines).
    committed_g = ([_g((committed_vi or {}).get(lot_names[d], 0)) for d in range(n_lots)]
                   if per_machine else None)
    # Semantics shared by the main solve AND the two remedy solves below — a remedy
    # model built on a LOOSER capacity answers "add 0 kg" to a shortage the main solve
    # really found (measured 2026-07-31 on AY02-DKG350: cap 55.971 incl. picked creel
    # instead of the 22 kg warehouse → deficit 0.0 instead of 4.0).
    cap_kw = dict(per_machine=per_machine, mounted=mounted_vi, lot_names=lot_names,
                  committed_g=committed_g,
                  inprod_pool=bool(config.get("enable_dyelot_inprod_pool", True)),
                  machine_span=machine_span)
    _add_roll_capacity(model, x, orders, n_lots, demand_g, cap_g, pk_g, machine_order,
                       **cap_kw)

    # used[d]: 1 iff lot d carries any order.  Drives tier-3 (min lots) + tier-4/5.
    # EXACT indicator (both bounds): tier 4 rewards buffer via +used[d]·buffer_g, a
    # positive coefficient, so used[d] must not be free to float to 1 on an empty
    # buffer lot to harvest the reward — the upper bound ties it to real assignment.
    # (Tiers 3/5 push used down, so the extra bound is slack there; it only matters
    # once a positive term exists.)
    used = [model.NewBoolVar(f"used_{vi}_{d}") for d in range(n_lots)]
    for d in range(n_lots):
        for o in orders:
            model.Add(used[d] >= x[o][d])
        model.Add(used[d] <= sum(x[o][d] for o in orders))

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
    #   (1) +assigned  (2) +small-first  (3) −flush  (4) −lots opened
    #   (5) +prefer-buffer  (6) −drain-small
    # Each tier's weight must dominate the MAX possible value of every lower tier
    # summed, so the optimum is exactly lexicographic.
    assigned_sum = sum(assigned[o] for o in orders)
    flush_sum = sum(cut for (cut, *_rest) in flush_terms)   # number of cohort cuts
    lots_sum = sum(used)                                    # distinct lots opened
    buf_sum = sum(used[d] * buffer_g[d] for d in range(n_lots))     # prefer-buffer
    frag_sum = sum(used[d] * remaining_g[d] for d in range(n_lots))  # drain-small

    # small-first: keep the SMALL orders when the lot cannot host everyone. Weight is
    # the size RANK (smallest order → largest weight, ties by order id so it stays
    # deterministic), NOT grams: a grams weight would push k_assign past int64 on a
    # multi-tonne VI (max_small ≈ Σnet_g), while the rank sum is ≤ n²/2.
    if config.get("enable_dyelot_small_order_first", True):
        by_size = sorted(orders, key=lambda o: (demand_g[o], o))
        small_w = {o: len(orders) - i for i, o in enumerate(by_size)}
    else:
        small_w = {o: 0 for o in orders}                    # OFF → tier vanishes
    small_sum = sum(assigned[o] * small_w[o] for o in orders)

    n_pairs = len(flush_terms)                              # max value of flush_sum
    max_frag = sum(remaining_g)                             # max value of frag_sum
    max_buf = sum(buffer_g)                                 # max value of buf_sum
    max_small = sum(small_w.values())                       # max value of small_sum
    k_frag = 1
    # Draining one buffer-gram must beat any drain-small gain (buffer is staged;
    # consuming it avoids a fresh main-warehouse pull — worth more than emptying a
    # small raw lot). Below min-lots, so it never opens an extra lot just for buffer.
    k_buf = k_frag * max_frag + 1
    # Opening one fewer lot must always beat any (buffer + drain-small) gain.
    k_lots = k_buf * max_buf + k_frag * max_frag + 1
    # One fewer flush must beat the worst (lots + buffer + drain-small) combined.
    k_flush = k_lots * n_lots + k_buf * max_buf + k_frag * max_frag + 1
    # Serving a SMALLER order must beat everything below it (flush/lots/buffer/drain) —
    # a rank step is worth more than any waste it causes. Still strictly BELOW tier 1,
    # so small-first never trades away the number of orders served.
    k_small = k_flush * n_pairs + k_lots * n_lots + k_buf * max_buf + k_frag * max_frag + 1
    # Assigning one more order must beat the worst lower-tier sum — so feasibility is
    # never traded for a lower tier (also why a VI with no flush pairs still assigns).
    # max_small = 0 when small-first is OFF → the old k_assign, byte-identical.
    k_assign = (k_small * max_small + k_flush * n_pairs + k_lots * n_lots
                + k_buf * max_buf + k_frag * max_frag + 1)
    model.Maximize(
        k_assign * assigned_sum
        + k_small * small_sum
        - k_flush * flush_sum
        - k_lots * lots_sum
        + k_buf * buf_sum
        - k_frag * frag_sum
    )

    # gap=0: tier-1 is feasibility (max assigned orders); the default 1% gap on a
    # many-order VI lets CP-SAT stop with one order stranded (1/N < 1%), which
    # then reports a FALSE shortage.  Feasibility must not trade against the gap.
    solver = make_solver(config, relative_gap=0.0)
    status = solver.Solve(model)

    out: Dict[str, Any] = {"assignments": [], "flush_points": [],
                           "unassigned": [], "gross_demand_g": gross_demand_g}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Infeasible ONLY because in-production pins remove the unassigned escape
        # (x[o][pin]=1 ⇒ assigned[o]=1): the pinned lot cannot host its own pinned
        # demand + any co-lot under the current (reduced) stock, so no feasible
        # packing exists. Report every order unassigned, but STILL price the top-up —
        # the remedy solves add capacity until the pins fit, so they return a real
        # "add X kg to <pinned lot>" deficit. Bailing here (the old behaviour) left
        # deficit 0 / new_lot 0, which downstream reads as "dropped but nothing to
        # buy" — a misleading procurement signal.
        logger.warning(f"🎨 VI {vi}: main solve {solver.StatusName(status)} "
                       f"(in-production pins exceed lot capacity) — pricing top-up remedy.")
        for o in orders:
            out["unassigned"].append({"order": o, "vi": vi,
                                      "reason": f"solver_{solver.StatusName(status)}"})
        _price_shortage_remedies(orders, n_lots, demand_g, cap_g, pk_g, machine_order,
                                 config, cap_kw, pinned_lot_vi, lot_names, out)
        return out

    # STRICT small-first: the objective above maximises the COUNT of served orders, so
    # it still sacrifices one small order to serve two larger ones. The business rule is
    # that a small order must be completed whenever it CAN be — big orders take what is
    # left — so when anything was dropped, recompute the served set by walking the
    # orders smallest→largest and re-solve with that set pinned. Only runs on a VI that
    # actually has an unplaceable order, and only once (the recursion passes
    # forced_assigned, which short-circuits this block).
    if (forced_assigned is None
            and config.get("enable_dyelot_small_order_priority", True)
            and any(solver.Value(una[o]) for o in orders)):
        must = _lex_small_first_forced(orders, n_lots, demand_g, cap_g, pk_g,
                                       machine_order, config, cap_kw,
                                       pinned_lot_vi, lot_names)
        served = {o for o in orders if not solver.Value(una[o])}
        if must != served:
            dropped_small = sorted(must - served, key=lambda o: (demand_g[o], o))
            logger.info(
                f"🎨 VI {vi}: strict small-first re-solve — count-max stranded "
                f"{len(dropped_small)} order(s) that CAN be served "
                f"({', '.join(dropped_small[:5])}{'…' if len(dropped_small) > 5 else ''}); "
                f"serving {len(must)} order(s) instead of {len(served)}."
            )
            return _solve_vi(vi, order_kg, lots, machine_units, machine_order, config,
                             pinned_lot_vi=pinned_lot_vi, mounted_vi=mounted_vi,
                             committed_vi=committed_vi, machine_span=machine_span,
                             forced_assigned=must)

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
        _price_shortage_remedies(orders, n_lots, demand_g, cap_g, pk_g, machine_order,
                                 config, cap_kw, pinned_lot_vi, lot_names, out)
    return out


def _lex_small_first_forced(orders, n_lots, demand_g, cap_g, pk_g, machine_order,
                            config, cap_kw, pinned_lot_vi, lot_names):
    """The set of orders to serve under STRICT small-first priority: "a small order is
    always completed if it can be, whatever it costs the bigger ones".

    Walks the orders smallest→largest (ties by id, so it is deterministic) and keeps
    each one whenever the accumulated must-serve set is still feasible. That is true
    lexicographic priority — order k is only sacrificed if serving it is impossible
    given every smaller order, never because it would cost two larger ones.

    Why a walk and not another objective tier: the tier-2 weight is a rank SUM, so it
    still trades the single smallest order for two lower-ranked ones (weights n vs
    (n−1)+(n−2)); making the tiers truly lexicographic needs exponential weights
    (2^rank), which overflows past ~60 orders. n small feasibility solves do not.

    Cost: one tiny feasibility solve per order, and only on a VI that actually has an
    unplaceable order — a VI where everything fits never reaches this path."""
    kw = dict(cap_kw or {})
    per_machine = bool(kw.pop("per_machine",
                              config.get("enable_dyelot_per_machine_creel", True)))
    must: List[str] = []
    for o in sorted(orders, key=lambda oo: (demand_g[oo], oo)):
        candidate = must + [o]
        m = cp_model.CpModel()
        x = {oo: [m.NewBoolVar(f"lx_{oo}_{d}") for d in range(n_lots)] for oo in orders}
        una = {oo: m.NewBoolVar(f"lu_{oo}") for oo in orders}
        for oo in orders:
            m.Add(sum(x[oo]) + una[oo] == 1)
        _pin_orders(m, x, orders, pinned_lot_vi, lot_names)
        _add_roll_capacity(m, x, orders, n_lots, demand_g, cap_g, pk_g,
                           machine_order, per_machine=per_machine, **kw)
        for oo in candidate:
            m.Add(una[oo] == 0)
        s = make_solver(config, relative_gap=0.0)
        if s.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            must = candidate
    return set(must)


def _pin_orders(model, x, orders, pinned_lot_vi, lot_names) -> None:
    """Force in-production orders onto their committed lot (same as the main solve).
    A remedy model without the pins is free to relocate an order already on the
    machine, so it prices a move that procurement cannot actually buy."""
    if not pinned_lot_vi or lot_names is None:
        return
    idx = {str(lot_names[d]): d for d in range(len(lot_names))}
    for o in orders:
        want = pinned_lot_vi.get(o)
        if want is None:
            continue
        d = idx.get(str(want))
        if d is not None:
            model.Add(x[o][d] == 1)


def _topup_one_lot_g(d_fill, orders, n_lots, demand_g, cap_g, pk_g,
                     machine_order, config, cap_kw=None,
                     pinned_lot_vi=None, lot_names=None) -> Optional[int]:
    """Minimal extra capacity (grams) added to dyelot `d_fill` ALONE (all other
    lots fixed at their current stock) so that a one-dyelot-per-order assignment of
    ALL orders becomes feasible. This is a STANDALONE option — dyelots are never
    merged, so a real top-up replenishes exactly one lot — not a spread across
    several. Returns None if even an unbounded single-lot top-up can't host the
    orders (a pin forces an order onto a DIFFERENT, non-growing lot)."""
    kw = dict(cap_kw or {})
    per_machine = bool(kw.pop("per_machine",
                              config.get("enable_dyelot_per_machine_creel", True)))
    m = cp_model.CpModel()
    x = {o: [m.NewBoolVar(f"tx_{o}_{d}") for d in range(n_lots)] for o in orders}
    ub = _cap_slack_g(demand_g, pk_g, machine_order, per_machine)
    fill = m.NewIntVar(0, ub, "fill")
    extra = [fill if d == d_fill else 0 for d in range(n_lots)]  # only d_fill grows
    for o in orders:
        m.Add(sum(x[o]) == 1)  # every order must land on exactly one lot
    _pin_orders(m, x, orders, pinned_lot_vi, lot_names)
    _add_roll_capacity(m, x, orders, n_lots, demand_g, cap_g, pk_g,
                       machine_order, extra=extra, per_machine=per_machine, **kw)
    m.Minimize(fill)
    s = make_solver(config, relative_gap=0.0)   # exact minimal top-up, not within 1%
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return int(s.Value(fill))


def _new_lot_g(orders, n_lots, demand_g, cap_g, pk_g,
               machine_order, config, cap_kw=None,
               pinned_lot_vi=None, lot_names=None) -> int:
    """Minimal size (grams) of ONE fresh dyelot which, added alongside the
    existing lots, makes a one-dyelot-per-order assignment of ALL orders feasible.
    Same expandable-bin model but the free capacity lives ONLY on a new
    (n_lots+1)-th lot — the 'import a new dyelot of this size' answer. The new lot
    inherits the smallest existing roll size (whole-roll charge stays consistent)."""
    kw = dict(cap_kw or {})
    per_machine = bool(kw.pop("per_machine",
                              config.get("enable_dyelot_per_machine_creel", True)))
    nl = n_lots + 1
    cap2 = cap_g + [0]
    pk2 = pk_g + [min(pk_g)]
    # The fresh lot carries no picked creel and no mounted cones — extend the
    # per-lot vectors so the shared semantics stay index-aligned.
    if kw.get("committed_g") is not None:
        kw["committed_g"] = list(kw["committed_g"]) + [0]
    names2 = (list(lot_names) + ["__NEW_LOT__"]) if lot_names is not None else None
    if kw.get("lot_names") is not None:
        kw["lot_names"] = names2
    m = cp_model.CpModel()
    x = {o: [m.NewBoolVar(f"nx_{o}_{d}") for d in range(nl)] for o in orders}
    ub = _cap_slack_g(demand_g, pk2, machine_order, per_machine)  # safe bound (gross ≥ Σnet)
    new = m.NewIntVar(0, ub, "new_lot")
    extra = [0] * n_lots + [new]          # only the fresh lot is expandable
    for o in orders:
        m.Add(sum(x[o]) == 1)
    _pin_orders(m, x, orders, pinned_lot_vi, names2)
    _add_roll_capacity(m, x, orders, nl, demand_g, cap2, pk2,
                       machine_order, extra=extra, per_machine=per_machine, **kw)
    m.Minimize(new)
    s = make_solver(config, relative_gap=0.0)   # exact minimal fresh-lot size, not within 1%
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # A fresh lot CANNOT clear this shortage at any size — an in-production pin
        # holds an order on an existing lot that overflows, and a new lot cannot take
        # it (dyelots are never mixed within an order). Returning 0 here would read
        # downstream as "import 0 kg", i.e. nothing to buy; None says "no such
        # option" so the caller can report it as unavailable instead of free.
        return None
    return int(s.Value(new))


def _price_shortage_remedies(orders, n_lots, demand_g, cap_g, pk_g, machine_order,
                             config, cap_kw, pinned_lot_vi, lot_names, out) -> None:
    """Attach the honest procurement remedies onto `out`, under the SAME reuse-aware
    capacity + in-production pins as the main solve (a looser model silently answers
    "add 0 kg"). Shared by BOTH shortage paths so each surfaces a real number:
      - the main solve was FEASIBLE but stranded an order for capacity, and
      - the main solve was INFEASIBLE because the in-production pins alone overflow
        their lot (the pins remove the unassigned escape, so no packing exists) —
        the path that used to bail with deficit 0 / new_lot 0.
    Dyelots are never merged, so each option replenishes exactly ONE lot:
      topups_g[d] — extra kg to add to dyelot d ALONE (None when it can't be the
                    single sink, e.g. a pin forces an order onto a different lot);
      deficit_g   — the CHEAPEST single-lot top-up (min over topups_g);
      new_lot_g   — size of ONE fresh dyelot to import instead (None when a fresh lot
                    cannot clear it at any size, i.e. a pin holds an order on an
                    existing overflowing lot; 0 would read as "import 0 kg").
    """
    topups_g = [_topup_one_lot_g(d, orders, n_lots, demand_g, cap_g, pk_g,
                                 machine_order, config, cap_kw=cap_kw,
                                 pinned_lot_vi=pinned_lot_vi, lot_names=lot_names)
                for d in range(n_lots)]
    out["topups_g"] = topups_g
    feasible = [g for g in topups_g if g is not None]
    out["deficit_g"] = min(feasible) if feasible else 0
    out["topup_possible"] = bool(feasible)
    new_lot_g = _new_lot_g(
        orders, n_lots, demand_g, cap_g, pk_g, machine_order, config,
        cap_kw=cap_kw, pinned_lot_vi=pinned_lot_vi, lot_names=lot_names)
    out["new_lot_possible"] = new_lot_g is not None
    out["new_lot_g"] = new_lot_g or 0
