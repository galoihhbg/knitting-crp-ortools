"""
Washing machine-consolidation tests.

Root cause under fix: the washing objective penalises active SLOTS and rewards
early batch_starts/batch_ends, but has NO per-machine cost.  So when ≥2 batch
slots are needed the solver runs them PARALLEL on 2 machines (both finish early
→ cheaper end-terms) instead of SEQUENTIAL on 1 machine — wasting electricity
when there is no deadline pressure.

Fix = add `machine_ever_used[m] × machine_w` (machine_w = 500, flat) to the
washing objective so consolidation onto fewer machines is preferred UNLESS a
deadline forces parallelism (machine_w ≪ lateness coeff) or the legitimate
end-time saving of a large group exceeds the per-machine cost.

RED tests (E4, E3c) assert the DESIRED post-fix outcome and FAIL until the
machine_w term is added.  GUARD tests assert behaviour that must NOT regress
(large groups stay parallel, deadline-forced parallel stays, physically-disjoint
machines stay, clean single-slot consolidation keeps).

Mutation-check for every test is documented inline.
"""
from typing import Any, Dict, List

from app.engine.model import Engine


def _task(
    tid: str,
    dur: int,
    due: int,
    rids: List[str],
    qty: float,
    lb: int = 0,
) -> Dict[str, Any]:
    return {
        "task_id": tid,
        "original_order_id": tid,
        "group_id": tid,
        "operation": "washing",
        "qty": float(qty),
        "total_qty": float(qty),
        "priority": 3,
        "original_depends_on": [],
        "final_depends_on": [],
        "start_after_min": lb,
        "due_at_min": due,
        "duration": dur,
        "is_slice": False,
        "parent_task_id": "",
        "internal_dep": "",
        "slice_index": 0,
        "is_batch": False,
        "sub_tasks": None,
        "design_item_id": "",
        "color_config": "",
        "color": "red",
        "substance": "cotton",
        "compatible_resource_ids": rids,
        "sub_task_completion_offsets": None,
        "WaitOffsets": None,
        "is_pinned": False,
        "pinned_machine_id": None,
        "pinned_start_time": None,
        "pinned_end_time": None,
        "demand": 1,
        "material_demands": {},
    }


def _machine(m: str) -> Dict[str, Any]:
    return {"id": m, "design_item_id": "D1", "color_config": "Black"}


def _resource(m: str) -> Dict[str, Any]:
    return {
        "id": m,
        "type": "serial",
        "capacity": 1,
        "operation": "washing",
        "unavailability": [],
        "design_item_id": "",
        "color_config": "",
        "available_at_min": 0,
    }


def _config(**overrides) -> Dict[str, Any]:
    cfg = {
        "horizon_minutes": 5000,
        "max_search_time": 20,
        "setup_time_minutes": 0,
        "max_factory_machines": 5,
        "random_seed": 42,
        "num_search_workers": 1,
        "washing_batch_capacity": 3,
    }
    cfg.update(overrides)
    return cfg


def _solve(name: str, tasks: List[Dict], machines: List[str], capacity: int):
    payload = {
        "job_id": name,
        "config": _config(washing_batch_capacity=capacity),
        "machines": [_machine(m) for m in machines],
        "resources": [_resource(m) for m in machines],
        "tasks": tasks,
    }
    result = Engine(payload).solve()
    real = [a for a in result["assignments"] if not a["task_id"].startswith("__")]
    return result, real


def _machines_used(assignments: List[Dict]) -> int:
    return len({a["machine_id"] for a in assignments})


def _makespan(assignments: List[Dict]) -> int:
    return max((a["end_time"] for a in assignments), default=0)


# ─────────────────────────── RED tests ────────────────────────────────────

def test_E4_two_slots_loose_deadline_consolidate_to_one_machine():
    """RED: qty>capacity forces 2 batch slots; deadline is loose (5000).

    Current: solver runs both slots PARALLEL on 2 machines (both end at t=60,
    cheaper batch_ends) → 2 machines wastefully concurrent.
    After fix: sequential on 1 machine [0,60]+[60,120], well within deadline.

    Mutation: remove machine_w → parallel end-terms (60+60) beat sequential
    (60+120) by 60, no machine cost → solver picks 2 machines → this assert fails.
    """
    _, a = _solve(
        "E4",
        [_task("W1", 60, 5000, ["WA", "WB"], 2), _task("W2", 60, 5000, ["WA", "WB"], 2)],
        ["WA", "WB"],
        capacity=3,
    )
    assert _machines_used(a) == 1, (
        f"2 short batches with a loose deadline should run sequentially on ONE "
        f"machine, but used {_machines_used(a)}: "
        f"{[(x['task_id'], x['machine_id'], x['start_time'], x['end_time']) for x in a]}"
    )


def test_E3c_multislot_small_qty_consolidate_to_one_machine():
    """RED: 4 tasks qty=1, capacity=2 → 2 batch slots; loose deadline.

    Current: 2 machines, both slots at [0,60] (parallel) → wasteful.
    After fix: 1 machine, slots [0,60] and [60,120].

    Mutation: remove machine_w → tie/parallel preference → ≥2 machines → fails.
    """
    _, a = _solve(
        "E3c",
        [_task(f"W{i}", 60, 5000, ["WA", "WB"], 1) for i in range(4)],
        ["WA", "WB"],
        capacity=2,
    )
    assert _machines_used(a) == 1, (
        f"4 tiny tasks (2 slots) with a loose deadline should consolidate onto ONE "
        f"machine, but used {_machines_used(a)}: "
        f"{[(x['task_id'], x['machine_id'], x['start_time'], x['end_time']) for x in a]}"
    )


# ─────────────────────────── GUARD tests ──────────────────────────────────

def test_guard_large_group_stays_parallel():
    """GUARD: 12 single-batch tasks, capacity=3, 4 machines, loose deadline.

    A large group's cumulative early-finish saving (∝ N²) dwarfs the flat
    per-machine cost (machine_w=500), so it MUST stay parallel — forcing it onto
    1 machine would blow makespan to 12×60=720.

    Must stay GREEN before and after the fix.
    Mutation: machine_w ≈ lateness (e.g. 100_000) → sequential forced →
    makespan 720, machines_used 1 → both asserts fail.
    """
    _, a = _solve(
        "LARGE",
        [_task(f"L{i}", 60, 5000, ["WA", "WB", "WC", "WD"], 3) for i in range(12)],
        ["WA", "WB", "WC", "WD"],
        capacity=3,
    )
    assert _machines_used(a) >= 2, (
        f"Large group must stay parallel; only {_machines_used(a)} machine(s) used"
    )
    assert _makespan(a) <= 300, (
        f"Large group makespan blew up to {_makespan(a)} (≈sequential) — "
        f"machine_w is too high and is forcing serialisation"
    )


def test_guard_deadline_forces_parallel():
    """GUARD: 2 tasks qty=cap=3, duration=60, tight deadline=65.

    Sequential on 1 machine → 2nd ends at 120 (misses 65 by 55 min).  Lateness
    coeff (100_000/min, priority 3) ≫ machine_w (500) → solver MUST keep them
    parallel on 2 machines to meet the deadline.

    Must stay GREEN.  Mutation: machine_w > lateness → forced sequential → a task
    ends at 120 > 65 (late) → assert fails.
    """
    _, a = _solve(
        "DEADLINE",
        [_task("W1", 60, 65, ["WA", "WB"], 3), _task("W2", 60, 65, ["WA", "WB"], 3)],
        ["WA", "WB"],
        capacity=3,
    )
    assert _machines_used(a) == 2, "Deadline-forced parallel must use 2 machines"
    assert _makespan(a) <= 65, f"A task missed its deadline (end={_makespan(a)} > 65)"


def test_guard_physically_disjoint_machines_stay_parallel():
    """GUARD: W1 only compatible with WA, W2 only with WB (no common machine).

    machine_w is a SOFT cost; it must yield to physical machine assignment — the
    two tasks cannot share a machine, so 2 machines is unavoidable and correct.

    Must stay GREEN.  Mutation: if machine_w were a HARD constraint it would make
    this infeasible → status not in (feasible, optimal).
    """
    r, a = _solve(
        "PHYS",
        [_task("W1", 60, 5000, ["WA"], 1), _task("W2", 60, 5000, ["WB"], 1)],
        ["WA", "WB"],
        capacity=10,
    )
    assert r["status"] in ("feasible", "optimal")
    assert _machines_used(a) == 2, "Disjoint-machine tasks must each keep their machine"


def test_guard_clean_single_slot_consolidates():
    """GUARD (consolidation no-regression): 2 tiny tasks, qty<capacity, fit ONE
    slot; loose deadline; 2 machines available.

    Already correct today (1 machine, 1 slot); the machine_w term must not break
    it.  Must stay GREEN.  Mutation: a bug that splits a single feasible slot
    across machines → 2 machines → fails.
    """
    _, a = _solve(
        "CLEAN",
        [_task("W1", 60, 5000, ["WA", "WB"], 1), _task("W2", 60, 5000, ["WA", "WB"], 1)],
        ["WA", "WB"],
        capacity=10,
    )
    assert _machines_used(a) == 1, (
        f"Clean single-slot group should use ONE machine, used {_machines_used(a)}"
    )
    starts = {a0["start_time"] for a0 in a}
    assert len(starts) == 1, "Both tasks should share one batch start (one slot)"


# ─────────────────── One-slot-one-machine (hard constraint) ────────────────

def _slot_machines(assignments: List[Dict]) -> Dict[str, set]:
    """batch_slot_id → set of machines used by that slot."""
    out: Dict[str, set] = {}
    for a in assignments:
        out.setdefault(a["batch_slot_id"], set()).add(a["machine_id"])
    return out


def test_no_slot_straddles_two_machines():
    """A single batch slot (one washing cycle) must run on exactly ONE machine.

    Scenario forces 2 machines: 6 qty-1 tasks, capacity=2 → 3 slots; due=130 makes
    sequential-on-one-machine (3×60=180) late, so the solver parallelises across
    WA+WB.  Each slot's qty (≤2) fits one machine, so NONE may straddle.

    Mutation: drop `model.Add(sum(lits) <= 1)` → a slot may split across WA+WB
    (free objective tie) → a slot maps to 2 machines → this assert fails.
    """
    _, a = _solve(
        "NOSTRADDLE",
        [_task(f"W{i}", 60, 130, ["WA", "WB"], 1) for i in range(6)],
        ["WA", "WB"],
        capacity=2,
    )
    straddlers = {sid: ms for sid, ms in _slot_machines(a).items() if len(ms) > 1}
    assert not straddlers, f"Slot(s) straddle >1 machine: {straddlers}"
    assert _machines_used(a) == 2, "Scenario should still parallelise across 2 machines"


def test_disjoint_coloc_stays_feasible_under_hard_cap():
    """GUARD for the hard-cap EXCEPTION: when co-located tasks share NO common
    machine, the slot-only co-location branch deliberately puts them in the same
    slot on DIFFERENT machines.  The one-slot-one-machine cap is skipped there
    (slot_machine_cap_ok=False), so this must stay FEASIBLE.

    Mutation: apply Σ_m slot_on_m[k] ≤ 1 unconditionally (drop the gate) → the
    forced same-slot/different-machine tasks make the model INFEASIBLE.
    """
    r, a = _solve(
        "DISJOINT_COLOC",
        [_task("W1", 60, 5000, ["WA"], 1), _task("W2", 60, 5000, ["WB"], 1)],
        ["WA", "WB"],
        capacity=3,
    )
    assert r["status"] in ("feasible", "optimal"), f"disjoint co-location went {r['status']}"
    assert {a0["task_id"] for a0 in a} == {"W1", "W2"}, "both tasks must be scheduled"
