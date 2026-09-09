"""
Prompt-washing penalty tests.

Operational WIP risk: goods that finished linking but sit unwashed for a long
time can be mislaid.  The washing objective otherwise minimises batch count +
machines used (consolidation) and rewards early starts only weakly, so with loose
due dates an early-ready slice gets bundled into a late batch and waits.

`enable_washing_prompt` adds a per-task WAIT penalty (start − ready) weighted
min_weight // washing_prompt_weight_divisor.  Unlike the raw-start tie-breaker it
subtracts ready time, so it minimises TOTAL wait and breaks the batch-assignment
symmetry — it pulls early-ready items toward their ready minute.  It is banded far
below the lateness ladder, so it never trades a missed due date for prompt washing.
"""
from typing import Any, Dict, List

from app.engine.model import Engine


def _task(tid: str, dur: int, due: int, rids: List[str], qty: float, release: int = 0):
    return {
        "task_id": tid,
        "original_order_id": tid,
        "group_id": "G",                 # one washing group → one _solve_group
        "operation": "washing",
        "qty": float(qty),
        "total_qty": float(qty),
        "priority": 3,
        "original_depends_on": [],
        "final_depends_on": [],
        "start_after_min": release,      # ready time (no linking deps in this fixture)
        "due_at_min": due,
        "duration": dur,
        "is_slice": False,
        "parent_task_id": "",
        "slice_index": 0,
        "is_batch": False,
        "sub_tasks": None,
        "design_item_id": "",
        "color_config": "",
        "color": "red",
        "substance": "cotton",
        "compatible_resource_ids": rids,
        "WaitOffsets": None,
        "is_pinned": False,
        "pinned_machine_id": None,
        "pinned_start_time": None,
        "pinned_end_time": None,
        "demand": 1,
        "material_demands": {},
    }


def _resource(m: str):
    return {
        "id": m, "type": "serial", "capacity": 1, "operation": "washing",
        "unavailability": [], "design_item_id": "", "color_config": "",
        "available_at_min": 0,
    }


def _machine(m: str):
    return {"id": m, "design_item_id": "D1", "color_config": "Black"}


def _solve(tasks, machines, **cfg_over):
    cfg = {
        "horizon_minutes": 6000,
        "max_search_time": 20,
        "max_deterministic_time": 10,
        "setup_time_minutes": 0,
        "max_factory_machines": 5,
        "random_seed": 42,
        "num_search_workers": 1,
        "washing_batch_capacity": 2,
        "washing_num_slots": 8,
    }
    cfg.update(cfg_over)
    payload = {
        "job_id": "WPROMPT",
        "config": cfg,
        "machines": [_machine(m) for m in machines],
        "resources": [_resource(m) for m in machines],
        "tasks": tasks,
    }
    r = Engine(payload).solve()
    real = {a["task_id"]: a for a in r["assignments"] if not a["task_id"].startswith("__")}
    return r, real


def _make_tasks():
    """Capacity pressure + mixed ready times.

    Six early-ready (release 0) tasks and three late-ready (release 1800) tasks,
    capacity 2 → ≥5 slots.  With loose dues the consolidation objective is free to
    bundle an early task with the late ones into a late slot; the prompt penalty
    pulls early-ready tasks to early slots, lowering total wait.
    """
    tasks = [_task(f"E{i}", 60, 6000, ["WA", "WB"], 1, release=0) for i in range(6)]
    tasks += [_task(f"L{i}", 60, 6000, ["WA", "WB"], 1, release=1800) for i in range(3)]
    return tasks


def _total_wait(real: Dict[str, Dict], tasks: List[Dict]) -> int:
    rel = {t["task_id"]: int(t["start_after_min"]) for t in tasks}
    return sum(max(0, a["start_time"] - rel[tid]) for tid, a in real.items())


def test_prompt_reduces_total_wait():
    """ON must wash items closer to their ready time → strictly less total wait."""
    tasks = _make_tasks()
    _, off = _solve(tasks, ["WA", "WB"], enable_washing_prompt=False)
    _, on = _solve(tasks, ["WA", "WB"], enable_washing_prompt=True,
                   washing_prompt_weight_divisor=100)
    w_off = _total_wait(off, tasks)
    w_on = _total_wait(on, tasks)
    assert w_on <= w_off, f"prompt should not increase total wait: on={w_on} off={w_off}"
    assert w_on < w_off, (
        f"prompt should strictly reduce total wait on this backlog fixture: "
        f"on={w_on} off={w_off}"
    )


def test_prompt_does_not_delay_early_ready_tasks():
    """With the prompt penalty, the early-ready tasks (release 0) wash promptly —
    none is dragged deep into the schedule behind the late-ready batch."""
    tasks = _make_tasks()
    _, on = _solve(tasks, ["WA", "WB"], enable_washing_prompt=True,
                   washing_prompt_weight_divisor=100)
    early_starts = [on[f"E{i}"]["start_time"] for i in range(6)]
    # All six early-ready items wash before the late-ready material is even ready.
    assert max(early_starts) < 1800, (
        f"early-ready tasks should wash before the late batch (1800); got {sorted(early_starts)}"
    )


def test_prompt_never_trades_lateness():
    """Banding guard: a tight due on a late-ready task must still be met — the wait
    penalty is ~10000× lighter per minute than lateness, so it never delays a task
    past its due to wash something else early."""
    tasks = [_task(f"E{i}", 60, 6000, ["WA", "WB"], 1, release=0) for i in range(4)]
    # One urgent late-ready task: ready 1800, due 1900 — must not be pushed late.
    tasks.append(_task("URGENT", 60, 1900, ["WA", "WB"], 1, release=1800))
    _, on = _solve(tasks, ["WA", "WB"], enable_washing_prompt=True,
                   washing_prompt_weight_divisor=100)
    u = on["URGENT"]
    assert u["end_time"] <= 1900, f"urgent task missed its due: end={u['end_time']} > 1900"


def test_prompt_flag_off_is_noop_path():
    """Disabling the flag must still produce a feasible washing schedule."""
    tasks = _make_tasks()
    r, off = _solve(tasks, ["WA", "WB"], enable_washing_prompt=False)
    assert r["status"] in ("feasible", "optimal")
    assert len(off) == len(tasks)


# ─────────────────────────── FIFO fairness ────────────────────────────────

def _fifo_tasks():
    """Four early-ready (release 0) and four late-ready (release 1500) tasks on
    one washing machine, capacity 2 → four batch slots.  A flat wait penalty is
    symmetric about which item waits and may strand an early-ready item behind the
    late-ready material; the earliness-weighted (FIFO) penalty must not."""
    tasks = [_task(f"E{i}", 60, 6000, ["WA"], 1, release=0) for i in range(4)]
    tasks += [_task(f"L{i}", 60, 6000, ["WA"], 1, release=1500) for i in range(4)]
    return tasks


def test_fifo_early_ready_not_stranded_behind_late():
    """With FIFO weighting every early-ready (release 0) item washes before the
    late-ready material (1500) — none is stranded behind a later-ready item."""
    tasks = _fifo_tasks()
    _, on = _solve(tasks, ["WA"], enable_washing_prompt=True,
                   washing_prompt_weight_divisor=100, washing_prompt_fifo_span=50,
                   washing_num_slots=8)
    early_starts = [on[f"E{i}"]["start_time"] for i in range(4)]
    assert max(early_starts) < 1500, (
        f"an early-ready item was stranded behind the late batch: {sorted(early_starts)}"
    )


def test_fifo_not_worse_than_flat():
    """Earliness weighting must never increase total wait vs the flat penalty, and
    must not increase the single longest wait (it removes the stranding tail)."""
    tasks = _fifo_tasks()
    rel = {t["task_id"]: int(t["start_after_min"]) for t in tasks}
    _, flat = _solve(tasks, ["WA"], enable_washing_prompt=True,
                     washing_prompt_fifo_span=0, washing_num_slots=8)
    _, fifo = _solve(tasks, ["WA"], enable_washing_prompt=True,
                     washing_prompt_fifo_span=50, washing_num_slots=8)
    flat_max = max(a["start_time"] - rel[tid] for tid, a in flat.items())
    fifo_max = max(a["start_time"] - rel[tid] for tid, a in fifo.items())
    assert fifo_max <= flat_max, f"FIFO increased the longest wait: fifo={fifo_max} flat={flat_max}"
