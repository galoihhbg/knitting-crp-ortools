"""
Knitting order-contiguity penalty tests.

The knitting objective is otherwise indifferent to interleaving (gap→0 leaves it
unchanged), so the solver may split an order into several runs on a machine and
weave other orders through the middle.  Bosses prefer each order finished before
the next ("dứt điểm đơn đó").  `_apply_po_bounding_box(..., contiguity_w>0)` adds a
SOFT penalty on each order's per-(order,machine) footprint span, discouraging
interleaving while still yielding when shift windows force a split.
"""
import copy

from ortools.sat.python import cp_model

from app.engine.model import Engine
from app.engine.phases.phase1_knitting import _apply_po_bounding_box
from app.engine.shared import build_resource_model


def _knit(task_id, order, machine, dur=100, due=10000):
    return {
        "task_id": task_id,
        "original_order_id": order,
        "group_id": order,
        "operation": "Knitting",
        "qty": 10.0,
        "total_qty": 10.0,
        "priority": 3,
        "final_depends_on": [],
        "start_after_min": 0,
        "due_at_min": due,
        "duration": dur,
        "design_item_id": "D",
        "color_config": "C",
        "compatible_resource_ids": [machine] if isinstance(machine, str) else list(machine),
        "is_pinned": False,
    }


def _rmap(*ids):
    return {
        m: {"id": m, "type": "serial", "capacity": 1, "operation": "knitting",
            "unavailability": [], "available_at_min": 0}
        for m in ids
    }


# ── Unit: the penalty term is created only when contiguity_w > 0 ──────────────

class TestContiguityTerm:

    def _build(self, contiguity_w):
        model = cp_model.CpModel()
        # One order with 2 tasks both compatible with the same machine → an
        # (order, machine) box with ≥2 literals exists, so a span penalty applies.
        tasks = [_knit("A1", "OA", "KM_00"), _knit("A2", "OA", "KM_00")]
        rmap = _rmap("KM_00")
        task_vars, _, _ = build_resource_model(model, tasks, rmap, 5000, use_affinity=True)
        terms = _apply_po_bounding_box(model, task_vars, tasks, rmap, 5000, contiguity_w=contiguity_w)
        return terms

    def test_no_terms_when_disabled(self):
        assert self._build(contiguity_w=0) == []

    def test_terms_when_enabled(self):
        terms = self._build(contiguity_w=10)
        assert len(terms) >= 1  # at least one (order, machine) span penalty


# ── Integration: flag-on stays feasible and schedules everything ─────────────

def _payload(enable):
    tasks = [
        _knit("A1", "OA", ("KM_00", "KM_01")),
        _knit("A2", "OA", ("KM_00", "KM_01")),
        _knit("A3", "OA", ("KM_00", "KM_01")),
        _knit("B1", "OB", ("KM_00", "KM_01")),
        _knit("B2", "OB", ("KM_00", "KM_01")),
        _knit("B3", "OB", ("KM_00", "KM_01")),
    ]
    return {
        "job_id": "CONTIG",
        "config": {
            "horizon_minutes": 5000,
            "max_search_time": 10,
            "max_deterministic_time": 5,
            "random_seed": 42,
            "num_search_workers": 1,
            "max_factory_machines": 2,
            "enable_knitting_contiguity": enable,
            "knitting_contiguity_mult": 4,
        },
        "machines": [{"id": "KM_00", "design_item_id": "D", "color_config": "C"},
                     {"id": "KM_01", "design_item_id": "D", "color_config": "C"}],
        "resources": [{"id": m, "type": "serial", "capacity": 1, "operation": "knitting",
                       "unavailability": [], "design_item_id": "", "color_config": "",
                       "available_at_min": 0} for m in ("KM_00", "KM_01")],
        "tasks": tasks,
    }


def _fragmentation(result, tasks):
    info = {t["task_id"]: t for t in tasks}
    by_m = {}
    for a in result["assignments"]:
        t = info.get(a["task_id"])
        if not t or t["operation"].lower() != "knitting":
            continue
        by_m.setdefault(a["machine_id"], []).append((a["start_time"], t["original_order_id"]))
    frag = 0
    for items in by_m.values():
        items.sort()
        runs, prev = [], None
        for _s, oid in items:
            if oid != prev:
                runs.append(oid)
                prev = oid
        seen = {}
        for oid in runs:
            seen[oid] = seen.get(oid, 0) + 1
        frag += sum(1 for c in seen.values() if c > 1)
    return frag


def test_contiguity_feasible_and_no_worse_than_off():
    p_on = _payload(True)
    p_off = _payload(False)
    r_on = Engine(p_on).solve()
    r_off = Engine(p_off).solve()
    assert r_on["status"] in ("feasible", "optimal")
    # Every task scheduled.
    assert len([a for a in r_on["assignments"] if not a["task_id"].startswith("__")]) == 6
    # Contiguity must never INCREASE fragmentation vs off.
    on_frag = _fragmentation(r_on, p_on["tasks"])
    off_frag = _fragmentation(r_off, p_off["tasks"])
    assert on_frag <= off_frag, f"contiguity increased fragmentation: on={on_frag} off={off_frag}"
