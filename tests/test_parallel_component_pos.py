"""Parallel component-PO knitting candidate (`parallelize_component_pos`).

When a garment's component POs (front/back) are knit SERIALLY — every batch of one PO
before the other starts — the first complete panel (one batch from EACH PO at the same
index) can't be linked until the 2nd PO begins, idling linking workers.  This candidate
dedicates disjoint machine subsets to each component PO so they knit IN PARALLEL, making
the first/middle panels ready far sooner.  These tests call the generator directly on
hand-built assignments; the pipeline-level lateness verify is tested via the e2e path.
"""
from typing import Any, Dict, List

from app.engine.phases.phase1_knitting import parallelize_component_pos


def _knit(tid: str, group: str, machines: List[str], dur: int = 100) -> Dict[str, Any]:
    return {"task_id": tid, "operation": "Knitting", "group_id": group,
            "original_order_id": f"BATCH_{group}", "duration": dur,
            "compatible_resource_ids": machines, "start_after_min": 0, "is_pinned": False}


def _link(tid: str, garment: str, deps: List[str]) -> Dict[str, Any]:
    return {"task_id": tid, "operation": "Linking", "original_order_id": garment,
            "parent_task_id": garment, "final_depends_on": deps, "due_at_min": 99999}


def _asg(tid: str, m: str, s: int, e: int) -> Dict[str, Any]:
    return {"task_id": tid, "machine_id": m, "start_time": s, "end_time": e,
            "status": "ON_TIME"}


def _res(ids: List[str]) -> List[Dict[str, Any]]:
    return [{"id": m, "operation": "knitting", "capacity": 1,
             "unavailability": [], "available_at_min": 0} for m in ids]


def _po_window(cand, info, group):
    tids = [t for t in cand["start"] if info[t]["group_id"] == group]
    return (min(cand["start"][t] for t in tids), max(cand["end"][t] for t in tids),
            min(cand["end"][t] for t in tids))


def _build_serial(n=4):
    """Garment G: PO A then PO B, knit serially on 2 machines (A in [0,200], B in
    [200,400]) → first panel only at 300 though A's first batch finished at 100."""
    tasks: List[Dict[str, Any]] = []
    asg: List[Dict[str, Any]] = []
    for g in ("A", "B"):
        for k in range(1, n + 1):
            tasks.append(_knit(f"BATCH_{g}_{k}", g, ["M1", "M2"]))
    for k in range(1, n + 1):
        tasks.append(_link(f"L_{k}", "G", [f"BATCH_A_{k}", f"BATCH_B_{k}"]))
    # serial layout: A on [0,200] across M1/M2, then B on [200,400]
    half = n // 2
    for i, g in enumerate(("A", "B")):
        base = i * 200
        for k in range(1, n + 1):
            m = "M1" if (k - 1) < half else "M2"
            slot = (k - 1) % half
            s = base + slot * 100
            asg.append(_asg(f"BATCH_{g}_{k}", m, s, s + 100))
    return tasks, asg


def test_serial_pos_get_parallelized():
    tasks, asg = _build_serial(4)
    info = {t["task_id"]: t for t in tasks}
    cand = parallelize_component_pos(asg, tasks, _res(["M1", "M2"]), {"max_factory_machines": 2})
    assert cand is not None
    assert "machine" in cand  # dedication CHANGES machine — must be returned & applied
    a_s, a_e, a_first = _po_window(cand, info, "A")
    b_s, b_e, b_first = _po_window(cand, info, "B")
    # both POs now START in parallel (near t0=0), not B-after-A
    assert a_s == 0 and b_s == 0
    # first complete panel ready = max(earliest A end, earliest B end) dropped 300→100
    assert max(a_first, b_first) <= 200
    # no batch scheduled before the group's original earliest start
    assert all(v >= 0 for v in cand["start"].values())
    # each PO is DEDICATED to a disjoint machine set (each machine runs one PO)
    a_m = {cand["machine"][t] for t in cand["machine"] if info[t]["group_id"] == "A"}
    b_m = {cand["machine"][t] for t in cand["machine"] if info[t]["group_id"] == "B"}
    assert a_m and b_m and a_m.isdisjoint(b_m)
    # and the resulting per-machine schedule has no overlap
    by_m: Dict[str, list] = {}
    for t in cand["start"]:
        by_m.setdefault(cand["machine"][t], []).append((cand["start"][t], cand["end"][t]))
    for iv in by_m.values():
        iv.sort()
        assert all(iv[i][0] >= iv[i - 1][1] for i in range(1, len(iv)))


def test_already_parallel_is_noop():
    """If the two POs already overlap in time, there is nothing to parallelize."""
    tasks: List[Dict[str, Any]] = []
    for g in ("A", "B"):
        for k in (1, 2):
            tasks.append(_knit(f"BATCH_{g}_{k}", g, ["M1", "M2"]))
    for k in (1, 2):
        tasks.append(_link(f"L_{k}", "G", [f"BATCH_A_{k}", f"BATCH_B_{k}"]))
    # A on M1, B on M2, both [0,200] — already parallel
    asg = [
        _asg("BATCH_A_1", "M1", 0, 100), _asg("BATCH_A_2", "M1", 100, 200),
        _asg("BATCH_B_1", "M2", 0, 100), _asg("BATCH_B_2", "M2", 100, 200),
    ]
    cand = parallelize_component_pos(asg, tasks, _res(["M1", "M2"]), {"max_factory_machines": 2})
    assert cand is None


def test_single_component_is_noop():
    """A garment with only one component PO cannot be parallelized."""
    tasks = [_knit("BATCH_A_1", "A", ["M1"]), _knit("BATCH_A_2", "A", ["M1"]),
             _link("L_1", "G", ["BATCH_A_1"]), _link("L_2", "G", ["BATCH_A_2"])]
    asg = [_asg("BATCH_A_1", "M1", 0, 100), _asg("BATCH_A_2", "M1", 100, 200)]
    cand = parallelize_component_pos(asg, tasks, _res(["M1"]), {"max_factory_machines": 2})
    assert cand is None


def test_pinned_garment_left_untouched():
    tasks, asg = _build_serial(4)
    info = {t["task_id"]: t for t in tasks}
    info["BATCH_A_1"]["is_pinned"] = True  # a pinned knitting task on this garment
    cand = parallelize_component_pos(asg, tasks, _res(["M1", "M2"]), {"max_factory_machines": 2})
    assert cand is None
