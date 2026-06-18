"""
Same-qty re-link refinement (two-pass, Pareto-guarded) tests.

Domain rule under test: knitting panels are interchangeable ONLY within the same
(component=group_id, qty) bucket.  Go pairs linking SLICE_k to panel _k by index;
when knitting finishes out of index order, a slice waits for "its" panel while an
identical finished panel sits idle.  The refinement relaxes the linking floor to
the k-th-earliest panel end of the matching bucket (FIFO), re-solves linking with
per-task Pareto end-caps, re-solves downstream, and accepts ONLY if no task in the
whole pipeline finishes later than pass 1.

The synthetic fixture uses two components with CROSSED index pairing:
    L_s1 depends on (K_c1_late, K_c2_early)   → index floor 500
    L_s2 depends on (K_c1_early, K_c2_late)   → index floor 500
Same-qty floors: L_s1 → 100 (1st-finished panel of each bucket), L_s2 → 500.
The within-component bijection keeps order completion at 500, but the crossed
AND across components is exactly where the index floor over-constrains.
"""
import copy

from app.engine.model import Engine
from app.engine.phases.phase2_linking import (
    _compute_start_lb,
    compute_sameqty_start_lb,
)


# ── Fixture ────────────────────────────────────────────────────────────────

def _knit(task_id, group, machine, start, end, qty=10.0):
    return {
        "task_id": task_id,
        "original_order_id": f"O-{group}",
        "group_id": group,
        "operation": "Knitting",
        "qty": qty,
        "total_qty": qty,
        "priority": 3,
        "final_depends_on": [],
        "start_after_min": 0,
        "due_at_min": 10000,
        "duration": end - start,
        "design_item_id": "D1",
        "color_config": "MAT_WHT:1",
        "compatible_resource_ids": [machine],
        "is_pinned": True,
        "pinned_machine_id": machine,
        "pinned_start_time": start,
        "pinned_end_time": end,
    }


def _link(task_id, deps, qty=10.0):
    return {
        "task_id": task_id,
        "original_order_id": "O-LINK",
        "group_id": "GL",
        "operation": "Linking",
        "qty": qty,
        "total_qty": qty,
        "priority": 3,
        "final_depends_on": deps,
        "start_after_min": 0,
        "due_at_min": 300,
        "duration": 50,
        "design_item_id": "",
        "color_config": "",
        "compatible_resource_ids": ["LM_00"],
        "is_slice": True,
        "parent_task_id": "L_parent",
    }


def _resource(r_id, op):
    return {
        "id": r_id,
        "type": "serial",
        "capacity": 1,
        "operation": op,
        "unavailability": [],
        "design_item_id": "",
        "color_config": "",
        "available_at_min": 0,
    }


def make_crossed_payload(enable_relink=True):
    """2 components × 2 panels each (ends 100 / 500), crossed index pairing."""
    tasks = [
        _knit("K_c1_early", "c1", "KM_00", 0, 100),
        _knit("K_c1_late", "c1", "KM_00", 100, 500),
        _knit("K_c2_early", "c2", "KM_01", 0, 100),
        _knit("K_c2_late", "c2", "KM_01", 100, 500),
        _link("L_s1", ["K_c1_late", "K_c2_early"]),
        _link("L_s2", ["K_c1_early", "K_c2_late"]),
    ]
    return {
        "job_id": "TEST_SAMEQTY",
        "config": {
            "horizon_minutes": 20000,
            "max_search_time": 10,
            "max_deterministic_time": 5,
            "random_seed": 42,
            "num_search_workers": 1,
            "enable_sameqty_relink": enable_relink,
        },
        "machines": [],
        "resources": [
            _resource("KM_00", "knitting"),
            _resource("KM_01", "knitting"),
            _resource("LM_00", "linking"),
        ],
        "tasks": tasks,
        "material_capacities": {},
    }


def _ends(result):
    return {a["task_id"]: a["end_time"] for a in result["assignments"]}


# ── Floor unit tests ───────────────────────────────────────────────────────

class TestSameQtyFloor:

    def _times(self, payload):
        knit = [t for t in payload["tasks"] if t["operation"] == "Knitting"]
        starts = {t["task_id"]: t["pinned_start_time"] for t in knit}
        ends = {t["task_id"]: t["pinned_end_time"] for t in knit}
        return starts, ends

    def test_crossed_pairing_relaxes_middle_slice_only(self):
        payload = make_crossed_payload()
        linking = [t for t in payload["tasks"] if t["operation"] == "Linking"]
        starts, ends = self._times(payload)
        trans = {t["task_id"]: t["task_id"] for t in payload["tasks"]}

        lb_index = _compute_start_lb(linking, starts, ends, trans)
        lb_sameqty = compute_sameqty_start_lb(
            linking, starts, ends, trans, payload["tasks"]
        )

        assert lb_index == {"L_s1": 500, "L_s2": 500}
        # FIFO bucket assignment: s1 gets the 1st-finished panel of each bucket.
        assert lb_sameqty == {"L_s1": 100, "L_s2": 500}
        # Same-qty floor must never exceed the index floor.
        assert all(lb_sameqty[k] <= lb_index[k] for k in lb_index)

    def test_never_mixes_quantities(self):
        """A qty-10 dep must NOT borrow a qty-20 panel even if it finished first."""
        payload = make_crossed_payload()
        # Make the early c1 panel a DIFFERENT quantity → separate bucket.
        for t in payload["tasks"]:
            if t["task_id"] == "K_c1_early":
                t["qty"] = t["total_qty"] = 20.0
        linking = [t for t in payload["tasks"] if t["operation"] == "Linking"]
        starts, ends = self._times(payload)
        trans = {t["task_id"]: t["task_id"] for t in payload["tasks"]}

        lb_sameqty = compute_sameqty_start_lb(
            linking, starts, ends, trans, payload["tasks"]
        )
        # L_s1's c1 dep (K_c1_late, qty 10) is alone in its bucket → floor stays 500.
        assert lb_sameqty["L_s1"] == 500

    def test_waitoffsets_relax_with_reassigned_panel(self):
        """Regression: WaitOffsets must follow the bucket-reassigned panel, not
        re-pin to the index panel.

        Old bug: same-qty relaxed only the final_depends_on END but left the
        WaitOffsets START+offset pinned to the specific index panel, so the floor
        snapped back to the index value (no-op).  Now both end and start+offset are
        read from the SAME bucket-assigned panel.

        Crossed fixture + offset 300 on every dep.  L_s1 is reassigned the
        first-finished panels of each bucket (K_c1_early/K_c2_early, start 0):
          * end floor        = 100   (earliest panel end)
          * WaitOffset floor = 0 + 300 = 300   ← uses the REASSIGNED panel's start
          * result           = max(100, 300) = 300
        The old re-pin would have used the index panel start (K_c1_late=100) →
        100 + 300 = 400.  Index floor (no relaxation) is 500.
        """
        payload = make_crossed_payload()
        linking = [t for t in payload["tasks"] if t["operation"] == "Linking"]
        for t in linking:
            t["wait_offsets"] = {d: 300 for d in t["final_depends_on"]}
        starts, ends = self._times(payload)
        trans = {t["task_id"]: t["task_id"] for t in payload["tasks"]}

        lb_index = _compute_start_lb(linking, starts, ends, trans)
        lb_sameqty = compute_sameqty_start_lb(
            linking, starts, ends, trans, payload["tasks"]
        )
        assert lb_index == {"L_s1": 500, "L_s2": 500}
        # 300 (reassigned panel start 0 + 300), NOT 400 (old re-pin), NOT 100 (offset dropped).
        assert lb_sameqty["L_s1"] == 300
        assert lb_sameqty["L_s2"] == 500
        assert all(lb_sameqty[k] <= lb_index[k] for k in lb_index)

    def test_unresolvable_or_nonknit_dep_falls_back_to_index(self):
        payload = make_crossed_payload()
        linking = [t for t in payload["tasks"] if t["operation"] == "Linking"]
        starts, ends = self._times(payload)
        ends["GHOST"] = 777
        starts["GHOST"] = 700
        linking[0]["final_depends_on"] = ["GHOST"]  # not a knitting task
        trans = {t["task_id"]: t["task_id"] for t in payload["tasks"]}

        lb_sameqty = compute_sameqty_start_lb(
            linking, starts, ends, trans, payload["tasks"]
        )
        assert lb_sameqty[linking[0]["task_id"]] == 777


# ── Pipeline integration ──────────────────────────────────────────────────

class TestSameQtyRelinkPipeline:

    def test_relink_is_pointwise_pareto(self):
        """Refined schedule must never finish ANY task later than baseline."""
        r_off = Engine(make_crossed_payload(enable_relink=False)).solve()
        r_on = Engine(make_crossed_payload(enable_relink=True)).solve()
        assert r_off["status"] == "feasible"
        assert r_on["status"] == "feasible"

        e_off, e_on = _ends(r_off), _ends(r_on)
        assert set(e_off) == set(e_on)
        regressed = {k: (e_off[k], e_on[k]) for k in e_off if e_on[k] > e_off[k]}
        assert not regressed, f"Pareto guard violated: {regressed}"

    def test_relink_improves_crossed_instance(self):
        """The crossed fixture has real slack — refinement must harvest some."""
        r_off = Engine(make_crossed_payload(enable_relink=False)).solve()
        r_on = Engine(make_crossed_payload(enable_relink=True)).solve()
        e_off, e_on = _ends(r_off), _ends(r_on)
        link_ids = ("L_s1", "L_s2")
        assert sum(e_on[k] for k in link_ids) < sum(e_off[k] for k in link_ids), (
            f"expected linking improvement, got off={e_off} on={e_on}"
        )
        # Baseline floors are 500/500 → both end ≥ 550; the relaxed slice starts
        # at the 1st-finished same-qty panels (floor 100) → one slice ends ≤ 500.
        assert min(e_on[k] for k in link_ids) <= 500

    def test_relink_never_starts_before_sameqty_floor(self):
        """Physical guard: no linking slice may start before its bucket floor."""
        r_on = Engine(make_crossed_payload(enable_relink=True)).solve()
        starts = {a["task_id"]: a["start_time"] for a in r_on["assignments"]}
        # k-th slice (FIFO) needs the k-th-finished panel of EACH of its buckets:
        # even relaxed, L_s2 cannot start before 500 (2nd panels finish at 500).
        assert starts["L_s2"] >= 500
        assert starts["L_s1"] >= 100

    def test_relink_deterministic(self):
        payload = make_crossed_payload(enable_relink=True)
        r1 = Engine(copy.deepcopy(payload)).solve()
        r2 = Engine(copy.deepcopy(payload)).solve()
        a1 = sorted(r1["assignments"], key=lambda a: a["task_id"])
        a2 = sorted(r2["assignments"], key=lambda a: a["task_id"])
        assert a1 == a2
