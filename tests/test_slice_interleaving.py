"""
Tests for slice interleaving across orders on the same machine.

Bug: apply_order_flow_objective's span term (span = group_end - group_start)
encourages the solver to bunch all slices of one order together before starting
the next order. This delays downstream tasks that depend on slice_N from multiple
POs/batches.

Example:
  ORDER_A: A_s1, A_s2, A_s3 on machine KM_00
  ORDER_B: B_s1, B_s2, B_s3 on machine KM_00
  Linking_B_s1 waits for B_s1 to finish.

  Bunched (current): A_s1(0-60) A_s2(60-120) A_s3(120-180) B_s1(180-240) ...
    → B_s1 ends at 240; downstream Linking_B_s1 cannot start until 240.
  Interleaved (desired): A_s1(0-60) B_s1(60-120) A_s2(120-180) B_s2(180-240) ...
    → B_s1 ends at 120; downstream Linking_B_s1 can start at 120.

Fix: apply_slice_sync_objective minimizes max(end_times of all slice_N tasks)
per slice_index N, driving interleaving. The span term is removed for groups
where all tasks are is_slice=True.
"""
import pytest
from app.engine.phases.phase1_knitting import solve_knitting
from app.engine.phases.phase2_linking import solve_linking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _knit_task(task_id, order_id, slice_index, duration, machines, priority=3, due_at=5000):
    return {
        "task_id": task_id,
        "original_order_id": order_id,
        "group_id": order_id,
        "operation": "knitting",
        "is_slice": True,
        "slice_index": slice_index,
        "parent_task_id": f"K_{order_id}",
        "duration": duration,
        "compatible_resource_ids": machines,
        "due_at_min": due_at,
        "priority": priority,
        "is_pinned": False,
        "start_after_min": 0,
        "qty": 10.0,
        "total_qty": 30.0,
        "design_item_id": "",
        "color_config": "",
        "is_batch": False,
        "sub_tasks": None,
        "demand": 1,
    }


def _link_task(task_id, order_id, slice_index, duration, machines,
               start_after_min=0, priority=3, due_at=5000):
    return {
        "task_id": task_id,
        "original_order_id": order_id,
        "group_id": order_id,
        "operation": "linking",
        "is_slice": True,
        "slice_index": slice_index,
        "parent_task_id": f"L_{order_id}",
        "duration": duration,
        "compatible_resource_ids": machines,
        "due_at_min": due_at,
        "priority": priority,
        "is_pinned": False,
        "start_after_min": start_after_min,
        "qty": 10.0,
        "total_qty": 30.0,
        "design_item_id": "",
        "color_config": "",
        "is_batch": False,
        "sub_tasks": None,
        "WaitOffsets": None,
        "demand": 0,
    }


def _resource(resource_id):
    return {
        "id": resource_id,
        "type": "serial",
        "capacity": 1,
        "unavailability": [],
        "available_at_min": 0,
        "design_item_id": "",
        "color_config": "",
    }


def _config(horizon=5000, time_limit=30):
    return {
        "horizon_minutes": horizon,
        "max_search_time": time_limit,
        "num_search_workers": 1,
        "random_seed": 42,
        "max_factory_machines": 10,
    }


# ---------------------------------------------------------------------------
# Phase 1 (Knitting) slice interleaving tests
# ---------------------------------------------------------------------------

class TestKnittingSliceInterleaving:

    def test_two_orders_three_slices_interleaved(self):
        """
        Two orders, 3 is_slice tasks each, all on ONE machine.

        Expected (interleaved): A_s1(0-60) B_s1(60-120) A_s2 B_s2 A_s3 B_s3
          → max(A_s1.end, B_s1.end) = 120 = 2 × SLICE_DUR

        Bug behaviour (bunched): A_s1 A_s2 A_s3 B_s1 ...
          → max(A_s1.end, B_s1.end) = 240 = 4 × SLICE_DUR
        """
        SLICE_DUR = 60
        machine = "KM_00"

        tasks = []
        for order in ["ORDER_A", "ORDER_B"]:
            for s in range(1, 4):
                tasks.append(_knit_task(f"K_{order}_s{s}", order, s, SLICE_DUR, [machine]))

        result = solve_knitting(tasks, [_resource(machine)], _config())
        assert result.status == "feasible"

        ends = {a["task_id"]: a["end_time"] for a in result.assignments}

        max_s1_end = max(ends["K_ORDER_A_s1"], ends["K_ORDER_B_s1"])
        assert max_s1_end <= 2 * SLICE_DUR, (
            f"Slice bunching detected! max slice_1 end = {max_s1_end}, "
            f"expected ≤ {2 * SLICE_DUR}.\n"
            f"  A: s1={ends['K_ORDER_A_s1']}, s2={ends['K_ORDER_A_s2']}, s3={ends['K_ORDER_A_s3']}\n"
            f"  B: s1={ends['K_ORDER_B_s1']}, s2={ends['K_ORDER_B_s2']}, s3={ends['K_ORDER_B_s3']}"
        )

    def test_three_orders_three_slices_interleaved(self):
        """
        Three orders competing for the same machine.
        All three slice_1 tasks must complete within 3 × SLICE_DUR.
        """
        SLICE_DUR = 60
        machine = "KM_00"

        tasks = []
        for order in ["ORDER_A", "ORDER_B", "ORDER_C"]:
            for s in range(1, 4):
                tasks.append(_knit_task(f"K_{order}_s{s}", order, s, SLICE_DUR, [machine]))

        result = solve_knitting(tasks, [_resource(machine)], _config())
        assert result.status == "feasible"

        ends = {a["task_id"]: a["end_time"] for a in result.assignments}
        orders = ["ORDER_A", "ORDER_B", "ORDER_C"]

        max_s1_end = max(ends[f"K_{o}_s1"] for o in orders)
        assert max_s1_end <= 3 * SLICE_DUR, (
            f"Bunching detected with 3 orders! max slice_1 end = {max_s1_end}, "
            f"expected ≤ {3 * SLICE_DUR}.\n"
            + "\n".join(
                f"  {o}: s1={ends[f'K_{o}_s1']}, s2={ends[f'K_{o}_s2']}, s3={ends[f'K_{o}_s3']}"
                for o in orders
            )
        )

    def test_slice_2_also_interleaved(self):
        """Slice_2 of all orders also finishes within 5 slots (2 × 2 + 1 guard)."""
        SLICE_DUR = 60
        machine = "KM_00"

        tasks = []
        for order in ["ORDER_A", "ORDER_B"]:
            for s in range(1, 4):
                tasks.append(_knit_task(f"K_{order}_s{s}", order, s, SLICE_DUR, [machine]))

        result = solve_knitting(tasks, [_resource(machine)], _config())
        assert result.status == "feasible"

        ends = {a["task_id"]: a["end_time"] for a in result.assignments}

        # After interleaving slice_1 (slots 1-2), both slice_2 fit in slots 3-4
        max_s2_end = max(ends["K_ORDER_A_s2"], ends["K_ORDER_B_s2"])
        assert max_s2_end <= 4 * SLICE_DUR, (
            f"Slice_2 bunched! max s2 end = {max_s2_end}, expected ≤ {4 * SLICE_DUR}.\n"
            f"  A: s1={ends['K_ORDER_A_s1']}, s2={ends['K_ORDER_A_s2']}, s3={ends['K_ORDER_A_s3']}\n"
            f"  B: s1={ends['K_ORDER_B_s1']}, s2={ends['K_ORDER_B_s2']}, s3={ends['K_ORDER_B_s3']}"
        )

    def test_single_order_slices_no_regression(self):
        """Single-order slice schedule should still be compact."""
        SLICE_DUR = 60
        machine = "KM_00"

        tasks = [
            _knit_task(f"K_ORDER_A_s{s}", "ORDER_A", s, SLICE_DUR, [machine])
            for s in range(1, 4)
        ]

        result = solve_knitting(tasks, [_resource(machine)], _config())
        assert result.status == "feasible"

        ends = {a["task_id"]: a["end_time"] for a in result.assignments}
        assert ends["K_ORDER_A_s3"] <= 3 * SLICE_DUR

    def test_non_slice_tasks_not_affected(self):
        """is_slice=False tasks must still schedule correctly (regression guard)."""
        tasks = [
            {
                "task_id": "K_A",
                "original_order_id": "ORDER_A",
                "group_id": "ORDER_A",
                "operation": "knitting",
                "is_slice": False,
                "slice_index": 0,
                "parent_task_id": "",
                "duration": 120,
                "compatible_resource_ids": ["KM_00"],
                "due_at_min": 5000,
                "priority": 3,
                "is_pinned": False,
                "start_after_min": 0,
                "qty": 10.0,
                "total_qty": 10.0,
                "design_item_id": "",
                "color_config": "",
                "is_batch": False,
                "sub_tasks": None,
                "demand": 1,
            },
            {
                "task_id": "K_B",
                "original_order_id": "ORDER_B",
                "group_id": "ORDER_B",
                "operation": "knitting",
                "is_slice": False,
                "slice_index": 0,
                "parent_task_id": "",
                "duration": 120,
                "compatible_resource_ids": ["KM_00"],
                "due_at_min": 5000,
                "priority": 3,
                "is_pinned": False,
                "start_after_min": 0,
                "qty": 10.0,
                "total_qty": 10.0,
                "design_item_id": "",
                "color_config": "",
                "is_batch": False,
                "sub_tasks": None,
                "demand": 1,
            },
        ]
        result = solve_knitting(tasks, [_resource("KM_00")], _config())
        assert result.status == "feasible"
        assert len(result.assignments) == 2

    def test_different_machines_both_interleaved(self):
        """
        Orders with slices spread across TWO machines.
        ORDER_A: 14G machine. ORDER_B: also 14G. ORDER_C: 7G machine.
        14G slices from A and B should be interleaved.
        """
        SLICE_DUR = 60

        tasks = []
        for order in ["ORDER_A", "ORDER_B"]:
            for s in range(1, 3):
                tasks.append(_knit_task(f"K_{order}_s{s}", order, s, SLICE_DUR, ["14G"]))
        for s in range(1, 3):
            tasks.append(_knit_task(f"K_ORDER_C_s{s}", "ORDER_C", s, SLICE_DUR, ["7G"]))

        resources = [_resource("14G"), _resource("7G")]
        result = solve_knitting(tasks, resources, _config())
        assert result.status == "feasible"

        ends = {a["task_id"]: a["end_time"] for a in result.assignments}

        # A and B compete for 14G — slice_1 of both must finish within 2 slots
        max_s1_14g = max(ends["K_ORDER_A_s1"], ends["K_ORDER_B_s1"])
        assert max_s1_14g <= 2 * SLICE_DUR, (
            f"A/B bunched on 14G! max s1 end = {max_s1_14g}, "
            f"A_s1={ends['K_ORDER_A_s1']}, B_s1={ends['K_ORDER_B_s1']}"
        )


# ---------------------------------------------------------------------------
# Phase 2 (Linking) slice interleaving tests
# ---------------------------------------------------------------------------

class TestLinkingSliceInterleaving:

    def test_two_orders_linking_slices_interleaved(self):
        """
        Two orders, 3 linking slices each on the same linking machine.
        Linking slice_1 of both orders must finish within 2 × SLICE_DUR.
        """
        SLICE_DUR = 45
        machine = "LM_00"

        tasks = []
        for order in ["ORDER_A", "ORDER_B"]:
            for s in range(1, 4):
                tasks.append(_link_task(f"L_{order}_s{s}", order, s, SLICE_DUR, [machine]))

        result = solve_linking(
            tasks, [_resource(machine)], _config(),
            p1_start_times={}, p1_end_times={}, translation_map={},
            horizon=5000,
        )
        assert result.status == "feasible"

        ends = {a["task_id"]: a["end_time"] for a in result.assignments}

        max_s1_end = max(ends["L_ORDER_A_s1"], ends["L_ORDER_B_s1"])
        assert max_s1_end <= 2 * SLICE_DUR, (
            f"Linking slice bunching! max s1 end = {max_s1_end}, "
            f"expected ≤ {2 * SLICE_DUR}.\n"
            f"  A: s1={ends['L_ORDER_A_s1']}, s2={ends['L_ORDER_A_s2']}, s3={ends['L_ORDER_A_s3']}\n"
            f"  B: s1={ends['L_ORDER_B_s1']}, s2={ends['L_ORDER_B_s2']}, s3={ends['L_ORDER_B_s3']}"
        )

    def test_three_orders_linking_slices_interleaved(self):
        """Three orders, 3 linking slices each — slice_1 of all three within 3 slots."""
        SLICE_DUR = 45
        machine = "LM_00"
        orders = ["ORDER_A", "ORDER_B", "ORDER_C"]

        tasks = []
        for order in orders:
            for s in range(1, 4):
                tasks.append(_link_task(f"L_{order}_s{s}", order, s, SLICE_DUR, [machine]))

        result = solve_linking(
            tasks, [_resource(machine)], _config(),
            p1_start_times={}, p1_end_times={}, translation_map={},
            horizon=5000,
        )
        assert result.status == "feasible"

        ends = {a["task_id"]: a["end_time"] for a in result.assignments}
        max_s1_end = max(ends[f"L_{o}_s1"] for o in orders)
        assert max_s1_end <= 3 * SLICE_DUR, (
            f"Linking bunched with 3 orders! max s1 end = {max_s1_end}, "
            f"expected ≤ {3 * SLICE_DUR}"
        )


# ---------------------------------------------------------------------------
# FIFO-by-PO linking floor (enable_fifo_linking_floor) — in-solver default
# ---------------------------------------------------------------------------

class TestFifoLinkingFloor:
    """Reproduces the WLJPELMsPp bug: two same-panel component POs (643/644) whose
    panels finish OUT of index order.  A linking slice needs one panel of each PO and
    they are interchangeable, but the index pairing (SLICE_k ↔ 643_k AND 644_k) forces
    a slice to wait for its specific late index-panel while an early sibling sits ready.
    The FIFO-by-PO floor (k-th slice ← k-th-FINISHED panel of each PO bucket) frees the
    middle slices; per-order completion (the LAST slice) is unchanged.
    """

    # PO 643 panel ends: _1 late (1076), the rest early.   PO 644: _4 late (2756).
    P643 = {"BATCH_643_1": 1076, "BATCH_643_2": 255, "BATCH_643_3": 407,
            "BATCH_643_4": 255, "BATCH_643_5": 407}
    P644 = {"BATCH_644_1": 407, "BATCH_644_2": 152, "BATCH_644_3": 407,
            "BATCH_644_4": 2756, "BATCH_644_5": 152}

    def _knit_panels(self):
        tasks, starts, ends = [], {}, {}
        for po, panels in (("643", self.P643), ("644", self.P644)):
            for tid, end in panels.items():
                tasks.append({
                    "task_id": tid, "operation": "knitting", "group_id": po,
                    "qty": 50.0, "final_depends_on": [],
                })
                starts[tid] = 0
                ends[tid] = end
        return tasks, starts, ends

    def _link_slices(self, machines):
        slices = []
        for k in range(1, 6):
            t = _link_task(f"L_WLJ_s{k}", "WLJ", k, 100, machines)
            t["parent_task_id"] = "L_WLJ"
            t["final_depends_on"] = [f"BATCH_643_{k}", f"BATCH_644_{k}"]
            slices.append(t)
        return slices

    def _run(self, fifo: bool):
        knit, p1s, p1e = self._knit_panels()
        workers = [f"LM_{i:02d}" for i in range(5)]  # plenty: capacity never the limiter
        slices = self._link_slices(workers)
        kwargs = dict(
            p1_start_times=p1s, p1_end_times=p1e, translation_map={}, horizon=6000,
        )
        if fifo:
            kwargs["all_pipeline_tasks"] = knit + slices
        result = solve_linking(slices, [_resource(w) for w in workers], _config(horizon=6000), **kwargs)
        assert result.status == "feasible"
        return sorted(a["start_time"] for a in result.assignments)

    def test_fifo_floor_frees_two_slices_to_earliest(self):
        """Under FIFO at least TWO slices start at 255 (643_2/4 ↔ 644_2/5); under index
        only ONE can (the second-ready pair is index-mismatched 643_4 vs 644_5)."""
        fifo_starts = self._run(fifo=True)
        index_starts = self._run(fifo=False)

        assert sum(1 for s in fifo_starts if s <= 255) >= 2, (
            f"FIFO floor should free 2 slices to ≤255, got {fifo_starts}"
        )
        assert sum(1 for s in index_starts if s <= 255) == 1, (
            f"index pairing should allow only 1 slice ≤255, got {index_starts}"
        )

    def test_fifo_floor_preserves_order_completion(self):
        """The order's LAST linking slice still waits for the last panel (644_4@2756) —
        FIFO only nudges the middle slices earlier, never the makespan."""
        fifo_starts = self._run(fifo=True)
        index_starts = self._run(fifo=False)
        # last slice gated by the genuinely-late panel in both models
        assert max(fifo_starts) == max(index_starts) == 2756
        # FIFO is element-wise no-later than index (pure relaxation)
        for f, i in zip(fifo_starts, index_starts):
            assert f <= i, f"FIFO start {f} later than index {i}: {fifo_starts} vs {index_starts}"
