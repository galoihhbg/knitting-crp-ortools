"""FIFO-swap iron/packing — core dùng chung, guard khác nhau có chủ ý.

Iron: guard proxy ``new_e ≤ max iron-end của đơn blocker`` (packing chưa solve,
không so due được).  Packing (op CUỐI): guard due-cap ``new_e ≤ due của blocker``
— end đơn = max packing-end nên chỉ cần không vượt due là tardiness đơn không
tăng và không task nào lật LATE; proxy iron ở đây sẽ từ chối oan blocker tí hon
mà đơn của nó chỉ có một task (case thật CP_1783586686707912847).
"""
from typing import Any, Dict, List

from app.engine.phases.phase4_downstream import (
    fifo_swap_ironing,
    fifo_swap_packing,
)


def _task(tid: str, op: str, due: int, deps: List[str],
          pinned: bool = False, compat: List[str] = None) -> Dict[str, Any]:
    return {
        "task_id": tid,
        "operation": op,
        "due_at_min": due,
        "final_depends_on": deps,
        "is_pinned": pinned,
        "compatible_resource_ids": compat or ["M1", "M2"],
        "start_after_min": 0,
    }


def _asg(tid: str, m: str, s: int, e: int, order: str = None) -> Dict[str, Any]:
    return {
        "task_id": tid, "machine_id": m, "start_time": s, "end_time": e,
        "order_id": order or tid,
        "status": "ON_TIME",
    }


def _overlap_free(assigns: List[Dict[str, Any]]) -> bool:
    by_m: Dict[str, List] = {}
    for a in assigns:
        by_m.setdefault(a["machine_id"], []).append(
            (a["start_time"], a["end_time"]))
    for ivs in by_m.values():
        ivs.sort()
        for i in range(1, len(ivs)):
            if ivs[i][0] < ivs[i - 1][1]:
                return False
    return True


class TestFifoSwapPacking:
    def _scenario(self):
        """Chưng cất từ CP_1783586686707912847: waiter 51′ ready sớm, cửa sổ duy
        nhất bị task 3′ ready-muộn (due lỏng) chiếm giữa; máy kia bận task thật."""
        tasks = [
            _task("W", "Packing", due=160, deps=["IW"]),          # waiter, ready 100
            _task("B", "Packing", due=1000, deps=["IB"]),          # blocker 3′, ready 110
            _task("C", "Packing", due=1000, deps=["IC"]),          # máy 2 bận, KHÍT (ready 100)
        ]
        assigns = [
            _asg("B", "M1", 110, 113),
            _asg("C", "M2", 100, 150),
            _asg("W", "M1", 130, 180),   # solver đặt sau blocker → WAIT 30, LATE 20
        ]
        dep_ends = {"IW": 100, "IB": 110, "IC": 100}
        return tasks, assigns, dep_ends

    def test_log_case_swap_accepted(self):
        tasks, assigns, dep_ends = self._scenario()
        n = fifo_swap_packing(assigns, tasks, {}, dep_ends)
        by = {a["task_id"]: a for a in assigns}
        assert n == 1
        # waiter chiếm cửa sổ tại release → hết trễ
        assert (by["W"]["machine_id"], by["W"]["start_time"], by["W"]["end_time"]) \
            == ("M1", 100, 150)
        assert by["W"]["status"] == "ON_TIME"
        # blocker re-seat ≥ ready của nó, không vượt due, không lật LATE
        assert by["B"]["start_time"] >= 110
        assert by["B"]["end_time"] <= 1000
        assert by["B"]["status"] == "ON_TIME"
        assert _overlap_free(assigns)

    def test_iron_proxy_would_reject_same_layout(self):
        """Cùng layout nhưng op = iron → guard proxy (order_max_end=113) từ chối
        re-seat blocker tới 153.  Đây là KHÁC BIỆT CÓ CHỦ Ý giữa hai wrapper."""
        tasks, assigns, dep_ends = self._scenario()
        for t in tasks:
            t["operation"] = "Iron"
        n = fifo_swap_ironing(assigns, tasks, {}, dep_ends)
        by = {a["task_id"]: a for a in assigns}
        assert n == 0
        assert by["W"]["start_time"] == 130  # giữ nguyên

    def test_no_swap_when_blocker_ready_before_waiter(self):
        tasks, assigns, dep_ends = self._scenario()
        dep_ends["IB"] = 95  # blocker ready TRƯỚC waiter → không phải inversion
        n = fifo_swap_packing(assigns, tasks, {}, dep_ends)
        assert n == 0
        assert {a["task_id"]: a["start_time"] for a in assigns}["W"] == 130

    def test_no_swap_when_blocker_would_cross_its_due(self):
        tasks, assigns, dep_ends = self._scenario()
        tasks[1]["due_at_min"] = 120  # blocker due chặt: re-seat 150-153 vượt due
        n = fifo_swap_packing(assigns, tasks, {}, dep_ends)
        assert n == 0

    def test_no_swap_when_blocker_pinned(self):
        tasks, assigns, dep_ends = self._scenario()
        tasks[1]["is_pinned"] = True
        n = fifo_swap_packing(assigns, tasks, {}, dep_ends)
        assert n == 0

    def test_blocker_new_end_bounded_by_waiter_old_end(self):
        # blocker dài: re-seat sẽ end vượt end cũ của waiter → từ chối
        tasks, assigns, dep_ends = self._scenario()
        assigns[0]["end_time"] = 113 + 60  # blocker 63′ (110-173)
        n = fifo_swap_packing(assigns, tasks, {}, dep_ends)
        assert n == 0

    def test_disabled_by_flag(self):
        tasks, assigns, dep_ends = self._scenario()
        n = fifo_swap_packing(
            assigns, tasks, {"enable_packing_fifo_swap": False}, dep_ends)
        assert n == 0


class TestFifoSwapIroningRegression:
    def test_iron_swap_within_order_proxy_still_works(self):
        """Hành vi iron GIỮ NGUYÊN sau refactor: blocker re-seat được khi đơn nó
        còn iron-end muộn hơn (proxy cho phép)."""
        tasks = [
            _task("W", "Iron", due=160, deps=["XW"]),
            _task("B", "Iron", due=1000, deps=["XB"]),   # blocker ready 110
            _task("B2", "Iron", due=1000, deps=["XB2"]),  # cùng đơn blocker, end muộn
            _task("C", "Iron", due=1000, deps=["XC"]),
        ]
        assigns = [
            _asg("B", "M1", 110, 113, order="OB"),
            _asg("B2", "M2", 150, 200, order="OB"),   # order_max_end[OB] = 200
            _asg("C", "M2", 100, 150),
            _asg("W", "M1", 130, 180),
        ]
        dep_ends = {"XW": 100, "XB": 110, "XB2": 150, "XC": 90}
        n = fifo_swap_ironing(assigns, tasks, {}, dep_ends)
        by = {a["task_id"]: a for a in assigns}
        assert n == 1
        assert by["W"]["start_time"] == 100
        assert by["B"]["end_time"] <= 200   # trong proxy order_max_end
        assert _overlap_free(assigns)
