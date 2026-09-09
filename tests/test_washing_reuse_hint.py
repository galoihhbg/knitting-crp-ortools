"""_washing_reuse_from_hint — reuse washing từ hint external reschedule.

Bất động điểm: workload giặt không đổi → giữ nguyên văn vị trí hint (fix
permutation drift, xem project_washing_permutation_drift).  Test khoá các
điều kiện từ-chối (→ None → re-solve như cũ) và bộ field tái tạo.
"""
from app.engine.pipeline import _washing_reuse_from_hint


def _wtask(tid, due=1000, pinned=False, **kw):
    t = {
        "task_id": tid, "operation": "Washing", "due_at_min": due,
        "group_id": "G1", "original_order_id": f"O-{tid}", "qty": 5.0,
        "is_pinned": pinned, "final_depends_on": [],
    }
    t.update(kw)
    return t


def _prev(tid, m="W_WASHING_01", s=100, e=160):
    return {"task_id": tid, "machine_id": m, "start_time": s, "end_time": e,
            "original_order_id": f"O-{tid}"}


class TestWashingReuseFromHint:
    def test_full_match_builds_verbatim(self):
        tasks = [_wtask("W1"), _wtask("W2", due=150)]
        hint = {"previous_assignments": [_prev("W1"), _prev("W2", s=100, e=160)]}
        out = _washing_reuse_from_hint(hint, tasks)
        assert out is not None and len(out) == 2
        by = {a["task_id"]: a for a in out}
        a = by["W1"]
        assert (a["machine_id"], a["start_time"], a["end_time"]) == ("W_WASHING_01", 100, 160)
        assert a["group_id"] == "G1" and a["order_id"] == "O-W1" and a["quantity"] == 5.0
        assert a["batch_slot_id"] == "keep_100"
        assert a["status"] == "ON_TIME"
        assert by["W2"]["status"] == "LATE"  # end 160 > due 150 (due tính lại)

    def test_new_washing_task_not_in_hint_returns_none(self):
        tasks = [_wtask("W1"), _wtask("W_NEW")]
        hint = {"previous_assignments": [_prev("W1")]}
        assert _washing_reuse_from_hint(hint, tasks) is None

    def test_prev_missing_fields_returns_none(self):
        tasks = [_wtask("W1")]
        hint = {"previous_assignments": [
            {"task_id": "W1", "machine_id": None, "start_time": 100, "end_time": 160},
        ]}
        assert _washing_reuse_from_hint(hint, tasks) is None

    def test_pinned_mismatch_returns_none(self):
        tasks = [_wtask("W1", pinned=True, pinned_start_time=90,
                        pinned_end_time=150, pinned_machine_id="W_WASHING_01")]
        hint = {"previous_assignments": [_prev("W1", s=100, e=160)]}
        assert _washing_reuse_from_hint(hint, tasks) is None

    def test_pinned_matching_ok(self):
        tasks = [_wtask("W1", pinned=True, pinned_start_time=100,
                        pinned_end_time=160, pinned_machine_id="W_WASHING_01")]
        hint = {"previous_assignments": [_prev("W1", s=100, e=160)]}
        out = _washing_reuse_from_hint(hint, tasks)
        assert out is not None and out[0]["start_time"] == 100

    def test_empty_tasks_returns_none(self):
        assert _washing_reuse_from_hint({"previous_assignments": []}, []) is None
