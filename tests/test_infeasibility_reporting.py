import pytest
from app.engine.shared import diagnose_infeasibility

def test_no_compatible_resource_gives_overload():
    tasks = [
        {
            "task_id": "T1",
            "operation": "knitting",
            "compatible_resource_ids": [],
            "duration": 60,
            "is_pinned": False
        }
    ]
    overloads = diagnose_infeasibility(tasks, [], {}, 1000, "infeasible")
    assert len(overloads) == 1
    assert overloads[0]["root_cause_code"] == "NO_COMPATIBLE_RESOURCE"

def test_task_too_long_gives_overload():
    tasks = [
        {
            "task_id": "T1",
            "operation": "knitting",
            "compatible_resource_ids": ["M1"],
            "duration": 2000,
            "is_pinned": False
        }
    ]
    resources = [{"id": "M1"}]
    overloads = diagnose_infeasibility(tasks, resources, {}, 1000, "infeasible")
    assert len(overloads) == 1
    assert overloads[0]["root_cause_code"] == "TASK_TOO_LONG"

def test_start_after_exceeds_horizon():
    tasks = [
        {
            "task_id": "T1",
            "operation": "knitting",
            "compatible_resource_ids": ["M1"],
            "duration": 60,
            "start_after_min": 1500,
            "is_pinned": False
        }
    ]
    resources = [{"id": "M1"}]
    overloads = diagnose_infeasibility(tasks, resources, {}, 1000, "infeasible")
    assert len(overloads) == 1
    assert overloads[0]["root_cause_code"] == "START_AFTER_EXCEEDS_HORIZON"

def test_pinned_conflict_detected():
    tasks = [
        {
            "task_id": "P1",
            "operation": "knitting",
            "is_pinned": True,
            "pinned_machine_id": "M1",
            "pinned_start_time": 100,
            "pinned_end_time": 200
        },
        {
            "task_id": "P2",
            "operation": "knitting",
            "is_pinned": True,
            "pinned_machine_id": "M1",
            "pinned_start_time": 150,
            "pinned_end_time": 250
        }
    ]
    overloads = diagnose_infeasibility(tasks, [{"id": "M1"}], {}, 1000, "infeasible")
    assert len(overloads) == 2
    assert overloads[0]["root_cause_code"] == "PINNED_TASK_CONFLICT"
    assert overloads[1]["root_cause_code"] == "PINNED_TASK_CONFLICT"

def test_timeout_gives_overloads():
    tasks = [
        {
            "task_id": "T1",
            "operation": "knitting",
            "compatible_resource_ids": ["M1"],
            "duration": 60,
            "is_pinned": False
        }
    ]
    overloads = diagnose_infeasibility(tasks, [{"id": "M1"}], {}, 1000, "timeout")
    assert len(overloads) == 1
    assert overloads[0]["root_cause_code"] == "SOLVER_TIMEOUT"

def test_machine_overload_detection():
    # 20 tasks of 60 mins each = 1200 mins.
    # 1 machine, horizon = 1000. 1200 > 1000 => Overload
    tasks = [
        {
            "task_id": f"T{i}",
            "operation": "knitting",
            "compatible_resource_ids": ["M1"],
            "duration": 60,
            "is_pinned": False
        }
        for i in range(20)
    ]
    resources = [{"id": "M1"}]
    overloads = diagnose_infeasibility(tasks, resources, {}, 1000, "infeasible")
    assert len(overloads) == 20
    assert overloads[0]["root_cause_code"] == "MACHINE_OVERLOAD"

def test_partial_infeasibility_only_bad_tasks():
    tasks = [
        {
            "task_id": "GOOD",
            "operation": "knitting",
            "compatible_resource_ids": ["M1"],
            "duration": 60,
            "is_pinned": False
        },
        {
            "task_id": "BAD",
            "operation": "knitting",
            "compatible_resource_ids": [],
            "duration": 60,
            "is_pinned": False
        }
    ]
    resources = [{"id": "M1"}]
    # 2 tasks total = 120 mins. 1 machine. horizon = 1000 => no machine overload
    overloads = diagnose_infeasibility(tasks, resources, {}, 1000, "infeasible")
    assert len(overloads) == 2
    good_ol = next(o for o in overloads if o["task_id"] == "GOOD")
    bad_ol = next(o for o in overloads if o["task_id"] == "BAD")
    
    assert bad_ol["root_cause_code"] == "NO_COMPATIBLE_RESOURCE"
    # GOOD falls back to CAPACITY_FULL because we know it's infeasible overall but this task seems fine
    assert good_ol["root_cause_code"] == "CAPACITY_FULL"

def test_overload_fields_complete():
    tasks = [
        {
            "task_id": "T1",
            "original_order_id": "ORD-1",
            "operation": "knitting",
            "compatible_resource_ids": [],
            "duration": 60,
            "qty": 42.5,
            "is_pinned": False
        }
    ]
    overloads = diagnose_infeasibility(tasks, [], {}, 1000, "infeasible")
    assert len(overloads) == 1
    o = overloads[0]
    assert o["task_id"] == "T1"
    assert o["order_id"] == "ORD-1"
    assert o["status"] == "UNSCHEDULABLE"
    assert o["delay_minutes"] == 0
    assert o["root_cause_code"] == "NO_COMPATIBLE_RESOURCE"
    assert o["bottleneck_resource_id"] is None
    assert o["quantity"] == 42.5

# ── E2E integration tests for infeasible behavior ───────────────────────────

from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock
from tests.conftest import make_payload

client = TestClient(app)

def test_infeasible_webhook_has_overloads():
    # Construct an explicitly infeasible payload
    # e.g. task duration > horizon
    payload = make_payload(n_orders=1, max_search_time=5)
    payload["config"]["horizon_minutes"] = 100
    payload["tasks"][0]["duration"] = 200  # task duration > horizon
    
    captured = {}
    def fake_post(url, json, timeout):
        captured["body"] = json
        resp = MagicMock()
        resp.status_code = 200
        return resp

    def fake_solve_knitting(*args, **kwargs):
        from app.engine.phases.phase1_knitting import Phase1Result
        return Phase1Result(status="infeasible", assignments=[], overloads=[])
    
    def fake_compute_global_horizon(*args, **kwargs):
        return 100
    
    from app.tasks.solver_task import optimize_schedule
    with patch("app.tasks.solver_task.requests.post", side_effect=fake_post), \
         patch("app.tasks.solver_task.time.sleep"), \
         patch("app.engine.pipeline.solve_knitting", side_effect=fake_solve_knitting), \
         patch("app.engine.pipeline.compute_global_horizon", side_effect=fake_compute_global_horizon):
        optimize_schedule.apply(args=[payload])

    assert captured
    body = captured["body"]
    assert body["status"] == "infeasible"
    assert len(body["overloads"]) > 0
    
    # K1-ORDER_000 will be TASK_TOO_LONG
    ol = next((o for o in body["overloads"] if "K1" in o["task_id"]), None)
    assert ol is not None
    assert ol["root_cause_code"] in ("TASK_TOO_LONG", "MACHINE_OVERLOAD")

def test_exception_path_sends_overloads():
    payload = make_payload(n_orders=1)
    
    captured = {}
    def fake_post(url, json, timeout):
        captured["body"] = json
        resp = MagicMock()
        resp.status_code = 200
        return resp
        
    from app.tasks.solver_task import optimize_schedule
    with patch("app.engine.model.Engine.solve", side_effect=Exception("Boom")), \
         patch("app.tasks.solver_task.requests.post", side_effect=fake_post), \
         patch("app.tasks.solver_task.time.sleep"):
        try:
            optimize_schedule.apply(args=[payload])
        except Exception:
            pass # Task raises it further

    assert captured
    body = captured["body"]
    assert body["status"] == "infeasible"
    assert body["infeasibility_reason"] == "Solver exception: Boom"
    # Should run diagnose_infeasibility which will probably just give CAPACITY_FULL or something
    assert len(body["overloads"]) > 0
