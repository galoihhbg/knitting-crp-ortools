"""
End-to-end tests for the FastAPI → Celery → webhook pipeline.

These tests run without a live Redis or Go backend:
  - Celery task dispatch is mocked (no broker needed)
  - Webhook POST is mocked (no Go backend needed)
  - Engine.solve() runs for real (validates full solver path)

Three scenarios (matching TechDesign acceptance criteria):
  1. Happy path — solve + webhook delivery succeeds
  2. Webhook transient failure — retries and eventually succeeds
  3. Webhook permanent failure — exhausts retries, returns "Callback Failed"
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tasks.solver_task import _post_webhook, optimize_schedule
from tests.conftest import make_payload

client = TestClient(app)


# ── HTTP route tests ──────────────────────────────────────────────────────────

class TestSolverRoute:

    def test_solve_endpoint_queues_task(self):
        """`POST /api/v1/solve` must return 200 with a celery_task_id."""
        payload = make_payload(n_orders=2, max_search_time=5)

        with patch("app.api.v1.solver_route.optimize_schedule") as mock_task:
            mock_result = MagicMock()
            mock_result.id = "mock-celery-id-123"
            mock_task.delay.return_value = mock_result

            resp = client.post("/api/v1/solve", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "celery_task_id" in body
        assert "job_id" in body
        mock_task.delay.assert_called_once()

    def test_health_returns_ok(self):
        """`GET /health` must always return 200 with status=ok."""
        # health.py does a late import from app.core.celery_app; patch at source.
        with patch("app.core.celery_app.celery_app") as mock_celery:
            mock_celery.control.inspect.return_value.active.return_value = {}
            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "active_tasks" in body

    def test_health_degrades_gracefully_when_redis_down(self):
        """Health endpoint must still return 200 when Redis is unreachable."""
        # Simulate Redis being down by making inspect() raise.
        with patch("app.core.celery_app.celery_app") as mock_celery:
            mock_celery.control.inspect.side_effect = Exception("Redis connection refused")
            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"          # liveness is not Redis-dependent
        assert body["queue_status"] == "degraded"


# ── Webhook delivery tests ────────────────────────────────────────────────────

class TestWebhookDelivery:

    def _ok_response(self):
        mock = MagicMock()
        mock.status_code = 200
        return mock

    def _fail_response(self, code: int = 503):
        mock = MagicMock()
        mock.status_code = code
        mock.text = f"Service Unavailable ({code})"
        return mock

    def test_successful_delivery_returns_true(self):
        """_post_webhook returns True on first 200 response."""
        with patch("app.tasks.solver_task.requests.post",
                   return_value=self._ok_response()) as mock_post:
            result = _post_webhook({"job_id": "J1"}, "task-abc")

        assert result is True
        mock_post.assert_called_once()

    def test_retries_on_transient_failure(self):
        """_post_webhook retries and succeeds on the third attempt."""
        responses = [
            self._fail_response(503),
            self._fail_response(503),
            self._ok_response(),
        ]
        with patch("app.tasks.solver_task.requests.post",
                   side_effect=responses), \
             patch("app.tasks.solver_task.time.sleep"):   # don't actually sleep
            result = _post_webhook({"job_id": "J2"}, "task-xyz")

        assert result is True

    def test_exhausts_retries_returns_false(self):
        """_post_webhook returns False after all attempts fail."""
        with patch("app.tasks.solver_task.requests.post",
                   return_value=self._fail_response(503)), \
             patch("app.tasks.solver_task.time.sleep"):
            result = _post_webhook({"job_id": "J3"}, "task-fail")

        assert result is False

    def test_connection_error_retries(self):
        """_post_webhook retries on RequestException and succeeds."""
        import requests as req_lib
        responses = [
            req_lib.ConnectionError("refused"),
            self._ok_response(),
        ]
        with patch("app.tasks.solver_task.requests.post",
                   side_effect=responses), \
             patch("app.tasks.solver_task.time.sleep"):
            result = _post_webhook({"job_id": "J4"}, "task-conn")

        assert result is True


# ── Full task integration (Engine + webhook mock) ─────────────────────────────

class TestOptimizeScheduleTask:

    def test_happy_path_calls_webhook_once(self):
        """
        optimize_schedule runs the solver for real and POSTs the result.
        Uses Task.apply() which runs synchronously in-process (no broker needed).
        """
        payload = make_payload(n_orders=3, max_search_time=10)
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["body"] = json
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch("app.tasks.solver_task.requests.post", side_effect=fake_post), \
             patch("app.tasks.solver_task.time.sleep"):
            # Task.apply() runs the task body synchronously without a broker.
            optimize_schedule.apply(args=[payload])

        assert captured, "Webhook was never called"
        assert captured["body"]["status"] in ("feasible", "optimal")
        assert "assignments" in captured["body"]
        assert "overloads" in captured["body"]
        assert captured["body"]["job_id"] == payload["job_id"]

    def test_result_contains_solve_metadata(self, payload_smoke):
        """Engine.solve() must return objective_value and solve_time_seconds."""
        from app.engine.model import Engine
        import copy

        result = Engine(copy.deepcopy(payload_smoke)).solve()

        assert "objective_value" in result, "objective_value missing from solve result"
        assert "solve_time_seconds" in result, "solve_time_seconds missing from solve result"
        assert result["solve_time_seconds"] >= 0
