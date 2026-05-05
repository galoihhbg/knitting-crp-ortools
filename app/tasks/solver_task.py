import json
import logging
import os
import time

import requests

from ..core.celery_app import celery_app
from ..engine.model import Engine
from ..engine.utils import filter_dummy_tasks, filter_dummy_overloads

logger = logging.getLogger(__name__)

WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "http://backend:8082/api/webhook/solver")

# Exponential backoff schedule for webhook retries (seconds between attempts).
# Total max wait before giving up: 2 + 5 + 15 = 22 s on top of the base timeout.
_RETRY_DELAYS = (2, 5, 15)

_ROOT_CAUSE_CODES = (
    "MACHINE_OVERLOAD",
    "WORKFORCE_SHORTAGE",
    "PINNED_TASK_CONFLICT",
    "CAPACITY_FULL",
    "NO_COMPATIBLE_RESOURCE",
)


def _post_webhook(response_data: dict, task_id: str) -> bool:
    """
    POST result to Go backend with exponential-backoff retries.

    Returns True on the first successful 2xx response.
    Logs a warning after each failed attempt and an error after all retries
    are exhausted.  Never raises — the caller decides what to return to Celery.
    """
    delays = list(_RETRY_DELAYS)
    max_attempts = len(delays) + 1

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(WEBHOOK_URL, json=response_data, timeout=10)
            if resp.status_code == 200:
                logger.info(f"[Task {task_id}] ✅ Webhook delivered (attempt {attempt})")
                return True
            logger.warning(
                f"[Task {task_id}] ⚠️ Webhook attempt {attempt}/{max_attempts} "
                f"non-2xx: {resp.status_code}"
            )
        except requests.RequestException as exc:
            logger.warning(
                f"[Task {task_id}] ⚠️ Webhook attempt {attempt}/{max_attempts} "
                f"connection error: {exc}"
            )

        if delays:
            time.sleep(delays.pop(0))

    logger.error(
        f"[Task {task_id}] ❌ Webhook failed after {max_attempts} attempts — "
        f"job_id={response_data.get('job_id')}"
    )
    return False


@celery_app.task(bind=True, name="optimize_schedule")
def optimize_schedule(self, payload: dict):
    try:
        job_id = payload.get("job_id")
        logger.info(f"[Task {self.request.id}] Starting job_id={job_id}")

        try:
            os.makedirs("/app/logs", exist_ok=True)
            with open(f"/app/logs/solver_input_{job_id}.json", "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"[Task {self.request.id}] Dumped input payload to /app/logs/solver_input_{job_id}.json")
        except Exception as e:
            logger.error(f"[Task {self.request.id}] Failed to dump input payload: {e}")

        result = Engine(payload).solve()

        raw_assignments = result.get("assignments", [])
        raw_overloads = result.get("overloads", [])
        clean_assignments = filter_dummy_tasks(raw_assignments)
        clean_overloads = filter_dummy_overloads(raw_overloads)

        # Structured solve-complete log (parseable by log aggregators)
        logger.info({
            "event": "solve_complete",
            "job_id": job_id,
            "celery_task_id": self.request.id,
            "status": result["status"],
            "n_assignments": len(clean_assignments),
            "n_overloads": len(clean_overloads),
            "root_cause_breakdown": {
                code: sum(1 for o in clean_overloads if o.get("root_cause_code") == code)
                for code in _ROOT_CAUSE_CODES
            },
            "objective_value": result.get("objective_value"),
            "solve_time_seconds": result.get("solve_time_seconds"),
        })

        response_data = {
            "job_id": job_id,
            "task_id": self.request.id,
            "status": result["status"],
            "assignments": clean_assignments,
            "overloads": clean_overloads,
        }

        try:
            os.makedirs("/app/logs", exist_ok=True)
            with open(f"/app/logs/solver_output_{job_id}.json", "w") as f:
                json.dump(response_data, f, indent=2)
            logger.info(f"[Task {self.request.id}] Dumped output payload to /app/logs/solver_output_{job_id}.json")
        except Exception as e:
            logger.error(f"[Task {self.request.id}] Failed to dump output payload: {e}")

        success = _post_webhook(response_data, self.request.id)
        return "Callback Successful" if success else "Callback Failed"

    except Exception as exc:
        logger.error(f"[Task {self.request.id}] Error: {exc}", exc_info=True)
        # Notify the Go backend so it doesn't wait forever for a callback that
        # will never arrive.  The task is still re-raised so Celery marks it failed.
        _post_webhook(
            {
                "job_id": payload.get("job_id"),
                "task_id": self.request.id,
                "status": "infeasible",
                "assignments": [],
                "overloads": [],
            },
            self.request.id,
        )
        raise
