import json
import logging
import os
import time

import requests

from ..core.celery_app import celery_app
from ..engine.model import Engine
from ..engine.shared import diagnose_infeasibility
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


def _truthy_env(name: str, default: bool) -> bool:
    """Read a boolean env var with a sane default (1/true/yes/on → True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _hint_from_assignments(assignments: list) -> dict:
    """Dựng reschedule_hint từ assignments của lượt solve trước.

    Tái hiện đúng cái Go gửi lại ở lần re-schedule kế tiếp: mỗi assignment
    thành một previous_assignment khớp theo task_id. order_id → original_order_id.
    Chỉ giữ assignment có đủ task_id + machine_id (bỏ qua bản ghi thiếu trường).

    Đánh dấu `stabilize_pass=True`: đây là lượt 2 NỘI BỘ của double-solve (không phải
    re-schedule thật từ Go), nên pipeline được phép chạy các post-pass nén (linking/
    ironing left-shift, knitting balance) để UI nhận lịch KHÍT — trong khi re-schedule
    thật (Go gửi kế hoạch cũ) vẫn ưu tiên ỔN ĐỊNH máy, bỏ qua các pass dời máy đó.
    """
    previous = [
        {
            "task_id": a["task_id"],
            "machine_id": a["machine_id"],
            "start_time": int(a["start_time"]),
            "end_time": int(a["end_time"]),
            "original_order_id": a.get("order_id", "") or "",
        }
        for a in assignments
        if a.get("task_id") and a.get("machine_id")
    ]
    return {"previous_assignments": previous, "stabilize_pass": True}


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

        # ─── MOCK MODE ────────────────────────────────────────────────────────
        # Khi MOCK_RESPONSE_FILE set, bỏ qua solver hoàn toàn, trả về nguyên
        # nội dung file đó cho Go (giữ nguyên job_id từ payload để Go route được).
        # Dùng để debug downstream (vd: stock check) với dữ liệu cố định.
        mock_file = os.getenv("MOCK_RESPONSE_FILE")
        if mock_file:
            try:
                with open(mock_file) as f:
                    mock_data = json.load(f)
                mock_data["job_id"] = job_id
                mock_data["task_id"] = self.request.id
                logger.warning(
                    f"[Task {self.request.id}] 🧪 MOCK_RESPONSE_FILE active: "
                    f"trả về nguyên nội dung {mock_file} (bỏ qua solver)"
                )
                _post_webhook(mock_data, self.request.id)
                return "Mock Callback Sent"
            except Exception as e:
                logger.error(f"[Task {self.request.id}] MOCK_RESPONSE_FILE lỗi: {e}")
                # rớt xuống chạy solver bình thường

        result = Engine(payload).solve()

        # ─── DOUBLE-SOLVE STABILIZATION ──────────────────────────────────────
        # Lịch lần 1 (cold) khác lần 2 một chút vì lần 2 thực chất là re-schedule
        # của cold (kích hoạt conditional hard-keep ở phase 4); từ lần 2 trở đi ổn
        # định. Để UI nhận thẳng lịch ổn định, ta tự chạy lượt 2 nội bộ: dùng kết
        # quả lượt 1 làm reschedule_hint rồi solve lại, trả về lượt 2.
        #
        # Chỉ áp dụng cho cold-solve (payload chưa có reschedule_hint sẵn — re-schedule
        # đã ổn định) và khi lượt 1 cho ra assignments khả dụng. Bật/tắt qua
        # ENABLE_DOUBLE_SOLVE (mặc định bật).
        if (
            _truthy_env("ENABLE_DOUBLE_SOLVE", True)
            and not payload.get("reschedule_hint")
            and result.get("status") in ("feasible", "optimal")
            and result.get("assignments")
        ):
            try:
                stabilize_payload = dict(payload)
                stab_hint = _hint_from_assignments(result["assignments"])
                # Lượt 2 giải LẠI washing thường KẸT ở FEASIBLE → gom quá đà đồ
                # sẵn-sàng-sớm vào mẻ muộn, tạo khe máy giặt đứng im hàng ngày (lượt 1
                # cold đã left-shift đúng rồi).  Nên KHÔNG giặt lại ở lượt 2: đính kèm
                # nguyên assignments washing đầy đủ của lượt 1 (giữ group_id/batch_slot_id)
                # để pipeline tái dùng thay vì re-solve.  Bonus: bỏ được lượt-2-giặt chậm.
                _wash_ids = {
                    t["task_id"] for t in payload.get("tasks", [])
                    if str(t.get("operation", "")).lower() == "washing"
                }
                stab_hint["_pass1_washing_full"] = [
                    dict(a) for a in result["assignments"]
                    if a.get("task_id") in _wash_ids
                ]
                # Mang theo cả overloads washing của lượt 1 (chẩn đoán LATE/root-cause
                # cho Go) — vì lượt 2 không giặt lại nên không tự sinh ra chúng nữa.
                stab_hint["_pass1_washing_overloads"] = [
                    dict(o) for o in result.get("overloads", [])
                    if o.get("task_id") in _wash_ids
                ]
                stabilize_payload["reschedule_hint"] = stab_hint
                logger.info(
                    f"[Task {self.request.id}] 🔁 Double-solve: chạy lượt 2 với "
                    f"{len(stabilize_payload['reschedule_hint']['previous_assignments'])} "
                    f"previous assignments để ổn định lịch trước khi trả UI"
                )
                result = Engine(stabilize_payload).solve()
            except Exception as exc:
                # Lượt 2 lỗi → giữ nguyên lượt 1 (vẫn là lịch hợp lệ), không làm hỏng job.
                logger.error(
                    f"[Task {self.request.id}] ⚠️ Double-solve lượt 2 lỗi, "
                    f"dùng kết quả lượt 1: {exc}",
                    exc_info=True,
                )

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
            # Dyelot allocation post-pass output (empty lists when no dyelot_stock).
            "order_dyelot_assignment": result.get("order_dyelot_assignment", []),
            "dyelot_flush_points": result.get("dyelot_flush_points", []),
            "dyelot_unassigned": result.get("dyelot_unassigned", []),
            "dyelot_shortage": result.get("dyelot_shortage", []),
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
        tasks = payload.get("tasks", [])
        resources = payload.get("resources", [])
        config = payload.get("config", {})
        horizon = int(config.get("horizon_minutes", 40320))
        exc_overloads = diagnose_infeasibility(tasks, resources, config, horizon, "infeasible")
        # Notify the Go backend so it doesn't wait forever for a callback that
        # will never arrive.  The task is still re-raised so Celery marks it failed.
        _post_webhook(
            {
                "job_id": payload.get("job_id"),
                "task_id": self.request.id,
                "status": "infeasible",
                "infeasibility_reason": f"Solver exception: {exc}",
                "assignments": [],
                "overloads": exc_overloads,
            },
            self.request.id,
        )
        raise
