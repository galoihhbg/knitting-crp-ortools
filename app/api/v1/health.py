import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Liveness + queue-depth probe.

    Returns active Celery task count so Go backend can backpressure when the
    solver queue is saturated.  Falls back gracefully when Redis is unreachable
    (e.g. during startup or network partition) — the service is still 'ok'
    from a liveness perspective; the caller should interpret missing
    `active_tasks` as unknown rather than zero.
    """
    active_tasks: int | None = None
    queue_status = "unknown"

    try:
        from app.core.celery_app import celery_app  # late import avoids circular dep
        inspector = celery_app.control.inspect(timeout=1.0)
        active = inspector.active() or {}
        active_tasks = sum(len(v) for v in active.values())
        queue_status = "ok"
    except Exception as exc:
        logger.warning(f"Health: Celery inspect failed — {exc}")
        queue_status = "degraded"

    return {
        "status": "ok",
        "service": "cp-solver",
        "queue_status": queue_status,
        "active_tasks": active_tasks,
    }
