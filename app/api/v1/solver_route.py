from fastapi import APIRouter, HTTPException

from ...schemas.request_schema import SolverPayload
from ...tasks.solver_task import optimize_schedule

router = APIRouter()


@router.post("/api/v1/solve")
async def create_solve_task(payload: SolverPayload):
    """Queue a fresh optimization task to the Celery worker.

    Any reschedule_hint accidentally sent on this endpoint is stripped so the
    queued task is a clean cold-start solve, regardless of what the caller put
    in the payload.
    """
    data = payload.model_dump(by_alias=False)
    data["reschedule_hint"] = None
    task = optimize_schedule.delay(data)
    return {
        "message": "Optimization task queued",
        "celery_task_id": task.id,
        "job_id": payload.job_id,
    }


@router.post("/api/v1/re-schedule")
async def re_schedule_task(payload: SolverPayload):
    """Queue a stability-preserving re-schedule.

    Requires a non-empty `reschedule_hint.previous_assignments`. Without it the
    behaviour is indistinguishable from `/solve`, so we reject early with 400 to
    surface caller mistakes instead of silently degrading.
    """
    if (
        payload.reschedule_hint is None
        or not payload.reschedule_hint.previous_assignments
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "re-schedule requires a non-empty reschedule_hint.previous_assignments. "
                "Use /api/v1/solve for fresh schedules."
            ),
        )
    task = optimize_schedule.delay(payload.model_dump(by_alias=False))
    return {
        "message": "Re-scheduling task queued",
        "celery_task_id": task.id,
        "job_id": payload.job_id,
    }


# Keep the legacy unversioned alias temporarily so existing Go clients don't
# break.  Deprecated — migrate callers to /api/v1/re-schedule.
@router.post("/api/re-schedule")
async def re_schedule_task_legacy(payload: SolverPayload):
    return await re_schedule_task(payload)
