from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .celery_app import celery_app
from .tasks import optimize_schedule

app = FastAPI()

# Type definitions matching Go structs
class TimeWindow(BaseModel):
    start: int
    end: int

class SolverResource(BaseModel):
    id: str
    type: str = "serial"
    capacity: int = 1
    operation: Optional[str] = None
    unavailability: List[TimeWindow] = Field(default_factory=list)

class MachineRoute(BaseModel):
    operation: str
    design_item_id: str
    duration: float
    setup_time: float = 0.0

class Machine(BaseModel):
    id: str
    capacity: Optional[int] = None
    type: Optional[str] = None
    worker_req: int = 1
    routing: List[MachineRoute]

class SolverTask(BaseModel):
    task_id: str = Field(alias="task_id")
    original_order_id: str = Field(alias="original_order_id")
    group_id: str = Field(alias="group_id")
    operation: str = Field(alias="operation")
    qty: float = Field(alias="qty")
    total_qty: float = Field(alias="total_qty")
    priority: int = Field(alias="priority")
    original_depends_on: List[str] = Field(default=[], alias="original_depends_on")
    final_depends_on: List[str] = Field(default=[], alias="final_depends_on")
    start_after_min: int = Field(default=0, alias="start_after_min")
    due_at_min: int = Field(default=0, alias="due_at_min")
    duration: int = Field(alias="duration")
    is_slice: bool = Field(default=False, alias="is_slice")
    parent_task_id: str = Field(default="", alias="parent_task_id")
    internal_dep: str = Field(default="", alias="internal_dep")
    slice_index: int = Field(default=0, alias="slice_index")
    is_batch: bool = Field(default=False, alias="is_batch")
    sub_tasks: Optional[List['SolverTask']] = Field(default=None, alias="sub_tasks")
    design_item_id: str = Field(alias="design_item_id")
    compatible_resource_ids: List[str] = Field(default=[], alias="compatible_resource_ids")
    class Config:
        populate_by_name = True

class SolverConfig(BaseModel):
    horizon_minutes: int = 57600
    max_search_time: int = 300
    setup_time_minutes: int = 60

class SolverPayload(BaseModel):
    job_id: str
    config: Dict[str, Any]
    machines: List[Machine]
    resources: List[SolverResource] = Field(default_factory=list)
    tasks: List[SolverTask]

@app.post("/api/v1/solve")
async def create_solve_task(payload: SolverPayload):
    """Queue optimization task to Celery"""
    task = optimize_schedule.delay(payload.model_dump(by_alias=False))
    
    return {
        "message": "Optimization task queued",
        "celery_task_id": task.id,
        "job_id": payload.job_id
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cp-solver"}

# Enable SolverTask to reference itself
SolverTask.model_rebuild()