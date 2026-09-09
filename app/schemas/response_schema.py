from pydantic import BaseModel
from typing import List, Dict, Any


class Assignment(BaseModel):
    task_id: str
    machine_id: str
    start_time: int
    end_time: int
    group_id: str = ""
    order_id: str = ""
    quantity: float = 0
    status: str = "ON_TIME"
    batch_slot_id: str = ""


class Overload(BaseModel):
    task_id: str
    order_id: str = ""
    status: str = "LATE"
    delay_minutes: int
    root_cause_code: str = "CAPACITY_FULL"
    bottleneck_resource_id: str = ""
    quantity: float = 0


class SolverResponse(BaseModel):
    job_id: str
    task_id: str
    status: str
    assignments: List[Dict[str, Any]] = []
    overloads: List[Dict[str, Any]] = []
