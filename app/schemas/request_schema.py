from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional



class TimeWindow(BaseModel):
    start: int
    end: int


class SolverResource(BaseModel):
    id: str
    type: str = "serial"
    capacity: int = 1
    operation: Optional[str] = None
    unavailability: List[TimeWindow] = Field(default_factory=list)
    design_item_id: str = ""
    color_config: str = ""
    available_at_min: Optional[int] = 0


class Machine(BaseModel):
    id: str
    design_item_id: str
    color_config: str


class SolverTask(BaseModel):
    task_id: str = Field(alias="task_id")
    original_order_id: str = Field(alias="original_order_id")
    group_id: str = Field(alias="group_id")
    operation: str = Field(alias="operation")
    qty: float = Field(alias="qty")
    total_qty: float = Field(alias="total_qty")
    priority: int = Field(alias="priority")
    final_depends_on: List[str] = Field(default=[], alias="final_depends_on")
    start_after_min: int = Field(default=0, alias="start_after_min")
    due_at_min: int = Field(default=0, alias="due_at_min")
    duration: int = Field(alias="duration")
    is_batch: bool = Field(default=False, alias="is_batch")
    sub_tasks: Optional[List["SolverTask"]] = Field(default=None, alias="sub_tasks")
    design_item_id: str = Field(alias="design_item_id")
    color_config: str = Field(alias="color_config")
    color: str = Field(default="", alias="color")
    substance: str = Field(default="", alias="substance")
    compatible_resource_ids: List[str] = Field(default=[], alias="compatible_resource_ids")
    wait_offsets: Optional[Dict[str, int]] = Field(default=None, alias="WaitOffsets")

    is_slice: bool = Field(default=False, alias="is_slice")
    slice_index: int = Field(default=0, alias="SliceIndex")
    parent_task_id: str = Field(default="", alias="parent_task_id")

    is_pinned: bool = Field(default=False, alias="is_pinned")
    pinned_machine_id: Optional[str] = Field(default=None, alias="pinned_machine_id")
    pinned_start_time: Optional[int] = Field(default=None, alias="pinned_start_time")
    pinned_end_time: Optional[int] = Field(default=None, alias="pinned_end_time")
    demand: int = Field(default=1, alias="demand")
    material_demands: Dict[str, int] = Field(default_factory=dict, alias="material_demands")

    class Config:
        populate_by_name = True


class SolverConfig(BaseModel):
    horizon_minutes: int = 57600
    max_search_time: int = 300
    max_factory_machines: int = 40
    random_seed: int = 42       # Fixed seed for deterministic output across runs
    num_search_workers: int = 8  # Set to 1 for byte-identical replay; 8 for production speed
    washing_batch_capacity: int = 10
    max_washing_batches: Optional[int] = None
    # Số slot (K) tối đa cho washing batching.
    # Nếu set: ghi đè hoàn toàn auto-calculation (K = min(n_tasks, giá trị này)).
    # Nếu None: tự tính từ ceil(total_qty / capacity) × 3, tối thiểu 5.
    washing_num_slots: Optional[int] = None
    # Virtual-time points (minutes) where each work shift ends.
    # Washing tasks must complete before, or start at/after, each boundary
    # because the backend strips breaks from the timeline and washing cannot be interrupted.
    shift_ends_min: List[int] = Field(default_factory=list)


class SolverPayload(BaseModel):
    job_id: str
    config: SolverConfig
    machines: List[Machine]
    resources: List[SolverResource] = Field(default_factory=list)
    tasks: List[SolverTask]
    material_capacities: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-material creel capacity: material_code → total available rolls/slots",
    )
