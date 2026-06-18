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


class YarnConsumption(BaseModel):
    """Sợi tiêu thụ cho một task (chuẩn bị cho post-pass dyelot — chưa tiêu thụ).

    is_main: True = sợi chính (được cấp dyelot); False = sợi phụ (không cấp dyelot).
    Default True để tương thích payload cũ chưa có cờ — khi đó mọi entry là sợi chính.
    """
    vi: str
    kg: float
    # slots = số creel position (cone) task lắp cho vi (Go MinSlots). Dùng cho
    # creel-up gross trong dyelot allocator; 0/absent = payload cũ → bỏ creel-up.
    slots: int = 0
    is_main: bool = True


class SolverTask(BaseModel):
    task_id: str = Field(alias="task_id")
    original_order_id: str = Field(alias="original_order_id")
    group_id: str = Field(alias="group_id")
    # Đơn (sales-order) mà task này thuộc về — khóa gom dyelot: mọi batch/panel
    # cùng order_group_id PHẢI dùng chung 1 dyelot per VI (tránh lệch màu khi ghép
    # thành 1 sản phẩm).  Một đơn có thể bị rolling-wave tách thành nhiều batch
    # (vd BATCH_0-665 + BATCH_0-666 cùng đơn "W9xTMuuLxR-1-200-200").  Để trống ""
    # khi payload cũ chưa gửi — khi đó dyelot gom theo original_order_id như cũ.
    order_group_id: str = Field(default="", alias="order_group_id")
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
    main_yarn_consumption: List[YarnConsumption] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class SolverConfig(BaseModel):
    horizon_minutes: int = 57600
    max_search_time: int = 300
    max_factory_machines: int = 40
    random_seed: int = 42       # Fixed seed for deterministic output across runs
    num_search_workers: int = 8  # Set to 1 for byte-identical replay; 8 for production speed
    # Primary stop criterion: deterministic units (≈ single-core seconds).
    # None → auto-derive as max_search_time × num_search_workers (sum-across-workers
    # heuristic).  Set explicit value to tune for unusual payload sizes / hardware.
    # See make_solver() for full semantics.
    max_deterministic_time: Optional[float] = None
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
    # Two-pass same-qty re-link refinement (cold solve only).  Pass 2 relaxes the
    # linking floor so a slice may consume the earliest-finished knitting panel of
    # the SAME (component, qty) bucket instead of its index-paired panel, under
    # per-task Pareto end-caps + a whole-pipeline pointwise verify: the refined
    # schedule is accepted only if NO task finishes later than pass 1.
    enable_sameqty_relink: bool = True
    # Cold-solve knitting EDD warm-start (hints-only AddHint seed; zero new
    # constraints/objective terms).  Cold knitting routinely stops at FEASIBLE
    # with due-inversions on the machines; an earliest-due-date incumbent seed
    # removed 79-85% of total order lateness in offline replays.
    enable_edd_knitting_hint: bool = True
    # Linking worker load-balance post-pass (cold solve only).  Re-assigns linking
    # tasks across interchangeable linking machines to even out per-worker load,
    # keeping every task's [start, end] fixed — downstream byte-identical, no order
    # finishes later.  Fixes severe worker idle/imbalance (machine-load stdev 965→40
    # on real payloads).
    enable_linking_balance: bool = True
    # Panel co-completion (cold solve only).  A linking SLICE_k depends on the
    # knitting batches of EVERY component (front/back/sleeve …) at the same index;
    # linking can only start once the LAST of that set finishes.  The solver
    # otherwise has no incentive to finish a whole panel together, so component
    # ends drift apart (measured ~1228 min mean spread on a 660-task payload) and
    # linking waits on the straggler.  This phase-1 objective term minimises each
    # panel's max component-end (the BOM-ready time that gates linking), pulling the
    # straggler component earlier so its linking slice can start sooner — WITHOUT
    # extra machines (a sequencing nudge).  Weighted at the flow/slice-sync scale,
    # so it is a commensurate secondary term (the dominant lateness penalty, ×10^7
    # per minute vs panel ×10^4, still wins) — architecturally identical to the
    # already-shipped apply_order_flow / apply_slice_sync objectives.
    # Measured on cold payloads (78/322/612 tasks, production budget): component-end
    # spread −61..−65%, linking starts −11..−25%, total lateness unchanged.
    enable_panel_sync_objective: bool = True


class PreviousAssignment(BaseModel):
    """One task↔machine assignment from a prior solve, used to seed re-schedule hints."""
    task_id: str
    machine_id: str
    start_time: int
    end_time: int
    original_order_id: str = ""


class RescheduleHint(BaseModel):
    """Optional hint payload that drives the re-schedule stability mechanism.

    Calibration (B.4 — measured on the symmetric fixture):
      * start tie-breaker (per task per start-minute) = max(1, 10**(6-priority)//100).
        With priority=3 (default) → 10.  Uniform priority=1 → 1000.
      * lateness coeff (per task per minute late)     = 10**(6-priority)*100.
        With priority=3 (default) → 100_000.

    Calibration window required: start_coeff  «  w_time  «  lateness_coeff.
      → w_time = 500 (50× the typical start tie-breaker, 200× below lateness).
      → w_machine = 50_000 (100× w_time, ≈ 5_000 start-minutes worth — set
        empirically after observing production payload (732 tasks, 110 machines,
        60s search) where w_machine=20_000 left keep_rate at 86%.  Combined
        with solver `repair_hint=True` in make_solver, this raises keep_rate
        toward the ≥95% target.  Still < lateness coeff (100_000) so no
        accidental "ổn định thay vì kịp deadline" tradeoff.
    """
    previous_assignments: List[PreviousAssignment] = Field(default_factory=list)
    stability_weight_time_per_min: int = 500
    stability_weight_machine_swap: int = 50_000
    match_by_order_fallback: bool = True


class DyelotStock(BaseModel):
    """Tồn kho theo dyelot (chuẩn bị cho post-pass dyelot — chưa tiêu thụ)."""
    vi: str
    dyelot: str
    remaining_kg: float
    packing_size: float


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
    dyelot_stock: List[DyelotStock] = Field(default_factory=list)
    # Default roll size (kg) per thread vi, incl. vis with zero current stock.
    # The dyelot post-pass uses it to size a fresh dyelot for a zero-stock vi
    # (whole-roll / creel-up gross needs a roll size). Empty → net floor fallback.
    vi_packing_size: Dict[str, float] = Field(default_factory=dict)
    reschedule_hint: Optional[RescheduleHint] = None
