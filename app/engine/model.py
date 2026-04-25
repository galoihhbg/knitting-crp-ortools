import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

from ortools.sat.python import cp_model

from .builder import TaskModelBuilder

# Per-run log file so each solve session is independently traceable
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) or "."
_log_dir = os.path.join(_base_dir, "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, f"scheduling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger(__name__)
logger.handlers.clear()
logger.setLevel(logging.INFO)

_file_handler = logging.FileHandler(_log_file)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(_console_handler)

logger.info(f"🔵 Logs will be saved to: {_log_file}")


class Engine:
    """
    Orchestrates the CP-SAT solving pipeline.
    Parses the raw JSON payload from Go, delegates model construction to
    TaskModelBuilder, then runs the solver and returns the result dict.
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        logger.info("📥 Parsing config from payload...")
        raw_config = payload.get("config", "{}")
        if isinstance(raw_config, str):
            try:
                self.config: Dict[str, Any] = json.loads(raw_config)
            except Exception:
                self.config = {}
        else:
            self.config = raw_config

        machines_data = payload.get("machines", [])
        logger.info(f"📦 RECEIVED {len(machines_data)} MACHINES FROM PAYLOAD")

        # Capture each machine's current design/color state for affinity scoring
        self.machine_states: Dict[str, Dict[str, str]] = {}
        for m in machines_data:
            m_id = m.get("id")
            if m_id:
                self.machine_states[m_id] = {
                    "current_design": m.get("design_item_id", ""),
                    "current_color": m.get("color_config", ""),
                }

        self.resources = payload.get("resources", [])
        self.tasks = payload.get("tasks", [])
        # Top-level material creel capacities: material_code → max concurrent rolls/slots.
        # Empty dict when the Go backend does not send the field (feature disabled).
        self.material_capacities: Dict[str, int] = payload.get("material_capacities") or {}
        if self.material_capacities:
            logger.info(
                f"📦 material_capacities received: {self.material_capacities} "
                f"({len(self.material_capacities)} material(s))"
            )
        else:
            logger.info("📦 material_capacities: not present in payload — material constraints disabled")

    def solve(self) -> Dict[str, Any]:
        if not self.tasks:
            return {"status": "feasible", "assignments": [], "overloads": []}

        builder = TaskModelBuilder(
            self.config, self.resources, self.tasks, self.machine_states,
            self.material_capacities,
        )
        builder.build_time_variables()

        # If any tasks have no compatible resource, the schedule is infeasible.
        # Return structured overloads so the Go backend / operator can act on them.
        if builder.no_resource_tasks:
            ids = [t["task_id"] for t in builder.no_resource_tasks]
            logger.error(
                f"❌ {len(ids)} task(s) have no compatible resource — "
                f"returning infeasible: {ids}"
            )
            return {
                "status": "infeasible",
                "assignments": [],
                "overloads": [
                    {
                        "task_id": t["task_id"],
                        "order_id": t.get("original_order_id", ""),
                        "status": "UNSCHEDULABLE",
                        "delay_minutes": 0,
                        "root_cause_code": "NO_COMPATIBLE_RESOURCE",
                        "bottleneck_resource_id": None,
                        "quantity": t.get("qty", 0),
                    }
                    for t in builder.no_resource_tasks
                ],
                "objective_value": None,
                "solve_time_seconds": 0.0,
            }

        (
            builder
            .build_resource_allocations()
            .build_material_constraints()
            .apply_routing_constraints()
            .apply_dependency_constraints()
            .apply_batch_offset_constraints()
            .apply_smart_batching_constraints()
            .apply_shift_boundary_constraints()
            .define_objective()
        )

        # After full model build, check again — build_resource_allocations may discover
        # tasks whose compatible_resource_ids are all absent from resource_map.
        if builder.no_resource_tasks:
            ids = [t["task_id"] for t in builder.no_resource_tasks]
            logger.error(
                f"❌ {len(ids)} task(s) found no matching resources in resource_map — "
                f"returning infeasible: {ids}"
            )
            return {
                "status": "infeasible",
                "assignments": [],
                "overloads": [
                    {
                        "task_id": t["task_id"],
                        "order_id": t.get("original_order_id", ""),
                        "status": "UNSCHEDULABLE",
                        "delay_minutes": 0,
                        "root_cause_code": "NO_COMPATIBLE_RESOURCE",
                        "bottleneck_resource_id": None,
                        "quantity": t.get("qty", 0),
                    }
                    for t in builder.no_resource_tasks
                ],
                "objective_value": None,
                "solve_time_seconds": 0.0,
            }

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = int(self.config.get("max_search_time", 60))
        # Stop early when the gap to the proven lower-bound is ≤ 1 %
        solver.parameters.relative_gap_limit = 0.01
        # num_search_workers=1 + fixed random_seed → byte-identical output for replay tests
        # Use num_search_workers=8 (default) in production for speed
        solver.parameters.num_search_workers = int(self.config.get("num_search_workers", 8))
        solver.parameters.random_seed = int(self.config.get("random_seed", 42))

        # ── PRE-SOLVE DIAGNOSTICS ──────────────────────────────────────────────
        _proto = builder.model.Proto()
        logger.info(
            f"📐 Model size: {len(_proto.variables)} vars, "
            f"{len(_proto.constraints)} constraints, "
            f"horizon={builder.horizon}min"
        )

        # Per-operation machine load — flag trước khi solver chạy
        # Load > 1.0 = guaranteed infeasible (không đủ máy); > 0.85 = cảnh báo
        _ops_duration: Dict[str, int] = {}
        for t in self.tasks:
            op = t.get("operation", "").lower()
            _ops_duration[op] = _ops_duration.get(op, 0) + int(t.get("duration", 0))

        _max_machines = int(self.config.get("max_factory_machines", 100))
        _op_machine_count: Dict[str, int] = {"knitting": _max_machines}
        for r in self.resources:
            op = (r.get("operation") or "").lower()
            if op and op != "knitting":
                _op_machine_count[op] = _op_machine_count.get(op, 0) + 1

        for op, total_dur in sorted(_ops_duration.items()):
            n_machines = _op_machine_count.get(op, 1)
            avail = n_machines * builder.horizon
            load = total_dur / avail if avail > 0 else float("inf")
            icon = "🔴" if load > 1.0 else ("🟡" if load > 0.85 else "🟢")
            logger.info(
                f"   {icon} [{op}] load={load:.1%}  "
                f"({total_dur}min demand / {n_machines} machines × {builder.horizon}min horizon)"
            )
            if load > 1.0:
                logger.error(
                    f"   ❌ GUARANTEED INFEASIBLE: [{op}] cần {total_dur}min "
                    f"nhưng chỉ có {avail}min ({n_machines} machines). "
                    "Tăng horizon, thêm máy, hoặc giảm số đơn."
                )

        validation_err = builder.model.Validate()
        if validation_err:
            logger.error(
                f"❌ MODEL_INVALID (Validate):\n{validation_err}\n"
                "Nguyên nhân thường gặp: duration âm, bound min > max, "
                "objective overflow (horizon quá lớn), mảng rỗng trong AddMaxEquality."
            )
            return {
                "status": "model_invalid",
                "assignments": [],
                "overloads": [],
                "objective_value": None,
                "solve_time_seconds": 0.0,
            }

        status = solver.Solve(builder.model)

        # Overload-ratio diagnostic (knitting-specific, post-solve)
        _total_knitting = _ops_duration.get("knitting", 0)
        _capacity = _max_machines * builder.horizon
        if _capacity > 0:
            _load = _total_knitting / _capacity
            if _load > 0.85:
                logger.warning(
                    f"🏭 Factory load {_load:.1%} exceeds 85 % "
                    f"({_total_knitting} knitting-min / {_capacity} available-min). "
                    "Consider extending shifts or adding machines."
                )

        return builder.extract_results(solver, status)
