import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from .pipeline import Pipeline

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

        # Optional re-schedule stability hint (None → fresh solve, behavior unchanged).
        rh = payload.get("reschedule_hint")
        self.reschedule_hint: Optional[Dict[str, Any]] = rh if rh else None
        if self.reschedule_hint:
            logger.info(
                f"🎯 reschedule_hint received: "
                f"{len(self.reschedule_hint.get('previous_assignments') or [])} previous assignments"
            )

        # Fallback for empty task operations (defensive programming against Go backend bugs)
        resource_ops = {r.get("id"): r.get("operation") for r in self.resources if r.get("id")}
        for t in self.tasks:
            if not t.get("operation"):
                m_id = t.get("pinned_machine_id") or (t.get("compatible_resource_ids", [None])[0] if t.get("compatible_resource_ids") else None)
                if m_id and m_id in resource_ops and resource_ops[m_id]:
                    t["operation"] = resource_ops[m_id]
                    logger.warning(f"⚠️ Task {t.get('task_id')} missing operation, inferred as '{t['operation']}' from machine {m_id}")
                else:
                    t["operation"] = "knitting"
                    logger.warning(f"⚠️ Task {t.get('task_id')} missing operation and machine inference failed, defaulting to 'knitting'")

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

        # Top-level dyelot stock — carried through to the dyelot post-pass at the
        # end of Pipeline.run().  Not consumed by any CP-SAT model (additive data
        # only).  Empty list when the Go backend does not send the field.
        self.dyelot_stock = payload.get("dyelot_stock") or []
        # Default roll size per thread vi (incl. zero-stock vis) — lets the dyelot
        # post-pass size a fresh lot for a vi with no current stock.
        self.vi_packing_size: Dict[str, float] = payload.get("vi_packing_size") or {}

    def solve(self) -> Dict[str, Any]:
        if not self.tasks:
            return {"status": "feasible", "assignments": [], "overloads": []}

        logger.info(
            f"🚀 Pipeline solve: {len(self.tasks)} tasks, "
            f"{len(self.resources)} resources"
        )
        pipeline = Pipeline(
            self.config,
            self.resources,
            self.tasks,
            self.material_capacities,
            reschedule_hint=self.reschedule_hint,
            dyelot_stock=self.dyelot_stock,
        )
        result = pipeline.run()

        # Dyelot allocation post-pass (orchestration; no scheduling CP-SAT touched).
        # Runs after knitting is scheduled: reads the knitting machine-sequence from
        # the assignments + per-task main_yarn_consumption + dyelot_stock, and emits
        # one flush-optimized dyelot per (order, VI).  Skipped (returns empty) when
        # the payload carries no dyelot_stock.
        from .dyelot_allocator import allocate_dyelots
        dyelot_out = allocate_dyelots(
            self.tasks, result.get("assignments", []), self.dyelot_stock, self.config,
            vi_packing=self.vi_packing_size,
        )
        result.update(dyelot_out)
        return result
