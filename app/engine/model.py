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

# Attach the per-run FileHandler to the `app.engine` PACKAGE logger, not just this
# module. Phase diagnostics (the "Phase 1 … INFEASIBLE / bottleneck / workforce"
# lines that explain WHY a solve fails) are emitted by sibling loggers such as
# app.engine.phases.phase1_knitting and app.engine.shared. Those propagate up to
# app.engine — not to app.engine.model — so anchoring the handler here lets the
# scheduling_*.log capture the full engine trace instead of only model.py's lines.
#
# Only a FileHandler is attached, and propagate is left True on purpose: console
# output keeps flowing through whatever root handler the runtime installs
# (celery/uvicorn), so there is no double-printing, and pytest's `caplog` — which
# captures via propagation to the root logger — keeps seeing engine records.
_engine_logger = logging.getLogger("app.engine")
_engine_logger.setLevel(logging.INFO)
# Drop any FileHandler left by a previous import of this module (defensive against
# module reloads in tests); leave non-file handlers (e.g. pytest's) untouched.
for _h in list(_engine_logger.handlers):
    if isinstance(_h, logging.FileHandler):
        _engine_logger.removeHandler(_h)

_file_handler = logging.FileHandler(_log_file)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
_engine_logger.addHandler(_file_handler)

logger = logging.getLogger(__name__)
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
        # In-production (converted/pinned) orders already committed to a dye lot:
        # [{order, vi, dyelot, machine_id, start_time, net_kg, slots, committed_kg}].
        # Lets the dyelot post-pass pin them to their committed lot and co-lot a
        # sharing new order. Empty when the Go backend sends no pinned work.
        self.in_production = payload.get("in_production") or []

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
            in_production=self.in_production,
        )
        result.update(dyelot_out)
        return result
