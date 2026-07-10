"""
4-Phase Sequential Pipeline Orchestrator.

Replaces the monolithic TaskModelBuilder for large payloads by decomposing
the scheduling problem into four independent CP-SAT models:

  Phase 1 — Knitting:    machine allocation + workforce capacity
  Phase 2 — Linking:     waits for knitting via start_lb (no shared CP-SAT vars)
  Phase 3 — Washing:     group-isolated batching (one model per color+substance)
  Phase 4 — Ironing:     waits for washing via start_lb
  Phase 5 — Packing:     waits for ironing; also any other downstream op

Each phase receives only the tasks and resources relevant to it. Dependencies
between phases are passed as plain integers (start/end times), not shared
CP-SAT variables — this is what eliminates combinatorial explosion.

Entry point: Pipeline(config, resources, tasks, material_capacities).run()
Returns the same dict format as the legacy Engine.solve():
  {"status", "assignments", "overloads", "objective_value", "solve_time_seconds"}
"""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .phases.phase1_knitting import (
    PHASE1_OPS,
    Phase1Result,
    balance_cold_knitting,
    left_shift_cold_knitting,
    parallelize_component_pos,
    reorder_contiguous_knitting,
    repair_yarn_config_reentry,
    solve_knitting,
    spread_cold_knitting,
)
from .shared import compute_global_horizon, diagnose_infeasibility
from .phases.phase2_linking import (
    PHASE2_OPS,
    Phase2Result,
    _compute_start_lb as _compute_linking_start_lb,
    balance_linking_load,
    compute_sameqty_start_lb,
    left_shift_cold_linking,
    solve_linking,
)
from .phases.phase3_batching import (
    PHASE3_OPS,
    Phase3Result,
    flush_unwashed_end_of_shift,
    left_shift_cold_washing,
    solve_washing,
)
from .phases.phase4_downstream import (
    UPSTREAM_OPS,
    Phase4Result,
    balance_downstream_load,
    fifo_swap_ironing,
    fifo_swap_packing,
    left_shift_cold_ironing,
    left_shift_cold_packing,
    solve_downstream,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the 4-phase scheduling pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        resources: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        material_capacities: Optional[Dict[str, int]] = None,
        reschedule_hint: Optional[Dict[str, Any]] = None,
        dyelot_stock: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.resources = resources
        # Top-level dyelot stock for the (future) dyelot post-pass.  Carried as-is;
        # no CP-SAT model reads it.  Empty list when absent.
        self.dyelot_stock = dyelot_stock or []
        # Determinism leg 3 (pairs with num_search_workers=1 + PYTHONHASHSEED=0):
        # normalise task order by the STABLE key `task_id` at ingest.  build_resource_model
        # creates CP-SAT start/interval/machine vars + NoOverlap/Cumulative in task-list
        # order, and the fixed-seed search keys off that creation order — so without this,
        # the same logical request produces a different schedule whenever Go serialises
        # `tasks` in a different order (measured: shuffling tasks moved 2–8/90 tasks).
        # task_id — NOT due/priority — because task_id is invariant across re-schedules:
        # sorting by due would reshuffle the build order whenever a due changes, churning
        # unrelated tasks.  task_id gives both determinism AND re-schedule stability.
        self.tasks = sorted(_sanitize_dummy_tasks(tasks), key=lambda t: t["task_id"])
        self.material_capacities = material_capacities or {}
        self.translation_map: Dict[str, str] = _build_translation_map(self.tasks)
        self.reschedule_hint = reschedule_hint
        # `stabilize_pass` marks the /solve double-solve's INTERNAL pass-2 (a re-schedule
        # of cold pass-1 whose result the UI actually receives) — as opposed to a GENUINE
        # external /re-schedule where Go sends a committed plan and machine stability is
        # paramount.  The cold compaction passes (linking/ironing left-shift, knitting
        # balance) reassign machines / pull tasks earlier for tightness; that is wanted on
        # cold + the internal stabilize pass, but would churn a genuine re-schedule, so
        # gate them on `_apply_cold_passes`.
        self._stabilize_pass = bool(reschedule_hint and reschedule_hint.get("stabilize_pass"))
        self._apply_cold_passes = (not reschedule_hint) or self._stabilize_pass
        # Detect BEFORE partitioning: partition_hint_for_pipeline drops previous
        # assignments whose task no longer exists, so the shrink signal must be
        # read from the raw hint vs the full current task list.
        self.workload_shrank: bool = _detect_workload_shrink(reschedule_hint, self.tasks)
        self.partitioned_hint: Dict[str, Any] = (
            partition_hint_for_pipeline(reschedule_hint, self.tasks)
            if reschedule_hint else {}
        )

    def run(self) -> Dict[str, Any]:
        if not self.tasks:
            return {"status": "feasible", "assignments": [], "overloads": [],
                    "objective_value": None, "solve_time_seconds": 0.0}

        # Each phase receives ALL resources — tasks' compatible_resource_ids is the
        # sole gating mechanism, not the resource's operation tag.  This mirrors the
        # old monolithic model where every resource was in one shared resource_map.
        all_resources = self.resources

        # Compute the global horizon from per-operation makespans so the
        # horizon is tight (avoids large lateness variable domains) while
        # remaining large enough for all tasks to complete.  A single consistent
        # horizon across all phases preserves CP-SAT determinism (fixed seed + 1 worker).
        global_horizon = compute_global_horizon(self.tasks, all_resources, self.config)
        logger.info(f"🌐 Global horizon: {global_horizon} min")
        if self.reschedule_hint:
            logger.info(
                f"🔁 Re-schedule: workload_shrank={self.workload_shrank} "
                f"(orders removed since previous plan → relax to compaction)"
                if self.workload_shrank else
                f"🔁 Re-schedule: workload_shrank=False (same/grown set → keep layout)"
            )

        # ── Phase 1: Knitting ─────────────────────────────────────────────
        p1_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE1_OPS]
        p1_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("knitting"))
        p1: Phase1Result = solve_knitting(
            p1_tasks, all_resources, self.config,
            material_capacities=self.material_capacities,
            horizon=global_horizon,
            reschedule_hint=p1_hint,
            all_pipeline_tasks=self.tasks,
            workload_shrank=self.workload_shrank,
            translation_map=self.translation_map,
        )
        logger.info(f"✅ Phase 1 complete: {len(p1.assignments)} assignments, status={p1.status}")

        if p1.status not in ("feasible", "empty"):
            return _phase_failure_result(p1.status, p1_tasks, all_resources, self.config, global_horizon)

        # Improve knitting BEFORE linking solves on it: cross-machine spread (parallel
        # serial tails) + same-machine idle compaction.  Running it here (not as a final
        # post-pass) means linking — and every downstream phase — schedules on the
        # pulled-earlier knitting, so the whole pipeline flows tight (no re-solve).
        self._improve_knitting(p1)

        # ── Phase 2: Linking ──────────────────────────────────────────────
        p2_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE2_OPS]
        p2_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("linking"))
        p2: Phase2Result = solve_linking(
            p2_tasks, all_resources, self.config,
            p1_start_times=p1.start_times,
            p1_end_times=p1.end_times,
            translation_map=self.translation_map,
            horizon=global_horizon,
            reschedule_hint=p2_hint,
            workload_shrank=self.workload_shrank,
            all_pipeline_tasks=self.tasks,
        )
        logger.info(f"✅ Phase 2 complete: {len(p2.assignments)} assignments, status={p2.status}")

        if p2.status not in ("feasible", "empty"):
            return _phase_failure_result(p2.status, p2_tasks, all_resources, self.config, global_horizon)

        # Tighten linking BEFORE washing/iron/packing solve: pull each linking task to
        # its earliest free worker at its (improved-)knitting-derived release, so the
        # downstream phases schedule on the tight linking and FOLLOW it — no re-solve.
        self._tighten_linking(p1, p2, all_resources)

        # ── Phases 3→5: Washing → Ironing → Packing ───────────────────────
        chain = self._solve_phases_3_to_5(
            all_resources, global_horizon, {**p1.end_times, **p2.end_times}
        )
        if isinstance(chain, dict):
            return chain  # phase-failure result
        p3, p4, p5 = chain

        # ── Knitting relayout refinement (cold solve only) ────────────────
        # ONE verified pass combining three knitting-layout transforms that share a
        # SINGLE phases-2–5 re-solve (instead of one re-solve each):
        #   1. parallel component-PO — the solver may knit a garment's component POs
        #      (front 0-641 / back 0-642) SERIALLY, so the first complete panel isn't
        #      ready until the 2nd PO starts → linking idles through the whole first PO.
        #      Dedicate disjoint machines per PO so they knit in PARALLEL (first panel
        #      far sooner; feeds linking early).
        #   2. order-contiguity — the solver stalls at FEASIBLE so its secondary
        #      contiguity term never optimises; re-sequence each machine so an order's
        #      tasks run contiguously ("dứt điểm đơn đó").
        #   3. yarn-config re-entry repair — a slack-due filler task dropped after a
        #      different-config block forces the machine back into a yarn config it
        #      already left (:2→:5→:2 = double creel change); move it to a compatible
        #      machine whose tail already holds that config.
        # None of these is monotone (each can push a yielding order later), so the merged
        # candidate is re-solved (cheaper verify budget) and accepted ONLY if total
        # pipeline lateness does not increase.
        if not self.reschedule_hint and (
            self.config.get("enable_knitting_parallel_pos", True)
            or self.config.get("enable_knitting_contiguity_reorder", False)
            or self.config.get("enable_knitting_yarn_config_repair", True)
        ):
            refined = self._try_knitting_relayout(
                p2_tasks, all_resources, global_horizon, p1, p2, p3, p4, p5,
            )
            if refined is not None:
                p1, p2, p3, p4, p5 = refined
                # Relayout relocates knitting across machines (parallel component-POs
                # / contiguity) but does NOT re-compact: moving a task to another
                # machine leaves an idle slot behind it on its old machine, so the
                # task that followed now shows a gap that should be tight (observed:
                # 662_3 moved off a machine → 657_1 stranded at 1080 with an empty
                # 928–1080 slot).  Re-run the cold left-shift so each machine's tasks
                # pull into the freed slots.  Safe: left-shift only moves knitting
                # EARLIER and never across machines → parallelisation is preserved,
                # downstream is untouched, and end-to-end lateness is non-increasing.
                if left_shift_cold_knitting(p1.assignments, self.tasks, self.config):
                    # Keep p1's time maps in sync (mirrors _improve_knitting) so the
                    # same-qty relink below reads the compacted knitting release.
                    for a in p1.assignments:
                        p1.start_times[a["task_id"]] = a["start_time"]
                        p1.end_times[a["task_id"]] = a["end_time"]

        # ── Knitting makespan balance (all paths) ─────────────────────────
        # spread/left-shift run BEFORE the relayout, which re-concentrates a PO's tail
        # onto one machine → 1–2 machines finish far later than the rest while compatible
        # machines sit idle.  Peel those critical tails onto the earliest-free compatible
        # machine (workforce-validated per move, only moves EARLIER → makespan monotone
        # ↓, downstream release relaxes).  Runs on RE-SCHEDULE too so the double-solve
        # pass-2 (returned to the UI) is balanced: pass-2 hard-keeps knitting to pass-1's
        # balanced layout and re-solves washing→packing on it, so downstream follows.
        if self._apply_cold_passes and self.config.get("enable_knitting_load_balance", True):
            if balance_cold_knitting(p1.assignments, self.tasks, self.config):
                for a in p1.assignments:
                    p1.start_times[a["task_id"]] = a["start_time"]
                    p1.end_times[a["task_id"]] = a["end_time"]

        # ── Same-qty re-link refinement (two-pass, cold solve only) ───────
        # Panel knitting cùng (component, qty) là thay-thế-được; floor same-qty
        # nới các slice GIỮA của mỗi đơn (per-order completion bất biến) → máy
        # linking rảnh sớm → hạ nguồn hưởng.  Pass 2 chạy với Pareto end-caps
        # (end ≤ pass-1 per task linking) + verify điểm-theo-điểm TOÀN pipeline:
        # sai một task là giữ nguyên pass 1 — không-regression theo cấu trúc.
        if (
            not self.reschedule_hint
            and p2.status == "feasible"
            and self.config.get("enable_sameqty_relink", True)
        ):
            refined = self._try_sameqty_relink(
                p2_tasks, all_resources, global_horizon, p1, p2, p3, p4, p5,
            )
            if refined is not None:
                p2, p3, p4, p5 = refined

        # ── Aggregate results ─────────────────────────────────────────────
        all_assignments = (
            p1.assignments + p2.assignments + p3.assignments
            + p4.assignments + p5.assignments
        )

        # Knitting spread/left-shift and linking left-shift now run FORWARD — knitting
        # before linking (_improve_knitting), linking before phases 3–5
        # (_tighten_linking) — so each downstream phase solves on the pulled-earlier
        # upstream and follows it, instead of being fixed first and left with a gap.
        # Only worker-relabel + washing compaction remain as final touch-ups here.
        if not self.reschedule_hint:
            # Linking worker load-balance: machine-relabel only (timing unchanged),
            # so downstream stays byte-identical and no order can finish later.
            # Fixes severe linking-worker idle/imbalance (measured stdev 965→40).
            # Skipped on re-schedule — machine assignment is part of stability there.
            if self.config.get("enable_linking_balance", True):
                balance_linking_load(all_assignments, self.tasks, all_resources, self.config)
        # Washing left-shift, FINAL pass — washing itself stalls at FEASIBLE and can
        # leave idle gaps even though linking was already tight when phase 3 solved.
        # This compacts washing on the final ends; washing only moves EARLIER →
        # iron/packing release bounds relax → their assignments stay valid (slack).
        # Runs on cold + real re-schedule (real re-schedules over-consolidate washing
        # into late batches — see the flush gate in _solve_phases_3_to_5); only the
        # internal stabilize pass is skipped so it can't shatter pass-1 consolidation.
        if not self._stabilize_pass and self.config.get("enable_washing_left_shift", True):
            wash_moved = left_shift_cold_washing(
                all_assignments, self.tasks, self.config,
                [int(s) for s in self.config.get("shift_ends_min", [])],
            )
            # Re-glue iron → packing to the washing this FINAL pass just pulled earlier.
            # The in-phase iron/packing left-shifts ran BEFORE this washing compaction,
            # so a wash batch moved earlier here leaves its iron stranded a full cycle
            # late (measured: wash re-solve drifted a batch 1020→1080, this pass pulled
            # it back, iron stayed at 1140 — tester saw a 60-min wash→iron hole).  Both
            # left-shifts are monotone + idempotent (already-tight tasks don't move),
            # so re-running them here only closes holes this pass opened.
            if wash_moved:
                cur_ends = {x["task_id"]: int(x["end_time"]) for x in all_assignments}
                if self.config.get("enable_ironing_left_shift", True):
                    left_shift_cold_ironing(
                        all_assignments, self.tasks, self.config, cur_ends,
                    )
                    cur_ends = {x["task_id"]: int(x["end_time"]) for x in all_assignments}
                if self.config.get("enable_packing_left_shift", True):
                    left_shift_cold_packing(
                        all_assignments, self.tasks, self.config, cur_ends,
                    )
        # Iron/packing worker load-balance: machine-relabel only (timing unchanged →
        # downstream byte-identical, zero regression by construction, like the linking
        # balance above).  The left-shifts glue slices to their ready times but
        # tie-break onto the lowest machine id, piling ~90% of tasks on 2 of 5 workers
        # (measured iron 49/38/6/3/1) while nobody actually waits.  Runs on cold +
        # stabilize (UI gets the balanced layout); skipped on real Go re-schedule —
        # worker assignment is part of stability there.
        if self._apply_cold_passes:
            changed = 0
            if self.config.get("enable_ironing_balance", True):
                changed += balance_downstream_load(
                    all_assignments, self.tasks, all_resources, self.config,
                    frozenset(_PHASE4_OP_SET), "Ironing",
                )
            if self.config.get("enable_packing_balance", True):
                changed += balance_downstream_load(
                    all_assignments, self.tasks, all_resources, self.config,
                    frozenset({"pack", "packing"}), "Packing",
                )
            # Hole-closing left-shift AFTER the balance: the relabel keeps every
            # task's time but REDISTRIBUTES the busy intervals, so a machine the
            # earlier left-shift saw as busy can end up free in the final layout —
            # a slice then visibly waits while a machine idles (measured: 1 of 4
            # ready-together slices started +21 min while W_IRONING_03 sat empty).
            # One more monotone left-shift on the final labels closes exactly those
            # holes (zero moves when already tight); no re-balance afterwards, so
            # no new holes can open.
            if changed:
                cur_ends = {x["task_id"]: int(x["end_time"]) for x in all_assignments}
                if self.config.get("enable_ironing_left_shift", True):
                    left_shift_cold_ironing(
                        all_assignments, self.tasks, self.config, cur_ends,
                    )
                    cur_ends = {x["task_id"]: int(x["end_time"]) for x in all_assignments}
                if self.config.get("enable_packing_left_shift", True):
                    left_shift_cold_packing(
                        all_assignments, self.tasks, self.config, cur_ends,
                    )
        all_overloads = (
            p1.overloads + p2.overloads + p3.overloads
            + p4.overloads + p5.overloads
        )
        total_time = (
            p1.solve_time_seconds + p2.solve_time_seconds + p3.solve_time_seconds
            + p4.solve_time_seconds + p5.solve_time_seconds
        )

        # Objective: sum of per-phase objective values where available
        obj_vals = [
            v for v in [p1.objective_value, p2.objective_value,
                        p4.objective_value, p5.objective_value]
            if v is not None
        ]
        combined_obj = sum(obj_vals) if obj_vals else None

        return {
            "status": "feasible",
            "assignments": all_assignments,
            "overloads": all_overloads,
            "objective_value": combined_obj,
            "solve_time_seconds": total_time,
        }

    def _improve_knitting(self, p1r: Phase1Result) -> None:
        """Cold knitting cross-machine spread + same-machine left-shift, applied to a
        Phase-1 result IN PLACE before linking solves on it.  Both transforms are
        deterministic and monotone (every task only moves earlier), so linking and all
        downstream phases simply schedule on the pulled-earlier knitting.  Skipped on
        re-schedule (knitting is hard-kept there)."""
        if self.reschedule_hint:
            return
        if self.config.get("enable_knitting_spread", True):
            spread_cold_knitting(p1r.assignments, self.tasks, self.config)
        left_shift_cold_knitting(p1r.assignments, self.tasks, self.config)
        for a in p1r.assignments:
            p1r.start_times[a["task_id"]] = a["start_time"]
            p1r.end_times[a["task_id"]] = a["end_time"]

    def _tighten_linking(
        self,
        p1r: Phase1Result,
        p2r: Phase2Result,
        all_resources: List[Dict[str, Any]],
    ) -> None:
        """Linking left-shift applied to a Phase-2 result IN PLACE before the downstream
        phases solve on it, so washing/iron/packing schedule on the tight linking and
        follow it.  Monotone (linking only moves earlier — to its earliest free worker at
        the knitting-derived release).  Reads the knitting timing from `p1r`.

        Runs on RE-SCHEDULE too (not cold-only): the /solve double-solve returns pass-2 (a
        re-schedule of cold pass-1) to the UI, where the linking solve stalls at FEASIBLE
        and staggers each slice minutes past its panel-ready time even though a linking
        worker is free (user: linking không bắt đầu ngay khi panel dệt xong).  The
        left-shift is monotone + deterministic + idempotent, so running it every pass glues
        linking to the knitting panel ends without harming re-schedule stability."""
        if not self._apply_cold_passes or not self.config.get("enable_linking_left_shift", True):
            return
        combined = p1r.assignments + p2r.assignments
        moved = left_shift_cold_linking(combined, self.tasks, all_resources, self.config)
        if moved:
            for a in p2r.assignments:
                p2r.start_times[a["task_id"]] = a["start_time"]
                p2r.end_times[a["task_id"]] = a["end_time"]

    def _solve_phases_3_to_5(
        self,
        all_resources: List[Dict[str, Any]],
        global_horizon: int,
        combined_end_times: Dict[str, int],
    ):
        """Washing → Ironing → Packing trên end-times đã cho (P1+P2 merged).

        Returns (p3, p4, p5) khi thành công, hoặc dict failure-result (giữ đúng
        hành vi cũ của run()).  Tách riêng để two-pass refinement gọi lại được
        với linking end-times của pass 2.
        """
        # ── Phase 3: Washing ──────────────────────────────────────────────
        p3_tasks = [t for t in self.tasks if t.get("operation", "").lower() in PHASE3_OPS]
        shift_ends: List[int] = [int(s) for s in self.config.get("shift_ends_min", [])]

        # Double-solve stabilize pass: REUSE the pass-1 washing verbatim instead of
        # re-solving it.  Re-solving washing on the stabilize pass stalls at FEASIBLE
        # and over-consolidates early-ready goods into a late batch, leaving the wash
        # machine idle for ~a day (pass-1 cold already left-shifted washing correctly:
        # max idle gap 184 vs 1965 min).  We can't just re-run the left-shift on pass 2
        # — in the full-batch-consolidation case that SHATTERS correct batches (see
        # [[project_washing_left_shift_reschedule]]).  Reusing pass-1's assignments is
        # right for BOTH regimes: it never re-decides consolidation, keeps every field
        # (group_id/batch_slot_id), and skips the slow (>7 min) stabilize washing solve.
        # Iron/packing below then solve on these (earlier, correct) washing ends.
        p1_wash = (self.reschedule_hint or {}).get("_pass1_washing_full") \
            if self._stabilize_pass else None
        _mode = "stabilize" if self._stabilize_pass else "reuse-hint"
        # Re-schedule NGOÀI (Go hint) với workload giặt KHÔNG ĐỔI: reuse nguyên văn
        # vị trí hint thay vì re-solve.  Re-solve group lớn kẹt FEASIBLE và HOÁN VỊ
        # các slice thay-thế-được giữa các cycle y nguyên (giữ-nguyên-100% là nghiệm
        # phạt-0 nhưng solver không tới được) → mỗi re-schedule một hoán vị khác,
        # chuỗi run không đạt bất động điểm (đo CP_1783648035252240672→..315..:
        # B đổi 49 task vs hint, A đổi 98, +1 LATE — iron/packing dời theo 1:1).
        # Reuse giữ washing từng byte; guard/repair dep + merge-only + flush +
        # left-shift phía dưới VẪN chạy (idempotent: hint đã khít → zero move;
        # hint lỏng/over-consolidated → vẫn được kéo sớm như trước — không mất
        # khả năng phục hồi của [[project_washing_reschedule_leftshift_fix]]).
        # Chỉ áp khi MỌI washing task hiện tại có prev khớp task_id (task giặt
        # mới → re-solve như cũ) và workload không co (shrink cần compaction).
        if (
            p1_wash is None
            and self.reschedule_hint and not self._stabilize_pass
            and not self.workload_shrank
            and self.config.get("enable_washing_reschedule_reuse", True)
        ):
            p1_wash = _washing_reuse_from_hint(self.reschedule_hint, p3_tasks)
            if p1_wash is not None:
                logger.info(
                    f"♻️ Phase 3 (reuse-hint): washing workload unchanged — reusing "
                    f"{len(p1_wash)} hint positions verbatim (no re-solve)."
                )
        if p1_wash:
            # Dependency guard: pass-1 washing was feasible against PASS-1 linking ends.
            # Pass-2 linking is hard-kept + left-shifted so it should only be equal or
            # EARLIER — but a dropped keep could land a linking task later, making a
            # reused wash start precede its dependency.  In that (rare) case fall back
            # to the normal washing solve instead of emitting an invalid schedule.
            _info = {t["task_id"]: t for t in p3_tasks}
            _violated = [
                a["task_id"] for a in p1_wash
                if any(
                    int(a["start_time"]) < int(combined_end_times.get(d, 0))
                    for d in (_info.get(a["task_id"], {}).get("final_depends_on") or [])
                    if d in combined_end_times
                )
            ]
            if _violated:
                # LOCAL REPAIR first: delay ONLY the violating cycles to their pass-2
                # dependency ends (earliest boundary-safe free slot on their machine),
                # keeping every other reused cycle untouched.  Discarding the whole
                # pass-1 layout over one drifted linking task re-opens the stabilize
                # re-solve disaster (measured: 1/87 violated → full re-solve stalled
                # FEASIBLE → 29 slices held >1 day, machine idle 2737 min).  Only if a
                # violating cycle genuinely cannot be re-placed do we fall back.
                repaired = _repair_reused_washing(
                    p1_wash, _info, combined_end_times, shift_ends,
                )
                if repaired is not None:
                    logger.info(
                        f"🩹 Phase 3 ({_mode}): {len(_violated)} reused washing "
                        f"task(s) started before their upstream dependency "
                        f"(e.g. {_violated[0]}) — delayed just their cycle(s); the "
                        f"rest of the reused layout is kept."
                    )
                    p1_wash = repaired
                else:
                    logger.warning(
                        f"⚠️ Phase 3 ({_mode}): {len(_violated)} reused washing task(s) "
                        f"would start before their upstream dependency "
                        f"(e.g. {_violated[0]}) and their cycles cannot be re-placed — "
                        f"falling back to a normal washing solve."
                    )
                    p1_wash = None
        if p1_wash:
            p3 = Phase3Result(
                status="feasible",
                assignments=[dict(a) for a in p1_wash],
                overloads=[
                    dict(o) for o in
                    (self.reschedule_hint or {}).get("_pass1_washing_overloads") or []
                ],
                end_times={a["task_id"]: int(a["end_time"]) for a in p1_wash},
            )
            logger.info(
                f"✅ Phase 3 ({_mode}): reused {len(p3.assignments)} washing "
                f"assignments (no re-solve — keeps the previous compacted layout)"
            )
            # Pass-2 linking is left-shifted TIGHTER than pass-1's, so the reused
            # washing gains fold opportunities pass-1 could not see (a slice whose
            # linking now ends before an under-filled cycle: "lần giặt sau còn chỗ mà
            # không nhét thêm vào").  Run the MERGE-ONLY washing left-shift: folds
            # into existing cycles with spare capacity — no new cycles, so it cannot
            # shatter consolidation (the reason the full left-shift stays off here);
            # starts only move EARLIER, and iron/packing solve after this on the
            # refreshed ends.
            if self.config.get("enable_washing_left_shift", True):
                merged = left_shift_cold_washing(
                    p3.assignments, self.tasks, self.config, shift_ends,
                    dep_ends=combined_end_times, merge_only=True,
                )
                if merged:
                    for a in p3.assignments:
                        p3.end_times[a["task_id"]] = int(a["end_time"])
        else:
            # Phase 3 is group-isolated; pass the full hint (with washing-group partition
            # already computed) so each group filter its own previous_assignments.
            p3_hint = _hint_for_phase_washing(self.reschedule_hint, self.partitioned_hint.get("washing"))
            p3 = solve_washing(
                p3_tasks, all_resources, self.config,
                p2_end_times=combined_end_times,
                shift_ends=shift_ends,
                horizon=global_horizon,
                reschedule_hint=p3_hint,
                workload_shrank=self.workload_shrank,
                all_pipeline_tasks=self.tasks,
            )
            logger.info(
                f"✅ Phase 3 complete: {len(p3.assignments)} assignments, "
                f"{len(p3.batches)} batches, status={p3.status}"
            )

        if p3.status not in ("feasible", "empty"):
            return _phase_failure_result(p3.status, p3_tasks, all_resources, self.config, global_horizon)

        # ── End-of-shift washing flush, BEFORE downstream ────────────────
        # Pull washing that became ready before a shift boundary but spilled into a
        # later shift into a pre-break batch, THEN solve ironing/packing once on the
        # earlier washing ends so they follow.  No re-solve/gate needed: flush only
        # moves washing EARLIER, so every downstream release bound relaxes and the
        # phase 4–5 optimum cannot get worse.
        #
        # Runs on cold AND real Go re-schedule (gate `not _stabilize_pass`): a real
        # re-schedule re-solves the large per-color washing groups from scratch and,
        # stalling at FEASIBLE, over-consolidates early-ready goods into much later
        # batches — drifting them days past the stable plan Go sent (measured: 35/97
        # washing tasks pushed >500 min later, worst +5550 → held ~4 days).  Because
        # washing only moves EARLIER, running the flush/left-shift there recovers the
        # early placement without hurting downstream.  ONLY the internal double-solve
        # stabilize pass is skipped: there pass-1 (cold) is already well-consolidated
        # AND early, and re-running the left-shift shatters that consolidation.
        if not self._stabilize_pass and self.config.get("enable_washing_flush", True):
            moved = flush_unwashed_end_of_shift(
                p3.assignments, self.tasks, self.config, shift_ends,
                dep_ends=combined_end_times,
            )
            if moved:
                for a in p3.assignments:
                    p3.end_times[a["task_id"]] = a["end_time"]

        # Washing left-shift: catch ready goods the solver bundled into a later batch
        # (machine idle in between) that the flush couldn't pull before a break.  Pulls
        # them to the earliest boundary-safe free wash slot.  Monotone (only earlier) →
        # downstream solves below on the earlier washing ends.  Same gating as the flush
        # above: cold + real re-schedule, skip only the internal stabilize pass.
        if not self._stabilize_pass and self.config.get("enable_washing_left_shift", True):
            moved = left_shift_cold_washing(
                p3.assignments, self.tasks, self.config, shift_ends,
                dep_ends=combined_end_times,
            )
            if moved:
                for a in p3.assignments:
                    p3.end_times[a["task_id"]] = a["end_time"]

        all_end_times = {**combined_end_times, **p3.end_times}
        p4, p5 = self._solve_phases_4_5(all_resources, global_horizon, all_end_times)
        return p3, p4, p5

    def _solve_phases_4_5(
        self,
        all_resources: List[Dict[str, Any]],
        global_horizon: int,
        end_through_washing: Dict[str, int],
    ) -> Tuple["Phase4Result", "Phase4Result"]:
        """Ironing (P4) → Packing (P5) given end-times through washing (P1+P2+P3).

        Split out so the washing-flush refinement can re-run downstream on the
        flushed (earlier) washing ends.  Downstream is split into ironing → packing
        so each CP-SAT model stays small (one combined model was heavy / could time
        out).  Cross-phase handoff is integers (end_times), no shared CP-SAT vars.
        """
        # ── Phase 4: Ironing ──────────────────────────────────────────────
        p4_tasks = [t for t in self.tasks if t.get("operation", "").lower() in _PHASE4_OP_SET]
        p4_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("ironing"))
        p4: Phase4Result = solve_downstream(
            p4_tasks, all_resources, self.config,
            p3_end_times=end_through_washing,
            horizon=global_horizon,
            reschedule_hint=p4_hint,
            workload_shrank=self.workload_shrank,
        )
        logger.info(f"✅ Phase 4 (Ironing) complete: {len(p4.assignments)} assignments, status={p4.status}")

        # Tighten ironing BEFORE packing solves: the downstream solver only weakly
        # rewards early starts, so with loose due dates it stalls at FEASIBLE and
        # staggers iron a few minutes past its washing-ready time even though the
        # serial iron machine is free (tester: iron starts 1–5 min after wash done).
        # Pull each iron task to its earliest feasible start; monotone (iron only
        # moves earlier → packing release relaxes, lateness non-increasing).
        #
        # Applied on EVERY pass (cold, internal stabilize, AND real Go re-schedule):
        # wherever washing can move earlier (stabilize: washing reused from pass-1;
        # real re-schedule: washing flush/left-shift now run there too), an iron solve
        # hard-kept to previous positions would strand iron minutes-to-hours after its
        # wash finishes (the 154-min-gap trap).  Since the left-shift is monotone,
        # deterministic and idempotent (pinned iron stays anchored; a schedule that is
        # already tight yields ZERO moves — no needless machine churn), running it every
        # pass keeps iron glued to the actual washing ends without harming stability.
        if self.config.get("enable_ironing_left_shift", True):
            moved = left_shift_cold_ironing(
                p4.assignments, self.tasks, self.config, end_through_washing,
            )
            if moved:
                p4.end_times = {a["task_id"]: a["end_time"] for a in p4.assignments}

        # FIFO-swap: sau left-shift vẫn còn ca một slice ready-TRƯỚC phải chờ vì slice
        # ready-SAU chiếm đúng cửa sổ máy (left-shift không được dời muộn blocker nên
        # bó tay).  Swap có guard "không đơn nào muộn đi"; chạy TRƯỚC packing solve
        # để packing bám theo end mới — vì vậy CHỈ gọi ở đây, không gọi ở các site
        # re-glue/hole-closing muộn hơn (packing đã chốt ở đó).
        moved = fifo_swap_ironing(
            p4.assignments, self.tasks, self.config, end_through_washing,
        )
        if moved:
            p4.end_times = {a["task_id"]: a["end_time"] for a in p4.assignments}

        # ── Phase 5: Packing (+ any other downstream op) ──────────────────
        # Packing waits for ironing via end_times (start_lb).  Anything downstream
        # that is NOT ironing lands here, preserving the old "any other op" catch-all.
        all_end_times_iron = {**end_through_washing, **p4.end_times}
        p5_tasks = [
            t for t in self.tasks
            if t.get("operation", "").lower() not in UPSTREAM_OPS
            and t.get("operation", "").lower() not in _PHASE4_OP_SET
        ]
        p5_hint = _hint_for_phase(self.reschedule_hint, self.partitioned_hint.get("downstream"))
        p5: Phase4Result = solve_downstream(
            p5_tasks, all_resources, self.config,
            p3_end_times=all_end_times_iron,
            horizon=global_horizon,
            reschedule_hint=p5_hint,
            workload_shrank=self.workload_shrank,
        )
        logger.info(f"✅ Phase 5 (Packing) complete: {len(p5.assignments)} assignments, status={p5.status}")

        # Tighten packing: same FEASIBLE-stall as ironing/linking, worst here because packing
        # is the LAST phase (loosest due dates → least early-start incentive), so slices slip
        # minutes past their ironing-ready time while a compatible packing machine is idle.
        # Terminal op ⇒ nothing downstream to disturb; monotone by construction.  Runs EVERY
        # pass incl. real Go re-schedule (same reasoning as the ironing left-shift above:
        # iron can now move earlier on any pass, and a tight schedule yields zero moves).
        if self.config.get("enable_packing_left_shift", True):
            moved = left_shift_cold_packing(
                p5.assignments, self.tasks, self.config, all_end_times_iron,
            )
            if moved:
                p5.end_times = {a["task_id"]: a["end_time"] for a in p5.assignments}

        # FIFO-swap packing: sau left-shift vẫn còn ca một slice ready-TRƯỚC phải chờ
        # vì task ready-SAU chiếm đúng cửa sổ máy (CP_1783586686707912847: 2 đơn trễ
        # 2-3 phút vì các task 3-phút slack >1000' chiếm giữa cửa sổ; left-shift bị
        # luật monotone cấm dời blocker).  Packing là op CUỐI nên guard dùng thẳng
        # DUE của blocker (không cần proxy như iron) và không có hạ nguồn phải bám
        # theo — gọi ở đây là chốt.
        moved = fifo_swap_packing(
            p5.assignments, self.tasks, self.config, all_end_times_iron,
        )
        if moved:
            p5.end_times = {a["task_id"]: a["end_time"] for a in p5.assignments}

        return p4, p5

    def _total_lateness(self, ends: Dict[str, int]) -> Tuple[int, int]:
        """(Σ tardiness, # late orders) at the ORDER level.

        Every task of an order carries the same ship-date due, so an intermediate
        task finishing "late" vs that date is meaningless — what matters is when the
        ORDER completes (its last task).  Tardiness = max(0, max-end-of-order − due).
        Used to gate the contiguity refinement: a candidate is accepted only if
        NEITHER figure increases, i.e. no order ships later and none newly slips.

        PO knit thành phần (BATCH_0-655…) được CUỘN vào garment mà nó nuôi (map qua
        final_depends_on của linking): dệt song song các PO của một garment tất yếu
        kéo dài span TỪNG PO trong khi garment ship sớm hơn — đếm PO trung gian như
        một "đơn" riêng làm gate tự chặn đúng transform nó verify
        (CP_1783308395880305537: BATCH_0-655 "lật trễ" +1 → reject dù W8jjpJWSUc
        −810).  Knit end ≤ linking end theo dependency nên sau khi cuộn nó không bao
        giờ là max — pseudo-order biến mất khỏi cả Σ lẫn count; knit-order không nuôi
        linking nào vẫn đếm như cũ.
        """
        remap = getattr(self, "_knit_oid_remap", None)
        if remap is None:
            remap = {}
            _info = {t["task_id"]: t for t in self.tasks}
            for t in self.tasks:
                if t.get("operation", "").lower() != "linking":
                    continue
                parent = t.get("original_order_id") or t["task_id"]
                for dep in (t.get("final_depends_on") or []):
                    kt = _info.get(dep)
                    if kt and kt.get("operation", "").lower() == "knitting":
                        koid = kt.get("original_order_id") or kt["task_id"]
                        remap.setdefault(koid, parent)
            self._knit_oid_remap = remap
        order_end: Dict[str, int] = {}
        order_due: Dict[str, int] = {}
        for t in self.tasks:
            due = t.get("due_at_min")
            if not due:
                continue
            e = ends.get(t["task_id"])
            if e is None:
                continue
            raw_oid = t.get("original_order_id") or t["task_id"]
            oid = remap.get(raw_oid, raw_oid)
            order_end[oid] = max(order_end.get(oid, 0), e)
            if oid == raw_oid:
                order_due[oid] = int(due)  # uniform per order
            else:
                # knit thành phần cuộn vào garment: góp end, không ghi đè due garment
                order_due.setdefault(oid, int(due))
        total = 0
        n_late = 0
        for oid, e in order_end.items():
            late = e - order_due[oid]
            if late > 0:
                total += late
                n_late += 1
        return total, n_late

    def _rebuild_p1_from_knitting(
        self, p1: Phase1Result, new_start: Dict[str, int], new_end: Dict[str, int],
        new_machine: Optional[Dict[str, str]] = None,
    ) -> Phase1Result:
        """Build a candidate Phase-1 result with knitting tasks re-timed to new_* (and
        re-machined to new_machine where given — the parallel-PO dedication changes the
        machine, so it MUST be applied or old-machine + new-time would overlap)."""
        info = {t["task_id"]: t for t in self.tasks}
        new_machine = new_machine or {}
        p1b_assignments: List[Dict[str, Any]] = []
        for a in p1.assignments:
            tid = a["task_id"]
            if tid in new_start:
                b = dict(a)
                b["start_time"] = new_start[tid]
                b["end_time"] = new_end[tid]
                if tid in new_machine:
                    b["machine_id"] = new_machine[tid]
                due = int(info.get(tid, {}).get("due_at_min", new_end[tid] + 1) or (new_end[tid] + 1))
                b["status"] = "LATE" if new_end[tid] > due else "ON_TIME"
                p1b_assignments.append(b)
            else:
                p1b_assignments.append(a)
        return Phase1Result(
            status="feasible",
            assignments=p1b_assignments,
            overloads=p1.overloads,
            start_times={**p1.start_times, **new_start},
            end_times={**p1.end_times, **new_end},
            solve_time_seconds=0.0,
            solver_status_name=p1.solver_status_name,
        )

    def _verify_config(self) -> Dict[str, Any]:
        """Cheaper solver budget for CANDIDATE re-solves (reorder verifies).  The verify
        re-runs phases 2–5 only to check that a knitting relayout does not raise lateness
        — it is not the production solve, and the deterministic left-shift post-passes
        repair tightness afterward, so it does not need the full det-time budget.  Caps
        ``max_deterministic_time`` to ``knitting_reorder_verify_det`` (or a fraction of
        the configured budget).  Everything else (shift_ends, seeds, flags) is copied."""
        cfg = dict(self.config)
        explicit = cfg.get("max_deterministic_time")
        base = float(explicit) if explicit is not None else float(
            min(cfg.get("max_search_time", 12) or 12, 12)
        )
        cap = cfg.get("knitting_reorder_verify_det")
        cfg["max_deterministic_time"] = (
            float(cap) if cap is not None
            else max(2.0, base * float(cfg.get("knitting_reorder_verify_frac", 0.5)))
        )
        return cfg

    def _try_knitting_relayout(
        self,
        p2_tasks: List[Dict[str, Any]],
        all_resources: List[Dict[str, Any]],
        global_horizon: int,
        p1: Phase1Result,
        p2: Phase2Result,
        p3: "Phase3Result",
        p4: "Phase4Result",
        p5: "Phase4Result",
    ):
        """ONE verified knitting-relayout pass that COMBINES three candidate transforms,
        so they share a SINGLE phases-2–5 re-solve instead of one each:

          1. parallel component-PO  — dedicate machines per component PO so a garment's
             POs knit in parallel (first panel ready sooner; feeds linking early).
          2. order-contiguity        — re-sequence each machine so an order's tasks run
             contiguously (no A→B→A interleave).
          3. yarn-config repair      — move tasks that force a machine back into a yarn
             config it already left onto a machine whose tail matches their config.

        (2) runs ON TOP of (1)'s layout and (3) on the merge of both, then the merged
        candidate is re-solved through phases 2–5 (on the cheaper ``_verify_config``
        budget) and accepted ONLY if total pipeline lateness (Σ tardiness AND late-order
        count) does not increase.  Any subset of the transforms may apply.  Returns the
        refined phase results or None (keep the solver plan)."""
        info = {t["task_id"]: t for t in self.tasks}

        cand_p = (
            parallelize_component_pos(p1.assignments, self.tasks, all_resources, self.config)
            if self.config.get("enable_knitting_parallel_pos", True) else None
        )
        # Interim assignments after the parallel-PO dedication (start/end/machine).
        if cand_p is not None:
            interim: List[Dict[str, Any]] = []
            for a in p1.assignments:
                tid = a["task_id"]
                if tid in cand_p["start"]:
                    b = dict(a)
                    b["start_time"] = cand_p["start"][tid]
                    b["end_time"] = cand_p["end"][tid]
                    b["machine_id"] = cand_p["machine"][tid]
                    interim.append(b)
                else:
                    interim.append(a)
        else:
            interim = p1.assignments

        cand_c = (
            reorder_contiguous_knitting(interim, self.tasks, self.config)
            if self.config.get("enable_knitting_contiguity_reorder", False) else None
        )

        # Merge: machine + times from the parallel layout, then overlay contiguity times.
        final_start: Dict[str, int] = {}
        final_end: Dict[str, int] = {}
        final_machine: Dict[str, str] = {}
        for a in interim:
            t = info.get(a["task_id"])
            if t is None or t.get("operation", "").lower() != "knitting":
                continue
            tid = a["task_id"]
            final_start[tid] = a["start_time"]
            final_end[tid] = a["end_time"]
            final_machine[tid] = a["machine_id"]
        if cand_c is not None:
            final_start.update(cand_c["start"])
            final_end.update(cand_c["end"])  # machine unchanged by per-machine reorder

        # 3. yarn-config re-entry repair — runs on the MERGED layout (it needs the
        #    final per-machine sequences): relocate slack-due tasks that force a
        #    machine back into a yarn config it already left (:2→:5→:2) onto a
        #    compatible machine whose tail holds the matching config.
        cand_y = None
        if self.config.get("enable_knitting_yarn_config_repair", True):
            merged: List[Dict[str, Any]] = []
            for a in interim:
                tid = a["task_id"]
                if tid in final_start:
                    b = dict(a)
                    b["start_time"] = final_start[tid]
                    b["end_time"] = final_end[tid]
                    b["machine_id"] = final_machine[tid]
                    merged.append(b)
                else:
                    merged.append(a)
            cand_y = repair_yarn_config_reentry(merged, self.tasks, self.config)
        if cand_p is None and cand_c is None and cand_y is None:
            return None
        if cand_y is not None:
            final_start.update(cand_y["start"])
            final_end.update(cand_y["end"])
            final_machine.update(cand_y["machine"])

        p1b = self._rebuild_p1_from_knitting(p1, final_start, final_end, final_machine)

        verify_cfg = self._verify_config()
        saved_cfg = self.config
        try:
            self.config = verify_cfg
            p2b: Phase2Result = solve_linking(
                p2_tasks, all_resources, verify_cfg,
                p1_start_times=p1b.start_times,
                p1_end_times=p1b.end_times,
                translation_map=self.translation_map,
                horizon=global_horizon,
                reschedule_hint=None,
                workload_shrank=False,
                all_pipeline_tasks=self.tasks,
            )
            if p2b.status != "feasible":
                logger.info(f"🧩 Knitting relayout: linking re-solve status={p2b.status} — keeping solver plan.")
                return None
            self._tighten_linking(p1b, p2b, all_resources)
            chain = self._solve_phases_3_to_5(
                all_resources, global_horizon, {**p1b.end_times, **p2b.end_times}
            )
        finally:
            self.config = saved_cfg
        if isinstance(chain, dict):
            logger.info("🧩 Knitting relayout: downstream re-solve failed — keeping solver plan.")
            return None
        p3b, p4b, p5b = chain

        base_ends = {**p1.end_times, **p2.end_times, **p3.end_times, **p4.end_times, **p5.end_times}
        cand_ends = {**p1b.end_times, **p2b.end_times, **p3b.end_times, **p4b.end_times, **p5b.end_times}
        if set(base_ends) != set(cand_ends):
            logger.info("🧩 Knitting relayout: task-set mismatch — keeping solver plan.")
            return None
        base_late, base_n = self._total_lateness(base_ends)
        cand_late, cand_n = self._total_lateness(cand_ends)
        if cand_late > base_late or cand_n > base_n:
            logger.info(
                f"🧩 Knitting relayout: REJECTED — lateness would rise "
                f"(Σ {base_late}→{cand_late}, late orders {base_n}→{cand_n}). Keeping solver plan."
            )
            return None
        what = "+".join(
            x for x in (
                ("parallel-PO" if cand_p else None),
                ("contiguity" if cand_c else None),
                ("yarn-config" if cand_y else None),
            ) if x
        )
        logger.info(
            f"✨ Knitting relayout ACCEPTED ({what}): Σ lateness {base_late}→{cand_late}, "
            f"late orders {base_n}→{cand_n}."
        )
        return p1b, p2b, p3b, p4b, p5b

    def _try_sameqty_relink(
        self,
        p2_tasks: List[Dict[str, Any]],
        all_resources: List[Dict[str, Any]],
        global_horizon: int,
        p1: Phase1Result,
        p2: Phase2Result,
        p3: "Phase3Result",
        p4: "Phase4Result",
        p5: "Phase4Result",
    ):
        """Pass 2: linking với floor same-qty + Pareto caps, rồi 3→5 lại.

        Nhận pass 2 CHỈ KHI mọi task (linking + toàn hạ nguồn) có end ≤ end
        pass 1 và có ít nhất một task sớm hơn thật.  Mọi nhánh khác → None
        (giữ pass 1).  Không bao giờ làm lịch xấu đi — guard theo cấu trúc.
        """
        linking_tasks = [
            t for t in p2_tasks if t.get("operation", "").lower() in PHASE2_OPS
        ]
        if not linking_tasks:
            return None

        lb_index = _compute_linking_start_lb(
            linking_tasks, p1.start_times, p1.end_times, self.translation_map
        )
        lb_sameqty = compute_sameqty_start_lb(
            linking_tasks, p1.start_times, p1.end_times, self.translation_map, self.tasks
        )
        slack = sum(
            1 for k, v in lb_index.items() if lb_sameqty.get(k, v) < v
        )
        if slack == 0:
            logger.info("🔁 Same-qty re-link: floor identical to index — skipped (no slack).")
            return None

        logger.info(f"🔁 Same-qty re-link: {slack} linking task(s) have earlier same-qty floor — pass 2…")
        p2b: Phase2Result = solve_linking(
            p2_tasks, all_resources, self.config,
            p1_start_times=p1.start_times,
            p1_end_times=p1.end_times,
            translation_map=self.translation_map,
            horizon=global_horizon,
            reschedule_hint=None,
            workload_shrank=False,
            start_lb_override=lb_sameqty,
            end_caps=p2.end_times,
        )
        if p2b.status != "feasible":
            logger.warning(f"🔁 Same-qty re-link: pass 2 status={p2b.status} — keeping pass 1.")
            return None
        if all(p2b.end_times.get(k) == v for k, v in p2.end_times.items()):
            logger.info("🔁 Same-qty re-link: pass 2 changed nothing — keeping pass 1.")
            return None

        self._tighten_linking(p1, p2b, all_resources)
        chain = self._solve_phases_3_to_5(
            all_resources, global_horizon, {**p1.end_times, **p2b.end_times}
        )
        if isinstance(chain, dict):
            logger.warning("🔁 Same-qty re-link: downstream re-solve failed — keeping pass 1.")
            return None
        p3b, p4b, p5b = chain

        # ── Pareto verify: điểm-theo-điểm trên TOÀN bộ task của phases 2–5 ──
        ends1: Dict[str, int] = {**p2.end_times, **p3.end_times, **p4.end_times, **p5.end_times}
        ends2: Dict[str, int] = {**p2b.end_times, **p3b.end_times, **p4b.end_times, **p5b.end_times}
        if set(ends1) != set(ends2):
            logger.warning("🔁 Same-qty re-link: task set mismatch — keeping pass 1.")
            return None
        regressed = [k for k, v in ends1.items() if ends2[k] > v]
        improved = sum(1 for k, v in ends1.items() if ends2[k] < v)
        if regressed:
            logger.info(
                f"🔁 Same-qty re-link: {len(regressed)} task(s) would finish later "
                f"(e.g. {regressed[0]}) — keeping pass 1 (Pareto guard)."
            )
            return None
        if improved == 0:
            logger.info("🔁 Same-qty re-link: no task improved — keeping pass 1.")
            return None

        total_gain = sum(v - ends2[k] for k, v in ends1.items())
        logger.info(
            f"✨ Same-qty re-link ACCEPTED: {improved} task(s) earlier, 0 later, "
            f"total {total_gain} task-min pulled forward."
        )
        return p2b, p3b, p4b, p5b


# ---------------------------------------------------------------------------
# Reschedule-hint partitioning (B.3)
# ---------------------------------------------------------------------------

_PHASE1_OP_SET = {"knitting", "capacity_block"}
_PHASE2_OP_SET = {"linking"}
_PHASE3_OP_SET = {"washing"}
_PHASE4_OP_SET = {"iron", "ironing"}  # Phase 5 (packing + rest) takes everything else downstream
                                      # (payloads use op "Iron"→"iron"; accept "ironing" too)


def _washing_reuse_from_hint(
    reschedule_hint: Dict[str, Any],
    p3_tasks: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Dựng lại washing assignments NGUYÊN VĂN từ ``previous_assignments`` của một
    re-schedule hint ngoài (Go), cùng bộ field solve_washing sinh ra.

    Trả None (→ caller re-solve như cũ) khi: không có washing task; BẤT KỲ washing
    task hiện tại nào thiếu prev khớp task_id / prev thiếu machine/start/end (task
    giặt mới xuất hiện); hoặc một task pinned có cửa sổ pin LỆCH prev (để solve xử
    pin chuẩn qua normalize_pinned_window thay vì đoán).

    ``batch_slot_id`` không có trong hint (Go không gửi lại) → tổng hợp
    ``keep_<start>``: các thành viên một cycle chung (machine, start) nhận cùng
    slot id, và vì start giữ nguyên run-over-run nên id ổn định — không phá tính
    bất động điểm.  ``status`` tính lại theo due hiện tại (due có thể đã đổi).
    """
    if not p3_tasks:
        return None
    prev_by_id = {
        p["task_id"]: p
        for p in (reschedule_hint.get("previous_assignments") or [])
    }
    out: List[Dict[str, Any]] = []
    for t in p3_tasks:
        tid = t["task_id"]
        p = prev_by_id.get(tid)
        if (
            p is None
            or p.get("machine_id") is None
            or p.get("start_time") is None
            or p.get("end_time") is None
        ):
            return None  # washing task mới / prev thiếu dữ liệu → re-solve
        s, e = int(p["start_time"]), int(p["end_time"])
        if t.get("is_pinned"):
            ps, pe = t.get("pinned_start_time"), t.get("pinned_end_time")
            pm = t.get("pinned_machine_id")
            if (
                (ps is not None and int(ps) != s)
                or (pe is not None and int(pe) != e)
                or (pm and pm != p["machine_id"])
            ):
                return None  # pin lệch prev → re-solve xử pin chuẩn
        due = int(t.get("due_at_min", e + 1) or (e + 1))
        out.append({
            "task_id": tid,
            "machine_id": p["machine_id"],
            "start_time": s,
            "end_time": e,
            "group_id": t.get("group_id", ""),
            "order_id": t.get("original_order_id", ""),
            "quantity": t.get("qty", 0),
            "status": "LATE" if e > due else "ON_TIME",
            "batch_slot_id": f"keep_{s}",
        })
    return out


def _repair_reused_washing(
    p1_wash: List[Dict[str, Any]],
    info: Dict[str, Dict[str, Any]],
    dep_ends: Dict[str, int],
    shift_ends: List[int],
) -> Optional[List[Dict[str, Any]]]:
    """Repair a reused pass-1 washing layout whose members start before their pass-2
    linking dependency: delay ONLY the violating cycles (whole batch moves together)
    to the earliest boundary-safe free slot ≥ their latest dependency end on the SAME
    machine; every other cycle keeps its pass-1 position byte-identical.

    Returns a NEW assignment list on success, or None when some violating cycle cannot
    be re-placed (caller then falls back to a full washing re-solve).  A delayed batch
    only moves LATER, so it can never race ahead of its linking; downstream (iron/
    packing) solves AFTER this on the returned end_times, so it follows the delay.
    Deterministic (cycles processed by required start, then machine id).
    """
    wash = [dict(a) for a in p1_wash]
    bounds = sorted({int(s) for s in shift_ends})

    cyc: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for a in wash:
        cyc.setdefault((a["machine_id"], int(a["start_time"])), []).append(a)

    def _span(members: List[Dict[str, Any]]) -> Tuple[int, int]:
        return (
            min(int(a["start_time"]) for a in members),
            max(int(a["end_time"]) for a in members),
        )

    # Required (dependency-safe) start per violating cycle.
    need: Dict[Tuple[str, int], int] = {}
    for key, members in cyc.items():
        req = key[1]
        for a in members:
            t = info.get(a["task_id"], {})
            for d in (t.get("final_depends_on") or []):
                if d in dep_ends:
                    req = max(req, int(dep_ends[d]))
        if req > key[1]:
            need[key] = req

    # Occupancy from the cycles that stay put.
    busy: Dict[str, List[Tuple[int, int]]] = {}
    for key, members in cyc.items():
        if key in need:
            continue
        busy.setdefault(key[0], []).append(_span(members))

    for key in sorted(need, key=lambda k: (need[k], k[0])):
        members = cyc[key]
        s0, e0 = _span(members)
        dur = e0 - s0
        m = key[0]
        occupied = busy.setdefault(m, [])
        # Earliest boundary-safe free slot ≥ the dependency end: try the requirement
        # itself, then each busy-interval end and shift boundary after it.
        cands = sorted(
            {need[key]}
            | {be for (_bs, be) in occupied if be >= need[key]}
            | {b for b in bounds if b >= need[key]}
        )
        placed = None
        for c in cands:
            e = c + dur
            if any(c < b < e for b in bounds):
                continue  # washing must not straddle a break
            if any(c < be and bs < e for (bs, be) in occupied):
                continue
            placed = c
            break
        if placed is None:
            return None
        delta = placed - s0
        for a in members:
            a["start_time"] = int(a["start_time"]) + delta
            a["end_time"] = int(a["end_time"]) + delta
            due = info.get(a["task_id"], {}).get("due_at_min")
            if due is not None:
                a["status"] = "LATE" if a["end_time"] > int(due) else "ON_TIME"
        occupied.append((placed, placed + dur))
    return wash


def _detect_workload_shrink(
    reschedule_hint: Optional[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> bool:
    """True when orders present in the previous plan are GONE from the current
    task set (e.g. re-scheduling 10 orders down to 5).

    Detected at ORDER level so it is rename-safe: slicing changes task_ids but
    keeps the original_order_id, so a renamed task is NOT counted as removed.
    Falls back to task_id level only when the hint carries no order metadata.

    Why it matters: the previous absolute start times describe a denser plan.
    Keeping them — hard pin OR the symmetric |Δ| soft penalty — freezes the
    surviving tasks at their old clock positions and leaves GAPS where the
    removed work used to sit.  On a shrink every phase relaxes to a one-sided
    (late-only) anchor + cold-style compaction so the schedule re-packs forward.
    """
    if not reschedule_hint:
        return False
    previous = reschedule_hint.get("previous_assignments") or []
    if not previous:
        return False

    current_orders = {
        t.get("original_order_id", "") for t in tasks if t.get("original_order_id")
    }
    prev_orders = {
        p.get("original_order_id", "") for p in previous if p.get("original_order_id")
    }
    if prev_orders:
        return bool(prev_orders - current_orders)

    # No order metadata in the hint — fall back to task_id level (rename-unsafe,
    # but better than silently missing a genuine order removal).
    current_tids = {t["task_id"] for t in tasks}
    prev_tids = {p.get("task_id") for p in previous}
    return bool(prev_tids - current_tids)


def partition_hint_for_pipeline(
    reschedule_hint: Optional[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Split a global reschedule_hint into per-phase / per-group sub-hints.

    Output shape:
        {
          "knitting":   [PreviousAssignment dicts],
          "linking":    [...],
          "washing":    { (color, substance): [...], ... },
          "ironing":    [...],
          "downstream": [...],   # packing + any other downstream op
        }

    The previous_assignments are matched to the current task list by task_id.
    Tasks whose task_id is unknown go to whichever bucket their original_order_id
    maps to via existing tasks (best-effort); otherwise they are dropped so a
    later fallback in apply_stability_objective can still match them by order.

    Always returns the full skeleton so callers can `.get()` safely.
    """
    result: Dict[str, Any] = {
        "knitting": [],
        "linking": [],
        "washing": {},
        "ironing": [],
        "downstream": [],
    }
    if not reschedule_hint:
        return result

    previous = reschedule_hint.get("previous_assignments") or []
    if not previous:
        return result

    task_op: Dict[str, str] = {
        t["task_id"]: t.get("operation", "").lower() for t in tasks
    }
    task_wash_key: Dict[str, Tuple[str, str]] = {
        t["task_id"]: (t.get("color", ""), t.get("substance", ""))
        for t in tasks
        if t.get("operation", "").lower() == "washing"
    }
    # Order-keyed fallback for rename cases (slicing changed → task_id no longer
    # matches but original_order_id still does).  Map order → set of ops it
    # touches, plus per-op washing keys, so a renamed prev can still land in
    # the right partition bucket.
    order_ops: Dict[str, Set[str]] = {}
    order_wash_keys: Dict[str, Set[Tuple[str, str]]] = {}
    for t in tasks:
        oid = t.get("original_order_id", "")
        if not oid:
            continue
        op = t.get("operation", "").lower()
        order_ops.setdefault(oid, set()).add(op)
        if op == "washing":
            order_wash_keys.setdefault(oid, set()).add(
                (t.get("color", ""), t.get("substance", ""))
            )

    def _route(op: str, prev: Dict[str, Any], wash_key: Tuple[str, str]) -> None:
        if op in _PHASE1_OP_SET:
            result["knitting"].append(prev)
        elif op in _PHASE2_OP_SET:
            result["linking"].append(prev)
        elif op in _PHASE3_OP_SET:
            result["washing"].setdefault(wash_key, []).append(prev)
        elif op in _PHASE4_OP_SET:
            result["ironing"].append(prev)
        elif op:
            result["downstream"].append(prev)

    for prev in previous:
        op = task_op.get(prev["task_id"], "").lower()
        if op:
            wash_key = task_wash_key.get(prev["task_id"], ("", ""))
            _route(op, prev, wash_key)
            continue

        # Rename / slicing case — task_id unknown.  Fan out by order's ops so
        # apply_stability_objective's machine-compatibility fallback can pick
        # a sensible match within each phase.
        oid = prev.get("original_order_id", "")
        if not oid or oid not in order_ops:
            continue
        for cand_op in order_ops[oid]:
            if cand_op == "washing":
                for wk in order_wash_keys.get(oid, {("", "")}):
                    _route(cand_op, prev, wk)
            else:
                _route(cand_op, prev, ("", ""))

    return result


def _hint_for_phase(
    base_hint: Optional[Dict[str, Any]],
    subset_previous: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Build a per-phase hint dict carrying the same weights as the global one."""
    if not base_hint or not subset_previous:
        return None
    return {
        "previous_assignments": list(subset_previous),
        "stability_weight_time_per_min": base_hint.get("stability_weight_time_per_min", 500),
        "stability_weight_machine_swap": base_hint.get("stability_weight_machine_swap", 5000),
        "match_by_order_fallback": base_hint.get("match_by_order_fallback", True),
    }


def _hint_for_phase_washing(
    base_hint: Optional[Dict[str, Any]],
    washing_groups: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    """Phase 3 hint: keep the per-group breakdown so _solve_group can filter."""
    if not base_hint or not washing_groups:
        return None
    # Flatten across groups; solve_washing will filter per (color, substance)
    flat = [p for prevs in washing_groups.values() for p in prevs]
    if not flat:
        return None
    h = _hint_for_phase(base_hint, flat)
    if h is None:
        return None
    # Carry the per-group partition for the per-group solver.
    h["_washing_groups"] = {k: list(v) for k, v in washing_groups.items()}
    return h


# ---------------------------------------------------------------------------
# Helpers (mirrors of builder.py preprocessing)
# ---------------------------------------------------------------------------

def _build_translation_map(tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map sub-task / original-order IDs → parent batch task ID."""
    translation: Dict[str, str] = {}
    for t in tasks:
        translation[t["task_id"]] = t["task_id"]
        if t.get("is_batch") and t.get("sub_tasks"):
            for sub in t["sub_tasks"]:
                translation[sub["task_id"]] = t["task_id"]
                if sub.get("original_order_id"):
                    translation[sub["original_order_id"]] = t["task_id"]
    return translation


def _sanitize_dummy_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge overlapping pinned dummy tasks (qty=0, is_pinned=True) on the same
    machine to prevent AddNoOverlap INFEASIBLE when Go maps multiple shift
    blockers to the same virtual machine slot.
    """
    from collections import defaultdict

    dummies = [t for t in tasks if t.get("is_pinned") and t.get("qty", 0) == 0]
    real_tasks = [t for t in tasks if not (t.get("is_pinned") and t.get("qty", 0) == 0)]

    if not dummies:
        return tasks

    grouped: Dict[str, List] = defaultdict(list)
    for t in dummies:
        r_ids = t.get("compatible_resource_ids") or []
        m_id = t.get("pinned_machine_id") or (r_ids[0] if r_ids else None)
        if not m_id:
            real_tasks.append(t)
            continue
        grouped[m_id].append(t)

    merged_dummies: List[Dict[str, Any]] = []
    for m_id, m_tasks in sorted(grouped.items()):
        intervals = [
            (int(t["pinned_start_time"]), int(t["pinned_end_time"]), t)
            for t in m_tasks
            if t.get("pinned_start_time") is not None and t.get("pinned_end_time") is not None
        ]
        if not intervals:
            continue

        intervals.sort(key=lambda x: x[0])
        merged = []
        cs, ce, bt = intervals[0]
        for s, e, t in intervals[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                merged.append((cs, ce, bt))
                cs, ce, bt = s, e, t
        merged.append((cs, ce, bt))

        for idx, (s, e, base_t) in enumerate(merged):
            new_dummy = dict(base_t)
            new_dummy["task_id"] = f"DUMMY_MERGED_{m_id}_{s}_{e}_{idx}"
            new_dummy["original_order_id"] = new_dummy["task_id"]
            new_dummy["pinned_start_time"] = s
            new_dummy["pinned_end_time"] = e
            new_dummy["duration"] = e - s
            merged_dummies.append(new_dummy)

        if len(intervals) > len(merged):
            logger.info(
                f"🧹 Merged {len(intervals)} dummy tasks → {len(merged)} for machine {m_id}"
            )

    return real_tasks + merged_dummies


def _phase_failure_result(
    status: str,
    tasks: List[Dict[str, Any]],
    resources: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a structured failure response when a phase cannot find a schedule."""
    if resources is not None and config is not None and horizon is not None:
        overloads = diagnose_infeasibility(tasks, resources, config, horizon, status)
    else:
        overloads = [
            {
                "task_id": t["task_id"],
                "order_id": t.get("original_order_id", ""),
                "status": "UNSCHEDULABLE",
                "delay_minutes": 0,
                "root_cause_code": status.upper(),
                "bottleneck_resource_id": None,
                "quantity": t.get("qty", 0),
            }
            for t in tasks
        ]
    return {
        "status": status,
        "assignments": [],
        "overloads": overloads,
        "objective_value": None,
        "solve_time_seconds": 0.0,
    }
