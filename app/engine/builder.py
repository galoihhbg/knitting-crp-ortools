import re
import logging
from typing import Dict, Any, List, Optional, Set

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)

# Affinity penalty constants — higher score = less preferred assignment
PENALTY_CHANGE_DESIGN: int = 500  # Must remove current mold and install a different one
PENALTY_COLD_START: int = 200     # Machine is idle and needs a mold installed from scratch
PENALTY_CHANGE_COLOR: int = 100   # Same mold but thread color must be changed
PENALTY_SETUP_COLOR: int = 50     # Machine has no thread color yet — light setup needed


class TaskModelBuilder:
    """
    Incrementally builds a CP-SAT scheduling model using the Builder Pattern.
    Each method returns `self` so calls can be chained fluently.

    Usage:
        builder = (
            TaskModelBuilder(config, resources, tasks, machine_states)
            .build_time_variables()
            .build_resource_allocations()
            .apply_routing_constraints()
            .apply_dependency_constraints()
            .apply_batch_offset_constraints()
            .define_objective()
        )
        status = solver.Solve(builder.model)
        result = builder.extract_results(solver, status)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        resources: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        machine_states: Dict[str, Dict[str, str]],
        material_capacities: Optional[Dict[str, int]] = None,
    ) -> None:
        self.model = cp_model.CpModel()
        self.config = config
        self.tasks = tasks
        self.machine_states = machine_states

        # resource_map: id -> resource dict; "intervals" list is added during allocation
        self.resource_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in resources}

        # task_vars: task_id -> {start, end, literals, r_ids, due, ...}
        self.task_vars: Dict[str, Dict[str, Any]] = {}

        # Accumulated terms for the minimization objective
        self.objective_terms: List[Any] = []

        # Use config horizon, but guarantee it is large enough for all task durations
        total_duration = sum(int(t.get("duration", 0)) for t in self.tasks)
        config_horizon = int(self.config.get("horizon_minutes", 40320))
        self.horizon: int = max(config_horizon, total_duration + 5000)

        # Dynamic weight calibration — keeps LATENESS : AFFINITY : ACTIVATION ≈ 1000 : 10 : 2
        # regardless of horizon size. Larger horizons need a bigger absolute coefficient so that
        # a 1-minute delay doesn't appear negligible relative to the huge time scale.
        self.lateness_scale: int = max(1, self.horizon // 1000)

        # Tasks that were skipped because no compatible resource exists.
        # Populated by build_time_variables() and build_resource_allocations().
        # Engine.solve() checks this after the builder chain to return a structured
        # infeasible result instead of silently proceeding with a partial model.
        self.no_resource_tasks: List[Dict[str, Any]] = []

        # Per-material creel capacities: material_code → max concurrent rolls/slots.
        # Populated from SolverPayload.material_capacities; empty dict = feature disabled.
        self.material_capacities: Dict[str, int] = material_capacities or {}

        # Build the sub-task → batch-task translation map once during construction
        self.task_translation_map: Dict[str, str] = {}
        self._build_translation_map()
        
        # Merge overlapping dummy shift tasks to prevent AddNoOverlap infeasible crashes
        self._sanitize_dummy_tasks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize_dummy_tasks(self) -> None:
        """
        Groups and merges any overlapping dummy tasks (is_pinned=True, qty=0) that are
        assigned to the same machine.
        
        Why: When worker count spikes exceed the available W_LINKING_XX virtual slots,
        the Go backend maps multiple shift blockers to the same virtual machine.
        If these blockers overlap in time, OR-Tools will fail immediately on AddNoOverlap.
        Since dummy tasks just represent 'blocked time' rather than real jobs,
        overlapping them just means a continuous blocked interval.
        """
        # Separate dummies and real tasks
        dummies = [t for t in self.tasks if t.get("is_pinned") and t.get("qty", 0) == 0]
        real_tasks = [t for t in self.tasks if not (t.get("is_pinned") and t.get("qty", 0) == 0)]
        
        if not dummies:
            return

        # Group by effective machine ID
        from collections import defaultdict
        grouped_dummies = defaultdict(list)
        for t in dummies:
            _rids = t.get("compatible_resource_ids") or []
            m_id = t.get("pinned_machine_id") or (_rids[0] if _rids else None)
            if not m_id:
                real_tasks.append(t) # Should be caught by normal no_resource block later
                continue
            grouped_dummies[m_id].append(t)
            
        merged_dummies = []
        for m_id, tasks_for_machine in grouped_dummies.items():
            # Filter and sort intervals
            intervals = []
            for t in tasks_for_machine:
                s = t.get("pinned_start_time")
                e = t.get("pinned_end_time")
                if s is not None and e is not None:
                    intervals.append((int(s), int(e), t))
                else:
                    real_tasks.append(t) # Malformed dummy, leave it for normal workflow
            
            if not intervals:
                continue
                
            intervals.sort(key=lambda x: x[0])
            
            # Sweep line merge
            merged = []
            current_start, current_end, base_t = intervals[0]
            
            for s, e, t in intervals[1:]:
                if s <= current_end:  # Overlap or contiguous
                    current_end = max(current_end, e)
                else:
                    merged.append((current_start, current_end, base_t))
                    current_start, current_end, base_t = s, e, t
            merged.append((current_start, current_end, base_t))
            
            # Create consolidated dummy tasks
            for idx, (s, e, base_t) in enumerate(merged):
                new_dummy = dict(base_t) # Copy to preserve group_id, operation, etc
                new_dummy["task_id"] = f"DUMMY_MERGED_{m_id}_{s}_{e}_{idx}"
                new_dummy["original_order_id"] = new_dummy["task_id"]
                new_dummy["pinned_start_time"] = s
                new_dummy["pinned_end_time"] = e
                new_dummy["duration"] = e - s
                merged_dummies.append(new_dummy)
                
            if len(intervals) > len(merged):
                logger.info(f"🧹 Merged {len(intervals)} overlapping dummy tasks into {len(merged)} for machine {m_id}")
                
        # Rebuild full task list
        self.tasks = real_tasks + merged_dummies

    def _build_translation_map(self) -> None:
        """
        Map every sub-task / original-order ID back to its parent batch task ID.
        This lets dependency resolution work even when Go sends raw sub-task IDs
        inside final_depends_on before batching IDs are available downstream.
        """
        for t in self.tasks:
            self.task_translation_map[t["task_id"]] = t["task_id"]
            if t.get("is_batch") and t.get("sub_tasks"):
                for sub in t["sub_tasks"]:
                    self.task_translation_map[sub["task_id"]] = t["task_id"]
                    if sub.get("original_order_id"):
                        self.task_translation_map[sub["original_order_id"]] = t["task_id"]

    def _compute_affinity_penalty(
        self,
        resource: Dict[str, Any],
        task_design: str,
        task_color_str: str, # Lúc này Go đang gửi chuỗi YarnConfig qua field 'color_config'
    ) -> int:
        """
        Tính toán điểm phạt (penalty) khi gán task vào một máy cụ thể.
        Điểm càng cao -> Setup càng lâu -> Solver càng né tránh máy này.
        """
        curr_design = resource.get("design_item_id", "")
        curr_color_str = resource.get("color_config", "")
        penalty = 0

        # ---------------------------------------------------------
        # 1. SETUP FILE THIẾT KẾ (Rất nhanh, chỉ cần nạp USB)
        # ---------------------------------------------------------
        PENALTY_CHANGE_DESIGN = 10 
        if task_design and curr_design and curr_design != task_design:
            penalty += PENALTY_CHANGE_DESIGN

        # ---------------------------------------------------------
        # 2. SETUP DÀN CỌC SỢI (Rất lâu, tốn công xỏ dây)
        # ---------------------------------------------------------
        PENALTY_COLD_START = 200     # Phạt máy trống, phải lên dàn cọc từ đầu
        PENALTY_PER_ROLL_SWAP = 100  # Phạt nặng cho MỖI MỘT cuộn sợi phải tháo/lắp

        if not task_color_str:
            return penalty # Không có thông tin sợi -> Bỏ qua

        if curr_color_str == task_color_str:
            # Dàn cọc giống HỆT NHAU -> Tuyệt vời, 0 điểm phạt sợi!
            pass
            
        elif curr_color_str == "":
            # Máy chưa có sợi, tốn công xỏ dây mới hoàn toàn
            # Đếm tổng số cuộn cần xỏ từ chuỗi (Ví dụ: "MAT_A:2|MAT_B:1" -> 3 cuộn)
            total_rolls = sum(int(item.split(':')[1]) for item in task_color_str.split('|') if ':' in item)
            
            # Xỏ mới từ đầu thường nhanh hơn việc tháo cái cũ ra rồi lắp cái mới vào
            penalty += PENALTY_COLD_START + (total_rolls * int(PENALTY_PER_ROLL_SWAP / 2))
            
        else:
            # CÓ SỰ LỆCH PHA -> TÍNH TOÁN SỐ CUỘN CẦN THAY THẾ
            def parse_yarns(y_str: str) -> Dict[str, int]:
                res = {}
                if not y_str: return res
                for item in y_str.split('|'):
                    if ':' in item:
                        mat, qty = item.split(':')
                        res[mat] = int(qty)
                    else:
                        res[item] = 1 # Fallback an toàn nếu chuỗi bị lỗi format
                return res

            # Parse chuỗi thành Dictionary. VD: {'MAT_BLK_05': 1, 'MAT_RED_01': 2}
            curr_yarns = parse_yarns(curr_color_str)
            task_yarns = parse_yarns(task_color_str)

            swaps_needed = 0
            
            # So sánh Sợi của Task vs Sợi đang cắm trên Máy
            for mat, target_qty in task_yarns.items():
                current_qty = curr_yarns.get(mat, 0)
                if target_qty > current_qty:
                    # Nếu thiếu bao nhiêu cuộn thì phải mất công chạy đi lấy và xỏ vào
                    swaps_needed += (target_qty - current_qty)
            
            # Cộng dồn hình phạt dựa trên số cuộn phải thao tác
            penalty += (swaps_needed * PENALTY_PER_ROLL_SWAP)

        return penalty

    # ------------------------------------------------------------------
    # Builder steps
    # ------------------------------------------------------------------

    def build_time_variables(self) -> "TaskModelBuilder":
        """
        Create CP-SAT start/end/interval variables and a lateness penalty variable
        for every task that has at least one compatible resource.
        Tasks with no compatible resources are skipped with a warning so the
        solver never crashes on bad input (defensive programming).
        """
        # Deduplicate task_ids — Go backend may emit duplicate IDs for split segments.
        # Duplicate CP-SAT variable names cause silent constraint mis-binding and
        # double-add NoOverlap intervals on the same machine, making the model INFEASIBLE.
        seen_task_ids: Set[str] = set()
        for t in self.tasks:
            orig = t["task_id"]
            if orig in seen_task_ids:
                counter = 2
                candidate = f"{orig}_dup{counter}"
                while candidate in seen_task_ids:
                    counter += 1
                    candidate = f"{orig}_dup{counter}"
                t["task_id"] = candidate
                logger.warning(
                    f"⚠️ Duplicate task_id '{orig}' renamed → '{t['task_id']}' "
                    f"(pinned={t.get('is_pinned')}, start={t.get('pinned_start_time')}, end={t.get('pinned_end_time')})"
                )
            seen_task_ids.add(t["task_id"])

        for t in self.tasks:
            t_id = t["task_id"]
            compatible_ids = t.get("compatible_resource_ids", [])
            operation = t.get("operation", "").lower()
            if t.get("operation") != "capacity_block" and not compatible_ids:
                logger.warning(f"⚠️ Task {t_id} has NO compatible resources — skipping.")
                self.no_resource_tasks.append(t)
                continue

            priority = int(t.get("priority", 5))
            due_at = int(t.get("due_at_min", self.horizon))
            start_after = int(t.get("start_after_min", 0))
            deps = t.get("final_depends_on") or []
            is_pinned = t.get("is_pinned", False)

            # NẾU BỊ GHIM (PINNED)
            if t.get("is_pinned") and t.get("pinned_start_time") is not None and t.get("pinned_end_time") is not None:
                start_val = int(t["pinned_start_time"])
                end_val = int(t["pinned_end_time"])

                # Dùng biến hằng số (Constant), không cho Solver xê dịch
                start_var = self.model.NewConstant(start_val)
                end_var = self.model.NewConstant(end_val)
                
                # Không cần ép `end_var == start_var + duration` nữa vì nó đã cố định.

            # NẾU LÀ TASK MỚI (TỰ DO) — hoặc pinned nhưng thiếu thời gian ghim
            else:
                start_var = self.model.NewIntVar(0, self.horizon, f"start_{t_id}")
                end_var = self.model.NewIntVar(0, self.horizon, f"end_{t_id}")
                if operation != "capacity_block":
                    # Ép duration rõ ràng: end == start + duration
                    duration_val = int(t.get("duration", 0))
                    self.model.Add(end_var == start_var + duration_val)
                    if start_after > 0 and not is_pinned:
                        self.model.Add(start_var >= start_after)

            # Weighted lateness (Only for real tasks, avoid horizon violations on pure dummy blockers)
            is_dummy = is_pinned and (t.get("qty", 0) == 0 or operation == "capacity_block")
            if not is_dummy:
                weight = 10 ** (6 - priority)
                lateness = self.model.NewIntVar(0, self.horizon, f"lat_{t_id}")
                self.model.Add(lateness >= end_var - due_at)
                self.objective_terms.append(lateness * weight * 1000 * self.lateness_scale)

            # Early-start preference: push every free task to start as early as
            # its constraints allow.  Without this, the solver accepts lazy
            # schedules where tasks sit idle for no reason (because lateness is
            # only penalised when end_var > due_at, not when there's slack).
            #
            # Weight = priority_weight × lateness_scale (1/1000 of lateness
            # per minute).  This is strong enough to outweigh the machine
            # activation cost (2 × lateness_scale) for any delay > ~1 minute,
            # so the solver will prefer assigning batches to separate machines
            # (parallelism) over serialising them on one machine to save
            # activation — which is the root cause of delayed linking.
            if not is_pinned and operation != "capacity_block":
                self.objective_terms.append(start_var * weight * self.lateness_scale)

            self.task_vars[t_id] = {
                "start": start_var,
                "end": end_var,
                "literals": [],        
                "r_ids": compatible_ids,
                "due": due_at,
                "original_order_id": t.get("original_order_id", ""),
                "group_id": t.get("group_id", ""),
                "depends_on": deps,
                "qty": t.get("qty", 0),
            }

            if deps:
                logger.info(f"   📌 {t_id} final_depends_on={deps}")

        return self
    
    def build_workforce_constraints(self):
        # Lấy giới hạn tuyệt đối của xưởng từ data truyền sang (Ví dụ: 100 máy)
        # Nếu không có, mặc định là 100
        print(f"\n🔢 MAX_FACTORY_MACHINES from config: {self.config.get('max_factory_machines', 'Not set, defaulting to 100')}")
        MAX_FACTORY_MACHINES = self.config.get("max_factory_machines", 100)

        # Ghost-task count guard: each capacity_block adds a fixed interval to AddCumulative.
        # Beyond 200 the cumulative propagator slows noticeably; log early so ops can split shifts.
        _ghost_count = sum(
            1 for t in self.tasks if t.get("operation", "").lower() == "capacity_block"
        )
        if _ghost_count > 200:
            logger.warning(
                f"⚠️ capacity_block count={_ghost_count} exceeds 200 — "
                "AddCumulative propagation may slow. Consider splitting shift windows."
            )

        # Choose workforce constraint strategy.
        # Boolean exclusion (slot-based NoOverlap) is lighter on RAM when there are many
        # capacity_block ghost tasks, because it avoids adding them all to one large
        # AddCumulative propagator.  It trades RAM for slightly more BoolVars.
        # Triggered automatically when ghost_count > 200, or by explicit config flag.
        _use_bool = bool(self.config.get("use_boolean_exclusion", False)) or _ghost_count > 200

        if _use_bool:
            logger.info(
                f"⚙️  Workforce mode: BOOLEAN EXCLUSION (ghost_count={_ghost_count}, "
                f"max_machines={MAX_FACTORY_MACHINES})"
            )
            self._build_workforce_boolean(MAX_FACTORY_MACHINES)
        else:
            self._build_workforce_cumulative(MAX_FACTORY_MACHINES)

    def _build_workforce_cumulative(self, MAX_FACTORY_MACHINES: int) -> None:
        """
        AddCumulative approach — uses OptionalIntervalVar for free knitting tasks
        so that demand is only counted when the task is actually assigned to a machine.

        BUG FIX: Previously used hard NewIntervalVar for knitting tasks. This caused
        ALL knitting tasks to always contribute demand=1 to the cumulative constraint,
        even at t=0 when no machine was selected yet, resulting in all tasks appearing
        to start simultaneously and violating the capacity limit.
        """
        knitting_intervals = []
        demands = []

        for t_id, tv in self.task_vars.items():
            task_info = next((t for t in self.tasks if t["task_id"] == t_id), {})
            operation = task_info.get("operation", "").lower()

            # Use actual duration: for pinned tasks, end-start may differ from "duration" field
            pinned_start = task_info.get("pinned_start_time")
            pinned_end = task_info.get("pinned_end_time")
            is_fully_pinned = (
                task_info.get("is_pinned") and pinned_start is not None and pinned_end is not None
            )
            duration = int(pinned_end) - int(pinned_start) if is_fully_pinned else int(task_info.get("duration", 0))

            # 1. TASK DỆT THỰC TẾ — dùng OptionalInterval để demand chỉ tính khi task đã được assign
            if operation == "knitting":
                literals = tv.get("literals", [])
                if literals:
                    # any_assigned = True khi task được gán vào bất kỳ máy nào
                    any_assigned = self.model.NewBoolVar(f"cumul_active_{t_id}")
                    self.model.AddMaxEquality(any_assigned, literals)
                    interval = self.model.NewOptionalIntervalVar(
                        tv["start"], duration, tv["end"], any_assigned,
                        f"global_interval_{t_id}"
                    )
                else:
                    # Pinned knitting task — luôn active, dùng hard interval
                    interval = self.model.NewIntervalVar(
                        tv["start"], duration, tv["end"],
                        f"global_interval_{t_id}"
                    )
                knitting_intervals.append(interval)
                demands.append(1)

            # 2. GHOST TASK KHÓA NĂNG LỰC — luôn active (pinned), dùng hard interval
            elif operation == "capacity_block":
                interval = self.model.NewIntervalVar(
                    tv["start"], duration, tv["end"],
                    f"global_interval_{t_id}"
                )
                knitting_intervals.append(interval)
                blocked_demand = int(task_info.get("demand", 0))
                demands.append(blocked_demand)

        # Lấp đầy khoảng TRỐNG giữa các ca làm (inter-shift gaps).
        # Trong khoảng trống không có capacity_block nào, solver có thể xếp tất cả
        # MAX_FACTORY_MACHINES task đồng thời (không có ràng buộc nào chặn).
        # Fix: với mỗi khoảng trống [gap_start, gap_end), thêm một interval cứng với
        # demand=MAX_FACTORY_MACHINES để chiếm hết slot → không task knitting nào
        # được lên lịch trong ca nghỉ.
        horizon = int(self.config.get("horizon_minutes", 57600))
        # Thu thập các khoảng thời gian của capacity_block (đã được pinned)
        block_windows: List[tuple] = []
        for t in self.tasks:
            if t.get("operation", "").lower() == "capacity_block":
                ps = t.get("pinned_start_time")
                pe = t.get("pinned_end_time")
                if ps is not None and pe is not None and int(pe) > int(ps):
                    block_windows.append((int(ps), int(pe)))
        # Sắp xếp và hợp nhất các block window (merge overlapping)
        block_windows.sort()
        merged: List[tuple] = []
        for s, e in block_windows:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        # Lấp khoảng TRỐNG giữa các ca làm bằng demand của ca có ràng buộc chặt nhất.
        # Chỉ áp dụng khi Go đã gửi ít nhất một capacity_block (có cấu trúc ca làm).
        # Nếu không có block nào → môi trường test/dev, bỏ qua.
        #
        # Dùng max_block_demand thay vì MAX_FACTORY_MACHINES để các task có thể
        # tiếp tục chạy qua ranh giới ca (autonomous machines), nhưng số lượng
        # đồng thời vẫn bị giới hạn như trong ca (không tăng đột biến trong gap).
        gap_count = 0
        if merged:
            max_block_demand = max(
                int(t.get("demand", 0))
                for t in self.tasks
                if t.get("operation", "").lower() == "capacity_block"
            )
            prev_end = 0
            for s, e in merged:
                if s > prev_end:
                    gap_iv = self.model.NewIntervalVar(
                        self.model.NewConstant(prev_end),
                        s - prev_end,
                        self.model.NewConstant(s),
                        f"gap_block_{prev_end}_{s}"
                    )
                    knitting_intervals.append(gap_iv)
                    demands.append(max_block_demand)
                    gap_count += 1
                prev_end = max(prev_end, e)
            # Sau block cuối cùng đến hết horizon
            if prev_end < horizon:
                gap_iv = self.model.NewIntervalVar(
                    self.model.NewConstant(prev_end),
                    horizon - prev_end,
                    self.model.NewConstant(horizon),
                    f"gap_block_{prev_end}_{horizon}"
                )
                knitting_intervals.append(gap_iv)
                demands.append(max_block_demand)
                gap_count += 1

        if knitting_intervals:
            _total_block_demand = sum(d for t, d in zip(knitting_intervals, demands) if d > 1)
            logger.info(
                f"📊 AddCumulative: {len(knitting_intervals)} intervals total "
                f"(capacity_block demand sum={_total_block_demand}, "
                f"gap_blocks_added={gap_count}), "
                f"capacity={MAX_FACTORY_MACHINES}"
            )
            self.model.AddCumulative(knitting_intervals, demands, MAX_FACTORY_MACHINES)

    def _build_workforce_boolean(self, MAX_FACTORY_MACHINES: int) -> None:
        """
        Slot-based NoOverlap alternative to AddCumulative.

        Each knitting task is assigned to exactly one of MAX_FACTORY_MACHINES virtual
        machine slots via a BoolVar.  capacity_block tasks occupy their first `demand`
        slots as fixed intervals, preventing knitting tasks on those slots from running
        concurrently with the block.

        RAM trade-off vs. AddCumulative:
          - Saves: no large cumulative propagator with many ghost-task intervals
          - Costs: MAX_FACTORY_MACHINES BoolVars per knitting task
          - Break-even: typically around 200 ghost tasks (hence the auto-trigger)
        """
        n_slots = max(1, int(MAX_FACTORY_MACHINES))
        slot_intervals: List[List] = [[] for _ in range(n_slots)]

        for t_id, tv in self.task_vars.items():
            task_info = next((t for t in self.tasks if t["task_id"] == t_id), {})
            operation = task_info.get("operation", "").lower()

            # Use actual duration: for pinned tasks, end-start may differ from "duration" field
            pinned_start = task_info.get("pinned_start_time")
            pinned_end = task_info.get("pinned_end_time")
            is_fully_pinned = (
                task_info.get("is_pinned") and pinned_start is not None and pinned_end is not None
            )
            duration = int(pinned_end) - int(pinned_start) if is_fully_pinned else int(task_info.get("duration", 0))

            if operation == "capacity_block":
                # Occupy first `demand` slots — same semantics as AddCumulative demand.
                blocked = min(int(task_info.get("demand", 0)), n_slots)
                for s in range(blocked):
                    iv = self.model.NewIntervalVar(
                        tv["start"], duration, tv["end"],
                        f"block_{t_id}_vslot_{s}"
                    )
                    slot_intervals[s].append(iv)

            elif operation == "knitting":
                slot_bools = [
                    self.model.NewBoolVar(f"{t_id}_vslot_{s}")
                    for s in range(n_slots)
                ]
                self.model.AddExactlyOne(slot_bools)
                for s, sb in enumerate(slot_bools):
                    opt_iv = self.model.NewOptionalIntervalVar(
                        tv["start"], duration, tv["end"], sb,
                        f"k_{t_id}_vslot_{s}"
                    )
                    slot_intervals[s].append(opt_iv)

        for s in range(n_slots):
            if slot_intervals[s]:
                self.model.AddNoOverlap(slot_intervals[s])

    def build_resource_allocations(self) -> "TaskModelBuilder":
        resource_usage_literals: Dict[str, List[cp_model.IntVar]] = {
            r_id: [] for r_id in self.resource_map.keys()
        }

        for t in self.tasks:
            t_id = t["task_id"]
            if t_id not in self.task_vars:
                continue 
            
            operation = t.get("operation", "").lower()
            if operation == "capacity_block":
                continue

            task_design = t.get("design_item_id", "")
            task_color = t.get("color_config", "")
            tv = self.task_vars[t_id]
            start_var = tv["start"]
            end_var = tv["end"]
            
            # Nếu ghim, tính toán effective_id thay vì dùng trực tiếp
            if t.get("is_pinned"):
                effective_id = t.get("pinned_machine_id") or (
                    t.get("compatible_resource_ids", [None])[0]
                )
                if not effective_id:
                    logger.warning(f"⚠️ Pinned {t_id}: no machine ID — skipping")
                    self.no_resource_tasks.append(t)
                    continue

                tv["r_ids"] = [effective_id]
                
                # Auto-register: Guard if pinned machine is omitted from resources list
                if effective_id not in self.resource_map:
                    logger.warning(f"⚠️ Auto-registering omitted resource {effective_id} for pinned {t_id}")
                    self.resource_map[effective_id] = {
                        "id": effective_id,
                        "type": "serial",
                        "capacity": 1,
                        "unavailability": [],
                        "available_at_min": 0,
                    }
                    self.resource_map[effective_id].setdefault("intervals", [])
                    # Track literals list for Objective activation constraints later
                    if "resource_usage_literals" in locals(): 
                        pass # Fixed cleanly by tracking within resource_map if needed, 
                             # but we don't activate phantom resources, we just avoid crashing

            literals = []
            actual_r_ids = []

            for r_id in tv["r_ids"]:
                if r_id not in self.resource_map:
                    continue

                # Tạo biến is_selected CHUẨN (Chỉ tạo 1 lần)
                is_selected = self.model.NewBoolVar(f"{t_id}_on_{r_id}")
                literals.append(is_selected)
                actual_r_ids.append(r_id)

                # NẾU TASK BỊ GHIM -> ÉP BIẾN NÀY = 1
                if t.get("is_pinned"):
                    self.model.Add(is_selected == 1)

                available_at = int(self.resource_map[r_id].get("available_at_min", 0))
                # Skip available_at for pinned tasks — start_var is a NewConstant;
                # is_selected is forced to 1, so OnlyEnforceIf has no effect and
                # available_at > pinned_start would make the model INFEASIBLE.
                if available_at > 0 and not t.get("is_pinned"):
                    self.model.Add(start_var >= available_at).OnlyEnforceIf(is_selected)

                if "resource_usage_literals" in locals() and r_id in resource_usage_literals:
                    resource_usage_literals[r_id].append(is_selected)
                # Dành cho resource mới tạo từ auto-register (safe access)
                elif "resource_usage_literals" in locals():
                    resource_usage_literals.setdefault(r_id, []).append(is_selected)

                # OptionalIntervalVar dùng cho AddNoOverlap
                # Cần tính lại duration thực tế nếu đã ghim để tránh lỗi Infeasible
                pinned_start = t.get("pinned_start_time")
                pinned_end = t.get("pinned_end_time")
                is_fully_pinned = t.get("is_pinned") and pinned_start is not None and pinned_end is not None
                actual_duration = int(pinned_end) - int(pinned_start) if is_fully_pinned else int(t["duration"])
                
                opt_interval = self.model.NewOptionalIntervalVar(
                    start_var, actual_duration, end_var, is_selected, f"int_{t_id}_{r_id}"
                )
                self.resource_map[r_id].setdefault("intervals", []).append(opt_interval)

                penalty = self._compute_affinity_penalty(
                    self.resource_map[r_id], task_design, task_color
                )
                if penalty > 0:
                    self.objective_terms.append(is_selected * penalty * 10 * self.lateness_scale)

            if not literals:
                logger.warning(
                    f"⚠️ Task {t_id}: none of {tv['r_ids']} found in resource_map — "
                    "no assignment possible. Marking as unschedulable."
                )
                self.no_resource_tasks.append(t)
                continue

            self.model.AddExactlyOne(literals)
            tv["literals"] = literals
            tv["r_ids"] = actual_r_ids

        # Activation weights scale with lateness_scale to preserve LATENESS : ACTIVATION ≈ 1000 : 2.
        # Labor activation is a pure tie-breaker — keep it zero.
        _w_activate = 2 * self.lateness_scale
        for r_id, assigned_literals in resource_usage_literals.items():
            if not assigned_literals:
                continue
            is_labor = r_id.startswith("W_")

            is_resource_activated = self.model.NewBoolVar(f"activated_{r_id}")
            self.model.AddMaxEquality(is_resource_activated, assigned_literals)

            if not is_labor:
                self.objective_terms.append(is_resource_activated * _w_activate)

        # ------------------------------------------------------------------
        # NEW: CONTIGUOUS ON SAME MACHINE, PARALLEL ACROSS MACHINES
        # ------------------------------------------------------------------
        # 1. Gom nhóm task Knitting theo PO
        po_knitting_groups = {}
        for t in self.tasks:
            if t.get("operation", "").lower() == "knitting":
                po_id = t.get("original_order_id")
                if po_id:
                    po_knitting_groups.setdefault(po_id, []).append(t)

        # 2. Tạo "Bounding Box" cục bộ trên từng máy
        for po_id, tasks_in_po in po_knitting_groups.items():
            if len(tasks_in_po) <= 1:
                continue

            for r_id in self.resource_map.keys():
                task_lits = []
                task_durations = []
                task_starts = []
                task_ends = []
                
                # Lọc các biến (variables) của PO này trên máy r_id
                for t in tasks_in_po:
                    tid = t["task_id"]
                    if tid not in self.task_vars:
                        # Task was skipped (no compatible resources) — do not crash
                        continue
                    tv = self.task_vars[tid]
                    lit = next((l for l in tv["literals"] if l.Name().endswith(f"_on_{r_id}")), None)
                    if lit is not None:
                        task_lits.append(lit)
                        task_durations.append(int(t["duration"]))
                        task_starts.append(tv["start"])
                        task_ends.append(tv["end"])

                if len(task_lits) > 1:
                    # Tạo biến bao trùm cục bộ
                    po_start = self.model.NewIntVar(0, self.horizon, f"po_{po_id}_{r_id}_start")
                    po_end = self.model.NewIntVar(0, self.horizon, f"po_{po_id}_{r_id}_end")
                    po_active = self.model.NewBoolVar(f"po_{po_id}_{r_id}_active")
                    
                    # Nếu máy này nhận ít nhất 1 task của PO -> po_active = 1
                    self.model.AddMaxEquality(po_active, task_lits)
                    
                    # Ép Start/End của các task được chọn không vượt ra ngoài Bounding Box
                    for lit, st, en in zip(task_lits, task_starts, task_ends):
                        self.model.Add(st >= po_start).OnlyEnforceIf(lit)
                        self.model.Add(en <= po_end).OnlyEnforceIf(lit)
                    
                    # Ràng buộc thép: Khoảng cách Box = Tổng thời gian chạy thực tế của các task trên máy này
                    total_dur_expr = sum(lit * dur for lit, dur in zip(task_lits, task_durations))
                    self.model.Add(po_end - po_start == total_dur_expr).OnlyEnforceIf(po_active)
        
        self.build_workforce_constraints()
        return self

    def build_material_constraints(self) -> "TaskModelBuilder":
        """
        Apply per-material Creel capacity constraints via AddCumulative (Profile Sweep).

        For each material declared in self.material_capacities this method:
          1. Collects fixed IntervalVars bound strictly to each task's [start, end].
          2. Collects the integer demand for that material from task.material_demands.
          3. Calls model.AddCumulative(intervals, demands, capacity) — O(n log n) sweep,
             no combinatorial BoolVar explosion, no tracking between task pairs.

        Material is released automatically when the task ends because the interval
        is anchored to tv["end"]. Called after build_resource_allocations() so
        task_vars are fully populated.

        Guardrails (per implementation spec):
          - NO O(N²) transition BoolVars between tasks.
          - ALL demands and capacities are int — no float.
          - Intervals are strictly [task.start, task.end] — material released on finish.
        """
        if not self.material_capacities:
            logger.info("📦 build_material_constraints: material_capacities is empty — skipped")
            return self

        logger.info(
            f"\n📦 MATERIAL CONSTRAINTS: {len(self.material_capacities)} material(s) declared\n"
            + "\n".join(
                f"   capacity['{mat}'] = {cap}"
                for mat, cap in self.material_capacities.items()
            )
        )

        # Step 1: Initialize per-material interval/demand lists
        mat_intervals: Dict[str, List] = {mat: [] for mat in self.material_capacities}
        mat_demands: Dict[str, List[int]] = {mat: [] for mat in self.material_capacities}

        # Step 2: Flatten task demands into per-material lists
        tasks_with_demands = 0
        for t in self.tasks:
            t_id = t["task_id"]
            if t_id not in self.task_vars:
                continue  # No compatible resource — excluded from the model

            task_mat_demands: Dict[str, int] = t.get("material_demands") or {}
            if not task_mat_demands:
                continue

            tasks_with_demands += 1
            tv = self.task_vars[t_id]

            # Compute actual duration — pinned tasks use their exact pinned window
            pinned_start = t.get("pinned_start_time")
            pinned_end = t.get("pinned_end_time")
            is_fully_pinned = (
                t.get("is_pinned") and pinned_start is not None and pinned_end is not None
            )
            duration: int = (
                int(pinned_end) - int(pinned_start)
                if is_fully_pinned
                else int(t.get("duration", 0))
            )

            if duration <= 0:
                logger.warning(
                    f"   ⚠️ Task {t_id}: non-positive duration ({duration}) — "
                    "skipping material interval"
                )
                continue

            for mat_code, demand in task_mat_demands.items():
                if mat_code not in mat_intervals:
                    logger.warning(
                        f"   ⚠️ Task {t_id}: material '{mat_code}' not in material_capacities "
                        f"(declared: {list(self.material_capacities)}) — skipping"
                    )
                    continue

                demand_int = int(demand)
                if demand_int <= 0:
                    continue

                # Strictly bound to this task's start/end — material released when task finishes
                interval = self.model.NewIntervalVar(
                    tv["start"], duration, tv["end"],
                    f"mat_{mat_code}_{t_id}",
                )
                mat_intervals[mat_code].append(interval)
                mat_demands[mat_code].append(demand_int)
                logger.info(
                    f"   📌 {t_id} → '{mat_code}': demand={demand_int}, duration={duration}"
                )

        logger.info(
            f"   📊 Tasks with material_demands: {tasks_with_demands}/{len(self.tasks)} total tasks"
        )

        # Step 3: Apply AddCumulative per material
        for mat_code, capacity in self.material_capacities.items():
            intervals = mat_intervals.get(mat_code, [])
            demands = mat_demands.get(mat_code, [])

            if not intervals:
                logger.warning(
                    f"   ⚠️ '{mat_code}': capacity declared ({capacity}) but NO task has "
                    "material_demands for this material — AddCumulative skipped. "
                    "Check that task.material_demands keys match material_capacities keys."
                )
                continue

            capacity_int = int(capacity)
            self.model.AddCumulative(intervals, demands, capacity_int)
            logger.info(
                f"   ✅ '{mat_code}': AddCumulative applied "
                f"({len(intervals)} tasks, capacity={capacity_int})"
            )

        return self

    def apply_routing_constraints(self) -> "TaskModelBuilder":
        """
        Enforce that no two tasks overlap on the same resource, and that tasks
        cannot be placed inside declared unavailability windows.
        """
        for r_id, resource in self.resource_map.items():
            intervals = resource.get("intervals", [])
            for window in resource.get("unavailability", []):
                w_start, w_end = int(window["start"]), int(window["end"])
                if w_end > w_start:
                    unavail = self.model.NewFixedSizeIntervalVar(
                        w_start, w_end - w_start, f"unavail_{r_id}"
                    )
                    intervals.append(unavail)
            if intervals:
                self.model.AddNoOverlap(intervals)

        return self

    def apply_dependency_constraints(self) -> "TaskModelBuilder":
        """
        Apply two kinds of ordering constraints:
        1. Explicit: task.start >= parent.end  (from final_depends_on)
        2. Inferred: Linking tasks must wait for all Knitting batches of the same
           order to finish — used as a fallback when Go hasn't yet set
           wait_for_batch_task_id on older payload versions.

        Flow continuity: for each A→B dependency where B is not a washing
        operation, a gap penalty (start_B - end_A) is added to the objective so
        the solver schedules B immediately after A instead of drifting lazily.
        Weight = priority_weight * lateness_scale (1000× lighter than lateness
        per minute, so gaps are closed when cheap but never override urgency).
        """
        _gap_ctr: List[int] = [0]  # mutable counter for unique CP-SAT var names

        def _add_gap_penalty(parent_id: str, child_id: str) -> None:
            """Add a flow-continuity gap penalty for the parent→child edge."""
            child_info = next((t for t in self.tasks if t["task_id"] == child_id), {})
            # Washing tasks intentionally batch up before starting — no gap pressure.
            if child_info.get("operation", "").lower() == "washing":
                return
            # Pinned tasks can't move; adding a penalty is pointless.
            if child_info.get("is_pinned", False):
                return
            priority = int(child_info.get("priority", 5))
            # Gap weight is 1/10 of early-start weight so that when L has
            # multiple K predecessors the solver never delays an early-finishing
            # K just to shrink its gap (perverse incentive).  The early-start
            # term on each K already handles unnecessary idle time.
            _gap_w = max(1, (10 ** (6 - priority)) * self.lateness_scale // 10)
            _gap_ctr[0] += 1
            gap_var = self.model.NewIntVar(0, self.horizon, f"gap_{_gap_ctr[0]}")
            self.model.Add(
                gap_var >= self.task_vars[child_id]["start"] - self.task_vars[parent_id]["end"]
            )
            self.objective_terms.append(gap_var * _gap_w)
            logger.info(f"   ⏩ GAP: {child_id} should follow {parent_id} immediately (w={_gap_w})")

        def _resolve_dependency(upstream_id: str, downstream_id: str) -> None:
            """
            Add the standard end-dependency: downstream must wait for upstream
            to fully complete before it can start.

            Linking requires full sub-batch output (material is only ready when
            the knitting run finishes), so no mid-batch pipelining is allowed.
            """
            self.model.Add(
                self.task_vars[downstream_id]["start"] >= self.task_vars[upstream_id]["end"]
            )
            _add_gap_penalty(upstream_id, downstream_id)

        # 1. Explicit final_depends_on
        logger.info("\n📋 APPLYING DEPENDENCY CONSTRAINTS:")
        for t_id, tv in self.task_vars.items():
            for parent_id in tv["depends_on"]:
                # Direct-lookup preference: if the dependency ID is already a
                # schedulable unit (in task_vars), use it as-is so Go can send
                # granular sub-batch IDs (e.g. K1_b1) and bypass batch aggregation.
                # Fall back to the translation map only when the direct ID is absent.
                if parent_id in self.task_vars:
                    actual = parent_id
                else:
                    actual = self.task_translation_map.get(parent_id, parent_id)

                if actual in self.task_vars:
                    logger.info(f"   ✅ DEP: {t_id} ← {actual} (raw: '{parent_id}')")
                    _resolve_dependency(actual, t_id)
                else:
                    logger.warning(
                        f"⚠️ Task '{t_id}' depends on '{parent_id}' "
                        f"(resolved: '{actual}') — not found in task_vars!"
                    )

        # 2. Inferred K→L fallback
        # Collect task IDs that are already handled by batch-offset constraints
        # so we don't add a redundant (and potentially weaker) end-dependency.
        tasks_with_offset: Set[str] = {
            t["task_id"]
            for t in self.tasks
            if t.get("wait_offsets") or t.get("WaitOffsets") or t.get("wait_for_batch_task_id") or t.get("WaitForBatchTaskID")
        }
        print(f"\n🔍 Tasks with batch offsets (will skip inferred K→L constraints): {tasks_with_offset}")
        for l_id, l_tv in self.task_vars.items():
            if l_id in tasks_with_offset:
                continue
            m_l = re.match(r"^L\d+-(.+)$", l_id)
            if not m_l:
                continue
            l_base = re.sub(r"(?:_b\d+)?(?:_SLICE_\d+)?$", "", m_l.group(1))

            k_batch_ids: Set[str] = set()
            for key, batch_id in self.task_translation_map.items():
                if key == batch_id:
                    continue
                km = re.match(r"^K\d+-(.+?)(?:_SLICE_\d+)?$", key)
                if not km:
                    continue
                k_base = re.sub(r"_b\d+$", "", km.group(1))
                if l_base == k_base and batch_id in self.task_vars:
                    k_batch_ids.add(batch_id)

            for k_batch_id in sorted(k_batch_ids):
                logger.info(f"   🔗 INFERRED: {l_id} ← {k_batch_id}")
                _resolve_dependency(k_batch_id, l_id)

            if not k_batch_ids:
                logger.warning(f"   ⚠️ No K batch found for L task '{l_id}' (base: '{l_base}')")

        return self

    def apply_batch_offset_constraints(self) -> "TaskModelBuilder":
        """
        Apply pipelining constraints: a downstream slice (e.g. Linking) can start
        once the upstream batch (e.g. Knitting) has advanced past a known offset.
        Now supports multiple batch dependencies via the 'wait_offsets' dictionary.

        Go backend provides:
            WaitOffsets — dictionary mapping BatchTaskID -> offset in minutes

        Overload-adaptive mode:
            When factory load > 85 %, hard offsets risk making the model infeasible
            because the solver cannot find a slot that simultaneously satisfies both
            the machine capacity and the pipeline window.  In that case we relax to
            soft constraints: a penalty proportional to the pipeline violation is
            added to the objective (weight = P5_lateness / 10) so the solver can
            slide the linking start slightly without being declared infeasible.
        """
        # Compute factory load once to decide hard vs. soft mode.
        _total_knitting = sum(
            int(t.get("duration", 0))
            for t in self.tasks
            if t.get("operation", "").lower() == "knitting"
        )
        _cap = int(self.config.get("max_factory_machines", 100)) * self.horizon
        _is_overloaded = (_total_knitting / max(1, _cap)) > 0.85

        # PIPELINE_VIOLATION_WEIGHT ≈ P5 lateness weight / 10 (less severe than real lateness)
        _pipeline_w = 100 * self.lateness_scale

        mode_label = "SOFT (factory overloaded)" if _is_overloaded else "HARD"
        logger.info(f"\n⏱  BATCH OFFSET CONSTRAINTS (WaitOffsets) — {mode_label}:")

        for t in self.tasks:
            t_id = t["task_id"]
            if t_id not in self.task_vars:
                continue

            wait_offsets = t.get("wait_offsets") or t.get("WaitOffsets") or {}

            if not wait_offsets:
                continue

            for raw_batch_id, offset in wait_offsets.items():
                actual_batch = self.task_translation_map.get(raw_batch_id, raw_batch_id)

                if actual_batch not in self.task_vars:
                    logger.warning(
                        f"   ⚠️ '{t_id}': wait_for_batch '{raw_batch_id}' "
                        f"(resolved: '{actual_batch}') not found — skipping."
                    )
                    continue

                offset_val = int(offset)

                if _is_overloaded:
                    # Soft: penalise pipeline violations but never block the solver.
                    # violation = max(0, (batch.start + offset) - downstream.start)
                    viol = self.model.NewIntVar(
                        0, self.horizon, f"pipe_viol_{t_id}_{actual_batch}"
                    )
                    self.model.Add(
                        viol >= self.task_vars[actual_batch]["start"]
                               + offset_val
                               - self.task_vars[t_id]["start"]
                    )
                    self.objective_terms.append(viol * _pipeline_w)
                    logger.info(
                        f"   ⚠️  SOFT {t_id} ← {actual_batch} +{offset_val}"
                    )
                else:
                    # Hard: strict pipeline dependency.
                    self.model.Add(
                        self.task_vars[t_id]["start"]
                        >= self.task_vars[actual_batch]["start"] + offset_val
                    )
                    logger.info(f"   ⏱  {t_id} waits for {actual_batch} at offset +{offset_val}")

        return self

    def define_objective(self) -> "TaskModelBuilder":
        """
        Minimise the weighted sum of:
        - Lateness penalties (higher weight for higher-priority tasks)
        - Early-start preference (secondary tie-breaker)
        - Affinity penalties (prefer machines already set up for the same design/color)
        """
        self.model.Minimize(sum(self.objective_terms))
        return self

    # ------------------------------------------------------------------
    # Post-solve diagnostics
    # ------------------------------------------------------------------

    def _classify_root_cause(
        self,
        t_id: str,
        selected_res: str,
        start_val: int,
        solver: cp_model.CpSolver,
    ) -> str:
        """
        Heuristic post-solve classifier: why did this task end up LATE?

        Priority order (first match wins):
          1. PINNED_TASK_CONFLICT  — a locked task on the same machine displaced us
          2. WORKFORCE_SHORTAGE    — global factory cap was saturated at our deadline
          3. MACHINE_OVERLOAD      — at least one other task competed for this machine
          4. CAPACITY_FULL         — fallback / unclassified

        Called only after a FEASIBLE/OPTIMAL solve, so solver.Value() is safe.
        O(n) per late task — acceptable at MVP scale (< 1000 tasks).
        """
        # The "desired" start is the earliest the task was allowed to begin.
        # Using this (rather than start_val) lets us check what blocked an
        # on-time start, not just what was concurrent at the actual start.
        task_info = next((t for t in self.tasks if t["task_id"] == t_id), {})
        desired_start = int(task_info.get("start_after_min", 0))

        # ── 1. PINNED_TASK_CONFLICT ────────────────────────────────────────
        # A locked task started before us AND was still running at our earliest
        # possible start, proving it displaced us into a late slot.
        for t in self.tasks:
            if (
                t.get("is_pinned")
                and t.get("pinned_machine_id") == selected_res
                and t["task_id"] != t_id
            ):
                ps = t.get("pinned_start_time") or 0
                pe = t.get("pinned_end_time") or 0
                if ps < start_val and pe > desired_start:
                    return "PINNED_TASK_CONFLICT"

        # ── 2. WORKFORCE_SHORTAGE ─────────────────────────────────────────
        # Check whether the factory was already at cap at our desired start time.
        # If so, we couldn't have started on time even on a free machine.
        config_max = int(self.config.get("max_factory_machines", 100))
        concurrent_knitting = 0
        blocked_demand = 0

        for other_id, other_tv in self.task_vars.items():
            if other_id == t_id:
                continue
            other_start = solver.Value(other_tv["start"])
            other_end   = solver.Value(other_tv["end"])
            if other_start <= desired_start < other_end:   # running at our desired start
                other_info = next(
                    (t for t in self.tasks if t["task_id"] == other_id), {}
                )
                op = other_info.get("operation", "").lower()
                if op == "knitting":
                    concurrent_knitting += 1
                elif op == "capacity_block":
                    blocked_demand += int(other_info.get("demand", 0))

        if (concurrent_knitting + blocked_demand) >= config_max:
            return "WORKFORCE_SHORTAGE"

        # ── 3. MACHINE_OVERLOAD ───────────────────────────────────────────
        # Another task ran on this machine and finished at or before our start —
        # it directly occupied the slot before us, pushing our start past due_at.
        for other_id, other_tv in self.task_vars.items():
            if other_id == t_id:
                continue
            if not other_tv.get("literals"):
                continue
            other_on_res = any(
                solver.Value(lit) == 1
                for lit in other_tv["literals"]
                if lit.Name().endswith(f"_on_{selected_res}")
            )
            if other_on_res and solver.Value(other_tv["end"]) <= start_val:
                return "MACHINE_OVERLOAD"

        return "CAPACITY_FULL"

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def extract_results(
        self,
        solver: cp_model.CpSolver,
        status: int,
    ) -> Dict[str, Any]:
        """Extract assignments and overloads from the solved model."""
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.warning("❌ Infeasible solution")
            return {
                "status": "infeasible",
                "assignments": [],
                "overloads": [],
                "objective_value": None,
                "solve_time_seconds": solver.WallTime(),
            }

        logger.info(f"✅ Feasible! Objective value: {solver.ObjectiveValue()}")
        assignments = []
        overloads = []

        # ── POST-SOLVE CONCURRENCY DIAGNOSTIC ───────────────────────────────
        # Scan all knitting tasks grouped by their start time.
        # Flags any time point where concurrent active knitting tasks + blocked slots > max.
        _config_max = int(self.config.get("max_factory_machines", 100))
        _kt_times: Dict[str, tuple] = {}  # t_id -> (start, end)
        _block_times: list = []           # (start, end, demand)
        for t_id_d, tv_d in self.task_vars.items():
            task_info_d = next((t for t in self.tasks if t["task_id"] == t_id_d), {})
            op_d = task_info_d.get("operation", "").lower()
            s_d, e_d = solver.Value(tv_d["start"]), solver.Value(tv_d["end"])
            if op_d == "knitting":
                _kt_times[t_id_d] = (s_d, e_d)
            elif op_d == "capacity_block":
                _block_times.append((s_d, e_d, int(task_info_d.get("demand", 0))))
        # Check unique start times of knitting tasks
        _checked = set()
        for t_id_d, (s_d, e_d) in _kt_times.items():
            if s_d in _checked:
                continue
            _checked.add(s_d)
            concurrent = sum(1 for (s2, e2) in _kt_times.values() if s2 <= s_d < e2)
            blocked = sum(dem for (bs, be, dem) in _block_times if bs <= s_d < be)
            if concurrent + blocked > _config_max:
                logger.warning(
                    f"⚠️ CAPACITY VIOLATION at t={s_d}: {concurrent} knitting tasks + "
                    f"{blocked} blocked demand = {concurrent+blocked} > {_config_max} max"
                )

        for t_id, tv in self.task_vars.items():
            start_val = solver.Value(tv["start"])
            end_val = solver.Value(tv["end"])

            selected_res = None
            for i, lit in enumerate(tv["literals"]):
                if solver.Value(lit) == 1:
                    selected_res = tv["r_ids"][i]
                    break

            if selected_res:
                is_late = end_val > tv["due"]
                assignments.append({
                    "task_id": t_id,
                    "machine_id": selected_res,
                    "start_time": start_val,
                    "end_time": end_val,
                    "group_id": tv.get("group_id", ""),
                    "order_id": tv.get("original_order_id", ""),
                    "quantity": tv.get("qty", 0),
                    "status": "LATE" if is_late else "ON_TIME",
                })
                if is_late:
                    overloads.append({
                        "task_id": t_id,
                        "order_id": tv.get("original_order_id", ""),
                        "status": "LATE",
                        "delay_minutes": end_val - tv["due"],
                        "root_cause_code": self._classify_root_cause(
                            t_id, selected_res, start_val, solver
                        ),
                        "bottleneck_resource_id": selected_res,
                        "quantity": tv.get("qty", 0),
                    })

        return {
            "status": "feasible",
            "assignments": assignments,
            "overloads": overloads,
            "objective_value": solver.ObjectiveValue(),
            "solve_time_seconds": solver.WallTime(),
        }
