from ortools.sat.python import cp_model
from typing import Dict, List, Any
import sys

# Buffer time (phút) an toàn giữa các task phụ thuộc
BUFFER_TIME = 0 
# Phạt nặng nếu bỏ qua task (để Solver cố gắng xếp bằng được)
DROP_PENALTY = 1000000 

class Engine:
    def __init__(self, payload: Dict[str, Any]):
        self.config = payload.get("config", {})
        self.resources = payload.get("resources", []) 
        self.tasks = payload.get("tasks", [])
        
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        print(f"Engine initialized with {len(self.tasks)} tasks and {len(self.resources)} resources.")

    def solve(self) -> Dict[str, Any]:
        if not self.tasks:
            return {"status": "feasible", "assignments": []}

        # --- STEP 0: DIAGNOSE INPUT DATA (Tìm nguyên nhân Infeasible trước) ---
        self._diagnose_input_issues()

        print("\n🕵️ DIAGNOSING DROPPED BATCHES:")
        resource_map = {r["id"]: r for r in self.resources}
        
        # Chỉ check các task có trong danh sách bị drop
        target_ids = ['BATCH_0-646_1', 'BATCH_0-646_2'] # ID bạn thấy trong log

        for t in self.tasks:
            if t.get("task_id") not in target_ids: continue
            
            print(f"\n>> ANALYZING {t.get('task_id')}:")
            
            # 1. Check Duration
            duration = int(t.get("duration") or 0)
            print(f"   - Duration Required: {duration} mins")
            
            # 2. Check Resources
            comp_res = t.get("compatible_resource_ids") or []
            print(f"   - Compatible Resources: {comp_res}")
            
            if not comp_res:
                print("   ❌ ERROR: No compatible resources found! Check Mapping Logic.")
                continue

            # 3. Check Slot trên từng máy
            for r_id in comp_res:
                if r_id not in resource_map: continue
                res = resource_map[r_id]
                windows = res.get("unavailability", [])
                
                # Tính Max Gap
                sorted_windows = sorted(windows, key=lambda x: int(x['start']))
                current_time = 0
                max_gap = 0
                
                print(f"   - Machine {r_id} Breaks:")
                for w in sorted_windows:
                    start, end = int(w['start']), int(w['end'])
                    gap = start - current_time
                    if gap > max_gap: max_gap = gap
                    print(f"     [{current_time} -> {start}] (Gap: {gap}m) | Break: {start}->{end}")
                    current_time = end
                
                # Check đoạn cuối
                gap = 100000 - current_time
                if gap > max_gap: max_gap = gap
                
                print(f"   => Max Continuous Slot on {r_id}: {max_gap} mins")
                
                if duration > max_gap:
                    print(f"   ❌ FAIL: Task duration ({duration}) > Max Slot ({max_gap})")
                else:
                    print(f"   ✅ PASS: Task fits in slot!")
        # ---------------------------------------------------------------------

        # --- STEP 1: BUILD MODEL ---
        task_vars = self._build_model()
        
        # --- STEP 2: CONFIGURE SOLVER ---
        max_time = int(self.config.get("max_search_time", 60))
        self.solver.parameters.max_time_in_seconds = max_time
        self.solver.parameters.log_search_progress = True 
        # self.solver.parameters.linearization_level = 2 # Uncomment để debug sâu hơn nếu cần

        print("🚀 Solving...")
        status = self.solver.Solve(self.model)
        print(f"🏁 Solver Status: {self.solver.StatusName(status)}")

        # --- STEP 3: RESULT ---
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self._analyze_overlaps(task_vars)
            assignments, overloads = self._extract_solution(task_vars)
            return {
                "status": "feasible",
                "objective_value": int(self.solver.ObjectiveValue()),
                "assignments": assignments,
                "overloads": overloads
            }
        else:
            print("❌ MODEL IS INFEASIBLE. Please check the Diagnosis logs above.")
            return {
                "status": "infeasible",
                "assignments": [],
                "overloads": []
            }

    def _build_model(self):
        task_vars = {}
        
        resource_map = {r["id"]: r for r in self.resources}
        resource_intervals = {r["id"]: [] for r in self.resources}
        resource_demands = {r["id"]: [] for r in self.resources}
        
        # Tự động tính Horizon nếu config quá bé
        total_duration = sum(int(t.get("duration") or 0) for t in self.tasks)
        config_horizon = int(self.config.get("horizon_minutes", 100000))
        horizon = max(config_horizon, total_duration + 10080) # Min là 1 tuần dư ra

        # ---------------------------------------------------------
        # 1. BUILD UNAVAILABILITY INTERVALS (DUMMY TASKS)
        # ---------------------------------------------------------
        resource_unavail_intervals = {}
        for r in self.resources:
            r_id = r["id"]
            unavail_list = []
            
            for window in r.get("unavailability", []):
                start = int(window["start"]) # Force int
                end = int(window["end"])     # Force int
                size = end - start
                if size > 0:
                    # Tạo Fixed Interval chặn giờ nghỉ
                    ival = self.model.NewFixedSizeIntervalVar(start, size, f"Unavail_{r_id}_{start}")
                    unavail_list.append(ival)
            
            resource_unavail_intervals[r_id] = unavail_list

        # ---------------------------------------------------------
        # 2. BUILD TASKS VARIABLES
        # ---------------------------------------------------------
        for t in self.tasks:
            t_id = t.get("task_id") or t.get("TaskID")
            if not t_id: continue

            start_var = self.model.NewIntVar(0, horizon, f"{t_id}_start")
            end_var = self.model.NewIntVar(0, horizon, f"{t_id}_end")
            
            # Start After Constraint
            start_after = int(t.get("start_after_min") or t.get("StartAfterMin") or 0)
            if start_after > 0:
                self.model.Add(start_var >= start_after)

            literals = []
            r_ids = []
            
            # [SOFT CONSTRAINT] Biến cho phép Drop task nếu không thể xếp lịch
            is_dropped = self.model.NewBoolVar(f"{t_id}_dropped")
            
            compatible_ids = t.get("compatible_resource_ids") or t.get("CompatibleResourceIDs") or []
            
            for r_id in compatible_ids:
                if r_id not in resource_map: continue
                
                is_selected = self.model.NewBoolVar(f"{t_id}_on_{r_id}")
                literals.append(is_selected)
                r_ids.append(r_id)

                # [FIX] Force int cho duration
                duration = int(t.get("duration") or t.get("Duration") or 0)
                
                # Interval Optional: Chỉ active nếu chọn máy này
                interval = self.model.NewOptionalIntervalVar(
                    start_var, duration, end_var, is_selected, f"Int_{t_id}_{r_id}"
                )
                resource_intervals[r_id].append(interval)
                
                # Logic Demand
                is_batch = t.get("is_batch") or t.get("IsBatch")
                qty_raw = t.get("qty") or t.get("Qty") or 1
                demand = int(qty_raw) if is_batch else 1
                resource_demands[r_id].append(demand)

            # [QUAN TRỌNG] Ràng buộc chọn máy: Chọn 1 máy HOẶC bị Drop
            if literals:
                self.model.AddExactlyOne(literals + [is_dropped])
            else:
                # Không có máy nào tương thích -> Buộc phải Drop
                self.model.Add(is_dropped == 1)

            # Store info
            prio_val = int(t.get("priority") or t.get("Priority") or 3)
            due_val = int(t.get("due_at_min") or t.get("DueAtMin") or horizon)

            task_vars[t_id] = {
                "start": start_var,
                "end": end_var,
                "literals": literals,
                "is_dropped": is_dropped, # Lưu biến drop
                "r_ids": r_ids,
                "due": due_val,      
                "priority": prio_val, 
                
                "depends_on": t.get("final_depends_on") or t.get("FinalDependsOn") or [],
                "internal_dep": t.get("internal_dep") or t.get("InternalDep"),
                
                "original_order_id": t.get("original_order_id") or t.get("OriginalOrderID"),
                "sub_task_completion_offsets": t.get("sub_task_completion_offsets") or t.get("SubTaskCompletionOffsets") or {}
            }

        # ---------------------------------------------------------
        # 3. APPLY RESOURCE CONSTRAINTS
        # ---------------------------------------------------------
        for r_id, r in resource_map.items():
            intervals = resource_intervals[r_id]
            unavail_intervals = resource_unavail_intervals.get(r_id, [])
            all_intervals = intervals + unavail_intervals
            
            if not all_intervals: continue
            
            if r.get("type") == "batch" or r.get("operation") == "washing":
                # Cumulative Constraint (Máy giặt)
                machine_capacity = int(r.get("capacity", 100))
                task_demands = resource_demands[r_id]
                
                # Giờ nghỉ phải chiếm toàn bộ capacity để chặn
                unavail_demands = [machine_capacity] * len(unavail_intervals)
                
                self.model.AddCumulative(
                    all_intervals, 
                    task_demands + unavail_demands, 
                    machine_capacity
                )
            else:
                # No Overlap Constraint (Máy thường)
                self.model.AddNoOverlap(all_intervals)

        # ---------------------------------------------------------
        # 4. APPLY DEPENDENCY CONSTRAINTS
        # ---------------------------------------------------------
        for t_id, tv in task_vars.items():
            # Chỉ áp dụng dependency nếu task KHÔNG bị drop (Enforce if not dropped)
            # Tuy nhiên trong CP-SAT, biến start/end của optional interval bị disable không xác định
            # Nên ta chỉ add constraint đơn giản, nếu drop thì constraint vẫn đúng vì start/end tự do
            
            # A. General Dependencies
            for parent_id in tv["depends_on"]:
                if parent_id not in task_vars: continue
                parent_tv = task_vars[parent_id]
                
                offsets = parent_tv["sub_task_completion_offsets"]
                child_order_id = tv["original_order_id"]

                # Logic Interleaved Batching
                if offsets and child_order_id and child_order_id in offsets:
                    lag_minutes = int(offsets[child_order_id]) 
                    self.model.Add(
                        tv["start"] >= parent_tv["start"] + lag_minutes + BUFFER_TIME
                    )
                else:
                    self.model.Add(
                        tv["start"] >= parent_tv["end"] + BUFFER_TIME
                    )

            # B. Internal Slice Dependencies
            prev_id = tv["internal_dep"]
            if prev_id and prev_id in task_vars:
                prev_tv = task_vars[prev_id]
                self.model.Add(tv["start"] >= prev_tv["end"])
                
                # Ràng buộc slice cùng task phải cùng máy (nếu không bị drop)
                for i, lit in enumerate(tv["literals"]):
                    r_id = tv["r_ids"][i]
                    if r_id in prev_tv["r_ids"]:
                        idx_prev = prev_tv["r_ids"].index(r_id)
                        # lit == prev_lit
                        self.model.Add(lit == prev_tv["literals"][idx_prev])

        # ---------------------------------------------------------
        # 5. OBJECTIVE FUNCTION
        # ---------------------------------------------------------
        makespan = self.model.NewIntVar(0, horizon, "makespan")
        objective_terms = []
        
        # A. Penalty for Dropped Tasks (Ưu tiên cao nhất: Hạn chế Drop)
        for _, tv in task_vars.items():
            objective_terms.append(tv["is_dropped"] * DROP_PENALTY)
        
        # B. Minimize Makespan
        for _, tv in task_vars.items():
            self.model.Add(makespan >= tv["end"])
        objective_terms.append(makespan * 100)
        
        # C. Minimize Lateness
        for _, tv in task_vars.items():
            if tv["due"] < horizon:
                delay = self.model.NewIntVar(0, horizon, f"delay")
                self.model.Add(delay >= tv["end"] - tv["due"])
                
                prio = tv["priority"]
                weight = int((6 - prio) * 1000) 
                objective_terms.append(delay * weight)
            
        self.model.Minimize(sum(objective_terms))
        return task_vars

    def _extract_solution(self, task_vars):
        assignments = []
        overloads = []

        for t_id, tv in task_vars.items():
            # 1. Check if Task is Dropped
            if self.solver.Value(tv["is_dropped"]) == 1:
                overloads.append({
                    "task_id": t_id,
                    "order_id": tv.get("original_order_id", ""),
                    "status": "DROPPED",
                    "delay_minutes": 0,
                    "root_cause_code": self._determine_drop_cause(t_id, tv),
                    "bottleneck_resource_id": ""
                })
                continue

            # 2. Task is successfully assigned
            start_val = self.solver.Value(tv["start"])
            end_val = self.solver.Value(tv["end"])
            
            selected_res = None
            for i, lit in enumerate(tv["literals"]):
                if self.solver.Value(lit) == 1:
                    selected_res = tv["r_ids"][i]
                    break
            
            if selected_res:
                assignments.append({
                    "task_id": t_id,
                    "machine_id": selected_res,
                    "start_min": start_val,
                    "end_min": end_val,
                    "order_id": tv.get("original_order_id", "")
                })

                # Check if task is LATE
                due_min = tv.get("due", 0)
                if due_min > 0 and end_val > due_min:
                    overloads.append({
                        "task_id": t_id,
                        "order_id": tv.get("original_order_id", ""),
                        "status": "LATE",
                        "delay_minutes": end_val - due_min,
                        "root_cause_code": "CAPACITY_FULL",
                        "bottleneck_resource_id": selected_res
                    })
        
        if overloads:
            dropped_count = sum(1 for o in overloads if o["status"] == "DROPPED")
            late_count = sum(1 for o in overloads if o["status"] == "LATE")
            print(f"\n⚠️  OVERLOAD SUMMARY: {dropped_count} DROPPED, {late_count} LATE")
            if dropped_count > 0:
                dropped_ids = [o["task_id"] for o in overloads if o["status"] == "DROPPED"][:10]
                print(f"    -> Dropped: {dropped_ids} ...")

        return assignments, overloads

    def _determine_drop_cause(self, t_id: str, tv: dict) -> str:
        """Determine why a task was dropped"""
        if not tv.get("literals"):
            return "NO_COMPATIBLE_RESOURCE"
        return "SLOT_TOO_SMALL_OR_CAPACITY_FULL"

    def _diagnose_input_issues(self):
        """Kiểm tra sơ bộ xem có task nào bất khả thi ngay từ đầu không"""
        print("\n--- DIAGNOSING INPUT DATA ---")
        resource_map = {r["id"]: r for r in self.resources}
        
        issues_found = False
        for t in self.tasks:
            duration = int(t.get("duration") or 0)
            compatible_ids = t.get("compatible_resource_ids") or []
            
            if not compatible_ids:
                print(f"❌ Task '{t.get('task_id')}' has NO compatible resources!")
                issues_found = True
                continue

            max_slot_found = 0
            # Tìm khoảng trống lớn nhất trên các máy tương thích
            for r_id in compatible_ids:
                if r_id not in resource_map: continue
                res = resource_map[r_id]
                windows = res.get("unavailability", [])
                
                # Tính max gap
                current_time = 0
                local_max = 0
                sorted_windows = sorted(windows, key=lambda x: int(x['start']))
                
                for w in sorted_windows:
                    w_start = int(w['start'])
                    gap = w_start - current_time
                    if gap > local_max: local_max = gap
                    current_time = int(w['end'])
                
                # Check đoạn cuối đến vô cực (horizon giả định 1 tuần)
                gap = 10080 - current_time 
                if gap > local_max: local_max = gap
                
                if local_max > max_slot_found: max_slot_found = local_max
            
            if duration > max_slot_found:
                print(f"⚠️  Task '{t.get('task_id')}' duration ({duration}m) > Max Slot ({max_slot_found}m). It will likely be DROPPED.")
                issues_found = True
        
        if not issues_found:
            print("✅ Input diagnosis passed. No obvious impossible tasks.")
        print("-----------------------------\n")

    def _analyze_overlaps(self, task_vars):
        resource_map = {r["id"]: r for r in self.resources}
        for t_id, tv in task_vars.items():
            if self.solver.Value(tv["is_dropped"]) == 1: continue

            start = self.solver.Value(tv["start"])
            end = self.solver.Value(tv["end"])
            
            selected_res = None
            for i, lit in enumerate(tv["literals"]):
                if self.solver.Value(lit) == 1:
                    selected_res = tv["r_ids"][i]
                    break
            
            if selected_res:
                windows = resource_map[selected_res].get("unavailability", [])
                for w in windows:
                    w_start = int(w["start"])
                    w_end = int(w["end"])
                    if max(start, w_start) < min(end, w_end):
                        print(f"❌ LOGIC ERROR: Task {t_id} overlaps break on {selected_res}")