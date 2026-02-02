from ortools.sat.python import cp_model
from typing import Dict, List, Any

BUFFER_TIME = 10 

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

        # --- VALIDATE INPUT ---
        # Kiểm tra xem resources có mang theo unavailability không
        print("\n--- CHECKING RESOURCE UNAVAILABILITY ---")
        count_unavail = 0
        for r in self.resources:
            windows = r.get("unavailability", [])
            if windows:
                count_unavail += 1
                print(f"Resource [{r['id']}] has {len(windows)} break windows. First: {windows[0]}")
        
        if count_unavail == 0:
            print("⚠️ WARNING: No unavailability windows found in payload! Shifts will be IGNORED.")
        # ----------------------

        task_vars = self._build_model()
        
        # ... (Giữ nguyên phần validate duration = 0) ...

        max_time = self.config.get("max_search_time", 60)
        self.solver.parameters.max_time_in_seconds = max_time
        self.solver.parameters.log_search_progress = True

        status = self.solver.Solve(self.model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Analyze solution
            self._analyze_overlaps(task_vars) # [NEW] Hàm check lấn giờ
            
            return {
                "status": "feasible",
                "assignments": self._extract_solution(task_vars)
            }
        else:
            return {
                "status": "infeasible",
                "assignments": []
            }

    def _build_model(self):
        task_vars = {}
        
        resource_map = {r["id"]: r for r in self.resources}
        resource_intervals = {r["id"]: [] for r in self.resources}
        resource_demands = {r["id"]: [] for r in self.resources}
        
        horizon = self.config.get("horizon_minutes", 100000)

        # 1. BUILD UNAVAILABILITY INTERVALS (THE "DUMMY TASKS")
        resource_unavail_intervals = {}
        for r in self.resources:
            r_id = r["id"]
            unavail_list = []
            
            for window in r.get("unavailability", []):
                start = int(window["start"]) # Force int
                end = int(window["end"])     # Force int
                size = end - start
                if size > 0:
                    # Tạo Fixed Interval (Đây chính là Dummy Task "cứng")
                    ival = self.model.NewFixedSizeIntervalVar(start, size, f"Unavail_{r_id}_{start}")
                    unavail_list.append(ival)
            
            resource_unavail_intervals[r_id] = unavail_list

        # 2. BUILD TASKS
        for t in self.tasks:
            # ... (Giữ nguyên logic tạo task variables) ...
            t_id = t.get("task_id") or t.get("TaskID")
            if not t_id: continue

            start_var = self.model.NewIntVar(0, horizon, f"{t_id}_start")
            end_var = self.model.NewIntVar(0, horizon, f"{t_id}_end")
            
            start_after = t.get("start_after_min") or t.get("StartAfterMin") or 0
            if start_after > 0:
                self.model.Add(start_var >= start_after)

            literals = []
            r_ids = []
            
            compatible_ids = t.get("compatible_resource_ids") or t.get("CompatibleResourceIDs") or []
            
            for r_id in compatible_ids:
                if r_id not in resource_map:
                    continue
                is_selected = self.model.NewBoolVar(f"{t_id}_on_{r_id}")
                literals.append(is_selected)
                r_ids.append(r_id)

                duration = t.get("duration") or t.get("Duration") or 0
                
                interval = self.model.NewOptionalIntervalVar(
                    start_var, duration, end_var, is_selected, f"Int_{t_id}_{r_id}"
                )
                resource_intervals[r_id].append(interval)
                
                # Logic Demand
                demand = t.get("qty") if t.get("is_batch") else 1
                resource_demands[r_id].append(demand)

            if literals:
                self.model.AddExactlyOne(literals)
                task_vars[t_id] = {
                    "start": start_var,
                    "end": end_var,
                    "literals": literals,
                    "r_ids": r_ids,
                    "due": t.get("due_at_min") or t.get("DueAtMin") or horizon,
                    "priority": t.get("priority") or t.get("Priority") or 3,
                    "depends_on": t.get("final_depends_on") or t.get("FinalDependsOn") or [],
                    "internal_dep": t.get("internal_dep") or t.get("InternalDep")
                }

        # 3. APPLY CONSTRAINTS (FIXED)
        for r_id, r in resource_map.items():
            intervals = resource_intervals[r_id]
            unavail_intervals = resource_unavail_intervals.get(r_id, [])
            
            all_intervals = intervals + unavail_intervals
            
            if not all_intervals: continue
            
            # [FIX] Logic Batch Capacity
            if r.get("type") == "batch":
                # Lấy capacity thực tế, nếu không có thì mặc định 100
                machine_capacity = r.get("capacity", 100) 
                
                task_demands = resource_demands[r_id]
                
                # [FIX] Demand của giờ nghỉ phải BẰNG capacity của máy để chặn hoàn toàn
                unavail_demands = [machine_capacity] * len(unavail_intervals)
                
                self.model.AddCumulative(
                    all_intervals, 
                    task_demands + unavail_demands, 
                    machine_capacity
                )
            else:
                # [CRITICAL] AddNoOverlap là ràng buộc cứng.
                # Nếu unavail_intervals có dữ liệu, task KHÔNG THỂ lấn vào.
                self.model.AddNoOverlap(all_intervals)

        # ... (Giữ nguyên logic Dependencies và Objective) ...
        # ... (Dependency Buffer) ...
        for t_id, tv in task_vars.items():
             for parent_id in tv["depends_on"]:
                if parent_id in task_vars:
                    self.model.Add(tv["start"] >= task_vars[parent_id]["end"] + BUFFER_TIME)

             prev_id = tv["internal_dep"]
             if prev_id and prev_id in task_vars:
                prev_tv = task_vars[prev_id]
                for i, lit in enumerate(tv["literals"]):
                    r_id = tv["r_ids"][i]
                    if r_id in prev_tv["r_ids"]:
                        idx_prev = prev_tv["r_ids"].index(r_id)
                        self.model.Add(lit == prev_tv["literals"][idx_prev])
                self.model.Add(tv["start"] >= prev_tv["end"])

        # --- Objective ---
        makespan = self.model.NewIntVar(0, horizon, "makespan")
        objective_terms = []
        for _, tv in task_vars.items():
            self.model.Add(makespan >= tv["end"])
        objective_terms.append(makespan * 100)
        
        # Penalize lateness
        for _, tv in task_vars.items():
            delay = self.model.NewIntVar(0, horizon, f"delay")
            self.model.Add(delay >= tv["end"] - tv["due"])
            weight = (6 - tv["priority"]) * 1000 
            objective_terms.append(delay * weight)
            
        self.model.Minimize(sum(objective_terms))
        
        return task_vars

    # ... (Giữ nguyên _extract_solution) ...
    def _extract_solution(self, task_vars):
        results = []
        for t_id, tv in task_vars.items():
            start_val = self.solver.Value(tv["start"])
            end_val = self.solver.Value(tv["end"])
            
            selected_res = None
            for i, lit in enumerate(tv["literals"]):
                if self.solver.Value(lit) == 1:
                    selected_res = tv["r_ids"][i]
                    break
            
            if selected_res:
                results.append({
                    "task_id": t_id,
                    "machine_id": selected_res,
                    "start_min": start_val,
                    "end_min": end_val
                })
        return results

    # [NEW] Helper function để debug overlap
    def _analyze_overlaps(self, task_vars):
        print("\n🔍 ANALYZING SCHEDULE VS BREAKS...")
        resource_map = {r["id"]: r for r in self.resources}
        
        for t_id, tv in task_vars.items():
            start = self.solver.Value(tv["start"])
            end = self.solver.Value(tv["end"])
            
            selected_res = None
            for i, lit in enumerate(tv["literals"]):
                if self.solver.Value(lit) == 1:
                    selected_res = tv["r_ids"][i]
                    break
            
            if selected_res:
                # Check against windows of this resource
                windows = resource_map[selected_res].get("unavailability", [])
                for w in windows:
                    w_start = int(w["start"])
                    w_end = int(w["end"])
                    
                    # Logic check overlap
                    if max(start, w_start) < min(end, w_end):
                        overlap_amount = min(end, w_end) - max(start, w_start)
                        print(f"❌ OVERLAP DETECTED: Task {t_id} [{start}-{end}] on {selected_res}")
                        print(f"   -> Break Window: [{w_start}-{w_end}]")
                        print(f"   -> Overlap by: {overlap_amount} mins")