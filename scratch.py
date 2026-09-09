import json
import logging
logging.basicConfig(level=logging.DEBUG)

with open("/tmp/dump.json") as f:
    data = json.load(f)

tasks = data["tasks"]
p3_end_times = data["p3_end_times"]
resources = data["resources"]
config = data["config"]

from app.engine.phases.phase4_downstream import _compute_start_lb
from app.engine.shared import build_resource_model, compute_horizon
from ortools.sat.python import cp_model

horizon = compute_horizon(tasks, config)
start_lb = _compute_start_lb(tasks, p3_end_times)

print("Tasks count:", len(tasks))

model = cp_model.CpModel()
resource_map = {r["id"]: r for r in resources}

task_vars, _, _ = build_resource_model(model, tasks, resource_map, horizon, start_lb=start_lb)

print("build_resource_model done. Validating model...")
validation = model.Validate()
if validation:
    print("MODEL_INVALID:", validation)

# Add intra-phase constraints
for t in tasks:
    t_id = t["task_id"]
    if t_id not in task_vars:
        continue
    for dep_id in (t.get("final_depends_on") or []):
        if dep_id in task_vars:
            model.Add(task_vars[t_id]["start"] >= task_vars[dep_id]["end"])

solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True
solver.parameters.cp_model_presolve = True # Enable presolve to see conflicts
print("Solving...")
status = solver.Solve(model)
print("Status:", solver.StatusName(status))
