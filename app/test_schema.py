from app.schemas.request_schema import SolverTask
t = SolverTask(task_id="T1", original_order_id="O1", group_id="G1", operation="test", qty=1, total_qty=1, priority=1, start_after_min=0, duration=10, is_slice=False, design_item_id="D1", color_config="C1")

d = t.model_dump(by_alias=False)
print("DUMP:", d)

horizon = 50000
due_at = int(d.get("due_at_min", horizon))
print("DUE_AT:", due_at)
print("IS_LATE (end=10):", 10 > due_at)
