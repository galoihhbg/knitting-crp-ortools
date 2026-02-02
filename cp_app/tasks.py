import requests
import os
from .celery_app import celery_app
from .engine import Engine

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://backend:8082/api/webhook/solver")

def filter_dummy_tasks(assignments):
    """
    Lọc bỏ các task có ID bắt đầu bằng 'DUMMY_' hoặc operation là 'Unavailability'
    trước khi trả về cho Backend.
    """
    if not assignments:
        return []
    
    cleaned_assignments = []
    for task in assignments:
        task_id = str(task.get("task_id", ""))
        operation = str(task.get("operation", ""))
        
        if not task_id.startswith("DUMMY_") and operation != "Unavailability":
            cleaned_assignments.append(task)
            
    return cleaned_assignments

@celery_app.task(bind=True, name="optimize_schedule")
def optimize_schedule(self, payload: dict):
    try:
        print(f"[Task {self.request.id}] Starting...")

        engine = Engine(payload)
        result = engine.solve()
        
        raw_assignments = result.get("assignments", [])
        clean_assignments = filter_dummy_tasks(raw_assignments)

        response_data = {
            "job_id": payload.get("job_id"),
            "task_id": self.request.id,
            "status": result["status"],
            "assignments": clean_assignments
        }

        print(f"Sending back to main service: {WEBHOOK_URL}")
        
        resp = requests.post(WEBHOOK_URL, json=response_data, timeout=10)
        
        if resp.status_code == 200:
            return "Callback Successful"
        else:
            print(f"⚠️ Callback Failed: {resp.text}")
            return "Callback Failed"

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        try:
            requests.post(WEBHOOK_URL, json={
                "job_id": payload.get("job_id"),
                "status": "failed",
                "error": str(e)
            }, timeout=5)
        except:
            pass
        raise e