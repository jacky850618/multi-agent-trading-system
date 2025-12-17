# backend/api.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from .storage import create_task, get_task
from .tasks import run_analysis

app = FastAPI(title="Deep Thinking Trading API")

class AnalysisRequest(BaseModel):
    ticker: str
    trade_date: str


@app.post("/start")
def start_analysis(req: AnalysisRequest, background_tasks: BackgroundTasks):
    task_id = create_task(req.ticker, req.trade_date)
    background_tasks.add_task(run_analysis, task_id, req.ticker, req.trade_date)
    return {"task_id": task_id, "status": "started"}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    task = get_task(task_id)
    if not task:
        return {"status": "not_found"}
    return {
        "status": task["status"],
        "logs": task["logs"],
        "final_result": task.get("final_result")
    }
