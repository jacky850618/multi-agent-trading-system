from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from .storage import create_task, get_task
from .tasks import run_analysis
from .config_user import get_user_config
from .storage import task_storage

app = FastAPI(title="Deep Thinking Trading API")
user_config = get_user_config()

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


@app.get("/tasks")
def list_tasks(status: str = None):
    # 返回当前内存中的任务列表，可选按 status 过滤
    items = []
    for tid, t in task_storage.items():
        if status and t.get("status") != status:
            continue
        created = t.get("created_at")
        items.append({
            "task_id": tid,
            "ticker": t.get("ticker"),
            "trade_date": t.get("trade_date"),
            "status": t.get("status"),
            "logs_count": len(t.get("logs", [])),
            "created_at": created.isoformat() if created is not None else None,
        })
    return {"tasks": items}


print(
    f"当前 LLM 配置: {user_config['llm_provider']} | 复杂模型: {user_config['deep_think_llm']} | 快速模型: {user_config['quick_think_llm']}")
