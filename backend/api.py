from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from .storage import create_task, get_task
from .tasks import run_analysis
from .config_user import get_user_config

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


print(
    f"当前 LLM 配置: {user_config['llm_provider']} | 复杂模型: {user_config['deep_think_llm']} | 快速模型: {user_config['quick_think_llm']}")
