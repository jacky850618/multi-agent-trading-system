from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
import asyncio
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


@app.websocket("/ws/status/{task_id}")
async def websocket_status(websocket: WebSocket, task_id: str):
    """WebSocket endpoint that streams task logs in real-time.

    Clients should connect to `/ws/status/{task_id}` and will receive JSON
    messages of the form: {"type": "log", "line": "..."} for each log
    line, and a final message {"type":"status","status":"completed",...}
    when the task finishes.
    """
    await websocket.accept()
    try:
        last_idx = 0
        sent_report_keys = set()
        last_progress = None
        last_progress_status = None
        while True:
            task = get_task(task_id)
            if not task:
                await websocket.send_json({"type": "status", "status": "not_found"})
                await asyncio.sleep(0.5)
                continue

            logs = task.get("logs", [])
            # send any new logs
            if last_idx < len(logs):
                for line in logs[last_idx:]:
                    await websocket.send_json({"type": "log", "line": line})
                last_idx = len(logs)

            # send any new structured reports
            reports = task.get("reports", {}) or {}
            for label, body in reports.items():
                if label not in sent_report_keys:
                    await websocket.send_json({"type": "report", "label": label, "markdown": body})
                    sent_report_keys.add(label)

            # send progress updates when changed
            try:
                prog = float(task.get("progress", 0.0) or 0.0)
            except Exception:
                prog = 0.0
            prog_status = task.get("progress_status")
            if last_progress is None or abs(prog - (last_progress or 0.0)) > 1e-6 or (prog_status != last_progress_status):
                await websocket.send_json({"type": "progress", "progress": prog, "status": prog_status})
                last_progress = prog
                last_progress_status = prog_status

            status = task.get("status")
            if status == "completed":
                await websocket.send_json({"type": "status", "status": "completed", "final_result": task.get("final_result")})
                break
            if status == "error":
                await websocket.send_json({"type": "status", "status": "error", "error": task.get("error")})
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
