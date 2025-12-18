# backend/storage.py
from typing import Dict, Any
from datetime import datetime
import uuid

# 生产环境建议换成 Redis
task_storage: Dict[str, Dict[str, Any]] = {}

def create_task(ticker: str, trade_date: str) -> str:
    task_id = str(uuid.uuid4())
    task_storage[task_id] = {
        "ticker": ticker,
        "trade_date": trade_date,
        "status": "running",
        "logs": [f"[{datetime.now().strftime('%H:%M:%S')}] 任务启动：分析 {ticker} 于 {trade_date}"],
        "final_result": None,
        "created_at": datetime.now()
    }
    return task_id

def append_log(task_id: str, log_line: str):
    if task_id in task_storage:
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] {log_line}"
        logs = task_storage[task_id]["logs"]
        # Avoid appending the same log line consecutively (dedupe by message text)
        if logs:
            try:
                last = logs[-1]
                # strip leading timestamp like [HH:MM:SS] <space>
                last_text = last.split('] ', 1)[1] if '] ' in last else last
            except Exception:
                last_text = logs[-1]
            if last_text == log_line:
                return
        logs.append(entry)

def get_task(task_id: str):
    if task_id in task_storage:
        return task_storage[task_id]
    return None

def complete_task(task_id: str, final_state: dict, signal: str):
    if task_id in task_storage:
        task_storage[task_id]["status"] = "completed"
        task_storage[task_id]["final_result"] = {
            "decision": final_state.get('final_trade_decision', ''),
            "signal": signal
        }
        append_log(task_id, f"分析完成！最终信号: {signal}")