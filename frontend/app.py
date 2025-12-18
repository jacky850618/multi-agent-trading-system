# frontend/app.py
import streamlit as st
import requests
import time
import importlib.util
from datetime import datetime, timedelta
import os
def _load_local_module(name, filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load settings/intro modules from files in the same folder
try:
    settings_mod = _load_local_module("settings", "settings.py")
except Exception as e:
    settings_mod = None
    st.error(f"无法加载 settings.py: {e}")

try:
    intro_mod = _load_local_module("intro", "intro.py")
except Exception as e:
    intro_mod = None
    st.error(f"无法加载 intro.py: {e}")

st.set_page_config(
    page_title="深度思考股票分析系统",  # 浏览器标签页标题
    page_icon="🧠",  # 图标
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 深度思考股票分析系统")
# 加载配置（使用 settings 模块中的实现）
if settings_mod is None:
    st.stop()
user_config = settings_mod.load_config()

tabs = st.tabs(["🧠 首页", "⚙️ 系统设置", "📘 系统介绍"])
home_tab, settings_tab, intro_tab = tabs

with settings_tab:
    if settings_mod is None:
        st.error("settings 模块未加载，无法显示设置界面。")
    else:
        settings_mod.render_settings(user_config)

with intro_tab:
    if intro_mod is None:
        st.error("intro 模块未加载，无法显示介绍页面。")
    else:
        intro_mod.render_intro()

with home_tab:
    # ========================== 主分析界面 ==========================
    if not settings_mod.is_configured(user_config):
        st.error("🚫 请先完成 API Key 配置！")
        st.info("请在左侧侧边栏填写 OpenAI、Finnhub 和 Tavily 的 API Key，然后点击保存。")
        st.stop()

    st.success("✅ 系统配置完成，可以开始分析, 请输入股票代码和交易日期，然后点击 **开始深度分析** 按钮！")

    # ------------------ 分析中任务面板 ------------------
    api_base = user_config.get("API_BASE", settings_mod.DEFAULT_CONFIG["API_BASE"]).rstrip("/")
    session = settings_mod.get_smart_session(user_config)
    try:
        tasks_resp = session.get(f"{api_base}/tasks", timeout=3)
        if tasks_resp.status_code == 200:
            tasks = tasks_resp.json().get("tasks", [])
        else:
            tasks = []
    except Exception as e:
        tasks = []

    running_tasks = [t for t in tasks if t.get("status") != "completed"]
    if running_tasks:
        st.markdown("### 🔄 分析中任务")
        for t in running_tasks:
            title = f"{t.get('ticker')}  — {t.get('status')}  — {t.get('created_at') or ''}"
            with st.expander(title, expanded=False):
                st.write(f"Task ID: {t.get('task_id')}")
                st.write(f"日志行数: {t.get('logs_count')}")
                if st.button("查看详情", key=f"view_{t.get('task_id')}"):
                    try:
                        s = session.get(f"{api_base}/status/{t.get('task_id')}", timeout=5)
                        if s.status_code == 200:
                            data = s.json()
                            logs = data.get("logs", []) or []
                            st.text_area("实时分析日志", "\n".join(logs), height=400)
                            final = data.get("final_result") or {}
                            st.write("最终结果:", final)
                        else:
                            st.error(f"无法获取任务详情：{s.status_code}")
                    except Exception as e:
                        st.error(f"获取任务详情失败：{e}")


    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("股票代码", value="NVDA", help="例如：NVDA, AAPL, 0700.HK")
    with col2:
        trade_date_input = st.date_input(
            "交易日期",
            value=datetime.now().date() - timedelta(days=2)
        )
    trade_date = trade_date_input.strftime('%Y-%m-%d')

    if st.button("🚀 开始深度分析", type="primary", use_container_width=True):
        st.info("正在提交分析任务...")
        api_base = user_config["API_BASE"]
        resp = requests.post(f"{api_base}/start", json={"ticker": ticker, "trade_date": trade_date})
        if resp.status_code != 200:
            st.error("后端服务不可用")
        else:
            task_id = resp.json()["task_id"]
            st.success(f"任务提交成功！Task ID: {task_id}")

            log_placeholder = st.empty()
            progress = st.progress(0)
            status_text = st.empty()

            logs = []
            while True:
                status_resp = requests.get(f"{api_base}/status/{task_id}")
                if status_resp.status_code == 200:
                    data = status_resp.json()
                    # 防御性获取字段，避免后端未包含某些键导致前端崩溃
                    new_logs = data.get("logs", []) or []
                    if new_logs != logs:
                        logs = new_logs
                        log_placeholder.text_area("实时分析日志", "\n".join(logs), height=600)

                    status = data.get("status")
                    if status == "completed":
                        result = data.get("final_result", {}) or {}
                        signal = result.get('signal') if isinstance(result, dict) else None
                        st.balloons()
                        st.success(f"最终信号：**{signal}**")
                        st.markdown("### 最终决策")
                        st.markdown(result.get("decision", ""))
                        st.download_button("下载日志", "\n".join(logs), f"analysis_{ticker}.txt")
                        break
                    elif status == "error":
                        # 显示后端返回的错误信息（如果有）
                        err = data.get("error") or data.get("message") or "分析失败"
                        st.error(f"分析失败: {err}")
                        break
                    else:
                        progress.progress(min(len(logs) / 25, 0.95))
                        status_text.text("分析进行中...")
                time.sleep(2)  # 每2秒轮询一次
