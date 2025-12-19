# frontend/app.py
import streamlit as st
import requests
import time
import importlib.util
import threading
import queue
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
        submit_ph = st.empty()
        submit_ph.info("正在提交分析任务...")
        api_base = user_config["API_BASE"]
        resp = requests.post(f"{api_base}/start", json={"ticker": ticker, "trade_date": trade_date})
        if resp.status_code != 200:
            submit_ph.error("后端服务不可用")
        else:
            task_id = resp.json()["task_id"]
            # persist the success message in a placeholder so it doesn't vanish on reruns
            submit_ph.success(f"任务提交成功！Task ID: {task_id}")
            # ---------------- websocket listener ----------------
            q = queue.Queue()
            stop_event = threading.Event()

            def _ws_listener(api_base_url, task_id, out_q, stop_evt):
                # Try to use websocket-client; fallback to pushing an error into queue
                try:
                    import websocket
                except Exception as e:
                    out_q.put(f"**WebSocket 库未安装，回退到 HTTP 轮询 (错误: {e})**")
                    out_q.put(None)
                    return

                ws_url = api_base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/status/{task_id}"

                def on_message(ws, message):
                    # backend may send either plain markdown text or a JSON string like
                    # {"type":"log","line":"...markdown..."}
                    try:
                        import json as _json
                        parsed = _json.loads(message)
                        # If structured message contains 'line' or 'markdown', use that
                        if isinstance(parsed, dict):
                            # structured report message
                            if parsed.get("type") == "report":
                                out_q.put(parsed)
                                return
                            if parsed.get("line"):
                                out_q.put(parsed.get("line"))
                                return
                            if parsed.get("markdown"):
                                out_q.put(parsed.get("markdown"))
                                return
                            # If final result object included
                            if parsed.get("type") == "final_result" or parsed.get("final"):
                                out_q.put(parsed)
                                return
                        # fallback: put raw message
                        out_q.put(message)
                    except Exception:
                        # not JSON, treat as raw markdown/text
                        out_q.put(message)

                def on_error(ws, error):
                    out_q.put(f"**WS_ERROR:** {error}")

                def on_close(ws, close_status_code, close_msg):
                    out_q.put(None)

                try:
                    wsapp = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
                    wsapp.run_forever()
                except Exception as e:
                    out_q.put(f"**WS_RUN_ERROR:** {e}")
                    out_q.put(None)

            t = threading.Thread(target=_ws_listener, args=(api_base, task_id, q, stop_event), daemon=True)
            t.start()

            log_placeholder = st.empty()
            progress = st.progress(0)
            status_text = st.empty()

            # keep a current compact status (e.g. timestamped "任务启动: ..." lines)
            current_status = None

            # Placeholder for report tabs
            reports_placeholder = st.empty()

            logs = []
            finished = False
            seen = set()
            reports_contents = {}  # label -> markdown body
            # read from queue until None sentinel
            while True:
                try:
                    item = q.get(timeout=1)
                except queue.Empty:
                    item = None

                if item is None:
                    if not t.is_alive():
                        finished = True
                        break
                    # no new message, continue polling
                    # update progress placeholder periodically
                    # progress updated each loop below as well
                    if current_status:
                        status_text.text(f"分析进行中({current_status})...")
                    else:
                        status_text.text("分析进行中...")
                    continue

                # handle structured final messages (dict) from WS
                if isinstance(item, dict):
                    # handle structured messages
                    # progress messages -> update progress bar and status
                    if item.get("type") == "progress":
                        try:
                            prog = float(item.get("progress", 0.0) or 0.0)
                        except Exception:
                            prog = 0.0
                        try:
                            progress.progress(min(max(prog, 0.0), 1.0))
                        except Exception:
                            pass
                        pstatus = item.get("status")
                        try:
                            if pstatus:
                                status_text.text(f"分析进行中({pstatus})... {int(prog*100)}%")
                            else:
                                status_text.text(f"分析进行中... {int(prog*100)}%")
                        except Exception:
                            pass
                        continue

                    # report messages -> add to report tabs
                    if item.get("type") == "report":
                        label = item.get("label") or item.get("name") or "报告"
                        body = item.get("markdown") or item.get("body") or ""
                        try:
                            reports_contents[label] = body
                        except Exception:
                            reports_contents[label] = str(body)
                        # refresh tabs display immediately
                        try:
                            titles = list(reports_contents.keys())
                            combined = "\n\n".join(logs)
                            tabs = reports_placeholder.tabs(["日志"] + titles)
                            tabs[0].markdown(combined, unsafe_allow_html=False)
                            for idx, title in enumerate(titles, start=1):
                                tabs[idx].markdown(reports_contents.get(title, ""), unsafe_allow_html=False)
                        except Exception:
                            reports_placeholder.markdown("\n\n".join(logs), unsafe_allow_html=False)
                        continue

                    # if backend sent a final_result payload, render final decision
                    final = item.get("final_result") or item.get("final") or item.get("result")
                    if final:
                        signal = final.get("signal") if isinstance(final, dict) else None
                        st.balloons()
                        st.success(f"最终信号：**{signal}**")
                        st.markdown("### 最终决策")
                        st.markdown(final.get("decision", "") if isinstance(final, dict) else str(final))
                        st.download_button("下载日志", "\n\n".join(logs), f"analysis_{ticker}.md")
                        break
                    # otherwise ignore unknown dicts
                    continue

                # received a markdown fragment/string
                if isinstance(item, str):
                    text = item.strip()
                    if not text:
                        continue

                    # If message is a short status line (timestamped or startup/info messages),
                    # treat it as the status summary and do NOT append to the main logs.
                    try:
                        import re
                        # timestamped lines like: [02:01:06] 任务启动：分析 NVDA 于 2025-12-17
                        m_ts = re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s*(.+)$", text)
                        if m_ts:
                            status_msg = m_ts.group(1).strip()
                            current_status = status_msg
                            status_text.text(f"分析进行中({status_msg})...")
                            # do not add to logs
                            continue

                        # plain short status lines or lines starting with emoji or key phrases
                        if text.startswith("✅") or "任务启动" in text or "开始执行" in text or "独立工作流" in text:
                            current_status = text
                            status_text.text(f"分析进行中({text})...")
                            continue

                        # update status if message contains node info like '执行节点: ...'
                        m = re.search(r"执行节点:\s*(.*)$", text)
                        if m:
                            node_name = m.group(1).strip()
                            current_status = node_name
                            status_text.text(f"分析进行中({node_name})...")
                            # continue processing this line as a normal log as it may contain useful details
                    except Exception:
                        pass

                    # skip exact duplicates
                    if text in seen:
                        continue
                    # skip repeating the very last appended block
                    if logs and logs[-1].strip() == text:
                        continue
                    seen.add(text)
                    logs.append(item)

                    # detect generated report blocks of form "<label>已生成:\n<markdown>"
                    if "已生成:\n" in text:
                        try:
                            label, body = text.split("已生成:\n", 1)
                            label = label.strip()
                            body = body.strip()
                            # store/overwrite report content
                            reports_contents[label] = body
                        except Exception:
                            pass

                    # join logs as markdown and render inside a single tabs area
                    combined = "\n\n".join(logs)
                    try:
                        titles = list(reports_contents.keys())
                        tabs = reports_placeholder.tabs(["日志"] + titles)
                        tabs[0].markdown(combined or "(无日志)", unsafe_allow_html=False)
                        for idx, title in enumerate(titles, start=1):
                            body_md = reports_contents.get(title, "")
                            tabs[idx].markdown(body_md, unsafe_allow_html=False)
                    except Exception:
                        # fallback: plain markdown
                        reports_placeholder.markdown(combined or "(无日志)", unsafe_allow_html=False)

                # update progress every loop iteration so the progress bar moves
                try:
                    progress.progress(min(len(logs) / 25, 0.95))
                except Exception:
                    pass

            # ws closed; fetch final status by HTTP as fallback
            try:
                status_resp = requests.get(f"{api_base}/status/{task_id}")
                if status_resp.status_code == 200:
                    data = status_resp.json()
                    result = data.get("final_result", {}) or {}
                    signal = result.get("signal") if isinstance(result, dict) else None
                    st.balloons()
                    st.success(f"最终信号：**{signal}**")
                    st.markdown("### 最终决策")
                    st.markdown(result.get("decision", ""))
                    st.download_button("下载日志", "\n\n".join(logs), f"analysis_{ticker}.md")
                else:
                    st.warning("无法通过 HTTP 获取最终结果，可能已通过 WebSocket 完成。")
            except Exception as e:
                st.error(f"获取最终结果失败：{e}")
