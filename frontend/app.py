# frontend/app.py
import streamlit as st
import requests
import time
import datetime

API_BASE = "http://127.0.0.1:8000"  # 部署时改成您的后端地址

st.title("🧠 深度思考股票分析系统")
st.info("请输入股票代码和交易日期，然后点击 **开始深度分析** 按钮")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("股票代码", "NVDA")
with col2:
    trade_date = st.date_input("交易日期", datetime.date.today() - datetime.timedelta(days=2))

if st.button("🚀 开始深度分析", type="primary"):
    resp = requests.post(f"{API_BASE}/start", json={"ticker": ticker, "trade_date": trade_date.strftime('%Y-%m-%d')})
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
            status_resp = requests.get(f"{API_BASE}/status/{task_id}")
            if status_resp.status_code == 200:
                data = status_resp.json()
                new_logs = data["logs"]
                if new_logs != logs[-len(new_logs):] if logs else True:
                    logs = new_logs
                    log_placeholder.text_area("实时分析日志", "\n".join(logs), height=600)

                if data["status"] == "completed":
                    result = data["final_result"]
                    st.balloons()
                    st.success(f"最终信号：**{result['signal']}**")
                    st.markdown("### 最终决策")
                    st.markdown(result["decision"])
                    st.download_button("下载日志", "\n".join(logs), f"analysis_{ticker}.txt")
                    break
                elif data["status"] == "error":
                    st.error("分析失败")
                    break
                else:
                    progress.progress(min(len(logs) / 25, 0.95))
                    status_text.text("分析进行中...")
            time.sleep(2)  # 每2秒轮询一次

st.markdown("""
    ## 
    **Deep Thinking Trading System** 是一个基于 **LangGraph 多智能体框架** 的高级智能交易决策系统，模拟真实投资机构的完整决策流程，具备以下突出优势：

    ### 🧠 多智能体深度协作（Society of Agents）
    - 系统由 **12+ 专业智能体** 组成，包括：市场分析师、社交媒体分析师、新闻分析师、基本面分析师、多空研究员、研究员主管、交易员、三方风控分析师（激进/稳健/平衡）、投资组合经理。
    - 每个智能体拥有独立角色、专业工具和长期记忆，协作完成从情报收集 → 辩论 → 交易提案 → 风险审核 → 最终决策的全链路。

    ### ⚔️ 内置对抗性辩论机制
    - **多空辩论**：多头研究员与空头研究员进行多轮激烈辩论，充分暴露机会与风险。
    - **三方风控辩论**：激进/稳健/平衡三位风控分析师对交易提案进行挑战，确保决策稳健。

    ### 🌐 实时多源数据驱动
    - 集成 **Yahoo Finance、Finnhub、Tavily 实时搜索**，获取最新股价、技术指标、新闻、社交媒体情绪、基本面数据。
    - 支持 ReAct 智能循环：智能体可多次调用工具，直至获得足够信息。

    ### 🧬 自进化学习能力
    - 每个关键智能体拥有独立的 **长期记忆系统**（基于 ChromaDB 向量存储）。
    - 每次交易结束后进行反思，将经验教训存档，系统随使用次数增多而越来越聪明。

    ### 🔍 多维度质量评估
    - **LLM-as-a-Judge**：大模型评分决策的逻辑性、证据支持和可执行性。
    - **真实市场验证**：对比实际股价表现评估信号正确性。
    - **事实一致性审计**：防止代理“幻觉”，确保报告与数据源一致。
    
    ### 📊 支持的市场
    | 市场          | 支持程度 | 示例代码                  | 推荐度    | 说明                                      |
    |---------------|----------|---------------------------|-----------|-------------------------------------------|
    | **美国股市**（NYSE/NASDAQ） | ★★★★★ | NVDA, AAPL, TSLA, MSFT   | 强烈推荐 | 数据最完整、实时性最强、分析最可靠        |
    | **港股**（HKEX）            | ★★★★  | 0700.HK（腾讯）、9988.HK（阿里） | 推荐     | 支持良好，适合中概股和港股通标的          |
    | **A股**（沪深）             | ★★★   | 600519.SS（茅台）、000001.SZ | 可尝试   | 数据部分延迟，建议结合国内资讯验证        |
    | **其他国际市场**（欧股、日股等） | ★★    | ASML.AS, 7203.T          | 参考使用 | 数据覆盖有限，仅作初步分析                |

    > **最佳分析对象**：美国 NYSE / NASDAQ 上市股票（FAANG、科技、AI、新能源等热门板块）  
    > **次佳选择**：香港主板上市股票（尤其是中概互联、科技股）

    ### 📊 完全透明的可视化过程
    - 实时展示每个智能体的执行节点、工具调用和思考过程。
    - 用户像观看“交易团队会议”一样，看到决策是如何一步步形成的。

    ### 🚀 适用于
    - 量化交易研究、投资决策辅助、算法交易策略验证、AI 金融教育演示。
    > **这不仅仅是一个交易信号生成器，而是一个可解释、可学习、可扩展的“智能投资研究机构”。**
    """)
