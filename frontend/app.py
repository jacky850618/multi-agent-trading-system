# frontend/app.py
import streamlit as st
import requests
import time
from datetime import datetime, timedelta
import os
import json

# ========================== 默认配置 ==========================
DEFAULT_CONFIG = {
    "OPENAI_API_KEY": "",
    "FINNHUB_API_KEY": "",
    "TAVILY_API_KEY": "",
    "LANGSMITH_API_KEY": "",
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "online_tools": True,
    "prompts": {
        "bull": "您是一位多头分析师。您的目标是论证投资该股票的合理性。请重点关注增长潜力、竞争优势以及报告中的积极指标。有效反驳看跌分析师的论点。",
        "bear": "您是一位空头分析师。您的目标是论证投资该股票的不合理性。请重点关注风险、挑战以及负面指标。有效反驳看涨分析师的论点。",
        "risky": "您是冒险型风险分析师。您主张高回报机会和大胆策略。",
        "safe": "您是稳健 / 保守型风险分析师。您优先考虑资本保值和最小化波动性。",
        "neutral": "您是平衡型风险分析师。您提供平衡的视角，权衡收益和风险。",
        "market_analyst": "您是一位专门分析金融市场的交易助理。您的职责是选择最相关的技术指标来分析股票的价格走势、动量和波动性。您必须使用工具获取历史数据，然后生成一份包含分析结果的报告，其中包括一个汇总表。",
        "social_analyst": "您是一名社交媒体分析师。您的工作是分析过去一周内特定公司的社交媒体帖子和公众情绪。使用您的工具查找相关讨论，并撰写一份全面的报告，详细说明您的分析、见解以及对交易者的影响，包括一份汇总表。",
        "news_analyst": "您是一名新闻研究员，负责分析过去一周的最新新闻和趋势。请撰写一份关于当前世界形势的综合报告，内容需与交易和宏观经济相关。请使用您的工具提供全面、详细的分析，包括汇总表。",
        "fundamentals_analyst": "您是一名研究员，正在分析公司的基本面信息。请撰写一份关于公司财务状况、内部人士情绪和交易情况的综合报告，以全面了解其基本面状况，并附上汇总表。"
    }
}

CONFIG_FILE = "config_user.json"


# ========================== 配置加载与保存 ==========================
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 合并默认值，确保新增字段不会缺失
            config = {**DEFAULT_CONFIG, **data}
            config["prompts"] = {**DEFAULT_CONFIG["prompts"], **data.get("prompts", {})}
            return config
        except Exception as e:
            st.error(f"配置文件加载失败，将使用默认配置: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def is_configured(config):
    required = ["OPENAI_API_KEY", "FINNHUB_API_KEY", "TAVILY_API_KEY"]
    return all(config.get(key, "").strip() != "" for key in required)


API_BASE = "http://127.0.0.1:8000"  # 部署时改成您的后端地址

st.title("🧠 深度思考股票分析系统")
st.info("请输入股票代码和交易日期，然后点击 **开始深度分析** 按钮")

# 加载配置
user_config = load_config()

# ========================== 侧边栏设置面板 ==========================
with st.sidebar:
    st.header("⚙️ 系统设置")

    with st.expander("🔑 API Keys（必须填写）", expanded=not is_configured(user_config)):
        openai_key = st.text_input(
            "OpenAI API Key",
            value=user_config.get("OPENAI_API_KEY", ""),
            type="password",
            help=(
                "**用途**：驱动所有大语言模型（GPT-4o、GPT-4o-mini），负责智能体的推理、辩论、报告生成和最终决策。\n\n"
                "[申请 OpenAI API Key](https://platform.openai.com/api-keys)"
            )
        )
        finnhub_key = st.text_input(
            "Finnhub API Key",
            value=user_config.get("FINNHUB_API_KEY", ""),
            type="password",
            help=(
                "**用途**：获取公司新闻、财报事件、基本面数据（如市值、PE 等），是新闻分析师和基本面分析的核心数据源。\n\n"
                "[免费申请 Finnhub API Key](https://finnhub.io/register)"
            )
        )
        tavily_key = st.text_input(
            "Tavily API Key",
            value=user_config.get("TAVILY_API_KEY", ""),
            type="password",
            help=(
                "**用途**：实时网页搜索，用于获取社交媒体情绪、最新基本面分析、宏观新闻等，是社交媒体分析师和基本面分析师的关键工具。\n\n"
                "[申请 Tavily API Key](https://app.tavily.com/home)"
            )
        )
        langsmith_key = st.text_input(
            "LangSmith API Key（可选）",
            value=user_config.get("LANGSMITH_API_KEY", ""),
            type="password",
            help=(
                "**用途**：用于 LangSmith 追踪和调试代理链路（可视化每个智能体的调用过程），非必需，但强烈推荐开启以便调试。\n\n"
                "[申请 LangSmith API Key](https://smith.langchain.com/settings/api-keys)"
            )
        )

    with st.expander("🛠️ 系统参数"):
        max_debate = st.slider("多空辩论轮数", 1, 5, user_config.get("max_debate_rounds", 2))
        max_risk = st.slider("风控辩论轮数", 1, 3, user_config.get("max_risk_discuss_rounds", 1))
        max_recur = st.number_input("最大递归限制", 50, 500, user_config.get("max_recur_limit", 100))
        online_tools = st.checkbox("启用在线工具", value=user_config.get("online_tools", True))

    with st.expander("✍️ 智能体提示词自定义"):
        prompts = user_config.get("prompts", DEFAULT_CONFIG["prompts"]).copy()
        for key, label in [
            ("bull", "多头研究员"),
            ("bear", "空头研究员"),
            ("risky", "激进风控"),
            ("safe", "保守风控"),
            ("neutral", "中立风控"),
            ("market_analyst", "市场分析师"),
            ("social_analyst", "社交媒体分析师"),
            ("news_analyst", "新闻分析师"),
            ("fundamentals_analyst", "基本面分析师")
        ]:
            prompts[key] = st.text_area(f"{label}提示词", value=prompts.get(key, DEFAULT_CONFIG["prompts"][key]),
                                        height=100)

    if st.button("💾 保存所有设置", type="primary", use_container_width=True):
        new_config = {
            "OPENAI_API_KEY": openai_key.strip(),
            "FINNHUB_API_KEY": finnhub_key.strip(),
            "TAVILY_API_KEY": tavily_key.strip(),
            "LANGSMITH_API_KEY": langsmith_key.strip(),
            "max_debate_rounds": max_debate,
            "max_risk_discuss_rounds": max_risk,
            "max_recur_limit": max_recur,
            "online_tools": online_tools,
            "prompts": prompts
        }
        save_config(new_config)
        st.success("✅ 设置已保存,正在应用新配置...")
        st.balloons()

        time.sleep(3)
        st.rerun()

    st.markdown("---")
    st.caption(f"配置文件路径：`{os.path.abspath(CONFIG_FILE)}`")

# ========================== 主分析界面 ==========================
if not is_configured(user_config):
    st.error("🚫 请先完成 API Key 配置！")
    st.info("请在左侧侧边栏填写 OpenAI、Finnhub 和 Tavily 的 API Key，然后点击保存。")
    st.stop()

st.success("✅ 系统配置完成，可以开始分析！")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("股票代码", value="NVDA", help="例如：NVDA, AAPL, 0700.HK")
with col2:
    trade_date_input = st.date_input(
        "交易日期",
        value=datetime.date.today() - timedelta(days=2)
    )
trade_date = trade_date_input.strftime('%Y-%m-%d')

if st.button("🚀 开始深度分析", type="primary", use_container_width=True):
    st.info("正在提交分析任务...")
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
