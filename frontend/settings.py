import streamlit as st
import os
import requests
import time
import json


# ========================== 默认配置 ==========================
DEFAULT_CONFIG = {
    "FINNHUB_API_KEY": "",
    "TAVILY_API_KEY": "",
    "LANGSMITH_API_KEY": "",
    "API_BASE": "http://127.0.0.1:8000",
    "llm_provider": "ChatGPT(Openai)",
    "deep_think_llm": "gpt-4o",  # 用于复杂推理和最终决策的强大模型。
    "quick_think_llm": "gpt-4o-mini",  # 用于数据处理和初步分析的快速、低成本模型。
    "backend_url": "https://api.openai.com/v1",
    "proxy_enabled": False,
    "proxy_host": "127.0.0.1",
    "proxy_port": "7890",
    "max_debate_rounds": 2,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "online_tools": True,
    "prompts": {
        "bull": "您是一位多头分析师。您的目标是论证投资该股票的合理性。请重点关注增长潜力、竞争优势以及报告中的积极指标。有效反驳看跌分析师的论点。",
        "bear": "您是一位空头分析师。您的目标是论证投资该股票的不合理性。请重点关注风险、挑战以及负面指标。有效反驳看涨分析师的论点。",
        "risky": "您是冒险型风险分析师。您主张高回报机会和大胆策略。",
        "safe": "您是稳健型风险分析师。您优先考虑资本保值和最小化波动性。",
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
    """
        检查是否已完成必要配置：
        - Finnhub 和 Tavily 必须填写（数据源）
        - LLM 平台（OpenAI / DeepSeek / 通义千问 / 豆包）至少配置一个 API Key
        """
    # 必填数据源
    data_required = ["FINNHUB_API_KEY", "TAVILY_API_KEY"]
    if not all(config.get(key, "").strip() != "" for key in data_required):
        return False

    # LLM 平台至少配置一个
    llm_keys = [
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "QWEN_API_KEY",
        "DOUBAO_API_KEY"
    ]
    if not any(config.get(key, "").strip() != "" for key in llm_keys):
        return False

    return True

# ========================== 测试连接函数（测试 Google 首页 + 本地后端） ==========================
def test_connections(session):
    results = []
    try:
        resp = session.get("https://www.google.com", timeout=10)
        if resp.status_code == 200:
            results.append(("✅ 外部网络（Google）", "连接成功，代理工作正常"))
        else:
            results.append(("⚠️ 外部网络（Google）", f"状态码 {resp.status_code}"))
    except Exception as e:
        results.append(("❌ 外部网络（Google）", f"连接失败：{str(e)}"))

    return results


def get_smart_session(config):
    """
    智能代理会话：
    - 如果目标是 127.0.0.1 或 localhost → 直连（不走代理）
    - 其他所有请求 → 走用户配置的代理
    """
    session = requests.Session()

    if config.get("proxy_enabled", False):
        host = config.get("proxy_host", "").strip()
        port = config.get("proxy_port", "").strip()
        if host and port:
            proxy_url = f"http://{host}:{port}"
            # 设置全局代理
            session.proxies.update({
                "http": proxy_url,
                "https": proxy_url,
            })
            # st.success(f"代理已启用：{proxy_url}（外部服务走代理，本地直连）")
        else:
            st.warning("代理启用但地址/端口为空，将直连所有服务")

        # 关键：添加 NO_PROXY 环境变量，绕过本地地址
        # requests 尊重 NO_PROXY
        import os
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"

    else:
        st.sidebar.info("代理未启用（所有服务直连）")

    return session


def render_settings(user_config):

    with st.expander("🌐 后端服务地址"):
        api_base = st.text_input("后端 FastAPI 服务地址（API_BASE）", value=user_config.get("API_BASE", "http://127.0.0.1:8000"))

    with st.expander("🌐 网络代理设置"):
        proxy_enabled = st.checkbox("启用网络代理（仅外部服务）", value=user_config.get("proxy_enabled", False))
        proxy_host = st.text_input("代理地址（Host）", value=user_config.get("proxy_host", "127.0.0.1"))
        proxy_port = st.text_input("代理端口（Port）", value=user_config.get("proxy_port", "7890"))

        if st.button("🧪 测试网络连接", type="secondary"):
            temp = user_config.copy()
            temp.update({"proxy_enabled": proxy_enabled, "proxy_host": proxy_host, "proxy_port": proxy_port})
            sess = get_smart_session(temp)
            results = test_connections(sess)
            for icon, msg in results:
                if "成功" in icon or icon == "Google":
                    st.success(f"{icon} {msg}")
                else:
                    st.warning(f"{icon} {msg}")

    with st.expander("🔑 API Keys", expanded=not is_configured(user_config)):
        finnhub_key = st.text_input("Finnhub API Key", value=user_config.get("FINNHUB_API_KEY", ""), type="password")
        st.caption("用途：用于获取实时与历史市场数据（行情、财务、指标等）。申请地址：https://finnhub.io/")

        tavily_key = st.text_input("Tavily API Key", value=user_config.get("TAVILY_API_KEY", ""), type="password")
        st.caption("用途：用于访问社交媒体与另类数据源（情绪、话题热度等）。申请/文档地址：请参考 Tavily 官方网站（例如 https://tavily.ai 或您的服务提供商控制台）。")

        langsmith_key = st.text_input("LangSmith API Key（可选）", value=user_config.get("LANGSMITH_API_KEY", ""), type="password")
        st.caption("用途：用于将模型调用与运行时追踪发送到 LangSmith（调试、可观测性与运行记录）。申请地址： https://www.langsmith.com/ 或 LangSmith 控制台。")

    with st.expander("🤖 大语言模型配置"):
        llm_provider_options = ["ChatGPT(Openai)", "Deepseek", "通义千问(qwen)", "豆包(doubao)"]
        llm_provider = st.selectbox("LLM 提供商 (llm_provider)", options=llm_provider_options, index=llm_provider_options.index(user_config.get("llm_provider", "ChatGPT(Openai)")))
        if llm_provider == "ChatGPT(Openai)":
            openai_api_key = st.text_input("OpenAI API Key", value=user_config.get("OPENAI_API_KEY", ""), type="password")
        elif llm_provider == "Deepseek":
            deepseek_api_key = st.text_input("DeepSeek API Key", value=user_config.get("DEEPSEEK_API_KEY", ""), type="password")
        elif llm_provider == "通义千问(qwen)":
            qwen_api_key = st.text_input("通义千问 API Key", value=user_config.get("QWEN_API_KEY", ""), type="password")
        elif llm_provider == "豆包(doubao)":
            doubao_api_key = st.text_input("豆包 API Key", value=user_config.get("DOUBAO_API_KEY", ""), type="password")

        deep_think_llm = st.text_input("复杂推理模型 (deep_think_llm)", value=user_config.get("deep_think_llm", ""))
        quick_think_llm = st.text_input("快速处理模型 (quick_think_llm)", value=user_config.get("quick_think_llm", ""))
        backend_url = st.text_input("模型基地址 (backend_url)", value=user_config.get("backend_url", ""))

        with st.expander("LLM 帮助 / 推荐配置（点击查看）", expanded=False):
            provider_tips = {
                "ChatGPT(Openai)": {
                    "deep": "gpt-4o",
                    "quick": "gpt-4o-mini",
                    "backend": "https://api.openai.com/v1",
                    "apply": "https://platform.openai.com/account/api-keys",
                    "note": "OpenAI 适合高质量复杂推理与决策；按需选择模型规格以平衡成本与性能。"
                },
                "Deepseek": {
                    "deep": "deepseek-chat",
                    "quick": "deepseek-coder",
                    "backend": "https://api.deepseek.com/v1",
                    "apply": "https://platform.deepseek.com/api_keys",
                    "note": "Deepseek 提供低延迟的企业模型（示例链接，请参考供应商文档）。"
                },
                "通义千问(qwen)": {
                    "deep": "qwen-7b",
                    "quick": "qwen-mini",
                    "backend": "请参考阿里云通义千问控制台（阿里云）",
                    "apply": "https://www.aliyun.com/（在阿里云控制台搜索 “通义千问” 以获取 API Key）",
                    "note": "通义千问由阿里巴巴提供，适合中文场景；请在阿里云控制台创建并查看接入文档。"
                },
                "豆包(doubao)": {
                    "deep": "（示例模型，依据供应商）",
                    "quick": "（示例模型，依据供应商）",
                    "backend": "请参考您的豆包供应商或私有部署文档",
                    "apply": "请咨询豆包供应商或查看其开发者控制台/文档",
                    "note": "“豆包” 在此作为示例占位（不同机构实现不同）。如需我把具体供应商链接或默认模型写入配置，请提供准确 URL 或说明。"
                }
            }
            tip = provider_tips.get(llm_provider, {})
            if tip:
                st.markdown(f"**建议复杂推理模型:** {tip.get('deep')}  ")
                st.markdown(f"**建议快速处理模型:** {tip.get('quick')}  ")
                st.markdown(f"**建议模型基地址:** {tip.get('backend')}  ")
                st.markdown(f"**API Key 申请/文档:** [{tip.get('apply')}]({tip.get('apply')})  ")
                st.caption(tip.get('note'))
            else:
                st.write("请参考所选提供商的官方文档以获取推荐模型与申请链接。")

 

    with st.expander("🛠️ 系统参数"):
        max_debate = st.slider("多空辩论轮数", 1, 5, user_config.get("max_debate_rounds", 2))
        max_risk = st.slider("风控辩论轮数", 1, 3, user_config.get("max_risk_discuss_rounds", 1))
        max_recur = st.number_input("最大递归限制", 50, 500, user_config.get("max_recur_limit", 100))
        online_tools = st.checkbox("启用在线工具", value=user_config.get("online_tools", True))

    with st.expander("✍️ 自定义提示词"):
        prompts = user_config.get("prompts", {}).copy()
        for key, label in [("bull", "多头分析员"), ("bear", "空头分析员"), ("risky", "激进风控研究员"), ("safe", "稳健风控研究员"), ("neutral", "平衡风控研究员"), ("market_analyst", "市场分析师"), ("social_analyst", "社交媒体分析师"), ("news_analyst", "新闻分析师"), ("fundamentals_analyst", "基本面分析师")]:
            prompts[key] = st.text_area(f"{label}提示词", value=prompts.get(key, ""), height=100)

    if st.button("💾 保存所有设置", type="primary", use_container_width=True):
        new_config = user_config.copy()
        new_config.update({
            "FINNHUB_API_KEY": finnhub_key.strip(),
            "TAVILY_API_KEY": tavily_key.strip(),
            "LANGSMITH_API_KEY": langsmith_key.strip(),
            "API_BASE": api_base.strip().rstrip("/"),
            "llm_provider": llm_provider,
            "deep_think_llm": deep_think_llm.strip(),
            "quick_think_llm": quick_think_llm.strip(),
            "backend_url": backend_url.strip().rstrip("/"),
            "proxy_enabled": proxy_enabled,
            "proxy_host": proxy_host.strip(),
            "proxy_port": proxy_port.strip(),
            "max_debate_rounds": max_debate,
            "max_risk_discuss_rounds": max_risk,
            "max_recur_limit": max_recur,
            "online_tools": online_tools,
            "prompts": prompts
        })

        # provider-specific keys
        if llm_provider == "ChatGPT(Openai)":
            new_config["OPENAI_API_KEY"] = openai_api_key.strip()
        elif llm_provider == "Deepseek":
            new_config["DEEPSEEK_API_KEY"] = deepseek_api_key.strip()
        elif llm_provider == "通义千问(qwen)":
            new_config["QWEN_API_KEY"] = qwen_api_key.strip()
        elif llm_provider == "豆包(doubao)":
            new_config["DOUBAO_API_KEY"] = doubao_api_key.strip()

        save_config(new_config)
        st.success("✅ 设置已保存,正在应用新配置...")
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()
