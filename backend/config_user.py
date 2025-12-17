import json
import os
from typing import Dict, Any

CONFIG_USER_FILE = "config_user.json"  # 与前端共用同一个文件

DEFAULT_USER_CONFIG = {
    "FINNHUB_API_KEY": "",
    "TAVILY_API_KEY": "",
    "LANGSMITH_API_KEY": "",
    "max_debate_rounds": 2,  # 牛市与熊市的辩论将进行2轮。
    "max_risk_discuss_rounds": 1,  # 风险团队进行1轮辩论。
    "max_recur_limit": 100,  # 智能体循环的安全限制。
    "online_tools": True,  # 使用实时 API；设置为 False 可使用缓存数据以更快、更便宜地运行。
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
    },
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    "OPENAI_API_KEY": "",
    "DEEPSEEK_API_KEY": "",
    "QWEN_API_KEY": "",
    "DOUBAO_API_KEY": "",
}


def set_env(config: Dict[str, Any], key: str):
    os.environ[key] = config[key]


def apply_proxy_settings(config: Dict[str, Any]):
    if config.get("proxy_enabled", False):
        host = config.get("proxy_host", "").strip()
        port = config.get("proxy_port", "").strip()
        if host and port:
            proxy_url = f"http://{host}:{port}"
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
            os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
            print(f"🌐 全局代理已启用：{proxy_url}")
        else:
            print("⚠️ 代理配置不完整")
    else:
        # 清理残留
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"]:
            os.environ.pop(key, None)
        print("🌐 代理已禁用")


def load_user_config() -> Dict[str, Any]:
    """加载用户自定义配置，如果文件不存在返回默认"""
    if os.path.exists(CONFIG_USER_FILE):
        try:
            with open(CONFIG_USER_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            # 合并默认值，确保新字段不会缺失
            config = {**DEFAULT_USER_CONFIG, **user_config}

            set_env(config, "FINNHUB_API_KEY")
            set_env(config, "TAVILY_API_KEY")
            set_env(config, "LANGSMITH_API_KEY")

            # 确保 prompts 完整
            config["prompts"] = {**DEFAULT_USER_CONFIG["prompts"], **user_config.get("prompts", {})}
            return config
        except Exception as e:
            print(f"加载配置文件失败，使用默认配置: {e}")
    return DEFAULT_USER_CONFIG.copy()


# 全局配置（后端启动时加载一次）
USER_CONFIG = load_user_config()
apply_proxy_settings(USER_CONFIG)

# 提供获取函数，便于其他模块导入
def get_user_config() -> Dict[str, Any]:
    return USER_CONFIG
