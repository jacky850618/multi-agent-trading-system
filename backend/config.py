
# 项目的全局配置中心和环境初始化。

import os
from getpass import getpass

def set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"请输入您的 {var}: ")

set_env("OPENAI_API_KEY")
set_env("FINNHUB_API_KEY")
set_env("TAVILY_API_KEY")
set_env("LANGSMITH_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "深度思考量化交易系统"

CONFIG = {
    "results_dir": "./results",
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",       # 用于复杂推理和最终决策的强大模型。
    "quick_think_llm": "gpt-4o-mini", # 用于数据处理和初步分析的快速、低成本模型。
    "backend_url": "https://api.openai.com/v1",
    # 辩论和讨论设置控制协作智能体的流程。
    "max_debate_rounds": 2,# 牛市与熊市的辩论将进行2轮。
    "max_risk_discuss_rounds": 1,# 风险团队进行1轮辩论。
    "max_recur_limit": 100,   # 智能体循环的安全限制。
    # 工具设置控制数据获取行为。
    "online_tools": True, # 使用实时 API；设置为 False 可使用缓存数据以更快、更便宜地运行。
    "data_cache_dir": "./data_cache"# 用于缓存在线数据的目录。
}

# 如果缓存目录不存在，则创建它。
os.makedirs(CONFIG["data_cache_dir"], exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)

print("Configuration dictionary created:")
print(CONFIG)