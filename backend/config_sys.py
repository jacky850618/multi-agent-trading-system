
# 项目的全局配置中心和环境初始化。

import os
from getpass import getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"请输入您的 {var}: ")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "深度思考量化交易系统"

CONFIG_SYS = {
    "results_dir": "./results",
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o",       # 用于复杂推理和最终决策的强大模型。
    "quick_think_llm": "gpt-4o-mini", # 用于数据处理和初步分析的快速、低成本模型。
    "backend_url": "https://api.openai.com/v1",
    "data_cache_dir": "./data_cache"# 用于缓存在线数据的目录。
}

# 如果缓存目录不存在，则创建它。
os.makedirs(CONFIG_SYS["data_cache_dir"], exist_ok=True)
os.makedirs(CONFIG_SYS["results_dir"], exist_ok=True)

print("Configuration dictionary created:")
print(CONFIG_SYS)