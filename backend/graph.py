# 构建并编译完整的 LangGraph 工作流（核心调度引擎）。
import functools

from backend.config_user import get_user_config
from langgraph.graph import StateGraph, END
# 创建 StateGraph，将所有智能体节点、工具节点、消息清理节点连接起来。
# 定义条件路由逻辑（ConditionalLogic）：
# 分析师是否需要继续调用工具（ReAct 循环）。
# 多空辩论何时结束、转向经理。
# 风控辩论的轮转顺序和结束条件。
# 设置入口点、边（edges）和条件边，最终编译成可执行的 trading_graph。
# 这是整个系统的“大脑”，控制智能体协作的顺序和流转。

from .agents import *
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, RemoveMessage
from .models import AgentState
from .tools import Toolkit


# ConditionalLogic 类包含我们图的路由函数。
class ConditionalLogic:
    def __init__(self, max_debate_rounds=2, max_risk_discuss_rounds=1):
        # 存储配置中的最大轮数。
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    # 此函数决定分析智能体是应该继续还是调用工具。
    def should_continue_analyst(self, state: AgentState) -> str:
        # tools_condition 辅助函数检查状态中的最后一条消息。
        # 如果是工具调用，则返回"tools"。否则，返回"continue"。
        return "tools" if tools_condition(state) == "tools" else "continue"

    # 此函数控制投资辩论的流程。
    def should_continue_debate(self, state: AgentState) -> str:
        if state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds:
            return "Research Manager"
        return "Bear Researcher" if state["investment_debate_state"]["current_response"].startswith("Bull") \
            else "Bull Researcher"

    # 此函数控制风险管理讨论的流程。
    def should_continue_risk_analysis(self, state: AgentState) -> str:
        if state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds:
            return "Risk Judge"
        speaker = state["risk_debate_state"]["latest_speaker"]
        if speaker == "Risky Analyst": return "Safe Analyst"
        if speaker == "Safe Analyst": return "Neutral Analyst"
        return "Risky Analyst"


# 此函数创建一个节点，用于清除状态中的消息。
def create_msg_delete():
    # 此辅助函数旨在用作图中的一个节点。
    def delete_messages(state):
        # 我们使用RemoveMessage来指定要删除的消息。
        # 在这里，我们删除所有现有消息，并添加一条新的 HumanMessage 以继续流程。
        return {"messages": [RemoveMessage(id=m.id) for m in state["messages"]] + [HumanMessage(content="Continue")]}

    return delete_messages


# ==================== 核心工厂函数：为每个任务创建独立的 graph ====================
def create_trading_graph():
    """
        为每个并发任务创建一个全新的、独立的 trading_graph
        所有 toolkit、memory、节点都是独立的，避免状态污染
        """
    # 每个任务独立的工具包
    user_config = get_user_config()
    prompts = user_config["prompts"]

    toolkit = Toolkit()
    print(f"定义并实例化了包含实时数据工具的工具包类。")

    # 每个任务独立的记忆（关键！）
    memories = {
        "bull": FinancialSituationMemory(f"bull_memory_{id(toolkit)}"),  # 用唯一标识避免冲突
        "bear": FinancialSituationMemory(f"bear_memory_{id(toolkit)}"),
        "trader": FinancialSituationMemory(f"trader_memory_{id(toolkit)}"),
        "invest_judge": FinancialSituationMemory(f"invest_judge_memory_{id(toolkit)}"),
        "risk_manager": FinancialSituationMemory(f"risk_manager_memory_{id(toolkit)}"),
    }

    # 独立的工具节点
    all_tools = [
        toolkit.get_yfinance_data,
        toolkit.get_technical_indicators,
        toolkit.get_finnhub_news,
        toolkit.get_social_media_sentiment,
        toolkit.get_fundamental_analysis,
        toolkit.get_macroeconomic_news
    ]
    tool_node = ToolNode(all_tools)

    print(f"启用分析师节点...")
    # 独立的分析师节点
    market_analyst_node = create_analyst_node(
        quick_thinking_llm, toolkit,
        prompts["market_analyst"],
        [toolkit.get_yfinance_data, toolkit.get_technical_indicators],
        "market_report"
    )
    social_analyst_node = create_analyst_node(
        quick_thinking_llm, toolkit,
        prompts["social_analyst"],
        [toolkit.get_social_media_sentiment],
        "sentiment_report"
    )
    news_analyst_node = create_analyst_node(
        quick_thinking_llm, toolkit,
        prompts["news_analyst"],
        [toolkit.get_finnhub_news, toolkit.get_macroeconomic_news],
        "news_report"
    )
    fundamentals_analyst_node = create_analyst_node(
        quick_thinking_llm, toolkit,
        prompts["fundamentals_analyst"],
        [toolkit.get_fundamental_analysis],
        "fundamentals_report"
    )

    # 独立的消息清理节点
    msg_clear_node = create_msg_delete()

    print(f"启用研究员节点...")
    bull_researcher_node = create_researcher_node(quick_thinking_llm,
                                                  memories["bull"],
                                                  prompts["bull"],
                                                  "Bull Analyst")
    bear_researcher_node = create_researcher_node(quick_thinking_llm, memories["bear"],
                                                  prompts["bear"],
                                                  "Bear Analyst")
    research_manager_node = create_research_manager(deep_thinking_llm, memories["invest_judge"])

    print(f"启用交易员和风控节点...")
    trader_node = functools.partial(create_trader(quick_thinking_llm, memories["trader"]), name="Trader")
    risky_node = create_risk_debator(quick_thinking_llm, prompts["risky"],
                                     "Risky Analyst")
    safe_node = create_risk_debator(quick_thinking_llm, prompts["safe"],
                                    "Safe Analyst")
    neutral_node = create_risk_debator(quick_thinking_llm, prompts["neutral"],
                                       "Neutral Analyst")
    risk_manager_node = create_risk_manager(deep_thinking_llm, memories["risk_manager"])

    # 独立的条件逻辑
    conditional_logic = ConditionalLogic(
        max_debate_rounds=user_config['max_debate_rounds'],
        max_risk_discuss_rounds=user_config['max_risk_discuss_rounds']
    )

    print(f"开始构建Workflow...")

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("Market Analyst", market_analyst_node)
    workflow.add_node("Social Analyst", social_analyst_node)
    workflow.add_node("News Analyst", news_analyst_node)
    workflow.add_node("Fundamentals Analyst", fundamentals_analyst_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("Msg Clear", msg_clear_node)
    workflow.add_node("Bull Researcher", bull_researcher_node)
    workflow.add_node("Bear Researcher", bear_researcher_node)
    workflow.add_node("Research Manager", research_manager_node)
    workflow.add_node("Trader", trader_node)
    workflow.add_node("Risky Analyst", risky_node)
    workflow.add_node("Safe Analyst", safe_node)
    workflow.add_node("Neutral Analyst", neutral_node)
    workflow.add_node("Risk Judge", risk_manager_node)

    # 设置入口和边（与原逻辑完全一致）
    workflow.set_entry_point("Market Analyst")
    workflow.add_conditional_edges("Market Analyst", conditional_logic.should_continue_analyst,
                                   {"tools": "tools", "continue": "Msg Clear"})
    workflow.add_edge("tools", "Market Analyst")
    workflow.add_edge("Msg Clear", "Social Analyst")
    workflow.add_conditional_edges("Social Analyst", conditional_logic.should_continue_analyst,
                                   {"tools": "tools", "continue": "News Analyst"})
    workflow.add_edge("tools", "Social Analyst")
    workflow.add_conditional_edges("News Analyst", conditional_logic.should_continue_analyst,
                                   {"tools": "tools", "continue": "Fundamentals Analyst"})
    workflow.add_edge("tools", "News Analyst")
    workflow.add_conditional_edges("Fundamentals Analyst", conditional_logic.should_continue_analyst,
                                   {"tools": "tools", "continue": "Bull Researcher"})
    workflow.add_edge("tools", "Fundamentals Analyst")

    workflow.add_conditional_edges("Bull Researcher", conditional_logic.should_continue_debate)
    workflow.add_conditional_edges("Bear Researcher", conditional_logic.should_continue_debate)
    workflow.add_edge("Research Manager", "Trader")

    workflow.add_edge("Trader", "Risky Analyst")
    workflow.add_conditional_edges("Risky Analyst", conditional_logic.should_continue_risk_analysis)
    workflow.add_conditional_edges("Safe Analyst", conditional_logic.should_continue_risk_analysis)
    workflow.add_conditional_edges("Neutral Analyst", conditional_logic.should_continue_risk_analysis)
    workflow.add_edge("Risk Judge", END)

    print(f"开始编译Workflow...")
    return workflow.compile()


def draw_trading_graph(trading_graph):
    # 要进行可视化，需要安装 pygraphviz：`pip install pygraphviz`
    try:
        from IPython.display import Image, display
        # `get_graph()` 方法返回图表结构的表示。
        # `draw_png()` 方法将此结构渲染为 PNG 图像。
        png_image = trading_graph.get_graph().draw_png()
        return Image(png_image)
    except Exception as e:
        print(f"Graph visualization failed: {e}. Please ensure pygraphviz is installed.")
