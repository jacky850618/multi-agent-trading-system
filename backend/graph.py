# 构建并编译完整的 LangGraph 工作流（核心调度引擎）。
import functools

from langgraph.graph import StateGraph, END
# 创建 StateGraph，将所有代理节点、工具节点、消息清理节点连接起来。
# 定义条件路由逻辑（ConditionalLogic）：
# 分析师是否需要继续调用工具（ReAct 循环）。
# 多空辩论何时结束、转向经理。
# 风控辩论的轮转顺序和结束条件。
# 设置入口点、边（edges）和条件边，最终编译成可执行的 trading_graph。
# 这是整个系统的“大脑”，控制代理协作的顺序和流转。

from .agents import *
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, RemoveMessage
from models import AgentState
from tools import Toolkit


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



toolkit = Toolkit(CONFIG)
print(f"定义并实例化了包含实时数据工具的工具包类。")


all_tools = [
    toolkit.get_yfinance_data,
    toolkit.get_technical_indicators,
    toolkit.get_finnhub_news,
    toolkit.get_social_media_sentiment,
    toolkit.get_fundamental_analysis,
    toolkit.get_macroeconomic_news
]
tool_node = ToolNode(all_tools)

# 市场分析师：专注于技术指标和价格走势。
market_analyst_system_message = "您是一位专门分析金融市场的交易助理。您的职责是选择最相关的技术指标来分析股票的价格走势、动量和波动性。您必须使用工具获取历史数据，然后生成一份包含分析结果的报告，其中包括一个汇总表。"
market_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    market_analyst_system_message,
    [toolkit.get_yfinance_data, toolkit.get_technical_indicators],
    "market_report"
)

# 社交媒体分析师：评估公众情绪。
social_analyst_system_message = "您是一名社交媒体分析师。您的工作是分析过去一周内特定公司的社交媒体帖子和公众情绪。使用您的工具查找相关讨论，并撰写一份全面的报告，详细说明您的分析、见解以及对交易者的影响，包括一份汇总表。"
social_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    social_analyst_system_message,
    [toolkit.get_social_media_sentiment],
    "sentiment_report"
)

# 新闻分析师：负责公司相关新闻和宏观经济新闻。
news_analyst_system_message = "您是一名新闻研究员，负责分析过去一周的最新新闻和趋势。请撰写一份关于当前世界形势的综合报告，内容需与交易和宏观经济相关。请使用您的工具提供全面、详细的分析，包括汇总表。"
news_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    news_analyst_system_message,
    [toolkit.get_finnhub_news, toolkit.get_macroeconomic_news],
    "news_report"
)

# 基本面分析师：深入分析公司的财务状况。
fundamentals_analyst_system_message = "您是一名研究员，正在分析公司的基本面信息。请撰写一份关于公司财务状况、内部人士情绪和交易情况的综合报告，以全面了解其基本面状况，并附上汇总表。"
fundamentals_analyst_node = create_analyst_node(
    quick_thinking_llm,
    toolkit,
    fundamentals_analyst_system_message,
    [toolkit.get_fundamental_analysis],
    "fundamentals_report"
)

# 创建可调用的消息清除节点。
msg_clear_node = create_msg_delete()


# 为每个学习的智能体创建一个专用的内存实例。
bull_memory = FinancialSituationMemory("bull_memory", CONFIG)
bear_memory = FinancialSituationMemory("bear_memory", CONFIG)
trader_memory = FinancialSituationMemory("trader_memory", CONFIG)
invest_judge_memory = FinancialSituationMemory("invest_judge_memory", CONFIG)
risk_manager_memory = FinancialSituationMemory("risk_manager_memory", CONFIG)

# 研究员
# 多头/空头研究员
bull_prompt = "您是一位多头分析师。您的目标是论证投资该股票的合理性。请重点关注增长潜力、竞争优势以及报告中的积极指标。有效反驳看跌分析师的论点。"
bear_prompt = "您是一位空头分析师。您的目标是论证投资该股票的不合理性。请重点关注风险、挑战以及负面指标。有效反驳看涨分析师的论点。"

# 使用我们的工厂函数创建可调用节点。
bull_researcher_node = create_researcher_node(quick_thinking_llm, bull_memory, bull_prompt, "Bull Analyst")
bear_researcher_node = create_researcher_node(quick_thinking_llm, bear_memory, bear_prompt, "Bear Analyst")

# 创建可调用的研究员主管节点。
research_manager_node = create_research_manager(deep_thinking_llm, invest_judge_memory)
print("研究员和研究员主管智能体创建功能现已可用。")

# 创建交易者节点。我们使用 functools.partial 预先填充 'name' 参数。
trader_node_func = create_trader(quick_thinking_llm, trader_memory)
trader_node = functools.partial(trader_node_func, name="Trader")

# “冒险型”风险分析师主张最大化收益，即使这意味着更高的风险。
risky_prompt = "您是冒险型风险分析师。您主张高回报机会和大胆策略。"

# “稳健型”风险分析师将资本保值放在首位。
safe_prompt = "您是稳健 / 保守型风险分析师。您优先考虑资本保值和最小化波动性。"

# “平衡型”风险分析师提供平衡、客观的视角。
neutral_prompt = "您是平衡型风险分析师。您提供平衡的视角，权衡收益和风险。"

# 使用各自的提示创建三个风险辩论者节点。
risky_node = create_risk_debator(quick_thinking_llm, risky_prompt, "Risky Analyst")
safe_node = create_risk_debator(quick_thinking_llm, safe_prompt, "Safe Analyst")
neutral_node = create_risk_debator(quick_thinking_llm, neutral_prompt, "Neutral Analyst")

# 创建可调用的投资组合经理节点。
risk_manager_node = create_risk_manager(deep_thinking_llm, risk_manager_memory)

# 使用中央配置中的值实例化逻辑类。
conditional_logic = ConditionalLogic(
    max_debate_rounds=CONFIG['max_debate_rounds'],
    max_risk_discuss_rounds=CONFIG['max_risk_discuss_rounds']
)

def build_workflow():
    # 使用我们的主 AgentState 初始化一个新的 StateGraph。
    workflow = StateGraph(AgentState)

    # --- 将所有节点添加到图中 ---
    # 添加分析师团队节点
    workflow.add_node("Market Analyst", market_analyst_node)
    workflow.add_node("Social Analyst", social_analyst_node)
    workflow.add_node("News Analyst", news_analyst_node)
    workflow.add_node("Fundamentals Analyst", fundamentals_analyst_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("Msg Clear", msg_clear_node)

    # 添加研究员团队节点
    workflow.add_node("Bull Researcher", bull_researcher_node)
    workflow.add_node("Bear Researcher", bear_researcher_node)
    workflow.add_node("Research Manager", research_manager_node)

    # 添加交易员和风险团队节点
    workflow.add_node("Trader", trader_node)
    workflow.add_node("Risky Analyst", risky_node)
    workflow.add_node("Safe Analyst", safe_node)
    workflow.add_node("Neutral Analyst", neutral_node)
    workflow.add_node("Risk Judge", risk_manager_node)

    # --- 使用边将节点连接起来 ---
    # 设置整个工作流的入口点。
    workflow.set_entry_point("Market Analyst")

    # 定义分析师团队的顺序流程和 ReAct 循环。
    workflow.add_conditional_edges("Market Analyst", conditional_logic.should_continue_analyst,
                                   {"tools": "tools", "continue": "Msg Clear"})
    workflow.add_edge("tools",
                      "Market Analyst")  # After a tool call, loop back to the analyst for it to reason about the new data.
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

    # 定义研究员辩论循环。
    workflow.add_conditional_edges("Bull Researcher", conditional_logic.should_continue_debate)
    workflow.add_conditional_edges("Bear Researcher", conditional_logic.should_continue_debate)
    workflow.add_edge("Research Manager", "Trader")

    # 定义风险辩论循环。
    workflow.add_edge("Trader", "Risky Analyst")
    workflow.add_conditional_edges("Risky Analyst", conditional_logic.should_continue_risk_analysis)
    workflow.add_conditional_edges("Safe Analyst", conditional_logic.should_continue_risk_analysis)
    workflow.add_conditional_edges("Neutral Analyst", conditional_logic.should_continue_risk_analysis)

    # 定义到工作流末尾的最终边。
    workflow.add_edge("Risk Judge", END)

    return workflow

def draw_workflow(graph):
    # 要进行可视化，需要安装 pygraphviz：`pip install pygraphviz`
    try:
        from IPython.display import Image, display

        # `get_graph()` 方法返回图表结构的表示。
        # `draw_png()` 方法将此结构渲染为 PNG 图像。
        png_image = graph.get_graph().draw_png()
        return Image(png_image)
    except Exception as e:
        print(f"Graph visualization failed: {e}. Please ensure pygraphviz is installed.")

trading_workflow = build_workflow()
trading_graph = trading_workflow.compile()
print("Graph compiled successfully.")
