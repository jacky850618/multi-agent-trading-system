# 定义 LangGraph 工作流的状态结构。
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


# 记录多空辩论的历史和轮次
# 研究团队辩论的状态，用作专门的草稿本。
class InvestDebateState(TypedDict):
    bull_history: str  # 存储多头智能体的论点。
    bear_history: str  # 存储空头智能体的论点。
    history: str  # 辩论的完整记录。
    current_response: str  # 最新提出的论点。
    judge_decision: str  # 经理的最终决定。
    count: int  # 用于跟踪辩论轮数的计数器。


# 记录风控团队（激进、稳健、平衡）的辩论历史和轮次
# 风险管理团队辩论的状态
class RiskDebateState(TypedDict):
    risky_history: str  # 激进型风险承担者的历史记录。
    safe_history: str  # 稳健型智能体的历史记录。
    neutral_history: str  # 平衡型智能体的历史记录。
    history: str  # 风险讨论的完整记录。
    latest_speaker: str  # 跟踪最后一位发言的智能体。
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str
    judge_decision: str  # 投资组合经理的最终决定。
    count: int  # 风险讨论轮次的计数器。


# 将在整个图中传递的主要状态, 继承自 LangGraph 的 MessagesState，包含所有分析师报告、辩论状态、投资计划、最终决策等字段。
class AgentState(MessagesState):
    company_of_interest: str  # 我们正在分析的股票代码。
    trade_date: str  # 分析日期。
    sender: str  # 跟踪哪个智能体最后修改了状态。
    # 每个分析师都会填充自己的报告字段。
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    # 用于辩论的嵌套状态。
    investment_debate_state: InvestDebateState
    investment_plan: str  # 研究经理的计划。
    trader_investment_plan: str  # 交易员的可执行计划。
    risk_debate_state: RiskDebateState
    final_trade_decision: str  # 投资组合经理的最终决策。
