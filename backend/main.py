# 程序的执行起点。
# 设置股票代码和交易日期。
# 初始化输入状态。
# 调用 trading_graph.stream() 运行完整工作流。
# 获取最终状态，提取信号，进行反思和评估。
# 相当于“一键运行整个智能交易公司”。

import datetime
import functools

from langgraph.graph import StateGraph, END
from rich.markdown import Markdown
from evaluation import *
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console
from graph import ConditionalLogic, create_msg_delete
from evaluation import SignalProcessor, Reflector
from agents import *
from models import InvestDebateState, RiskDebateState
from tools import Toolkit
from config_sys import CONFIG_SYS
from config_user import get_user_config
from memory import FinancialSituationMemory

toolkit = Toolkit()
print(f"定义并实例化了包含实时数据工具的工具包类。")

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

# 为每个学习的智能体创建一个专用的内存实例。
bull_memory = FinancialSituationMemory("bull_memory")
bear_memory = FinancialSituationMemory("bear_memory")
trader_memory = FinancialSituationMemory("trader_memory")
invest_judge_memory = FinancialSituationMemory("invest_judge_memory")
risk_manager_memory = FinancialSituationMemory("risk_manager_memory")

print("FinancialSituationMemory instances created for 5 agents.")

all_tools = [
    toolkit.get_yfinance_data,
    toolkit.get_technical_indicators,
    toolkit.get_finnhub_news,
    toolkit.get_social_media_sentiment,
    toolkit.get_fundamental_analysis,
    toolkit.get_macroeconomic_news
]

# 初始化一个用于富文本打印的控制台。
console = Console()


# 运行单个分析师的 ReAct 循环的辅助函数。
def run_analyst(analyst_node, initial_state):
    state = initial_state
    # 从我们的工具包实例中获取所有可用的工具。
    all_tools_in_toolkit = [getattr(toolkit, name) for name in dir(toolkit) if
                            callable(getattr(toolkit, name)) and not name.startswith("__")]
    # ToolNode 是一个特殊的 LangGraph 节点，用于执行工具调用。
    tool_node = ToolNode(all_tools_in_toolkit)
    # ReAct 循环最多可以包含 5 个推理和工具调用步骤。
    for _ in range(5):
        result = analyst_node(state)
        # tools_condition 检查 LLM 的最后一条消息是否是工具调用。
        if tools_condition(result) == "tools":
            # 如果是，则执行工具并更新状态。
            state = tool_node.invoke(result)
        else:
            # 如果不是，则智能体已完成，因此我们跳出循环。
            state = result
            break
    return state


if __name__ == "__main__":
    TICKER = "NVDA"
    # 使用最近的日期获取实时数据
    TRADE_DATE = (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')

    # 定义图表运行的初始状态。
    initial_state = AgentState(
        messages=[HumanMessage(content=f"Analyze {TICKER} for trading on {TRADE_DATE}")],
        company_of_interest=TICKER,
        trade_date=TRADE_DATE,
        # 使用默认空值初始化辩论状态。
        investment_debate_state=InvestDebateState(
            {'history': '', 'current_response': '', 'count': 0, 'bull_history': '', 'bear_history': '',
             'judge_decision': ''}),
        risk_debate_state=RiskDebateState(
            {'history': '', 'latest_speaker': '', 'current_risky_response': '', 'current_safe_response': '',
             'current_neutral_response': '', 'count': 0, 'risky_history': '', 'safe_history': '', 'neutral_history': '',
             'judge_decision': ''})
    )

    # 运行市场分析
    print("运行市场分析...")
    market_analyst_result = run_analyst(market_analyst_node, initial_state)
    initial_state['market_report'] = market_analyst_result.get('market_report', 'Failed to generate report.')
    console.print("----- 市场分析报告 -----")
    console.print(Markdown(initial_state['market_report']))

    # 运行社交媒体分析
    print("\n运行社交媒体情绪分析...")
    social_analyst_result = run_analyst(social_analyst_node, initial_state)
    initial_state['sentiment_report'] = social_analyst_result.get('sentiment_report', 'Failed to generate report.')
    console.print("----- 社交媒体情绪分析报告 -----")
    console.print(Markdown(initial_state['sentiment_report']))

    # 运行新闻分析
    print("\n运行新闻分析...")
    news_analyst_result = run_analyst(news_analyst_node, initial_state)
    initial_state['news_report'] = news_analyst_result.get('news_report', 'Failed to generate report.')
    console.print("----- 新闻分析报告 -----")
    console.print(Markdown(initial_state['news_report']))

    # 运行基本面分析
    print("\n运行基本面分析...")
    fundamentals_analyst_result = run_analyst(fundamentals_analyst_node, initial_state)
    initial_state['fundamentals_report'] = fundamentals_analyst_result.get('fundamentals_report',
                                                                           'Failed to generate report.')
    console.print("----- 基本面分析报告 -----")
    console.print(Markdown(initial_state['fundamentals_report']))

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

    # 我们将使用分析师部分末尾的状态。
    current_state = initial_state

    user_config = get_user_config()
    # 循环遍历配置中定义的辩论轮数。
    for i in range(user_config['max_debate_rounds']):
        print(f"--- 投资辩论环节 {i + 1} ---")
        # 多头先发言。
        bull_result = bull_researcher_node(current_state)
        current_state['investment_debate_state'] = bull_result['investment_debate_state']
        console.print("\n**多头观点:**")
        # 我们解析回复，只打印新的论点
        console.print(
            Markdown(current_state['investment_debate_state']['current_response'].replace('Bull Analyst: ', '')))

        # 然后，空头反驳
        bear_result = bear_researcher_node(current_state)
        current_state['investment_debate_state'] = bear_result['investment_debate_state']
        console.print("\n**空头反驳:**")
        console.print(
            Markdown(current_state['investment_debate_state']['current_response'].replace('Bear Analyst: ', '')))
        print("\n")

    # 循环结束后，将最终的辩论状态存储回主初始状态
    initial_state['investment_debate_state'] = current_state['investment_debate_state']

    print("执行研究员主管...")
    # 研究员主管接收包含完整辩论历史的最终状态。
    manager_result = research_manager_node(initial_state)
    # 研究员主管的输出存储在 'investment_plan' 字段中。
    initial_state['investment_plan'] = manager_result['investment_plan']

    console.print("\n----- 研究员主管的投资计划 -----")
    console.print(Markdown(initial_state['investment_plan']))

    # “冒险型”风险分析师主张最大化收益，即使这意味着更高的风险。
    risky_prompt = "您是冒险型风险分析师。您主张高回报机会和大胆策略。"

    # “稳健型”风险分析师将资本保值放在首位。
    safe_prompt = "您是稳健 / 保守型风险分析师。您优先考虑资本保值和最小化波动性。"

    # “平衡型”风险分析师提供平衡、客观的视角。
    neutral_prompt = "您是平衡型风险分析师。您提供平衡的视角，权衡收益和风险。"

    # 创建交易者节点。我们使用 functools.partial 预先填充 'name' 参数。
    trader_node_func = create_trader(quick_thinking_llm, trader_memory)
    trader_node = functools.partial(trader_node_func, name="Trader")

    # 使用各自的提示创建三个风险辩论者节点。
    risky_node = create_risk_debator(quick_thinking_llm, risky_prompt, "Risky Analyst")
    safe_node = create_risk_debator(quick_thinking_llm, safe_prompt, "Safe Analyst")
    neutral_node = create_risk_debator(quick_thinking_llm, neutral_prompt, "Neutral Analyst")

    print("交易员和风险管理智能体功能已经准备好。")

    print("运行交易员...")
    trader_result = trader_node(initial_state)
    initial_state['trader_investment_plan'] = trader_result['trader_investment_plan']

    console.print("----- 交易员提案 -----")
    console.print(Markdown(initial_state['trader_investment_plan']))

    print("--- 风险管理辩论赛, 第1轮 ---")

    risk_state = initial_state
    # 我们按照配置中指定的轮数（当前为 1）进行辩论。
    for _ in range(user_config['max_risk_discuss_rounds']):
        # 冒险型分析师首先发言。
        risky_result = risky_node(risk_state)
        risk_state['risk_debate_state'] = risky_result['risk_debate_state']
        console.print("\n**冒险分析师观点:**")
        console.print(Markdown(risk_state['risk_debate_state']['current_risky_response']))

        # 然后是稳健型分析师。
        safe_result = safe_node(risk_state)
        risk_state['risk_debate_state'] = safe_result['risk_debate_state']
        console.print("\n**稳健分析师观点:**")
        console.print(Markdown(risk_state['risk_debate_state']['current_safe_response']))

        # 最后，平衡分析师。
        neutral_result = neutral_node(risk_state)
        risk_state['risk_debate_state'] = neutral_result['risk_debate_state']
        console.print("\n**平衡分析师观点:**")
        console.print(Markdown(risk_state['risk_debate_state']['current_neutral_response']))

    # 使用最终辩论记录更新主状态。
    initial_state['risk_debate_state'] = risk_state['risk_debate_state']

    # 创建可调用的投资组合经理节点。
    risk_manager_node = create_risk_manager(deep_thinking_llm, risk_manager_memory)

    print("正在运行投资组合经理进行最终决策...")

    # 投资组合经理经理会收到包含交易员计划和完整风险讨论的最终状态。
    risk_manager_result = risk_manager_node(initial_state)

    # 投资组合经理经理的输出存储在'final_trade_decision'字段中。
    initial_state['final_trade_decision'] = risk_manager_result['final_trade_decision']

    console.print(" \n-----投资组合经理的最终决策-----")
    console.print(Markdown(initial_state['final_trade_decision']))

    # 使用中央配置中的值实例化逻辑类。
    conditional_logic = ConditionalLogic(
        max_debate_rounds=user_config['max_debate_rounds'],
        max_risk_discuss_rounds=user_config['max_risk_discuss_rounds']
    )

    # 创建可调用的消息清除节点。
    msg_clear_node = create_msg_delete()

    # ToolNode 是一个预构建的 LangGraph 节点，它接受一个工具列表，
    # 并执行在智能体的上一条消息中找到的任何工具调用。
    tool_node = ToolNode(all_tools)

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

    # `compile()` 方法最终确定图表并使其可以执行。
    trading_graph = workflow.compile()
    print("Graph compiled successfully.")

    # 要进行可视化，需要安装 pygraphviz：`pip install pygraphviz`
    try:
        from IPython.display import Image, display

        # `get_graph()` 方法返回图表结构的表示。
        # `draw_png()` 方法将此结构渲染为 PNG 图像。
        png_image = trading_graph.get_graph().draw_png()
        display(Image(png_image))
    except Exception as e:
        print(f"Graph visualization failed: {e}. Please ensure pygraphviz is installed.")

    # 为了保持一致性，我们将使用与手动测试中相同的股票代码和日期。
    graph_input = AgentState(
        messages=[HumanMessage(content=f"Analyze {TICKER} for trading on {TRADE_DATE}")],
        company_of_interest=TICKER,
        trade_date=TRADE_DATE,
        # 使用默认空值初始化辩论状态，以确保干净的开始。
        investment_debate_state=InvestDebateState(
            {'history': '', 'current_response': '', 'count': 0, 'bull_history': '', 'bear_history': '',
             'judge_decision': ''}),
        risk_debate_state=RiskDebateState(
            {'history': '', 'latest_speaker': '', 'current_risky_response': '', 'current_safe_response': '',
             'current_neutral_response': '', 'count': 0, 'risky_history': '', 'safe_history': '', 'neutral_history': '',
             'judge_decision': ''})
    )

    print(f"Running full analysis for {TICKER} on {TRADE_DATE}")

    final_state = None
    print("\n--- Invoking Graph Stream ---")
    # Set the recursion limit from our config, a safety measure for complex graphs.
    graph_config = {"recursion_limit": user_config['max_recur_limit']}

    # The .stream() method executes the graph and yields the output of each node as it completes.
    for chunk in trading_graph.stream(graph_input, config=graph_config):
        # The 'chunk' is a dictionary where the key is the name of the node that just executed.
        node_name = list(chunk.keys())[0]
        print(f"Executing Node: {node_name}")
        # We keep track of the final state to analyze it after the run.
        final_state = chunk[node_name]
    print("\n--- Graph Stream Finished ---")

    console.print("\n----- Final Raw Output from Portfolio Manager -----")
    console.print(Markdown(final_state['final_trade_decision']))

    # 智能体反思
    # 使用我们的 quick_thinking_llm 实例化处理器。
    signal_processor = SignalProcessor(quick_thinking_llm)
    final_signal = signal_processor.process_signal(final_state['final_trade_decision'])
    print(f"Extracted Signal: {final_signal}")

    print("基于 1000 美元的假设利润模拟反思...")
    reflector = Reflector(quick_thinking_llm)
    hypothetical_returns = 1000

    # 为每个具有记忆的智能体理运行反思过程。
    print("正在反思和更新牛市研究员的记忆...")
    reflector.reflect(final_state, hypothetical_returns, bull_memory,
                      lambda s: s['investment_debate_state']['bull_history'])

    print("正在反思和更新熊市研究员的记忆...")
    reflector.reflect(final_state, hypothetical_returns, bear_memory,
                      lambda s: s['investment_debate_state']['bear_history'])

    print("正在反思和更新交易员的记忆...")
    reflector.reflect(final_state, hypothetical_returns, trader_memory, lambda s: s['trader_investment_plan'])

    print("正在反思和更新风险管理者的记忆...")
    reflector.reflect(final_state, hypothetical_returns, risk_manager_memory, lambda s: s['final_trade_decision'])

    # 用法示例：事后评估代理的信号是否正确
    ground_truth_report = evaluate_ground_truth(TICKER, TRADE_DATE, final_signal)
    print(ground_truth_report)

    # 根据 final_state 构建分析师报告的全文摘要。
    reports_summary = (
        f"Market: {final_state['market_report']}\n"
        f"Sentiment: {final_state['sentiment_report']}\n"
        f"News: {final_state['news_report']}\n"
        f"Fundamentals: {final_state['fundamentals_report']}"
    )

    # 准备评估器输入，包含报告和待评估的决策。
    eval_input = {
        "reports": reports_summary,
        "final_decision": final_state['final_trade_decision']
    }

    # 运行评估器链 — 返回结构化的评估对象。
    evaluation_result = evaluator_chain.invoke(eval_input)

    # 以可读格式打印评估报告。
    print("----- LLM-as-a-Judge Evaluation Report -----")
    print(evaluation_result.dict())

    # 用法示例：事后评估代理的信号是否正确
    ground_truth_report = evaluate_ground_truth(TICKER, TRADE_DATE, final_signal)
    print(ground_truth_report)

    # 提取交易日前约 60 天的技术指标数据，以了解交易背景。
    start_date_audit = (datetime.strptime(TRADE_DATE, "%Y-%m-%d") - timedelta(days=60)).strftime('%Y-%m-%d')

    raw_market_data_for_audit = toolkit.get_technical_indicators(TICKER, start_date_audit, TRADE_DATE)

    # 为审计员准备输入资料（原始技术数据 + 代理人的叙述性报告）
    audit_input = {
        "raw_data": raw_market_data_for_audit,
        "agent_report": final_state['market_report']
    }

    # 运行链 → 结构化审计输出
    audit_result = auditor_chain.invoke(audit_input)

    # 以美观的方式打印审计结果
    print("----- Factual Consistency Audit Report -----")
    print(audit_result.dict())
