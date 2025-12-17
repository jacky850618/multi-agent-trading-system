# 程序的执行起点。
# 设置股票代码和交易日期。
# 初始化输入状态。
# 调用 trading_graph.stream() 运行完整工作流。
# 获取最终状态，提取信号，进行反思和评估。
# 相当于“一键运行整个智能交易公司”。

from rich.markdown import Markdown

from evaluation import *
from rich.console import Console
from graph import *
from evaluation import SignalProcessor, Reflector
from models import InvestDebateState, RiskDebateState, AgentState
from config import CONFIG

print("FinancialSituationMemory instances created for 5 agents.")

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
    graph_config = {"recursion_limit": CONFIG['max_recur_limit']}

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
    reflector.reflect(final_state, hypothetical_returns, bull_memory, lambda s: s['investment_debate_state']['bull_history'])

    print("正在反思和更新熊市研究员的记忆...")
    reflector.reflect(final_state, hypothetical_returns, bear_memory, lambda s: s['investment_debate_state']['bear_history'])

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
