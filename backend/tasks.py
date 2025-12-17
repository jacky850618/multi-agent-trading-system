# backend/tasks.py
from .storage import append_log, complete_task, task_storage
from .graph import trading_graph
from evaluation import SignalProcessor
from agents import quick_thinking_llm
from models import AgentState, InvestDebateState, RiskDebateState
from langchain_core.messages import HumanMessage


def run_analysis(task_id: str, ticker: str, trade_date: str):
    try:
        append_log(task_id, "正在初始化输入状态...")

        graph_input = AgentState(
            messages=[HumanMessage(content=f"Analyze {ticker} for trading on {trade_date}")],
            company_of_interest=ticker,
            trade_date=trade_date,
            investment_debate_state=InvestDebateState({
                'history': '', 'current_response': '', 'count': 0,
                'bull_history': '', 'bear_history': '', 'judge_decision': ''
            }),
            risk_debate_state=RiskDebateState({
                'history': '', 'latest_speaker': '', 'current_risky_response': '',
                'current_safe_response': '', 'current_neutral_response': '', 'count': 0,
                'risky_history': '', 'safe_history': '', 'neutral_history': '', 'judge_decision': ''
            })
        )

        append_log(task_id, "开始执行多代理工作流...")

        final_state = None

        # 节点图标美化映射
        node_icons = {
            "Market Analyst": "📈 市场分析师",
            "Social Analyst": "💬 社交媒体分析师",
            "News Analyst": "📰 新闻分析师",
            "Fundamentals Analyst": "📊 基本面分析师",
            "Bull Researcher": "🐂 多头研究员",
            "Bear Researcher": "🐻 空头研究员",
            "Research Manager": "👔 研究主管",
            "Trader": "💰 交易员",
            "Risky Analyst": "⚡ 激进风控分析师",
            "Safe Analyst": "🛡️ 保守风控分析师",
            "Neutral Analyst": "⚖️ 中立风控分析师",
            "Risk Judge": "⚖️ 投资组合经理（最终决策）",
            "tools": "🔧 工具调用",
            "Msg Clear": "🧹 消息清理",
        }

        for i, chunk in enumerate(trading_graph.stream(graph_input, {"recursion_limit": 100})):
            node_name = list(chunk.keys())[0]
            icon = node_icons.get(node_name, "▶️")
            append_log(task_id, f"{icon} [{i + 1}] {node_name}")
            final_state = chunk[node_name]

        # 提取信号
        processor = SignalProcessor(quick_thinking_llm)
        signal = processor.process_signal(final_state['final_trade_decision'])

        complete_task(task_id, final_state, signal)

    except Exception as e:
        append_log(task_id, f"错误：{str(e)}")
        if task_id in task_storage:
            task_storage[task_id]["status"] = "error"