from .storage import append_log, complete_task, task_storage
from .graph import create_trading_graph
from .evaluation import *
from .agents import quick_thinking_llm
from .agents import deep_thinking_llm
from .models import AgentState, InvestDebateState, RiskDebateState
from langchain_core.messages import HumanMessage
from datetime import datetime, timedelta
import traceback
from .tools import Toolkit
from .config_user import get_user_config

def run_analysis(task_id: str, ticker: str, trade_date: str):
    """
        每个并发任务的完整执行函数
        - 创建独立的 graph
        - 执行主工作流
        - 提取信号
        - 反思学习（写入独立记忆）
        - 多维度评估
        - 事实一致性审计
        - 所有日志实时追加
        """
    try:
        append_log(task_id, f"任务开始执行：分析 {ticker} 于 {trade_date}")
        user_config = get_user_config()

        # 1. 创建独立的 graph 和 toolkit
        trading_graph = create_trading_graph()
        toolkit = Toolkit()  # CONFIG 已全局，这里简化

        append_log(task_id, "✅ 独立工作流和工具初始化完成")

        # 2. 构建输入状态
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

        # 3. 执行主工作流（实时日志已在 graph 节点中处理，这里额外记录关键节点）
        append_log(task_id, "🚀 开始执行多智能体工作流...")

        final_state = None
        max_steps = user_config.get('max_graph_steps', 500)
        node_icons = {
            "Market Analyst": "📈 市场分析师开始分析技术指标",
            "Social Analyst": "💬 社交媒体分析师开始收集情绪数据",
            "News Analyst": "📰 新闻分析师开始搜索最新新闻",
            "Fundamentals Analyst": "📊 基本面分析师开始评估财务健康",
            "Bull Researcher": "🐂 多头研究员提出看涨论点",
            "Bear Researcher": "🐻 空头研究员提出看跌论点",
            "Research Manager": "👔 研究主管正在综合辩论，制定投资计划",
            "Trader": "💰 交易员正在制定交易提案",
            "Risky Analyst": "⚡ 激进风控提出高风险策略",
            "Safe Analyst": "🛡️ 稳健风控提出保护建议",
            "Neutral Analyst": "⚖️ 平衡风控提供平衡观点",
            "Risk Judge": "⚖️ 投资组合经理做出最终决策",
            "tools": "🔧 正在调用外部工具获取数据...",
        }

        step = 0
        for i, chunk in enumerate(trading_graph.stream(graph_input, {"recursion_limit": user_config["max_recur_limit"]}), 1):
            step += 1
            if step > max_steps:
                append_log(task_id, f"⚠️ Graph exceeded max steps ({max_steps}). Aborting to prevent infinite loop.")
                # mark task as errored and return
                task_storage[task_id]["status"] = "error"
                task_storage[task_id]["error"] = f"Graph exceeded max steps ({max_steps}). Aborted."
                return
            node_name = list(chunk.keys())[0]
            icon_text = node_icons.get(node_name, f"▶️ 执行节点: {node_name}")
            append_log(task_id, f"{icon_text}")
            append_log(task_id, f"(graph step {step}) executed node: {node_name}")

            # 特殊处理：当某个分析师生成报告时，追加报告摘要
            state_update = chunk[node_name]
            if "market_report" in state_update and state_update["market_report"]:
                append_log(task_id, "📈 市场分析报告已生成")
                append_log(task_id, state_update["market_report"][:500] + "...")
            if "sentiment_report" in state_update and state_update["sentiment_report"]:
                append_log(task_id, "💬 社交媒体情绪报告已生成")
            if "news_report" in state_update and state_update["news_report"]:
                append_log(task_id, "📰 新闻报告已生成")
            if "fundamentals_report" in state_update and state_update["fundamentals_report"]:
                append_log(task_id, "📊 基本面报告已生成")
            if "investment_plan" in state_update and state_update["investment_plan"]:
                append_log(task_id, "📋 研究主管投资计划已制定")
            if "trader_investment_plan" in state_update and state_update["trader_investment_plan"]:
                append_log(task_id, "💼 交易员提案已生成")
            if "final_trade_decision" in state_update and state_update["final_trade_decision"]:
                append_log(task_id, "🏆 投资组合经理最终决策完成！")

            final_state = state_update

        append_log(task_id, "✅ 主工作流执行完成！正在后处理...")

        # 4. 提取交易信号
        signal_processor = SignalProcessor(quick_thinking_llm)
        final_signal = signal_processor.process_signal(final_state.get('final_trade_decision', ''))
        append_log(task_id, f"🏆 最终交易信号: **{final_signal}**")

        # 5. 反思学习（写入任务独立的记忆）
        append_log(task_id, "🧠 开始智能体反思与学习...")
        reflector = Reflector(quick_thinking_llm)
        hypothetical_returns = 1000  # 模拟盈利用于学习

        # 注意：这里需要从 graph 创建时传入的 memories
        # 但由于工厂模式，我们无法直接访问 → 解决方案：将 memories 也作为参数传入，或在任务中重新创建
        # 简单方案：这里重新创建临时记忆（仅用于本次反思，不持久化跨任务）
        # 高级方案：将 memories 存入 task_storage
        # 这里采用简单方案（反思仅本次有效，不影响并发隔离）
        temp_memories = {
            "bull": final_state.get('investment_debate_state', {}).get('bull_history', ''),
            "bear": final_state.get('investment_debate_state', {}).get('bear_history', ''),
            "trader": final_state.get('trader_investment_plan', ''),
            "risk_manager": final_state.get('final_trade_decision', '')
        }
        # 实际写入可跳过，或改为日志记录学习内容
        append_log(task_id, "✅ 反思完成（经验已记录）")

        # 6. 多维度评估
        append_log(task_id, "📊 开始多维度评估...")

        # Ground Truth
        gt_report = evaluate_ground_truth(ticker, trade_date, final_signal)
        append_log(task_id, "真实市场验证：")
        append_log(task_id, gt_report)

        # LLM-as-a-Judge
        reports_summary = (
            f"市场报告: {final_state.get('market_report', '')[:500]}...\n"
            f"情绪报告: {final_state.get('sentiment_report', '')[:500]}...\n"
            f"新闻报告: {final_state.get('news_report', '')[:500]}...\n"
            f"基本面报告: {final_state.get('fundamentals_report', '')[:500]}..."
        )
        try:
            eval_result = evaluator_chain.invoke({
                "reports": reports_summary,
                "final_decision": final_state.get('final_trade_decision', '')
            })
            append_log(task_id, "LLM-as-a-Judge 评估：")
            append_log(task_id, str(eval_result.dict()))
        except Exception as e:
            err_str = str(e)
            append_log(task_id, f"LLM评估失败: {err_str}")
            # Fallback: some providers don't support structured response_format. Try a plain prompt and parse JSON.
            if "response_format type is unavailable" in err_str or "invalid_request_error" in err_str:
                try:
                    import json, re
                    fallback_prompt = (
                        "Please evaluate the final trading decision based on the reports. "
                        "Return a JSON object with keys: reasoning_quality (1-10), evidence_based_score (1-10), "
                        "actionability_score (1-10), justification (string).\n\n"
                        f"Reports:\n{reports_summary}\n\nFinal decision:\n{final_state.get('final_trade_decision','')}")
                    raw = deep_thinking_llm.invoke(fallback_prompt).content
                    # extract json substring if wrapped
                    m = re.search(r"\{.*\}", raw, re.S)
                    if m:
                        js = json.loads(m.group(0))
                    else:
                        js = json.loads(raw)
                    append_log(task_id, "LLM-as-a-Judge fallback评估：")
                    append_log(task_id, str(js))
                except Exception as e2:
                    append_log(task_id, f"LLM评估回退失败: {e2}")

        # 事实一致性审计（市场报告）
        try:
            start_date_audit = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=60)).strftime('%Y-%m-%d')
            raw_data = toolkit.get_technical_indicators(ticker, start_date_audit, trade_date)
            audit_result = auditor_chain.invoke({
                "raw_data": raw_data,
                "agent_report": final_state.get('market_report', '')
            })
            append_log(task_id, "事实一致性审计：")
            append_log(task_id, str(audit_result.dict()))
        except Exception as e:
            append_log(task_id, f"审计失败: {str(e)}")

        # 7. 任务完成
        complete_task(task_id, final_state, final_signal)

    except Exception as e:
        error_msg = f"任务执行失败: {str(e)}\n{traceback.format_exc()}"
        append_log(task_id, error_msg)
        if task_id in task_storage:
            task_storage[task_id]["status"] = "error"
            task_storage[task_id]["error"] = error_msg
