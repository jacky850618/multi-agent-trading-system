# 实现运行后的多维度评估与学习闭环。
# 包含多种评估方法：
# LLM-as-a-Judge（主观评分）。
# Ground Truth（与真实市场表现对比）。
# Factual Consistency Audit（事实一致性审计，防幻觉）。

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import yfinance as yf
from datetime import datetime, timedelta
from .agents import deep_thinking_llm


# 从最终自然语言决策中提取干净的 BUY/SELL/HOLD 信号
class SignalProcessor:
    """信号处理器"""

    def __init__(self, llm):
        self.llm = llm

    def process_signal(self, full_signal: str) -> str:
        messages = [
            ("system",
             "您是一个助手，旨在从财务报告中提取最终的投资决策：SELL,BUY或HOLD。请仅以一个词来回答该决策。"),
            ("human", full_signal),
        ]
        result = self.llm.invoke(messages).content.strip().upper()
        if result in ["BUY", "SELL", "HOLD"]:
            return result
        return "ERROR_UNPARSABLE_SIGNAL"


# 根据实际盈亏进行反思，生成经验教训并存入智能体记忆（实现系统学习）。
class Reflector:
    """反思引擎"""

    def __init__(self, llm):
        self.llm = llm
        self.reflection_prompt = """您是一位资深的金融分析师。请回顾交易决策/分析、市场背景以及最终的财务结果。
        - 首先，根据结果判断决策是否正确。
        - 分析导致成功或失败的关键因素。
        - 最后，总结出一条简洁明了的经验教训或启发式方法，以便在类似情况下改进未来的决策。

        市场背景及分析： {situation}
        结果（盈利/亏损）： {returns_losses}"""

    def reflect(self, current_state, returns_losses, memory, component_key_func):
        situation = f"Reports: {current_state['market_report']} {current_state['sentiment_report']} {current_state['news_report']} {current_state['fundamentals_report']}\nDecision/Analysis Text: {component_key_func(current_state)}"
        prompt = self.reflection_prompt.format(situation=situation, returns_losses=returns_losses)
        result = self.llm.invoke(prompt).content
        memory.add_situations([(situation, result)])


class Evaluation(BaseModel):
    reasoning_quality: int = Field(description="Score 1-10 on the coherence and logic.")
    evidence_based_score: int = Field(description="Score 1-10 on citation of evidence from reports.")
    actionability_score: int = Field(description="Score 1-10 on how clear and actionable the decision is.")
    justification: str = Field(description="A brief justification for the scores.")


# 为评估模型创建一个提示模板。
# 该提示指示 LLM 像财务审计员一样工作。
evaluator_prompt = ChatPromptTemplate.from_template(
    """You are an expert financial auditor. Evaluate the 'Final Trading Decision' based on the provided 'Analyst Reports'.
    Analyst Reports:
    {reports}
    Final Trading Decision to Evaluate:
    {final_decision}
    """
)

# 将提示与 LLM 结合，通过评估模式强制执行结构化输出。
# 此处假定 `deep_thinking_llm` 是先前定义的 LLM 实例。
evaluator_chain = evaluator_prompt | deep_thinking_llm.with_structured_output(Evaluation)


def evaluate_ground_truth(ticker, trade_date, signal):
    try:
        # Import locally to avoid accidental shadowing of the datetime name
        from datetime import datetime, timedelta

        start_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        # Try a longer window to ensure we can find 5 trading days (markets have weekends/holidays)
        end_date = start_date + timedelta(days=14)

        data = yf.download(ticker, start=start_date.isoformat(), end=end_date.isoformat(), progress=False)

        # If initial window returns fewer than 5 trading days, expand to 30 days as a fallback
        if len(data) < 5:
            end_date = start_date + timedelta(days=30)
            data = yf.download(ticker, start=start_date.isoformat(), end=end_date.isoformat(), progress=False)

        if len(data) < 5:
            return f"Insufficient data for ground truth evaluation. Found only {len(data)} days."

        # Ensure the first row corresponds to the trade_date or the next trading day
        first_trading_day_index = 0
        while data.index[first_trading_day_index].date() < start_date:
            first_trading_day_index += 1
            if first_trading_day_index >= len(data) - 5:
                return "Could not align trade date."

        open_price = data['Open'].iloc[first_trading_day_index]
        close_price_5_days_later = data['Close'].iloc[first_trading_day_index + 4]
        performance = ((close_price_5_days_later - open_price) / open_price) * 100

        result = "INCORRECT DECISION"
        # Define success criteria: >1% for BUY, <-1% for SELL, within +/-1% for HOLD
        if (signal == "BUY" and performance > 1) or \
                (signal == "SELL" and performance < -1) or \
                (signal == "HOLD" and -1 <= performance <= 1):
            result = "CORRECT DECISION"

        return (
            f"----- Ground Truth Evaluation Report -----\n"
            f"Agent Signal: {signal} on {trade_date}\n"
            f"Opening Price on {data.index[first_trading_day_index].strftime('%Y-%m-%d')}: ${open_price:.2f}\n"
            f"Closing Price 5 days later ({data.index[first_trading_day_index + 4].strftime('%Y-%m-%d')}): ${close_price_5_days_later:.2f}\n"
            f"Actual Market Performance: {performance:+.2f}%\n"
            f"Evaluation Result: {result}"
        )
    except Exception as e:
        return f"Ground truth evaluation failed: {e}"


class Audit(BaseModel):
    is_consistent: bool = Field(description="Whether the report is factually consistent with the data.")
    discrepancies: list[str] = Field(description="A list of any identified discrepancies.")
    justification: str = Field(description="A brief justification for the audit result.")


auditor_prompt = ChatPromptTemplate.from_template(
    """You are an auditor. Compare the 'Agent Report' against the 'Raw Data' and check for factual consistency.
    Ignore differences in formatting or summarization, but flag any direct contradictions or claims in the report that are not supported by the data.

    Raw Data:
    {raw_data}

    Agent Report to Audit:
    {agent_report}
    """
)
auditor_chain = auditor_prompt | deep_thinking_llm.with_structured_output(Audit)
