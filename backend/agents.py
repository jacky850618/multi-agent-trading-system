# 定义所有智能智能体（Agent）的逻辑和节点函数。
# 包含创建各种智能体的工厂函数：
# 四个分析师（市场、社交媒体、新闻、基本面）。
# 多空研究员（Bull/Bear Researcher）。
# 研究员主管（Research Manager）。
# 交易员（Trader）。
# 风控三方辩手（Risky/Safe/Neutral Analyst）。
# 风控经理/最终决策者（Risk Manager）。
#
# 每个智能体使用特定的 Prompt + LLM + 工具/记忆，实现其专业角色和决策行为。

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .config_sys import CONFIG_SYS
from .models import AgentState
from .memory import FinancialSituationMemory

# 初始化功能强大的 LLM，用于高风险推理任务。
deep_thinking_llm = ChatOpenAI(
    model=CONFIG_SYS["deep_think_llm"],
    base_url=CONFIG_SYS["backend_url"],
    temperature=0.1
)

# 初始化速度更快、成本更低的 LLM，用于常规数据处理。
quick_thinking_llm = ChatOpenAI(
    model=CONFIG_SYS["quick_think_llm"],
    base_url=CONFIG_SYS["backend_url"],
    temperature=0.1
)


# 此函数是一个工厂，用于为特定类型的分析师创建一个 LangGraph 节点。
def create_analyst_node(llm, toolkit, system_message, tools, output_field):
    """
        Creates a node for an analyst agent.
        Args:
            llm: 智能体要使用的语言模型实例。
            toolkit: 智能体可用的工具集合。
            system_message: 定义智能体角色和目标的具体指令。
            tools: 此智能体可以使用的工具包中的特定工具列表。
            output_field: AgentState中用于存储此智能体最终报告的键。
        """

    """创建分析师节点"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "您是一位乐于助人的AI助手，与其他助手协作。使用提供的工具来逐步回答问题。"
         " 如果您无法完全回答，没关系；另一位拥有不同工具的助手会从您中断的地方继续提供帮助。尽您所能，继续前进。"
         " 您可以使用以下工具：{tool_names}。\n{system_message}"
         " 供您参考，当前日期为{current_date}。我们要查看的公司是{ticker}。"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    chain = prompt | llm.bind_tools(tools)

    def analyst_node(state: AgentState):
        # 传入完整的消息历史 + 动态变量
        result = chain.invoke({
            "messages": state["messages"],
            "current_date": state["trade_date"],
            "ticker": state["company_of_interest"]
        })

        # 现在 result 是 AIMessage，才有 tool_calls 属性
        report = ""
        if hasattr(result, "tool_calls") and result.tool_calls:
            report = ""  # 有 tool_calls，继续 ReAct 循环，不生成报告
        else:
            report = result.content  # 最终回答，生成报告

        return {"messages": [result], output_field: report}

    return analyst_node


# 此函数是一个工厂，用于为研究者智能体（牛市或熊市）创建一个 LangGraph 节点。
def create_researcher_node(llm, memory, role_prompt, agent_name):
    def researcher_node(state):
        # 首先，将所有分析师报告合并成一个摘要，以便提供上下文。
        situation_summary = f"""
        市场分析报告: {state['market_report']}
        社交媒体情绪分析报告: {state['sentiment_report']}
        新闻分析报告: {state['news_report']}
        基本面分析报告: {state['fundamentals_report']}
        """
        past_memories = memory.get_memories(situation_summary)
        past_memory_str = "\n".join([mem['recommendation'] for mem in past_memories])

        prompt = f"""{role_prompt}
        以下是当前的分析状态：
        {situation_summary}
        对话历史：{state['investment_debate_state']['history']}
        对方的最后论点：{state['investment_debate_state']['current_response']}
        对类似过往情境的反思：{past_memory_str or '未找到过往记忆'}
        基于以上信息，以对话的形式陈述你的论点。"""

        # 调用 LLM 生成论点。
        response = llm.invoke(prompt)
        argument = f"{agent_name}: {response.content}"

        # 使用新论点更新辩论状态。
        debate_state = state['investment_debate_state'].copy()
        debate_state['history'] += "\n" + argument

        # 更新此智能体（多头或空头）的特定历史记录。
        if agent_name == 'Bull Analyst':
            debate_state['bull_history'] += "\n" + argument
        else:
            debate_state['bear_history'] += "\n" + argument
        debate_state['current_response'] = argument
        debate_state['count'] += 1
        return {"investment_debate_state": debate_state}

    return researcher_node


# 此函数创建研究员主管节点。
def create_research_manager(llm, memory: FinancialSituationMemory):
    """创建研究员主管节点"""

    def research_manager_node(state: AgentState) -> dict:
        prompt = f"""作为研究员主管，您的职责是批判性地评估多空分析师之间的辩论，并做出明确的决策。
            总结要点，然后给出明确的建议：买入、卖出或持有。为交易者制定详细的投资计划，包括您的投资逻辑和策略行动。
            辩论历史：
            {state['investment_debate_state']['history']}"""
        response = llm.invoke(prompt)

        # 输出是最终的投资计划，将传递给交易员。
        return {"investment_plan": response.content}

    return research_manager_node


# 此函数创建交易智能体节点。
def create_trader(llm, memory):
    def trader_node(state, name):
        # 提示很简单：根据计划创建交易建议。
        # 关键指令是必须的 final 标签。
        prompt = f"""您是一名交易代理。根据提供的投资计划，创建一个简洁的交易建议。
        您的回复必须以“最终交易建议：**BUY/HOLD/SELL**”结尾'.

        建议的投资计划： {state['investment_plan']}"""
        result = llm.invoke(prompt)

        # 输出使用交易员的计划更新状态并标识发送者。
        return {"trader_investment_plan": result.content, "sender": name}

    return trader_node


# 此函数是创建风险辩论者节点的工厂。
def create_risk_debator(llm, role_prompt, agent_name):
    def risk_debator_node(state):
        # 首先，从状态中获取其他两个辩论者的论点。
        risk_state = state['risk_debate_state']
        opponents_args = []
        if agent_name != 'Risky Analyst' and risk_state['current_risky_response']: opponents_args.append(
            f"Risky: {risk_state['current_risky_response']}")
        if agent_name != 'Safe Analyst' and risk_state['current_safe_response']: opponents_args.append(
            f"Safe: {risk_state['current_safe_response']}")
        if agent_name != 'Neutral Analyst' and risk_state['current_neutral_response']: opponents_args.append(
            f"Neutral: {risk_state['current_neutral_response']}")

        # 使用交易者的计划、辩论历史构建提示信息以及对手的论点。
        prompt = f"""{role_prompt}
        以下是交易者的计划：{state['trader_investment_plan']}
        辩论历史：{risk_state['history']}
        对手的最后论点：\n {'\n'.join(opponents_args)}
        请从您的角度评价或支持该计划。"""

        response = llm.invoke(prompt).content

        # 使用新论点更新风险辩论状态。
        new_risk_state = risk_state.copy()
        new_risk_state['history'] += f"\n{agent_name}: {response}"
        new_risk_state['latest_speaker'] = agent_name

        # 将响应存储在此智能体智能体的特定字段中。
        if agent_name == 'Risky Analyst':
            new_risk_state['current_risky_response'] = response
        elif agent_name == 'Safe Analyst':
            new_risk_state['current_safe_response'] = response
        else:
            new_risk_state['current_neutral_response'] = response
        new_risk_state['count'] += 1
        return {"risk_debate_state": new_risk_state}

    return risk_debator_node


# 此函数创建投资组合经理节点。
def create_risk_manager(llm, memory):
    def risk_manager_node(state):
        prompt = f"""作为投资组合经理，您的决定是最终的。请查看交易员的计划和风险讨论。
        请提供最终的、具有约束力的决定：买入、卖出或持有，并简要说明理由。
        交易员计划：{state['trader_investment_plan']}
        风险讨论：{state['risk_debate_state']['history']} """

        response = llm.invoke(prompt).content

        # 输出存储在 state 的 'final_trade_decision' 字段中。
        return {"final_trade_decision": response}

    return risk_manager_node
