# 定义系统所有外部数据获取工具。
# 这些工具是分析师实现 ReAct（Reasoning + Acting）循环的核心，允许智能体在需要时调用真实世界数据。

import os
from typing import Annotated
import yfinance as yf
import finnhub
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from stockstats import wrap as stockstats_wrap

tavily_tool = TavilySearchResults(max_results=3)


@tool
def get_yfinance_data(
        symbol: Annotated[str, "股票代码"],
        start_date: Annotated[str, "开始日期(格式:yyyy-mm-dd)"],
        end_date: Annotated[str, "结束日期(格式:yyyy-mm-dd)"],
) -> str:
    """从雅虎财经获取指定股票代码的股票价格数据。"""
    try:
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        return data.to_csv()
    except Exception as e:
        return f"Error fetching Yahoo Finance data: {e}"


@tool
def get_technical_indicators(
        symbol: Annotated[str, "股票代码"],
        start_date: Annotated[str, "开始日期(格式:yyyy-mm-dd)"],
        end_date: Annotated[str, "结束日期(格式:yyyy-mm-dd)"],
) -> str:
    """使用 stockstats 库检索股票的关键技术指标。"""
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            return "No data to calculate indicators."
        stock_df = stockstats_wrap(df)
        indicators = stock_df[['macd', 'rsi_14', 'boll', 'boll_ub', 'boll_lb', 'close_50_sma', 'close_200_sma']]
        return indicators.tail().to_csv()  # Return last 5 days for brevity
    except Exception as e:
        return f"Error calculating stockstats indicators: {e}"


@tool
def get_finnhub_news(ticker: str, start_date: str, end_date: str) -> str:
    """从 Finnhub 获取指定日期范围内的公司新闻。"""
    try:
        finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])
        news_list = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        news_items = []
        for news in news_list[:5]:  # Limit to 5 results
            news_items.append(f"Headline: {news['headline']}\nSummary: {news['summary']}")
        return "\n\n".join(news_items) if news_items else "No Finnhub news found."
    except Exception as e:
        return f"Error fetching Finnhub news: {e}"


# 以下三个工具使用 Tavily 进行实时网络搜索。
tavily_tool = TavilySearchResults(max_results=3)


@tool
def get_social_media_sentiment(ticker: str, trade_date: str) -> str:
    """对股票相关的社交媒体情绪进行实时网络搜索。"""
    query = f"social media sentiment and discussions for {ticker} stock around {trade_date}"
    return tavily_tool.invoke({"query": query})


@tool
def get_fundamental_analysis(ticker: str, trade_date: str) -> str:
    """对股票的最新基本面分析进行实时网络搜索。"""
    query = f"fundamental analysis and key financial metrics for {ticker} stock published around {trade_date}"
    return tavily_tool.invoke({"query": query})


@tool
def get_macroeconomic_news(trade_date: str) -> str:
    """对与股市相关的宏观经济新闻进行实时网络搜索。"""
    query = f"macroeconomic news and market trends affecting the stock market on {trade_date}"
    return tavily_tool.invoke({"query": query})


# --- Toolkit Class ---
class Toolkit:
    def __init__(self):
        self.get_yfinance_data = get_yfinance_data
        self.get_technical_indicators = get_technical_indicators
        self.get_finnhub_news = get_finnhub_news
        self.get_social_media_sentiment = get_social_media_sentiment
        self.get_fundamental_analysis = get_fundamental_analysis
        self.get_macroeconomic_news = get_macroeconomic_news
