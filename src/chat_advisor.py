"""
AI Chat Advisor for personalized investment recommendations.

Provides a conversational interface to analyze holdings, get buy/sell recommendations
based on your specific position (gains, cost basis, allocation), and integrate
fundamentals, technicals, margin of safety, etc.
"""
from __future__ import annotations

import json
import os
from typing import Any

from google import genai

from src import data_client
from src.database import init_db, Account
from src.ticker_map import resolve_us_ticker
from src.alpha_vantage import (
    get_company_overview,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_technical_indicator,
)


def _gemini_generate(prompt: str, model: str = "gemini-2.5-flash") -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text
    except Exception as exc:
        print(f"Error calling Gemini: {exc}")
        return None


def get_position_context(ticker: str) -> dict[str, Any]:
    init_db()
    context = {
        "ticker": ticker.upper(),
        "has_position": False,
        "accounts": [],
        "total_shares": 0.0,
        "avg_cost": 0.0,
        "total_cost_basis": 0.0,
        "current_value": 0.0,
        "unrealized_gain_loss": 0.0,
        "unrealized_gain_loss_pct": 0.0,
    }

    for acc in Account.select():
        for h in acc.holdings:
            if h.ticker.upper() == ticker.upper():
                live, _ = data_client.get_current_price(ticker)
                if live <= 0:
                    live = h.avg_price
                cost = h.shares * h.avg_price
                value = h.shares * live
                gain = value - cost
                gain_pct = (gain / cost * 100) if cost > 0 else 0.0
                context["accounts"].append({
                    "account": acc.name,
                    "shares": h.shares,
                    "avg_price": h.avg_price,
                    "cost_basis": cost,
                    "current_price": live,
                    "market_value": value,
                    "unrealized_gain_loss": gain,
                    "unrealized_gain_loss_pct": gain_pct,
                })
                context["total_shares"] += h.shares
                context["total_cost_basis"] += cost
                context["current_value"] += value
                context["has_position"] = True

    if context["has_position"] and context["total_shares"] > 0:
        context["avg_cost"] = context["total_cost_basis"] / context["total_shares"]
        context["unrealized_gain_loss"] = context["current_value"] - context["total_cost_basis"]
        context["unrealized_gain_loss_pct"] = (context["unrealized_gain_loss"] / context["total_cost_basis"] * 100) if context["total_cost_basis"] > 0 else 0.0

    live_price, _ = data_client.get_current_price(ticker)
    context["current_price"] = live_price if live_price > 0 else context["avg_cost"]
    return context


def get_comprehensive_analysis(ticker: str) -> dict[str, Any]:
    us_ticker = resolve_us_ticker(ticker)
    analysis = {
        "ticker": ticker.upper(),
        "us_ticker": us_ticker,
        "position": get_position_context(ticker),
        "overview": {},
        "fundamentals": {},
        "technicals": {},
    }

    try:
        overview = get_company_overview(us_ticker) or {}
        analysis["overview"] = overview
    except Exception:
        pass

    try:
        income = get_income_statement(us_ticker, period="annual")[:2]
        balance = get_balance_sheet(us_ticker, period="annual")[:2]
        cashflow = get_cash_flow(us_ticker, period="annual")[:2]
        analysis["fundamentals"] = {
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
        }
    except Exception:
        pass

    try:
        rsi = get_technical_indicator(us_ticker, "RSI", interval="daily", time_period=14)
        sma20 = get_technical_indicator(us_ticker, "SMA", interval="daily", time_period=20)
        sma50 = get_technical_indicator(us_ticker, "SMA", interval="daily", time_period=50)
        sma200 = get_technical_indicator(us_ticker, "SMA", interval="daily", time_period=200)
        macd = get_technical_indicator(us_ticker, "MACD", interval="daily", fastperiod=12, slowperiod=26, signalperiod=9)
        analysis["technicals"] = {
            "rsi": rsi,
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "macd": macd,
        }
    except Exception:
        pass

    return analysis


def chat_advice(ticker: str, question: str | None = None) -> str:
    analysis = get_comprehensive_analysis(ticker)
    pos = analysis["position"]
    overview = analysis["overview"]

    prompt = f"""You are a seasoned investment advisor and portfolio manager with expertise in value investing, risk management, and position sizing.

Your job is to analyze a specific stock position in the user's portfolio and give concrete, actionable advice about whether to BUY MORE, HOLD, or SELL.

### STOCK
{ticker.upper()}

### USER'S CURRENT POSITION
{json.dumps(pos, indent=2)}

### COMPANY OVERVIEW
{json.dumps({k: v for k, v in overview.items() if k in ['Symbol', 'Name', 'Sector', 'Industry', 'Description', 'MarketCapitalization', 'PERatio', 'PEGRatio', 'PriceToBookRatio', 'DividendYield', 'EPS', 'RevenueTTM', 'ProfitMargin', 'ReturnOnEquityTTM', '52WeekHigh', '52WeekLow', 'AnalystTargetPrice', 'Recommendation', 'SharesOutstanding']}, indent=2)}

### QUESTION FROM USER (if any)
{question or 'Provide comprehensive buy/sell/hold recommendation'}

### ANALYSIS REQUIREMENTS
Give a thorough response with these sections:

1. **POSITION ASSESSMENT** - Your unrealized gain/loss, cost basis vs current price, how significant is this position (% of portfolio if weight known)
2. **FUNDAMENTAL ANALYSIS** - P/E, PEG, P/B, margins, ROE, debt, growth, competitive moat
3. **VALUATION & MARGIN OF SAFETY** - Is it overvalued/fair/undervalued? Calculate/estimate Margin of Safety vs intrinsic value if possible
4. **TECHNICAL ANALYSIS** - Trend, support/resistance, momentum (RSI), moving averages
5. **RISK ASSESSMENT** - Downside risk, volatility, concentration risk
6. **RECOMMENDATION** - Be explicit: BUY MORE / HOLD / SELL / PARTIALLY SELL. Give specific reasoning
7. **ACTION PLAN** - If sell: HOW MUCH to sell (e.g., trim 25%, exit fully, take profits)? If buy: how much? If hold: what price levels to watch?
8. **PRICE TARGETS & TRIGGERS** - Specific sell targets, stop-loss, buy targets
9. **DISCLAIMER** - Not financial advice, do your own research

Be decisive and specific. Quantify everything you can. Consider tax implications conceptually.

### ANSWER IN MARKDOWN FORMAT
"""

    response = _gemini_generate(prompt)
    if not response:
        return "Unable to generate AI advice. Please ensure GEMINI_API_KEY is configured and try again."
    return response
