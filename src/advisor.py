from __future__ import annotations

# src/advisor.py
import os
import json
import warnings
from datetime import date
from typing import Any, List
from google import genai
from rich.panel import Panel
from rich.markdown import Markdown
from .sentiment import get_sentiment_engine
from .alpha_vantage import (
    get_macro_news as av_get_macro_news,
    get_ticker_news as av_get_ticker_news,
    get_company_overview,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_technical_indicator,
)
from . import data_client
from .ticker_map import resolve_tickers


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="urllib3")


# ----------------------------------------------------------------------
# Helper: tiny wrapper around the Gemini SDK
# ----------------------------------------------------------------------
def _gemini_generate(prompt: str) -> str | None:
    """
    Sends *prompt* to Gemini‑2.5‑flash and returns the plaintext response.
    Returns ``None`` when the API key is missing or any error occurs.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        # ``resp.text`` is the generated markdown string
        return resp.text
    except Exception:
        return None


# ----------------------------------------------------------------------
# 1️⃣  Market‑wide advice (macro news)
# ----------------------------------------------------------------------
def generate_market_advice(macro_news: list) -> str:
    """
    Builds a detailed market outlook from the supplied macro headlines.
    If Gemini is unavailable we fall back to a lightweight VADER‑based
    sentiment score.
    """
    if not macro_news:
        return "Could not fetch macro market data today."

    # ------------------------------------------------------------------
    # Build a compact, human‑readable list of headlines for the prompt.
    # ------------------------------------------------------------------
    headline_lines = []
    for i, article in enumerate(macro_news[:10], start=1):
        title = article.get("title", "").strip()
        source = ""
        # NewsAPI provides a dict under "source"; yfinance uses "publisher"
        src_obj = article.get("source", {})
        if isinstance(src_obj, dict):
            source = src_obj.get("name", "")
        else:
            source = article.get("publisher", "")
        headline_lines.append(f"{i}. {title} ({source})")
    headlines_block = "\n".join(headline_lines)

    # ------------------------------------------------------------------
    # Prompt sent to Gemini – we ask for a full markdown report.
    # ------------------------------------------------------------------
    prompt = f"""
You are a senior financial analyst. Using the macro‑level headlines below, produce
a **comprehensive market outlook** in clean Markdown. Include:

1️⃣  **Key Economic Themes** – interest‑rates, inflation, geopolitics, sector
    movers, etc.  
2️⃣  **Overall Market Sentiment** – bullish, bearish, or neutral, with a short
    justification.  
3️⃣  **Strategic Recommendations** – what a conservative, moderate and aggressive
    investor should consider (e.g., sector bias, defensive positioning,
    opportunistic buying).  
4️⃣  **Brief Disclaimer** – typical “not investment advice” clause.

**MACRO HEADLINES**  
{headlines_block}
"""

    gemini_response = _gemini_generate(prompt)
    if gemini_response:
        return gemini_response.strip()

    # ------------------------------------------------------------------
    # Fallback – FinBERT sentiment score
    # ------------------------------------------------------------------
    engine = get_sentiment_engine()
    titles = [a.get("title", "") for a in macro_news if a.get("title")]
    scores = engine.score_batch(titles)
    avg_score = sum(s["compound"] for s in scores) / len(scores) if scores else 0.0
    sentiment_word = (
        "optimistic"
        if avg_score > 0.15
        else "pessimistic" if avg_score < -0.15 else "neutral"
    )
    return (
        f"[bold]Market Sentiment Score:[/bold] {avg_score:.2f} " f"({sentiment_word})."
    )


# ----------------------------------------------------------------------
# 2️⃣  Ticker‑specific sentiment analysis
# ----------------------------------------------------------------------
def analyze_ticker_sentiment(ticker: str, news: list) -> tuple[str, list[dict]]:
    """
    Uses the Alpha Vantage news-sentiment endpoint. If a sentiment score
    is present we weight the final recommendation by that score, otherwise
    we fall back to FinBERT.

    Returns (advice_string, per_headline_scores).
    """
    if not news:
        return (
            "[yellow]No recent news found. Hold current position.[/yellow]",
            [],
        )

    # ------------------------------------------------------------------
    # Build a nice headline block for the Gemini prompt (same as before)
    # ------------------------------------------------------------------
    news_lines = []
    sentiment_scores = []
    for i, article in enumerate(news[:10], start=1):
        title = article.get("title", "").strip()
        src = article.get("publisher", "")
        score = article.get("sentiment_score")
        # Save scores that are actually present (Alpha returns None for some)
        if isinstance(score, (int, float)):
            sentiment_scores.append(score)
        news_lines.append(f"{i}. {title} ({src})")
    news_block = "\n".join(news_lines)

    # --------------------------------------------------------------
    # Prompt for Gemini (unchanged – we only changed the data source)
    # --------------------------------------------------------------
    prompt = f"""
    You are an experienced equity analyst. Evaluate the following recent headlines
    for **{ticker.upper()}** and provide a concise markdown report containing:

    * **Themes Summary** – the main stories driving sentiment.
    * **Overall Sentiment** – bullish, neutral, or bearish (with a short rationale).
    * **Short‑Term Impact** – likely price direction in the next few weeks.
    * **Recommendation** – BUY, HOLD, or SELL, with a clear justification.
    * **Disclaimer** – brief “not investment advice” statement.

    **NEWS HEADLINES**  
    {news_block}
    """

    gemini_response = _gemini_generate(prompt)
    if gemini_response:
        return gemini_response.strip(), []

    # ------------------------------------------------------------------
    # Fallback – FinBERT + simple averaging of Alpha sentiment scores
    # ------------------------------------------------------------------
    # If Alpha gave us numeric scores we blend them with FinBERT.
    if sentiment_scores:
        # Alpha scores are already in the range -1..1
        alpha_avg = sum(sentiment_scores) / len(sentiment_scores)
    else:
        alpha_avg = 0.0

    # FinBERT part
    engine = get_sentiment_engine()
    titles = [a.get("title", "") for a in news if a.get("title")]
    headline_scores = engine.score_batch(titles)
    finbert_avg = (
        sum(s["compound"] for s in headline_scores) / len(headline_scores)
        if headline_scores
        else 0.0
    )

    # Blend – 70% Alpha, 30% FinBERT
    blended = 0.7 * alpha_avg + 0.3 * finbert_avg

    if blended >= 0.20:
        advice = f"[bold green]Bullish (Score: {blended:.2f})[/bold green] – consider buying or holding."
    elif blended <= -0.20:
        advice = f"[bold red]Bearish (Score: {blended:.2f})[/bold red] – consider reducing exposure."
    else:
        advice = f"[bold yellow]Neutral (Score: {blended:.2f})[/bold yellow] – maintain current position."

    return advice, headline_scores


# ----------------------------------------------------------------------
# 4️⃣  Portfolio‑wide evaluation (rich output)
# ----------------------------------------------------------------------
def evaluate_portfolio(current_portfolio: dict, user_settings: dict):
    """
    This is the *new* entry point used by the ``analyze`` command.
    It builds a rich prompt that includes:
      • macro‑news (Alpha Vantage NEWS_SENTIMENT)
      • ticker‑specific news (Alpha Vantage)
      • sector exposure derived from company overview (Alpha)
      • a lightweight risk‑allocation breakdown
    """
    allocations = user_settings.get("risk_allocation", {})
    accounts = current_portfolio.get("accounts", {})

    # ----- basic numbers -------------------------------------------------
    total_holdings = sum(len(acc.get("holdings", {})) for acc in accounts.values())

    # ----- macro news ----------------------------------------------------
    macro_news = av_get_macro_news(limit=7)  # returns list of dicts

    # ----- ticker news + sector map ---------------------------------------
    tickers = {
        tk.upper() for acc in accounts.values() for tk in acc.get("holdings", {}).keys()
    }

    # Resolve Canadian tickers to US equivalents for Alpha Vantage calls.
    # AV does not support .NE/.TO tickers (returns 0/empty).
    ticker_us_map = resolve_tickers(list(tickers))

    ticker_news: dict[str, List[dict[str, Any]]] = {}
    sector_map: dict[str, str] = {}
    for t in tickers:
        us = ticker_us_map.get(t, t)
        # news + sentiment (Alpha)
        ticker_news[t] = av_get_ticker_news(us, limit=3)

        # sector extraction – cheap, cached
        ov = get_company_overview(us) or {}
        sector_map[t] = ov.get("Sector", "Other")

    # ----- prices & valuation -------------------------------------------
    # Value every holding at its *actual* traded price (CAD for .TO/.NE
    # listings, incl. CDRs) rather than the full US underlying price —
    # a CDR is not a full US share and trades in CAD.  US-account cash and
    # holdings are converted to CAD via the current FX rate, matching the
    # view-portfolio global summary.
    fx_rate = data_client.get_usd_to_cad()
    all_tickers = [tk for acc in accounts.values() for tk in acc.get("holdings", {})]
    all_tickers = list(dict.fromkeys(all_tickers))  # dedupe, preserve order
    prices = data_client.get_current_prices_batch(all_tickers) if all_tickers else {}

    total_cash = 0.0
    total_portfolio_value = 0.0
    sector_exposure: dict[str, float] = {}
    for acc_name, acc_data in accounts.items():
        multiplier = fx_rate if acc_name.upper() == "USD" else 1.0
        total_cash += acc_data.get("cash", 0.0) * multiplier
        for t, data in acc_data.get("holdings", {}).items():
            price, _ = prices.get(t, (0.0, 0.0))
            if price <= 0.0:
                price, _ = data_client.get_current_price(t)
            pos_value = price * data["shares"] * multiplier
            total_portfolio_value += pos_value
            sector = sector_map.get(t.upper(), "Other")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + pos_value

    # normalise to percentages for the prompt
    if total_portfolio_value > 0:
        for sector in sector_exposure:
            sector_exposure[sector] = round(
                100 * sector_exposure[sector] / total_portfolio_value, 2
            )

    # ------------------------------------------------------------------
    # Build the big Gemini prompt
    # ------------------------------------------------------------------
    prompt = f"""
You are a professional financial‑advisor AI.  The client has the following
portfolio (all values are in CAD unless otherwise noted):

**TOTAL CASH (Buying Power):** ${total_cash:,.2f}
**TOTAL HOLDINGS VALUE:** ${total_portfolio_value:,.2f}
**UNIQUE POSITIONS:** {total_holdings}
**RISK ALLOCATION Preference:** {allocations.get('conservative',0)}% Conservative,
{allocations.get('moderate',100)}% Moderate,
{allocations.get('aggressive',0)}% Aggressive.

**SECTOR EXPOSURE (%)**
{json.dumps(sector_exposure, indent=2)}

**MACRO ECONOMIC HEADLINES**
"""
    for i, a in enumerate(macro_news[:6], 1):
        prompt += f"{i}. {a['title']} ({a.get('publisher','')})\n"

    prompt += "\n**TICKER‑SPECIFIC NEWS (with sentiment scores)**\n"
    for tk, articles in ticker_news.items():
        if not articles:
            continue
        prompt += f"\n***{tk}***\n"
        for i, art in enumerate(articles[:3], 1):
            score = art.get("sentiment_score", 0.0)
            label = art.get("sentiment_label", "NEUTRAL")
            prompt += f"{i}. {art['title']} ({art.get('publisher','')}) – {label} ({score:+.2f})\n"

    prompt += """
**YOUR TASK**
Provide a concise, markdown‑formatted analysis containing:

1. Portfolio health & diversification – comment on any sector overweight/underweight.
2. How the macro headlines could affect the holdings.
3. Short‑term sentiment for each ticker (based on the scores above).
4. 2‑3 clear actionable recommendations that respect the client’s risk‑allocation.
5. A brief disclaimer.

**Tone** – Objective, professional, bullet‑point heavy, use **bold** for headings.
"""

    # ------------------------------------------------------------------
    # Call Gemini
    # ------------------------------------------------------------------
    ai_response = _gemini_generate(prompt)

    # ------------------------------------------------------------------
    # Render – if Gemini failed we fall back to the original Panel logic
    # ------------------------------------------------------------------
    if not ai_response:
        # keep the original structural advice as a safety net
        structural_advice = (
            f"You hold {total_holdings} unique positions and have "
            f"[green]${total_cash:,.2f}[/green] in total buying power.\n"
            f"Target Allocation: {allocations.get('conservative',0)}% Conservative | "
            f"{allocations.get('moderate',100)}% Moderate | "
            f"{allocations.get('aggressive',0)}% Aggressive.\n"
        )
        return Panel(structural_advice, title="Portfolio Analysis", border_style="blue")
    else:
        return Markdown(ai_response)


def generate_stock_report(ticker: str, current_portfolio: dict) -> str:
    """
    Uses Alpha Vantage fundamentals (OVERVIEW, INCOME, BALANCE, CASHFLOW)
    plus a short list of technical indicators (SMA, RSI, MACD) to give the
    LLM more concrete numbers.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Basic portfolio context (unchanged)
    # ------------------------------------------------------------------
    accounts = current_portfolio.get("accounts", {})
    position_info = "The client does not currently hold this stock."
    for acc in accounts.values():
        if ticker.upper() in acc.get("holdings", {}):
            h = acc["holdings"][ticker.upper()]
            position_info = (
                f"Current Position: {h['shares']} shares @ ${h['avg_price']:.2f} avg."
            )

    # ------------------------------------------------------------------
    # 2️⃣ Fundamentals from Alpha Vantage
    # ------------------------------------------------------------------
    overview = get_company_overview(ticker) or {}
    income = get_income_statement(ticker, period="annual")[:1]  # most recent year
    balance = get_balance_sheet(ticker, period="annual")[:1]
    cashflow = get_cash_flow(ticker, period="annual")[:1]

    # Build a concise markdown bullet list – the LLM will expand it
    fundamentals_md = "\n".join(
        [
            f"- **Sector:** {overview.get('Sector', 'N/A')}",
            f"- **Industry:** {overview.get('Industry', 'N/A')}",
            f"- **Market Cap:** {overview.get('MarketCapitalization', 'N/A')}",
            f"- **PE Ratio (TTM):** {overview.get('PERatio', 'N/A')}",
            f"- **PEG Ratio:** {overview.get('PEGRatio', 'N/A')}",
            f"- **Dividend Yield:** {overview.get('DividendYield', 'N/A')}",
            f"- **52‑Week High / Low:** ${overview.get('52WeekHigh', 'N/A')} / ${overview.get('52WeekLow', 'N/A')}",
            f"- **Profit Margin:** {overview.get('ProfitMargin', 'N/A')}",
            f"- **Return on Equity:** {overview.get('ReturnOnEquityTTM', 'N/A')}",
            f"- **Latest Revenue:** {income[0].get('totalRevenue') if income else 'N/A'}",
            f"- **Net Income:** {income[0].get('netIncome') if income else 'N/A'}",
            f"- **Total Assets:** {balance[0].get('totalAssets') if balance else 'N/A'}",
            f"- **Total Liabilities:** {balance[0].get('totalLiabilities') if balance else 'N/A'}",
            f"- **Operating Cash Flow:** {cashflow[0].get('operatingCashFlow') if cashflow else 'N/A'}",
        ]
    )

    # ------------------------------------------------------------------
    # 3️⃣ Light technical snapshot (SMA‑20, RSI‑14, MACD)
    # ------------------------------------------------------------------
    sma = get_technical_indicator(ticker, "SMA", interval="daily", time_period=20)
    rsi = get_technical_indicator(ticker, "RSI", interval="daily", time_period=14)
    macd = get_technical_indicator(
        ticker, "MACD", interval="daily", fastperiod=12, slowperiod=26, signalperiod=9
    )

    tech_md = ""
    try:
        sma_val = list(sma["Technical Analysis: SMA"].values())[0]["SMA"]
        rsi_val = list(rsi["Technical Analysis: RSI"].values())[0]["RSI"]
        macd_vals = list(macd["Technical Analysis: MACD"].values())[0]
        macd_val = macd_vals["MACD"]
        macd_signal = macd_vals["MACD_Signal"]
        tech_md = (
            f"- **SMA‑20:** {float(sma_val):.2f}\n"
            f"- **RSI‑14:** {float(rsi_val):.2f}\n"
            f"- **MACD:** {float(macd_val):.2f} (Signal: {float(macd_signal):.2f})"
        )
    except Exception:
        tech_md = "- Technical indicator data not available."

    # ------------------------------------------------------------------
    # 4️⃣ Prompt for Gemini (the same persona as before)
    # ------------------------------------------------------------------
    prompt = f"""
Role: Senior Equity Research Analyst (Value & Growth focus).

**TICKER:** {ticker.upper()}
**DATE:** {date.today().strftime('%B %d, %Y')}
**PORTFOLIO CONTEXT:** {position_info}

**FUNDAMENTAL SNAPSHOT**
{fundamentals_md}

**TECHNICAL SNAPSHOT**
{tech_md}

Please produce a markdown report covering:

1️⃣ **Company Profile & Moat** – key business model, competitive advantages.  
2️⃣ **Financial Health** – comment on profitability, balance‑sheet strength and cash‑flow.  
3️⃣ **Valuation** – interpret the PE/PEG and suggest a fair‑value range.  
4️⃣ **Analyst Sentiment** – use the *Recommendation* field from the overview.  
5️⃣ **Macro‑Economic Impact** – briefly note how current rates/inflation affect the sector.  
6️⃣ **Technical Outlook** – bullish/neutral/bearish based on SMA/RSI/MACD.  
7️⃣ **Red‑Flags** – up to three material risks.  

**CONCLUSION** – final Verdict (Strong Buy / Buy / Hold / Sell) + concise action plan.

**DISCLAIMER** – standard “not personal investment advice” clause.
"""

    response = _gemini_generate(prompt)
    return response if response else "AI Advisor is currently unavailable."
