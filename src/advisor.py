from __future__ import annotations

# src/advisor.py
import os
import json
import warnings
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google import genai
from rich.panel import Panel
from rich.markdown import Markdown
from .data_client import cache, get_advanced_metrics, get_ticker_news

# New import – we need live news & macro headlines inside the advisor
from . import data_client

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="urllib3")

# ----------------------------------------------------------------------
# Sentiment Analyzer (fallback only)
# ----------------------------------------------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


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
    # Fallback – simple VADER sentiment score
    # ------------------------------------------------------------------
    total_compound = sum(
        sia.polarity_scores(a.get("title", ""))["compound"] for a in macro_news
    )
    avg_score = total_compound / len(macro_news)
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
def analyze_ticker_sentiment(ticker: str, news: list) -> str:
    """
    Takes a list of news items for *ticker* and returns a deep sentiment &
    actionable recommendation using Gemini.  If Gemini is not reachable the
    original VADER‑based logic is used as a graceful fallback.
    """
    if not news:
        return "[yellow]No recent news found. Hold current position.[/yellow]"

    # Prepare a short, ordered list of headlines for the model.
    news_lines = []
    for i, article in enumerate(news[:10], start=1):
        title = article.get("title", "").strip()
        src = article.get("publisher", "")
        news_lines.append(f"{i}. {title} ({src})")
    news_block = "\n".join(news_lines)

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
        return gemini_response.strip()

    # ------------------------------------------------------------------
    # Fallback – original VADER based implementation
    # ------------------------------------------------------------------
    total_compound = 0.0
    valid_articles = 0

    for article in news:
        title = article.get("title", "")
        if title:
            score = sia.polarity_scores(title)
            total_compound += score["compound"]
            valid_articles += 1

    if valid_articles == 0:
        return "[yellow]Could not analyze sentiment (missing article titles).[/yellow]"

    avg_score = total_compound / valid_articles
    if avg_score >= 0.2:
        return (
            f"[bold green]Bullish (Score: {avg_score:.2f})[/bold green] - "
            "News is positive. Consider accumulating or holding."
        )
    elif avg_score <= -0.2:
        return (
            f"[bold red]Bearish (Score: {avg_score:.2f})[/bold red] - "
            "News is negative. Consider a stop‑loss or reducing exposure."
        )
    else:
        return (
            f"[bold yellow]Neutral (Score: {avg_score:.2f})[/bold yellow] - "
            "News is mixed or quiet. Maintain current position."
        )


def compute_portfolio_stats(portfolio):
    from collections import defaultdict
    from .data_client import get_current_price

    sector_exposure = defaultdict(float)
    total_value = 0

    for acc in portfolio.get("accounts", {}).values():
        for ticker, data in acc.get("holdings", {}).items():
            price = get_current_price(ticker)
            value = price * data["shares"]

            total_value += value

            # naive sector tagging (you can improve later)
            if ticker in ["VFV.TO", "VCE.TO"]:
                sector = "ETF"
            elif "MSFT" in ticker:
                sector = "Tech"
            else:
                sector = "Other"

            sector_exposure[sector] += value

    return total_value, sector_exposure


def get_gemini_analysis(
    current_portfolio: dict,
    user_settings: dict,
    total_cash: float,
    total_holdings: int,
    macro_news: list | None = None,
    ticker_news: dict | None = None,
) -> str:
    """
    Sends a highly structured text summary of the portfolio, risk profile, and news
    to Gemini‑2.5‑flash for analysis.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "[yellow]GEMINI_API_KEY not found in environment. Skipping advanced AI analysis.[/yellow]"

    allocations = user_settings.get("risk_allocation", {})
    c = allocations.get("conservative", 0)
    m = allocations.get("moderate", 100)
    a = allocations.get("aggressive", 0)

    accounts_data = current_portfolio.get("accounts", {})

    total_value, sector_exposure = compute_portfolio_stats(current_portfolio)

    # Translate the JSON into a clean string so the AI doesn't hallucinate
    portfolio_summary = ""
    for acc_name, acc_data in accounts_data.items():
        cash = acc_data.get("cash", 0.0)
        init = acc_data.get("initial_cash", 0.0)
        holdings = acc_data.get("holdings", {})

        portfolio_summary += f"\n### {acc_name.upper()} Account\n"
        portfolio_summary += f"- **Initial Investment All-Time:** ${init:,.2f}\n"
        portfolio_summary += f"- **Current Available Cash:** ${cash:,.2f}\n"
        portfolio_summary += "- **Holdings:**\n"

        if not holdings:
            portfolio_summary += "  - No active positions.\n"
        else:
            for tk, hk in holdings.items():
                portfolio_summary += f"  - {tk}: {hk['shares']} shares @ ${hk['avg_price']:.2f} avg cost\n"

    prompt = f"""
You are an expert, professional financial‑advisor AI specializing in Value and Growth analysis.

**CLIENT PROFILE**
- Target Risk Allocation: {c}% Conservative, {m}% Moderate, {a}% Aggressive.
- Total Unique Stock Positions: {total_holdings}
**PORTFOLIO BREAKDOWN**

Total Value: ${total_value:,.2f}

Sector Exposure:
{json.dumps(sector_exposure, indent=2)}
**PORTFOLIO HOLDINGS**
{portfolio_summary}
"""

    if macro_news:
        macro_section = "\n**MACRO ECONOMIC HEADLINES**\n"
        for i, article in enumerate(macro_news[:10], start=1):
            title = article.get("title", "").strip()
            src = (
                article.get("source", {}).get("name", "")
                if isinstance(article.get("source"), dict)
                else article.get("publisher", "")
            )
            macro_section += f"{i}. {title} ({src})\n"
        prompt += macro_section

    if ticker_news:
        ticker_section = "\n**TICKER‑SPECIFIC NEWS**\n"
        for tk, articles in ticker_news.items():
            if articles:
                ticker_section += f"\n***{tk.upper()}***\n"
                for i, article in enumerate(articles[:3], start=1):
                    title = article.get("title", "").strip()
                    src = article.get("publisher", "")
                    ticker_section += f"{i}. {title} ({src})\n"
        prompt += ticker_section

    prompt += """
**YOUR TASK**
Provide a concise, highly‑structured analysis formatted in clean Markdown. Include:

1. **Portfolio Health & Diversification** – assess exposure across sectors and comment on their all-time performance vs current cash drag.
2. **Macro Environment Impact** – how the headlines above could specifically affect their current holdings.
3. **Ticker‑Level Sentiment** – brief take on the news for the specific holdings.
4. **Actionable Recommendations** – 2‑3 concrete steps tailored strictly to their target risk allocation.
5. **Disclaimer** – a short note that the output is not personal investment advice.

**Tone** – Professional, objective, and direct. Use headings, bullet points, and bold for emphasis.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"[red]Error contacting Gemini API: {e}[/red]"


# ----------------------------------------------------------------------
# 4️⃣  Portfolio‑wide evaluation (rich output)
# ----------------------------------------------------------------------
def evaluate_portfolio(current_portfolio: dict, user_settings: dict):
    """
    Evaluates the portfolio structure, pulls macro & ticker news,
    and fetches an advanced Gemini analysis.  Returns a Rich ``Panel``
    (fallback) or a Rich ``Markdown`` object (when Gemini succeeds).
    """
    allocations = user_settings.get("risk_allocation", {})
    accounts = current_portfolio.get("accounts", {})

    total_holdings = sum(
        len(acc_data.get("holdings", {})) for acc_data in accounts.values()
    )
    total_cash = sum(acc_data.get("cash", 0.0) for acc_data in accounts.values())

    c = allocations.get("conservative", 0)
    m = allocations.get("moderate", 100)
    a = allocations.get("aggressive", 0)

    # ------------------------------------------------------------------
    #   Base structural advice (hard‑coded, colour‑coded)
    # ------------------------------------------------------------------
    structural_advice = (
        f"You hold {total_holdings} unique positions and have [green]${total_cash:,.2f}[/green] in total buying power.\n"
        f"Target Allocation: {c}% Conservative | {m}% Moderate | {a}% Aggressive.\n\n"
    )

    if total_cash > 0 and a >= 60:
        structural_advice += "- [yellow]Cash Drag Warning:[/yellow] You have an aggressive growth target, but holding cash limits upside during bull markets.\n"
    elif total_cash == 0:
        structural_advice += "- [red]Liquidity Warning:[/red] You have $0 in buying power. Consider raising cash.\n"

    # ------------------------------------------------------------------
    #   Fetch contextual news for a richer Gemini prompt
    # ------------------------------------------------------------------
    try:
        macro_news = data_client.get_macro_news()
    except Exception:
        macro_news = None

    # Gather a set of all tickers in the portfolio
    tickers = {
        ticker.upper()
        for acc in accounts.values()
        for ticker in acc.get("holdings", {}).keys()
    }

    ticker_news = {}
    for t in tickers:
        try:
            ticker_news[t] = data_client.get_ticker_news(t, limit=3)
        except Exception:
            ticker_news[t] = []

    # ------------------------------------------------------------------
    #   Gemini call – enriched with the news we just gathered
    # ------------------------------------------------------------------
    ai_response = get_gemini_analysis(
        current_portfolio,
        user_settings,
        total_cash,
        total_holdings,
        macro_news=macro_news,
        ticker_news=ticker_news,
    )

    # ------------------------------------------------------------------
    #   Render results – if Gemini failed we fall back to a Panel
    # ------------------------------------------------------------------
    if (
        "GEMINI_API_KEY not found" in ai_response
        or "Error contacting Gemini API" in ai_response
    ):
        final_output = structural_advice + "\n" + ai_response
        return Panel(final_output, title="Portfolio Analysis", border_style="blue")
    else:
        # The Gemini output is already markdown‑formatted
        return Markdown(ai_response)


# ----------------------------------------------------------------------
# 5️⃣  (Deprecated) Simple static version – kept for backward compatibility
# ----------------------------------------------------------------------
def evaluate_portfolio_legacy(current_portfolio: dict, user_settings: dict) -> str:
    """
    Simple static version kept for backward‑compatibility.  It is superseded
    by the richer ``evaluate_portfolio`` implementation above.
    """
    allocations = user_settings.get("risk_allocation", {})

    if not allocations:
        return "[yellow]No risk allocation found in settings. Run 'python main.py settings' to configure.[/yellow]"

    accounts = current_portfolio.get("accounts", {})

    if not accounts:
        return (
            "[yellow]Your portfolio is empty. Add stocks to get an analysis.[/yellow]"
        )

    total_holdings = sum(
        len(acc_data.get("holdings", {})) for acc_data in accounts.values()
    )
    total_cash = sum(acc_data.get("cash", 0.0) for acc_data in accounts.values())

    c = allocations.get("conservative", 0)
    m = allocations.get("moderate", 100)
    a = allocations.get("aggressive", 0)

    advice = (
        f"[bold]Portfolio Structural Analysis:[/bold]\n"
        f"You currently hold {total_holdings} unique positions and have [green]${total_cash:,.2f}[/green] in total buying power.\n\n"
        f"Target Allocation: {c}% Conservative, {m}% Moderate, {a}% Aggressive.\n"
    )

    if total_cash > 0:
        if a >= 60:
            advice += "- [yellow]Cash Drag Warning:[/yellow] You have an aggressive growth target, but holding cash limits upside during bull markets.\n"
        elif c >= 50:
            advice += "- [green]Cash Reserve:[/green] Your cash balance perfectly aligns with your conservative strategy, acting as a buffer against volatility.\n"
    else:
        advice += "- [red]Liquidity Warning:[/red] You have $0 in buying power. You cannot capitalize on market dips without selling existing assets. Consider raising cash.\n"

    advice += "\n[italic]Tip: In a future update, tag your individual stocks by risk category to get mathematically precise rebalancing advice![/italic]"

    return advice


# Add this function to src/advisor.py


@cache.memoize(expire=43200)
def generate_stock_report(ticker: str, current_portfolio: dict) -> str:
    """Generates a comprehensive investment thesis using hard data."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Cannot generate report.")

    # 1. Get Hard Data
    metrics = get_advanced_metrics(ticker)
    metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])

    # 2. Portfolio Context
    accounts = current_portfolio.get("accounts", {})
    owned_shares = 0
    avg_price = 0.0

    for acc_data in accounts.values():
        holdings = acc_data.get("holdings", {})
        for h_ticker, h_data in holdings.items():
            if ticker.upper() in h_ticker.upper():
                owned_shares += h_data["shares"]
                avg_price = h_data["avg_price"]

    # 3. News Context
    recent_news = get_ticker_news(ticker, limit=5)
    news_text = "\n".join(
        [f"- {n.get('title')} ({n.get('publisher')})" for n in recent_news]
    )

    prompt = f"""
    Role: Act as a Senior Equity Research Analyst.
    Task: Conduct a comprehensive investment thesis for {ticker.upper()}.

    **HARD FINANCIAL METRICS (Do not hallucinate these numbers, use them):**
    {metrics_text}

    **CLIENT PORTFOLIO CONTEXT:**
    - Current Position: {owned_shares} shares @ ${avg_price:.2f} average.

    **RECENT NEWS:**
    {news_text}

    Please provide a detailed report covering:
    1. Company Profile & Moat
    2. Financial Health (Reference the Hard Financial Metrics provided)
    3. Analyst Sentiment
    4. Macro-Economic State
    5. Technical Analysis & Entry Price
    6. Red Flags
    7. Conclusion & Action Plan tailored to the Client Portfolio Context.
    """

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {e}"


def generate_stock_report(ticker: str, current_portfolio: dict) -> str:
    """Generates the Ultimate Stock Deep-Dive report using the Senior Analyst persona."""
    metrics = get_advanced_metrics(ticker)

    # Context: Is the user currently holding this?
    accounts = current_portfolio.get("accounts", {})
    position_info = "The client does not currently hold this stock."
    for acc_data in accounts.values():
        if ticker.upper() in acc_data.get("holdings", {}):
            h = acc_data["holdings"][ticker.upper()]
            position_info = f"Current Position: {h['shares']} shares at an average cost of ${h['avg_price']:.2f}."

    news = get_ticker_news(ticker, limit=5)
    news_text = "\n".join([f"- {n.get('title')} ({n.get('publisher')})" for n in news])

    prompt = f"""
Role: Act as a Senior Equity Research Analyst specializing in Value and Growth investing.
Task: Conduct a comprehensive investment thesis and risk assessment for {ticker.upper()}.

**DATA CONTEXT**
- Live Financial Metrics: {json.dumps(metrics, indent=2)}
- Recent News Headlines: {news_text}
- Portfolio Context: {position_info}

Please provide a detailed report covering these six pillars:

1  **Company Profile & Moat:** What is their primary revenue model? What is their "Economic Moat"? Mention recent strategy shifts.
2  **Financial Health & Key Stats:** Analyze the provided P/E, Debt-to-Equity, FCF, and Revenue Growth (YoY). Comment on dividend sustainability.
3  **Analyst Sentiment & Institutional Ownership:** Summarize the consensus based on the analyst target and recommendation provided.
4  **Macro-Economic State:** How do current interest rates and inflation specifically impact {ticker.upper()}'s sector? 
5  **Technical Analysis & Entry Price:** Identify "Fair Value" and a "Margin of Safety" price (20% below intrinsic value).
6  **Red Flags:** List the top 3 specific risks that could break the thesis over the next 12–24 months.

**CONCLUSION:** Provide a definitive Verdict (Strong Buy, Buy, Hold, Sell) and a recommended Action Plan tailored to the Portfolio Context.
"""

    response = _gemini_generate(prompt)
    return response if response else "AI Advisor is currently unavailable."
