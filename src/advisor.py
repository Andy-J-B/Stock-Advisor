import os
import json
import warnings
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google import genai
from rich.panel import Panel
from rich.markdown import Markdown

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="urllib3")


# Automatically download the VADER lexicon if it's missing
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Initialize the Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


def evaluate_portfolio(current_portfolio: dict, user_settings: dict) -> str:
    allocations = user_settings.get("risk_allocation", {})
    if not allocations:
        return "[yellow]No risk allocation found in settings. Run 'python main.py settings' to configure.[/yellow]"

    accounts = current_portfolio.get("accounts", {})
    if not accounts:
        return (
            "[yellow]Your portfolio is empty. Add stocks to get an analysis.[/yellow]"
        )

    # Tally up unique positions and total cash
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

    # AI logic regarding cash and buying power
    if total_cash > 0:
        if a >= 60:
            advice += "- [yellow]Cash Drag Warning:[/yellow] You have an aggressive growth target, but holding cash limits upside during bull markets. Consider deploying buying power if market sentiment is positive.\n"
        elif c >= 50:
            advice += "- [green]Cash Reserve:[/green] Your cash balance perfectly aligns with your conservative strategy, acting as a buffer against volatility.\n"
    else:
        advice += "- [red]Liquidity Warning:[/red] You have $0 in buying power. You cannot capitalize on market dips without selling existing assets. Consider raising cash.\n"

    advice += "\n[italic]Tip: In a future update, tag your individual stocks by risk category to get mathematically precise rebalancing advice![/italic]"

    return advice


def analyze_ticker_sentiment(ticker: str, news: list) -> str:
    """
    Analyzes the sentiment of recent news articles for a specific ticker.
    """
    if not news or len(news) == 0:
        return "[yellow]No recent news found. Hold current position.[/yellow]"

    total_compound = 0
    valid_articles = 0

    for article in news:
        title = article.get("title", "")
        if title:
            # VADER returns a 'compound' score from -1 (most negative) to +1 (most positive)
            score = sia.polarity_scores(title)
            total_compound += score["compound"]
            valid_articles += 1

    if valid_articles == 0:
        return "[yellow]Could not analyze sentiment (missing article titles).[/yellow]"

    avg_score = total_compound / valid_articles

    # Map the numerical score to actionable advice
    if avg_score >= 0.2:
        return f"[bold green]Bullish (Score: {avg_score:.2f})[/bold green] - News is positive. Consider accumulating or holding."
    elif avg_score <= -0.2:
        return f"[bold red]Bearish (Score: {avg_score:.2f})[/bold red] - News is negative. Consider a stop-loss or reducing exposure."
    else:
        return f"[bold yellow]Neutral (Score: {avg_score:.2f})[/bold yellow] - News is mixed or quiet. Maintain current position."


def generate_market_advice(macro_news: list) -> str:
    """
    Analyzes broad market headlines and gives a general market health summary.
    """
    if not macro_news:
        return "Could not fetch macro market data today."

    total_compound = 0

    for article in macro_news:
        title = article.get("title", "")
        score = sia.polarity_scores(title)
        total_compound += score["compound"]

    avg_score = total_compound / len(macro_news)

    summary = f"[bold]Market Sentiment Score:[/bold] {avg_score:.2f}\n"

    if avg_score > 0.15:
        summary += "[green]The broader market outlook is currently optimistic. Favorable conditions for growth assets.[/green]"
    elif avg_score < -0.15:
        summary += "[red]The broader market outlook is currently pessimistic. Consider defensive positioning.[/red]"
    else:
        summary += "[yellow]The broader market is currently neutral or showing mixed signals. Stick to your core strategy.[/yellow]"

    return summary


def get_gemini_analysis(
    current_portfolio: dict, user_settings: dict, total_cash: float, total_holdings: int
) -> str:
    """Builds a prompt from user data and fetches advice from Gemini."""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[yellow]GEMINI_API_KEY not found in environment. Skipping advanced AI analysis.[/yellow]"

    # Format the data for the prompt
    allocations = user_settings.get("risk_allocation", {})
    c = allocations.get("conservative", 0)
    m = allocations.get("moderate", 100)
    a = allocations.get("aggressive", 0)
    accounts_data = current_portfolio.get("accounts", {})

    # The Prompt Engineering
    prompt = f"""
    You are an expert, professional financial advisor AI. 
    Your client has requested an analysis of their portfolio based on their target risk profile.
    
    CLIENT PROFILE:
    - Target Risk Allocation: {c}% Conservative, {m}% Moderate, {a}% Aggressive.
    - Total Cash (Buying Power): ${total_cash:,.2f}
    - Total Unique Stock Positions: {total_holdings}
    
    PORTFOLIO DATA (JSON):
    {json.dumps(accounts_data, indent=2)}
    
    YOUR TASK:
    Provide a concise, highly structured analysis formatted in clean Markdown.
    
    1. Diversification: Briefly assess their current diversification based on the specific tickers provided.
    2. Cash Position: Evaluate their cash position relative to their stated risk profile.
    3. Actionable Advice: Provide 2-3 specific, actionable recommendations (e.g., sectors to explore, whether to deploy cash, rebalancing suggestions). Do not recommend specific stock buying prices, keep it strategic.
    4. Disclaimer: End with a standard, brief financial disclaimer.
    
    Tone: Professional, objective, insightful, and concise. Do not use filler introductions like "Here is your analysis".
    """

    # NEW SDK SYNTAX
    try:
        # Initialize the new client
        client = genai.Client(api_key=api_key)

        # Use the newest, fastest model available via the new SDK
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"[red]Error contacting Gemini API: {e}[/red]"


def evaluate_portfolio(current_portfolio: dict, user_settings: dict):
    """
    Evaluates the portfolio structure and fetches Gemini AI advice.
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

    structural_advice = (
        f"You hold {total_holdings} unique positions and have [green]${total_cash:,.2f}[/green] in total buying power.\n"
        f"Target Allocation: {c}% Conservative | {m}% Moderate | {a}% Aggressive.\n\n"
    )

    if total_cash > 0 and a >= 60:
        structural_advice += "- [yellow]Cash Drag Warning:[/yellow] You have an aggressive growth target, but holding cash limits upside during bull markets.\n"
    elif total_cash == 0:
        structural_advice += "- [red]Liquidity Warning:[/red] You have $0 in buying power. Consider raising cash to capitalize on dips.\n"

    ai_response = get_gemini_analysis(
        current_portfolio, user_settings, total_cash, total_holdings
    )

    if "GEMINI_API_KEY not found" in ai_response or "Error contacting" in ai_response:
        final_output = structural_advice + "\n" + ai_response
        return Panel(final_output, title="Portfolio Analysis", border_style="blue")
    else:
        return Markdown(ai_response)


def evaluate_portfolio(current_portfolio: dict, user_settings: dict):
    """
    Evaluates the portfolio structure and fetches Gemini AI advice.
    Returns a Rich renderable object (Panel or Markdown) for beautiful CLI output.
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

    # Base Structural Advice (The hardcoded logic)
    structural_advice = (
        f"You hold {total_holdings} unique positions and have [green]${total_cash:,.2f}[/green] in total buying power.\n"
        f"Target Allocation: {c}% Conservative | {m}% Moderate | {a}% Aggressive.\n\n"
    )

    if total_cash > 0 and a >= 60:
        structural_advice += "- [yellow]Cash Drag Warning:[/yellow] You have an aggressive growth target, but holding cash limits upside during bull markets.\n"
    elif total_cash == 0:
        structural_advice += "- [red]Liquidity Warning:[/red] You have $0 in buying power. Consider raising cash to capitalize on dips.\n"

    # Fetch advanced Gemini Advice
    ai_response = get_gemini_analysis(
        current_portfolio, user_settings, total_cash, total_holdings
    )

    # If the API key is missing, it returns a plain string warning.
    # If successful, it returns Markdown. We use Rich to format it nicely.
    if "GEMINI_API_KEY not found" in ai_response or "Error contacting" in ai_response:
        final_output = structural_advice + "\n" + ai_response
        return Panel(final_output, title="Portfolio Analysis", border_style="blue")
    else:
        # Render the Gemini markdown beautifully in the terminal
        return Markdown(ai_response)
