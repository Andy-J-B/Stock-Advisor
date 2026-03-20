import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Automatically download the VADER lexicon if it's missing
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Initialize the Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


def evaluate_portfolio(current_portfolio: dict, user_settings: dict) -> str:
    """
    Evaluates the portfolio based on the user's risk settings.
    (Note: A full version would categorize individual stocks. This provides structural advice.)
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

    c = allocations.get("conservative", 0)
    m = allocations.get("moderate", 100)
    a = allocations.get("aggressive", 0)

    # Generate a baseline structural recommendation
    advice = (
        f"[bold]Portfolio Structural Analysis:[/bold]\n"
        f"You currently hold {total_holdings} unique positions across your accounts.\n\n"
        f"Based on your target allocation ({c}% Conservative, {m}% Moderate, {a}% Aggressive), "
        f"ensure you are weighting your capital accordingly:\n"
    )

    if a > 40:
        advice += "- [red]High Aggression:[/red] You are targeting high growth. Expect high volatility. Ensure you are taking profits during market rallies.\n"
    if c > 40:
        advice += "- [green]High Conservation:[/green] You are targeting safety. Ensure your portfolio is heavily weighted in ETFs, bonds, or dividend aristocrats rather than individual tech stocks.\n"

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
