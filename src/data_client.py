import yfinance as yf
import requests
from diskcache import Cache
from pathlib import Path

# Setup local cache
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
cache = Cache(CACHE_DIR)


# Cache FX rates for 1 hour (3600 seconds)
@cache.memoize(expire=3600)
def get_usd_to_cad() -> float:
    primary_url = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
    fallback_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
    try:
        try:
            response = requests.get(primary_url, timeout=5)
            response.raise_for_status()
        except:
            response = requests.get(fallback_url, timeout=5)
            response.raise_for_status()
        return response.json()["usd"]["cad"]
    except Exception:
        return 1.35


# Cache live prices for 5 minutes (300 seconds)
@cache.memoize(expire=300)
def get_current_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        return round(stock.fast_info["lastPrice"], 2)
    except Exception:
        return 0.0


# NEW: Cache fundamental data for 24 hours (86400 seconds)
@cache.memoize(expire=86400)
def get_advanced_metrics(ticker: str) -> dict:
    """Fetches hard financial data to inject into the AI prompt."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Trailing P/E": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Debt-to-Equity": info.get("debtToEquity", "N/A"),
            "Dividend Yield": (
                f"{info.get('dividendYield', 0) * 100:.2f}%"
                if info.get("dividendYield")
                else "N/A"
            ),
            "Free Cash Flow": info.get("freeCashflow", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception:
        return {}


@cache.memoize(expire=3600)
def get_ticker_news(ticker: str, limit: int = 3) -> list:
    try:
        stock = yf.Ticker(ticker)
        return [
            {
                "title": a.get("title", ""),
                "publisher": a.get("publisher", ""),
                "link": a.get("link", ""),
            }
            for a in stock.news[:limit]
        ]
    except Exception:
        return [{"title": f"Could not fetch news for {ticker}.", "publisher": "System"}]
