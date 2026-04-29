import os
import yfinance as yf
import requests
from diskcache import Cache
from pathlib import Path

# Setup local cache
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
cache = Cache(CACHE_DIR)

# Load APIs securely
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
FMP_KEY = os.getenv("FMP_API_KEY", "")


@cache.memoize(expire=3600)
def get_usd_to_cad() -> float:
    primary_url = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
    fallback_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
    try:
        try:
            response = requests.get(primary_url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException:  # Fix applied here
            response = requests.get(fallback_url, timeout=5)
            response.raise_for_status()
        return response.json()["usd"]["cad"]
    except Exception:
        return 1.35


@cache.memoize(expire=300)
def get_current_price(ticker: str) -> tuple[float, float]:
    """Returns a tuple of (current_price, previous_close)"""
    try:
        stock = yf.Ticker(ticker)
        current = round(stock.fast_info["lastPrice"], 2)
        try:
            prev_close = round(stock.fast_info["previousClose"], 2)
        except KeyError:
            prev_close = current  # Fallback to avoid division by zero errors
        return current, prev_close
    except Exception:
        return 0.0, 0.0


@cache.memoize(expire=86400)
def get_advanced_metrics(ticker: str) -> dict:
    """Fetches comprehensive financial data for the Deep-Dive report."""
    base_ticker = ticker.split(".")[0]
    if base_ticker == "VISA":
        base_ticker = "V"

    # Attempt FMP first for key ratios if API key exists
    if FMP_KEY:
        try:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{base_ticker}?apikey={FMP_KEY}"
            res = requests.get(url, timeout=5).json()
            if res:
                m = res[0]
                return {
                    "P/E Ratio (TTM)": round(m.get("peRatioTTM", 0), 2),
                    "Debt to Equity": round(m.get("debtToEquityTTM", 0), 2),
                    "Revenue Growth (YoY)": "See yfinance",  # FMP requires separate growth endpoint
                    "Dividend Yield": f"{m.get('dividendYieldPercentageTTM', 0):.2f}%",
                    "Free Cash Flow": f"${m.get('freeCashFlowYieldTTM', 0)*100:.2f}% Yield",
                }
        except Exception:
            pass

    # Fallback/Primary data source: yfinance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Calculate Margin of Safety based on analyst targets
        current = info.get("currentPrice", 0)
        target = info.get("targetMeanPrice", 0)
        mos = ((target - current) / target * 100) if target else 0

        return {
            "Trailing P/E": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Price to Book": info.get("priceToBook", "N/A"),
            "Debt-to-Equity": info.get("debtToEquity", "N/A"),
            "Revenue Growth (YoY)": (
                f"{info.get('revenueGrowth', 0) * 100:.2f}%"
                if info.get("revenueGrowth")
                else "N/A"
            ),
            "Profit Margins": (
                f"{info.get('profitMargins', 0) * 100:.2f}%"
                if info.get("profitMargins")
                else "N/A"
            ),
            "Free Cash Flow": info.get("freeCashflow", "N/A"),
            "52-Week High/Low": f"${info.get('fiftyTwoWeekHigh')} / ${info.get('fiftyTwoWeekLow')}",
            "Analyst Target (Mean)": f"${target}",
            "Recommendation": info.get("recommendationKey", "N/A").upper(),
            "Margin of Safety": f"{mos:.2f}%",
        }
    except Exception:
        return {"Error": "Could not retrieve fundamental data."}


@cache.memoize(expire=3600)
def get_macro_news() -> list:
    """Fetches macro market news. Prefers Finnhub, falls back to NewsAPI."""
    if FINNHUB_KEY:
        try:
            res = requests.get(
                f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}",
                timeout=5,
            )
            data = res.json()
            return [
                {
                    "title": d.get("headline", ""),
                    "publisher": d.get("source", ""),
                    "link": d.get("url", ""),
                }
                for d in data[:5]
            ]
        except Exception:
            pass

    # Fallback to NewsAPI Mock
    try:
        res = requests.get(
            "https://newsapi.org/v2/top-headlines?category=business&apiKey=70d39de976cc4625bc3929766d7a6720",
            timeout=5,
        )
        return res.json().get("articles", [])[:5]
    except Exception:
        return [
            {
                "title": "Markets await new data as volatility continues.",
                "publisher": "System",
            }
        ]


@cache.memoize(expire=3600)
def get_ticker_news(ticker: str, limit: int = 3) -> list:
    """Fetches specific ticker news via yfinance."""
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
