import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import yfinance as yf
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.database import init_db, cache_get, cache_set

log = logging.getLogger(__name__)

init_db()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
FMP_KEY = os.getenv("FMP_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_API_KEY", "")

_http = httpx.Client(timeout=10)

# TTL constants (seconds)
TTL_PRICE = 300        # 5 min – live prices
TTL_FX = 3600          # 1 h – exchange rates
TTL_NEWS = 3600        # 1 h – news headlines
TTL_INFO = 86400       # 24 h – ticker fundamentals
TTL_METRICS = 86400    # 24 h – advanced metrics
TTL_HISTORY = 3600      # 1 h – daily OHLCV bars


# ---------------------------------------------------------------------------
# FX rate
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def get_usd_to_cad() -> float:
    cached = cache_get("fx:usd_cad", TTL_FX)
    if cached is not None:
        return cached

    primary_url = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
    fallback_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
    try:
        try:
            response = _http.get(primary_url)
            response.raise_for_status()
        except httpx.HTTPError:
            response = _http.get(fallback_url)
            response.raise_for_status()
        rate = response.json()["usd"]["cad"]
    except Exception:
        rate = 1.35

    cache_set("fx:usd_cad", rate)
    return rate


# ---------------------------------------------------------------------------
# Ticker info / fundamentals
# ---------------------------------------------------------------------------

def get_ticker_info(ticker: str) -> dict:
    """Returns the full yfinance info dict for a ticker (cached 24h)."""
    cache_key = f"info:{ticker}"
    cached = cache_get(cache_key, TTL_INFO)
    if cached is not None:
        return cached

    try:
        result = yf.Ticker(ticker).info or {}
    except Exception:
        result = {}

    cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Live price
# ---------------------------------------------------------------------------

def get_current_price(ticker: str) -> tuple[float, float]:
    """Returns (current_price, previous_close)."""
    cache_key = f"price:{ticker}"
    cached = cache_get(cache_key, TTL_PRICE)
    if cached is not None:
        return tuple(cached)

    try:
        stock = yf.Ticker(ticker)
        current = round(stock.fast_info["lastPrice"], 2)
        try:
            prev_close = round(stock.fast_info["previousClose"], 2)
        except KeyError:
            prev_close = current
        result = [current, prev_close]
    except Exception:
        result = [0.0, 0.0]

    cache_set(cache_key, result)
    return tuple(result)


def get_current_prices_batch(tickers: list[str]) -> dict[str, tuple[float, float]]:
    """Fetch prices for multiple tickers concurrently via thread pool."""
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as pool:
        results = list(pool.map(get_current_price, tickers))
    return dict(zip(tickers, results))


# ---------------------------------------------------------------------------
# Historical prices
# ---------------------------------------------------------------------------

def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV history for *ticker* (cached 1 h).

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume.
    """
    cache_key = f"history:{ticker}:{period}"
    cached = cache_get(cache_key, TTL_HISTORY)
    if cached is not None:
        if not cached.get("dates"):
            return pd.DataFrame()
        return pd.DataFrame(
            cached["data"], index=pd.DatetimeIndex(cached["dates"])
        )

    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is not None and not df.empty:
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            # Normalize to tz-naive UTC to avoid "Mixed timezones" errors in pandas 3.x
            dates = df.index
            if dates.tz is not None:
                dates = dates.tz_convert("UTC").tz_localize(None)
            result = {
                "dates": [str(d) for d in dates],
                "data": {c: df[c].tolist() for c in cols},
            }
        else:
            result = {"dates": [], "data": {}}
    except Exception:
        result = {"dates": [], "data": {}}

    cache_set(cache_key, result)

    if result["dates"]:
        return pd.DataFrame(
            result["data"], index=pd.DatetimeIndex(result["dates"])
        )
    return pd.DataFrame()


def get_close_prices(
    tickers: list[str], period: str = "1y", min_rows: int = 1
) -> pd.DataFrame:
    """Fetch aligned daily close prices for multiple tickers.

    Tickers with fewer than *min_rows* of history are excluded, so one
    sparse listing doesn't collapse the shared date index.

    Returns a DataFrame with tickers as columns and a shared date index.
    Rows with any NaN are dropped.
    """
    if not tickers:
        return pd.DataFrame()
    frames = {}
    for ticker in tickers:
        df = get_price_history(ticker, period)
        if not df.empty and "Close" in df.columns and len(df) >= min_rows:
            frames[ticker] = df["Close"]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna()


# ---------------------------------------------------------------------------
# Advanced metrics (FMP + yfinance fallback)
# ---------------------------------------------------------------------------

def get_advanced_metrics(ticker: str) -> dict:
    """Fetches comprehensive financial data for the Deep-Dive report."""
    cache_key = f"metrics:{ticker}"
    cached = cache_get(cache_key, TTL_METRICS)
    if cached is not None:
        return cached

    base_ticker = ticker.split(".")[0]
    if base_ticker == "VISA":
        base_ticker = "V"

    result = None
    if FMP_KEY:
        try:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{base_ticker}?apikey={FMP_KEY}"
            res = _http.get(url).json()
            if res:
                m = res[0]
                result = {
                    "P/E Ratio (TTM)": round(m.get("peRatioTTM", 0), 2),
                    "Debt to Equity": round(m.get("debtToEquityTTM", 0), 2),
                    "Revenue Growth (YoY)": "See yfinance",
                    "Dividend Yield": f"{m.get('dividendYieldPercentageTTM', 0):.2f}%",
                    "Free Cash Flow": f"${m.get('freeCashFlowYieldTTM', 0)*100:.2f}% Yield",
                }
        except Exception:
            pass

    if result is None:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current = info.get("currentPrice", 0)
            target = info.get("targetMeanPrice", 0)
            mos = ((target - current) / target * 100) if target else 0
            result = {
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
                    if info.get('profitMargins')
                    else "N/A"
                ),
                "Free Cash Flow": info.get("freeCashflow", "N/A"),
                "52-Week High/Low": f"${info.get('fiftyTwoWeekHigh')} / ${info.get('fiftyTwoWeekLow')}",
                "Analyst Target (Mean)": f"${target}",
                "Recommendation": info.get("recommendationKey", "N/A").upper(),
                "Margin of Safety": f"{mos:.2f}%",
            }
        except Exception:
            result = {"Error": "Could not retrieve fundamental data."}

    cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def get_macro_news() -> list:
    """Fetches macro market news (Finnhub → NewsAPI → fallback)."""
    cached = cache_get("macro_news", TTL_NEWS)
    if cached is not None:
        return cached

    result = None
    if FINNHUB_KEY:
        try:
            res = _http.get(
                f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
            )
            data = res.json()
            result = [
                {"title": d.get("headline", ""), "publisher": d.get("source", ""), "link": d.get("url", "")}
                for d in data[:5]
            ]
        except Exception:
            pass

    if result is None and NEWSAPI_KEY:
        try:
            res = _http.get(
                f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWSAPI_KEY}"
            )
            result = res.json().get("articles", [])[:5]
        except Exception:
            pass

    if result is None:
        result = [{"title": "Markets await new data as volatility continues.", "publisher": "System"}]

    cache_set("macro_news", result)
    return result


def get_ticker_news(ticker: str, limit: int = 3) -> list:
    """Fetches specific ticker news via yfinance."""
    cache_key = f"news:{ticker}:{limit}"
    cached = cache_get(cache_key, TTL_NEWS)
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(ticker)
        result = []
        for a in stock.news[:limit]:
            # yfinance nests news under "content" — handle both old and new formats
            c = a.get("content") or a
            result.append({
                "title": c.get("title", ""),
                "publisher": (c.get("provider") or {}).get("displayName", ""),
                "link": (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url", ""),
            })
    except Exception:
        result = [{"title": f"Could not fetch news for {ticker}.", "publisher": "System"}]

    cache_set(cache_key, result)
    return result


def get_ticker_news_batch(tickers: list[str], limit: int = 3) -> dict[str, list]:
    """Fetch news for multiple tickers concurrently via thread pool."""
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as pool:
        results = list(pool.map(lambda t: get_ticker_news(t, limit), tickers))
    return dict(zip(tickers, results))
