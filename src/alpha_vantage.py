# src/alpha_vantage.py
"""
Alpha Vantage thin client used by the Stock‑Advisor.

All public functions are cached (disk‑cache) to stay well‑inside the
free‑tier limits and to make the CLI snappy.  If the environment variable
ALPHAVANTAGE_API_KEY is missing the functions return ``None`` or an empty
list – the advisor falls back to the existing yfinance / VADER path.
"""

from __future__ import annotations
import traceback
import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional


import requests
from diskcache import Cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# Cache set‑up – reuse the same folder that data_client uses
# ----------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
cache = Cache(CACHE_DIR)

# ----------------------------------------------------------------------
# Helper – load the API key once
# ----------------------------------------------------------------------
_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
log = logging.getLogger(__name__)


def _has_key() -> bool:
    return bool(_API_KEY)


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET request with timeout/exception handling; returns parsed JSON."""
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.debug("AlphaVantage request failed: %s – %s", url, exc)
        raise


# ----------------------------------------------------------------------
# 1️⃣ Core market data
# ----------------------------------------------------------------------
@cache.memoize(expire=300)  # 5 min cache – price updates are frequent
def get_daily_price(ticker: str) -> Tuple[float, float]:
    """
    Returns ``(current_price, previous_close)`` using the free
    TIME_SERIES_DAILY endpoint.  If the call fails ``(0.0, 0.0)`` is
    returned.
    """
    if not _has_key():
        return 0.0, 0.0

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": _API_KEY,
        "outputsize": "compact",  # latest 100 days – more than enough for “today”
    }

    try:
        data = _get(url, params)
        series = data.get("Time Series (Daily)", {})
        if not series:
            return 0.0, 0.0

        # The *most recent* trading day is the first key (sorted descending)
        sorted_dates = sorted(series.keys(), reverse=True)
        today = series[sorted_dates[0]]
        current = float(today["4. close"])
        if len(sorted_dates) > 1:
            prev = float(series[sorted_dates[1]]["4. close"])
        else:
            prev = current
        return round(current, 2), round(prev, 2)
    except Exception:
        return 0.0, 0.0


# ----------------------------------------------------------------------
# 2️⃣ FX – USD → CAD (used throughout the app)
# ----------------------------------------------------------------------
@cache.memoize(expire=3600)  # 1 h cache – FX moves slowly
def get_usd_to_cad() -> float:
    """Realtime USD‑to‑CAD rate via the CURRENCY_EXCHANGE_RATE endpoint."""
    if not _has_key():
        return 1.35  # a sensible static fallback

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "USD",
        "to_currency": "CAD",
        "apikey": _API_KEY,
    }

    try:
        data = _get(url, params)
        rate_info = data.get("Realtime Currency Exchange Rate", {})
        return float(rate_info.get("5. Exchange Rate", 1.35))
    except Exception:
        return 1.35


# ----------------------------------------------------------------------
# 3️⃣ Macro‑level news (Alpha “NEWS_SENTIMENT”)
# ----------------------------------------------------------------------
@cache.memoize(expire=1800)  # 30 min – news isn’t refreshed every second
def get_macro_news(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Returns the latest headline list from Alpha Vantage’s **NEWS_SENTIMENT**
    endpoint (no ticker filter).  If the endpoint is unavailable it falls
    back to the existing news‑api mock (so the CLI never crashes).
    """
    if not _has_key():
        # fallback to the same mock structure data_client used previously
        return [
            {
                "title": "Markets await new data as volatility continues.",
                "publisher": "System",
                "link": "",
            }
        ]

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": _API_KEY,
        "limit": str(limit),
    }

    try:
        news = _get(url, params).get("feed", [])
        # Normalise to a tiny dict the rest of the code expects
        return [
            {
                "title": n.get("title", ""),
                "publisher": n.get("source", ""),
                "link": n.get("url", ""),
                "sentiment_score": n.get("overall_sentiment_score", 0.0),
                "sentiment_label": n.get("overall_sentiment_label", ""),
            }
            for n in news[:limit]
        ]
    except Exception:
        # very defensive – return a single placeholder so the UI still works
        return [
            {
                "title": "Markets await new data as volatility continues.",
                "publisher": "System",
                "link": "",
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
            },
        ]


# ----------------------------------------------------------------------
# 4️⃣ Ticker‑specific news & sentiment
# ----------------------------------------------------------------------
@cache.memoize(expire=1800)
def get_ticker_news(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Alpha’s news endpoint can filter by ticker.  It returns a short list
    with the headline and an *overall_sentiment_score* (‑1 → +1).
    """
    if not _has_key():
        # simple fallback – keeps the CLI usable without Alpha
        return [{"title": f"Could not fetch news for {ticker}.", "publisher": "System"}]

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": _API_KEY,
        "tickers": ticker,
        "limit": str(limit),
    }

    try:
        raw = _get(url, params).get("feed", [])
        return [
            {
                "title": n.get("title", ""),
                "publisher": n.get("source", ""),
                "link": n.get("url", ""),
                "sentiment_score": n.get("overall_sentiment_score", 0.0),
                "sentiment_label": n.get("overall_sentiment_label", ""),
            }
            for n in raw[:limit]
        ]
    except Exception:
        return [{"title": f"Could not fetch news for {ticker}.", "publisher": "System"}]


# ----------------------------------------------------------------------
# 5️⃣ Company fundamental endpoints (overview, income, balance, cash)
# ----------------------------------------------------------------------
@cache.memoize(expire=86400)
def get_company_overview(ticker: str) -> Dict[str, Any]:
    """
    Returns the raw JSON from the *OVERVIEW* endpoint.  If Alpha cannot be
    reached an empty dict is returned.
    """
    if not _has_key():
        return {}

    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": ticker, "apikey": _API_KEY}
    try:

        return _get(url, params)
    except Exception as exc:
        print(exc)

        log.error(
            "Alpha Vantage OVERVIEW failed for %s – %s",
            ticker,
            exc,
        )
        log.debug(traceback.format_exc())
        return {"_error": str(exc), "_traceback": traceback.format_exc()}


@cache.memoize(expire=86400)
def get_income_statement(ticker: str, period: str = "annual") -> List[Dict[str, Any]]:
    """period ∈ {annual, quarterly} – returns a list of dict rows."""
    if not _has_key():
        return []
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker,
        "apikey": _API_KEY,
    }
    try:
        data = _get(url, params)
        key = "annualReports" if period == "annual" else "quarterlyReports"
        return data.get(key, [])
    except Exception:
        return []


@cache.memoize(expire=86400)
def get_balance_sheet(ticker: str, period: str = "annual") -> List[Dict[str, Any]]:
    if not _has_key():
        return []
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "BALANCE_SHEET",
        "symbol": ticker,
        "apikey": _API_KEY,
    }
    try:
        data = _get(url, params)
        key = "annualReports" if period == "annual" else "quarterlyReports"
        return data.get(key, [])
    except Exception:
        return []


@cache.memoize(expire=86400)
def get_cash_flow(ticker: str, period: str = "annual") -> List[Dict[str, Any]]:
    if not _has_key():
        return []
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CASH_FLOW",
        "symbol": ticker,
        "apikey": _API_KEY,
    }
    try:
        data = _get(url, params)
        key = "annualReports" if period == "annual" else "quarterlyReports"
        return data.get(key, [])
    except Exception:
        return []


# ----------------------------------------------------------------------
# 6️⃣ Technical indicators – single‑function generic wrapper
# ----------------------------------------------------------------------
@cache.memoize(expire=300)  # 5 min – short‑term indicator data
def get_technical_indicator(
    ticker: str,
    indicator: str,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close",
    **extra: Any,
) -> Dict[str, Any]:
    """
    Generic wrapper for Alpha Vantage’s **Technical Indicator** APIs.
    Example::

        get_technical_indicator('AAPL', 'SMA', interval='daily',
                               time_period=20, series_type='close')
    Returns the raw JSON dictionary (or empty dict on failure).
    """
    if not _has_key():
        return {}

    url = "https://www.alphavantage.co/query"
    params = {
        "function": indicator,
        "symbol": ticker,
        "interval": interval,
        "time_period": str(time_period),
        "series_type": series_type,
        "apikey": _API_KEY,
    }
    # merge any optional params (e.g. fastperiod, slowperiod for MACD)
    params.update({k: str(v) for k, v in extra.items()})

    try:
        return _get(url, params)
    except Exception:
        return {}


# ----------------------------------------------------------------------
# 7️⃣ Advanced analytics – fixed‑window (e.g. mean, stddev, corr)
# ----------------------------------------------------------------------
@cache.memoize(expire=86400)  # analytics are heavy, cache 1 day
def get_analytics_fixed_window(
    symbols: List[str],
    interval: str = "daily",
    start: Optional[str] = None,
    end: Optional[str] = None,
    calculations: str = "MEAN,STDDEV",
) -> Dict[str, Any]:
    """
    Calls the ANALYTICS_FIXED_WINDOW endpoint. ``symbols`` may be a single
    ticker or a comma‑separated list (max 5 for free keys).  The response
    contains a nested JSON with the requested calculations.
    """
    if not _has_key():
        return {}

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "ANALYTICS_FIXED_WINDOW",
        "symbol": ",".join(symbols),
        "interval": interval,
        "calculations": calculations,
        "apikey": _API_KEY,
    }
    if start:
        params["range"] = start
    if end:
        params["range"] = end

    try:
        return _get(url, params)
    except Exception:
        return {}


# ----------------------------------------------------------------------
# 8️⃣ Market‑wide lists – top gainers / losers (premium endpoint)
# ----------------------------------------------------------------------
@cache.memoize(expire=1800)
def get_top_gainers_losers() -> Dict[str, List[Dict[str, Any]]]:
    """Returns ``{'gainers': [...], 'losers': [...], 'active': [...]}``."""
    if not _has_key():
        return {"gainers": [], "losers": [], "active": []}
    url = "https://www.alphavantage.co/query"
    params = {"function": "TOP_GAINERS_LOSERS", "apikey": _API_KEY}
    try:
        data = _get(url, params)
        # The free version returns a nested dict under each key
        return {
            "gainers": data.get("top_gainers", []),
            "losers": data.get("top_losers", []),
            "active": data.get("most_active", []),
        }
    except Exception:
        return {"gainers": [], "losers": [], "active": []}
