# src/alpha_vantage.py
"""
Alpha Vantage thin client used by the Stock-Advisor.

All public functions are cached (SQLite-backed) to stay well inside the
free-tier limits and to make the CLI snappy.  If the environment variable
ALPHAVANTAGE_API_KEY is missing the functions return ``None`` or an empty
list -- the advisor falls back to the existing yfinance / VADER path.
"""

from __future__ import annotations
import traceback
import os
import logging
from typing import Any, Dict, List, Tuple, Optional

import httpx
from dotenv import load_dotenv

from src.database import cache_get, cache_set

load_dotenv()

# TTL constants (seconds)
TTL_PRICE = 300       # 5 min
TTL_NEWS = 1800       # 30 min
TTL_FUNDAMENTALS = 86400  # 24 h
TTL_TECHNICAL = 300   # 5 min
TTL_ANALYTICS = 86400  # 24 h
TTL_TOP_MOVES = 1800  # 30 min

# Helper -- load the API key once
_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
log = logging.getLogger(__name__)

_http = httpx.Client(timeout=10)


def _has_key() -> bool:
    return bool(_API_KEY)


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET request with timeout/exception handling; returns parsed JSON."""
    try:
        resp = _http.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.debug("AlphaVantage request failed: %s -- %s", url, exc)
        raise


# ----------------------------------------------------------------------
# 1. Core market data
# ----------------------------------------------------------------------
def get_daily_price(ticker: str) -> Tuple[float, float]:
    """
    Returns ``(current_price, previous_close)`` using the free
    TIME_SERIES_DAILY endpoint.  If the call fails ``(0.0, 0.0)`` is
    returned.
    """
    cache_key = f"av:price:{ticker}"
    cached = cache_get(cache_key, TTL_PRICE)
    if cached is not None:
        return tuple(cached)

    if not _has_key():
        return 0.0, 0.0

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": _API_KEY,
        "outputsize": "compact",
    }

    try:
        data = _get(url, params)
        series = data.get("Time Series (Daily)", {})
        if not series:
            return 0.0, 0.0

        sorted_dates = sorted(series.keys(), reverse=True)
        today = series[sorted_dates[0]]
        current = float(today["4. close"])
        if len(sorted_dates) > 1:
            prev = float(series[sorted_dates[1]]["4. close"])
        else:
            prev = current
        result = [round(current, 2), round(prev, 2)]
    except Exception:
        result = [0.0, 0.0]

    cache_set(cache_key, result)
    return tuple(result)


# ----------------------------------------------------------------------
# 2. FX -- USD -> CAD (used throughout the app)
#    NOTE: FX rates are served by data_client.get_usd_to_cad().
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 3. Macro-level news (Alpha "NEWS_SENTIMENT")
# ----------------------------------------------------------------------
def get_macro_news(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Returns the latest headline list from Alpha Vantage's NEWS_SENTIMENT
    endpoint (no ticker filter).
    """
    cache_key = f"av:macro_news:{limit}"
    cached = cache_get(cache_key, TTL_NEWS)
    if cached is not None:
        return cached

    if not _has_key():
        fallback = [{"title": "Markets await new data as volatility continues.", "publisher": "System", "link": ""}]
        cache_set(cache_key, fallback)
        return fallback

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": _API_KEY,
        "limit": str(limit),
    }

    try:
        news = _get(url, params).get("feed", [])
        result = [
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
        result = [
            {
                "title": "Markets await new data as volatility continues.",
                "publisher": "System",
                "link": "",
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
            },
        ]

    cache_set(cache_key, result)
    return result


# ----------------------------------------------------------------------
# 4. Ticker-specific news & sentiment
# ----------------------------------------------------------------------
def get_ticker_news(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Alpha's news endpoint filtered by ticker."""
    cache_key = f"av:news:{ticker}:{limit}"
    cached = cache_get(cache_key, TTL_NEWS)
    if cached is not None:
        return cached

    if not _has_key():
        fallback = [{"title": f"Could not fetch news for {ticker}.", "publisher": "System"}]
        cache_set(cache_key, fallback)
        return fallback

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": _API_KEY,
        "tickers": ticker,
        "limit": str(limit),
    }

    try:
        raw = _get(url, params).get("feed", [])
        result = [
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
        result = [{"title": f"Could not fetch news for {ticker}.", "publisher": "System"}]

    cache_set(cache_key, result)
    return result


# ----------------------------------------------------------------------
# 5. Company fundamental endpoints (overview, income, balance, cash)
# ----------------------------------------------------------------------
def get_company_overview(ticker: str) -> Dict[str, Any]:
    """Returns the raw JSON from the OVERVIEW endpoint."""
    cache_key = f"av:overview:{ticker}"
    cached = cache_get(cache_key, TTL_FUNDAMENTALS)
    if cached is not None:
        return cached

    if not _has_key():
        return {}

    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": ticker, "apikey": _API_KEY}
    try:
        result = _get(url, params)
    except Exception as exc:
        log.error("Alpha Vantage OVERVIEW failed for %s -- %s", ticker, exc)
        log.debug(traceback.format_exc())
        result = {}

    cache_set(cache_key, result)
    return result


def get_income_statement(ticker: str, period: str = "annual") -> List[Dict[str, Any]]:
    """period in {annual, quarterly} -- returns a list of dict rows."""
    cache_key = f"av:income:{ticker}:{period}"
    cached = cache_get(cache_key, TTL_FUNDAMENTALS)
    if cached is not None:
        return cached

    if not _has_key():
        return []

    url = "https://www.alphavantage.co/query"
    params = {"function": "INCOME_STATEMENT", "symbol": ticker, "apikey": _API_KEY}
    try:
        data = _get(url, params)
        key = "annualReports" if period == "annual" else "quarterlyReports"
        result = data.get(key, [])
        if "Information" in data:
            log.warning("AV rate-limited on INCOME_STATEMENT for %s", ticker)
            result = []
    except Exception as exc:
        log.debug("INCOME_STATEMENT failed for %s: %s", ticker, exc)
        result = []

    cache_set(cache_key, result)
    return result


def get_balance_sheet(ticker: str, period: str = "annual") -> List[Dict[str, Any]]:
    cache_key = f"av:balance:{ticker}:{period}"
    cached = cache_get(cache_key, TTL_FUNDAMENTALS)
    if cached is not None:
        return cached

    if not _has_key():
        return []

    url = "https://www.alphavantage.co/query"
    params = {"function": "BALANCE_SHEET", "symbol": ticker, "apikey": _API_KEY}
    try:
        data = _get(url, params)
        key = "annualReports" if period == "annual" else "quarterlyReports"
        result = data.get(key, [])
        if "Information" in data:
            log.warning("AV rate-limited on BALANCE_SHEET for %s", ticker)
            result = []
    except Exception as exc:
        log.debug("BALANCE_SHEET failed for %s: %s", ticker, exc)
        result = []

    cache_set(cache_key, result)
    return result


def get_cash_flow(ticker: str, period: str = "annual") -> List[Dict[str, Any]]:
    cache_key = f"av:cashflow:{ticker}:{period}"
    cached = cache_get(cache_key, TTL_FUNDAMENTALS)
    if cached is not None:
        return cached

    if not _has_key():
        return []

    url = "https://www.alphavantage.co/query"
    params = {"function": "CASH_FLOW", "symbol": ticker, "apikey": _API_KEY}
    try:
        data = _get(url, params)
        key = "annualReports" if period == "annual" else "quarterlyReports"
        result = data.get(key, [])
        if "Information" in data:
            log.warning("AV rate-limited on CASH_FLOW for %s", ticker)
            result = []
    except Exception as exc:
        log.debug("CASH_FLOW failed for %s: %s", ticker, exc)
        result = []

    cache_set(cache_key, result)
    return result


# ----------------------------------------------------------------------
# 6. Technical indicators -- single-function generic wrapper
# ----------------------------------------------------------------------
def get_technical_indicator(
    ticker: str,
    indicator: str,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close",
    **extra: Any,
) -> Dict[str, Any]:
    """Generic wrapper for Alpha Vantage Technical Indicator APIs."""
    extra_part = ":".join(f"{k}={v}" for k, v in sorted(extra.items()))
    cache_key = f"av:tech:{ticker}:{indicator}:{interval}:{time_period}:{series_type}:{extra_part}"
    cached = cache_get(cache_key, TTL_TECHNICAL)
    if cached is not None:
        return cached

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
    params.update({k: str(v) for k, v in extra.items()})

    try:
        result = _get(url, params)
        if "Information" in result:
            log.warning("AV rate-limited on %s for %s", indicator, ticker)
            result = {}
    except Exception as exc:
        log.debug("Technical indicator %s failed for %s: %s", indicator, ticker, exc)
        result = {}

    cache_set(cache_key, result)
    return result


# ----------------------------------------------------------------------
# 7. Advanced analytics -- fixed-window
# ----------------------------------------------------------------------
def get_analytics_fixed_window(
    symbols: List[str],
    interval: str = "daily",
    start: Optional[str] = None,
    end: Optional[str] = None,
    calculations: str = "MEAN,STDDEV",
) -> Dict[str, Any]:
    """Calls the ANALYTICS_FIXED_WINDOW endpoint."""
    sym_key = ",".join(sorted(symbols))
    cache_key = f"av:analytics:{sym_key}:{interval}:{start}:{end}:{calculations}"
    cached = cache_get(cache_key, TTL_ANALYTICS)
    if cached is not None:
        return cached

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
        result = _get(url, params)
    except Exception:
        result = {}

    cache_set(cache_key, result)
    return result


# ----------------------------------------------------------------------
# 8. Market-wide lists -- top gainers / losers
# ----------------------------------------------------------------------
def get_top_gainers_losers() -> Dict[str, List[Dict[str, Any]]]:
    """Returns {'gainers': [...], 'losers': [...], 'active': [...]}."""
    cache_key = "av:top_gainers_losers"
    cached = cache_get(cache_key, TTL_TOP_MOVES)
    if cached is not None:
        return cached

    if not _has_key():
        return {"gainers": [], "losers": [], "active": []}

    url = "https://www.alphavantage.co/query"
    params = {"function": "TOP_GAINERS_LOSERS", "apikey": _API_KEY}
    try:
        data = _get(url, params)
        result = {
            "gainers": data.get("top_gainers", []),
            "losers": data.get("top_losers", []),
            "active": data.get("most_active", []),
        }
    except Exception:
        result = {"gainers": [], "losers": [], "active": []}

    cache_set(cache_key, result)
    return result
