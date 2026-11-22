"""Market leaders snapshot: top market-cap names (US + Canada) with all-time-high
distance and analyst opinion counts.

Drop this file in as ``src/market_leaders.py``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf

from src.database import cache_get, cache_set
from src.data_client import get_ticker_info, get_current_price

log = logging.getLogger(__name__)

TTL_ATH = 6 * 3600  # 6 h
TTL_ANALYST = 6 * 3600  # 6 h

# Candidate pools. They are re-ranked by live market cap at runtime, so the
# order here doesn't matter and a name that drifts in or out is handled
# automatically. Add tickers freely if you want a wider net.
US_CANDIDATES = [
    "NVDA",
    "MSFT",
    "AAPL",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "BRK-B",
    "LLY",
    "WMT",
    "JPM",
    "V",
    "ORCL",
    "MA",
    "XOM",
    "NFLX",
    "COST",
    "JNJ",
    "PLTR",
    "HD",
    "BAC",
    "ABBV",
    "PG",
    "CVX",
    "AMD",
    "KO",
    "UNH",
]
CA_CANDIDATES = [
    "RY.TO",
    "TD.TO",
    "SHOP.TO",
    "ENB.TO",
    "CNQ.TO",
    "BN.TO",
    "CP.TO",
    "BMO.TO",
    "BNS.TO",
    "CNR.TO",
    "CSU.TO",
    "ATD.TO",
    "TRI.TO",
    "SU.TO",
    "MFC.TO",
    "AEM.TO",
]


def currency_of(ticker: str) -> str:
    return "CAD" if ticker.upper().endswith((".TO", ".V", ".NE")) else "USD"


# ---------------------------------------------------------------------------
# All-time high
# ---------------------------------------------------------------------------


def get_all_time_high(ticker: str) -> dict:
    """Return {"ath": float, "ath_date": "YYYY-MM-DD"} using the full price history.

    Uses intraday highs and split-adjusted (but NOT dividend-adjusted) prices so
    the number matches what charting sites show as the "all-time high".
    """
    key = f"ath:{ticker}"
    cached = cache_get(key, TTL_ATH)
    if cached is not None:
        return cached

    result = {"ath": 0.0, "ath_date": None}
    try:
        df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        if df is not None and not df.empty and "High" in df.columns:
            highs = df["High"].dropna()
            if not highs.empty:
                result = {
                    "ath": round(float(highs.max()), 2),
                    "ath_date": str(highs.idxmax().date()),
                }
    except Exception as e:  # network / rate limit
        log.warning("ATH fetch failed for %s: %s", ticker, e)
        return result  # don't cache failures

    if result["ath"] > 0:
        cache_set(key, result)
    return result


# ---------------------------------------------------------------------------
# Analyst opinions
# ---------------------------------------------------------------------------


def get_analyst_breakdown(ticker: str) -> dict:
    """Return counts of strong buy / buy / hold / sell / strong sell."""
    key = f"analyst_breakdown:{ticker}"
    cached = cache_get(key, TTL_ANALYST)
    if cached is not None:
        return cached

    out = {
        "strong_buy": 0,
        "buy": 0,
        "hold": 0,
        "sell": 0,
        "strong_sell": 0,
        "total": 0,
    }
    try:
        df = yf.Ticker(ticker).recommendations_summary
        if df is not None and not df.empty:
            # Row "0m" is the current month; fall back to the first row.
            row = (
                df[df["period"] == "0m"].iloc[0]
                if "0m" in set(df["period"])
                else df.iloc[0]
            )
            out = {
                "strong_buy": int(row.get("strongBuy", 0)),
                "buy": int(row.get("buy", 0)),
                "hold": int(row.get("hold", 0)),
                "sell": int(row.get("sell", 0)),
                "strong_sell": int(row.get("strongSell", 0)),
            }
            out["total"] = sum(out.values())
    except Exception as e:
        log.warning("Analyst breakdown failed for %s: %s", ticker, e)

    if out["total"] > 0:
        cache_set(key, out)
    return out


def consensus_label(b: dict) -> tuple[str, str]:
    """(label, rich color) from the breakdown, using a -2..+2 weighted average."""
    n = b.get("total", 0)
    if not n:
        return "N/A", "dim"
    score = (2 * b["strong_buy"] + b["buy"] - b["sell"] - 2 * b["strong_sell"]) / n
    if score >= 1.2:
        return "Strong Buy", "bold green"
    if score >= 0.5:
        return "Buy", "green"
    if score > -0.5:
        return "Hold", "yellow"
    if score > -1.2:
        return "Sell", "red"
    return "Strong Sell", "bold red"


# ---------------------------------------------------------------------------
# Per-ticker status + ranking
# ---------------------------------------------------------------------------


def build_status(ticker: str) -> dict:
    info = get_ticker_info(ticker)
    price, prev = get_current_price(ticker)
    if price <= 0:
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        prev = float(info.get("previousClose") or price)

    ath = get_all_time_high(ticker)
    ath_price = max(ath["ath"], price)  # price may already be a fresh ATH
    from_ath = ((price / ath_price) - 1) * 100 if ath_price > 0 else None

    target = info.get("targetMeanPrice")
    upside = ((target / price) - 1) * 100 if target and price > 0 else None

    return {
        "symbol": ticker,
        "company": info.get("shortName") or info.get("longName") or ticker,
        "currency": currency_of(ticker),
        "price": price,
        "day_pct": ((price / prev) - 1) * 100 if prev > 0 else 0.0,
        "market_cap": info.get("marketCap") or 0,
        "pe": info.get("trailingPE"),
        "fwd_pe": info.get("forwardPE"),
        # yfinance >= 0.2.54 reports dividendYield already in percent.
        "div_yield": info.get("dividendYield"),
        "high_52w": info.get("fiftyTwoWeekHigh"),
        "low_52w": info.get("fiftyTwoWeekLow"),
        "ath": ath_price if ath["ath"] > 0 else None,
        "ath_date": ath["ath_date"],
        "from_ath_pct": from_ath,
        "target": target,
        "upside_pct": upside,
        "analysts": get_analyst_breakdown(ticker),
    }


def top_by_market_cap(candidates: list[str], n: int) -> list[str]:
    """Rank candidates by live market cap and return the top *n* symbols."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        infos = list(pool.map(get_ticker_info, candidates))
    caps = [(t, (i or {}).get("marketCap") or 0) for t, i in zip(candidates, infos)]
    caps.sort(key=lambda x: x[1], reverse=True)
    return [t for t, c in caps[:n] if c > 0]


def get_market_leaders(
    us_n: int = 15, ca_n: int = 5, custom: list[str] | None = None
) -> dict[str, list[dict]]:
    """Return {"US": [...], "Canada": [...]} (or {"Custom": [...]}), sorted by market cap."""
    if custom:
        groups = {"Custom": [t.strip().upper() for t in custom if t.strip()]}
    else:
        groups = {}
        if us_n > 0:
            groups["US"] = top_by_market_cap(US_CANDIDATES, us_n)
        if ca_n > 0:
            groups["Canada"] = top_by_market_cap(CA_CANDIDATES, ca_n)

    out: dict[str, list[dict]] = {}
    for name, tickers in groups.items():
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(build_status, tickers))
        out[name] = sorted(rows, key=lambda r: r["market_cap"], reverse=True)
    return out
