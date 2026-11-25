"""Market leaders snapshot: top market-cap names (US + Canada) with all-time-high
distance and analyst opinion counts.

Drop this file in as ``src/market_leaders.py``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any

import httpx
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
            row = df[df["period"] == "0m"].iloc[0] if "0m" in set(df["period"]) else df.iloc[0]
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


# ---------------------------------------------------------------------------
# Persistence to Supabase (market-leaders / stock-status history)
# ---------------------------------------------------------------------------

_SUPABASE_TABLE_STATUS_RUNS = "stock_status_runs"
_SUPABASE_TABLE_STATUS_ROWS = "stock_status_rows"


def _status_rest_config() -> tuple[str, str]:
    """Return (rest_url, api_key) for Supabase REST, or ('', '')."""
    import os

    rest_url = os.getenv("DATABASE_REST_URL", "")
    api_key = os.getenv(
        "SUPABASE_PUBLISHABLE_KEY", os.getenv("SUPABASE_ANON_KEY", "")
    )
    return rest_url, api_key


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars into plain JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _status_payload(
    row: dict[str, Any], run_id: Any, group_name: str, rank: int
) -> dict[str, Any]:
    """Flatten one market-leader status into a `stock_status_rows` row."""
    analysts = row.get("analysts") or {}
    consensus, _ = consensus_label(analysts)
    return _json_safe({
        "run_id": run_id,
        "group_name": group_name,
        "rank": rank,
        "symbol": row.get("symbol"),
        "company": row.get("company") or None,
        "currency": row.get("currency") or None,
        "price": row.get("price"),
        "day_pct": row.get("day_pct"),
        "market_cap": row.get("market_cap"),
        "pe": row.get("pe"),
        "fwd_pe": row.get("fwd_pe"),
        "div_yield": row.get("div_yield"),
        "high_52w": row.get("high_52w"),
        "low_52w": row.get("low_52w"),
        "ath": row.get("ath"),
        "ath_date": row.get("ath_date"),
        "from_ath_pct": row.get("from_ath_pct"),
        "target": row.get("target"),
        "upside_pct": row.get("upside_pct"),
        "strong_buy": analysts.get("strong_buy"),
        "buy": analysts.get("buy"),
        "hold": analysts.get("hold"),
        "sell": analysts.get("sell"),
        "strong_sell": analysts.get("strong_sell"),
        "total_analysts": analysts.get("total"),
        "consensus": consensus,
    })


def persist_snapshot(
    groups: dict[str, list[dict[str, Any]]],
    run_date: date | str | None = None,
    us_n: int = 15,
    ca_n: int = 5,
    custom: list[str] | None = None,
    sort: str = "mcap",
) -> bool:
    """Upsert a market-leaders snapshot (run + status rows) into Supabase.

    Requires ``DATABASE_REST_URL`` and a Supabase API key.  The run is keyed on
    ``run_date`` so a same-day re-run overwrites it, and its rows are replaced
    so stale names never linger.  Returns True on success.
    """
    if not groups or not any(groups.values()):
        log.warning("No market-leader rows to persist — skipping stock-status persist.")
        return False

    rest_url, api_key = _status_rest_config()
    if not rest_url or not api_key:
        log.warning(
            "DATABASE_REST_URL / SUPABASE_PUBLISHABLE_KEY not set — "
            "skipping stock-status persist."
        )
        return False

    if isinstance(run_date, str):
        run_date = date.fromisoformat(run_date)
    run_date = run_date or date.today()
    generated_at = datetime.now().astimezone()

    payloads: list[dict[str, Any]] = []
    rows_by_group = {name: rows for name, rows in groups.items() if rows}
    total = sum(len(rows) for rows in rows_by_group.values())
    run_payload = _json_safe({
        "run_date": run_date.isoformat(),
        "generated_at": generated_at.isoformat(),
        "universe": "custom" if custom else ", ".join(rows_by_group),
        "sort": sort,
        "us_count": us_n if not custom else 0,
        "ca_count": ca_n if not custom else 0,
        "total_count": total,
        "groups": list(rows_by_group),
    })

    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{rest_url}/{_SUPABASE_TABLE_STATUS_RUNS}?on_conflict=run_date",
                json=run_payload,
                headers={
                    **headers,
                    "Prefer": "resolution=merge-duplicates,return=representation",
                },
            )
            resp.raise_for_status()
            rows = resp.json()
            run_id = rows[0]["id"] if rows else None
            if run_id is None:
                log.error("Could not determine stock-status run_id from Supabase response.")
                return False

            # Replace the run's rows so a shrinking list cannot leave stale names.
            try:
                resp = client.delete(
                    f"{rest_url}/{_SUPABASE_TABLE_STATUS_ROWS}?run_id=eq.{run_id}",
                    headers={"apikey": api_key},
                )
                resp.raise_for_status()
            except Exception as exc:
                log.warning("Could not clear previous stock-status rows: %s", exc)

            for group_name, group_rows in rows_by_group.items():
                for rank, row in enumerate(group_rows, start=1):
                    payloads.append(
                        _status_payload(row, run_id, group_name, rank)
                    )
            for i in range(0, len(payloads), 50):
                resp = client.post(
                    f"{rest_url}/{_SUPABASE_TABLE_STATUS_ROWS}"
                    "?on_conflict=run_id,group_name,symbol",
                    json=payloads[i : i + 50],
                    headers=headers,
                )
                resp.raise_for_status()

        log.info(
            "Persisted stock-status snapshot (id=%s) for %s with %d rows.",
            run_id, run_date, len(payloads),
        )
        return True
    except Exception as exc:
        log.error("Failed to persist stock-status snapshot: %s", exc)
        return False


def fetch_snapshot(run_date: date | str | None = None) -> dict[str, Any] | None:
    """Load a persisted market-leaders snapshot (run + rows) from Supabase.

    Returns None when REST is unconfigured, nothing has been persisted yet, or
    the request fails.  With *run_date* the exact day is preferred, falling
    back to the most recent run so the newsletter/dashboard always show
    something useful.
    """
    rest_url, api_key = _status_rest_config()
    if not rest_url or not api_key:
        return None

    if isinstance(run_date, date):
        run_date = run_date.isoformat()

    headers = {"apikey": api_key, "Accept": "application/json"}
    try:
        with httpx.Client(timeout=20) as client:
            rows: list[dict[str, Any]] = []
            if run_date:
                resp = client.get(
                    f"{rest_url}/{_SUPABASE_TABLE_STATUS_RUNS}"
                    f"?select=*&run_date=eq.{run_date}&limit=1",
                    headers=headers,
                )
                resp.raise_for_status()
                rows = resp.json()
                if not rows:
                    log.warning(
                        "No stock-status snapshot persisted for %s — using latest.",
                        run_date,
                    )
            if not rows:
                resp = client.get(
                    f"{rest_url}/{_SUPABASE_TABLE_STATUS_RUNS}"
                    "?select=*&order=run_date.desc&limit=1",
                    headers=headers,
                )
                resp.raise_for_status()
                rows = resp.json()
            if not rows:
                return None

            run = rows[0]
            resp = client.get(
                f"{rest_url}/{_SUPABASE_TABLE_STATUS_ROWS}"
                f"?select=*&run_id=eq.{run['id']}&order=group_name.asc,rank.asc",
                headers=headers,
            )
            resp.raise_for_status()
            status_rows = resp.json()

        return {"run": run, "rows": status_rows}
    except Exception as exc:
        log.warning("Could not fetch stock-status snapshot from Supabase: %s", exc)
        return None
