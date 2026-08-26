"""
Stock screener that surfaces high-conviction "buy right now" candidates.

Uses a static universe (S&P 500, TSX 60) or a custom ticker list, then screens
each name on live analyst consensus (strong-buy/buy/hold/sell counts) and the
mean price-target upside via yfinance.  Candidates that pass a net-bullish
filter are ranked by a blended score and enriched with fundamentals
(market cap, P/E, sector) so the CLI can render a buy list.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx
import yfinance as yf
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src import data_client, ticker_map
from src.database import cache_get, cache_set

log = logging.getLogger(__name__)

# yfinance prints per-symbol HTTP 404 errors to stderr for names with no
# coverage (e.g. class shares we can't resolve).  Suppress that noise.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

_UA = "StockAdvisor/1.0 (personal finance research CLI)"
_http = httpx.Client(timeout=20, headers={"User-Agent": _UA})

# Cache TTLs (seconds)
TTL_UNIVERSE = 7 * 86400    # constituent list — rarely changes
TTL_ANALYST = 24 * 3600     # analyst consensus — updates daily

# Universe name → (Wikipedia page, column header holding the company name).
UNIVERSES: dict[str, tuple[str, str]] = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Security"),
    "tsx60": ("https://en.wikipedia.org/wiki/S%26P/TSX_60", "Company"),
}

DEFAULT_WORKERS = 10


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def fetch_universe(name: str) -> dict[str, str]:
    """Return ``{symbol: company_name}`` for a named universe (cached 7d)."""
    if name not in UNIVERSES:
        return {}

    cache_key = f"universe:{name}"
    cached = cache_get(cache_key, TTL_UNIVERSE)
    if cached is not None:
        return cached

    url, name_col = UNIVERSES[name]
    result: dict[str, str] = {}
    try:
        resp = _http.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tbl in soup.find_all("table", class_="wikitable"):
            headers = [th.get_text(strip=True) for th in tbl.find_all("th")][:12]
            sym_idx = next(
                (i for i, h in enumerate(headers) if "symbol" in h.lower() or "ticker" in h.lower()),
                None,
            )
            name_idx = next(
                (i for i, h in enumerate(headers) if h.lower() == name_col.lower()),
                None,
            )
            if sym_idx is None:
                continue
            for row in tbl.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) <= (name_idx if name_idx is not None else sym_idx):
                    continue
                # Normalize: "BRK.B" → "BRK-B" (yfinance class-share format);
                # Wikipedia TSX/NASDAQ lists never contain a "." in the symbol.
                sym = cells[sym_idx].get_text(strip=True).replace(".", "-").upper().strip()
                if not sym:
                    continue
                company = cells[name_idx].get_text(strip=True) if name_idx is not None else ""
                result[sym] = company
            break
    except Exception as exc:
        log.debug("universe fetch failed for %s: %s", name, exc)
        result = {}

    cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Analyst consensus
# ---------------------------------------------------------------------------

def _analyst(ticker: str) -> Optional[dict[str, Any]]:
    """Fetch analyst consensus + price target for one ticker (cached 24h)."""
    cache_key = f"screen:analyst:{ticker}"
    cached = cache_get(cache_key, TTL_ANALYST)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker)
        rec = t.recommendations_summary
        pt = t.analyst_price_targets
    except Exception:
        return None

    if rec is None or rec.empty:
        return None
    row = rec.iloc[0]
    current = (pt or {}).get("current")
    target = (pt or {}).get("mean")
    if not target or not current:
        return None

    out = {
        "current": float(current),
        "target": float(target),
        "strong_buy": int(row.get("strongBuy", 0) or 0),
        "buy": int(row.get("buy", 0) or 0),
        "hold": int(row.get("hold", 0) or 0),
        "sell": int(row.get("sell", 0) or 0),
        "strong_sell": int(row.get("strongSell", 0) or 0),
    }
    cache_set(cache_key, out)
    return out


def _score(d: dict[str, Any]) -> float:
    """Blend analyst consensus, target upside and coverage into 0..4-ish rank."""
    total = max(d["total"], 1)
    consensus = (2.0 * d["strong_buy"] + d["buy"]) / total
    upside = max(-1.0, min(d["upside"], 3.0))
    coverage = min(d["total"], 40) / 40.0
    return 2.0 * consensus + 1.5 * upside + 0.5 * coverage


def rating_label(d: dict[str, Any]) -> tuple[str, str]:
    """Return (label, rich color) for an analyst-consensus row."""
    total = max(d["total"], 1)
    ratio = (2.0 * d["strong_buy"] + d["buy"]) / total
    if ratio >= 1.0 and d["strong_buy"] >= 2:
        return "Strong Buy", "green"
    if ratio >= 0.8:
        return "Buy", "green"
    if ratio >= 0.6:
        return "Moderate Buy", "yellow"
    return "Hold", "yellow"


# ---------------------------------------------------------------------------
# Screening pipeline
# ---------------------------------------------------------------------------

def _resolve_symbols(symbols: list[str]) -> tuple[list[str], dict[str, str]]:
    """Map Canadian tickers to US equivalents for screening; dedupe."""
    resolved = ticker_map.resolve_tickers(symbols)
    out: list[str] = []
    names: dict[str, str] = {}
    for s in symbols:
        us = resolved.get(s, s)
        if us not in out:
            out.append(us)
            names[us] = s
    return out, names


def screen(
    universe: str = "sp500",
    custom: Optional[str] = None,
    limit: int = 10,
    min_analysts: int = 3,
    workers: int = DEFAULT_WORKERS,
) -> list[dict[str, Any]]:
    """Screen a universe for high-conviction buys and return the top *limit*.

    Returns enriched rows sorted by score descending.  Each row contains:
    symbol, company, name, current, target, upside, rating counts, total,
    score, market_cap, pe, sector.
    """
    if custom:
        symbols = [s.strip().upper() for s in custom.split(",") if s.strip()]
        if not symbols:
            raise ValueError("No tickers provided.")
        symbols, names = _resolve_symbols(symbols)
    else:
        names: dict[str, str] = {}
        for part in universe.split(","):
            part = part.strip()
            if part:
                names.update(fetch_universe(part))
        symbols = list(names.keys())
        if not symbols:
            raise ValueError(f"Could not fetch universe {universe!r}. Try --tickers.")

    if len(symbols) < 2:
        raise ValueError("Need at least 2 tickers to screen.")

    with ThreadPoolExecutor(max_workers=min(workers, len(symbols))) as pool:
        results = list(pool.map(_analyst, symbols))

    rows: list[dict[str, Any]] = []
    for sym, data in zip(symbols, results):
        if data is None:
            continue
        data = dict(data)
        data["total"] = (
            data["strong_buy"] + data["buy"] + data["hold"]
            + data["sell"] + data["strong_sell"]
        )
        if data["total"] < min_analysts:
            continue
        bull = data["strong_buy"] + data["buy"]
        bear = data["hold"] + data["sell"] + data["strong_sell"]
        if bull <= bear:
            continue
        cur = data["current"]
        data["upside"] = (data["target"] - cur) / cur if cur > 0 else 0.0
        if data["upside"] <= 0:
            continue
        data["symbol"] = sym
        data["company"] = names.get(sym, "")
        data["score"] = _score(data)
        rows.append(data)

    rows.sort(key=lambda r: r["score"], reverse=True)
    return _enrich(rows[:limit], workers=workers)


def _enrich(rows: list[dict[str, Any]], workers: int = 8) -> list[dict[str, Any]]:
    """Attach fundamentals + company name to each row (via 24h-cached info)."""
    if not rows:
        return rows

    def info(ticker: str) -> dict[str, Any]:
        d = data_client.get_ticker_info(ticker) or {}
        return {
            "name": d.get("longName") or d.get("shortName") or ticker,
            "market_cap": d.get("marketCap"),
            "pe": d.get("trailingPE"),
            "sector": d.get("sector", ""),
        }

    with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as pool:
        extras = list(pool.map(lambda r: info(r["symbol"]), rows))
    for row, e in zip(rows, extras):
        row["company"] = e["name"] or row.get("company", "")
        row["name"] = e["name"]
        row["market_cap"] = e["market_cap"]
        row["pe"] = e["pe"]
        row["sector"] = e["sector"]
    return rows


# ---------------------------------------------------------------------------
# Deep-dive narrative
# ---------------------------------------------------------------------------

def deep_dive(picks: list[dict[str, Any]], top: int = 3) -> str:
    """Ask the AI to explain *why* each top pick is a buy right now."""
    if not picks:
        return ""
    picks = picks[:top]

    news_batch = data_client.get_ticker_news_batch(
        [p["symbol"] for p in picks], limit=4
    )

    picks_block = ""
    for p in picks:
        mcap = (p.get("market_cap") or 0) / 1e9
        pe = p.get("pe") if p.get("pe") else "n/a"
        label, _ = rating_label(p)
        picks_block += (
            f"- **{p['symbol']}** — {p.get('company') or p.get('name', '')} | "
            f"Price ${p['current']:,.2f}, Target ${p['target']:,.2f} "
            f"(+{p['upside'] * 100:.0f}%) | {label} ({p['total']} analysts) | "
            f"{p.get('sector', 'n/a')} | P/E {pe} | "
            f"Market cap ${mcap:,.1f}B\n"
        )

    news_block = ""
    for p in picks:
        articles = news_batch.get(p["symbol"], [])
        if articles:
            news_block += f"\n**{p['symbol']} — recent headlines:**\n"
            for a in articles[:3]:
                news_block += f"- {a.get('title', '')}\n"

    prompt = f"""
You are a senior equity analyst. For each stock below, give a concise
**"Why it's a buy right now"** note: key catalysts, valuation take,
one material risk, and a 1-line thesis. 2-4 sentences each, bullet points,
Markdown. Be specific and avoid generic platitudes.

**TOP PICKS (analyst consensus + fundamentals)**
{picks_block}

**RECENT NEWS**
{news_block}
"""

    from src.advisor import _gemini_generate

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
    def _generate(prompt: str) -> str:
        response = _gemini_generate(prompt)
        if not response:
            raise RuntimeError("Gemini returned no content")
        return response

    try:
        response = _generate(prompt)
    except Exception:
        response = None
    return response.strip() if response else "AI analysis unavailable right now."
