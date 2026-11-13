"""
Multi-factor stock screener that surfaces high-conviction "buy right now"
candidates instead of simply ranking analyst price-target upside.

The screen is deliberately two-stage so it stays usable on a 500-name universe:

1. **Cheap analyst pass** — fetch consensus ratings and price targets for every
   ticker, drop names with thin coverage / net-bearish ratings / no upside, and
   keep a candidate pool (default: 6× the requested result count).
2. **Deep investigation** — for that pool only, pull fundamentals, price
   history, recommendation trends and recent rating/price-target revisions.
   Each name is then scored on five weighted pillars:

   * Analyst conviction (28%) — consensus, rating trend, target confidence,
     coverage and upside.
   * Business quality (24%) — profitability, returns on capital, cash generation,
     balance-sheet strength.
   * Valuation (18%) — multiples relative to growth and cash flow (PEG, forward
     P/E, EV/EBITDA, FCF yield, etc.).
   * Growth (18%) — revenue, earnings and forward-EPS growth.
   * Momentum (12%) — trend, relative returns and RSI positioning.

   Material risks (negative FCF, over-leverage, high beta, stale-looking targets,
   very high short interest, micro-cap liquidity) subtract from the final score.

The result is a ranked list with component scores, tags like **Underrated**,
**Quality**, **Value**, **Growth** and **Revisions up**, plus a plain-English
reason for each rank.  The optional AI deep-dive is then asked to challenge the
quant model — not merely repeat the analyst target.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx
import pandas as pd
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
TTL_ANALYST = 24 * 3600     # analyst consensus / revisions — updates daily
TTL_MARKET = 3600           # technical snapshot — intraday is enough

# Universe name → (Wikipedia page, column header holding the company name).
UNIVERSES: dict[str, tuple[str, str]] = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Security"),
    "tsx60": ("https://en.wikipedia.org/wiki/S%26P/TSX_60", "Company"),
}

DEFAULT_WORKERS = 10
DEFAULT_POOL_MULTIPLIER = 6
MIN_POOL_SIZE = 50

# Weights for the final 0-100 conviction score.  They are intentionally
# dominated by analyst conviction + business quality: a high target price on a
# low-quality, over-levered business should not automatically win.
SCORE_WEIGHTS = {
    "analyst": 0.28,
    "quality": 0.24,
    "valuation": 0.18,
    "growth": 0.18,
    "momentum": 0.12,
}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    """Return *value* as a finite float, else ``None``."""
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    n, d = _num(numerator), _num(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scale(
    value: Optional[float],
    low: float,
    high: float,
    invert: bool = False,
) -> Optional[float]:
    """Linearly map *value* from [low, high] to 0..100, clamped.

    ``invert=True`` means lower is better (e.g. P/E, debt).
    """
    if value is None:
        return None
    if high == low:
        return 50.0
    x = (value - low) / (high - low)
    x = _clamp(x, 0.0, 1.0)
    if invert:
        x = 1.0 - x
    return round(x * 100, 1)


def _weighted(parts: list[tuple[Optional[float], float]]) -> tuple[float, float]:
    """Weighted average of present components.

    Returns ``(score_0_100, coverage_0_1)`` where coverage is the share of the
    intended weight that had usable data.
    """
    present = [(v, w) for v, w in parts if v is not None and w > 0]
    total_intended = sum(w for _, w in parts)
    if not present or total_intended <= 0:
        return 50.0, 0.0
    total_present = sum(w for _, w in present)
    score = sum(v * w for v, w in present) / total_present
    return round(score, 1), round(total_present / total_intended, 2)


def _exp_upside_score(upside: Optional[float]) -> Optional[float]:
    """Diminishing-returns score for analyst upside.

    10% upside ≈ 28, 25% ≈ 57, 50% ≈ 81, 100% ≈ 96.  A 300% target does not
    score three times as well as a 100% target (that is usually a stale or
    low-float name, not a better investment).
    """
    u = _num(upside)
    if u is None:
        return None
    return round((1.0 - math.exp(-max(0.0, u) / 0.30)) * 100, 1)


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
# Analyst consensus + revisions
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
        "target_low": _num((pt or {}).get("low")),
        "target_high": _num((pt or {}).get("high")),
        "target_median": _num((pt or {}).get("median")),
        "strong_buy": int(row.get("strongBuy", 0) or 0),
        "buy": int(row.get("buy", 0) or 0),
        "hold": int(row.get("hold", 0) or 0),
        "sell": int(row.get("sell", 0) or 0),
        "strong_sell": int(row.get("strongSell", 0) or 0),
    }
    cache_set(cache_key, out)
    return out


def _revision_snapshot(ticker: str) -> dict[str, Any]:
    """Recent rating/price-target revisions for *ticker* (cached 24h).

    Uses two cheap yfinance tables:
      * ``recommendations`` — 4 months of buy/hold/sell counts, so we can tell
        whether consensus is improving or deteriorating.
      * ``upgrades_downgrades`` — individual analyst actions, so we can count
        net upgrades and price-target raises over the last 90 days.

    Returns an empty dict when the data is unavailable.
    """
    cache_key = f"screen:revisions:{ticker}"
    cached = cache_get(cache_key, TTL_ANALYST)
    if cached is not None:
        return cached

    out: dict[str, Any] = {}
    try:
        t = yf.Ticker(ticker)
        rec = t.recommendations
    except Exception:
        rec = None

    def _bull_ratio(r: pd.Series) -> Optional[float]:
        sb = _num(r.get("strongBuy")) or 0.0
        b = _num(r.get("buy")) or 0.0
        h = _num(r.get("hold")) or 0.0
        s = _num(r.get("sell")) or 0.0
        ss = _num(r.get("strongSell")) or 0.0
        total = sb + b + h + s + ss
        return (2 * sb + b) / (2 * total) if total > 0 else None

    if rec is not None and not rec.empty:
        try:
            current_ratio = _bull_ratio(rec.iloc[0])
            old_ratio = _bull_ratio(rec.iloc[-1]) if len(rec) > 1 else None
            out["bull_ratio"] = current_ratio
            if current_ratio is not None and old_ratio is not None:
                out["bull_ratio_delta"] = round(current_ratio - old_ratio, 4)
        except Exception:
            pass

    try:
        actions = t.upgrades_downgrades
    except Exception:
        actions = None

    if actions is not None and not actions.empty:
        try:
            idx = pd.to_datetime(actions.index, utc=True).tz_convert(None)
            cutoff = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=90)
            recent = actions.loc[idx >= cutoff]
            out["rating_changes"] = int(len(recent))
            if "Action" in recent.columns:
                act = recent["Action"].astype(str).str.lower()
                out["upgrades"] = int((act == "up").sum())
                out["downgrades"] = int((act == "down").sum())
                out["initiations"] = int((act == "init").sum())
                out["net_upgrades"] = out["upgrades"] - out["downgrades"]
            if "priceTargetAction" in recent.columns:
                pta = recent["priceTargetAction"].astype(str).str.lower()
                out["pt_raises"] = int(pta.str.startswith(("rais", "up")).sum())
                out["pt_cuts"] = int(pta.str.startswith(("cut", "lower", "down")).sum())
        except Exception:
            pass

    cache_set(cache_key, out)
    return out


def _prelim_score(d: dict[str, Any]) -> float:
    """Cheap analyst-only score used to choose the deep-analysis pool.

    This is *not* the final ranking.  It deliberately keeps the expensive
    fundamental/technical work for a smaller candidate list.
    """
    total = max(d["total"], 1)
    consensus = (2.0 * d["strong_buy"] + d["buy"]) / total
    consensus_n = _clamp(consensus / 2.0, 0.0, 1.0)
    upside_n = (_exp_upside_score(d.get("upside", 0.0)) or 0.0) / 100.0
    coverage = min(total, 40) / 40.0
    return round(0.55 * consensus_n + 0.30 * upside_n + 0.15 * coverage, 4)


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


def conviction_label(d: dict[str, Any]) -> tuple[str, str]:
    """Return (verdict, rich color) for the final multi-factor score."""
    score = _num(d.get("score")) or 0.0
    risk = _num(d.get("risk_penalty")) or 0.0
    if score >= 80 and risk < 10:
        return "Strong Buy", "green"
    if score >= 65 and risk < 15:
        return "Buy", "green"
    if risk >= 15 and score < 70:
        return "Speculative", "red"
    if score >= 50:
        return "Accumulate", "yellow"
    if risk >= 12:
        return "Speculative", "red"
    return "Watch", "yellow"


# ---------------------------------------------------------------------------
# Fundamental / technical enrichment
# ---------------------------------------------------------------------------

def _fundamentals(ticker: str) -> dict[str, Any]:
    """Normalize the useful parts of yfinance's large ``info`` dict."""
    d = data_client.get_ticker_info(ticker) or {}
    if not d:
        return {}

    fcf = _num(d.get("freeCashflow"))
    ocf = _num(d.get("operatingCashflow"))
    mcap = _num(d.get("marketCap"))
    profit_margin = _num(d.get("profitMargins"))
    debt_equity = _num(d.get("debtToEquity"))
    # yfinance reports debt/equity as a percentage (e.g. 28.5 for 0.285x).
    if debt_equity is not None and debt_equity > 3:
        debt_equity = debt_equity / 100.0

    # yfinance occasionally returns a single quarter (or otherwise broken)
    # free-cash-flow figure.  A highly profitable company whose FCF is <10% of
    # operating cash flow is almost certainly a data glitch — don't let bad
    # data drive the quality/valuation score.
    if (
        fcf is not None
        and ocf is not None
        and ocf > 0
        and fcf < 0.10 * ocf
        and (profit_margin or 0) > 0.15
    ):
        fcf = None

    trailing_eps = _num(d.get("trailingEps"))
    forward_eps = _num(d.get("forwardEps"))
    eps_growth = None
    if trailing_eps and forward_eps and trailing_eps > 0:
        eps_growth = forward_eps / trailing_eps - 1.0

    return {
        "name": d.get("longName") or d.get("shortName") or ticker,
        "sector": d.get("sector") or "",
        "industry": d.get("industry") or "",
        "market_cap": mcap,
        "pe": _num(d.get("trailingPE")),
        "forward_pe": _num(d.get("forwardPE")),
        "peg": _num(d.get("pegRatio")),
        "pb": _num(d.get("priceToBook")),
        "ps": _num(d.get("priceToSalesTrailing12Months")),
        "ev_ebitda": _num(d.get("enterpriseToEbitda")),
        "roe": _num(d.get("returnOnEquity")),
        "roa": _num(d.get("returnOnAssets")),
        "gross_margin": _num(d.get("grossMargins")),
        "operating_margin": _num(d.get("operatingMargins")),
        "profit_margin": _num(d.get("profitMargins")),
        "ebitda_margin": _num(d.get("ebitdaMargins")),
        "fcf": fcf,
        "ocf": ocf,
        "fcf_yield": _safe_div(fcf, mcap),
        "debt_equity": debt_equity,
        "current_ratio": _num(d.get("currentRatio")),
        "quick_ratio": _num(d.get("quickRatio")),
        "revenue_growth": _num(d.get("revenueGrowth")),
        "earnings_growth": _num(d.get("earningsGrowth")),
        "earnings_quarterly_growth": _num(d.get("earningsQuarterlyGrowth")),
        "eps_growth": eps_growth,
        "dividend_rate": _num(d.get("dividendRate")),
        "payout_ratio": _num(d.get("payoutRatio")),
        "beta": _num(d.get("beta")),
        "week52_high": _num(d.get("fiftyTwoWeekHigh")),
        "week52_low": _num(d.get("fiftyTwoWeekLow")),
        "sma50": _num(d.get("fiftyDayAverage")),
        "sma200": _num(d.get("twoHundredDayAverage")),
        "short_float": _num(d.get("shortPercentOfFloat")),
        "institutional": _num(d.get("heldPercentInstitutions")),
        "recommendation_mean": _num(d.get("recommendationMean")),
        "analyst_count": _num(d.get("numberOfAnalystOpinions")),
        "total_cash": _num(d.get("totalCash")),
        "total_debt": _num(d.get("totalDebt")),
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
    }


def _rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    avg_gain = _num(delta.clip(lower=0).rolling(period).mean().iloc[-1])
    avg_loss = _num((-delta.clip(upper=0)).rolling(period).mean().iloc[-1])
    if avg_gain is None or avg_loss is None:
        return None
    # Monotonic series produce a zero denominator; handle them explicitly.
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _technical_snapshot(ticker: str) -> dict[str, Any]:
    """Compute trend, returns and RSI from one year of price history."""
    df = data_client.get_price_history(ticker, period="1y")
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 30:
        return {}

    last = float(close.iloc[-1])
    sma20 = float(close.tail(20).mean()) if len(close) >= 20 else None
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
    high52 = float(close.tail(252).max())
    low52 = float(close.tail(252).min())

    def _ret(days: int) -> Optional[float]:
        if len(close) <= days:
            return None
        start = float(close.iloc[-days - 1])
        return (last / start - 1.0) if start > 0 else None

    if last > (sma50 or last) and (sma50 is None or sma200 is None or sma50 > sma200):
        trend = "uptrend"
        trend_score = 90.0
    elif last < (sma50 or last) and (sma50 is None or sma200 is None or sma50 < sma200):
        trend = "downtrend"
        trend_score = 20.0
    else:
        trend = "mixed"
        trend_score = 55.0

    rets = close.pct_change().dropna()
    vol = float(rets.tail(63).std() * math.sqrt(252)) if len(rets) >= 20 else None

    return {
        "last": last,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "vs_sma50": (last / sma50 - 1.0) if sma50 else None,
        "vs_sma200": (last / sma200 - 1.0) if sma200 else None,
        "from_high": (last / high52 - 1.0) if high52 > 0 else None,
        "from_low": (last / low52 - 1.0) if low52 > 0 else None,
        "week52_high": high52,
        "week52_low": low52,
        "ret_1m": _ret(21),
        "ret_3m": _ret(63),
        "ret_6m": _ret(126),
        "rsi": _rsi(close),
        "volatility": vol,
        "trend": trend,
        "trend_score": trend_score,
    }


def _enrich(
    rows: list[dict[str, Any]],
    workers: int = 8,
    market: bool = False,
) -> list[dict[str, Any]]:
    """Attach fundamentals (and optionally technicals/revisions) to each row.

    ``market=False`` keeps the old fast behaviour used by tests and callers that
    only need company name / market cap / P/E / sector.
    """
    if not rows:
        return rows

    def enrich_one(row: dict[str, Any]) -> dict[str, Any]:
        sym = row["symbol"]
        fund = _fundamentals(sym)
        out = dict(row)
        out["fundamentals"] = fund
        out["company"] = fund.get("name") or row.get("company", "")
        out["name"] = out["company"]
        out["market_cap"] = fund.get("market_cap")
        out["pe"] = fund.get("pe")
        out["sector"] = fund.get("sector", "")
        if market:
            out["technical"] = _technical_snapshot(sym)
            out["revisions"] = _revision_snapshot(sym)
        return out

    with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as pool:
        enriched = list(pool.map(enrich_one, rows))
    return enriched


# ---------------------------------------------------------------------------
# Scoring pillars (each 0..100)
# ---------------------------------------------------------------------------

def _score_analyst(
    row: dict[str, Any],
    fund: dict[str, Any],
    rev: dict[str, Any],
) -> tuple[float, float, str]:
    """Analyst conviction: consensus, target, coverage, confidence, revisions."""
    total = max(int(row.get("total") or 0), 1)
    sb = int(row.get("strong_buy") or 0)
    b = int(row.get("buy") or 0)
    s = int(row.get("sell") or 0)
    ss = int(row.get("strong_sell") or 0)

    # -1..1 weighted consensus → 0..100
    raw = (2 * sb + b - s - 2 * ss) / (2 * total)
    consensus = round((raw + 1.0) / 2.0 * 100, 1)

    mean = fund.get("recommendation_mean")
    mean_score = _scale(mean, 1.0, 5.0, invert=True)  # 1=strong buy → 100

    upside_score = _exp_upside_score(row.get("upside"))
    coverage = _scale(math.log10(total), math.log10(3), math.log10(30))

    low, high, target = row.get("target_low"), row.get("target_high"), row.get("target")
    dispersion = (high - low) / target if (low and high and target and target > 0) else None
    confidence = _scale(dispersion, 0.15, 0.75, invert=True)

    revision_parts: list[tuple[Optional[float], float]] = []
    if rev:
        revision_parts = [
            (_scale(rev.get("bull_ratio_delta"), -0.20, 0.20), 0.45),
            (_scale(rev.get("net_upgrades"), -3, 3), 0.30),
            (_scale((rev.get("pt_raises") or 0) - (rev.get("pt_cuts") or 0), -5, 5), 0.25),
        ]
    rev_score, rev_cov = _weighted(revision_parts) if revision_parts else (None, 0.0)

    score, coverage_ratio = _weighted([
        (consensus, 0.30),
        (mean_score, 0.15),
        (upside_score, 0.25),
        (coverage, 0.10),
        (confidence, 0.10),
        (rev_score if rev_cov > 0 else None, 0.10),
    ])

    label, _ = rating_label({
        "strong_buy": sb, "buy": b, "hold": int(row.get("hold") or 0),
        "sell": s, "strong_sell": ss, "total": total,
    })
    bits = [f"{label} ({total} analysts)"]
    if upside_score is not None:
        bits.append(f"{row.get('upside', 0) * 100:.0f}% mean upside")
    if rev.get("net_upgrades"):
        bits.append(f"{rev['net_upgrades']:+d} net upgrades")
    if rev.get("pt_raises") or rev.get("pt_cuts"):
        bits.append(f"{rev.get('pt_raises', 0)} PT raises / {rev.get('pt_cuts', 0)} cuts")
    if confidence is not None and confidence < 40:
        bits.append("wide target dispersion")
    return score, coverage_ratio, "; ".join(bits)


def _is_financial(fund: dict[str, Any]) -> bool:
    sector = (fund.get("sector") or "").lower()
    return any(k in sector for k in ("financial", "bank", "insurance"))


def _score_quality(fund: dict[str, Any]) -> tuple[float, float, str]:
    """Business quality: profitability, returns, cash generation, balance sheet."""
    if not fund:
        return 50.0, 0.0, "Fundamentals unavailable"

    fin = _is_financial(fund)
    parts = [
        (_scale(fund.get("roe"), 0.05, 0.25), 0.20),
        (_scale(fund.get("roa"), 0.02, 0.12), 0.10),
        (_scale(fund.get("gross_margin"), 0.25, 0.65), 0.12),
        (_scale(fund.get("operating_margin"), 0.08, 0.30), 0.15),
        (_scale(fund.get("profit_margin"), 0.05, 0.25), 0.12),
        (_scale(fund.get("fcf_yield"), 0.0, 0.08), 0.16),
        (None if fin else _scale(fund.get("debt_equity"), 0.2, 1.5, invert=True), 0.10),
        (None if fin else _scale(fund.get("current_ratio"), 1.0, 2.0), 0.05),
    ]
    score, cov = _weighted(parts)

    bits = []
    if fund.get("roe") is not None:
        bits.append(f"ROE {fund['roe'] * 100:.0f}%")
    if fund.get("operating_margin") is not None:
        bits.append(f"op margin {fund['operating_margin'] * 100:.0f}%")
    if fund.get("fcf_yield") is not None:
        bits.append(f"FCF yield {fund['fcf_yield'] * 100:.1f}%")
    if not fin and fund.get("debt_equity") is not None:
        bits.append(f"debt/equity {fund['debt_equity']:.2f}x")
    return score, cov, "; ".join(bits) if bits else "Limited quality data"


def _score_valuation(fund: dict[str, Any]) -> tuple[float, float, str]:
    """Valuation: multiples relative to growth and cash flow."""
    if not fund:
        return 50.0, 0.0, "Valuation unavailable"

    fin = _is_financial(fund)
    peg = fund.get("peg")
    if peg is not None and peg <= 0:
        peg = None
    fwd_pe = fund.get("forward_pe")
    if fwd_pe is not None and fwd_pe <= 0:
        fwd_pe = None
    ev = fund.get("ev_ebitda")
    if ev is not None and ev <= 0:
        ev = None
    ps = fund.get("ps")
    if ps is not None and ps <= 0:
        ps = None

    parts = [
        (_scale(peg, 0.7, 3.0, invert=True), 0.30),
        (_scale(fwd_pe, 10.0, 35.0, invert=True), 0.15),
        (_scale(ev, 8.0, 25.0, invert=True), 0.25),
        (None if fin else _scale(ps, 1.0, 8.0, invert=True), 0.10),
        (_scale(fund.get("pb"), 1.0, 8.0, invert=True), 0.05),
        (_scale(fund.get("fcf_yield"), 0.0, 0.08), 0.15),
    ]
    score, cov = _weighted(parts)

    bits = []
    if peg is not None:
        bits.append(f"PEG {peg:.1f}")
    if fwd_pe is not None:
        bits.append(f"fwd P/E {fwd_pe:.1f}")
    if ev is not None:
        bits.append(f"EV/EBITDA {ev:.1f}")
    if fund.get("fcf_yield") is not None:
        bits.append(f"FCF yield {fund['fcf_yield'] * 100:.1f}%")
    return score, cov, "; ".join(bits) if bits else "Limited valuation data"


def _score_growth(fund: dict[str, Any]) -> tuple[float, float, str]:
    """Growth: revenue, earnings and forward-EPS trajectory."""
    if not fund:
        return 50.0, 0.0, "Growth unavailable"

    parts = [
        (_scale(fund.get("revenue_growth"), -0.05, 0.25), 0.35),
        (_scale(fund.get("earnings_growth"), -0.10, 0.30), 0.30),
        (_scale(fund.get("earnings_quarterly_growth"), -0.10, 0.30), 0.15),
        (_scale(fund.get("eps_growth"), -0.05, 0.25), 0.20),
    ]
    score, cov = _weighted(parts)

    bits = []
    if fund.get("revenue_growth") is not None:
        bits.append(f"revenue {fund['revenue_growth'] * 100:+.0f}%")
    if fund.get("earnings_growth") is not None:
        bits.append(f"earnings {fund['earnings_growth'] * 100:+.0f}%")
    if fund.get("eps_growth") is not None:
        bits.append(f"fwd EPS {fund['eps_growth'] * 100:+.0f}%")
    return score, cov, "; ".join(bits) if bits else "Limited growth data"


def _rsi_positioning(rsi: Optional[float]) -> Optional[float]:
    """Reward constructive RSI zones, not just 'low is good' or 'high is good'."""
    if rsi is None:
        return None
    if rsi <= 25:
        return 35.0
    if rsi < 40:
        return 55.0
    if rsi <= 65:
        return 90.0
    if rsi <= 75:
        return 60.0
    return 25.0


def _score_momentum(tech: dict[str, Any]) -> tuple[float, float, str]:
    """Momentum: trend, 3/6-month returns, RSI and distance from the high."""
    if not tech:
        return 50.0, 0.0, "Price history unavailable"

    parts = [
        (tech.get("trend_score"), 0.30),
        (_scale(tech.get("ret_3m"), -0.15, 0.20), 0.20),
        (_scale(tech.get("ret_6m"), -0.20, 0.30), 0.15),
        (_rsi_positioning(tech.get("rsi")), 0.20),
        (_scale(tech.get("from_high"), -0.40, -0.02), 0.15),
    ]
    score, cov = _weighted(parts)

    bits = []
    if tech.get("trend"):
        bits.append(str(tech["trend"]))
    if tech.get("rsi") is not None:
        bits.append(f"RSI {tech['rsi']:.0f}")
    if tech.get("ret_3m") is not None:
        bits.append(f"3m {tech['ret_3m'] * 100:+.0f}%")
    if tech.get("from_high") is not None:
        bits.append(f"{abs(tech['from_high']) * 100:.0f}% below 52w high")
    return score, cov, "; ".join(bits) if bits else "Limited momentum data"


def _risk_penalty(
    row: dict[str, Any],
    fund: dict[str, Any],
    tech: dict[str, Any],
) -> tuple[float, list[str]]:
    """Material risks subtract from the composite score (capped at 25)."""
    penalty = 0.0
    flags: list[str] = []

    beta = fund.get("beta")
    if beta is not None and beta > 2.5:
        penalty += 8
        flags.append(f"very high beta ({beta:.1f})")
    elif beta is not None and beta > 1.8:
        penalty += 4
        flags.append(f"high beta ({beta:.1f})")

    if fund.get("fcf") is not None and fund["fcf"] < 0:
        penalty += 10
        flags.append("negative free cash flow")

    de = fund.get("debt_equity")
    if de is not None and not _is_financial(fund):
        if de > 3.0:
            penalty += 8
            flags.append(f"over-levered ({de:.1f}x debt/equity)")
        elif de > 2.0:
            penalty += 4
            flags.append(f"high leverage ({de:.1f}x debt/equity)")

    margin = fund.get("profit_margin")
    if margin is not None and margin < 0:
        penalty += 8
        flags.append("unprofitable")

    rg = fund.get("revenue_growth")
    if rg is not None and rg < -0.10:
        penalty += 5
        flags.append(f"revenue shrinking ({rg * 100:.0f}%)")

    short = fund.get("short_float")
    if short is not None and short > 0.25:
        penalty += 6
        flags.append(f"very high short interest ({short * 100:.0f}%)")
    elif short is not None and short > 0.15:
        penalty += 3
        flags.append(f"high short interest ({short * 100:.0f}%)")

    low, high, target = row.get("target_low"), row.get("target_high"), row.get("target")
    if low and high and target and target > 0 and (high - low) / target > 0.90:
        penalty += 4
        flags.append("analyst targets are widely dispersed")

    upside = row.get("upside")
    if upside is not None and upside > 1.50:
        penalty += 4
        flags.append("target upside may be stale (>150%)")

    mcap = fund.get("market_cap")
    if mcap is not None and mcap < 2e9:
        penalty += 3
        flags.append("small-cap liquidity risk")

    if (row.get("total") or 0) < 5:
        penalty += 2
        flags.append("thin analyst coverage")

    vol = tech.get("volatility")
    if vol is not None and vol > 0.60:
        penalty += 3
        flags.append(f"high volatility ({vol * 100:.0f}%)")

    return min(penalty, 25.0), flags


def _tags(
    row: dict[str, Any],
    fund: dict[str, Any],
    tech: dict[str, Any],
    rev: dict[str, Any],
) -> list[str]:
    """Human-readable labels describing the investment case."""
    analyst = _num(row.get("analyst_score")) or 0
    quality = _num(row.get("quality_score")) or 0
    valuation = _num(row.get("valuation_score")) or 0
    growth = _num(row.get("growth_score")) or 0
    momentum = _num(row.get("momentum_score")) or 0
    underrated = _num(row.get("underrated_score")) or 0
    rsi = tech.get("rsi")

    tags: list[str] = []
    if underrated >= 65 and momentum < 60:
        tags.append("Underrated")
    if quality >= 70:
        tags.append("Quality")
    if valuation >= 65:
        tags.append("Value")
    if growth >= 70:
        tags.append("Growth")
    if momentum >= 70:
        tags.append("Momentum")
    if analyst >= 75:
        tags.append("Analyst favorite")
    if (rev.get("bull_ratio_delta") or 0) > 0.03 or (rev.get("net_upgrades") or 0) >= 2:
        tags.append("Revisions up")
    if rsi is not None and rsi < 38 and quality >= 60:
        tags.append("Oversold quality")
    div_rate = fund.get("dividend_rate")
    price = row.get("current")
    if div_rate and price and price > 0 and div_rate / price >= 0.02:
        tags.append("Dividend")
    if not tags:
        tags.append("Balanced")
    return tags[:5]


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compute every pillar, penalties and the final 0-100 score for one row."""
    fund = row.get("fundamentals") or {}
    tech = row.get("technical") or {}
    rev = row.get("revisions") or {}

    analyst, a_cov, a_reason = _score_analyst(row, fund, rev)
    quality, q_cov, q_reason = _score_quality(fund)
    valuation, v_cov, v_reason = _score_valuation(fund)
    growth, g_cov, g_reason = _score_growth(fund)
    momentum, m_cov, m_reason = _score_momentum(tech)

    components = {
        "analyst": analyst,
        "quality": quality,
        "valuation": valuation,
        "growth": growth,
        "momentum": momentum,
    }
    coverages = {
        "analyst": a_cov,
        "quality": q_cov,
        "valuation": v_cov,
        "growth": g_cov,
        "momentum": m_cov,
    }
    total_weight = sum(SCORE_WEIGHTS.values())
    score = sum(components[k] * SCORE_WEIGHTS[k] for k in components) / total_weight
    data_coverage = sum(coverages[k] * SCORE_WEIGHTS[k] for k in components) / total_weight

    risk_penalty, risk_flags = _risk_penalty(row, fund, tech)
    # Low data coverage is not a risk per se, but it lowers confidence in the rank.
    if data_coverage < 0.50:
        risk_flags.append("limited fundamental/technical data")
    score -= (1.0 - data_coverage) * 5.0
    score -= risk_penalty
    score = round(_clamp(score, 0.0, 100.0), 1)

    underrated = round(
        _clamp(
            0.35 * quality + 0.35 * valuation + 0.15 * growth + 0.15 * (100.0 - momentum),
            0.0,
            100.0,
        ),
        1,
    )

    row.update({
        "analyst_score": analyst,
        "quality_score": quality,
        "valuation_score": valuation,
        "growth_score": growth,
        "momentum_score": momentum,
        "underrated_score": underrated,
        "score": score,
        "data_coverage": round(data_coverage, 2),
        "risk_penalty": risk_penalty,
        "risk_flags": risk_flags,
        "score_reasons": [a_reason, q_reason, v_reason, g_reason, m_reason],
    })
    row["tags"] = _tags(row, fund, tech, rev)
    row["verdict"], row["verdict_color"] = conviction_label(row)
    return row


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
    pool_size: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Screen a universe for high-conviction buys and return the top *limit*.

    Stage 1 ranks every name on cheap analyst data; only the candidate pool is
    then enriched and scored on quality, valuation, growth and momentum.  This
    keeps the run practical for S&P 500 / TSX 60 universes while still doing a
    real investigation of the finalists.

    Returns enriched rows sorted by final score descending.  Each row contains
    the original analyst fields plus component scores, tags, risk flags and
    plain-English reasons.
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

    # --- Stage 1: cheap analyst pass over the whole universe -----------------
    with ThreadPoolExecutor(max_workers=min(workers, len(symbols))) as pool:
        results = list(pool.map(_analyst, symbols))

    candidates: list[dict[str, Any]] = []
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
        data["prelim_score"] = _prelim_score(data)
        candidates.append(data)

    if not candidates:
        return []

    candidates.sort(key=lambda r: r["prelim_score"], reverse=True)
    if pool_size is None:
        pool_size = max(limit * DEFAULT_POOL_MULTIPLIER, MIN_POOL_SIZE)
    pool_size = max(1, min(pool_size, len(candidates)))
    finalists = candidates[:pool_size]

    # --- Stage 2: investigate finalists on fundamentals / technicals ---------
    finalists = _enrich(finalists, workers=workers, market=True)
    finalists = [_score_row(row) for row in finalists]
    finalists.sort(key=lambda r: r["score"], reverse=True)
    return finalists[:limit]


# ---------------------------------------------------------------------------
# Deep-dive narrative
# ---------------------------------------------------------------------------

def _fmt_pct(value: Any, digits: int = 0) -> str:
    v = _num(value)
    return "n/a" if v is None else f"{v * 100:+.{digits}f}%"


def _fmt_money(value: Any) -> str:
    v = _num(value)
    if v is None:
        return "n/a"
    if abs(v) >= 1e12:
        return f"${v / 1e12:,.1f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:,.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.1f}M"
    return f"${v:,.0f}"


def _target_range(pick: dict[str, Any]) -> str:
    low, high = _num(pick.get("target_low")), _num(pick.get("target_high"))
    if low is None or high is None:
        return "n/a"
    return f"${low:,.0f}–${high:,.0f}"


def _pick_context(p: dict[str, Any]) -> str:
    """Compact, metric-rich context block for one pick (used by the AI)."""
    fund = p.get("fundamentals") or {}
    tech = p.get("technical") or {}
    rev = p.get("revisions") or {}
    verdict, _ = conviction_label(p)
    tags = ", ".join(p.get("tags") or [])

    lines = [
        f"- **{p['symbol']}** — {p.get('company') or p.get('name', '')} | "
        f"{p.get('sector') or 'n/a'} | Price ${p['current']:,.2f}, "
        f"target ${p['target']:,.2f} ({p.get('upside', 0) * 100:+.0f}%) | "
        f"**score {p.get('score', 0):.0f}/100 — {verdict}**",
        f"  - Pillars: analyst {p.get('analyst_score', 0):.0f}/100, "
        f"quality {p.get('quality_score', 0):.0f}, "
        f"valuation {p.get('valuation_score', 0):.0f}, "
        f"growth {p.get('growth_score', 0):.0f}, "
        f"momentum {p.get('momentum_score', 0):.0f}, "
        f"underrated {p.get('underrated_score', 0):.0f}",
        f"  - Analysts: {p.get('strong_buy', 0)} strong buy / {p.get('buy', 0)} buy / "
        f"{p.get('hold', 0)} hold / {p.get('sell', 0)} sell / "
        f"{p.get('strong_sell', 0)} strong sell ({p.get('total', 0)} total); "
        f"target range {_target_range(p)}",
    ]
    if rev:
        lines.append(
            f"  - Revisions (90d): {rev.get('upgrades', 0)} upgrades, "
            f"{rev.get('downgrades', 0)} downgrades, "
            f"{rev.get('pt_raises', 0)} price-target raises, "
            f"{rev.get('pt_cuts', 0)} cuts; consensus change "
            f"{(rev.get('bull_ratio_delta') or 0) * 100:+.1f} pts"
        )
    if fund:
        lines.append(
            f"  - Quality: ROE {_fmt_pct(fund.get('roe'))}, "
            f"op margin {_fmt_pct(fund.get('operating_margin'))}, "
            f"net margin {_fmt_pct(fund.get('profit_margin'))}, "
            f"FCF yield {_fmt_pct(fund.get('fcf_yield'))}, "
            f"debt/equity {fund.get('debt_equity') if fund.get('debt_equity') is not None else 'n/a'}"
        )
        lines.append(
            f"  - Valuation: P/E {fund.get('pe') if fund.get('pe') is not None else 'n/a'}, "
            f"fwd P/E {fund.get('forward_pe') if fund.get('forward_pe') is not None else 'n/a'}, "
            f"PEG {fund.get('peg') if fund.get('peg') is not None else 'n/a'}, "
            f"EV/EBITDA {fund.get('ev_ebitda') if fund.get('ev_ebitda') is not None else 'n/a'}, "
            f"market cap {_fmt_money(fund.get('market_cap'))}"
        )
        lines.append(
            f"  - Growth: revenue {_fmt_pct(fund.get('revenue_growth'))}, "
            f"earnings {_fmt_pct(fund.get('earnings_growth'))}, "
            f"fwd EPS {_fmt_pct(fund.get('eps_growth'))}"
        )
    if tech:
        lines.append(
            f"  - Technical: {tech.get('trend', 'n/a')}, "
            f"RSI {tech.get('rsi') if tech.get('rsi') is not None else 'n/a'}, "
            f"3m {_fmt_pct(tech.get('ret_3m'))}, 6m {_fmt_pct(tech.get('ret_6m'))}, "
            f"{_fmt_pct(tech.get('from_high'))} from 52w high"
        )
    lines.append(f"  - Tags: {tags or 'none'}")
    if p.get("risk_flags"):
        lines.append(f"  - Risk flags: {', '.join(p['risk_flags'])}")
    if p.get("score_reasons"):
        lines.append(f"  - Score drivers: {' | '.join(p['score_reasons'])}")
    return "\n".join(lines)


def _fallback_narrative(picks: list[dict[str, Any]]) -> str:
    """Deterministic, data-driven write-up when the AI is unavailable."""
    lines = ["_AI narrative unavailable right now — here is the quantitative case:_"]
    for p in picks:
        verdict, _ = conviction_label(p)
        lines.append(f"\n### {p['symbol']} — {p.get('company') or ''} ({verdict})")
        lines.append(f"- **Score:** {p.get('score', 0):.0f}/100 | tags: {', '.join(p.get('tags') or [])}")
        for reason in p.get("score_reasons") or []:
            lines.append(f"- {reason}")
        if p.get("risk_flags"):
            lines.append(f"- **Risk flags:** {', '.join(p['risk_flags'])}")
    return "\n".join(lines)


def deep_dive(picks: list[dict[str, Any]], top: int = 3) -> str:
    """Ask the AI to challenge the quant screen and build an investment case."""
    if not picks:
        return ""
    picks = picks[:top]

    news_batch = data_client.get_ticker_news_batch(
        [p["symbol"] for p in picks], limit=4
    )

    picks_block = "\n".join(_pick_context(p) for p in picks)

    news_block = ""
    for p in picks:
        articles = news_batch.get(p["symbol"], [])
        if articles:
            news_block += f"\n**{p['symbol']} — recent headlines:**\n"
            for a in articles[:3]:
                news_block += f"- {a.get('title', '')}\n"

    prompt = f"""
You are a skeptical senior equity analyst. A quantitative multi-factor screen
has already ranked the stocks below. Your job is NOT to repeat the analyst
target or the screen's score — it is to challenge the model and explain the
actual investment case.

For each stock, write a concise note covering:
1. **Business quality & moat** — what makes this a good company, and is it durable?
2. **Why the market may be underrating it** — specific mispricing, neglect,
   cyclical fear, temporary setback, or sentiment disconnect. Cite the metrics.
3. **Catalysts (6–12 months)** — concrete events or trends that could close the gap.
4. **Valuation check** — is the upside justified by growth, margins and returns,
   or is the target simply stale/too optimistic?
5. **Bear case / what would make you pass** — 2–3 material risks or disconfirming evidence.
6. **Verdict** — Strong Buy / Buy / Accumulate / Avoid, with a 12-month thesis
   and the one metric to monitor.

Be specific and use the supplied numbers. Flag when the score looks inflated by
a single pillar or when important data is missing. Finish with a short
**"Ranking challenge"** section: if you had to drop one name, which one and why.
5–7 sentences per stock, bullet points, Markdown.

**QUANTITATIVE SCREEN OUTPUT**
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
    return response.strip() if response else _fallback_narrative(picks)
