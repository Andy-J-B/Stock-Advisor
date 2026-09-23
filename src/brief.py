"""
Nightly brief: composite conviction scoring for every holding/watchlist ticker.

Orchestrates existing modules (sentiment, indicators, ML, screener, anomaly)
into a single [-100, +100] conviction score per ticker per day.

No new external API calls are introduced beyond what the individual modules
already make.
"""

from __future__ import annotations

import logging
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime


from src import data_client, features, indicators, ml_model, anomaly
from src.sentiment import get_sentiment_engine
from src.config import load_brief_weights
from src.database import (
    get_watchlist, init_db, Account,
)

log = logging.getLogger(__name__)

# Serializes the CPU-heavy native inference steps (FinBERT/transformers, LightGBM,
# scikit-learn) across the parallel scorer threads. Network fetches still overlap;
# only one native call runs at a time. Prevents thread oversubscription crashes
# (OpenMP + torch + sklearn interleaving under ThreadPoolExecutor segfaults).
_native_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TickerScore:
    ticker: str
    composite: float
    sentiment: float = 0.0
    technical: float = 0.0
    ml_pred: float = 0.0
    analyst: float = 0.0
    anomaly_flag: bool = False
    reasoning: str = ""
    # Enrichment fields (filled after scoring)
    prev_composite: float | None = None
    score_delta: float | None = None
    anomaly_detail: str = ""
    analyst_breakdown: str = ""
    top_headline: str = ""
    signal_agreement: str = ""
    price: float = 0.0
    day_change_pct: float = 0.0
    portfolio_weight: float = 0.0
    portfolio_value: float = 0.0
    new_entrant: bool = False
    anomaly_new: bool = False
    rank_change: int | None = None
    recommendation: str = ""
    overexposed: bool = False


@dataclass
class BriefRun:
    run_date: date
    generated_at: datetime
    weights_used: dict[str, float]
    scores: list[TickerScore] = field(default_factory=list)
    tickers: list[str] | None = None
    market: dict | None = None
    # Enrichment fields
    prev_run_date: date | None = None
    biggest_movers: list[TickerScore] = field(default_factory=list)
    new_entrants: list[str] = field(default_factory=list)
    exits: list[str] = field(default_factory=list)
    anomaly_flips: list[dict] = field(default_factory=list)
    portfolio_total_value: float = 0.0


# ---------------------------------------------------------------------------
# Component scorers – each normalizes to [-100, +100]
# ---------------------------------------------------------------------------


def score_sentiment(ticker: str) -> tuple[float, str, str]:
    """Score sentiment from recent news headlines.

    Returns (score, reasoning, top_headline).
    """
    resolved = data_client.get_ticker_news(ticker, limit=10)
    if not resolved:
        # Try US ticker via ticker_map
        from src.ticker_map import resolve_us_ticker
        us = resolve_us_ticker(ticker)
        if us != ticker:
            resolved = data_client.get_ticker_news(us, limit=10)
    if not resolved:
        return 0.0, "No news found", ""

    titles = [a.get("title", "") for a in resolved if a.get("title")]
    if not titles:
        return 0.0, "No valid headlines", ""

    headline = titles[0].strip()

    engine = get_sentiment_engine()
    with _native_lock:
        scores = engine.score_batch(titles)
    compounds = [s["compound"] for s in scores]
    avg = sum(compounds) / len(compounds) if compounds else 0.0

    # Scale compound [-1, 1] to [-100, 100]
    normalized = round(avg * 100, 1)
    pos = sum(1 for s in scores if s["label"] == "positive")
    neg = sum(1 for s in scores if s["label"] == "negative")
    reason = f"{len(titles)} headlines, {pos} positive, {neg} negative"
    return normalized, reason, headline


def score_technical(ticker: str) -> tuple[float, str]:
    """Score technical indicators. Returns (score, reasoning)."""
    ohlcv = data_client.get_price_history(ticker, period="6mo")
    if ohlcv.empty or ohlcv.shape[0] < 30:
        return 0.0, "Insufficient price history"

    hist = indicators.compute_indicators(ohlcv)
    sigs = indicators.interpret_signals(hist)
    if not sigs:
        return 0.0, "No signals computed"

    score = 0.0
    reasons = []

    for sig in sigs:
        name = sig["name"]
        signal = sig.get("signal", "")

        if name == "RSI (14)":
            try:
                rsi_val = float(sig["value"])
                if rsi_val <= 30:
                    score += 30  # oversold = bullish
                    reasons.append(f"RSI oversold ({rsi_val:.0f})")
                elif rsi_val >= 70:
                    score -= 30  # overbought = bearish
                    reasons.append(f"RSI overbought ({rsi_val:.0f})")
                else:
                    # Neutral zone: slight positive if <50
                    score += (50 - rsi_val) * 0.3
            except ValueError:
                pass

        if name == "EMA Cross":
            if signal == "Uptrend":
                score += 25
                reasons.append("EMA uptrend")
            elif signal == "Downtrend":
                score -= 25
                reasons.append("EMA downtrend")

        if name == "MACD" and signal:
            if "Bullish" in signal:
                score += 25
                reasons.append(signal)
            elif "Bearish" in signal:
                score -= 25
                reasons.append(signal)

    # Clamp to [-100, 100]
    score = max(-100, min(100, score))
    return round(score, 1), "; ".join(reasons) if reasons else "Neutral"


def score_ml(ticker: str) -> tuple[float, str]:
    """Score ML prediction probability. Returns (score, reasoning)."""
    horizon = 5
    payload = ml_model.load_model(ticker, horizon)
    if payload is None or ml_model.is_stale(ticker, horizon):
        # Train a fresh model
        ohlcv = data_client.get_price_history(ticker, period="2y")
        if ohlcv.empty or ohlcv.shape[0] < 60:
            return 0.0, "Insufficient data for ML"

        feat = features.build_features(ohlcv)
        target = features.build_target(ohlcv["Close"], horizon=horizon)
        feat_clean = feat.dropna(axis=1, how="all")

        mask = target.notna() & feat_clean.notna().all(axis=1)
        X_train = feat_clean.loc[mask]
        y_train = target.loc[mask]

        if len(X_train) < 30:
            return 0.0, "Not enough labeled data"

        try:
            with _native_lock:
                result = ml_model.train(X_train, y_train)
                ml_model.save_model(result["model"], ticker, horizon, metadata={
                    "cv_accuracy": result["cv_accuracy"],
                })
            model = result["model"]
        except Exception as exc:
            log.warning("ML training failed for %s: %s", ticker, exc)
            return 0.0, f"Training failed: {exc}"
    else:
        model = payload["model"]

    ohlcv = data_client.get_price_history(ticker, period="2y")
    if ohlcv.empty:
        return 0.0, "No price data for prediction"

    feat = features.build_features(ohlcv)
    feat_clean = feat.dropna(axis=1, how="all")
    latest = feat_clean.iloc[[-1]]
    if latest.empty:
        return 0.0, "No valid feature row"

    try:
        with _native_lock:
            pred = ml_model.predict(model, latest.iloc[0])
    except Exception:
        return 0.0, "Prediction failed"

    prob_up = pred["probability_up"]
    # Map P(up) [0, 1] to [-100, +100]
    normalized = round((prob_up - 0.5) * 200, 1)
    reason = f"P(up)={prob_up:.1%}, label={pred['label']}"
    return normalized, reason


def score_analyst(ticker: str) -> tuple[float, str, str]:
    """Score analyst consensus.

    Returns (score, reasoning, breakdown_summary e.g. "15 bull, 0 hold, 0 bear").
    """
    from src.screener import _analyst, rating_label

    data = _analyst(ticker)
    if data is None:
        return 0.0, "No analyst coverage", ""

    data["total"] = (
        data["strong_buy"] + data["buy"] + data["hold"]
        + data["sell"] + data["strong_sell"]
    )
    total = max(data["total"], 1)
    bull = data["strong_buy"] + data["buy"]
    bear = data["sell"] + data["strong_sell"]
    hold = data["hold"]
    breakdown = f"{bull} bull, {hold} hold, {bear} bear, {total} total"

    # Score: strong_buy=+2, buy=+1, hold=0, sell=-1, strong_sell=-2
    raw = (2 * data["strong_buy"] + data["buy"] - data["sell"] - 2 * data["strong_sell"]) / total
    # Normalize raw [-2, 2] to [-100, 100]
    normalized = round(raw * 50, 1)

    label, _ = rating_label(data)
    reason = f"{label} ({breakdown})"
    return normalized, reason, breakdown


def detect_anomaly(ticker: str) -> tuple[bool, str]:
    """Return (is_flagged, detail_text) for anomalous activity.

    The detail text comes from ``anomaly.summarize_anomalies`` so users can
    see *why* a ticker was flagged, not just that it was.
    """
    ohlcv = data_client.get_price_history(ticker, period="6mo")
    if ohlcv.empty or ohlcv.shape[0] < 30:
        return False, ""

    feat = features.build_features(ohlcv)
    with _native_lock:
        flagged = anomaly.detect_anomalies(feat)
    if flagged.empty:
        return False, ""
    return True, (anomaly.summarize_anomalies(flagged, ticker) or "")


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def compute_composite(
    ticker: str,
    weights: dict[str, float],
) -> TickerScore:
    """Compute the conviction score for a single ticker."""
    log.info("Scoring %s ...", ticker)

    sentiment_score, sentiment_reason, headline = score_sentiment(ticker)
    technical_score, technical_reason = score_technical(ticker)
    ml_score, ml_reason = score_ml(ticker)
    analyst_score, analyst_reason, analyst_breakdown = score_analyst(ticker)
    anomaly_flag, anomaly_detail = detect_anomaly(ticker)

    composite = (
        weights.get("sentiment", 0.25) * sentiment_score
        + weights.get("technical", 0.20) * technical_score
        + weights.get("ml_pred", 0.30) * ml_score
        + weights.get("analyst", 0.25) * analyst_score
    )

    # Anomaly penalty
    if anomaly_flag:
        composite -= 25

    composite = round(max(-100, min(100, composite)), 1)

    reasoning_parts = [
        f"Sentiment: {sentiment_reason}",
        f"Technical: {technical_reason}",
        f"ML: {ml_reason}",
        f"Analyst: {analyst_reason}",
    ]
    if anomaly_flag:
        reasoning_parts.append("ANOMALY PENALTY: -25")

    return TickerScore(
        ticker=ticker,
        composite=composite,
        sentiment=sentiment_score,
        technical=technical_score,
        ml_pred=ml_score,
        analyst=analyst_score,
        anomaly_flag=anomaly_flag,
        reasoning=" | ".join(reasoning_parts),
        anomaly_detail=anomaly_detail,
        analyst_breakdown=analyst_breakdown,
        top_headline=headline,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _get_all_tickers() -> list[str]:
    """Return holdings ∪ watchlist, deduplicated."""
    init_db()
    tickers: set[str] = set()

    for acc in Account.select():
        for h in acc.holdings:
            tickers.add(h.ticker)

    for t in get_watchlist():
        tickers.add(t)

    return sorted(tickers)


def run_brief(
    tickers: list[str] | None = None,
    workers: int = 5,
    fetch_market: bool = False,
) -> BriefRun:
    """Score every ticker and return a BriefRun.

    Parameters
    ----------
    tickers : override ticker list (default = holdings ∪ watchlist).
    workers : thread pool size for parallel scoring.
    fetch_market : also snapshot index moves + top market news into run.market.
    """
    if tickers is None:
        tickers = _get_all_tickers()
        if not tickers:
            remote = fetch_brief_universe()
            if remote:
                log.info(
                    "Local universe empty; scoring %d tickers synced to Supabase.",
                    len(remote),
                )
                tickers = remote
    if not tickers:
        log.warning("No tickers to score.")
        return BriefRun(
            run_date=date.today(),
            generated_at=datetime.now(),
            weights_used={},
        )

    weights = load_brief_weights()
    log.info("Scoring %d tickers with weights: %s", len(tickers), weights)

    scores: list[TickerScore] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(tickers))) as pool:
        futures = {
            pool.submit(compute_composite, t, weights): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                scores.append(future.result())
            except Exception as exc:
                log.error("Failed to score %s: %s", ticker, exc)
                scores.append(TickerScore(
                    ticker=ticker,
                    composite=0.0,
                    reasoning=f"Error: {exc}",
                ))

    scores.sort(key=lambda s: s.composite, reverse=True)

    market = None
    if fetch_market:
        market = fetch_market_overview()

    run = BriefRun(
        run_date=date.today(),
        generated_at=datetime.now(),
        weights_used=weights,
        scores=scores,
        tickers=tickers,
        market=market,
    )
    _enrich_run(run)
    return run


# ---------------------------------------------------------------------------
# Brief enrichment: history deltas, portfolio context, per-ticker details
# ---------------------------------------------------------------------------


def _signal_agreement(s: TickerScore) -> str:
    """Label how much the four component scores agree with each other.

    Low dispersion across components = high confidence; wide spread = the
    signals are pulling in different directions.
    """
    parts = [s.sentiment, s.technical, s.ml_pred, s.analyst]
    active = [p for p in parts if p != 0.0]
    if len(active) < 2:
        return "Low coverage" if not active else "Agreeing"
    spread = statistics.pstdev(active)
    if spread < 15:
        return "High agreement"
    if spread < 35:
        return "Moderate"
    return "Conflicted"


def _get_holdings_map() -> dict[str, dict]:
    """Return {ticker: {"shares": float, "account": str}} for current holdings."""
    holdings: dict[str, dict] = {}
    for acc in Account.select():
        for h in acc.holdings:
            holdings[h.ticker] = {"shares": h.shares, "account": acc.name}
    return holdings


def _fetch_prev_run_scores() -> tuple[date | None, dict[str, dict]]:
    """Best-effort fetch of the previous persisted brief run's scores.

    Returns (prev_run_date, {ticker: {"composite": float, "anomaly_flag": bool}}).
    Uses the same PostgREST pattern as ``fetch_brief_universe`` so scoring
    still works when Supabase is not configured or unreachable.
    """
    import httpx

    rest_url, api_key = _supabase_rest_config()
    if not rest_url or not api_key:
        return None, {}

    headers = {"apikey": api_key, "Accept": "application/json"}
    today = date.today().isoformat()
    try:
        with httpx.Client(timeout=15) as client:
            # Latest persisted run strictly before today.
            resp = client.get(
                f"{rest_url}/{_SUPABASE_TABLE_RUNS}"
                f"?select=id,run_date&run_date=lt.{today}"
                "&order=run_date.desc&limit=1",
                headers=headers,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return None, {}
            prev_date = rows[0].get("run_date")
            prev_id = rows[0].get("id")

            resp = client.get(
                f"{rest_url}/{_SUPABASE_TABLE_SCORES}"
                f"?select=ticker,composite,anomaly_flag&run_id=eq.{prev_id}",
                headers=headers,
            )
            resp.raise_for_status()
            prev: dict[str, dict] = {}
            for r in resp.json():
                ticker = r.get("ticker")
                if ticker:
                    prev[ticker] = {
                        "composite": float(r.get("composite") or 0.0),
                        "anomaly_flag": bool(r.get("anomaly_flag")),
                    }
        if prev_date:
            try:
                prev_dt = date.fromisoformat(str(prev_date))
            except ValueError:
                prev_dt = None
            return prev_dt, prev
    except Exception as exc:
        log.warning("Could not fetch previous brief run: %s", exc)
    return None, {}


def _recommendation(s: TickerScore) -> tuple[str, bool]:
    """Rule-based action hint based on conviction + position size.

    Returns (recommendation_text, overexposed_flag). Pure heuristics — useful
    as a nudge, not financial advice.
    """
    weight = s.portfolio_weight or 0.0
    overexposed = weight >= 20.0
    if s.composite >= 25:
        if overexposed:
            return "Hold — strong conviction but already a large position", True
        return "Add / accumulate", False
    if s.composite <= -25:
        if s.portfolio_weight and s.portfolio_weight > 0:
            return "Reduce / consider trimming", overexposed
        return "Avoid — no position", False
    if overexposed:
        return "Watch — large position, neutral conviction", True
    return "Hold / watch", False


def _enrich_run(run: BriefRun) -> None:
    """Fill in deltas, movers, portfolio context, and per-ticker details.

    Runs after scoring; every step is best-effort and never raises.
    """
    # --- Live prices for all scored tickers ---------------------------------
    scored_tickers = [s.ticker for s in run.scores]
    prices: dict[str, tuple[float, float]] = {}
    if scored_tickers:
        try:
            prices = data_client.get_current_prices_batch(scored_tickers)
        except Exception as exc:
            log.warning("Could not fetch live prices: %s", exc)

    for s in run.scores:
        price, prev_close = prices.get(s.ticker, (0.0, 0.0))
        s.price = round(float(price), 2)
        if price and prev_close:
            s.day_change_pct = round((price - prev_close) / prev_close * 100, 2)
        s.signal_agreement = _signal_agreement(s)

    # --- Portfolio weight / value for holdings ------------------------------
    holdings = _get_holdings_map()
    if holdings:
        try:
            usd_to_cad = data_client.get_usd_to_cad()
        except Exception:
            usd_to_cad = 1.0
        total_value = 0.0
        for t, info in holdings.items():
            price, _ = prices.get(t, (0.0, 0.0))
            multiplier = usd_to_cad if info["account"] == "USD" else 1.0
            info["value"] = info["shares"] * price * multiplier
            total_value += info["value"]
        run.portfolio_total_value = round(total_value, 2)
        if total_value > 0:
            for s in run.scores:
                info = holdings.get(s.ticker)
                if info:
                    s.portfolio_value = round(info["value"], 2)
                    s.portfolio_weight = round(info["value"] / total_value * 100, 2)

    # --- Actionability hints --------------------------------------------------
    for s in run.scores:
        s.recommendation, s.overexposed = _recommendation(s)

    # --- History deltas via the previous persisted run -----------------------
    prev_date, prev = _fetch_prev_run_scores()
    if prev_date is not None and prev:
        run.prev_run_date = prev_date
        prev_ranks = {t: i for i, t in enumerate(
            sorted(prev, key=lambda t: prev[t]["composite"], reverse=True)
        )}
        current_ranks = {s.ticker: i for i, s in enumerate(run.scores)}
        for rank, s in enumerate(run.scores):
            old = prev.get(s.ticker)
            if old is None:
                s.new_entrant = True
                run.new_entrants.append(s.ticker)
            else:
                s.prev_composite = old["composite"]
                s.score_delta = round(s.composite - old["composite"], 1)
                if old["composite"] is not None:
                    prev_rank = prev_ranks.get(s.ticker)
                    curr_rank = current_ranks[s.ticker]
                    if prev_rank is not None:
                        s.rank_change = prev_rank - curr_rank
                if not old["anomaly_flag"] and s.anomaly_flag:
                    s.anomaly_new = True
                    run.anomaly_flips.append({"ticker": s.ticker, "new": True})

        # Detected in the previous run but absent from ours.
        scored_set = {s.ticker for s in run.scores}
        run.exits = sorted(t for t in prev if t not in scored_set)

        movers = [s for s in run.scores if s.score_delta is not None]
        movers.sort(key=lambda s: abs(s.score_delta or 0.0), reverse=True)
        run.biggest_movers = movers[:5]


# ---------------------------------------------------------------------------
# Market overview (index moves + top market news)
# ---------------------------------------------------------------------------

_INDEX_SYMBOLS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "TSX 60": "^GSPTSE",
}


def fetch_market_overview() -> dict:
    """Snapshot major index closes and top market news.

    Best-effort: returns {"indices": {..}, "news": [..]} and never raises.
    Used by the nightly brief so the webhook and dashboard can show one line
    of non-stock-dependent context alongside the conviction scores.
    """
    indices: dict = {}
    for name, symbol in _INDEX_SYMBOLS.items():
        try:
            hist = data_client.get_price_history(symbol, period="1mo")
            closes = hist["Close"].dropna()
            if len(closes) >= 2:
                prev, last = closes.iloc[-2], closes.iloc[-1]
                chg_pct = round((last - prev) / prev * 100, 2) if prev else 0.0
                indices[name] = {"close": round(float(last), 2), "chg_pct": chg_pct}
        except Exception as exc:
            log.debug("Could not fetch %s index: %s", name, exc)

    try:
        news = data_client.get_macro_news()
    except Exception as exc:
        log.debug("Could not fetch macro news: %s", exc)
        news = []

    return {"indices": indices, "news": news}


def _market_summary_lines(run: BriefRun) -> list[str]:
    """One or two text lines summarizing run.market for webhook/CLI output."""
    if not run.market:
        return []
    idx = run.market.get("indices") or {}
    news = run.market.get("news") or []
    lines = [f"**Markets — {run.run_date.isoformat()}**"]
    parts = []
    for name in ("S&P 500", "NASDAQ", "TSX 60"):
        d = idx.get(name)
        if d:
            sign = "+" if d["chg_pct"] >= 0 else ""
            parts.append(f"{name} {sign}{d['chg_pct']:.2f}%")
    if parts:
        lines.append("  " + " · ".join(parts))
    if news:
        headline = (news[0].get("title") or "").strip()
        if headline:
            lines.append(f"  📰 {headline}")
    return lines


# ---------------------------------------------------------------------------
# Persistence to Supabase / Postgres
# ---------------------------------------------------------------------------

_SUPABASE_TABLE_RUNS = "brief_runs"
_SUPABASE_TABLE_SCORES = "ticker_scores"
_SUPABASE_TABLE_WATCHLIST = "watchlist"
_SUPABASE_TABLE_MARKET = "market_overview"
_SUPABASE_TABLE_PORTFOLIO = "portfolio_snapshots"


def _supabase_rest_config() -> tuple[str, str]:
    """Return (rest_url, api_key) used for Supabase REST calls, or ('', '')."""
    import os

    rest_url = os.getenv("DATABASE_REST_URL", "")
    api_key = os.getenv(
        "SUPABASE_PUBLISHABLE_KEY", os.getenv("SUPABASE_ANON_KEY", "")
    )
    return rest_url, api_key


def fetch_brief_universe() -> list[str]:
    """Return the ticker universe saved by the most recent persisted run.

    Each `brief --persist` uploads the holdings ∪ watchlist it scored, so a
    fresh CI runner with an empty local database can score the same stocks.
    The live Supabase watchlist is merged in so trackers added after the last
    persist are not missed. Returns [] when Supabase REST is not configured.
    """
    import httpx

    rest_url, api_key = _supabase_rest_config()
    if not rest_url or not api_key:
        return []

    headers = {
        "apikey": api_key,
        "Accept": "application/json",
    }
    tickers: list[str] = []
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{rest_url}/{_SUPABASE_TABLE_RUNS}"
                "?select=id,tickers&order=run_date.desc&limit=1",
                headers=headers,
            )
            resp.raise_for_status()
            rows = resp.json()
        if rows:
            tickers = rows[0].get("tickers") or []
            if not tickers and rows[0].get("id") is not None:
                # Legacy rows (pre-tickers column): fall back to the tickers
                # that were actually scored for the most recent run.
                try:
                    with httpx.Client(timeout=15) as client:
                        resp = client.get(
                            f"{rest_url}/{_SUPABASE_TABLE_SCORES}"
                            f"?select=ticker&run_id=eq.{rows[0]['id']}",
                            headers=headers,
                        )
                        resp.raise_for_status()
                        tickers = [r.get("ticker") for r in resp.json() if r.get("ticker")]
                except Exception as exc:
                    log.warning("Could not fetch scores fallback universe: %s", exc)
    except Exception as exc:
        log.warning("Could not fetch brief universe from Supabase: %s", exc)

    # Merge the live remote watchlist so newly-added trackers are not missed.
    live = fetch_brief_watchlist()
    tickers = sorted(set(tickers) | set(live))
    return [t for t in tickers if isinstance(t, str)]


def fetch_brief_watchlist() -> list[str]:
    """Return the watchlist currently mirrored in Supabase, or [].

    Best-effort: never raises, logs and returns [] on any failure.
    """
    import httpx

    rest_url, api_key = _supabase_rest_config()
    if not rest_url or not api_key:
        return []

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{rest_url}/{_SUPABASE_TABLE_WATCHLIST}?select=ticker",
                headers={"apikey": api_key, "Accept": "application/json"},
            )
            resp.raise_for_status()
        return sorted({r["ticker"] for r in resp.json() if r.get("ticker")})
    except Exception as exc:
        log.warning("Could not fetch Supabase watchlist: %s", exc)
        return []


def sync_watchlist_to_supabase(tickers: list[str] | None = None) -> bool:
    """Mirror the local watchlist into Supabase.

    Upserts every supplied ticker and deletes remote rows that no longer exist
    locally (skipped when the local list is empty, so a fresh CI runner cannot
    wipe the shared watchlist). Returns True on success, False on any failure.
    """
    import httpx

    rest_url, api_key = _supabase_rest_config()
    if not rest_url or not api_key:
        log.warning("Supabase REST not configured — skipping watchlist sync.")
        return False

    if tickers is None:
        tickers = get_watchlist()
    tickers = sorted({t.upper().strip() for t in tickers if t and t.strip()})

    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    try:
        with httpx.Client(timeout=15) as client:
            # Upsert local tickers.
            if tickers:
                resp = client.post(
                    f"{rest_url}/{_SUPABASE_TABLE_WATCHLIST}?on_conflict=ticker",
                    json=[{"ticker": t} for t in tickers],
                    headers=headers,
                )
                resp.raise_for_status()

            # Delete stale remote entries (only when we have a positive list,
            # so an empty local DB in CI cannot wipe the shared watchlist).
            if tickers:
                resp = client.get(
                    f"{rest_url}/{_SUPABASE_TABLE_WATCHLIST}?select=ticker",
                    headers={"apikey": api_key, "Accept": "application/json"},
                )
                resp.raise_for_status()
                remote = {r["ticker"] for r in resp.json() if r.get("ticker")}
                for stale in sorted(remote - set(tickers)):
                    resp = client.delete(
                        f"{rest_url}/{_SUPABASE_TABLE_WATCHLIST}?ticker=eq.{stale}",
                        headers={"apikey": api_key},
                    )
                    resp.raise_for_status()
        log.info("Synced %d watchlist tickers to Supabase.", len(tickers))
        return True
    except Exception as exc:
        log.error("Failed to sync watchlist to Supabase: %s", exc)
        return False


def persist_to_supabase(run: BriefRun) -> bool:
    """Write a BriefRun to Supabase Postgres via REST API.

    Returns True on success. Falls back gracefully if DATABASE_URL is missing.
    """
    import os
    import httpx

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        log.warning("DATABASE_URL not set — skipping Supabase persist.")
        return False

    # Use Supabase REST API (PostgREST) to insert rows with upsert semantics.
    # Requires DATABASE_REST_URL (e.g. https://<ref>.supabase.co/rest/v1)
    # and a project API key.  Prefer the new publishable key
    # (sb_publishable_...), falling back to the legacy anon key.
    rest_url, api_key = _supabase_rest_config()
    if not rest_url:
        log.warning("DATABASE_REST_URL not set — cannot persist via REST.")
        return False
    if not api_key:
        log.warning("Supabase API key not set — cannot persist via REST.")
        return False

    # NOTE: send the key on the `apikey` header ONLY.  New-style publishable
    # keys (sb_publishable_...) are not JWTs and are rejected if also passed
    # on the `Authorization: Bearer` header.
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    run_payload = {
        "run_date": run.run_date.isoformat(),
        "generated_at": run.generated_at.isoformat(),
        "weights_used": run.weights_used,
        "tickers": run.tickers or [],
    }

    try:
        with httpx.Client(timeout=30) as client:
            # Upsert run, return the row so we get the run_id for the scores.
            resp = client.post(
                f"{rest_url}/{_SUPABASE_TABLE_RUNS}?on_conflict=run_date",
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
                log.error("Could not determine run_id from Supabase response.")
                return False

            score_payloads = [
                {
                    "run_id": run_id,
                    "ticker": s.ticker,
                    "composite": s.composite,
                    "sentiment": s.sentiment,
                    "technical": s.technical,
                    "ml_pred": s.ml_pred,
                    "analyst": s.analyst,
                    "anomaly_flag": s.anomaly_flag,
                    "reasoning": s.reasoning,
                    "price": s.price or None,
                    "day_change_pct": s.day_change_pct or None,
                    "top_headline": s.top_headline or None,
                    "analyst_breakdown": s.analyst_breakdown or None,
                    "signal_agreement": s.signal_agreement or None,
                    "anomaly_detail": s.anomaly_detail or None,
                    "portfolio_weight": s.portfolio_weight or None,
                    "recommendation": s.recommendation or None,
                }
                for s in run.scores
            ]

            # Upsert scores in batches of 50
            for i in range(0, len(score_payloads), 50):
                batch = score_payloads[i : i + 50]
                resp = client.post(
                    f"{rest_url}/{_SUPABASE_TABLE_SCORES}?on_conflict=run_id,ticker",
                    json=batch,
                    headers=headers,
                )
                resp.raise_for_status()

            # Upsert the market overview (indices + top news) for this date.
            if run.market:
                market_payload = {
                    "run_date": run.run_date.isoformat(),
                    "indices": run.market.get("indices") or {},
                    "news": run.market.get("news") or [],
                    "generated_at": run.generated_at.isoformat(),
                }
                resp = client.post(
                    f"{rest_url}/{_SUPABASE_TABLE_MARKET}?on_conflict=run_date",
                    json=market_payload,
                    headers=headers,
                )
                resp.raise_for_status()

            # Upsert a live portfolio snapshot so CI/newsletter jobs can render
            # portfolio status without a local database (best-effort).
            try:
                from src import portfolio
                snap = portfolio.snapshot()
                if snap["rows"] or snap["cash"] or snap["invested"]:
                    snap_payload = {
                        "run_date": run.run_date.isoformat(),
                        "data": snap,
                        "generated_at": run.generated_at.isoformat(),
                    }
                    resp = client.post(
                        f"{rest_url}/{_SUPABASE_TABLE_PORTFOLIO}?on_conflict=run_date",
                        json=snap_payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
            except Exception as exc:
                log.warning("Could not persist portfolio snapshot: %s", exc)

        log.info(
            "Persisted brief run (id=%s) for %s with %d tickers.",
            run_id, run.run_date, len(run.scores),
        )
        return True

    except Exception as exc:
        log.error("Failed to persist brief run: %s", exc)
        return False


def fetch_remote_portfolio_snapshot() -> dict | None:
    """Latest portfolio snapshot from Supabase, or None when unavailable.

    Used by the newsletter when the local database has no holdings (e.g. CI),
    so the email can still include portfolio status from the last ``--persist``.
    """
    import httpx

    rest_url, api_key = _supabase_rest_config()
    if not rest_url or not api_key:
        return None

    try:
        headers = {"apikey": api_key, "Accept": "application/json"}
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{rest_url}/{_SUPABASE_TABLE_PORTFOLIO}"
                "?select=data&order=run_date.desc&limit=1",
                headers=headers,
            )
            resp.raise_for_status()
            rows = resp.json()
        return rows[0]["data"] if rows else None
    except Exception as exc:
        log.warning("Could not fetch portfolio snapshot: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Webhook notification
# ---------------------------------------------------------------------------


def notify_webhook(run: BriefRun, webhook_url: str | None = None) -> bool:
    """Post a short summary to a Discord/Slack/Telegram webhook."""
    import os
    import httpx

    url = webhook_url or os.getenv("NOTIFY_WEBHOOK_URL", "")
    if not url:
        return False

    top3 = run.scores[:3]
    bottom3 = run.scores[-3:]

    lines = [
        f"**Daily Brief — {run.run_date.isoformat()}**",
        f"Scored {len(run.scores)} tickers.\n",
    ]
    market_lines = _market_summary_lines(run)
    if market_lines:
        lines += market_lines + ["\n"]
    if top3:
        lines.append("**Top Conviction:**")
        for s in top3:
            emoji = "🟢" if s.composite >= 0 else "🔴"
            base = f"{emoji} {s.ticker}: {s.composite:+.1f}"
            if s.score_delta is not None:
                base += f" (Δ {s.score_delta:+.1f})"
            lines.append("  " + base)
    if bottom3:
        lines.append("\n**Lowest Conviction:**")
        for s in bottom3:
            emoji = "🟢" if s.composite >= 0 else "🔴"
            base = f"{emoji} {s.ticker}: {s.composite:+.1f}"
            if s.score_delta is not None:
                base += f" (Δ {s.score_delta:+.1f})"
            lines.append("  " + base)
    if run.biggest_movers:
        lines.append("\n**Biggest Movers:**")
        for s in run.biggest_movers:
            lines.append(f"  • {s.ticker}: {s.composite:+.1f} ({s.score_delta:+.1f} vs {run.prev_run_date})")
    if run.anomaly_flips:
        flips = ", ".join(f["ticker"] for f in run.anomaly_flips)
        lines.append(f"\n⚠️ Newly flagged anomalies: {flips}")
    if run.new_entrants:
        lines.append(f"🆕 New tickers: {', '.join(run.new_entrants)}")
    if run.exits:
        lines.append(f"➖ Dropped: {', '.join(run.exits)}")

    payload = {"content": "\n".join(lines)} if "discord" in url else {"text": "\n".join(lines)}

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
        return True
    except Exception as exc:
        log.error("Webhook notification failed: %s", exc)
        return False
