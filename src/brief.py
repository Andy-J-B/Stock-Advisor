"""
Nightly brief: composite conviction scoring for every holding/watchlist ticker.

Orchestrates existing modules (sentiment, indicators, ML, screener, anomaly)
into a single [-100, +100] conviction score per ticker per day.

No new external API calls are introduced beyond what the individual modules
already make.
"""

from __future__ import annotations

import logging
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


@dataclass
class BriefRun:
    run_date: date
    generated_at: datetime
    weights_used: dict[str, float]
    scores: list[TickerScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Component scorers – each normalizes to [-100, +100]
# ---------------------------------------------------------------------------


def score_sentiment(ticker: str) -> tuple[float, str]:
    """Score sentiment from recent news headlines. Returns (score, reasoning)."""
    resolved = data_client.get_ticker_news(ticker, limit=10)
    if not resolved:
        # Try US ticker via ticker_map
        from src.ticker_map import resolve_us_ticker
        us = resolve_us_ticker(ticker)
        if us != ticker:
            resolved = data_client.get_ticker_news(us, limit=10)
    if not resolved:
        return 0.0, "No news found"

    titles = [a.get("title", "") for a in resolved if a.get("title")]
    if not titles:
        return 0.0, "No valid headlines"

    engine = get_sentiment_engine()
    scores = engine.score_batch(titles)
    compounds = [s["compound"] for s in scores]
    avg = sum(compounds) / len(compounds) if compounds else 0.0

    # Scale compound [-1, 1] to [-100, 100]
    normalized = round(avg * 100, 1)
    pos = sum(1 for s in scores if s["label"] == "positive")
    neg = sum(1 for s in scores if s["label"] == "negative")
    reason = f"{len(titles)} headlines, {pos} positive, {neg} negative"
    return normalized, reason


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
        pred = ml_model.predict(model, latest.iloc[0])
    except Exception:
        return 0.0, "Prediction failed"

    prob_up = pred["probability_up"]
    # Map P(up) [0, 1] to [-100, +100]
    normalized = round((prob_up - 0.5) * 200, 1)
    reason = f"P(up)={prob_up:.1%}, label={pred['label']}"
    return normalized, reason


def score_analyst(ticker: str) -> tuple[float, str]:
    """Score analyst consensus. Returns (score, reasoning)."""
    from src.screener import _analyst, rating_label

    data = _analyst(ticker)
    if data is None:
        return 0.0, "No analyst coverage"

    data["total"] = (
        data["strong_buy"] + data["buy"] + data["hold"]
        + data["sell"] + data["strong_sell"]
    )
    total = max(data["total"], 1)
    bull = data["strong_buy"] + data["buy"]
    bear = data["sell"] + data["strong_sell"]
    hold = data["hold"]

    # Score: strong_buy=+2, buy=+1, hold=0, sell=-1, strong_sell=-2
    raw = (2 * data["strong_buy"] + data["buy"] - data["sell"] - 2 * data["strong_sell"]) / total
    # Normalize raw [-2, 2] to [-100, 100]
    normalized = round(raw * 50, 1)

    label, _ = rating_label(data)
    reason = f"{label} ({bull} bull, {hold} hold, {bear} bear, {total} total)"
    return normalized, reason


def detect_anomaly(ticker: str) -> bool:
    """Return True if the ticker shows anomalous activity."""
    ohlcv = data_client.get_price_history(ticker, period="6mo")
    if ohlcv.empty or ohlcv.shape[0] < 30:
        return False

    feat = features.build_features(ohlcv)
    flagged = anomaly.detect_anomalies(feat)
    return not flagged.empty


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def compute_composite(
    ticker: str,
    weights: dict[str, float],
) -> TickerScore:
    """Compute the conviction score for a single ticker."""
    log.info("Scoring %s ...", ticker)

    sentiment_score, sentiment_reason = score_sentiment(ticker)
    technical_score, technical_reason = score_technical(ticker)
    ml_score, ml_reason = score_ml(ticker)
    analyst_score, analyst_reason = score_analyst(ticker)
    anomaly_flag = detect_anomaly(ticker)

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
) -> BriefRun:
    """Score every ticker and return a BriefRun.

    Parameters
    ----------
    tickers : override ticker list (default = holdings ∪ watchlist).
    workers : thread pool size for parallel scoring.
    """
    if tickers is None:
        tickers = _get_all_tickers()
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

    return BriefRun(
        run_date=date.today(),
        generated_at=datetime.now(),
        weights_used=weights,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Persistence to Supabase / Postgres
# ---------------------------------------------------------------------------

_SUPABASE_TABLE_RUNS = "brief_runs"
_SUPABASE_TABLE_SCORES = "ticker_scores"


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
    rest_url = os.getenv("DATABASE_REST_URL", "")
    if not rest_url:
        log.warning("DATABASE_REST_URL not set — cannot persist via REST.")
        return False

    api_key = os.getenv(
        "SUPABASE_PUBLISHABLE_KEY", os.getenv("SUPABASE_ANON_KEY", "")
    )
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
    }

    try:
        with httpx.Client(timeout=30) as client:
            # Upsert run, return the row so we get the run_id for the scores.
            resp = client.post(
                f"{rest_url}/{_SUPABASE_TABLE_RUNS}",
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
                }
                for s in run.scores
            ]

            # Upsert scores in batches of 50
            for i in range(0, len(score_payloads), 50):
                batch = score_payloads[i : i + 50]
                resp = client.post(
                    f"{rest_url}/{_SUPABASE_TABLE_SCORES}",
                    json=batch,
                    headers=headers,
                )
                resp.raise_for_status()

        log.info(
            "Persisted brief run (id=%s) for %s with %d tickers.",
            run_id, run.run_date, len(run.scores),
        )
        return True

    except Exception as exc:
        log.error("Failed to persist brief run: %s", exc)
        return False


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
    if top3:
        lines.append("**Top Conviction:**")
        for s in top3:
            emoji = "🟢" if s.composite >= 0 else "🔴"
            lines.append(f"  {emoji} {s.ticker}: {s.composite:+.1f}")
    if bottom3:
        lines.append("\n**Lowest Conviction:**")
        for s in bottom3:
            emoji = "🟢" if s.composite >= 0 else "🔴"
            lines.append(f"  {emoji} {s.ticker}: {s.composite:+.1f}")

    payload = {"content": "\n".join(lines)} if "discord" in url else {"text": "\n".join(lines)}

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
        return True
    except Exception as exc:
        log.error("Webhook notification failed: %s", exc)
        return False
