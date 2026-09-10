"""Tests for the nightly brief aggregation module."""

import pandas as pd
import pytest
from src.database import init_db, db, Account, Holding, Setting
from src import brief


@pytest.fixture(autouse=True)
def memory_db():
    """Fresh in-memory DB for each test."""
    if not db.is_closed():
        db.close()
    original = db.database
    db.init(":memory:")
    db.connect()
    init_db(skip_migration=True)
    yield
    db.drop_tables([Account, Holding, Setting])
    db.close()
    db.init(original)


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def test_score_sentiment_no_news(monkeypatch):
    monkeypatch.setattr(brief.data_client, "get_ticker_news", lambda *a, **k: [])
    score, reason = brief.score_sentiment("MSFT")
    assert score == 0.0
    assert "No news" in reason


def test_score_sentiment_uses_finbert(monkeypatch):
    news = [
        {"title": "Company beats earnings expectations"},
        {"title": "Company announces new product line"},
    ]

    class FakeEngine:
        def score_batch(self, texts):
            return [
                {"label": "positive", "compound": 0.85, "positive": 0.9, "negative": 0.05, "neutral": 0.05},
                {"label": "positive", "compound": 0.70, "positive": 0.8, "negative": 0.1, "neutral": 0.1},
            ]

    monkeypatch.setattr(brief.data_client, "get_ticker_news", lambda *a, **k: news)
    monkeypatch.setattr(brief, "get_sentiment_engine", lambda: FakeEngine())

    score, reason = brief.score_sentiment("MSFT")
    assert score > 0  # positive headlines yield positive score
    assert score <= 100
    assert "positive" in reason


def test_score_technical_insufficient_data(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(brief.data_client, "get_price_history", lambda *a, **k: pd.DataFrame())
    score, reason = brief.score_technical("MSFT")
    assert score == 0.0
    assert "Insufficient" in reason


def test_score_analyst_no_coverage(monkeypatch):
    import src.screener as screener
    monkeypatch.setattr(screener, "_analyst", lambda t: None)
    score, reason = brief.score_analyst("MSFT")
    assert score == 0.0
    assert "No analyst coverage" in reason


def test_score_analyst_uses_consensus(monkeypatch):
    import src.screener as screener
    data = {
        "current": 100.0, "target": 120.0,
        "strong_buy": 5, "buy": 10, "hold": 2, "sell": 0, "strong_sell": 0,
    }
    monkeypatch.setattr(screener, "_analyst", lambda t: data)
    score, reason = brief.score_analyst("MSFT")
    assert score > 0
    assert "15 bull" in reason


def test_detect_anomaly_no_data(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(brief.data_client, "get_price_history", lambda *a, **k: pd.DataFrame())
    assert brief.detect_anomaly("MSFT") is False


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def test_compute_composite_basic(monkeypatch):
    monkeypatch.setattr(brief, "score_sentiment", lambda t: (10.0, "news"))
    monkeypatch.setattr(brief, "score_technical", lambda t: (20.0, "tech"))
    monkeypatch.setattr(brief, "score_ml", lambda t: (30.0, "ml"))
    monkeypatch.setattr(brief, "score_analyst", lambda t: (40.0, "analyst"))
    monkeypatch.setattr(brief, "detect_anomaly", lambda t: False)

    weights = {"sentiment": 0.25, "technical": 0.20, "ml_pred": 0.30, "analyst": 0.25}
    result = brief.compute_composite("MSFT", weights)

    expected = 0.25 * 10 + 0.20 * 20 + 0.30 * 30 + 0.25 * 40
    assert result.composite == round(expected, 1)
    assert result.sentiment == 10.0
    assert result.technical == 20.0
    assert result.ml_pred == 30.0
    assert result.analyst == 40.0
    assert result.anomaly_flag is False


def test_compute_composite_anomaly_penalty(monkeypatch):
    monkeypatch.setattr(brief, "score_sentiment", lambda t: (50.0, "news"))
    monkeypatch.setattr(brief, "score_technical", lambda t: (50.0, "tech"))
    monkeypatch.setattr(brief, "score_ml", lambda t: (50.0, "ml"))
    monkeypatch.setattr(brief, "score_analyst", lambda t: (50.0, "analyst"))
    monkeypatch.setattr(brief, "detect_anomaly", lambda t: True)

    weights = {"sentiment": 0.25, "technical": 0.20, "ml_pred": 0.30, "analyst": 0.25}
    result = brief.compute_composite("MSFT", weights)

    expected = 50.0 - 25.0  # all components = 50, minus 25 anomaly penalty
    assert result.composite == round(expected, 1)
    assert result.anomaly_flag is True
    assert "ANOMALY PENALTY" in result.reasoning


def test_compute_composite_clamps_to_100():
    result = brief.TickerScore(
        ticker="X", composite=150.0,
        sentiment=150, technical=120, ml_pred=140, analyst=130,
    )
    assert result.composite == 150.0  # dataclass doesn't clamp; compute does


def test_composite_clamped_in_compute(monkeypatch):
    monkeypatch.setattr(brief, "score_sentiment", lambda t: (100.0, "news"))
    monkeypatch.setattr(brief, "score_technical", lambda t: (100.0, "tech"))
    monkeypatch.setattr(brief, "score_ml", lambda t: (100.0, "ml"))
    monkeypatch.setattr(brief, "score_analyst", lambda t: (100.0, "analyst"))
    monkeypatch.setattr(brief, "detect_anomaly", lambda t: False)

    weights = {"sentiment": 0.25, "technical": 0.20, "ml_pred": 0.30, "analyst": 0.25}
    result = brief.compute_composite("MSFT", weights)
    assert result.composite <= 100.0


# ---------------------------------------------------------------------------
# run_brief
# ---------------------------------------------------------------------------

def test_run_brief_empty_portfolio():
    run = brief.run_brief(tickers=[])
    assert run.scores == []


def test_run_brief_scores_tickers(monkeypatch):
    monkeypatch.setattr(brief, "compute_composite", lambda t, w: brief.TickerScore(
        ticker=t,
        composite=50.0,
        sentiment=10.0,
        technical=20.0,
        ml_pred=30.0,
        analyst=40.0,
    ))
    run = brief.run_brief(tickers=["AAPL", "MSFT"])
    assert len(run.scores) == 2
    assert {s.ticker for s in run.scores} == {"AAPL", "MSFT"}


def test_run_brief_handles_errors(monkeypatch):
    def boom(t, w):
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(brief, "compute_composite", boom)
    run = brief.run_brief(tickers=["MSFT"])
    assert len(run.scores) == 1
    assert run.scores[0].composite == 0.0
    assert "Error" in run.scores[0].reasoning


def test_run_brief_uses_holdings_and_watchlist(monkeypatch):
    acc = Account.create(name="USD", cash=0.0, initial_cash=0.0)
    Holding.create(account=acc, ticker="AAPL", shares=10, avg_price=150.0)
    from src.database import add_to_watchlist
    add_to_watchlist("NVDA")

    monkeypatch.setattr(brief, "compute_composite", lambda t, w: brief.TickerScore(
        ticker=t, composite=0.0,
    ))

    run = brief.run_brief()
    assert {s.ticker for s in run.scores} == {"AAPL", "NVDA"}


# ---------------------------------------------------------------------------
# Persistence & notification fallbacks
# ---------------------------------------------------------------------------

def test_persist_without_env(monkeypatch):
    monkeypatch.setattr("os.getenv", lambda k, d="": "" if k == "DATABASE_URL" else d)
    run = brief.run_brief(tickers=[])
    assert brief.persist_to_supabase(run) is False


def test_notify_without_env(monkeypatch):
    monkeypatch.setattr("os.getenv", lambda k, d="": "")
    run = brief.run_brief(tickers=[])
    assert brief.notify_webhook(run) is False


def test_sync_watchlist_without_env(monkeypatch):
    monkeypatch.setattr("os.getenv", lambda k, d="": "")
    assert brief.sync_watchlist_to_supabase(["AAPL"]) is False


def test_fetch_brief_watchlist_without_env(monkeypatch):
    monkeypatch.setattr("os.getenv", lambda k, d="": "")
    assert brief.fetch_brief_watchlist() == []


# ---------------------------------------------------------------------------
# Market overview (index moves + top news)
# ---------------------------------------------------------------------------

def _fake_history(closes):
    idx = pd.date_range("2026-09-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_fetch_market_overview_best_effort(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(brief.data_client, "get_price_history", boom)
    monkeypatch.setattr(brief.data_client, "get_macro_news", boom)
    mkt = brief.fetch_market_overview()
    assert mkt == {"indices": {}, "news": []}


def test_fetch_market_overview_captures_data(monkeypatch):
    monkeypatch.setattr(
        brief.data_client, "get_price_history",
        lambda *a, **k: _fake_history([100.0, 101.5]),
    )
    monkeypatch.setattr(
        brief.data_client, "get_macro_news",
        lambda *a, **k: [{"title": "Markets rally", "publisher": "X", "link": "http://x"}],
    )
    mkt = brief.fetch_market_overview()
    assert mkt["indices"]["S&P 500"]["close"] == 101.5
    assert abs(mkt["indices"]["S&P 500"]["chg_pct"] - 1.5) < 1e-6
    assert mkt["news"][0]["title"] == "Markets rally"


def test_market_summary_lines_renders(monkeypatch):
    run = brief.run_brief(tickers=[])
    run.market = {
        "indices": {"S&P 500": {"close": 101.5, "chg_pct": 1.5},
                    "NASDAQ": {"close": 2.0, "chg_pct": -0.5}},
        "news": [{"title": "Markets rally", "publisher": "X", "link": ""}],
    }
    lines = brief._market_summary_lines(run)
    joined = "\n".join(lines)
    assert "Markets —" in joined
    assert "S&P 500 +1.50%" in joined
    assert "📰 Markets rally" in joined


def test_market_summary_lines_empty(monkeypatch):
    run = brief.run_brief(tickers=[])
    assert brief._market_summary_lines(run) == []