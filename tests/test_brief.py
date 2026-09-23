"""Tests for the nightly brief aggregation module."""

import pandas as pd
import pytest
from src.database import init_db, db, Account, Holding, Setting
from src import brief


@pytest.fixture(autouse=True)
def memory_db(monkeypatch):
    """Fresh in-memory DB for each test."""
    if not db.is_closed():
        db.close()
    original = db.database
    db.init(":memory:")
    db.connect()
    init_db(skip_migration=True)
    # Enrichment backfills live prices; stub the network calls so scoring
    # tests stay hermetic (no yfinance / FX requests).
    monkeypatch.setattr(
        brief.data_client, "get_current_prices_batch", lambda tickers: {
            t: (0.0, 0.0) for t in tickers
        }
    )
    monkeypatch.setattr(brief.data_client, "get_usd_to_cad", lambda: 1.0)
    yield
    db.drop_tables([Account, Holding, Setting])
    db.close()
    db.init(original)


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def test_score_sentiment_no_news(monkeypatch):
    monkeypatch.setattr(brief.data_client, "get_ticker_news", lambda *a, **k: [])
    score, reason, headline = brief.score_sentiment("MSFT")
    assert score == 0.0
    assert "No news" in reason
    assert headline == ""


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

    score, reason, headline = brief.score_sentiment("MSFT")
    assert score > 0  # positive headlines yield positive score
    assert score <= 100
    assert "positive" in reason
    assert headline == news[0]["title"]


def test_score_technical_insufficient_data(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(brief.data_client, "get_price_history", lambda *a, **k: pd.DataFrame())
    score, reason = brief.score_technical("MSFT")
    assert score == 0.0
    assert "Insufficient" in reason


def test_score_analyst_no_coverage(monkeypatch):
    import src.screener as screener
    monkeypatch.setattr(screener, "_analyst", lambda t: None)
    score, reason, breakdown = brief.score_analyst("MSFT")
    assert score == 0.0
    assert "No analyst coverage" in reason
    assert breakdown == ""


def test_score_analyst_uses_consensus(monkeypatch):
    import src.screener as screener
    data = {
        "current": 100.0, "target": 120.0,
        "strong_buy": 5, "buy": 10, "hold": 2, "sell": 0, "strong_sell": 0,
    }
    monkeypatch.setattr(screener, "_analyst", lambda t: data)
    score, reason, breakdown = brief.score_analyst("MSFT")
    assert score > 0
    assert "15 bull" in reason
    assert breakdown == "15 bull, 2 hold, 0 bear, 17 total"


def test_detect_anomaly_no_data(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(brief.data_client, "get_price_history", lambda *a, **k: pd.DataFrame())
    flag, detail = brief.detect_anomaly("MSFT")
    assert flag is False
    assert detail == ""


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def test_compute_composite_basic(monkeypatch):
    monkeypatch.setattr(brief, "score_sentiment", lambda t: (10.0, "news", "Headline A"))
    monkeypatch.setattr(brief, "score_technical", lambda t: (20.0, "tech"))
    monkeypatch.setattr(brief, "score_ml", lambda t: (30.0, "ml"))
    monkeypatch.setattr(brief, "score_analyst", lambda t: (40.0, "analyst", ""))
    monkeypatch.setattr(brief, "detect_anomaly", lambda t: (False, ""))

    weights = {"sentiment": 0.25, "technical": 0.20, "ml_pred": 0.30, "analyst": 0.25}
    result = brief.compute_composite("MSFT", weights)

    expected = 0.25 * 10 + 0.20 * 20 + 0.30 * 30 + 0.25 * 40
    assert result.composite == round(expected, 1)
    assert result.sentiment == 10.0
    assert result.technical == 20.0
    assert result.ml_pred == 30.0
    assert result.analyst == 40.0
    assert result.anomaly_flag is False
    assert result.top_headline == "Headline A"


def test_compute_composite_anomaly_penalty(monkeypatch):
    monkeypatch.setattr(brief, "score_sentiment", lambda t: (50.0, "news", ""))
    monkeypatch.setattr(brief, "score_technical", lambda t: (50.0, "tech"))
    monkeypatch.setattr(brief, "score_ml", lambda t: (50.0, "ml"))
    monkeypatch.setattr(brief, "score_analyst", lambda t: (50.0, "analyst", ""))
    monkeypatch.setattr(brief, "detect_anomaly", lambda t: (True, "Unusual volume"))

    weights = {"sentiment": 0.25, "technical": 0.20, "ml_pred": 0.30, "analyst": 0.25}
    result = brief.compute_composite("MSFT", weights)

    expected = 50.0 - 25.0  # all components = 50, minus 25 anomaly penalty
    assert result.composite == round(expected, 1)
    assert result.anomaly_flag is True
    assert result.anomaly_detail == "Unusual volume"
    assert "ANOMALY PENALTY" in result.reasoning


def test_compute_composite_clamps_to_100():
    result = brief.TickerScore(
        ticker="X", composite=150.0,
        sentiment=150, technical=120, ml_pred=140, analyst=130,
    )
    assert result.composite == 150.0  # dataclass doesn't clamp; compute does


def test_composite_clamped_in_compute(monkeypatch):
    monkeypatch.setattr(brief, "score_sentiment", lambda t: (100.0, "news", ""))
    monkeypatch.setattr(brief, "score_technical", lambda t: (100.0, "tech"))
    monkeypatch.setattr(brief, "score_ml", lambda t: (100.0, "ml"))
    monkeypatch.setattr(brief, "score_analyst", lambda t: (100.0, "analyst", ""))
    monkeypatch.setattr(brief, "detect_anomaly", lambda t: (False, ""))

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


# ---------------------------------------------------------------------------
# Rule-based action hints
# ---------------------------------------------------------------------------

def test_recommendation_bullish_no_position():
    s = brief.TickerScore(ticker="MSFT", composite=40.0)
    rec, over = brief._recommendation(s)
    assert rec == "Add / accumulate"
    assert not over


def test_recommendation_bullish_but_overexposed():
    s = brief.TickerScore(ticker="MSFT", composite=40.0, portfolio_weight=25.0)
    rec, over = brief._recommendation(s)
    assert "large position" in rec
    assert over


def test_recommendation_bearish_held():
    s = brief.TickerScore(ticker="MSFT", composite=-40.0, portfolio_weight=10.0)
    rec, over = brief._recommendation(s)
    assert "Reduce" in rec
    assert not over


def test_recommendation_bearish_no_position():
    s = brief.TickerScore(ticker="MSFT", composite=-40.0)
    rec, over = brief._recommendation(s)
    assert rec == "Avoid — no position"
    assert not over


def test_recommendation_overexposed_neutral():
    s = brief.TickerScore(ticker="MSFT", composite=5.0, portfolio_weight=30.0)
    rec, over = brief._recommendation(s)
    assert "Watch" in rec
    assert over


def test_recommendation_neutral():
    s = brief.TickerScore(ticker="MSFT", composite=5.0)
    rec, over = brief._recommendation(s)
    assert rec == "Hold / watch"


# ---------------------------------------------------------------------------
# Portfolio persistence (normalized snapshots + items + holdings)
# ---------------------------------------------------------------------------


class _CapResp:
    def __init__(self, json):
        self._json = json

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _CapClient:
    """Records every REST call; returns a fake id from the snapshot upsert."""

    def __init__(self):
        self.calls = []  # (method, url, payload)

    def post(self, url, json=None, headers=None):
        self.calls.append(("post", url, json))
        if "on_conflict=run_id" in url and "portfolio_snapshots" in url:
            return _CapResp([{"id": 7}])
        return _CapResp([])

    def get(self, url, headers=None):
        return _CapResp([])

    def delete(self, url, headers=None):
        self.calls.append(("delete", url, None))
        return _CapResp([])


def _fake_snap():
    return {
        "rows": [
            {"ticker": "MU.NE", "account": "CAD", "shares": 100, "avg_price": 40.0,
             "price": 45.0, "day_pct": 2.5, "day_chg_cad": 110.0,
             "value_cad": 4500.0, "return_pct": 12.5, "return_cad": 500.0},
            {"ticker": "MSFT.NE", "account": "CAD", "shares": 20, "avg_price": 30.0,
             "price": 32.0, "day_pct": 1.0, "day_chg_cad": 40.0,
             "value_cad": 640.0, "return_pct": 6.7, "return_cad": 40.0},
        ],
        "net_worth": 60000.0, "invested": 5140.0, "cash": 54860.0,
        "cost": 4600.0, "day_chg": 150.0, "day_pct": 3.0,
        "return_pct": 11.7, "return_cad": 540.0,
        "all_time_pct": 9.0, "all_time_cad": 5000.0, "fx_usd_cad": 1.36,
    }


def test_persist_portfolio_snapshot_payloads():
    client = _CapClient()
    from datetime import date, datetime
    run = brief.BriefRun(
        run_date=date(2026, 9, 22), generated_at=datetime(2026, 9, 22, 21, 0),
        weights_used={},
    )
    brief._persist_portfolio_snapshot(
        client, "https://db.example/rest/v1", {"apikey": "k"}, 5, run, _fake_snap()
    )
    posts = [c for c in client.calls if c[0] == "post"]

    # Affected tables: snapshots upsert + one items batch.
    assert posts[0][1].endswith("/portfolio_snapshots?on_conflict=run_id")
    assert posts[0][2]["run_id"] == 5
    assert posts[0][2]["net_worth_cad"] == 60000.0
    assert posts[1][1].endswith("/portfolio_snapshot_items?on_conflict=snapshot_id,account,ticker")
    items = posts[1][2]
    assert len(items) == 2
    assert items[0]["snapshot_id"] == 7 and items[0]["ticker"] == "MU.NE"
    assert items[0]["day_change_cad"] == 110.0 and items[0]["value_cad"] == 4500.0


def test_upsert_holdings_prunes_sold(monkeypatch, memory_db):
    from src.database import Account, Holding
    acc = Account.create(name="CAD", cash=0.0, initial_cash=0.0)
    Holding.create(account=acc, ticker="MU.NE", shares=100, avg_price=40.0)

    deleted_urls = []

    class Client:
        def post(self, url, json=None, headers=None):
            return _CapResp([])

        def get(self, url, headers=None):
            return _CapResp([{"account": "CAD", "ticker": "VINTAGE.TO"},
                             {"account": "USD"}])

        def delete(self, url, headers=None):
            deleted_urls.append(url)
            return _CapResp([])

    brief._upsert_portfolio_holdings(Client(), "https://db.example/rest/v1",
                                     {"apikey": "k"}, _fake_snap())
    # Keeps MU.NE, prunes the stale VINTAGE.TO (and wipes the empty USD account).
    assert any("not.in.(MU.NE)" in u for u in deleted_urls)
    assert any("account=eq.USD" in u and "not.in" not in u for u in deleted_urls)


def test_fetch_remote_portfolio_snapshot_mapping(monkeypatch):
    import sys
    import types

    monkeypatch.setattr(brief, "_supabase_rest_config",
                        lambda: ("https://db.example/rest/v1", "k"))

    snapshots_resp = [{
        "id": 7, "net_worth_cad": 60000, "invested_cad": 5140,
        "cash_cad": 54860, "cost_cad": 4600, "day_change_cad": 150,
        "day_pct": 3.0, "return_pct": 11.7, "return_cad": 540,
        "all_time_pct": 9.0, "all_time_cad": 5000, "fx_usd_cad": 1.36,
    }]
    items_resp = [
        {"ticker": "MU.NE", "account": "CAD", "shares": 100,
         "avg_price": 40.0, "price": 45.0, "day_change_pct": 2.5,
         "day_change_cad": 110.0, "value_cad": 4500.0,
         "return_pct": 12.5, "return_cad": 500.0},
    ]

    class CapturingClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url, headers=None):
            if "/portfolio_snapshots?" in url:
                assert "portfolio_snapshot_items(" not in url  # no embed
                return _CapResp(snapshots_resp)
            assert "snapshot_id=eq.7&order=value_cad.desc" in url
            return _CapResp(items_resp)

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = CapturingClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    snap = brief.fetch_remote_portfolio_snapshot()
    assert snap is not None
    assert snap["net_worth"] == 60000 and snap["day_chg"] == 150
    assert snap["rows"][0]["ticker"] == "MU.NE"
    assert snap["rows"][0]["day_chg_cad"] == 110.0
    assert snap["fx_usd_cad"] == 1.36


def test_fetch_remote_portfolio_snapshot_unconfigured(monkeypatch):
    monkeypatch.setattr(brief, "_supabase_rest_config", lambda: ("", ""))
    assert brief.fetch_remote_portfolio_snapshot() is None