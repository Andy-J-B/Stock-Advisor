"""Tests for the daily email newsletter module (hermetic — no SMTP/network)."""

from datetime import date, datetime


from src import newsletter

from src.brief import TickerScore, BriefRun


def _fake_run() -> BriefRun:
    return BriefRun(
        run_date=date(2026, 9, 14),
        generated_at=datetime(2026, 9, 14, 21, 30),
        weights_used={"sentiment": 0.25, "technical": 0.25, "ml_pred": 0.25, "analyst": 0.25},
        market={
            "indices": {
                "S&P 500": {"close": 5500.0, "chg_pct": 0.85},
                "NASDAQ": {"close": 18500.0, "chg_pct": -0.4},
            },
            "news": [{"title": "Rates steady", "publisher": "Reuters", "link": ""}],
        },
        scores=[
            TickerScore(
                ticker="MSFT", composite=42.0, sentiment=30.0, technical=40.0,
                ml_pred=50.0, analyst=50.0, top_headline="MSFT cloud grows",
                analyst_breakdown="10 bull, 1 hold, 0 bear, 11 total",
                signal_agreement="High agreement", recommendation="Add / accumulate",
                price=400, day_change_pct=1.2,
            ),
            TickerScore(
                ticker="VCE.TO", composite=-45.0, sentiment=0.0, technical=-30.0,
                ml_pred=-60.0, analyst=0.0, anomaly_flag=True,
                anomaly_detail="Unusual activity in VCE.TO (1 day):\n  2026-05-06: vol z=2.8",
                signal_agreement="Conflicted", recommendation="Reduce / consider trimming",
                portfolio_weight=16.0, price=76.9, day_change_pct=-2.1,
            ),
            TickerScore(
                ticker="WATCHX", composite=5.0, sentiment=5.0, technical=0.0,
                ml_pred=0.0, analyst=10.0, signal_agreement="Moderate",
                recommendation="Hold / watch", price=10.0, day_change_pct=0.0,
            ),
        ],
    )


def _fake_snapshot() -> dict:
    return {
        "rows": [
            {"ticker": "VCE.TO", "account": "CAD", "shares": 50, "avg_price": 70.0,
             "price": 76.9, "day_pct": -2.1, "day_chg_cad": -82.0,
             "value_cad": 3845.0, "return_pct": 9.9, "return_cad": 345.0},
        ],
        "net_worth": 50000.0, "invested": 3845.0, "cash": 46155.0,
        "cost": 3500.0, "day_chg": -82.0, "day_pct": -0.16,
        "return_pct": 9.9, "return_cad": 345.0,
        "all_time_pct": 8.1, "all_time_cad": 3750.0, "fx_usd_cad": 1.36,
    }


def _fake_funds() -> dict:
    return {
        "MSFT": {"formatted_market_cap": "2.50B", "trailingPE": "30.10",
                 "forwardPE": "28.00", "priceToBook": "12.00", "dividendYield": "0.80%",
                 "fiftyTwoWeekHigh": "420.00", "fiftyTwoWeekLow": "300.00",
                 "targetMeanPrice": "450.00"},
    }


# ---------------------------------------------------------------------------
# Bucketing + rendering
# ---------------------------------------------------------------------------

def test_classify_buckets():
    b = newsletter._classify(_fake_run())
    assert len(b["buys"]) == 1 and b["buys"][0]["ticker"] == "MSFT"
    assert len(b["alerts"]) == 1 and b["alerts"][0]["ticker"] == "VCE.TO"
    assert len(b["watch"]) == 1 and b["watch"][0]["ticker"] == "WATCHX"
    assert len(b["invested"]) == 1 and b["invested"][0]["ticker"] == "VCE.TO"


def test_render_email_contains_sections():
    html = newsletter.render_email(_fake_run(), _fake_snapshot(), _fake_funds(),
                                   company_news={"MSFT": [{"title": "MSFT up", "link": ""}]})
    assert "Executive Summary" in html
    assert "Market Snapshot" in html
    assert "Top Stocks To Consider" in html
    assert "Watch Out" in html
    assert "Valuation Snapshot" in html
    assert "Company News" in html
    assert "MSFT" in html
    assert "VCE.TO" in html
    assert "informational purposes only" in html


def test_render_email_empty_data():
    run = BriefRun(run_date=date(2026, 9, 14), generated_at=datetime.now(), weights_used={})
    snap = {k: v for k, v in _fake_snapshot().items()}
    snap["rows"] = []
    html = newsletter.render_email(run, snap, {})
    assert "No equity positions" in html


def test_exec_summary_rules_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    text = newsletter._exec_summary(_fake_run(), {"indices": {}, "news": []}, _fake_snapshot())
    assert "MSFT" in text and "VCE.TO" in text


def test_recipients_default(monkeypatch):
    monkeypatch.delenv("EMAIL_TO", raising=False)
    assert newsletter.recipients() == [newsletter.DEFAULT_RECIPIENT]


def test_recipients_csv(monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "a@x.com, b@y.com")
    assert newsletter.recipients() == ["a@x.com", "b@y.com"]  # dict.fromkeys? no — split
    assert "b@y.com" in newsletter.recipients()


def test_fundamentals_fallback(monkeypatch):
    monkeypatch.setattr(newsletter.data_client, "get_ticker_info", lambda t: {})
    funds = newsletter._fundamentals(["MSFT", "ZZZ"])
    assert funds["MSFT"]["formatted_market_cap"] == "N/A"
    assert funds["ZZZ"]["trailingPE"] == "N/A"


def test_fundamentals_dividend_cap(monkeypatch):
    def fake_info(t):
        return {
            "marketCap": 1234e6,
            "trailingPE": 15.0,
            "dividendYield": 0.71,  # bogus CDR/ETF yield -> clamped to N/A
            "fiftyTwoWeekHigh": 12.5,
        }

    monkeypatch.setattr(newsletter.data_client, "get_ticker_info", fake_info)
    funds = newsletter._fundamentals(["ABC.TO"])
    assert funds["ABC.TO"]["dividendYield"] == "N/A"
    assert funds["ABC.TO"]["formatted_market_cap"] == "1.23B"
    assert funds["ABC.TO"]["trailingPE"] == "15.00"


def test_fund_table_flat_renders_values():
    out = newsletter._fund_table_flat(_fake_funds())
    assert "2.50B" in out and "30.10" in out and "450.00" in out
    assert "Market Cap" in out and "P/E (trailing)" in out


def test_render_email_modern_layout():
    html = newsletter.render_email(_fake_run(), _fake_snapshot(), _fake_funds())
    assert 'class="card"' in html          # rounded card container
    assert ".thead" in html                # gray header rows
    assert "prefers-color-scheme: dark" in html  # dark-mode media query
    assert "max-width: 640px" in html      # responsive media query
    assert "MIMEText" not in html


def test_rec_pill_kinds():
    assert "pill buy" in newsletter._rec_pill("Add / accumulate")
    assert "pill alert" in newsletter._rec_pill("Avoid — no position")
    assert "pill watch" in newsletter._rec_pill("Hold / watch")
    assert "pill muted" in newsletter._rec_pill(None)


def test_ticker_info_retries_on_empty(monkeypatch):
    calls = {"n": 0, "stored": None}

    class FakeTicker:
        def __init__(self, t):
            self.t = t

        @property
        def info(self):
            calls["n"] += 1
            if calls["n"] < 2:
                return {}  # simulate a transient Yahoo rate-limit
            return {"marketCap": 1e9, "trailingPE": 20.0}

    monkeypatch.setattr(newsletter.data_client, "cache_get", lambda k, ttl: None)
    monkeypatch.setattr(newsletter.data_client, "cache_set",
                        lambda k, v: calls.__setitem__("stored", v))
    monkeypatch.setattr(newsletter.data_client.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(newsletter.data_client.time, "sleep", lambda s: None)

    info = newsletter.data_client.get_ticker_info("ZZZ")
    assert info["marketCap"] == 1e9
    assert calls["n"] == 2          # second attempt succeeded
    assert calls["stored"]["trailingPE"] == 20.0


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def test_send_email_with_smtp(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k):
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *a):
            pass

        def ehlo(self):
            pass

        def has_extn(self, name):
            return name == "starttls"

        def starttls(self):
            pass

        def login(self, user, password):
            assert user == "me@gmail.com"
            assert password == "app-secret"

        def sendmail(self, frm, to, msg):
            sent["from"] = frm
            sent["to"] = to

    monkeypatch.setattr(newsletter.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-secret")
    monkeypatch.setenv("EMAIL_TO", "080.abae@gmail.com")

    assert newsletter.send_email("<html><body>hi</body></html>") is True
    assert sent["to"] == ["080.abae@gmail.com"]


def test_send_email_not_configured(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    assert newsletter.send_email("<html></html>") is False


def test_run_newsletter_preview(monkeypatch, tmp_path):
    monkeypatch.setattr(newsletter.brief, "run_brief", lambda fetch_market=False: _fake_run())
    monkeypatch.setattr(newsletter.brief, "fetch_remote_portfolio_snapshot", lambda: None)
    monkeypatch.setattr(newsletter.portfolio, "snapshot", lambda: {k: v for k, v in _fake_snapshot().items()})
    monkeypatch.setattr(newsletter.data_client, "get_ticker_info", lambda t: {})
    monkeypatch.setattr(newsletter.data_client, "get_ticker_news_batch", lambda t, limit=3: {})
    monkeypatch.chdir(tmp_path)

    ok = newsletter.run_newsletter(send=False)
    assert ok is False  # preview-only path
    previews = list(tmp_path.glob("data/newsletter_*.html"))
    assert len(previews) == 1
    assert "Executive Summary" in previews[0].read_text()