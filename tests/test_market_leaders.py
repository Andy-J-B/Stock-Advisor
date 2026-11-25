"""Tests for the market-leaders snapshot + Supabase persistence."""

import pytest

from src import market_leaders


class _CapResp:
    def __init__(self, json):
        self._json = json

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _CapClient:
    """Records every Supabase REST call made by the stock-status layer."""

    def __init__(self, *, run_id=77, runs=None, rows=None, date_runs=None):
        self.calls = []
        self.run_id = run_id
        self.runs = runs or []
        self.rows = rows or []
        self.date_runs = date_runs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def post(self, url, json=None, headers=None):
        self.calls.append(("post", url, json))
        if "stock_status_runs" in url:
            return _CapResp([{"id": self.run_id}])
        return _CapResp([])

    def get(self, url, headers=None):
        self.calls.append(("get", url, None))
        if "stock_status_runs" in url:
            if "run_date=eq." in url and self.date_runs is not None:
                return _CapResp(self.date_runs)
            return _CapResp(self.runs)
        return _CapResp(self.rows)

    def delete(self, url, headers=None):
        self.calls.append(("delete", url, None))
        return _CapResp([])


def _analysts(**counts) -> dict:
    out = {
        "strong_buy": 0,
        "buy": 0,
        "hold": 0,
        "sell": 0,
        "strong_sell": 0,
    }
    out.update(counts)
    out["total"] = sum(out.values())
    return out


def _row(symbol: str, **overrides) -> dict:
    row = {
        "symbol": symbol,
        "company": f"{symbol} Inc.",
        "currency": "USD",
        "price": 100.0,
        "day_pct": 1.5,
        "market_cap": 2.5e12,
        "pe": 30.0,
        "fwd_pe": 25.0,
        "div_yield": 0.5,
        "high_52w": 120.0,
        "low_52w": 80.0,
        "ath": 130.0,
        "ath_date": "2026-11-01",
        "from_ath_pct": -23.1,
        "target": 120.0,
        "upside_pct": 20.0,
        "analysts": _analysts(strong_buy=20, buy=10, hold=2),
    }
    row.update(overrides)
    return row


def _groups() -> dict:
    return {
        "US": [_row("NVDA"), _row("MSFT", from_ath_pct=-10.0)],
        "Canada": [
            _row(
                "RY.TO",
                currency="CAD",
                market_cap=2.0e11,
                analysts=_analysts(buy=8, hold=3),
            )
        ],
    }


def test_persist_snapshot_upserts_run_and_rows(monkeypatch):
    monkeypatch.setenv("DATABASE_REST_URL", "https://db.example/rest/v1")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "k")
    client = _CapClient()
    monkeypatch.setattr(market_leaders.httpx, "Client", lambda *a, **k: client)

    ok = market_leaders.persist_snapshot(
        _groups(), run_date="2026-11-22", us_n=2, ca_n=1, sort="mcap"
    )

    assert ok is True

    run_post = next(
        c
        for c in client.calls
        if c[0] == "post" and c[1].endswith("/stock_status_runs?on_conflict=run_date")
    )
    assert run_post[2]["run_date"] == "2026-11-22"
    assert run_post[2]["universe"] == "US, Canada"
    assert run_post[2]["sort"] == "mcap"
    assert run_post[2]["us_count"] == 2
    assert run_post[2]["ca_count"] == 1
    assert run_post[2]["total_count"] == 3
    assert run_post[2]["groups"] == ["US", "Canada"]

    rows_post = next(
        c
        for c in client.calls
        if c[0] == "post"
        and c[1].endswith("/stock_status_rows?on_conflict=run_id,group_name,symbol")
    )
    first = rows_post[2][0]
    assert first["run_id"] == 77
    assert first["group_name"] == "US"
    assert first["rank"] == 1
    assert first["symbol"] == "NVDA"
    assert first["consensus"] == "Strong Buy"
    assert first["ath"] == 130.0
    assert first["from_ath_pct"] == -23.1
    assert first["total_analysts"] == 32

    assert any(c[0] == "delete" and "run_id=eq.77" in c[1] for c in client.calls)


def test_persist_snapshot_unconfigured(monkeypatch):
    monkeypatch.delenv("DATABASE_REST_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    assert market_leaders.persist_snapshot(_groups()) is False


def test_persist_snapshot_empty(monkeypatch):
    monkeypatch.setenv("DATABASE_REST_URL", "https://db.example/rest/v1")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "k")
    assert market_leaders.persist_snapshot({}) is False


def test_fetch_snapshot_loads_run_and_rows(monkeypatch):
    monkeypatch.setenv("DATABASE_REST_URL", "https://db.example/rest/v1")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "k")
    client = _CapClient(
        runs=[{"id": 9, "run_date": "2026-11-22", "universe": "US, Canada"}],
        rows=[
            {"run_id": 9, "group_name": "US", "rank": 1, "symbol": "NVDA"},
            {"run_id": 9, "group_name": "Canada", "rank": 1, "symbol": "RY.TO"},
        ],
    )
    monkeypatch.setattr(market_leaders.httpx, "Client", lambda *a, **k: client)

    snapshot = market_leaders.fetch_snapshot("2026-11-22")

    assert snapshot is not None
    assert snapshot["run"]["id"] == 9
    assert snapshot["rows"][0]["symbol"] == "NVDA"
    assert any("run_date=eq.2026-11-22" in c[1] for c in client.calls)
    assert any(
        "run_id=eq.9&order=group_name.asc,rank.asc" in c[1] for c in client.calls
    )


def test_fetch_snapshot_falls_back_to_latest(monkeypatch):
    monkeypatch.setenv("DATABASE_REST_URL", "https://db.example/rest/v1")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "k")
    client = _CapClient(
        runs=[{"id": 5, "run_date": "2026-11-21"}],
        rows=[{"run_id": 5, "group_name": "US", "rank": 1, "symbol": "AAPL"}],
        date_runs=[],
    )
    monkeypatch.setattr(market_leaders.httpx, "Client", lambda *a, **k: client)

    snapshot = market_leaders.fetch_snapshot("2026-11-22")

    assert snapshot is not None
    assert snapshot["run"]["id"] == 5
    assert any("order=run_date.desc" in c[1] for c in client.calls)


def test_fetch_snapshot_unconfigured(monkeypatch):
    monkeypatch.delenv("DATABASE_REST_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    assert market_leaders.fetch_snapshot() is None


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [("RY.TO", "CAD"), ("SHOP.V", "CAD"), ("NVDA", "USD"), ("ABC.NE", "CAD")],
)
def test_currency_of(ticker, expected):
    assert market_leaders.currency_of(ticker) == expected


def test_consensus_label_thresholds():
    assert market_leaders.consensus_label(_analysts(strong_buy=10, buy=2))[0] == "Strong Buy"
    assert market_leaders.consensus_label(_analysts(buy=5, hold=5))[0] == "Buy"
    assert market_leaders.consensus_label(_analysts(hold=10))[0] == "Hold"
    assert market_leaders.consensus_label(_analysts(sell=10))[0] == "Sell"
    assert market_leaders.consensus_label(_analysts(strong_sell=10))[0] == "Strong Sell"
    assert market_leaders.consensus_label(_analysts())[0] == "N/A"
