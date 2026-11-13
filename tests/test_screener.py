"""Tests for src/screener — analyst buy-list screening."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd

from src import screener


class TestRatingLabel:
    def test_strong_buy(self):
        d = {"strong_buy": 10, "buy": 48, "hold": 2, "sell": 1, "strong_sell": 0, "total": 61}
        label, color = screener.rating_label(d)
        assert label == "Strong Buy"
        assert color == "green"

    def test_buy(self):
        d = {"strong_buy": 5, "buy": 34, "hold": 12, "sell": 0, "strong_sell": 2, "total": 53}
        label, _ = screener.rating_label(d)
        assert label == "Buy"

    def test_moderate_buy(self):
        d = {"strong_buy": 1, "buy": 10, "hold": 8, "sell": 1, "strong_sell": 0, "total": 20}
        label, _ = screener.rating_label(d)
        assert label == "Moderate Buy"

    def test_hold(self):
        d = {"strong_buy": 1, "buy": 10, "hold": 20, "sell": 5, "strong_sell": 4, "total": 40}
        label, _ = screener.rating_label(d)
        assert label == "Hold"


class TestFetchUniverse:
    def test_parses_wikitable(self):
        html = (
            '<table class="wikitable">'
            "<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>"
            "<tr><td>ABC</td><td>ABC Company</td><td>Info Tech</td></tr>"
            "<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td></tr>"
            "</table>"
        )
        with patch("src.screener._http.get") as mock_get, \
             patch("src.screener.cache_get", return_value=None), \
             patch("src.screener.cache_set"):
            mock_get.return_value = MagicMock(text=html)
            result = screener.fetch_universe("sp500")

        assert result == {"ABC": "ABC Company", "BRK-B": "Berkshire"}

    def test_unknown_universe_returns_empty(self):
        assert screener.fetch_universe("nope") == {}


class TestResolveSymbols:
    def test_maps_canadian_to_us_and_dedupes(self):
        with patch(
            "src.screener.ticker_map.resolve_tickers",
            return_value={"MSFT.NE": "MSFT", "NVDA.NE": "NVDA", "AAPL": "AAPL"},
        ):
            symbols, names = screener._resolve_symbols(["MSFT.NE", "AAPL", "NVDA.NE"])
        assert symbols == ["MSFT", "AAPL", "NVDA"]
        assert names["MSFT"] == "MSFT.NE"


class TestScreen:
    def test_filters_net_bullish_and_ranks(self):
        fake = {
            "AAA": {"current": 100, "target": 150, "strong_buy": 5, "buy": 10, "hold": 2, "sell": 0, "strong_sell": 0},
            "BBB": {"current": 50, "target": 60, "strong_buy": 0, "buy": 3, "hold": 8, "sell": 2, "strong_sell": 1},
            "CCC": {"current": 20, "target": 21, "strong_buy": 2, "buy": 5, "hold": 3, "sell": 1, "strong_sell": 0},
            "DDD": None,
        }
        with patch(
            "src.screener.fetch_universe",
            return_value={"AAA": "A Co", "BBB": "B Co", "CCC": "C Co", "DDD": "D Co"},
        ), patch("src.screener._analyst", side_effect=lambda s: fake[s]), patch(
            "src.screener._enrich", side_effect=lambda rows, **kw: rows
        ):
            rows = screener.screen(limit=10, min_analysts=2)

        # BBB is net-bearish, DDD has no coverage — both must be dropped.
        symbols = [r["symbol"] for r in rows]
        assert symbols == ["AAA", "CCC"]
        # Best upside + consensus ranks first.
        assert rows[0]["symbol"] == "AAA"
        assert rows[0]["upside"] > rows[1]["upside"]

    def test_min_analysts_filter(self):
        fake = {
            "AAA": {"current": 100, "target": 150, "strong_buy": 0, "buy": 1, "hold": 0, "sell": 0, "strong_sell": 0},
            "BBB": {"current": 10, "target": 12, "strong_buy": 0, "buy": 1, "hold": 0, "sell": 0, "strong_sell": 0},
        }
        with patch(
            "src.screener.fetch_universe", return_value={"AAA": "A Co", "BBB": "B Co"}
        ), patch("src.screener._analyst", side_effect=lambda s: fake[s]), patch(
            "src.screener._enrich", side_effect=lambda rows, **kw: rows
        ):
            # Only 1 analyst per name — below min_analysts=2, so nothing passes.
            rows = screener.screen(limit=10, min_analysts=2)
        assert rows == []

    def test_no_upside_filtered(self):
        fake = {
            "AAA": {"current": 100, "target": 90, "strong_buy": 5, "buy": 5, "hold": 0, "sell": 0, "strong_sell": 0},
            "BBB": {"current": 10, "target": 9, "strong_buy": 5, "buy": 5, "hold": 0, "sell": 0, "strong_sell": 0},
        }
        with patch(
            "src.screener.fetch_universe", return_value={"AAA": "A Co", "BBB": "B Co"}
        ), patch("src.screener._analyst", side_effect=lambda s: fake[s]), patch(
            "src.screener._enrich", side_effect=lambda rows, **kw: rows
        ):
            rows = screener.screen(limit=10, min_analysts=1)
        assert rows == []


class TestEnrich:
    def test_attaches_fundamentals(self):
        rows = [{"symbol": "AAPL", "company": ""}]
        with patch(
            "src.screener.data_client.get_ticker_info",
            return_value={
                "longName": "Apple Inc.",
                "marketCap": 3.5e12,
                "trailingPE": 30.0,
                "sector": "Technology",
            },
        ):
            enriched = screener._enrich(rows)
        assert enriched[0]["company"] == "Apple Inc."
        assert enriched[0]["market_cap"] == 3.5e12
        assert enriched[0]["pe"] == 30.0
        assert enriched[0]["sector"] == "Technology"


class TestDeepDive:
    def _picks(self):
        return [
            {
                "symbol": "MU",
                "company": "Micron Technology, Inc.",
                "name": "Micron Technology, Inc.",
                "current": 823.03,
                "target": 1522.26,
                "upside": 0.85,
                "total": 45,
                "strong_buy": 10,
                "buy": 35,
                "hold": 0,
                "sell": 0,
                "strong_sell": 0,
                "sector": "Technology",
                "pe": 18.6,
                "market_cap": 9.3e11,
            }
        ]

    def test_fallback_when_gemini_down(self):
        with patch("src.screener.data_client.get_ticker_news_batch", return_value={}), patch(
            "src.advisor._gemini_generate", return_value=None
        ):
            result = screener.deep_dive(self._picks(), top=3)
        assert "unavailable" in result

    def test_returns_narrative(self):
        with patch("src.screener.data_client.get_ticker_news_batch", return_value={}), patch(
            "src.advisor._gemini_generate", return_value="**MU** — buy note here."
        ):
            result = screener.deep_dive(self._picks(), top=3)
        assert "**MU**" in result


# ---------------------------------------------------------------------------
# Multi-factor scoring
# ---------------------------------------------------------------------------

GOOD_FUND = {
    "sector": "Technology",
    "roe": 0.28,
    "roa": 0.15,
    "gross_margin": 0.65,
    "operating_margin": 0.30,
    "profit_margin": 0.22,
    "fcf_yield": 0.06,
    "debt_equity": 0.20,
    "current_ratio": 2.2,
    "peg": 0.9,
    "forward_pe": 16.0,
    "ev_ebitda": 9.0,
    "ps": 3.0,
    "pb": 3.0,
    "revenue_growth": 0.18,
    "earnings_growth": 0.25,
    "earnings_quarterly_growth": 0.22,
    "eps_growth": 0.18,
    "beta": 1.0,
    "market_cap": 50e9,
    "recommendation_mean": 1.5,
    "short_float": 0.02,
    "fcf": 3e9,
    "dividend_rate": 0.0,
}

WEAK_FUND = {
    "sector": "Consumer Cyclical",
    "roe": 0.02,
    "roa": 0.01,
    "gross_margin": 0.15,
    "operating_margin": 0.03,
    "profit_margin": -0.02,
    "fcf_yield": -0.02,
    "debt_equity": 3.5,
    "current_ratio": 0.8,
    "peg": 4.5,
    "forward_pe": 45.0,
    "ev_ebitda": 30.0,
    "ps": 12.0,
    "pb": 10.0,
    "revenue_growth": -0.15,
    "earnings_growth": -0.30,
    "earnings_quarterly_growth": -0.40,
    "eps_growth": -0.20,
    "beta": 2.8,
    "market_cap": 500e6,
    "recommendation_mean": 3.5,
    "short_float": 0.30,
    "fcf": -1e8,
    "dividend_rate": 0.0,
}

GOOD_TECH = {
    "trend": "uptrend",
    "trend_score": 90.0,
    "ret_3m": 0.12,
    "ret_6m": 0.20,
    "rsi": 58.0,
    "from_high": -0.05,
    "volatility": 0.20,
}

WEAK_TECH = {
    "trend": "downtrend",
    "trend_score": 20.0,
    "ret_3m": -0.25,
    "ret_6m": -0.40,
    "rsi": 28.0,
    "from_high": -0.55,
    "volatility": 0.80,
}


def _row(
    symbol: str,
    *,
    upside: float,
    fund: dict | None,
    tech: dict | None,
    strong_buy: int = 8,
    buy: int = 15,
    hold: int = 2,
    sell: int = 0,
    strong_sell: int = 0,
) -> dict:
    total = strong_buy + buy + hold + sell + strong_sell
    return {
        "symbol": symbol,
        "current": 100.0,
        "target": 100.0 * (1 + upside),
        "target_low": 90.0,
        "target_high": 120.0,
        "upside": upside,
        "strong_buy": strong_buy,
        "buy": buy,
        "hold": hold,
        "sell": sell,
        "strong_sell": strong_sell,
        "total": total,
        "fundamentals": fund or {},
        "technical": tech or {},
        "revisions": {},
        "risk_flags": [],
    }


class TestFundamentalsNormalization:
    def test_normalizes_debt_equity_percentage(self):
        info = {
            "longName": "Test Co",
            "marketCap": 1e9,
            "debtToEquity": 28.465,
            "freeCashflow": 50e6,
            "operatingCashflow": 100e6,
            "profitMargins": 0.10,
        }
        with patch("src.screener.data_client.get_ticker_info", return_value=info):
            fund = screener._fundamentals("TEST")
        assert abs(fund["debt_equity"] - 0.28465) < 1e-9
        assert fund["fcf"] == 50e6

    def test_ignores_implausible_fcf_data_glitch(self):
        info = {
            "longName": "Test Co",
            "marketCap": 1e12,
            "freeCashflow": 3e9,       # broken single-quarter figure
            "operatingCashflow": 196e9,
            "profitMargins": 0.41,
        }
        with patch("src.screener.data_client.get_ticker_info", return_value=info):
            fund = screener._fundamentals("TEST")
        assert fund["fcf"] is None
        assert fund["fcf_yield"] is None


class TestPillarScores:
    def test_quality_prefers_profitable_low_debt_business(self):
        good, _, _ = screener._score_quality(GOOD_FUND)
        weak, _, _ = screener._score_quality(WEAK_FUND)
        assert good > 75
        assert weak < 30
        assert good > weak

    def test_valuation_prefers_cheap_growth(self):
        good, _, _ = screener._score_valuation(GOOD_FUND)
        weak, _, _ = screener._score_valuation(WEAK_FUND)
        assert good > weak
        assert good > 65
        assert weak < 30

    def test_growth_prefers_compounding_business(self):
        good, _, _ = screener._score_growth(GOOD_FUND)
        weak, _, _ = screener._score_growth(WEAK_FUND)
        assert good > 70
        assert weak < 30

    def test_momentum_prefers_uptrend_not_overbought(self):
        good, _, _ = screener._score_momentum(GOOD_TECH)
        weak, _, _ = screener._score_momentum(WEAK_TECH)
        assert good > weak
        assert good > 70

    def test_missing_data_is_neutral_not_a_crash(self):
        score, coverage, _ = screener._score_quality({})
        assert score == 50.0
        assert coverage == 0.0


class TestRiskAndLabels:
    def test_weak_business_gets_risk_flags(self):
        row = _row("WEAK", upside=0.60, fund=WEAK_FUND, tech=WEAK_TECH)
        penalty, flags = screener._risk_penalty(
            {**row, "total": 10, "target_low": 10, "target_high": 300, "target": 100},
            WEAK_FUND,
            WEAK_TECH,
        )
        assert penalty > 10
        assert any("cash flow" in f for f in flags)
        assert any("lever" in f for f in flags)

    def test_conviction_labels(self):
        assert screener.conviction_label({"score": 85, "risk_penalty": 0})[0] == "Strong Buy"
        assert screener.conviction_label({"score": 70, "risk_penalty": 0})[0] == "Buy"
        assert screener.conviction_label({"score": 55, "risk_penalty": 0})[0] == "Accumulate"
        assert screener.conviction_label({"score": 40, "risk_penalty": 15})[0] == "Speculative"
        assert screener.conviction_label({"score": 40, "risk_penalty": 0})[0] == "Watch"
        # High risk should downgrade an otherwise decent score.
        assert screener.conviction_label({"score": 68, "risk_penalty": 18})[0] == "Speculative"

    def test_underrated_tag_for_cheap_quality_with_weak_momentum(self):
        row = {
            "analyst_score": 70,
            "quality_score": 85,
            "valuation_score": 80,
            "growth_score": 60,
            "momentum_score": 30,
            "underrated_score": 82,
            "current": 100.0,
        }
        tags = screener._tags(row, {"dividend_rate": 0.0}, {"rsi": 45.0}, {})
        assert "Underrated" in tags
        assert "Quality" in tags
        assert "Value" in tags

    def test_dividend_tag_uses_annual_rate_over_price(self):
        row = {"current": 100.0}
        tags = screener._tags(row, {"dividend_rate": 3.0}, {}, {})
        assert "Dividend" in tags


class TestCompositeRanking:
    def test_quality_beats_stale_high_upside(self):
        quality = _row("QUAL", upside=0.08, fund=GOOD_FUND, tech=GOOD_TECH)
        junk = _row(
            "JUNK", upside=0.60, fund=WEAK_FUND, tech=WEAK_TECH,
            strong_buy=1, buy=4, hold=5, sell=3, strong_sell=2,
        )
        quality = screener._score_row(quality)
        junk = screener._score_row(junk)
        assert quality["score"] > junk["score"]
        assert quality["quality_score"] > junk["quality_score"]
        assert junk["risk_penalty"] > 0
        assert quality["verdict"] in ("Buy", "Strong Buy")


class TestTechnicalSnapshot:
    def _history(self, step: float) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=260, freq="B")
        close = [100 + step * i for i in range(260)]
        return pd.DataFrame({"Close": close}, index=idx)

    def test_detects_uptrend(self):
        with patch("src.screener.data_client.get_price_history", return_value=self._history(1.0)):
            snap = screener._technical_snapshot("UP")
        assert snap["trend"] == "uptrend"
        assert snap["ret_3m"] > 0
        assert snap["rsi"] > 50

    def test_detects_downtrend(self):
        with patch("src.screener.data_client.get_price_history", return_value=self._history(-0.5)):
            snap = screener._technical_snapshot("DOWN")
        assert snap["trend"] == "downtrend"
        assert snap["ret_3m"] < 0
        assert snap["from_high"] < 0


class TestRevisionSnapshot:
    def test_counts_recent_upgrades_and_target_raises(self):
        rec = pd.DataFrame({
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [10, 9, 6, 5],
            "buy": [20, 20, 18, 18],
            "hold": [5, 6, 10, 11],
            "sell": [0, 0, 1, 1],
            "strongSell": [0, 0, 0, 0],
        })
        actions = pd.DataFrame({
            "Action": ["up", "down", "main", "init"],
            "priceTargetAction": ["Raises", "Lowers", "Raises", "Maintains"],
        }, index=pd.to_datetime([
            "2026-11-01", "2026-10-15", "2026-09-01", "2026-08-01",
        ]))
        fake = MagicMock()
        fake.recommendations = rec
        fake.upgrades_downgrades = actions
        with patch("src.screener.yf.Ticker", return_value=fake), \
             patch("src.screener.cache_get", return_value=None), \
             patch("src.screener.cache_set"):
            rev = screener._revision_snapshot("TEST")

        assert rev["net_upgrades"] == 0  # one up, one down in the 90-day window
        assert rev["pt_raises"] == 2
        assert rev["pt_cuts"] == 1
        assert rev["bull_ratio_delta"] > 0


class TestScreenMultiFactor:
    def test_investigation_changes_the_ranking(self):
        analyst = {
            "QUAL": {
                "current": 100, "target": 108, "strong_buy": 10, "buy": 15,
                "hold": 2, "sell": 0, "strong_sell": 0,
            },
            # JUNK still passes the cheap net-bullish filter — it is only the
            # fundamental/technical investigation that reveals the problem.
            "JUNK": {
                "current": 100, "target": 160, "strong_buy": 2, "buy": 9,
                "hold": 5, "sell": 1, "strong_sell": 0,
            },
        }

        def enrich(rows, **kwargs):
            out = []
            for r in rows:
                r = dict(r)
                if r["symbol"] == "QUAL":
                    r["fundamentals"] = GOOD_FUND
                    r["technical"] = GOOD_TECH
                else:
                    r["fundamentals"] = WEAK_FUND
                    r["technical"] = WEAK_TECH
                out.append(r)
            return out

        with patch("src.screener.fetch_universe", return_value={"QUAL": "Quality Co", "JUNK": "Junk Co"}), \
             patch("src.screener._analyst", side_effect=lambda s: analyst[s]), \
             patch("src.screener._enrich", side_effect=enrich):
            rows = screener.screen(limit=2, min_analysts=2)

        assert [r["symbol"] for r in rows] == ["QUAL", "JUNK"]
        assert rows[0]["score"] > rows[1]["score"]
        assert rows[0]["verdict"] in ("Buy", "Strong Buy")
        for key in ("analyst_score", "quality_score", "valuation_score", "growth_score", "momentum_score", "tags"):
            assert key in rows[0]
