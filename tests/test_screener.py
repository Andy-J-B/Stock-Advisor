"""Tests for src/screener — analyst buy-list screening."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

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
