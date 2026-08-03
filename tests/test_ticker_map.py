"""Tests for Canadian→US ticker mapping."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# _is_canadian helper
# ---------------------------------------------------------------------------
class TestIsCanadian:
    def test_ne_suffix(self):
        from src.ticker_map import _is_canadian
        assert _is_canadian("MSFT.NE") is True

    def test_to_suffix(self):
        from src.ticker_map import _is_canadian
        assert _is_canadian("VFV.TO") is True

    def test_us_ticker(self):
        from src.ticker_map import _is_canadian
        assert _is_canadian("AAPL") is False

    def test_case_insensitive(self):
        from src.ticker_map import _is_canadian
        assert _is_canadian("msft.ne") is True
        assert _is_canadian("vfv.to") is True


# ---------------------------------------------------------------------------
# resolve_us_ticker
# ---------------------------------------------------------------------------
class TestResolveUsTicker:
    def test_us_ticker_unchanged(self):
        from src.ticker_map import resolve_us_ticker
        assert resolve_us_ticker("AAPL") == "AAPL"
        assert resolve_us_ticker("MSFT") == "MSFT"

    def test_static_mapping_visa(self):
        from src.ticker_map import resolve_us_ticker
        with patch("src.ticker_map.cache_get", return_value=None), \
             patch("src.ticker_map.cache_set"):
            assert resolve_us_ticker("VISA.NE") == "V"

    def test_static_mapping_brk(self):
        from src.ticker_map import resolve_us_ticker
        with patch("src.ticker_map.cache_get", return_value=None), \
             patch("src.ticker_map.cache_set"):
            assert resolve_us_ticker("BRK.TO") == "BRK-B"

    @patch("src.ticker_map.yf.Ticker")
    @patch("src.ticker_map.cache_set")
    @patch("src.ticker_map.cache_get", return_value=None)
    def test_cdr_base_works(self, mock_get, mock_set, mock_ticker):
        from src.ticker_map import resolve_us_ticker
        mock_ticker.return_value.info = {"exchange": "NMS"}
        assert resolve_us_ticker("MSFT.NE") == "MSFT"

    @patch("src.ticker_map.yf.Ticker")
    @patch("src.ticker_map.cache_set")
    @patch("src.ticker_map.cache_get", return_value=None)
    def test_cdr_ubernet(self, mock_get, mock_set, mock_ticker):
        from src.ticker_map import resolve_us_ticker
        mock_ticker.return_value.info = {"exchange": "NYQ"}
        assert resolve_us_ticker("UBER.NE") == "UBER"

    @patch("src.ticker_map.yf.Ticker")
    @patch("src.ticker_map.cache_set")
    @patch("src.ticker_map.cache_get", return_value=None)
    def test_canadian_only_passthrough(self, mock_get, mock_set, mock_ticker):
        from src.ticker_map import resolve_us_ticker
        mock_ticker.return_value.info = {"exchange": "TOR"}
        assert resolve_us_ticker("VFV.TO") == "VFV.TO"

    @patch("src.ticker_map.yf.Ticker")
    @patch("src.ticker_map.cache_set")
    @patch("src.ticker_map.cache_get", return_value=None)
    def test_canadian_only_guard_beats_false_us_exchange(self, mock_get, mock_set, mock_ticker):
        from src.ticker_map import resolve_us_ticker
        # yfinance returns a phantom US exchange for these bases; the guard
        # must keep them on TSX instead of resolving to the phantom listing.
        mock_ticker.return_value.info = {"exchange": "NMS", "quoteType": "ECNQUOTE"}
        assert resolve_us_ticker("VCE.TO") == "VCE.TO"
        assert resolve_us_ticker("VFV.TO") == "VFV.TO"
        mock_ticker.assert_not_called()

    @patch("src.ticker_map.yf.Ticker")
    @patch("src.ticker_map.cache_set")
    @patch("src.ticker_map.cache_get", return_value=None)
    def test_no_exchange_passthrough(self, mock_get, mock_set, mock_ticker):
        from src.ticker_map import resolve_us_ticker
        mock_ticker.return_value.info = {}
        assert resolve_us_ticker("UNKNOWN.NE") == "UNKNOWN.NE"

    @patch("src.ticker_map.yf.Ticker")
    @patch("src.ticker_map.cache_set")
    @patch("src.ticker_map.cache_get", return_value=None)
    def test_yfinance_exception_passthrough(self, mock_get, mock_set, mock_ticker):
        from src.ticker_map import resolve_us_ticker
        mock_ticker.side_effect = Exception("network error")
        assert resolve_us_ticker("FAKE.NE") == "FAKE.NE"

    def test_result_cached(self):
        from src.ticker_map import resolve_us_ticker
        # Second call hits cache, returns without yfinance
        with patch("src.ticker_map.cache_get", return_value="CACHED"), \
             patch("src.ticker_map.yf.Ticker") as mock_ticker:
            result = resolve_us_ticker("CACHED.NE")
            assert result == "CACHED"
            mock_ticker.assert_not_called()

    def test_negative_result_cached(self):
        from src.ticker_map import resolve_us_ticker
        # Cached negative result (original returned)
        with patch("src.ticker_map.cache_get", return_value="NEGTEST.TO"), \
             patch("src.ticker_map.yf.Ticker") as mock_ticker:
            result = resolve_us_ticker("NEGTEST.TO")
            assert result == "NEGTEST.TO"
            mock_ticker.assert_not_called()


# ---------------------------------------------------------------------------
# resolve_tickers (batch)
# ---------------------------------------------------------------------------
class TestResolveTickers:
    @patch("src.ticker_map.yf.Ticker")
    @patch("src.ticker_map.cache_set")
    @patch("src.ticker_map.cache_get", return_value=None)
    def test_batch_mixed(self, mock_get, mock_set, mock_ticker):
        from src.ticker_map import resolve_tickers
        mock_ticker.return_value.info = {"exchange": "NMS"}
        result = resolve_tickers(["MSFT.NE", "AAPL", "VISA.NE"])
        assert result == {
            "MSFT.NE": "MSFT",
            "AAPL": "AAPL",
            "VISA.NE": "V",
        }

    def test_empty_list(self):
        from src.ticker_map import resolve_tickers
        assert resolve_tickers([]) == {}
