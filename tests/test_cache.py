"""Tests for the SQLite-backed cache and batch-fetch helpers."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
from peewee import SqliteDatabase

# ---------------------------------------------------------------------------
# Setup: in-memory DB before any model imports touch the real DB
# ---------------------------------------------------------------------------
_test_db = SqliteDatabase(":memory:")

import src.database as _db_mod

_db_mod.db = _test_db
_db_mod.CacheEntry._meta.database = _test_db
_db_mod.Account._meta.database = _test_db
_db_mod.Holding._meta.database = _test_db
_db_mod.Transaction._meta.database = _test_db
_db_mod.NetWorthSnapshot._meta.database = _test_db
_db_mod.Setting._meta.database = _test_db

_test_db.connect()
_test_db.create_tables([
    _db_mod.Account, _db_mod.Holding, _db_mod.Transaction,
    _db_mod.NetWorthSnapshot, _db_mod.Setting, _db_mod.CacheEntry,
])

from src.database import cache_get, cache_set, CacheEntry


# ---------------------------------------------------------------------------
# cache_get / cache_set
# ---------------------------------------------------------------------------

class TestCacheHelpers:
    def test_set_and_get(self):
        cache_set("test:hello", {"msg": "world"})
        result = cache_get("test:hello", ttl_seconds=60)
        assert result == {"msg": "world"}

    def test_cache_miss(self):
        result = cache_get("test:nonexistent_key", ttl_seconds=60)
        assert result is None

    def test_ttl_expiry(self):
        cache_set("test:expire_me", [1, 2, 3])
        # Backdate the fetched_at to 2 hours ago
        CacheEntry.update(
            fetched_at=datetime.now() - timedelta(hours=2)
        ).where(CacheEntry.key == "test:expire_me").execute()

        result = cache_get("test:expire_me", ttl_seconds=3600)
        assert result is None

    def test_upsert(self):
        cache_set("test:upsert", "first")
        cache_set("test:upsert", "second")
        result = cache_get("test:upsert", ttl_seconds=60)
        assert result == "second"
        # Only one row should exist
        count = CacheEntry.select().where(CacheEntry.key == "test:upsert").count()
        assert count == 1

    def test_stores_various_types(self):
        cache_set("test:str", "hello")
        cache_set("test:int", 42)
        cache_set("test:float", 3.14)
        cache_set("test:dict", {"a": 1, "b": [2, 3]})
        cache_set("test:list", [1.5, 2.5, 3.5])

        assert cache_get("test:str", 60) == "hello"
        assert cache_get("test:int", 60) == 42
        assert cache_get("test:float", 60) == 3.14
        assert cache_get("test:dict", 60) == {"a": 1, "b": [2, 3]}
        assert cache_get("test:list", 60) == [1.5, 2.5, 3.5]

    def test_corrupt_json_returns_none(self):
        CacheEntry.create(key="test:corrupt", value="NOT JSON {{{", fetched_at=datetime.now())
        result = cache_get("test:corrupt", ttl_seconds=60)
        assert result is None


# ---------------------------------------------------------------------------
# data_client cache integration (mocked yfinance)
# ---------------------------------------------------------------------------

class TestDataClientCache:
    """Verify data_client functions check cache before hitting the network."""

    def test_get_current_price_cache_hit(self):
        """If cache has a fresh price, yfinance should NOT be called."""
        from src.data_client import get_current_price

        cache_set("price:AAPL", [150.0, 148.0])
        with patch("src.data_client.yf") as mock_yf:
            result = get_current_price("AAPL")
            mock_yf.Ticker.assert_not_called()
        assert result == (150.0, 148.0)

    def test_get_current_price_cache_miss(self):
        """On cache miss, yfinance is called and result is cached."""
        from src.data_client import get_current_price

        # Clear any existing cache entry
        CacheEntry.delete().where(CacheEntry.key == "price:MSFT").execute()

        mock_stock = MagicMock()
        mock_stock.fast_info = {"lastPrice": 420.0, "previousClose": 415.0}
        with patch("src.data_client.yf") as mock_yf:
            mock_yf.Ticker.return_value = mock_stock
            result = get_current_price("MSFT")

        assert result == (420.0, 415.0)
        # Verify it was cached
        cached = cache_get("price:MSFT", ttl_seconds=300)
        assert cached == [420.0, 415.0]

    def test_get_ticker_news_cache_hit(self):
        from src.data_client import get_ticker_news

        fake_news = [{"title": "Test", "publisher": "TestPub", "link": ""}]
        cache_set("news:TSLA:3", fake_news)
        with patch("src.data_client.yf") as mock_yf:
            result = get_ticker_news("TSLA")
            mock_yf.Ticker.assert_not_called()
        assert result == fake_news

    def test_get_usd_to_cad_cache_hit(self):
        from src.data_client import get_usd_to_cad

        cache_set("fx:usd_cad", 1.3650)
        with patch("src.data_client._http") as mock_http:
            result = get_usd_to_cad()
            mock_http.get.assert_not_called()
        assert result == 1.3650

    def test_get_close_prices_filters_sparse_tickers(self):
        """Sparse tickers (fewer than min_rows) must not collapse the shared index."""
        from src.data_client import get_close_prices

        long_dates = pd.date_range("2025-01-01", periods=100, freq="B")
        short_dates = pd.date_range("2025-01-01", periods=3, freq="B")

        def fake_history(ticker, period="1y"):
            if ticker == "SPARSE":
                return pd.DataFrame(
                    {"Close": [10, 11, 12]},
                    index=pd.DatetimeIndex(short_dates),
                )
            return pd.DataFrame(
                {"Close": range(100)},
                index=pd.DatetimeIndex(long_dates),
            )

        with patch("src.data_client.get_price_history", side_effect=fake_history):
            result = get_close_prices(["GOOD1", "GOOD2", "SPARSE"], period="1y", min_rows=30)

        assert "SPARSE" not in result.columns
        assert list(result.columns) == ["GOOD1", "GOOD2"]
        assert result.shape[0] == 100

    def test_get_close_prices_default_no_filter(self):
        """Without min_rows, sparse tickers are kept (backward compatible)."""
        from src.data_client import get_close_prices

        long_dates = pd.date_range("2025-01-01", periods=50, freq="B")
        short_dates = pd.date_range("2025-01-01", periods=3, freq="B")

        def fake_history(ticker, period="1y"):
            dates = long_dates if ticker == "FULL" else short_dates
            return pd.DataFrame(
                {"Close": range(len(dates))},
                index=pd.DatetimeIndex(dates),
            )

        with patch("src.data_client.get_price_history", side_effect=fake_history):
            result = get_close_prices(["FULL", "SPARSE"], period="1y")

        assert "SPARSE" in result.columns
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# Batch fetching
# ---------------------------------------------------------------------------

class _SequentialPool:
    """Drop-in ThreadPoolExecutor replacement that runs tasks in-process (no threads)."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, fn, iterable):
        return [fn(x) for x in iterable]


class TestBatchFetching:
    def test_get_current_prices_batch(self):
        from src.data_client import get_current_prices_batch

        cache_set("price:A", [100.0, 99.0])
        cache_set("price:B", [200.0, 198.0])

        with patch("src.data_client.ThreadPoolExecutor", _SequentialPool):
            result = get_current_prices_batch(["A", "B"])
        assert result["A"] == (100.0, 99.0)
        assert result["B"] == (200.0, 198.0)

    def test_get_current_prices_batch_empty(self):
        from src.data_client import get_current_prices_batch
        assert get_current_prices_batch([]) == {}

    def test_get_ticker_news_batch(self):
        from src.data_client import get_ticker_news_batch

        cache_set("news:X:3", [{"title": "X news", "publisher": "X", "link": ""}])
        cache_set("news:Y:3", [{"title": "Y news", "publisher": "Y", "link": ""}])

        with patch("src.data_client.ThreadPoolExecutor", _SequentialPool):
            result = get_ticker_news_batch(["X", "Y"])
        assert len(result["X"]) == 1
        assert result["X"][0]["title"] == "X news"
        assert len(result["Y"]) == 1
        assert result["Y"][0]["title"] == "Y news"

    def test_batch_preserves_order(self):
        from src.data_client import get_current_prices_batch

        for i, t in enumerate(["Z1", "Z2", "Z3"]):
            cache_set(f"price:{t}", [float(i), float(i - 1)])

        with patch("src.data_client.ThreadPoolExecutor", _SequentialPool):
            result = get_current_prices_batch(["Z1", "Z2", "Z3"])
        keys = list(result.keys())
        assert keys == ["Z1", "Z2", "Z3"]


# ---------------------------------------------------------------------------
# DataProvider protocol
# ---------------------------------------------------------------------------

class TestDataProvider:
    def test_yfinance_adapter_satisfies_protocol(self):
        from src.providers import YFinanceAdapter, DataProvider
        adapter = YFinanceAdapter()
        assert isinstance(adapter, DataProvider)

    def test_alpha_vantage_adapter_satisfies_protocol(self):
        from src.providers import AlphaVantageAdapter, DataProvider
        adapter = AlphaVantageAdapter()
        assert isinstance(adapter, DataProvider)
