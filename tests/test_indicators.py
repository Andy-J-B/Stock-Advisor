"""Tests for src/indicators.py using synthetic OHLCV data."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import compute_indicators, interpret_signals


@pytest.fixture
def synthetic_ohlcv():
    """100-bar synthetic OHLCV DataFrame."""
    rng = np.random.RandomState(42)
    n = 100
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opn = close + rng.randn(n) * 0.3
    vol = rng.randint(1000, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


class TestComputeIndicators:
    def test_adds_columns(self, synthetic_ohlcv):
        result = compute_indicators(synthetic_ohlcv)
        assert "RSI_14" in result.columns
        assert "MACD_12_26_9" in result.columns
        assert "BBU_20_2.0_2.0" in result.columns
        assert "BBL_20_2.0_2.0" in result.columns
        assert "ATRr_14" in result.columns
        assert "EMA_20" in result.columns
        assert "EMA_50" in result.columns

    def test_rsi_bounded(self, synthetic_ohlcv):
        result = compute_indicators(synthetic_ohlcv)
        rsi = result["RSI_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_does_not_lose_original_columns(self, synthetic_ohlcv):
        original_cols = set(synthetic_ohlcv.columns)
        result = compute_indicators(synthetic_ohlcv)
        assert original_cols.issubset(set(result.columns))


class TestInterpretSignals:
    def test_returns_list(self, synthetic_ohlcv):
        compute_indicators(synthetic_ohlcv)
        signals = interpret_signals(synthetic_ohlcv)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_each_entry_has_required_keys(self, synthetic_ohlcv):
        compute_indicators(synthetic_ohlcv)
        signals = interpret_signals(synthetic_ohlcv)
        for s in signals:
            assert "name" in s
            assert "value" in s
            assert "signal" in s

    def test_empty_dataframe(self):
        assert interpret_signals(pd.DataFrame()) == []
