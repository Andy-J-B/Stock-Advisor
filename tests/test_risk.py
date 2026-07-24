"""Tests for src/risk.py using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.risk import historical_var, cvar, sharpe_ratio, sortino_ratio, max_drawdown


@pytest.fixture
def synthetic_returns():
    """A reproducible daily-return series with known properties."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0.0005, 0.015, 252), name="returns")


@pytest.fixture
def synthetic_prices():
    """Synthetic price series (geometric random walk)."""
    rng = np.random.RandomState(42)
    returns = rng.normal(0.0005, 0.015, 252)
    return pd.Series(100 * np.cumprod(1 + returns), name="price")


class TestHistoricalVaR:
    def test_returns_negative_for_normal_dist(self, synthetic_returns):
        var = historical_var(synthetic_returns, 0.95)
        assert var < 0

    def test_worse_confidence_gives_lower_var(self, synthetic_returns):
        var_95 = historical_var(synthetic_returns, 0.95)
        var_99 = historical_var(synthetic_returns, 0.99)
        assert var_99 <= var_95

    def test_empty_series(self):
        assert historical_var(pd.Series(dtype=float)) == 0.0


class TestCVaR:
    def test_cvar_worse_than_var(self, synthetic_returns):
        var = historical_var(synthetic_returns, 0.95)
        cv = cvar(synthetic_returns, 0.95)
        assert cv <= var

    def test_empty_series(self):
        assert cvar(pd.Series(dtype=float)) == 0.0


class TestSharpe:
    def test_positive_for_positive_drift(self):
        rng = np.random.RandomState(0)
        pos = pd.Series(rng.normal(0.002, 0.01, 252))
        assert sharpe_ratio(pos) > 0

    def test_zero_for_empty(self):
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0


class TestSortino:
    def test_positive_for_positive_drift(self):
        rng = np.random.RandomState(0)
        pos = pd.Series(rng.normal(0.002, 0.01, 252))
        assert sortino_ratio(pos) > 0

    def test_zero_for_empty(self):
        assert sortino_ratio(pd.Series(dtype=float)) == 0.0


class TestMaxDrawdown:
    def test_negative_value(self, synthetic_prices):
        dd = max_drawdown(synthetic_prices)
        assert dd <= 0

    def test_no_drawdown_if_monotonic(self):
        prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
        assert max_drawdown(prices) == 0.0

    def test_empty(self):
        assert max_drawdown(pd.Series(dtype=float)) == 0.0

    def test_known_drawdown(self):
        prices = pd.Series([100, 120, 90, 110], dtype=float)
        dd = max_drawdown(prices)
        assert abs(dd - (-0.25)) < 1e-9  # 100 -> 90 is -25% from 120 peak? No.
        # Peak is 120, trough is 90. DD = (90-120)/120 = -0.25
