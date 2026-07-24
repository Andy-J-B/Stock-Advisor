"""Tests for src/optimizer.py using synthetic price data."""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import optimize, discrete_allocation, SUPPORTED_OBJECTIVES


@pytest.fixture
def synthetic_prices_3t():
    """3-ticker, 252-day synthetic price DataFrame."""
    rng = np.random.RandomState(42)
    n = 252
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {
        "AAA": 100 * np.cumprod(1 + rng.normal(0.0008, 0.015, n)),
        "BBB": 100 * np.cumprod(1 + rng.normal(0.0005, 0.020, n)),
        "CCC": 100 * np.cumprod(1 + rng.normal(0.0003, 0.010, n)),
    }
    return pd.DataFrame(data, index=dates)


class TestOptimize:
    def test_max_sharpe_weights_sum_to_one(self, synthetic_prices_3t):
        result = optimize(synthetic_prices_3t, "max-sharpe")
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_min_volatility_weights_sum_to_one(self, synthetic_prices_3t):
        result = optimize(synthetic_prices_3t, "min-volatility")
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_result_has_required_keys(self, synthetic_prices_3t):
        result = optimize(synthetic_prices_3t, "max-sharpe")
        assert "weights" in result
        assert "expected_return" in result
        assert "volatility" in result
        assert "sharpe" in result

    def test_max_sharpe_beats_equal_weight(self, synthetic_prices_3t):
        opt = optimize(synthetic_prices_3t, "max-sharpe")
        # Equal-weight Sharpe
        eq_ret = synthetic_prices_3t.pct_change().dropna().mean(axis=1)
        eq_sharpe = eq_ret.mean() / eq_ret.std() * np.sqrt(252)
        assert opt["sharpe"] >= eq_sharpe - 0.01  # allow tiny numerical noise

    def test_unknown_objective_raises(self, synthetic_prices_3t):
        with pytest.raises(ValueError, match="Unknown objective"):
            optimize(synthetic_prices_3t, "bogus")

    def test_efficient_risk_requires_target(self, synthetic_prices_3t):
        with pytest.raises(ValueError, match="target_volatility"):
            optimize(synthetic_prices_3t, "efficient-risk")

    def test_too_few_tickers(self):
        prices = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 tickers"):
            optimize(prices)


class TestDiscreteAllocation:
    def test_allocations_are_integers(self, synthetic_prices_3t):
        result = optimize(synthetic_prices_3t, "max-sharpe")
        alloc = discrete_allocation(result["weights"], synthetic_prices_3t, 10000)
        for shares in alloc["allocations"].values():
            assert isinstance(shares, int) or isinstance(shares, (int, np.integer))

    def test_leftover_is_non_negative(self, synthetic_prices_3t):
        result = optimize(synthetic_prices_3t, "max-sharpe")
        alloc = discrete_allocation(result["weights"], synthetic_prices_3t, 10000)
        assert alloc["leftover"] >= 0
