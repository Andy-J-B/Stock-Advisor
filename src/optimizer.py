"""
Portfolio optimizer backed by PyPortfolioOpt.

Supports three objectives: max-sharpe (default), min-volatility,
and efficient-risk (target volatility).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import DiscreteAllocation, get_latest_prices

log = logging.getLogger(__name__)

SUPPORTED_OBJECTIVES = {"max-sharpe", "min-volatility", "efficient-risk"}


def optimize(
    prices: pd.DataFrame,
    objective: str = "max-sharpe",
    target_volatility: Optional[float] = None,
) -> dict[str, Any]:
    """Run mean-variance optimization and return results dict.

    Parameters
    ----------
    prices : DataFrame with tickers as columns and dates as index (adj close).
    objective : one of ``max-sharpe``, ``min-volatility``, ``efficient-risk``.
    target_volatility : required when objective is ``efficient-risk``.

    Returns dict with keys: weights, expected_return, volatility, sharpe.
    """
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 tickers to optimize.")

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, S)

    if objective == "max-sharpe":
        ef.max_sharpe()
    elif objective == "min-volatility":
        ef.min_volatility()
    elif objective == "efficient-risk":
        if target_volatility is None:
            raise ValueError("target_volatility required for efficient-risk.")
        ef.efficient_risk(target_volatility=target_volatility)
    else:
        raise ValueError(f"Unknown objective: {objective!r}. Choose from {SUPPORTED_OBJECTIVES}")

    weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance()

    return {
        "weights": dict(weights),
        "expected_return": float(ret),
        "volatility": float(vol),
        "sharpe": float(sharpe),
    }


def discrete_allocation(
    weights: dict[str, float],
    prices: pd.DataFrame,
    total_value: float,
) -> dict[str, Any]:
    """Given continuous weights and a total portfolio value, compute share counts.

    Returns dict with ``allocations`` (ticker -> shares) and ``leftover`` cash.
    """
    latest = get_latest_prices(prices)
    da = DiscreteAllocation(weights, latest, total_portfolio_value=total_value)
    alloc, leftover = da.greedy_portfolio()
    return {"allocations": dict(alloc), "leftover": float(leftover)}
