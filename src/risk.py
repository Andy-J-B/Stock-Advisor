"""
Quantitative risk metrics for portfolio analysis.

Uses historical simulation for VaR/CVaR -- the simplest defensible method
for a CLI tool.  Tradeoff noted in docstrings: historical method assumes
the past distribution repeats; parametric assumes normality which
underestimates tail risk; Monte Carlo is heavier and overkill here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk via historical simulation (annualized).

    Returns the loss threshold that *confidence* fraction of daily
    returns did not exceed, expressed as a signed daily return.
    """
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) -- average loss beyond VaR."""
    if returns.empty:
        return 0.0
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if not tail.empty else var


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio.  ``rf`` is daily risk-free rate."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - rf
    return float(excess.mean() / excess.std() * np.sqrt(252))


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sortino ratio -- penalizes only downside deviation."""
    if returns.empty:
        return 0.0
    excess = returns - rf
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown from a price series.  Returns a negative fraction."""
    if prices.empty or len(prices) < 2:
        return 0.0
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())
