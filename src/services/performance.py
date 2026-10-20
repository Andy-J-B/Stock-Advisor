from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def compute_returns(values: list[float]) -> dict[str, float]:
    if not values or len(values) < 2:
        return {"total_return": 0.0, "daily_returns": []}
    daily = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        cur = values[i]
        if prev == 0:
            daily.append(0.0)
        else:
            daily.append((cur - prev) / prev)
    total = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0.0
    return {"total_return": total, "daily_returns": daily}


def sharpe_ratio(daily_returns: list[float], risk_free_rate: float = 0.04) -> float:
    if not daily_returns:
        return 0.0
    try:
        import numpy as np
        arr = np.array(daily_returns)
        excess = arr - (risk_free_rate / 252.0)
        if arr.std(ddof=1) == 0:
            return 0.0
        return float((excess.mean() / arr.std(ddof=1)) * np.sqrt(252))
    except Exception:
        import statistics
        if len(daily_returns) < 2:
            return 0.0
        mean = statistics.mean(daily_returns) - (risk_free_rate / 252.0)
        try:
            stdev = statistics.pstdev(daily_returns)
        except Exception:
            return 0.0
        if stdev == 0:
            return 0.0
        return (mean / stdev) * 15.87  # sqrt(252) approx


def sortino_ratio(daily_returns: list[float], risk_free_rate: float = 0.04) -> float:
    if not daily_returns:
        return 0.0
    try:
        import numpy as np
        arr = np.array(daily_returns)
        excess = arr - (risk_free_rate / 252.0)
        downside = arr[arr < 0]
        if len(downside) == 0 or downside.std(ddof=1) == 0:
            return 0.0
        return float((excess.mean() / downside.std(ddof=1)) * np.sqrt(252))
    except Exception:
        return sharpe_ratio(daily_returns, risk_free_rate)


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak == 0:
            continue
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return -max_dd


def time_weighted_return(equity_curve: list[tuple[date, float]]) -> float:
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    t0, v0 = equity_curve[0]
    t1, v1 = equity_curve[-1]
    if v0 == 0:
        return 0.0
    return (v1 - v0) / v0


def money_weighted_return_simple(cash_flows: list[tuple[date, float]], values: list[tuple[date, float]]) -> float:
    # Simplified approximation - proper IRR would need solver
    if not cash_flows or not values:
        return 0.0
    total_invested = sum(abs(cf[1]) for cf in cash_flows if cf[1] < 0)
    if total_invested == 0 or not values:
        return 0.0
    final = values[-1][1]
    return (final - total_invested) / total_invested
