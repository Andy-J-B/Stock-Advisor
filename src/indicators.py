"""
Technical indicator pipeline using pandas-ta.

Computes a standard set of indicators in one call and returns a
human-readable summary dict for each indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_ta  # noqa: F401 – registers .ta accessor


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Run core indicators on an OHLCV DataFrame.

    Expects columns: Open, High, Low, Close, Volume.
    Returns the same DataFrame with indicator columns appended.
    """
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    return df


def interpret_signals(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract the latest row of each indicator and return a summary list.

    Each entry: ``{"name": ..., "value": ..., "signal": ...}``
    """
    if df.empty:
        return []

    last = df.iloc[-1]
    signals: list[dict[str, Any]] = []

    def _add(name: str, col: str, fmt: str = ".2f", signal_fn=None):
        if col in df.columns and pd.notna(last[col]):
            val = last[col]
            sig = signal_fn(val) if signal_fn else ""
            signals.append({"name": name, "value": f"{val:{fmt}}", "signal": sig})

    def _rsi_signal(v: float) -> str:
        if v >= 70:
            return "Overbought"
        if v <= 30:
            return "Oversold"
        return ""

    def _macd_signal(_: float) -> str:
        macd_col = "MACD_12_26_9"
        sig_col = "MACDs_12_26_9"
        if macd_col in df.columns and sig_col in df.columns:
            if pd.notna(last[macd_col]) and pd.notna(last[sig_col]):
                prev = df.iloc[-2] if len(df) > 1 else last
                if last[macd_col] > last[sig_col] and prev[macd_col] <= prev[sig_col]:
                    return "Bullish crossover"
                if last[macd_col] < last[sig_col] and prev[macd_col] >= prev[sig_col]:
                    return "Bearish crossover"
        return ""

    def _ema_signal(_: float) -> str:
        ema20 = "EMA_20"
        ema50 = "EMA_50"
        if ema20 in df.columns and ema50 in df.columns:
            if pd.notna(last.get(ema20)) and pd.notna(last.get(ema50)):
                return "Uptrend" if last[ema20] > last[ema50] else "Downtrend"
        return ""

    _add("RSI (14)", "RSI_14", signal_fn=_rsi_signal)
    _add("MACD", "MACD_12_26_9", signal_fn=_macd_signal)
    _add("MACD Signal", "MACDs_12_26_9")
    _add("BB Upper", "BBU_20_2.0_2.0")
    _add("BB Lower", "BBL_20_2.0_2.0")
    _add("ATR (14)", "ATRr_14")
    _add("EMA 20", "EMA_20")
    _add("EMA 50", "EMA_50")
    _add("EMA Cross", "EMA_20", signal_fn=_ema_signal)

    return signals
