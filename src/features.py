"""
Feature engineering pipeline for the ML alpha engine.

All price-derived features are shifted by 1 bar to avoid lookahead bias.
Sentiment features use current-day aggregates (news is exogenous, not from
the price series).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Price / technical features (all lagged by 1)
# ---------------------------------------------------------------------------


def _lagged_returns(close: pd.Series) -> pd.DataFrame:
    """1-day, 5-day, and 20-day lagged returns."""
    return pd.DataFrame(
        {
            "return_1d": close.pct_change(1).shift(1),
            "return_5d": close.pct_change(5).shift(1),
            "return_20d": close.pct_change(20).shift(1),
        },
        index=close.index,
    )


def _realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of daily returns, lagged by 1."""
    rets = close.pct_change()
    vol = rets.rolling(window).std()
    return vol.shift(1).rename("volatility_20d")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI lagged by 1 bar."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.shift(1).rename(f"rsi_{period}")


def _macd_histogram(close: pd.Series) -> pd.Series:
    """MACD histogram (MACD − signal) lagged by 1 bar."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return hist.shift(1).rename("macd_hist")


def _bollinger_pctb(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Bollinger %B lagged by 1 bar."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return pctb.shift(1).rename("bb_pctb")


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range lagged by 1 bar."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.shift(1).rename(f"atr_{period}")


def _ema_spread(close: pd.Series) -> pd.Series:
    """EMA 20/50 spread (as % of close) lagged by 1 bar."""
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    spread = (ema20 - ema50) / close
    return spread.shift(1).rename("ema_spread")


def _volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of volume, lagged by 1 bar (used by anomaly detector)."""
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    z = (volume - mu) / sigma.replace(0, np.nan)
    return z.shift(1).rename("volume_zscore")


# ---------------------------------------------------------------------------
# Macro features (optional)
# ---------------------------------------------------------------------------


def fetch_macro_features() -> pd.DataFrame:
    """Fetch VIX and 10Y yield levels from yfinance (cached via data_client)."""
    import yfinance as yf

    frames: dict[str, pd.Series] = {}
    for ticker, name in [("^VIX", "vix"), ("^TNX", "tnx_10y")]:
        try:
            s = yf.Ticker(ticker).history(period="1y")["Close"]
            if s is not None and not s.empty:
                frames[name] = s
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1, keys=frames.keys())
    # Lag by 1 to avoid lookahead
    return df.shift(1)


# ---------------------------------------------------------------------------
# Sentiment features
# ---------------------------------------------------------------------------


def aggregate_sentiment(
    sentiment_scores: list[dict],
) -> pd.DataFrame:
    """Convert a list of per-headline sentiment dicts to daily aggregates.

    Each dict must have ``compound`` and ``label`` keys.
    Returns a DataFrame with columns: sentiment_compound, sentiment_pos_pct,
    sentiment_neg_pct, sentiment_neu_pct.
    """
    if not sentiment_scores:
        return pd.DataFrame()

    compounds = [s["compound"] for s in sentiment_scores]
    n = len(compounds)
    pos_pct = sum(1 for s in sentiment_scores if s["label"] == "positive") / n
    neg_pct = sum(1 for s in sentiment_scores if s["label"] == "negative") / n
    neu_pct = sum(1 for s in sentiment_scores if s["label"] == "neutral") / n

    return pd.DataFrame(
        {
            "sentiment_compound": [np.mean(compounds)],
            "sentiment_pos_pct": [pos_pct],
            "sentiment_neg_pct": [neg_pct],
            "sentiment_neu_pct": [neu_pct],
        }
    )


def build_features(
    ohlcv: pd.DataFrame,
    sentiment_history: pd.DataFrame | None = None,
    include_macro: bool = False,
) -> pd.DataFrame:
    """Build the full feature matrix from OHLCV data.

    Parameters
    ----------
    ohlcv : DataFrame with Open, High, Low, Close, Volume columns.
    sentiment_history : optional DataFrame indexed by date with
        ``sentiment_compound`` column (daily aggregate).
    include_macro : if True, fetch VIX / 10Y yield.

    Returns
    -------
    DataFrame of features, aligned to the same date index.
    All price-derived features are shifted by 1 bar (no lookahead).
    """
    close = ohlcv["Close"]
    feats = pd.DataFrame(index=ohlcv.index)

    # Price / technical features (all shifted)
    feats = pd.concat(
        [
            feats,
            _lagged_returns(close),
            _realized_volatility(close),
            _rsi(close),
            _macd_histogram(close),
            _bollinger_pctb(close),
            _atr(ohlcv["High"], ohlcv["Low"], close),
            _ema_spread(close),
            _volume_zscore(ohlcv["Volume"]),
        ],
        axis=1,
    )

    # Sentiment rolling averages (if available)
    if sentiment_history is not None and not sentiment_history.empty:
        common = feats.index.intersection(sentiment_history.index)
        if len(common) > 0:
            sh = sentiment_history.loc[common]
            feats.loc[common, "sentiment_1d"] = sh["sentiment_compound"]
            feats.loc[common, "sentiment_5d"] = (
                sh["sentiment_compound"].rolling(5, min_periods=1).mean()
            )
        else:
            feats["sentiment_1d"] = np.nan
            feats["sentiment_5d"] = np.nan
    else:
        feats["sentiment_1d"] = np.nan
        feats["sentiment_5d"] = np.nan

    # Optional macro features
    if include_macro:
        macro = fetch_macro_features()
        if not macro.empty:
            common = feats.index.intersection(macro.index)
            for col in macro.columns:
                feats.loc[common, col] = macro.loc[common, col]
        else:
            feats["vix"] = np.nan
            feats["tnx_10y"] = np.nan
    else:
        feats["vix"] = np.nan
        feats["tnx_10y"] = np.nan

    return feats


def build_target(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary target: 1 if close[t+horizon] > close[t], else 0.

    The last ``horizon`` rows will be NaN (no future data available).
    """
    future_close = close.shift(-horizon)
    target = (future_close > close).astype(float)
    target.iloc[-horizon:] = np.nan
    return target.rename("target")
