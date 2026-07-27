"""Tests for src/features.py — no-lookahead assertion + basic sanity."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_features,
    build_target,
    _lagged_returns,
    _realized_volatility,
    _rsi,
    _macd_histogram,
    _bollinger_pctb,
    _atr,
    _ema_spread,
    _volume_zscore,
)


@pytest.fixture
def synthetic_ohlcv():
    """252-day synthetic OHLCV DataFrame with a clear uptrend."""
    rng = np.random.RandomState(42)
    n = 252
    # Uptrend + noise
    trend = np.linspace(100, 120, n)
    noise = rng.randn(n) * 2
    close = trend + noise
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opn = close + rng.randn(n) * 0.3
    vol = rng.randint(1000, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Lookahead-bias tests: feature at row t must NOT use close[t] or later
# ---------------------------------------------------------------------------


class TestNoLookahead:
    """Prove that each feature column uses only information from row t-1 or earlier."""

    def test_lagged_returns_never_use_same_bar(self, synthetic_ohlcv):
        close = synthetic_ohlcv["Close"]
        feats = _lagged_returns(close)
        # return_1d at row t = (close[t-1] - close[t-2]) / close[t-2]
        # Verify: for any row t, return_1d[t] equals the pct_change at t-1
        manual = close.pct_change(1).shift(1)
        pd.testing.assert_series_equal(feats["return_1d"], manual, check_names=False)

    def test_realized_vol_is_shifted(self, synthetic_ohlcv):
        close = synthetic_ohlcv["Close"]
        vol = _realized_volatility(close)
        # raw rolling std without shift
        raw = close.pct_change().rolling(20).std()
        # vol[t] == raw[t-1] for all t
        common = vol.dropna().index
        shifted_raw = raw.shift(1)
        pd.testing.assert_series_equal(
            vol.loc[common], shifted_raw.loc[common], check_names=False
        )

    def test_rsi_is_shifted(self, synthetic_ohlcv):
        close = synthetic_ohlcv["Close"]
        rsi = _rsi(close)
        # First valid RSI value should be NaN (shifted by 1)
        assert pd.isna(rsi.iloc[0])
        # RSI should be bounded [0, 100] for non-NaN values
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_hist_is_shifted(self, synthetic_ohlcv):
        close = synthetic_ohlcv["Close"]
        hist = _macd_histogram(close)
        # Row 0 should be NaN (shifted by 1); row 1 is valid
        assert pd.isna(hist.iloc[0])
        assert not pd.isna(hist.iloc[1])
        # All values should be bounded (MACD histogram is continuous)
        assert not np.isinf(hist.dropna()).any()

    def test_bb_pctb_is_shifted(self, synthetic_ohlcv):
        close = synthetic_ohlcv["Close"]
        pctb = _bollinger_pctb(close)
        # First 20 rows should be NaN (rolling window + shift)
        assert pd.isna(pctb.iloc[0])
        assert pd.isna(pctb.iloc[19])

    def test_atr_is_shifted(self, synthetic_ohlcv):
        row = synthetic_ohlcv.iloc[0]
        # ATR at row 0 should be NaN because it's shifted
        from src.features import _atr as atr_fn
        atr = atr_fn(synthetic_ohlcv["High"], synthetic_ohlcv["Low"], synthetic_ohlcv["Close"])
        assert pd.isna(atr.iloc[0])

    def test_ema_spread_is_shifted(self, synthetic_ohlcv):
        close = synthetic_ohlcv["Close"]
        spread = _ema_spread(close)
        assert pd.isna(spread.iloc[0])

    def test_volume_zscore_is_shifted(self, synthetic_ohlcv):
        vol = synthetic_ohlcv["Volume"]
        z = _volume_zscore(vol)
        # First 20 rows should be NaN (rolling window + shift)
        assert pd.isna(z.iloc[0])
        assert pd.isna(z.iloc[19])


# ---------------------------------------------------------------------------
# Full feature matrix
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_output_shape(self, synthetic_ohlcv):
        feat = build_features(synthetic_ohlcv)
        assert feat.shape[0] == synthetic_ohlcv.shape[0]
        assert feat.shape[1] > 10  # should have many features

    def test_no_infinite_values(self, synthetic_ohlcv):
        feat = build_features(synthetic_ohlcv)
        assert not np.isinf(feat.values).any()

    def test_feature_names(self, synthetic_ohlcv):
        feat = build_features(synthetic_ohlcv)
        expected_cols = {"return_1d", "return_5d", "return_20d", "volatility_20d",
                         "rsi_14", "macd_hist", "bb_pctb", "atr_14", "ema_spread",
                         "volume_zscore"}
        assert expected_cols.issubset(set(feat.columns))


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


class TestBuildTarget:
    def test_horizon_1(self, synthetic_ohlcv):
        target = build_target(synthetic_ohlcv["Close"], horizon=1)
        # Last row should be NaN
        assert pd.isna(target.iloc[-1])
        # All other values should be 0 or 1
        valid = target.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_horizon_5(self, synthetic_ohlcv):
        target = build_target(synthetic_ohlcv["Close"], horizon=5)
        # Last 5 rows should be NaN
        assert target.iloc[-5:].isna().all()
        valid = target.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})
