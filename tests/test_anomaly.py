"""Tests for src/anomaly.py — Isolation Forest + GMM."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly import detect_anomalies, summarize_anomalies


@pytest.fixture
def synthetic_ohlcv():
    """252-day synthetic OHLCV with a volume spike injected at row 100."""
    rng = np.random.RandomState(42)
    n = 252
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opn = close + rng.randn(n) * 0.3
    vol = rng.uniform(1000, 10000, n)
    vol[100] = 50000
    close[100] = close[99] * 1.15
    high[100] = close[100] * 1.05
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


@pytest.fixture
def features_df(synthetic_ohlcv):
    from src.features import build_features
    return build_features(synthetic_ohlcv)


class TestIsolationForest:
    def test_returns_dataframe(self, features_df):
        result = detect_anomalies(features_df, method="isolation-forest")
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_has_anomaly_score_column(self, features_df):
        result = detect_anomalies(features_df, method="isolation-forest")
        assert "anomaly_score" in result.columns

    def test_some_rows_flagged(self, features_df):
        result = detect_anomalies(features_df, method="isolation-forest")
        n_flagged = len(result)
        assert n_flagged >= 1


class TestDetectAnomalies:
    def test_volume_spike_flagged(self, synthetic_ohlcv):
        from src.features import build_features
        feat = build_features(synthetic_ohlcv)
        flagged = detect_anomalies(feat, method="isolation-forest", contamination=0.05)
        # Row 100 (the spike) should be near the anomaly region.
        # Use integer position since index is datetime.
        flagged_idx = flagged.index
        dates = feat.index
        spike_date = dates[100]
        near_spike = [d for d in flagged_idx if abs((d - spike_date).days) <= 5]
        assert len(near_spike) >= 1, f"No anomaly near spike. Flagged dates: {flagged_idx.tolist()}"

    def test_empty_features_returns_empty(self):
        empty = pd.DataFrame()
        result = detect_anomalies(empty)
        assert result.empty

    def test_too_few_rows_returns_empty(self):
        few = pd.DataFrame({"return_1d": [0.01, -0.02, 0.03]})
        result = detect_anomalies(few)
        assert result.empty


class TestSummarize:
    def test_returns_string(self, features_df):
        flagged = detect_anomalies(features_df)
        summary = summarize_anomalies(flagged, "AAPL")
        assert isinstance(summary, str)
        assert "AAPL" in summary

    def test_empty_returns_none(self):
        empty = pd.DataFrame()
        assert summarize_anomalies(empty, "AAPL") is None
