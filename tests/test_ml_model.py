"""Tests for src/ml_model.py — training, prediction, persistence."""

import time
import numpy as np
import pandas as pd
import pytest

from src.ml_model import train, predict, save_model, load_model, is_stale


@pytest.fixture
def synthetic_ohlcv():
    """252-day synthetic OHLCV with a momentum signal."""
    rng = np.random.RandomState(42)
    n = 252
    base_returns = rng.randn(n) * 0.01
    signal = np.zeros(n)
    for i in range(1, n):
        signal[i] = 0.3 * base_returns[i - 1]
    close = 100 * np.cumprod(1 + base_returns + signal)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    opn = close * (1 + rng.randn(n) * 0.002)
    vol = rng.randint(1000, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


@pytest.fixture
def training_data(synthetic_ohlcv):
    from src.features import build_features, build_target
    feat = build_features(synthetic_ohlcv)
    target = build_target(synthetic_ohlcv["Close"], horizon=1)
    # Drop always-NaN columns (no sentiment/macro in synthetic data)
    price_cols = [
        "return_1d", "return_5d", "return_20d", "volatility_20d",
        "rsi_14", "macd_hist", "bb_pctb", "atr_14", "ema_spread",
        "volume_zscore",
    ]
    feat_price = feat[price_cols]
    mask = target.notna() & feat_price.notna().all(axis=1)
    return feat_price.loc[mask], target.loc[mask]


class TestTrain:
    def test_returns_model(self, training_data):
        X, y = training_data
        result = train(X, y)
        assert "model" in result
        assert "cv_accuracy" in result
        assert "fold_accuracies" in result

    def test_cv_accuracy_above_random(self, training_data):
        X, y = training_data
        result = train(X, y)
        assert result["cv_accuracy"] > 0.45

    def test_walk_forward_indices_chronological(self, training_data):
        from sklearn.model_selection import TimeSeriesSplit
        X, y = training_data
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in tscv.split(X.values):
            assert train_idx[-1] < test_idx[0], "Train must be before test"

    def test_too_few_rows_raises(self):
        X = pd.DataFrame({"a": range(5)})
        y = pd.Series([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="Not enough data"):
            train(X, y, n_splits=5)


class TestPredict:
    def test_returns_expected_keys(self, training_data):
        X, y = training_data
        result = train(X, y)
        pred = predict(result["model"], X.iloc[[-1]])
        assert "probability_up" in pred
        assert "label" in pred
        assert "confidence" in pred
        assert pred["label"] in ("up", "down")

    def test_probability_range(self, training_data):
        X, y = training_data
        result = train(X, y)
        pred = predict(result["model"], X.iloc[[-1]])
        assert 0.0 <= pred["probability_up"] <= 1.0
        assert 0.0 <= pred["confidence"] <= 1.0


class TestPersistence:
    def test_save_and_load(self, training_data, tmp_path):
        import src.ml_model as mod
        original_dir = mod.MODEL_DIR
        mod.MODEL_DIR = tmp_path
        try:
            X, y = training_data
            result = train(X, y)
            path = save_model(result["model"], "TEST", 1, metadata={"cv_accuracy": 0.55})
            assert path.exists()
            loaded = load_model("TEST", 1)
            assert loaded is not None
            assert loaded["ticker"] == "TEST"
            assert loaded["horizon"] == 1
        finally:
            mod.MODEL_DIR = original_dir

    def test_load_nonexistent(self, tmp_path):
        import src.ml_model as mod
        original_dir = mod.MODEL_DIR
        mod.MODEL_DIR = tmp_path
        try:
            assert load_model("NONEXISTENT", 1) is None
            assert is_stale("NONEXISTENT", 1) is True
        finally:
            mod.MODEL_DIR = original_dir

    def test_staleness(self, training_data, tmp_path):
        import joblib
        import src.ml_model as mod
        original_dir = mod.MODEL_DIR
        mod.MODEL_DIR = tmp_path
        try:
            X, y = training_data
            result = train(X, y)
            save_model(result["model"], "STALE", 1)
            assert not is_stale("STALE", 1, max_age_days=7)

            payload = load_model("STALE", 1)
            payload["trained_at"] = time.time() - 86400 * 10
            joblib.dump(payload, mod.MODEL_DIR / "STALE_1.pkl")
            assert is_stale("STALE", 1, max_age_days=7)
        finally:
            mod.MODEL_DIR = original_dir
