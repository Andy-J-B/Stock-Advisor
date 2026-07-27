"""
LightGBM directional predictor.

Trains a classifier to forecast next-day/week up/down movement using
walk-forward (TimeSeriesSplit) validation.  Models are persisted to
``data/models/{ticker}_{horizon}.pkl`` with a training timestamp.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train a LightGBM classifier with walk-forward cross-validation.

    Returns a dict with keys: ``model``, ``cv_accuracy``, ``report``.
    """
    # Drop rows where target or any feature is NaN
    mask = target.notna() & features.notna().all(axis=1)
    X = features.loc[mask].values
    y = target.loc[mask].values.astype(int)
    feature_names = list(features.columns)

    if len(X) < n_splits + 1:
        raise ValueError(
            f"Not enough data for {n_splits}-fold TimeSeriesSplit: {len(X)} rows."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies: list[float] = []

    last_model = None
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbose=-1,
            random_state=42,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        fold_accuracies.append(acc)
        log.info("Fold %d: accuracy=%.3f", fold_idx + 1, acc)
        last_model = model

    # Retrain on full data for final model
    final_model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        verbose=-1,
        random_state=42,
    )
    final_model.fit(X, y)

    mean_acc = float(np.mean(fold_accuracies))
    report = classification_report(
        y, final_model.predict(X), target_names=["down", "up"], output_dict=True
    )

    return {
        "model": final_model,
        "feature_names": feature_names,
        "cv_accuracy": mean_acc,
        "fold_accuracies": fold_accuracies,
        "report": report,
        "n_train": len(X),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    model: lgb.LGBMClassifier,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Run a single-row prediction and return probability + label.

    Returns dict with keys: ``probability_up``, ``label``, ``confidence``.
    """
    X = features.values.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    prob_up = float(proba[1])
    label = "up" if prob_up >= 0.5 else "down"
    confidence = abs(prob_up - 0.5) * 2  # 0 = coin flip, 1 = certain

    return {
        "probability_up": prob_up,
        "label": label,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    model: lgb.LGBMClassifier,
    ticker: str,
    horizon: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist model + metadata to disk. Returns the file path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{ticker}_{horizon}.pkl"

    payload = {
        "model": model,
        "ticker": ticker,
        "horizon": horizon,
        "trained_at": time.time(),
        "metadata": metadata or {},
    }
    joblib.dump(payload, path)
    log.info("Model saved to %s", path)
    return path


def load_model(ticker: str, horizon: int) -> dict[str, Any] | None:
    """Load a persisted model. Returns None if not found."""
    path = MODEL_DIR / f"{ticker}_{horizon}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def is_stale(ticker: str, horizon: int, max_age_days: int = 7) -> bool:
    """Check if a saved model is older than *max_age_days* or missing."""
    payload = load_model(ticker, horizon)
    if payload is None:
        return True
    age_days = (time.time() - payload["trained_at"]) / 86400
    return age_days > max_age_days


def model_path(ticker: str, horizon: int) -> Path:
    """Return the expected path for a saved model."""
    return MODEL_DIR / f"{ticker}_{horizon}.pkl"
