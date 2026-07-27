"""
Anomaly detection for portfolio holdings.

Flags abnormal volume / volatility using Isolation Forest or Gaussian
Mixture Model.  Designed to run against each holding during
``market-update`` and surface a Rich warning panel.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

ANOMALY_COLS = ["volume_zscore", "volatility_20d", "return_1d"]


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def detect_anomalies_isolation_forest(
    features: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Flag anomalies using Isolation Forest.

    Parameters
    ----------
    features : DataFrame with at least ``volume_zscore``, ``volatility_20d``,
        ``return_1d`` columns.
    contamination : expected fraction of anomalies (0–1).

    Returns
    -------
    Subset of *features* where anomaly_score == -1 (anomalous).
    """
    cols = [c for c in ANOMALY_COLS if c in features.columns]
    if not cols:
        return pd.DataFrame()

    df = features[cols].dropna()
    if len(df) < 10:
        return pd.DataFrame()

    iso = IsolationForest(contamination=contamination, random_state=42)
    df = df.copy()
    df["anomaly_score"] = iso.fit_predict(df[cols])
    return df[df["anomaly_score"] == -1]


def detect_anomalies_gmm(
    features: pd.DataFrame,
    threshold_percentile: float = 95,
) -> pd.DataFrame:
    """Flag anomalies using Gaussian Mixture Model.

    Uses the negative log-likelihood as an anomaly score; points above
    the *threshold_percentile* are flagged.

    Returns a subset of *features* with an ``anomaly_score`` column
    (higher = more anomalous).
    """
    cols = [c for c in ANOMALY_COLS if c in features.columns]
    if not cols:
        return pd.DataFrame()

    df = features[cols].dropna()
    if len(df) < 10:
        return pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    log_likelihood = gmm.score_samples(X)
    scores = -log_likelihood  # higher = more anomalous

    threshold = np.percentile(scores, threshold_percentile)
    df = df.copy()
    df["anomaly_score"] = scores
    return df[df["anomaly_score"] >= threshold]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_anomalies(
    features: pd.DataFrame,
    method: str = "isolation-forest",
    **kwargs: Any,
) -> pd.DataFrame:
    """Detect anomalies using the specified method.

    Parameters
    ----------
    features : DataFrame with anomaly detection features.
    method : ``"isolation-forest"`` or ``"gmm"``.
    **kwargs : passed to the underlying detector.

    Returns
    -------
    Subset of *features* flagged as anomalous.
    """
    if method == "isolation-forest":
        return detect_anomalies_isolation_forest(features, **kwargs)
    elif method == "gmm":
        return detect_anomalies_gmm(features, **kwargs)
    else:
        raise ValueError(f"Unknown anomaly method: {method!r}")


def summarize_anomalies(
    anomalies: pd.DataFrame, ticker: str
) -> str | None:
    """Return a human-readable summary string for a single ticker's anomalies.

    Returns None if no anomalies were found.
    """
    if anomalies.empty:
        return None

    n = len(anomalies)
    rows = []
    for date, row in anomalies.iterrows():
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        parts = []
        if "volume_zscore" in row and pd.notna(row["volume_zscore"]):
            parts.append(f"vol z={row['volume_zscore']:.1f}")
        if "volatility_20d" in row and pd.notna(row["volatility_20d"]):
            parts.append(f"vol20d={row['volatility_20d']:.2%}")
        if "return_1d" in row and pd.notna(row["return_1d"]):
            parts.append(f"ret={row['return_1d']:+.2%}")
        detail = ", ".join(parts)
        rows.append(f"  {date_str}: {detail}")

    header = f"Unusual activity in {ticker} ({n} day{'s' if n != 1 else ''}):"
    return header + "\n" + "\n".join(rows)
