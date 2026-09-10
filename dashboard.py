"""
Streamlit dashboard for browsing nightly conviction score history.

Run locally:
    pip install streamlit sqlalchemy "psycopg[binary]"
    streamlit run dashboard.py

Requires a Supabase (or any Postgres) DATABASE_URL in Streamlit secrets
or as an environment variable.
"""

from __future__ import annotations

import ast
import json
import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

_WEIGHT_LABELS = {
    "sentiment": "Sentiment (news tone)",
    "technical": "Technical (price action)",
    "ml_pred": "ML (5-day forecast)",
    "analyst": "Analyst (street consensus)",
}

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------


def _get_engine():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        try:
            url = st.secrets["DATABASE_URL"]
        except (KeyError, FileNotFoundError):
            st.error(
                "Set DATABASE_URL as an environment variable or in "
                ".streamlit/secrets.toml to connect to your Supabase database."
            )
            st.stop()
    # psycopg (v3) is the supported driver; alias the Postgres scheme so
    # SQLAlchemy uses the psycopg3 dialect instead of the legacy psycopg2.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    return create_engine(url)


engine = _get_engine()

st.set_page_config(page_title="Brief History", layout="wide")

# ---------------------------------------------------------------------------
# Load available dates
# ---------------------------------------------------------------------------

try:
    dates_df = pd.read_sql(
        "SELECT run_date FROM brief_runs ORDER BY run_date",
        engine,
    )
except Exception as exc:
    st.error(f"Could not connect to database: {exc}")
    st.stop()

dates = dates_df["run_date"].tolist()

if not dates:
    st.info("No brief runs found. Run `python main.py brief --persist` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Date selection: prev/next buttons + slider
# ---------------------------------------------------------------------------

if "date_idx" not in st.session_state:
    st.session_state.date_idx = len(dates) - 1

col_prev, col_slider, col_next = st.columns([1, 8, 1])
with col_prev:
    if st.button("◀", disabled=st.session_state.date_idx <= 0):
        st.session_state.date_idx -= 1
        st.rerun()
with col_next:
    if st.button("▶", disabled=st.session_state.date_idx >= len(dates) - 1):
        st.session_state.date_idx += 1
        st.rerun()
with col_slider:
    st.session_state.date_idx = st.select_slider(
        "Date",
        options=range(len(dates)),
        value=st.session_state.date_idx,
        format_func=lambda i: dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], "strftime") else str(dates[i]),
    )

selected_date = dates[st.session_state.date_idx]
selected_date_str = selected_date.strftime("%Y-%m-%d") if hasattr(selected_date, "strftime") else str(selected_date)

# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

run_info = pd.read_sql(
    text("SELECT * FROM brief_runs WHERE run_date = :d"),
    engine,
    params={"d": selected_date_str},
)

if not run_info.empty:
    generated = str(run_info.iloc[0]['generated_at']).replace("+00:00", " UTC")
    weights = run_info.iloc[0]['weights_used']
    try:
        weights = ast.literal_eval(weights) if isinstance(weights, str) else dict(weights)
    except (ValueError, SyntaxError):
        try:
            weights = json.loads(weights)
        except (TypeError, json.JSONDecodeError):
            weights = {}
    if weights:
        weight_str = "  ·  ".join(
            f"{_WEIGHT_LABELS.get(k, k)} {int(round(v * 100))}%"
            for k, v in weights.items()
        )
        st.caption(f"Run generated {generated}  |  Weights: {weight_str}")
    else:
        st.caption(f"Run generated {generated}  |  Weights: {run_info.iloc[0]['weights_used']}")

    with st.expander("How to read these numbers"):
        st.markdown(
            """
Every factor scores the stock on a **-100 to +100** scale, and the **composite** is their
weighted average (then **-25 if an anomaly is flagged**). Weights always sum to 100%.

| Column | What it measures | Meaning |
|---|---|---|
| **Composite** | Blend of all four factors | **≥ +25 bullish** (strong buy-lean), **-25 to +25 neutral** (hold/watch), **≤ -25 bearish** (reduce/avoid) |
| **Sentiment** | News tone of the latest headlines (AI) | + = positive coverage, - = negative coverage |
| **Technical** | Momentum & trend across indicators | + = uptrend/healthy, - = downtrend/weak |
| **ML** | Forecasted probability price rises in ~5 days | +0..100 = chance price is up |
| **Analyst** | Street buy vs hold vs sell consensus | + = mostly buy, - = mostly sell |
| **Anomaly** | Unusual price/volume vs its history | checks = flagged, already subtracted 25 from composite |

Color coding: 🟢 green composite ≥ +25 · 🟡 yellow = neutral · 🔴 red ≤ -25.
"""
        )

# ---------------------------------------------------------------------------
# Day view: conviction scores table
# ---------------------------------------------------------------------------

st.subheader(f"Conviction Scores — {selected_date_str}")

day_df = pd.read_sql(
    text("""SELECT ts.ticker, ts.composite, ts.sentiment, ts.technical,
              ts.ml_pred, ts.analyst, ts.anomaly_flag, ts.reasoning
       FROM ticker_scores ts
       JOIN brief_runs br ON br.id = ts.run_id
       WHERE br.run_date = :d
       ORDER BY ts.composite DESC"""),
    engine,
    params={"d": selected_date_str},
)


def _color_composite(val):
    if val >= 25:
        return "color: #2ecc71; font-weight: bold"
    elif val <= -25:
        return "color: #e74c3c; font-weight: bold"
    else:
        return "color: #f39c12"


styled = day_df.style.map(_color_composite, subset=["composite"])
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "composite": st.column_config.NumberColumn(
            "Composite",
            format="%+.1f",
            help="Weighted average of the four factors (minus 25 if anomaly). ≥ +25 bullish, -25 to +25 neutral, ≤ -25 bearish.",
        ),
        "sentiment": st.column_config.NumberColumn(
            "Sentiment",
            format="%+.1f",
            help="News tone from -100 (very negative coverage) to +100 (very positive).",
        ),
        "technical": st.column_config.NumberColumn(
            "Technical",
            format="%+.1f",
            help="Price momentum & trend from -100 (downtrend) to +100 (uptrend).",
        ),
        "ml_pred": st.column_config.NumberColumn(
            "ML",
            format="%+.1f",
            help="Forecast probability (0-100) the price is higher in ~5 days.",
        ),
        "analyst": st.column_config.NumberColumn(
            "Analyst",
            format="%+.1f",
            help="Street consensus from -100 (mostly sell) to +100 (mostly buy).",
        ),
        "anomaly_flag": st.column_config.CheckboxColumn(
            "Anomaly",
            help="Unusual price/volume vs its history. If flagged, 25 points were already subtracted from composite.",
        ),
    },
)

# ---------------------------------------------------------------------------
# Ticker history: line chart
# ---------------------------------------------------------------------------

st.divider()

if not day_df.empty:
    ticker_options = day_df["ticker"].tolist()
    selected_ticker = st.selectbox("Ticker history", ticker_options)

    hist_df = pd.read_sql(
        text("""SELECT br.run_date, ts.composite, ts.sentiment, ts.technical,
                  ts.ml_pred, ts.analyst
           FROM ticker_scores ts
           JOIN brief_runs br ON br.id = ts.run_id
           WHERE ts.ticker = :t
           ORDER BY br.run_date"""),
        engine,
        params={"t": selected_ticker},
    )

    if not hist_df.empty:
        st.line_chart(
            hist_df.set_index("run_date")[["composite", "sentiment", "technical", "ml_pred", "analyst"]],
            height=350,
        )

        # Show reasoning for the selected date
        row = day_df[day_df["ticker"] == selected_ticker]
        if not row.empty and pd.notna(row.iloc[0].get("reasoning")):
            with st.expander(f"Reasoning — {selected_ticker} on {selected_date_str}"):
                st.markdown(row.iloc[0]["reasoning"].replace(" | ", "\n\n"))

    # Score breakdown bar chart
    if not hist_df.empty:
        st.subheader(f"{selected_ticker} — Component Breakdown Over Time")
        st.area_chart(
            hist_df.set_index("run_date")[["sentiment", "technical", "ml_pred", "analyst"]],
            height=300,
        )

# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

if not day_df.empty:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers Scored", len(day_df))
    col2.metric("Avg Composite", f"{day_df['composite'].mean():+.1f}")
    col3.metric("Anomalies", int(day_df["anomaly_flag"].sum()))
    bullish = (day_df["composite"] >= 25).sum()
    bearish = (day_df["composite"] <= -25).sum()
    col4.metric("Bullish / Bearish", f"{bullish} / {bearish}")
