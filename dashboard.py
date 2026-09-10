"""
Streamlit dashboard for browsing nightly conviction score history.

Run locally:
    pip install streamlit sqlalchemy psycopg2-binary
    streamlit run dashboard.py

Requires a Supabase (or any Postgres) DATABASE_URL in Streamlit secrets
or as an environment variable.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

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
    st.caption(
        f"Generated: {run_info.iloc[0]['generated_at']}  |  "
        f"Weights: {run_info.iloc[0]['weights_used']}"
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


styled = day_df.style.applymap(_color_composite, subset=["composite"])
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "composite": st.column_config.NumberColumn("Composite", format="%+.1f"),
        "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.1f"),
        "technical": st.column_config.NumberColumn("Technical", format="%+.1f"),
        "ml_pred": st.column_config.NumberColumn("ML", format="%+.1f"),
        "analyst": st.column_config.NumberColumn("Analyst", format="%+.1f"),
        "anomaly_flag": st.column_config.CheckboxColumn("Anomaly"),
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
