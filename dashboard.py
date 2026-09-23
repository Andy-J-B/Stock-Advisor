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


def _bollinger_bands(series, window: int = 20, k: float = 2.0):
    """Rolling mean / standard-deviation bands around a time series."""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    return mid, mid + k * std, mid - k * std

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
# Market overview: index moves + top news for the selected date
# ---------------------------------------------------------------------------

try:
    market_df = pd.read_sql(
        text("SELECT indices, news FROM market_overview WHERE run_date = :d"),
        engine,
        params={"d": selected_date_str},
    )
except Exception:
    market_df = pd.DataFrame()

if not market_df.empty:
    st.divider()
    st.subheader(f"Market — {selected_date_str}")
    indices = market_df.iloc[0]["indices"] or {}
    news = market_df.iloc[0]["news"] or []
    if isinstance(indices, str):
        try:
            indices = json.loads(indices)
        except (TypeError, json.JSONDecodeError):
            indices = {}
    if isinstance(news, str):
        try:
            news = json.loads(news)
        except (TypeError, json.JSONDecodeError):
            news = []

    index_items = [(n, indices[n]) for n in ("S&P 500", "NASDAQ", "TSX 60") if n in indices]
    if index_items:
        cols = st.columns(len(index_items))
        for col, (name, d) in zip(cols, index_items):
            col.metric(
                name,
                f"{d.get('close', 0.0):,.2f}",
                delta=f"{d.get('chg_pct', 0.0):+.2f}%",
            )

    if news:
        with st.expander(f"Top market news ({len(news)})"):
            for n in news[:7]:
                title = (n.get("title") or "").strip()
                pub = n.get("publisher") or ""
                link = n.get("link") or ""
                if not title:
                    continue
                if link:
                    st.markdown(f"**{title}** — {pub}  \n{link}")
                else:
                    st.markdown(f"**{title}** — {pub}")

# ---------------------------------------------------------------------------
# Portfolio — daily snapshots (CAD): overall net worth, allocation, per stock
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Portfolio — Daily Snapshots (CAD)")

try:
    snapshots_df = pd.read_sql(
        text("""SELECT ps.run_date, ps.net_worth_cad, ps.invested_cad, ps.cash_cad,
                       ps.cost_cad, ps.day_change_cad, ps.return_pct,
                       ps.all_time_pct, ps.fx_usd_cad
                FROM portfolio_snapshots ps
                ORDER BY ps.run_date"""),
        engine,
    )
    items_df = pd.read_sql(
        text("""SELECT ps.run_date, psi.ticker, psi.account, psi.value_cad,
                       psi.price, psi.shares, psi.avg_price
                FROM portfolio_snapshot_items psi
                JOIN portfolio_snapshots ps ON ps.id = psi.snapshot_id
                ORDER BY ps.run_date, psi.value_cad DESC"""),
        engine,
    )
except Exception:
    snapshots_df = pd.DataFrame()
    items_df = pd.DataFrame()

if snapshots_df.empty:
    st.info(
        "No portfolio snapshots yet — run `python main.py brief --persist` and "
        "the daily snapshot history will appear here (one row per date)."
    )
else:
    latest = snapshots_df.iloc[-1]
    m_cols = st.columns(5)
    m_cols[0].metric("Net Worth", f"${latest['net_worth_cad']:,.2f}")
    m_cols[1].metric("Invested", f"${latest['invested_cad']:,.2f}")
    m_cols[2].metric("Cash", f"${latest['cash_cad']:,.2f}")
    m_cols[3].metric("Day Change", f"${latest['day_change_cad']:+,.2f}")
    m_cols[4].metric("All-Time", f"{latest['all_time_pct']:+.2f}%")

    # A single daily snapshot is not enough to show a trend.
    has_history = len(snapshots_df) > 1

    tab_overall, tab_alloc, tab_stock = st.tabs(
        ["Net worth & bands", "Allocation over time", "Per stock"]
    )

    overall = snapshots_df.set_index("run_date")

    with tab_overall:
        if not has_history:
            st.info("More than one date of snapshots is needed to chart a trend.")
        chart_col, ctrl_col = st.columns([4, 1])
        with ctrl_col:
            show_bb = st.toggle("Bollinger bands", value=True)
            bb_window = st.slider("BB window (days)", 5, 60, 20,
                                  disabled=not has_history)
            bb_k = st.slider("BB ± std", 1.0, 3.0, 2.0, 0.5,
                             disabled=not has_history)
        chart = overall[["net_worth_cad", "invested_cad", "cash_cad"]]
        if show_bb and has_history:
            mid, upper, lower = _bollinger_bands(
                overall["net_worth_cad"], window=bb_window, k=bb_k
            )
            chart = chart.join(
                pd.DataFrame(
                    {"bb_upper": upper, "bb_mid": mid, "bb_lower": lower},
                    index=overall.index,
                )
            )
        st.line_chart(chart, height=380)
        if show_bb and has_history:
            st.caption(
                f"Bollinger bands ({bb_window}-day rolling mean ± {bb_k:g}σ) on "
                "net worth. Adjust the controls to taste."
            )

    with tab_alloc:
        if not items_df.empty and has_history:
            alloc = items_df.pivot_table(
                index="run_date", columns="ticker", values="value_cad"
            ).fillna(0.0)
            st.area_chart(alloc, height=320)
            st.caption("Position values per day (CAD). Sold positions drop out "
                       "of the snapshot set once removed.")
        else:
            st.info("Allocation history appears once multiple daily snapshots exist.")

    with tab_stock:
        if items_df.empty:
            st.info("No position data yet.")
        else:
            tickers_sorted = sorted(items_df["ticker"].unique().tolist())
            selected = st.selectbox("Stock", tickers_sorted)
            stock = items_df[items_df["ticker"] == selected].set_index("run_date")
            if has_history:
                st.subheader(f"{selected} — Position Value (CAD)")
                st.line_chart(stock["value_cad"], height=260)
                st.subheader(f"{selected} — Price (CAD)")
                price_col, price_ctrl = st.columns([4, 1])
                with price_ctrl:
                    show_p_bb = st.toggle("Bollinger", value=True,
                                          key="price_bb")
                    p_window = st.slider("BB window", 5, 60, 20,
                                         key="price_bb_w")
                    p_k = st.slider("BB ± std", 1.0, 3.0, 2.0, 0.5,
                                    key="price_bb_k")
                price_chart = stock[["price"]].rename(
                    columns={"price": selected}
                )
                if show_p_bb:
                    mid, upper, lower = _bollinger_bands(
                        stock["price"], window=p_window, k=p_k
                    )
                    price_chart = price_chart.join(
                        pd.DataFrame(
                            {"bb_upper": upper, "bb_mid": mid, "bb_lower": lower},
                            index=stock.index,
                        )
                    )
                st.line_chart(price_chart, height=300)
            else:
                st.metric(
                    "Latest",
                    f"{selected}: ${stock['price'].iloc[-1]:,.2f}"
                    f" × {stock['shares'].iloc[-1]:g} sh "
                    f"= ${stock['value_cad'].iloc[-1]:,.2f}",
                )
                st.info("Price/value trends appear once multiple daily snapshots exist.")

# ---------------------------------------------------------------------------
# Day view: conviction scores table
# ---------------------------------------------------------------------------

st.subheader(f"Conviction Scores — {selected_date_str}")

day_df = pd.read_sql(
    text("""SELECT ts.ticker, ts.composite, ts.sentiment, ts.technical,
              ts.ml_pred, ts.analyst, ts.anomaly_flag, ts.price,
              ts.day_change_pct, ts.top_headline, ts.analyst_breakdown,
              ts.signal_agreement, ts.anomaly_detail, ts.portfolio_weight,
              ts.recommendation, ts.reasoning
       FROM ticker_scores ts
       JOIN brief_runs br ON br.id = ts.run_id
       WHERE br.run_date = :d
       ORDER BY ts.composite DESC"""),
    engine,
    params={"d": selected_date_str},
)

# Delta vs the previous run (same day's table contains no history).
prev_dates = [d for d in dates if d < selected_date]
prev_date = prev_dates[-1] if prev_dates else None
if prev_date is not None:
    prev_date_str = prev_date.strftime("%Y-%m-%d") if hasattr(prev_date, "strftime") else str(prev_date)
else:
    prev_date_str = None

prev_scores = {}
if prev_date_str:
    prev_df = pd.read_sql(
        text("""SELECT ts.ticker, ts.composite
         FROM ticker_scores ts
         JOIN brief_runs br ON br.id = ts.run_id
         WHERE br.run_date = :d"""),
        engine,
        params={"d": prev_date_str},
    )
    prev_scores = dict(zip(prev_df["ticker"], prev_df["composite"]))
    day_df["score_delta"] = day_df["ticker"].map(lambda t: prev_scores.get(t, None))
    day_df["score_delta"] = day_df.apply(
        lambda r: round(r["composite"] - r["score_delta"], 1)
        if r["score_delta"] is not None else None,
        axis=1,
    )
    prev_ranks = {t: i for i, t in enumerate(
        prev_df.sort_values("composite", ascending=False)["ticker"]
    )}
    day_df["rank_change"] = day_df.apply(
        lambda r: prev_ranks[r["ticker"]] - r.name if r["ticker"] in prev_ranks else None,
        axis=1,
    )
else:
    day_df["score_delta"] = None
    day_df["rank_change"] = None

day_df = day_df.drop(columns=["rank_change"])


def _color_composite(val):
    if val >= 25:
        return "color: #2ecc71; font-weight: bold"
    elif val <= -25:
        return "color: #e74c3c; font-weight: bold"
    else:
        return "color: #f39c12"


def _color_delta(val):
    if val is None or pd.isna(val):
        return ""
    if val > 0:
        return "color: #2ecc71"
    if val < 0:
        return "color: #e74c3c"
    return "color: #888"


styled = (
    day_df.style
    .map(_color_composite, subset=["composite"])
    .map(_color_delta, subset=["score_delta"])
)
st.dataframe(
    styled,
    width="stretch",
    hide_index=True,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", pinned="left"),
        "composite": st.column_config.NumberColumn(
            "Composite",
            format="%+.1f",
            help="Weighted average of the four factors (minus 25 if anomaly). ≥ +25 bullish, -25 to +25 neutral, ≤ -25 bearish.",
        ),
        "score_delta": st.column_config.NumberColumn(
            "Δ vs prev",
            format="%+.1f",
            help=(
                f"Change in composite since {prev_date_str}."
                if prev_date_str else "Change in composite vs the previous run."
            ),
        ),
        "price": st.column_config.NumberColumn(
            "Price",
            format="$%.2f",
            help="Last close / live price captured with the run.",
        ),
        "day_change_pct": st.column_config.NumberColumn(
            "Day %",
            format="%+.2f%%",
            help="Intraday change vs the previous close.",
        ),
        "portfolio_weight": st.column_config.NumberColumn(
            "Pos %",
            format="%.1f%%",
            help="Share of total portfolio value this holding represents (CAD).",
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
        "signal_agreement": st.column_config.TextColumn(
            "Signal",
            help="How much the four factors agree: High agreement / Moderate / Conflicted / Low coverage.",
        ),
        "recommendation": st.column_config.TextColumn(
            "Action",
            help="Rule-based nudge from conviction + position size. Not financial advice.",
        ),
    },
)

if prev_date_str:
    st.caption(f"Δ vs previous run **{prev_date_str}** — green = improved, red = worsened.")

# ---------------------------------------------------------------------------
# Score distribution + biggest movers
# ---------------------------------------------------------------------------

if not day_df.empty:
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Score Distribution")
        buckets = [float(b) for b in [-100, -75, -50, -25, 0, 25, 50, 75, 100]]
        labels = [f"{b}+" for b in buckets[:-1]]
        counts = pd.cut(day_df["composite"], bins=buckets, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)
        st.bar_chart(counts, height=240)
    with c2:
        st.subheader("Biggest Movers")
        if prev_date_str and day_df["score_delta"].notna().any():
            movers = day_df.dropna(subset=["score_delta"]).copy()
            movers["abs_delta"] = movers["score_delta"].abs()
            movers = movers.sort_values("abs_delta", ascending=False).head(5)
            for _, r in movers.iterrows():
                arrow = "🟢" if r["score_delta"] >= 0 else "🔴"
                st.markdown(
                    f"{arrow} **{r['ticker']}**: {r['composite']:+.1f} "
                    f"({r['score_delta']:+.1f} vs {prev_date_str})"
                )
        else:
            st.info("No previous run yet — movers appear once a second brief is persisted.")

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

        # Show reasoning + enrichment for the selected date
        row = day_df[day_df["ticker"] == selected_ticker]
        if not row.empty:
            r = row.iloc[0]
            details = []
            if pd.notna(r.get("top_headline")) and str(r["top_headline"]).strip():
                details.append(f"**Top headline:** {r['top_headline']}")
            if pd.notna(r.get("analyst_breakdown")) and str(r["analyst_breakdown"]).strip():
                details.append(f"**Analyst breakdown:** {r['analyst_breakdown']}")
            if pd.notna(r.get("signal_agreement")) and str(r["signal_agreement"]).strip():
                details.append(f"**Signal agreement:** {r['signal_agreement']}")
            if pd.notna(r.get("recommendation")) and str(r["recommendation"]).strip():
                details.append(f"**Suggestion:** {r['recommendation']}")
            if bool(r.get("anomaly_flag")) and pd.notna(r.get("anomaly_detail")) and str(r["anomaly_detail"]).strip():
                details.append(f"**Anomaly detail:**\n\n{r['anomaly_detail']}")
            if pd.notna(r.get("reasoning")) and str(r["reasoning"]).strip():
                details.append(r["reasoning"].replace(" | ", "\n\n"))
            if details:
                with st.expander(f"Details — {selected_ticker} on {selected_date_str}"):
                    st.markdown("\n\n".join(details))

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
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Tickers Scored", len(day_df))
    col2.metric("Avg Composite", f"{day_df['composite'].mean():+.1f}")
    col3.metric("Anomalies", int(day_df["anomaly_flag"].sum()))
    bullish = int((day_df["composite"] >= 25).sum())
    bearish = int((day_df["composite"] <= -25).sum())
    col4.metric("Bullish / Bearish", f"{bullish} / {bearish}")
    if "score_delta" in day_df and day_df["score_delta"].notna().any():
        avg_delta = day_df["score_delta"].mean()
        up = int((day_df["score_delta"] > 0).sum())
        down = int((day_df["score_delta"] < 0).sum())
        col5.metric(f"Δ vs {prev_date_str}", f"{avg_delta:+.1f}")
        col6.metric("Up / Down", f"{up} / {down}")
    else:
        col5.metric("Δ vs prev", "—")
        col6.metric("Up / Down", "—")
    st.caption(
        "Bullish ≥ +25 / Bearish ≤ -25 on the composite — which already includes "
        "the -25 anomaly penalty where flagged."
    )
