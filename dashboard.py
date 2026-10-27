"""
Enhanced Streamlit dashboard with Plotly visualizations.
Run: streamlit run dashboard_new.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text


def _get_engine():
    url = os.getenv("DATABASE_URL", "")
    if not url:
        try:
            url = st.secrets["DATABASE_URL"]
        except Exception:
            st.error("Set DATABASE_URL")
            st.stop()
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    return create_engine(url)


engine = _get_engine()
st.set_page_config(page_title="Stock Advisor - Enhanced Dashboard", layout="wide")


@st.cache_data(ttl=300)
def load_dates():
    try:
        df = pd.read_sql("SELECT run_date FROM brief_runs ORDER BY run_date", engine)
        return df["run_date"].tolist()
    except Exception:
        return []


dates = load_dates()
if not dates:
    st.info("No brief runs found. Run `python main.py brief --persist` first.")
    st.stop()

if "date_idx" not in st.session_state:
    st.session_state.date_idx = len(dates) - 1

col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    if st.button("◀", disabled=st.session_state.date_idx <= 0):
        st.session_state.date_idx -= 1
        st.rerun()
with col3:
    if st.button("▶", disabled=st.session_state.date_idx >= len(dates) - 1):
        st.session_state.date_idx += 1
        st.rerun()
with col2:
    st.session_state.date_idx = st.select_slider(
        "Date", options=range(len(dates)), value=st.session_state.date_idx,
        format_func=lambda i: str(dates[i])[:10] if dates[i] else str(i)
    )

selected_date = dates[st.session_state.date_idx]
selected_date_str = str(selected_date)[:10]

tabs = st.tabs(["Overview", "Performance", "Risk & Attribution", "Allocation", "Conviction", "Ticker Deep Dive"])

with tabs[0]:
    st.header(f"Overview - {selected_date_str}")
    try:
        market_df = pd.read_sql(text("SELECT indices, news FROM market_overview WHERE run_date = :d"), engine, params={"d": selected_date_str})
        if not market_df.empty:
            indices = market_df.iloc[0]["indices"]
            if isinstance(indices, str):
                indices = json.loads(indices)
            news = market_df.iloc[0]["news"]
            if isinstance(news, str):
                news = json.loads(news)
            cols = st.columns(3)
            for i, name in enumerate(["S&P 500", "NASDAQ", "TSX 60"]):
                if name in indices:
                    d = indices[name]
                    cols[i].metric(name, f"{d.get('close',0):,.2f}", f"{d.get('chg_pct',0):+.2f}%")
            if news:
                with st.expander("Market News"):
                    for n in news[:5]:
                        st.markdown(f"**{n.get('title','')}** - {n.get('publisher','')}")
    except Exception as e:
        st.warning(f"Market data unavailable: {e}")

    try:
        snapshots_df = pd.read_sql(text("SELECT * FROM portfolio_snapshots ORDER BY run_date DESC LIMIT 30"), engine)
        if not snapshots_df.empty:
            latest = snapshots_df.iloc[0]
            current_nw = float(latest['net_worth_cad'] or 0)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Net Worth (CAD)", f"${current_nw:,.2f}")
            c2.metric("Invested", f"${latest['invested_cad']:,.2f}")
            c3.metric("Cash", f"${latest['cash_cad']:,.2f}")
            c4.metric("Day Change", f"${latest['day_change_cad']:+,.2f}", f"{latest.get('day_pct',0):+.2f}%")
            c5.metric("All-Time", f"{latest.get('all_time_pct',0):+.2f}%")
            # Peak net worth across full snapshot history + compare vs current
            peak_df = pd.read_sql(text("SELECT net_worth_cad FROM portfolio_snapshots"), engine)
            peak_nw = float(peak_df['net_worth_cad'].max()) if not peak_df.empty else 0.0
            if peak_nw > 0:
                change_from_peak_pct = (current_nw - peak_nw) / peak_nw * 100
                change_from_peak_amt = current_nw - peak_nw
                c6.metric("Peak Net Worth (CAD)", f"${peak_nw:,.2f}", delta=f"{change_from_peak_pct:+.2f}%")
                amt_str = f"+${abs(change_from_peak_amt):,.2f}" if change_from_peak_amt >= 0 else f"-${abs(change_from_peak_amt):,.2f}"
                st.caption(f"Current vs peak: {change_from_peak_pct:+.2f}% ({amt_str})")
            if len(snapshots_df) > 1:
                fig = go.Figure()
                df_plot = snapshots_df.sort_values('run_date')
                fig.add_trace(go.Scatter(x=df_plot['run_date'], y=df_plot['net_worth_cad'], name='Net Worth', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=df_plot['run_date'], y=df_plot['invested_cad'], name='Invested', mode='lines'))
                fig.add_trace(go.Scatter(x=df_plot['run_date'], y=df_plot['cash_cad'], name='Cash', mode='lines'))
                fig.update_layout(title="Net Worth Trend", height=400)
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Portfolio data unavailable: {e}")

with tabs[1]:
    st.header("Performance Metrics")
    try:
        snapshots_df = pd.read_sql(text("SELECT run_date, net_worth_cad FROM portfolio_snapshots ORDER BY run_date"), engine)
        if len(snapshots_df) >= 2:
            df = snapshots_df.copy()
            df['returns'] = df['net_worth_cad'].pct_change()
            daily_rets = df['returns'].dropna().tolist()
            values = df['net_worth_cad'].tolist()
            from src.services import performance
            sharpe = performance.sharpe_ratio(daily_rets)
            sortino = performance.sortino_ratio(daily_rets)
            mdd = performance.max_drawdown(values)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c2.metric("Sortino Ratio", f"{sortino:.2f}")
            c3.metric("Max Drawdown", f"{mdd:.2%}")
            c4.metric("Total Return", f"{((values[-1]/values[0])-1):.2%}" if values[0] != 0 else "N/A")
            fig = px.histogram(x=daily_rets, nbins=30, title="Daily Returns Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more history for performance metrics")
    except Exception as e:
        st.warning(f"Performance calculation error: {e}")

with tabs[2]:
    st.header("Risk & Attribution")
    try:
        items_df = pd.read_sql(text("""
            SELECT ps.run_date, psi.ticker, psi.value_cad, psi.shares, psi.price
            FROM portfolio_snapshot_items psi
            JOIN portfolio_snapshots ps ON ps.id = psi.snapshot_id
            ORDER BY ps.run_date, psi.value_cad DESC
        """), engine)
        if not items_df.empty and selected_date_str in items_df['run_date'].astype(str).values:
            day_items = items_df[items_df['run_date'].astype(str).str[:10] == selected_date_str]
            if not day_items.empty:
                fig = px.pie(day_items, values='value_cad', names='ticker', title="Current Allocation")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(day_items[['ticker', 'shares', 'price', 'value_cad']].rename(columns={'value_cad': 'Value (CAD)'}))
        else:
            st.info("No attribution data for selected date")
    except Exception as e:
        st.warning(f"Attribution error: {e}")

with tabs[3]:
    st.header("Allocation Over Time")
    try:
        items_df = pd.read_sql(text("""
            SELECT ps.run_date, psi.ticker, psi.value_cad
            FROM portfolio_snapshot_items psi
            JOIN portfolio_snapshots ps ON ps.id = psi.snapshot_id
            ORDER BY ps.run_date
        """), engine)
        if not items_df.empty:
            pivot = items_df.pivot_table(index='run_date', columns='ticker', values='value_cad', aggfunc='sum').fillna(0)
            fig = px.area(pivot, title="Allocation Over Time (CAD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation history")
    except Exception as e:
        st.warning(f"Allocation error: {e}")

with tabs[4]:
    st.header(f"Conviction Scores - {selected_date_str}")
    try:
        day_df = pd.read_sql(text("""
            SELECT ts.ticker, ts.composite, ts.sentiment, ts.technical, ts.ml_pred, ts.analyst,
                   ts.anomaly_flag, ts.price, ts.day_change_pct, ts.portfolio_weight, ts.recommendation
            FROM ticker_scores ts
            JOIN brief_runs br ON br.id = ts.run_id
            WHERE br.run_date = :d
            ORDER BY ts.composite DESC
        """), engine, params={"d": selected_date_str})
        if not day_df.empty:
            fig = px.bar(day_df, x='ticker', y='composite', color='composite',
                         color_continuous_scale=['red', 'yellow', 'green'],
                         title="Composite Scores")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(day_df)
        else:
            st.info("No scores for selected date")
    except Exception as e:
        st.warning(f"Conviction data error: {e}")

with tabs[5]:
    st.header("Ticker Deep Dive")
    try:
        day_df = pd.read_sql(text("""
            SELECT DISTINCT ts.ticker FROM ticker_scores ts
            JOIN brief_runs br ON br.id = ts.run_id
            ORDER BY ts.ticker
        """), engine)
        if not day_df.empty:
            ticker = st.selectbox("Select Ticker", day_df['ticker'].tolist())
            hist_df = pd.read_sql(text("""
                SELECT br.run_date, ts.composite, ts.sentiment, ts.technical, ts.ml_pred, ts.analyst
                FROM ticker_scores ts
                JOIN brief_runs br ON br.id = ts.run_id
                WHERE ts.ticker = :t
                ORDER BY br.run_date
            """), engine, params={"t": ticker})
            if not hist_df.empty:
                fig = go.Figure()
                for col in ['composite', 'sentiment', 'technical', 'ml_pred', 'analyst']:
                    fig.add_trace(go.Scatter(x=hist_df['run_date'], y=hist_df[col], name=col, mode='lines+markers'))
                fig.update_layout(title=f"{ticker} Score History", height=400)
                st.plotly_chart(fig, use_container_width=True)
            bars = pd.read_sql(text("SELECT * FROM price_bars_daily WHERE ticker = :t ORDER BY date DESC LIMIT 90"), engine, params={"t": ticker})
            if not bars.empty:
                bars = bars.sort_values('date')
                fig = go.Figure(data=[go.Candlestick(x=bars['date'], open=bars['open'], high=bars['high'], low=bars['low'], close=bars['close'])])
                fig.update_layout(title=f"{ticker} Price Chart (90 days)", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tickers available")
    except Exception as e:
        st.warning(f"Deep dive error: {e}")
