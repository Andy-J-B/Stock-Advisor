"""
Enhanced Streamlit dashboard with Plotly visualizations.
Run: streamlit run dashboard_new.py
"""

from __future__ import annotations

import html as html_escape
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
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
        "Date",
        options=range(len(dates)),
        value=st.session_state.date_idx,
        format_func=lambda i: str(dates[i])[:10] if dates[i] else str(i),
    )

selected_date = dates[st.session_state.date_idx]
selected_date_str = str(selected_date)[:10]

tabs = st.tabs(
    [
        "Overview",
        "Performance",
        "Risk & Attribution",
        "Allocation",
        "Conviction",
        "Top Buys",
        "Stock Status",
        "Ticker Deep Dive",
    ]
)

with tabs[0]:
    st.header(f"Overview - {selected_date_str}")
    try:
        market_df = pd.read_sql(
            text("SELECT indices, news FROM market_overview WHERE run_date = :d"),
            engine,
            params={"d": selected_date_str},
        )
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
                    cols[i].metric(
                        name, f"{d.get('close', 0):,.2f}", f"{d.get('chg_pct', 0):+.2f}%"
                    )
            if news:
                with st.expander("Market News"):
                    for n in news[:5]:
                        st.markdown(f"**{n.get('title', '')}** - {n.get('publisher', '')}")
    except Exception as e:
        st.warning(f"Market data unavailable: {e}")

    try:
        # Fetch snapshot matching selected date, falling back to latest if exact match missing
        snapshot_df = pd.read_sql(
            text(
                "SELECT * FROM portfolio_snapshots WHERE run_date::text LIKE :d"
                " ORDER BY run_date DESC LIMIT 1"
            ),
            engine,
            params={"d": f"{selected_date_str}%"},
        )
        if snapshot_df.empty:
            snapshots_df = pd.read_sql(
                text("SELECT * FROM portfolio_snapshots ORDER BY run_date DESC LIMIT 30"),
                engine,
            )
            if not snapshots_df.empty:
                latest = snapshots_df.iloc[0]
        else:
            snapshots_df = pd.read_sql(
                text("SELECT * FROM portfolio_snapshots ORDER BY run_date DESC LIMIT 30"),
                engine,
            )
            latest = snapshot_df.iloc[0]

        if not snapshots_df.empty and latest is not None:
            current_nw = float(latest["net_worth_cad"] or 0)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Net Worth (CAD)", f"${current_nw:,.2f}")
            c2.metric("Invested", f"${latest['invested_cad']:,.2f}")
            c3.metric("Cash", f"${latest['cash_cad']:,.2f}")
            c4.metric(
                "Day Change",
                f"${latest['day_change_cad']:+,.2f}",
                f"{latest.get('day_pct', 0):+.2f}%",
            )
            c5.metric("All-Time", f"{latest.get('all_time_pct', 0):+.2f}%")

            # Peak net worth across full snapshot history + compare vs selected date snapshot
            peak_df = pd.read_sql(text("SELECT net_worth_cad FROM portfolio_snapshots"), engine)
            peak_nw = float(peak_df["net_worth_cad"].max()) if not peak_df.empty else 0.0
            if peak_nw > 0:
                change_from_peak_pct = (current_nw - peak_nw) / peak_nw * 100
                change_from_peak_amt = current_nw - peak_nw
                c6.metric(
                    "Peak Net Worth (CAD)",
                    f"${peak_nw:,.2f}",
                    delta=f"{change_from_peak_pct:+.2f}%",
                )
                amt_str = (
                    f"+${abs(change_from_peak_amt):,.2f}"
                    if change_from_peak_amt >= 0
                    else f"-${abs(change_from_peak_amt):,.2f}"
                )
                st.caption(f"Selected vs peak net worth: {change_from_peak_pct:+.2f}% ({amt_str})")

            if len(snapshots_df) > 1:
                st.subheader("Net Worth Trend")
                autoscale = st.checkbox("Autoscale Y-Axis (Zoom to Visible Range)", value=False)

                fig = go.Figure()
                df_plot = snapshots_df.sort_values("run_date")
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["run_date"],
                        y=df_plot["net_worth_cad"],
                        name="Net Worth",
                        mode="lines+markers",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["run_date"],
                        y=df_plot["invested_cad"],
                        name="Invested",
                        mode="lines",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["run_date"],
                        y=df_plot["cash_cad"],
                        name="Cash",
                        mode="lines",
                    )
                )

                yaxis_config = {}
                if autoscale:
                    yaxis_config = {"autorange": True}

                fig.update_layout(title="", height=400, yaxis=yaxis_config, uirevision="constant")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Portfolio data unavailable: {e}")

with tabs[1]:
    st.header("Performance Metrics")
    try:
        snapshots_df = pd.read_sql(
            text("SELECT run_date, net_worth_cad FROM portfolio_snapshots ORDER BY run_date"),
            engine,
        )
        if len(snapshots_df) >= 2:
            df = snapshots_df.copy()
            df["returns"] = df["net_worth_cad"].pct_change()
            daily_rets = df["returns"].dropna().tolist()
            values = df["net_worth_cad"].tolist()
            from src.services import performance

            sharpe = performance.sharpe_ratio(daily_rets)
            sortino = performance.sortino_ratio(daily_rets)
            mdd = performance.max_drawdown(values)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c2.metric("Sortino Ratio", f"{sortino:.2f}")
            c3.metric("Max Drawdown", f"{mdd:.2%}")
            c4.metric(
                "Total Return",
                f"{((values[-1] / values[0]) - 1):.2%}" if values[0] != 0 else "N/A",
            )
            fig = px.histogram(x=daily_rets, nbins=30, title="Daily Returns Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more history for performance metrics")
    except Exception as e:
        st.warning(f"Performance calculation error: {e}")

with tabs[2]:
    st.header("Risk & Attribution")
    try:
        items_df = pd.read_sql(
            text("""
            SELECT ps.run_date, psi.ticker, psi.value_cad, psi.shares, psi.price
            FROM portfolio_snapshot_items psi
            JOIN portfolio_snapshots ps ON ps.id = psi.snapshot_id
            ORDER BY ps.run_date, psi.value_cad DESC
        """),
            engine,
        )

        snapshot_row = pd.read_sql(
            text(
                "SELECT cash_cad, invested_cad FROM portfolio_snapshots WHERE"
                " run_date::text LIKE :d LIMIT 1"
            ),
            engine,
            params={"d": f"{selected_date_str}%"},
        )

        if not items_df.empty and selected_date_str in items_df["run_date"].astype(str).values:
            day_items = items_df[items_df["run_date"].astype(str).str[:10] == selected_date_str]
            if not day_items.empty:
                cash_val = (
                    float(snapshot_row.iloc[0]["cash_cad"]) if not snapshot_row.empty else 0.0
                )

                col_a, col_b = st.columns(2)

                with col_a:
                    fig1 = px.pie(
                        day_items,
                        values="value_cad",
                        names="ticker",
                        title="Equities Allocation Only",
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col_b:
                    # Chart 2: Equities + Cash breakdown
                    cash_df = pd.DataFrame(
                        [
                            {
                                "ticker": "CASH",
                                "value_cad": cash_val,
                                "shares": 0,
                                "price": 1,
                            }
                        ]
                    )
                    combined_df = pd.concat([day_items, cash_df], ignore_index=True)
                    fig2 = px.pie(
                        combined_df,
                        values="value_cad",
                        names="ticker",
                        title="Allocation with Cash Included",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Chart 3: Invested vs Cash aggregate comparison
                invested_val = (
                    float(snapshot_row.iloc[0]["invested_cad"])
                    if not snapshot_row.empty
                    else day_items["value_cad"].sum()
                )
                summary_comp_df = pd.DataFrame(
                    {
                        "Asset Type": ["Equities", "Cash"],
                        "Value": [invested_val, cash_val],
                    }
                )
                fig3 = px.pie(
                    summary_comp_df,
                    values="Value",
                    names="Asset Type",
                    title="Total Portfolio Breakdown (Equities vs. Cash)",
                )
                st.plotly_chart(fig3, use_container_width=True)

                st.dataframe(
                    day_items[["ticker", "shares", "price", "value_cad"]].rename(
                        columns={"value_cad": "Value (CAD)"}
                    )
                )
        else:
            st.info("No attribution data for selected date")
    except Exception as e:
        st.warning(f"Attribution error: {e}")

with tabs[3]:
    st.header("Allocation Over Time")
    try:
        items_df = pd.read_sql(
            text("""
            SELECT ps.run_date, psi.ticker, psi.value_cad
            FROM portfolio_snapshot_items psi
            JOIN portfolio_snapshots ps ON ps.id = psi.snapshot_id
            ORDER BY ps.run_date
        """),
            engine,
        )
        if not items_df.empty:
            pivot = items_df.pivot_table(
                index="run_date", columns="ticker", values="value_cad", aggfunc="sum"
            ).fillna(0)
            fig = px.area(pivot, title="Allocation Over Time (CAD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation history")
    except Exception as e:
        st.warning(f"Allocation error: {e}")

with tabs[4]:
    st.header(f"Conviction Scores - {selected_date_str}")
    try:
        day_df = pd.read_sql(
            text("""
            SELECT ts.ticker, ts.composite, ts.sentiment, ts.technical, ts.ml_pred, ts.analyst,
                   ts.anomaly_flag, ts.price, ts.day_change_pct, ts.portfolio_weight, ts.recommendation
            FROM ticker_scores ts
            JOIN brief_runs br ON br.id = ts.run_id
            WHERE br.run_date = :d
            ORDER BY ts.composite DESC
        """),
            engine,
            params={"d": selected_date_str},
        )
        if not day_df.empty:
            fig = px.bar(
                day_df,
                x="ticker",
                y="composite",
                color="composite",
                color_continuous_scale=["red", "yellow", "green"],
                title="Composite Scores",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(day_df)
        else:
            st.info("No scores for selected date")
    except Exception as e:
        st.warning(f"Conviction data error: {e}")

with tabs[5]:
    st.header("Top Buys — Multi-Factor Screen")
    try:
        runs_df = pd.read_sql(
            text("""
            SELECT id, run_date, generated_at, universe, pick_limit, picks_count, narrative
            FROM top_buys_runs
            ORDER BY run_date DESC
        """),
            engine,
        )
        if runs_df.empty:
            st.info(
                "No top-buys screens found. Run `python main.py top-buys --persist` "
                "(or the nightly workflow) first."
            )
        else:
            dates_tb = runs_df["run_date"].astype(str).str[:10]
            match = runs_df[dates_tb == selected_date_str]
            row = match.iloc[0] if not match.empty else runs_df.iloc[0]
            if match.empty:
                st.caption(
                    f"No screen saved for {selected_date_str} — showing the latest "
                    f"({str(row['run_date'])[:10]})."
                )

            picks_df = pd.read_sql(
                text("""
                SELECT * FROM top_buys_picks
                WHERE run_id = :rid
                ORDER BY rank
            """),
                engine,
                params={"rid": int(row["id"])},
            )

            st.caption(
                f"{row['universe'] or 'screen'} · {len(picks_df)} picks · "
                f"generated {str(row['generated_at'])[:16]}"
            )

            if picks_df.empty:
                st.info("This run has no picks.")
            else:
                avg_score = picks_df["score"].fillna(0).mean()
                avg_upside = picks_df["upside"].fillna(0).mean() * 100
                strong = picks_df["verdict"].fillna("").str.lower().str.contains("strong").sum()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Picks", len(picks_df))
                c2.metric("Avg Score", f"{avg_score:.0f}/100")
                c3.metric("Avg Upside", f"{avg_upside:+.1f}%")
                c4.metric("Strong Buy", int(strong))

                fig = px.bar(
                    picks_df.sort_values("score"),
                    x="score",
                    y="symbol",
                    orientation="h",
                    color="score",
                    color_continuous_scale=["red", "yellow", "green"],
                    range_color=[0, 100],
                    title="Multi-Factor Score (0–100)",
                    hover_data=["verdict", "upside", "sector"],
                )
                st.plotly_chart(fig, use_container_width=True)

                show = picks_df.rename(
                    columns={
                        "rank": "#",
                        "symbol": "Ticker",
                        "company": "Company",
                        "sector": "Sector",
                        "price": "Price",
                        "target": "Target",
                        "upside": "Upside",
                        "score": "Score",
                        "verdict": "Verdict",
                        "analyst_score": "Analyst",
                        "quality_score": "Quality",
                        "valuation_score": "Valuation",
                        "growth_score": "Growth",
                        "momentum_score": "Momentum",
                        "underrated_score": "Underrated",
                        "data_coverage": "Coverage",
                        "risk_penalty": "Risk Penalty",
                    }
                )
                show["Upside"] = (show["Upside"].fillna(0) * 100).round(1)
                cols = [
                    "#",
                    "Ticker",
                    "Company",
                    "Sector",
                    "Price",
                    "Target",
                    "Upside",
                    "Score",
                    "Verdict",
                    "Analyst",
                    "Quality",
                    "Valuation",
                    "Growth",
                    "Momentum",
                    "Underrated",
                    "Coverage",
                    "Risk Penalty",
                ]
                st.dataframe(
                    show[[c for c in cols if c in show.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Why each name ranked (score drivers & risks)"):
                    for _, p in picks_df.iterrows():
                        tags = p.get("tags") or []
                        reasons = p.get("score_reasons") or []
                        risks = p.get("risk_flags") or []
                        st.markdown(
                            f"**{p['symbol']}** — {p.get('score', 0):.0f}/100 · "
                            f"{p.get('verdict') or ''}"
                            + (f" · _{', '.join(map(str, tags))}_" if tags else "")
                        )
                        for reason in reasons:
                            st.markdown(f"- {reason}")
                        if risks:
                            st.markdown(
                                "**Risk flags:** "
                                + ", ".join(html_escape.escape(str(r)) for r in risks)
                            )

                narrative = row.get("narrative")
                if isinstance(narrative, str) and narrative.strip():
                    with st.expander("AI Investment Case & Risks", expanded=False):
                        st.markdown(narrative)
    except Exception as e:
        st.info(
            f"Top-buys data unavailable: {e}. Run `sql/supabase_brief.sql` to create "
            "the `top_buys_runs` / `top_buys_picks` tables."
        )

with tabs[6]:
    st.header("Stock Status — Market Leaders")
    try:
        runs_df = pd.read_sql(
            text("""
            SELECT id, run_date, generated_at, universe, sort,
                   us_count, ca_count, total_count, groups
            FROM stock_status_runs
            ORDER BY run_date DESC
        """),
            engine,
        )
        if runs_df.empty:
            st.info(
                "No stock-status snapshots found. Run "
                "`python main.py stock-status --persist` (or the nightly workflow) first."
            )
        else:
            dates_ss = runs_df["run_date"].astype(str).str[:10]
            match = runs_df[dates_ss == selected_date_str]
            row = match.iloc[0] if not match.empty else runs_df.iloc[0]
            if match.empty:
                st.caption(
                    f"No snapshot saved for {selected_date_str} — showing the latest "
                    f"({str(row['run_date'])[:10]})."
                )

            rows_df = pd.read_sql(
                text("""
                SELECT * FROM stock_status_rows
                WHERE run_id = :rid
                ORDER BY group_name, rank
            """),
                engine,
                params={"rid": int(row["id"])},
            )

            st.caption(
                f"{row['universe'] or 'market leaders'} · {len(rows_df)} names · "
                f"generated {str(row['generated_at'])[:16]}"
            )

            if rows_df.empty:
                st.info("This run has no rows.")
            else:
                ath = rows_df["from_ath_pct"].dropna()
                upside = rows_df["upside_pct"].dropna()
                strong = (
                    rows_df["consensus"].fillna("").str.lower().str.contains("strong").sum()
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Names", len(rows_df))
                c2.metric("Avg From ATH", f"{ath.mean():+.1f}%" if len(ath) else "—")
                c3.metric("Avg Upside", f"{upside.mean():+.1f}%" if len(upside) else "—")
                c4.metric("Strong Buy", int(strong))

                groups = [g for g in rows_df["group_name"].dropna().unique().tolist()]
                choice = "All"
                if len(groups) > 1:
                    choice = st.selectbox("Group", ["All", *groups])
                view = rows_df if choice == "All" else rows_df[rows_df["group_name"] == choice]

                fig = px.bar(
                    view.sort_values("from_ath_pct", ascending=False),
                    x="from_ath_pct",
                    y="symbol",
                    orientation="h",
                    color="from_ath_pct",
                    color_continuous_scale=["red", "yellow", "green"],
                    title="Distance From All-Time High (%)",
                    hover_data=["company", "consensus", "upside_pct", "market_cap"],
                )
                st.plotly_chart(fig, use_container_width=True)

                show = view.rename(
                    columns={
                        "group_name": "Group",
                        "rank": "#",
                        "symbol": "Ticker",
                        "company": "Company",
                        "currency": "Cur",
                        "price": "Price",
                        "day_pct": "Day %",
                        "market_cap": "Mkt Cap",
                        "pe": "P/E",
                        "fwd_pe": "Fwd P/E",
                        "div_yield": "Div %",
                        "from_ath_pct": "From ATH",
                        "target": "Target",
                        "upside_pct": "Upside",
                        "consensus": "Consensus",
                        "total_analysts": "Analysts",
                    }
                )
                cols = [
                    "Group", "#", "Ticker", "Company", "Cur", "Price", "Day %",
                    "Mkt Cap", "P/E", "Fwd P/E", "Div %", "From ATH", "Target",
                    "Upside", "Consensus", "Analysts",
                ]
                st.dataframe(
                    show[[c for c in cols if c in show.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Analyst breakdown & all-time high"):
                    for _, p in view.iterrows():
                        target = p.get("target")
                        ath_price = p.get("ath")
                        target_txt = f"${float(target):,.2f}" if pd.notna(target) else "—"
                        ath_txt = f"${float(ath_price):,.2f}" if pd.notna(ath_price) else "—"
                        st.markdown(
                            f"**{p['symbol']}** — {p.get('consensus') or 'N/A'} · "
                            f"{int(p.get('total_analysts') or 0)} analysts · "
                            f"target {target_txt}"
                        )
                        st.markdown(
                            "Strong Buy "
                            f"{int(p.get('strong_buy') or 0)} · Buy "
                            f"{int(p.get('buy') or 0)} · Hold "
                            f"{int(p.get('hold') or 0)} · Sell "
                            f"{int(p.get('sell') or 0)} · Strong Sell "
                            f"{int(p.get('strong_sell') or 0)} · ATH "
                            f"{ath_txt} ({p.get('ath_date') or '—'})"
                        )
    except Exception as e:
        st.info(
            f"Stock-status data unavailable: {e}. Run `sql/supabase_brief.sql` to "
            "create the `stock_status_runs` / `stock_status_rows` tables."
        )

with tabs[7]:
    st.header("Ticker Deep Dive")
    try:
        day_df = pd.read_sql(
            text("""
            SELECT DISTINCT ts.ticker FROM ticker_scores ts
            JOIN brief_runs br ON br.id = ts.run_id
            ORDER BY ts.ticker
        """),
            engine,
        )
        if not day_df.empty:
            ticker = st.selectbox("Select Ticker", day_df["ticker"].tolist())
            hist_df = pd.read_sql(
                text("""
                SELECT br.run_date, ts.composite, ts.sentiment, ts.technical, ts.ml_pred, ts.analyst
                FROM ticker_scores ts
                JOIN brief_runs br ON br.id = ts.run_id
                WHERE ts.ticker = :t
                ORDER BY br.run_date
            """),
                engine,
                params={"t": ticker},
            )
            if not hist_df.empty:
                fig = go.Figure()
                for col in [
                    "composite",
                    "sentiment",
                    "technical",
                    "ml_pred",
                    "analyst",
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=hist_df["run_date"],
                            y=hist_df[col],
                            name=col,
                            mode="lines+markers",
                        )
                    )
                fig.update_layout(title=f"{ticker} Score History", height=400)
                st.plotly_chart(fig, use_container_width=True)
            bars = pd.read_sql(
                text(
                    "SELECT * FROM price_bars_daily WHERE ticker = :t ORDER BY date DESC LIMIT 90"
                ),
                engine,
                params={"t": ticker},
            )
            if not bars.empty:
                bars = bars.sort_values("date")
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=bars["date"],
                            open=bars["open"],
                            high=bars["high"],
                            low=bars["low"],
                            close=bars["close"],
                        )
                    ]
                )
                fig.update_layout(title=f"{ticker} Price Chart (90 days)", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tickers available")
    except Exception as e:
        st.warning(f"Deep dive error: {e}")
