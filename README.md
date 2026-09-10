# Terminal Stock Advisor

Python CLI tool for managing stock portfolios across CAD/USD accounts, tracking live prices, and getting AI-driven investment advice backed by quantitative analysis, FinBERT sentiment, and LightGBM predictions.

**Version:** 0.2.0

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py          # first launch runs setup wizard
```

Optional interactive menu:

```bash
.venv/bin/python launcher.py
```

## Environment Variables

`.env` file in project root (all optional; app works without them):

| Variable               | Purpose                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`       | AI reports via Gemini 2.5 Flash (`research`, `analyze`, `market-update`, `portfolio-news`, `rebalance`) |
| `ALPHAVANTAGE_API_KEY` | News sentiment, fundamentals, technical indicators, price fallback                                      |
| `FINNHUB_API_KEY`      | Macro news                                                                                              |
| `NEWSAPI_API_KEY`      | Macro news fallback                                                                                     |
| `FMP_API_KEY`          | Advanced fundamentals (key-metrics-ttm)                                                                                |
| `DATABASE_URL`         | Postgres connection string (Supabase) for `brief --persist` (direct connection URI) |
| `DATABASE_REST_URL`    | Supabase PostgREST base URL (e.g. `https://<ref>.supabase.co/rest/v1`)              |
| `SUPABASE_PUBLISHABLE_KEY` | Supabase **publishable** key (`sb_publishable_...`) for PostgREST inserts       |
| `NOTIFY_WEBHOOK_URL`   | Discord/Slack/Telegram webhook for `brief --notify` and nightly-failure alerts      |

Without `GEMINI_API_KEY`, commands fall back to locally-computed analysis.

## Commands

### Analysis & Advice

| Command              | Description                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `analyze`            | Full portfolio review: risk metrics, technical indicators, FinBERT sentiment, Gemini AI allocation advice |
| `optimize-portfolio` | Mean-variance optimization (max-sharpe, min-volatility, efficient-risk) via PyPortfolioOpt                |
| `predict TICKER`     | LightGBM directional prediction (1-day or 5-day horizon) with walk-forward CV                             |
| `market-update`      | Macro news sentiment + Isolation Forest anomaly detection on holdings                                     |
| `portfolio-news`     | Per-ticker news sentiment with headline-level FinBERT breakdown                                           |
| `research TICKER`    | Deep-dive analyst report on a single ticker via Gemini                                                    |
| `top-buys`           | Screen S&P 500 / TSX 60 for high-conviction analyst buys with AI deep-dive                                |

### Portfolio Management

| Command                                        | Description                                       |
| ---------------------------------------------- | ------------------------------------------------- |
| `add-stock TICKER SHARES PRICE --account USD`  | Add a position                                    |
| `sell-stock TICKER SHARES PRICE --account USD` | Sell shares (proceeds auto-convert to CAD)        |
| `deposit AMOUNT --currency USD`                | Deposit cash (auto-converts to CAD at live rates) |
| `view-portfolio`                               | Show all accounts + global CAD summary            |
| `remove-stock TICKER --account USD`            | Remove a holding                                  |
| `set-initial AMOUNT --account USD`             | Set initial cash balance                          |
| `update-cash AMOUNT --account USD`             | Adjust cash balance                               |
| `export`                                       | Export portfolio to CSV                           |
| `dividends` | Show projected annual dividend income |

### Nightly Brief & History

| Command | Description |
|---|---|
| `brief [--tickers ...] [--persist] [--notify]` | Score every holding/watchlist ticker with a conviction score [-100, +100] |
| `brief-weights --sentiment 0.25 --ml 0.30` | View/update conviction scoring weights |
| `watchlist show / add --ticker X / remove --ticker X` | Manage watchlist tickers (scored by `brief`) |
| `dashboard.py` | Streamlit app to browse conviction history day by day |

### Configuration

| Command     | Description                                                                |
| ----------- | -------------------------------------------------------------------------- |
| `settings`  | View/update risk allocation (conservative/moderate/aggressive percentages) |
| `rebalance` | Suggest rebalancing trades to match target allocation                      |
| `tui`       | Launch Textual terminal dashboard (holdings + net worth chart + AI report) |

## Architecture

```
Stock-Advisor/
├── main.py                 # Typer CLI entry point (20+ commands)
├── launcher.py             # Interactive command menu
├── tui.py                  # Textual dashboard (holdings table + chart)
├── requirements.txt
├── .env                    # API keys (git-ignored)
├── data/                   # Runtime state (git-ignored)
│   ├── portfolio.db        # SQLite database (Peewee ORM)
│   ├── settings.json       # Legacy config (migrated to DB)
│   ├── portfolio.json      # Legacy portfolio (migrated to DB)
│   └── models/             # Persisted LightGBM pickles
└── src/
    ├── database.py         # Peewee ORM models + cache helpers + JSON migration
    ├── portfolio.py        # Buy/sell/deposit CRUD + FX conversion + net worth
    ├── config.py           # Risk allocation settings
    ├── setup.py            # First-run setup wizard
    ├── data_client.py      # yfinance + FMP + Finnhub + FX data fetching
    ├── alpha_vantage.py    # Alpha Vantage API client (news, fundamentals, technicals)
    ├── providers.py        # DataProvider protocol + adapter implementations
    ├── ticker_map.py       # Canadian (.NE/.TO) to US ticker resolution
    ├── advisor.py          # Gemini AI orchestration + fallbacks
    ├── sentiment.py        # FinBERT singleton + headline caching
    ├── indicators.py       # RSI, MACD, Bollinger Bands, ATR, EMA (pure pandas)
    ├── risk.py             # VaR, CVaR, Sharpe, Sortino, Max Drawdown
    ├── optimizer.py        # PyPortfolioOpt wrapper + discrete allocation
    ├── features.py         # Lagged feature engineering (no lookahead bias)
    ├── ml_model.py         # LightGBM classifier (walk-forward CV, auto-retrain)
    ├── anomaly.py          # Isolation Forest + GMM anomaly detection
    ├── screener.py         # Analyst-consensus stock screener (top-buys)
    └── brief.py            # Composite conviction scoring (nightly brief)
```

## Nightly Brief & History Dashboard

Two-part feature: score every ticker nightly, then browse that history day by day.

### How the conviction score works

`brief` computes a single **[-100, +100] conviction score** per ticker per day by aggregating
existing modules — no new data sources. Weights are configurable via
`brief-weights` (defaults: sentiment 0.25, technical 0.20, ML 0.30, analyst 0.25).

```
conviction = 0.25*sentiment + 0.20*technical + 0.30*ml_pred + 0.25*analyst  (+ anomaly penalty −25)
```

| Component | Source | Normalized |
|---|---|---|
| Sentiment | FinBERT compound (avg of latest 10 headlines) | −100..+100 |
| Technical | `indicators.py` RSI/MACD/BB/EMA signals | −100..+100 |
| ML | LightGBM P(up) 5-day horizon | P(up) 0..1 → −100..+100 |
| Analyst | buy/hold/sell consensus ratio (`screener.py`) | −100..+100 |
| Anomaly | Isolation Forest flag | −25 if flagged |

Raw components are stored alongside the composite so the dashboard can show *why* a score
moved, and weightings can be backtested retroactively.

### Running the brief

```bash
python main.py brief                          # score holdings + watchlist, console only
python main.py brief --tickers MSFT,NVDA      # override ticker list
python main.py brief --persist                # write to Supabase Postgres
python main.py brief --persist --notify       # + post summary to Discord/Slack webhook

# manage the tickers the brief scores
python main.py watchlist add --ticker NVDA
python main.py watchlist add --ticker MSFT,MU
python main.py watchlist remove --ticker NVDA
python main.py watchlist show

# tune weights (normalized to sum 1.0)
python main.py brief-weights --sentiment 0.30 --ml 0.30
```

### Persisting to Supabase

1. Create a Supabase project, open the SQL editor, run `sql/supabase_brief.sql`.
2. Add these to `.env` (or GitHub Actions secrets):
   - `DATABASE_URL` — **Settings → Database → Connection string → Direct connection → URI** (replace the password placeholder)
   - `DATABASE_REST_URL` — `Settings → API → Project URL` + `/rest/v1`
   - `SUPABASE_PUBLISHABLE_KEY` — `Settings → API → Publishable key` (`sb_publishable_...`). Use the **publishable** key (not the legacy `anon`, which is deprecated by end of 2026; not the secret key, which bypasses RLS). The RLS policies in the schema grant it read + write on these two tables.
3. `python main.py brief --persist` upserts into `brief_runs` / `ticker_scores`
   (one row per run_date).

### Automated nightly runs

`.github/workflows/nightly-brief.yml` runs `brief --persist --notify` on a cron
(weekdays 05:30 UTC, after US close) and posts a failure alert to your webhook.
Add the API keys + `DATABASE_URL`/`DATABASE_REST_URL`/`SUPABASE_PUBLISHABLE_KEY`/
`NOTIFY_WEBHOOK_URL` as repo Actions secrets.

### History dashboard (Streamlit)

```bash
pip install -r requirements.txt
cp .streamlit/secrets.example.toml .streamlit/secrets.toml   # fill in DATABASE_URL
streamlit run dashboard.py
```

The dashboard is a pure read-only viewer over the Postgres history tables
(**no coupling to the local `portfolio.db`**):
- Prev/◀/▶/next buttons + date slider to scrub through days
- Color-coded conviction table with per-component breakdown and anomaly flags
- Per-ticker history line chart (composite + all 4 components over time)
- Expandable reasoning panel explaining each score

## Key Subsystems

### Data Layer

SQLite database (`data/portfolio.db`) stores holdings, settings, and a generic cache with TTLs (prices: 5min, news: 1h, fundamentals: 24h, ticker maps: 30d). Parallel fetching via `ThreadPoolExecutor`. FX conversion uses a free exchange-rate API.

### Canadian Ticker Handling

Portfolios support Canadian-listed securities (CDRs `.NE`, TSX `.TO`). `ticker_map.py` resolves them to US equivalents (`VISA.NE` to `V`, `BRK.TO` to `BRK-B`) using a static map, suffix stripping, and yfinance exchange validation. Unresolved tickers fall back to related market news (`KILO-B.TO` to `GLD`, `VFV.TO` to `SPY`).

### Technical Indicators

Pure-pandas implementation (no `pandas-ta`): RSI-14, MACD 12/26/9, Bollinger Bands 20/2, ATR-14, EMA 20/50. `interpret_signals()` produces readable overbought/oversold/crossover summaries.

### Sentiment Analysis

FinBERT (`ProsusAI/finbert`) scores news headlines as positive/neutral/negative. Compound scores cached per-headline in the database with 24h TTL.

### AI Reports

Gemini 2.5 Flash (`google-genai` SDK) generates market outlook, portfolio analysis, ticker deep-dives, and rebalancing suggestions. Prompts include sector exposure, macro context, and ticker-specific fundamentals.

### ML Prediction

LightGBM classifier trained on lagged price features (returns, volatility, RSI, MACD, Bollinger %B, ATR, EMA spread, volume z-score, optional VIX/10Y macro features). Walk-forward `TimeSeriesSplit` CV. Models persisted per ticker/horizon and auto-retrained after 7 days.

### Portfolio Optimization

PyPortfolioOpt wrapper for mean-variance optimization. Objectives: max-sharpe, min-volatility, efficient-risk. Outputs optimal weights and discrete share allocation given available cash.

### Anomaly Detection

Isolation Forest and Gaussian Mixture models on `volume_zscore`, `volatility_20d`, `return_1d` feature vectors. Flags unusual patterns in portfolio holdings.

### Stock Screener

Fetches S&P 500 / TSX 60 constituents from Wikipedia (cached 7d). Ranks by analyst consensus, enriches with fundamentals, and generates AI deep-dives for top candidates.

## Tech Stack

| Component              | Library                                             |
| ---------------------- | --------------------------------------------------- |
| CLI                    | Typer 0.25.0                                        |
| Terminal UI            | Rich 15.0.0, Textual 8.2.4                          |
| Database               | Peewee 4.0.5 (SQLite)                               |
| Market Data            | yfinance 1.3.0, Alpha Vantage, FMP, Finnhub         |
| Technical Indicators   | Pure pandas                                         |
| Portfolio Optimization | PyPortfolioOpt 1.6.0                                |
| Sentiment              | FinBERT (ProsusAI/finbert) via transformers + torch |
| ML Prediction          | LightGBM 4.7.0 + scikit-learn 1.9.0                 |
| Anomaly Detection      | scikit-learn (Isolation Forest, Gaussian Mixture)   |
| AI Reports             | google-genai (Gemini 2.5 Flash)                     |
| FX Rates               | Exchange-API (Fawaz Ahmed)                          |

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

198 tests across 16 files: cache, database, portfolio, indicators, risk, optimizer, sentiment, features, ML model, anomaly detection, screener, and Canadian-to-US ticker mapping.

CI: ruff lint + pytest with coverage (Python 3.12).
