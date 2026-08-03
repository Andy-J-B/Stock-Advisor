# Terminal Stock Advisor

A Python CLI tool for managing stock portfolios, tracking live prices across multiple currencies, and getting AI-driven investment advice backed by quantitative analysis, sentiment scoring, and machine learning.

## Tech Stack

- **Python 3.13**
- **CLI Framework:** [Typer](https://typer.tiangolo.com/)
- **Terminal UI:** [Rich](https://rich.readthedocs.io/)
- **Database:** [Peewee](http://docs.peewee-orm.com/) (SQLite)
- **Market Data:** [yfinance](https://pypi.org/project/yfinance/) + [Alpha Vantage](https://www.alphavantage.co/)
- **Technical Indicators:** Pure pandas (RSI, MACD, Bollinger Bands, EMA, ATR)
- **Portfolio Optimization:** [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
- **Sentiment Analysis:** [FinBERT](https://huggingface.co/ProsusAI/finbert) (transformers + torch)
- **ML Prediction:** [LightGBM](https://lightgbm.readthedocs.io/) + scikit-learn
- **Anomaly Detection:** scikit-learn (Isolation Forest, Gaussian Mixture)
- **FX Rates:** [Exchange-API (Fawaz Ahmed)](https://github.com/fawazahmed0/exchange-api)

## Installation

```bash
git clone https://github.com/yourusername/stock-advisor.git
cd stock-advisor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py   # runs setup wizard on first launch
```

> **Note:** FinBERT downloads ~440 MB on first use (model weights). Subsequent runs use the cached model.

## Configuration

Create a `.env` file in the project root for optional API keys:

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` | AI deep-dive reports (`research`) and rebalancing suggestions |
| `ALPHAVANTAGE_API_KEY` | Fallback price/news provider alongside yfinance |
| `FINNHUB_API_KEY` / `NEWSAPI_API_KEY` / `FMP_API_KEY` | Optional extra news providers |

Without `GEMINI_API_KEY`, the `research` and `rebalance` commands fall back to a locally-computed analysis instead of an AI report.

## Interactive Launcher

Prefer pointing and clicking over typing? Run the launcher to see every command
with a one-line description and run it interactively — pick a number or type a
full command line:

```bash
.venv/bin/python launcher.py
```

## Commands

### Analysis & Advice

| Command | Description |
|---|---|
| `analyze` | Full portfolio review: risk metrics, technical indicators, FinBERT sentiment, allocation |
| `optimize-portfolio` | Mean-variance optimization (max-sharpe, min-volatility, efficient-risk) |
| `predict AAPL` | LightGBM directional prediction (1-day or 5-day horizon) with walk-forward CV |
| `market-update` | Macro news sentiment + anomaly detection on portfolio holdings |
| `portfolio-news` | Per-ticker news sentiment with headline-level breakdown |
| `research AAPL` | Deep dive on a single ticker |
| `top-buys` | Screen S&P 500 / TSX 60 (or your own list) for high-conviction analyst buys with AI deep-dive |

### Portfolio Management

| Command | Description |
|---|---|
| `add-stock AAPL 10 150.00 --account USD` | Add a position |
| `sell-stock AAPL 5 175.00 --account USD` | Sell shares (proceeds auto-convert to CAD) |
| `deposit 1000 --currency USD` | Deposit cash (auto-converts to CAD at live rates) |
| `view-portfolio` | Show all accounts + Global CAD Summary |
| `remove-stock AAPL --account USD` | Remove a holding |
| `set-initial 50000 --account USD` | Set initial cash balance |
| `update-cash 1000 --account USD` | Adjust cash balance |
| `export` | Export portfolio to CSV |
| `dividends` | Show projected annual dividend income from holdings |

### Configuration

| Command | Description |
|---|---|
| `settings` | View/update risk allocation (conservative/moderate/aggressive) |
| `rebalance` | Suggest rebalancing trades to match target allocation |

### TUI

| Command | Description |
|---|---|
| `tui` | Launch the interactive terminal UI (Textual) |

## Architecture

```
stock_advisor/
├── main.py                     # Typer CLI entry point
├── launcher.py                 # Interactive command menu
├── requirements.txt
├── data/                       # Local storage (git-ignored)
│   ├── settings.json
│   └── portfolio.json
├── models/                     # Persisted ML models (git-ignored)
├── tui/                        # Textual dashboard app
└── src/
    ├── __init__.py
    ├── setup.py                # First-run wizard
    ├── config.py               # Settings management
    ├── database.py             # Peewee ORM + cache helpers
    ├── portfolio.py            # Buy/sell/deposit + FX conversion
    ├── data_client.py          # yfinance + Alpha Vantage data fetching
    ├── providers.py            # DataProvider protocol + adapters
    ├── alpha_vantage.py        # Alpha Vantage API client
    ├── ticker_map.py           # Canadian (.NE/.TO) → US ticker resolution
    ├── advisor.py              # Sentiment analysis orchestrator
    ├── sentiment.py            # FinBERT sentiment engine (singleton + caching)
    ├── indicators.py           # Pure pandas indicators (RSI, MACD, BB, EMA, ATR)
    ├── risk.py                 # VaR, CVaR, Sharpe, Sortino, Max Drawdown
    ├── optimizer.py            # PyPortfolioOpt wrapper + discrete allocation
    ├── features.py             # Lagged feature engineering (no lookahead bias)
    ├── ml_model.py             # LightGBM classifier (train/predict/save/load)
    └── anomaly.py              # Isolation Forest + GMM anomaly detection
```

## Canadian Ticker Handling

Portfolios hold Canadian-listed securities as CDRs (`.NE`, e.g. `MSFT.NE`,
`VISA.NE`) and ETFs/companies (`.TO`, e.g. `VFV.TO`, `BRK.TO`). Alpha Vantage
does not support these suffixes and news coverage is thinner, so:

- `src/ticker_map.py` resolves each ticker to its US equivalent
  (`VISA.NE → V`, `BRK.TO → BRK-B`) using a static map, CDR base-stripping, and
  a yfinance exchange check, cached 30 days in the database.
- `portfolio-news` shows the mapping (`MSFT.NE (→ MSFT)`) and uses the US
  ticker when fetching headlines.
- Unresolved Canadian tickers fall back to a related market's news
  (`KILO-B.TO → GLD` gold news, `VFV.TO → SPY`, `VCE.TO → XIU.TO`).

## How It Works

### Caching

All market data is cached locally in SQLite via Peewee (`CacheEntry` model) with configurable TTLs (prices: 5 min, news: 1 hour, sentiment: 24 h, ticker resolution: 30 days). Parallel fetching uses `ThreadPoolExecutor` for batch operations. News is fetched through yfinance's headline feed (data nested under the `content` field in current API versions) and cached per-ticker/limit.

### Technical Indicators

`indicators.compute_indicators()` appends RSI, MACD (12/26/9), Bollinger Bands (20/2), ATR, and EMA 20/50 to any OHLCV DataFrame. `interpret_signals()` produces a human-readable signal summary. The implementation is pure pandas — no `pandas-ta` dependency.

### Sentiment

FinBERT (`ProsusAI/finbert`) scores news headlines as positive/neutral/negative. Compound scores are cached per-headline in the database. The `analyze` and `portfolio-news` commands show weighted sentiment summaries.

### Risk Metrics

`risk.py` computes Historical VaR/CVaR, annualized Sharpe and Sortino ratios, and maximum drawdown from price series. Displayed in the `analyze` command's risk panel.

### Portfolio Optimization

PyPortfolioOpt wrapper supports max-sharpe, min-volatility, and efficient-risk objectives. Outputs optimal weights plus a discrete share allocation given available cash.

### ML Prediction

`predict` trains a LightGBM classifier on lagged price features (returns, volatility, RSI, MACD, Bollinger %B, ATR, EMA spread, volume z-score) using walk-forward time-series cross-validation. Models are persisted per ticker/horizon and auto-retrained when stale.

### Anomaly Detection

`market-update` runs Isolation Forest on portfolio holdings' feature vectors, flagging unusual volume/volatility/price patterns as a Rich warning panel.

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

180 tests covering cache, database, portfolio, indicators, risk, optimizer, sentiment, features, ML model, anomaly detection, and Canadian→US ticker mapping.

## Roadmap

- [ ] Backtesting engine (vectorbt + YAML strategy specs)
- [ ] Historical net worth tracking / charting
- [ ] Live quotes in the TUI dashboard
