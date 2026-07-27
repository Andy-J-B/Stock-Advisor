# Terminal Stock Advisor

A Python CLI tool for managing stock portfolios, tracking live prices across multiple currencies, and getting AI-driven investment advice backed by quantitative analysis, sentiment scoring, and machine learning.

## Tech Stack

- **Python 3.13**
- **CLI Framework:** [Typer](https://typer.tiangolo.com/)
- **Terminal UI:** [Rich](https://rich.readthedocs.io/)
- **Database:** [Peewee](http://docs.peewee-orm.com/) (SQLite)
- **Market Data:** [yfinance](https://pypi.org/project/yfinance/) + [Alpha Vantage](https://www.alphavantage.co/)
- **Technical Indicators:** [pandas-ta](https://github.com/twopirllc/pandas-ta)
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

## Commands

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

### Analysis & Advice

| Command | Description |
|---|---|
| `analyze` | Full portfolio review: risk metrics, technical indicators, FinBERT sentiment, allocation |
| `optimize-portfolio` | Mean-variance optimization (max-sharpe, min-volatility, efficient-risk) |
| `predict AAPL` | LightGBM directional prediction (1-day or 5-day horizon) with walk-forward CV |
| `market-update` | Macro news sentiment + anomaly detection on portfolio holdings |
| `portfolio-news` | Per-ticker news sentiment with headline-level breakdown |
| `research AAPL` | Deep dive on a single ticker |

### Configuration

| Command | Description |
|---|---|
| `settings` | View/update risk allocation (conservative/moderate/aggressive) |
| `rebalance` | Suggest rebalancing trades to match target allocation |
| `dividends` | Show dividend history for portfolio holdings |

### TUI

| Command | Description |
|---|---|
| `tui` | Launch the interactive terminal UI (Textual) |

## Architecture

```
stock_advisor/
├── main.py                     # Typer CLI entry point
├── requirements.txt
├── data/                       # Local storage (git-ignored)
│   ├── settings.json
│   └── portfolio.json
├── models/                     # Persisted ML models (git-ignored)
└── src/
    ├── __init__.py
    ├── setup.py                # First-run wizard
    ├── config.py               # Settings management
    ├── database.py             # Peewee ORM + cache helpers
    ├── portfolio.py            # Buy/sell/deposit + FX conversion
    ├── data_client.py          # yfinance + Alpha Vantage data fetching
    ├── providers.py            # DataProvider protocol + adapters
    ├── alpha_vantage.py        # Alpha Vantage API client
    ├── advisor.py              # Sentiment analysis orchestrator
    ├── sentiment.py            # FinBERT sentiment engine (singleton + caching)
    ├── indicators.py           # pandas-ta indicator pipeline (RSI, MACD, BB, EMA, ATR)
    ├── risk.py                 # VaR, CVaR, Sharpe, Sortino, Max Drawdown
    ├── optimizer.py            # PyPortfolioOpt wrapper + discrete allocation
    ├── features.py             # Lagged feature engineering (no lookahead bias)
    ├── ml_model.py             # LightGBM classifier (train/predict/save/load)
    ├── anomaly.py              # Isolation Forest + GMM anomaly detection
    └── providers.py            # Data provider protocol
```

## How It Works

### Caching

All market data is cached locally in SQLite via Peewee (`CacheEntry` model) with configurable TTLs (prices: 5 min, news: 1 hour, sentiment: 24 h). Parallel fetching uses `ThreadPoolExecutor` for batch operations.

### Technical Indicators

`indicators.compute_indicators()` appends RSI, MACD (12/26/9), Bollinger Bands (20/2), ATR, and EMA 20/50 to any OHLCV DataFrame. `interpret_signals()` produces a human-readable signal summary.

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

164 tests covering cache, database, portfolio, indicators, risk, optimizer, sentiment, features, ML model, and anomaly detection.

## Roadmap

- [ ] Backtesting engine (vectorbt + YAML strategy specs)
- [ ] Historical net worth tracking / charting
- [ ] Export to CSV
