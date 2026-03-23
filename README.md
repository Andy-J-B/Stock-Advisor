# 📈 Terminal Stock Advisor

A sleek, Python-based command-line interface (CLI) that helps you manage your stock portfolio, track live prices across multiple currencies, and get automated, sentiment-driven investment advice.

Built with a modular architecture, this tool keeps your financial data stored locally while leveraging external APIs to provide real-time market context and live currency exchange rates.

## 🛠 Tech Stack

- **Python 3.8+**
- **CLI Framework:** [Typer](https://typer.tiangolo.com/)
- **Terminal UI:** [Rich](https://rich.readthedocs.io/)
- **Market Data:** [yfinance](https://pypi.org/project/yfinance/)
- **FX Rates:** [Exchange-API (Fawaz Ahmed)](https://github.com/fawazahmed0/exchange-api)
- **Sentiment Analysis:** [NLTK (VADER)](https://www.nltk.org/)

## 🚀 Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/stock-advisor.git
   cd stock-advisor
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the application:**
   Simply run the main script to start the setup wizard.
   ```bash
   python main.py
   ```

## 💻 Usage & Commands

### Portfolio Management

- **Add Stock:** `python main.py add-stock AAPL 10 150.00 --account USD`
- **Sell Stock:** `python main.py sell-stock AAPL 5 175.00 --account USD`
  - _Note: Proceeds are automatically converted to CAD and added to your cash balance._
- **Deposit Cash:** `python main.py deposit 1000 --currency USD`
  - _Automatically converts USD to CAD at live market rates._
- **View Portfolio:** `python main.py view-portfolio`
  - _Displays individual account tables plus a **Global CAD Summary** (Total Net Worth)._

### Analysis & Advice

- **`python main.py analyze`** - Evaluates your current portfolio structure against your target risk allocation.
- **`python main.py market-update`** - Fetches macro market news and provides a general sentiment overview.
- **`python main.py portfolio-news`** - Fetches specific headlines for your tickers and advises Hold/Buy/Sell based on sentiment scores.

### Configuration

- **`python main.py settings`** - View/Update your risk allocation (e.g., `-c 20 -m 60 -a 20`).

## 📂 Project Structure

```text
stock_advisor/
│
├── main.py                 # Typer CLI application entry point
├── requirements.txt        # Project dependencies
│
├── data/                   # Local JSON storage (git-ignored)
│   ├── settings.json       # Risk profiles & user config
│   └── portfolio.json      # Current holdings and cash balances
│
└── src/                    # Core logic
    ├── setup.py            # Initialization wizard
    ├── config.py           # Settings management
    ├── portfolio.py        # Logic for Buy/Sell/Deposit & FX conversion
    ├── data_client.py      # yfinance and FX API integrations
    └── advisor.py          # NLTK sentiment analysis engine
```

## 🗺 Future Roadmap

- [ ] **Historical Tracking:** Save daily snapshots of Net Worth to visualize growth over time.
- [ ] **Technical Indicators:** Add RSI and Moving Average signals to the `analyze` command.
- [ ] **Export to CSV:** Ability to export portfolio data for tax or external accounting purposes.
- [ ] **Delete Command:** A `remove-stock` utility to quickly fix entry errors without triggering a "sale."
