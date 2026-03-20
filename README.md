# Terminal Stock Advisor

A sleek, Python-based command-line interface (CLI) that helps you manage your stock portfolio, track live prices, and get automated, sentiment-driven investment advice.

Built with a modular architecture, this tool keeps your financial data stored locally while leveraging external APIs to provide real-time market context.

## Features

- **Multi-Account Portfolio Management:** Track your holdings across different accounts (e.g., USD, CAD, TFSA, RRSP).
- **Live Market Data:** Fetches real-time stock prices using `yfinance` to calculate your total return.
- **Custom Risk Profiling:** Configure your personal risk tolerance (Conservative / Moderate / Aggressive) to receive tailored structural advice.
- **AI Sentiment Analysis:** Automatically pulls recent news articles for the stocks you own and uses Natural Language Processing (VADER) to score the sentiment as Bullish, Bearish, or Neutral.
- **Beautiful Terminal UI:** Utilizes `Rich` for gorgeous tables, colored text, and loading spinners.

## Tech Stack

- **Python 3.8+**
- **CLI Framework:** [Typer](https://typer.tiangolo.com/)
- **Terminal UI:** [Rich](https://rich.readthedocs.io/)
- **Market Data:** [yfinance](https://pypi.org/project/yfinance/)
- **Sentiment Analysis:** [NLTK (VADER)](https://www.nltk.org/)

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/yourusername/stock-advisor.git](https://github.com/yourusername/stock-advisor.git)
   cd stock-advisor
   ```

2. **Create a virtual environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the application:**
   Simply run the main script. On the first run, the built-in setup wizard will guide you through creating your local database and configuring your risk profile.
   ```bash
   python main.py
   ```

## 💻 Usage & Commands

The CLI is entirely self-documenting. You can run `python main.py --help` at any time to see available commands.

**Portfolio Management**

- `python main.py add-stock AAPL 10 150.00 --account USD` - Adds 10 shares of Apple bought at $150 to your USD account.
- `python main.py view-portfolio` - Displays a table of all your accounts, holdings, live prices, and profit/loss.
- `python main.py view-portfolio -a CAD` - Displays only the holdings in your CAD account.

**Analysis & Advice**

- `python main.py analyze` - Evaluates your current portfolio structure against your saved risk allocation limits.
- `python main.py market-update` - Fetches macro market news and provides a general sentiment overview.
- `python main.py portfolio-news` - Iterates through the tickers you own, fetches the latest headlines, and advises a Hold/Buy/Sell based on NLP sentiment scoring.

**Configuration**

- `python main.py settings` - Displays your current risk allocation.
- `python main.py settings -c 20 -m 60 -a 20` - Updates your risk allocation to 20% Conservative, 60% Moderate, and 20% Aggressive.

## 📂 Project Structure

```text
stock_advisor/
│
├── main.py                 # Typer CLI application entry point
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
├── data/                   # Auto-generated local storage (ignored in git)
│   ├── settings.json       # User risk profile
│   └── portfolio.json      # Holdings and account data
│
└── src/                    # Core modules
    ├── __init__.py
    ├── setup.py            # First-run initialization wizard
    ├── config.py           # State management for user settings
    ├── portfolio.py        # CRUD operations for local database
    ├── data_client.py      # yfinance and external API integrations
    └── advisor.py          # NLTK sentiment analysis and business logic
```

## Future Roadmap

- [ ] Integrate a dedicated News API (like Finnhub or NewsAPI) for richer macro market data.
- [ ] Add a `remove-stock` command to handle selling positions.
- [ ] Tag individual stocks with specific risk profiles for more granular rebalancing advice.

---

_Disclaimer: This software is for educational purposes only and does not constitute professional financial advice._

```

This completes the initial build of your stock advisor. Would you like to go over how to initialize a Git repository and push this to GitHub, or are you ready to start using the tool locally?
```
