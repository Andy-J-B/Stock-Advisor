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
