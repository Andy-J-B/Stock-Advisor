import json
from src.setup import PORTFOLIO_FILE


def load():
    """Reads the portfolio from disk."""
    if not PORTFOLIO_FILE.exists():
        return {"accounts": {"USD": {"holdings": {}}, "CAD": {"holdings": {}}}}
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def save(data):
    """Writes the updated portfolio back to disk."""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)


def ensure_account_exists(portfolio_data, account_name):
    """Helper to make sure an account exists before adding to it."""
    account_name = account_name.upper()
    if "accounts" not in portfolio_data:
        portfolio_data["accounts"] = {}
    if account_name not in portfolio_data["accounts"]:
        portfolio_data["accounts"][account_name] = {"holdings": {}}
    return portfolio_data, account_name


def add_position(account: str, ticker: str, shares: float, price: float):
    """Adds a new stock or updates an existing position's average cost."""
    portfolio_data = load()
    portfolio_data, account = ensure_account_exists(portfolio_data, account)

    ticker = ticker.upper()
    holdings = portfolio_data["accounts"][account]["holdings"]

    if ticker in holdings:
        # Calculate new average price
        old_shares = holdings[ticker]["shares"]
        old_price = holdings[ticker]["avg_price"]

        total_shares = old_shares + shares
        total_cost = (old_shares * old_price) + (shares * price)
        new_avg = total_cost / total_shares

        holdings[ticker]["shares"] = total_shares
        holdings[ticker]["avg_price"] = new_avg
    else:
        # Brand new position
        holdings[ticker] = {"shares": shares, "avg_price": price}

    save(portfolio_data)


def get_account_holdings(account: str):
    """Returns the holdings for a specific account."""
    portfolio_data = load()
    account = account.upper()
    return portfolio_data.get("accounts", {}).get(account, {}).get("holdings", {})
