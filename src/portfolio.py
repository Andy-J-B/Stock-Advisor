import json
from src.setup import PORTFOLIO_FILE
from datetime import date
from src import data_client


def load():
    if not PORTFOLIO_FILE.exists():
        return {
            "accounts": {"USD": {"holdings": {}}, "CAD": {"holdings": {"cash": 0.0}}}
        }
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def save(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)


def ensure_account_exists(portfolio_data, account_name):
    account_name = account_name.upper()
    if "accounts" not in portfolio_data:
        portfolio_data["accounts"] = {}
    if account_name not in portfolio_data["accounts"]:
        portfolio_data["accounts"][account_name] = {"holdings": {}, "cash": 0.0}
    return portfolio_data, account_name


def deposit_cash(amount: float, currency: str):
    """Adds cash to the CAD master account, converting USD if necessary."""
    portfolio_data = load()
    currency = currency.upper()

    # Ensure CAD account exists as the primary cash bucket
    portfolio_data, _ = ensure_account_exists(portfolio_data, "CAD")

    if currency == "USD":
        rate = data_client.get_usd_to_cad()
        converted_amount = amount * rate
        portfolio_data["accounts"]["CAD"]["cash"] += converted_amount
        return converted_amount, rate
    else:
        portfolio_data["accounts"]["CAD"]["cash"] += amount
        return amount, 1.0

    save(portfolio_data)


def sell_position(account: str, ticker: str, shares: float, price: float):
    """Sells stock and puts proceeds into the CAD cash balance."""
    portfolio_data = load()
    account = account.upper()
    ticker = ticker.upper()

    holdings = portfolio_data["accounts"].get(account, {}).get("holdings", {})

    if ticker not in holdings or holdings[ticker]["shares"] < shares:
        raise ValueError(f"Insufficient shares of {ticker} in {account} account.")

    # Calculate proceeds
    proceeds = shares * price
    rate = 1.0

    if account == "USD":
        rate = data_client.get_usd_to_cad()
        final_proceeds = proceeds * rate
    else:
        final_proceeds = proceeds

    # Update holdings
    holdings[ticker]["shares"] -= shares
    if holdings[ticker]["shares"] <= 0:
        del holdings[ticker]

    # Add to CAD cash bucket
    portfolio_data["accounts"]["CAD"]["cash"] = (
        portfolio_data["accounts"].get("CAD", {}).get("cash", 0.0) + final_proceeds
    )

    save(portfolio_data)
    return final_proceeds, rate


def update_cash(account: str, amount: float):
    """Sets the cash balance for a specific account."""
    portfolio_data = load()
    portfolio_data, account = ensure_account_exists(portfolio_data, account)

    portfolio_data["accounts"][account]["cash"] = float(amount)
    save(portfolio_data)


def get_cash(account: str) -> float:
    """Returns the available buying power for an account."""
    portfolio_data = load()
    return portfolio_data.get("accounts", {}).get(account.upper(), {}).get("cash", 0.0)


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

    # Calculate proceeds
    proceeds = shares * price

    if account == "USD":
        portfolio_data["accounts"]["USD"]["cash"] = (
            portfolio_data["accounts"].get("USD", {}).get("cash", 0.0) - proceeds
        )
    else:
        portfolio_data["accounts"]["CAD"]["cash"] = (
            portfolio_data["accounts"].get("CAD", {}).get("cash", 0.0) - proceeds
        )

    save(portfolio_data)


def get_account_holdings(account: str):
    """Returns the holdings for a specific account."""
    portfolio_data = load()
    account = account.upper()
    return portfolio_data.get("accounts", {}).get(account, {}).get("holdings", {})


HISTORY_FILE = PORTFOLIO_FILE.parent / "history.json"


def log_net_worth():
    """Calculates total CAD net worth and logs it for the current date."""
    portfolio_data = load()
    accounts = portfolio_data.get("accounts", {})
    fx_rate = data_client.get_usd_to_cad()

    total_net_worth_cad = 0.0

    for acc_name, acc_data in accounts.items():
        multiplier = fx_rate if acc_name == "USD" else 1.0
        total_net_worth_cad += acc_data.get("cash", 0.0) * multiplier

        for ticker, holding in acc_data.get("holdings", {}).items():
            live_price = data_client.get_current_price(ticker)
            if live_price > 0:
                total_net_worth_cad += (holding["shares"] * live_price) * multiplier

    # Load existing history
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = {}

    # Update today's value (overwrites if run multiple times in one day)
    today_str = date.today().isoformat()
    history[today_str] = round(total_net_worth_cad, 2)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def get_history() -> dict:
    """Returns the net worth history."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}
