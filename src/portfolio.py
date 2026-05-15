from __future__ import annotations

from datetime import date
from typing import Optional

from src import data_client
from src.database import (
    init_db,
    Account,
    Holding,
    NetWorthSnapshot,
)


def load() -> dict:
    """Returns the full portfolio dict (JSON-compatible structure for backward compat)."""
    init_db()
    result = {"accounts": {}}
    for acc in Account.select():
        acc_data = {"holdings": {}, "cash": acc.cash, "initial_cash": acc.initial_cash}
        for h in acc.holdings:
            acc_data["holdings"][h.ticker] = {"shares": h.shares, "avg_price": h.avg_price}
        result["accounts"][acc.name] = acc_data
    if not result["accounts"]:
        for name in ["USD", "CAD"]:
            result["accounts"][name] = {"holdings": {}, "cash": 0.0, "initial_cash": 0.0}
    return result


def save(data: dict):
    """Pushes a full dict back into the database (used for backward compat)."""
    init_db()
    for acc_name, acc_data in data.get("accounts", {}).items():
        account, _ = Account.get_or_create(
            name=acc_name.upper(),
            defaults={"cash": acc_data.get("cash", 0.0), "initial_cash": acc_data.get("initial_cash", 0.0)},
        )
        account.cash = acc_data.get("cash", 0.0)
        account.initial_cash = acc_data.get("initial_cash", 0.0)
        account.save()

        existing = {h.ticker: h for h in account.holdings}
        seen = set()
        for ticker, h_data in acc_data.get("holdings", {}).items():
            seen.add(ticker)
            if ticker in existing:
                h = existing[ticker]
                h.shares = h_data["shares"]
                h.avg_price = h_data["avg_price"]
                h.save()
            else:
                Holding.create(account=account, ticker=ticker, shares=h_data["shares"], avg_price=h_data["avg_price"])
        for ticker in set(existing) - seen:
            existing[ticker].delete_instance()


def ensure_account_exists(portfolio_data: dict, account_name: str) -> tuple[dict, str]:
    """Patches a dict to ensure an account exists (backward compat wrapper)."""
    account_name = account_name.upper()
    if "accounts" not in portfolio_data:
        portfolio_data["accounts"] = {}
    if account_name not in portfolio_data["accounts"]:
        portfolio_data["accounts"][account_name] = {"holdings": {}, "cash": 0.0, "initial_cash": 0.0}
    if "initial_cash" not in portfolio_data["accounts"][account_name]:
        portfolio_data["accounts"][account_name]["initial_cash"] = 0.0

    # Ensure DB account also exists
    Account.get_or_create(name=account_name, defaults={"cash": 0.0, "initial_cash": 0.0})
    return portfolio_data, account_name


def set_initial_cash(account: str, amount: float):
    Account.get_or_create(name=account.upper(), defaults={"cash": 0.0, "initial_cash": 0.0})
    Account.update(initial_cash=float(amount)).where(Account.name == account.upper()).execute()


def deposit_cash(amount: float, currency: str) -> tuple[float, float]:
    Account.get_or_create(name="CAD", defaults={"cash": 0.0, "initial_cash": 0.0})
    if currency.upper() == "USD":
        rate = data_client.get_usd_to_cad()
        converted = amount * rate
        Account.update(cash=Account.cash + converted).where(Account.name == "CAD").execute()
        return converted, rate
    else:
        Account.update(cash=Account.cash + amount).where(Account.name == "CAD").execute()
        return amount, 1.0


def sell_position(account: str, ticker: str, shares: float, price: float) -> tuple[float, float]:
    account = account.upper()
    ticker = ticker.upper()

    try:
        acc = Account.get(Account.name == account)
    except Account.DoesNotExist:
        raise ValueError(f"Account '{account}' not found.")

    try:
        holding = Holding.get(account=acc, ticker=ticker)
    except Holding.DoesNotExist:
        raise ValueError(f"Position {ticker} not found in {account} account.")

    if holding.shares < shares:
        raise ValueError(f"Insufficient shares of {ticker} in {account} account.")

    proceeds = shares * price
    rate = data_client.get_usd_to_cad() if account == "USD" else 1.0
    final_proceeds = proceeds * rate

    holding.shares -= shares
    if holding.shares <= 0:
        holding.delete_instance()
    else:
        holding.save()

    Account.update(cash=Account.cash + final_proceeds).where(Account.name == "CAD").execute()

    return final_proceeds, rate


def update_cash(account: str, amount: float):
    Account.get_or_create(name=account.upper(), defaults={"cash": 0.0, "initial_cash": 0.0})
    Account.update(cash=float(amount)).where(Account.name == account.upper()).execute()


def get_cash(account: str) -> float:
    try:
        acc = Account.get(Account.name == account.upper())
        return acc.cash
    except Account.DoesNotExist:
        return 0.0


def add_position(account: str, ticker: str, shares: float, price: float):
    account = account.upper()
    ticker = ticker.upper()
    acc, _ = Account.get_or_create(name=account, defaults={"cash": 0.0, "initial_cash": 0.0})

    holding, created = Holding.get_or_create(
        account=acc,
        ticker=ticker,
        defaults={"shares": shares, "avg_price": price},
    )
    if not created:
        total_shares = holding.shares + shares
        total_cost = (holding.shares * holding.avg_price) + (shares * price)
        holding.shares = total_shares
        holding.avg_price = total_cost / total_shares
        holding.save()

    proceeds = shares * price
    Account.update(cash=Account.cash - proceeds).where(Account.name == account).execute()


def remove_position(account: str, ticker: str, shares: Optional[float] = None):
    account = account.upper()
    ticker = ticker.upper()

    try:
        acc = Account.get(Account.name == account)
    except Account.DoesNotExist:
        raise ValueError(f"Account '{account}' not found.")

    try:
        holding = Holding.get(account=acc, ticker=ticker)
    except Holding.DoesNotExist:
        raise ValueError(f"Position {ticker} not found in {account} account.")

    if shares is None or shares >= holding.shares:
        holding.delete_instance()
    else:
        holding.shares -= shares
        holding.save()


def get_account_holdings(account: str) -> dict:
    try:
        acc = Account.get(Account.name == account.upper())
    except Account.DoesNotExist:
        return {}
    return {h.ticker: {"shares": h.shares, "avg_price": h.avg_price} for h in acc.holdings}


def log_net_worth():
    accounts = Account.select()
    fx_rate = data_client.get_usd_to_cad()
    total = 0.0

    for acc in accounts:
        multiplier = fx_rate if acc.name == "USD" else 1.0
        total += acc.cash * multiplier
        for h in acc.holdings:
            live_price, _ = data_client.get_current_price(h.ticker)
            if live_price > 0:
                total += (h.shares * live_price) * multiplier

    today_str = date.today()
    NetWorthSnapshot.get_or_create(date=today_str, defaults={"value": round(total, 2)})
    NetWorthSnapshot.update(value=round(total, 2)).where(NetWorthSnapshot.date == today_str).execute()


def get_history() -> dict:
    return {str(s.date): s.value for s in NetWorthSnapshot.select().order_by(NetWorthSnapshot.date)}
