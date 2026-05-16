"""Tests for portfolio operations using an in-memory SQLite database."""

import pytest
from src.database import init_db, db, Account, Holding, NetWorthSnapshot
from src import portfolio


@pytest.fixture(autouse=True)
def fresh_db():
    """Use a fresh in-memory database for each test."""
    if not db.is_closed():
        db.close()
    db.init(":memory:")
    db.connect()
    init_db(skip_migration=True)
    yield
    db.drop_tables([Account, Holding, NetWorthSnapshot])
    db.close()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def test_empty_portfolio():
    p = portfolio.load()
    assert "accounts" in p
    assert "USD" in p["accounts"]
    assert "CAD" in p["accounts"]
    assert p["accounts"]["USD"]["holdings"] == {}
    assert p["accounts"]["CAD"]["holdings"] == {}
    assert p["accounts"]["USD"]["cash"] == 0.0
    assert p["accounts"]["CAD"]["cash"] == 0.0


def test_load_with_accounts_in_db():
    Account.create(name="EUR", cash=500.0, initial_cash=1000.0)
    p = portfolio.load()
    assert "EUR" in p["accounts"]
    assert p["accounts"]["EUR"]["cash"] == 500.0
    assert p["accounts"]["EUR"]["initial_cash"] == 1000.0


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def test_save_and_load():
    data = {"accounts": {"USD": {"holdings": {}, "cash": 500.0, "initial_cash": 1000.0}}}
    portfolio.save(data)
    p = portfolio.load()
    assert p["accounts"]["USD"]["cash"] == 500.0
    assert p["accounts"]["USD"]["initial_cash"] == 1000.0


def test_save_empty_accounts():
    portfolio.save({"accounts": {}})
    p = portfolio.load()
    assert "USD" in p["accounts"]


def test_save_removes_stale_holdings():
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    data = {"accounts": {"USD": {"holdings": {}, "cash": 0.0, "initial_cash": 0.0}}}
    portfolio.save(data)
    p = portfolio.load()
    assert "AAPL" not in p["accounts"]["USD"]["holdings"]


# ---------------------------------------------------------------------------
# Add position
# ---------------------------------------------------------------------------

def test_add_position():
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    p = portfolio.load()
    h = p["accounts"]["USD"]["holdings"]["AAPL"]
    assert h["shares"] == 10
    assert h["avg_price"] == 150.0


def test_add_position_average_down():
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    portfolio.add_position("USD", "AAPL", 10, 130.0)
    p = portfolio.load()
    h = p["accounts"]["USD"]["holdings"]["AAPL"]
    assert h["shares"] == 20
    assert h["avg_price"] == 140.0


def test_add_position_cad_account():
    portfolio.add_position("CAD", "VFV.TO", 5, 100.0)
    p = portfolio.load()
    assert "VFV.TO" in p["accounts"]["CAD"]["holdings"]
    h = p["accounts"]["CAD"]["holdings"]["VFV.TO"]
    assert h["shares"] == 5
    assert h["avg_price"] == 100.0


def test_add_position_deducts_cash():
    portfolio.update_cash("USD", 5000.0)
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    assert portfolio.get_cash("USD") == 3500.0


def test_add_position_to_new_account():
    portfolio.add_position("EUR", "ABC", 100, 1.0)
    p = portfolio.load()
    assert "EUR" in p["accounts"]
    assert p["accounts"]["EUR"]["holdings"]["ABC"]["shares"] == 100


# ---------------------------------------------------------------------------
# Remove position
# ---------------------------------------------------------------------------

def test_remove_position_partial():
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    portfolio.remove_position("USD", "AAPL", shares=4)
    p = portfolio.load()
    assert p["accounts"]["USD"]["holdings"]["AAPL"]["shares"] == 6


def test_remove_position_full():
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    portfolio.remove_position("USD", "AAPL")
    p = portfolio.load()
    assert "AAPL" not in p["accounts"]["USD"]["holdings"]


def test_remove_position_exact_shares():
    """Removing exactly all shares should delete the holding."""
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    portfolio.remove_position("USD", "AAPL", shares=10)
    p = portfolio.load()
    assert "AAPL" not in p["accounts"]["USD"]["holdings"]


def test_remove_position_excess_shares():
    """Removing more shares than owned should delete the holding."""
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    portfolio.remove_position("USD", "AAPL", shares=99)
    p = portfolio.load()
    assert "AAPL" not in p["accounts"]["USD"]["holdings"]


def test_remove_position_not_found():
    with pytest.raises(ValueError, match="not found"):
        portfolio.remove_position("USD", "NONEXIST")


def test_remove_from_nonexistent_account():
    with pytest.raises(ValueError, match="not found"):
        portfolio.remove_position("EUR", "AAPL")


# ---------------------------------------------------------------------------
# Deposit cash
# ---------------------------------------------------------------------------

def test_deposit_cash_cad():
    amt, rate = portfolio.deposit_cash(1000, "CAD")
    assert amt == 1000.0
    assert rate == 1.0
    p = portfolio.load()
    assert p["accounts"]["CAD"]["cash"] == 1000.0


def test_deposit_cash_multiple():
    portfolio.deposit_cash(500, "CAD")
    portfolio.deposit_cash(300, "CAD")
    assert portfolio.get_cash("CAD") == 800.0


# ---------------------------------------------------------------------------
# Sell position
# ---------------------------------------------------------------------------

def test_sell_position_reduces_holdings():
    portfolio.add_position("CAD", "VFV.TO", 10, 50.0)
    portfolio.sell_position("CAD", "VFV.TO", 3, 55.0)
    p = portfolio.load()
    assert p["accounts"]["CAD"]["holdings"]["VFV.TO"]["shares"] == 7


def test_sell_position_full_liquidates():
    portfolio.add_position("CAD", "VFV.TO", 10, 50.0)
    portfolio.sell_position("CAD", "VFV.TO", 10, 55.0)
    p = portfolio.load()
    assert "VFV.TO" not in p["accounts"]["CAD"]["holdings"]


def test_sell_insufficient_shares():
    portfolio.add_position("CAD", "VFV.TO", 10, 50.0)
    with pytest.raises(ValueError, match="Insufficient"):
        portfolio.sell_position("CAD", "VFV.TO", 99, 55.0)


def test_sell_nonexistent_ticker():
    with pytest.raises(ValueError, match="not found"):
        portfolio.sell_position("CAD", "NONEXIST", 1, 10.0)


def test_sell_nonexistent_account():
    with pytest.raises(ValueError, match="not found"):
        portfolio.sell_position("EUR", "AAPL", 1, 10.0)


# ---------------------------------------------------------------------------
# Set initial cash
# ---------------------------------------------------------------------------

def test_set_initial_cash():
    portfolio.set_initial_cash("USD", 5000.0)
    p = portfolio.load()
    assert p["accounts"]["USD"]["initial_cash"] == 5000.0


def test_set_initial_cash_twice():
    portfolio.set_initial_cash("USD", 1000.0)
    portfolio.set_initial_cash("USD", 2000.0)
    assert portfolio.load()["accounts"]["USD"]["initial_cash"] == 2000.0


def test_set_initial_cash_new_account():
    portfolio.set_initial_cash("EUR", 999.0)
    assert portfolio.load()["accounts"]["EUR"]["initial_cash"] == 999.0


# ---------------------------------------------------------------------------
# Update cash
# ---------------------------------------------------------------------------

def test_update_cash():
    portfolio.update_cash("USD", 10000.0)
    p = portfolio.load()
    assert p["accounts"]["USD"]["cash"] == 10000.0


def test_update_cash_overwrites():
    portfolio.update_cash("CAD", 100.0)
    portfolio.update_cash("CAD", 200.0)
    assert portfolio.get_cash("CAD") == 200.0


# ---------------------------------------------------------------------------
# Get cash
# ---------------------------------------------------------------------------

def test_get_cash():
    portfolio.update_cash("CAD", 2500.0)
    assert portfolio.get_cash("CAD") == 2500.0


def test_get_cash_empty():
    assert portfolio.get_cash("NONEXIST") == 0.0


# ---------------------------------------------------------------------------
# Ensure account exists
# ---------------------------------------------------------------------------

def test_ensure_account_exists():
    p = portfolio.load()
    portfolio.ensure_account_exists(p, "EUR")
    assert "EUR" in p["accounts"]
    assert p["accounts"]["EUR"]["cash"] == 0.0


def test_ensure_account_exists_idempotent():
    p = portfolio.load()
    portfolio.ensure_account_exists(p, "USD")
    assert "USD" in p["accounts"]


def test_ensure_account_exists_creates_accounts_key():
    p = {}
    portfolio.ensure_account_exists(p, "USD")
    assert "accounts" in p


def test_ensure_account_exists_adds_initial_cash():
    p = {"accounts": {"USD": {"holdings": {}, "cash": 100.0}}}
    portfolio.ensure_account_exists(p, "USD")
    assert "initial_cash" in p["accounts"]["USD"]


# ---------------------------------------------------------------------------
# Get account holdings
# ---------------------------------------------------------------------------

def test_get_account_holdings_empty():
    assert portfolio.get_account_holdings("USD") == {}


def test_get_account_holdings_populated():
    portfolio.add_position("USD", "AAPL", 10, 150.0)
    h = portfolio.get_account_holdings("USD")
    assert "AAPL" in h
    assert h["AAPL"]["shares"] == 10


def test_get_account_holdings_nonexistent():
    assert portfolio.get_account_holdings("EUR") == {}


# ---------------------------------------------------------------------------
# Log net worth / history
# ---------------------------------------------------------------------------

def test_log_net_worth():
    portfolio.add_position("CAD", "VFV.TO", 10, 50.0)
    portfolio.update_cash("CAD", 1000.0)
    portfolio.log_net_worth()
    history = portfolio.get_history()
    assert len(history) >= 1


def test_log_net_worth_updates_same_day():
    portfolio.log_net_worth()
    portfolio.log_net_worth()
    history = portfolio.get_history()
    assert len(history) == 1


def test_get_history_empty():
    assert portfolio.get_history() == {}

