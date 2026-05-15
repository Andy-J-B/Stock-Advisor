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


def test_empty_portfolio():
    p = portfolio.load()
    assert "accounts" in p
    assert "USD" in p["accounts"]
    assert "CAD" in p["accounts"]
    assert p["accounts"]["USD"]["holdings"] == {}
    assert p["accounts"]["CAD"]["holdings"] == {}
    assert p["accounts"]["USD"]["cash"] == 0.0
    assert p["accounts"]["CAD"]["cash"] == 0.0


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
    assert h["avg_price"] == 140.0  # (10*150 + 10*130) / 20


def test_add_position_cad_account():
    portfolio.add_position("CAD", "VFV.TO", 5, 100.0)
    p = portfolio.load()
    assert "VFV.TO" in p["accounts"]["CAD"]["holdings"]
    h = p["accounts"]["CAD"]["holdings"]["VFV.TO"]
    assert h["shares"] == 5
    assert h["avg_price"] == 100.0


def test_deposit_cash_cad():
    amt, rate = portfolio.deposit_cash(1000, "CAD")
    assert amt == 1000.0
    assert rate == 1.0
    p = portfolio.load()
    assert p["accounts"]["CAD"]["cash"] == 1000.0


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


def test_remove_position_not_found():
    with pytest.raises(ValueError, match="not found"):
        portfolio.remove_position("USD", "NONEXIST")


def test_set_initial_cash():
    portfolio.set_initial_cash("USD", 5000.0)
    p = portfolio.load()
    assert p["accounts"]["USD"]["initial_cash"] == 5000.0


def test_update_cash():
    portfolio.update_cash("USD", 10000.0)
    p = portfolio.load()
    assert p["accounts"]["USD"]["cash"] == 10000.0


def test_get_cash():
    portfolio.update_cash("CAD", 2500.0)
    assert portfolio.get_cash("CAD") == 2500.0


def test_get_cash_empty():
    assert portfolio.get_cash("NONEXIST") == 0.0


def test_ensure_account_exists():
    p = portfolio.load()
    portfolio.ensure_account_exists(p, "EUR")
    assert "EUR" in p["accounts"]
    assert p["accounts"]["EUR"]["cash"] == 0.0


def test_log_net_worth():
    portfolio.add_position("CAD", "VFV.TO", 10, 50.0)
    portfolio.update_cash("CAD", 1000.0)
    portfolio.log_net_worth()
    history = portfolio.get_history()
    assert len(history) >= 1


def test_get_history_empty():
    assert portfolio.get_history() == {}


def test_save_and_load():
    data = {"accounts": {"USD": {"holdings": {}, "cash": 500.0, "initial_cash": 1000.0}}}
    portfolio.save(data)
    p = portfolio.load()
    assert p["accounts"]["USD"]["cash"] == 500.0
    assert p["accounts"]["USD"]["initial_cash"] == 1000.0
