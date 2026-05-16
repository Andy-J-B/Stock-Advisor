"""Tests for the database module."""

import pytest
from src.database import init_db, db, Account, Holding, NetWorthSnapshot, Setting


@pytest.fixture(autouse=True)
def memory_db():
    """Set up a fresh in-memory database for each test."""
    if not db.is_closed():
        db.close()
    original = db.database
    db.init(":memory:")
    db.connect()
    init_db(skip_migration=True)
    yield
    db.drop_tables([Account, Holding, NetWorthSnapshot, Setting])
    db.close()
    db.init(original)


def test_tables_created():
    assert Account.table_exists()
    assert Holding.table_exists()
    assert NetWorthSnapshot.table_exists()
    assert Setting.table_exists()


def test_init_db_idempotent():
    init_db(skip_migration=True)
    init_db(skip_migration=True)
    assert Account.table_exists()


def test_account_create_and_retrieve():
    acc = Account.create(name="TEST", cash=100.0, initial_cash=500.0)
    assert acc.name == "TEST"
    assert acc.cash == 100.0
    assert acc.initial_cash == 500.0


def test_account_unique_name():
    Account.create(name="UNIQUE", cash=0.0, initial_cash=0.0)
    with pytest.raises(Exception):
        Account.create(name="UNIQUE", cash=0.0, initial_cash=0.0)


def test_holding_relation():
    acc = Account.create(name="HOLD_TEST", cash=0.0, initial_cash=0.0)
    h = Holding.create(account=acc, ticker="AAPL", shares=10, avg_price=150.0)
    assert h.ticker == "AAPL"
    assert h.shares == 10
    assert h.avg_price == 150.0
    assert h.account.name == "HOLD_TEST"


def test_holding_unique_per_account():
    acc = Account.create(name="DUPE", cash=0.0, initial_cash=0.0)
    Holding.create(account=acc, ticker="AAPL", shares=10, avg_price=150.0)
    with pytest.raises(Exception):
        Holding.create(account=acc, ticker="AAPL", shares=5, avg_price=200.0)


def test_same_ticker_different_accounts():
    a1 = Account.create(name="A1", cash=0.0, initial_cash=0.0)
    a2 = Account.create(name="A2", cash=0.0, initial_cash=0.0)
    Holding.create(account=a1, ticker="AAPL", shares=10, avg_price=150.0)
    Holding.create(account=a2, ticker="AAPL", shares=5, avg_price=200.0)
    assert Holding.select().where(Holding.ticker == "AAPL").count() == 2


def test_net_worth_snapshot():
    from datetime import date
    ns = NetWorthSnapshot.create(date=date.today(), value=50000.0)
    assert ns.value == 50000.0


def test_setting_create():
    s = Setting.create(key="test_key", value="test_value")
    assert s.key == "test_key"
    assert s.value == "test_value"


def test_setting_unique_key():
    Setting.create(key="dup", value="v1")
    with pytest.raises(Exception):
        Setting.create(key="dup", value="v2")


def test_backref_holdings():
    acc = Account.create(name="BACKREF", cash=0.0, initial_cash=0.0)
    Holding.create(account=acc, ticker="A", shares=1, avg_price=10.0)
    Holding.create(account=acc, ticker="B", shares=2, avg_price=20.0)
    assert acc.holdings.count() == 2
    tickers = {h.ticker for h in acc.holdings}
    assert tickers == {"A", "B"}


def test_delete_account_leaves_orphan_holdings():
    """Deleting an Account does not cascade-delete its Holdings by default in peewee."""
    acc = Account.create(name="ORPHAN", cash=0.0, initial_cash=0.0)
    Holding.create(account=acc, ticker="LEFT_BEHIND", shares=1, avg_price=10.0)
    acc.delete_instance()
    assert Holding.select().where(Holding.ticker == "LEFT_BEHIND").count() == 1

