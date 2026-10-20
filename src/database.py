from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from peewee import (
    SqliteDatabase,
    Model,
    CharField,
    FloatField,
    DateTimeField,
    DateField,
    ForeignKeyField,
    TextField,
    IntegerField,
    BooleanField,
)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

db = SqliteDatabase(DATA_DIR / "portfolio.db")


class BaseModel(Model):
    class Meta:
        database = db


class Account(BaseModel):
    name = CharField(unique=True, max_length=10)
    cash = FloatField(default=0.0)
    initial_cash = FloatField(default=0.0)
    currency = CharField(max_length=3, default="CAD")


class Holding(BaseModel):
    account = ForeignKeyField(Account, backref="holdings")
    ticker = CharField(max_length=20)
    shares = FloatField(default=0.0)
    avg_price = FloatField(default=0.0)

    class Meta:
        indexes = ((("account", "ticker"), True),)


class Transaction(BaseModel):
    account = ForeignKeyField(Account, backref="transactions")
    ticker = CharField(max_length=20, null=True)
    type = CharField(max_length=20)
    shares = FloatField(default=0.0)
    price = FloatField(default=0.0)
    currency = CharField(max_length=3, default="USD")
    fee = FloatField(default=0.0)
    timestamp = DateTimeField(default=datetime.now)
    notes = TextField(null=True)


class PositionLot(BaseModel):
    """Lot-level tracking for accurate cost basis."""
    account = ForeignKeyField(Account, backref="lots")
    holding = ForeignKeyField(Holding, backref="lots", null=True)
    ticker = CharField(max_length=20)
    quantity = FloatField(default=0.0)
    remaining_quantity = FloatField(default=0.0)
    cost_price = FloatField(default=0.0)
    cost_currency = CharField(max_length=3, default="USD")
    acquired_at = DateTimeField(default=datetime.now)
    fee = FloatField(default=0.0)
    lot_source = CharField(max_length=20, default="buy")
    is_closed = BooleanField(default=False)

    class Meta:
        indexes = ((("ticker", "account", "acquired_at"), False),)


class PriceBar(BaseModel):
    """Historical OHLCV price bars."""
    ticker = CharField(max_length=20)
    date = DateField()
    open = FloatField(null=True)
    high = FloatField(null=True)
    low = FloatField(null=True)
    close = FloatField(null=True)
    adj_close = FloatField(null=True)
    volume = FloatField(null=True)
    source = CharField(max_length=20, default="yfinance")

    class Meta:
        indexes = ((("ticker", "date"), True),)


class CorporateAction(BaseModel):
    ticker = CharField(max_length=20)
    ex_date = DateField()
    type = CharField(max_length=20)
    ratio = FloatField(null=True)
    amount = FloatField(null=True)
    currency = CharField(max_length=3, default="USD")
    processed = BooleanField(default=False)


class Dividend(BaseModel):
    account = ForeignKeyField(Account, backref="dividends")
    ticker = CharField(max_length=20)
    ex_date = DateField(null=True)
    pay_date = DateField(null=True)
    amount_per_share = FloatField(default=0.0)
    total_amount = FloatField(default=0.0)
    currency = CharField(max_length=3, default="USD")
    received_at = DateTimeField(default=datetime.now)


class RealizedPnL(BaseModel):
    account = ForeignKeyField(Account, backref="realized_pnl")
    ticker = CharField(max_length=20)
    lot = ForeignKeyField(PositionLot, backref="realized_trades", null=True)
    quantity = FloatField(default=0.0)
    cost_basis = FloatField(default=0.0)
    proceeds = FloatField(default=0.0)
    realized_gain_loss = FloatField(default=0.0)
    currency = CharField(max_length=3, default="USD")
    realized_at = DateTimeField(default=datetime.now)


class PositionPerformanceDaily(BaseModel):
    account = ForeignKeyField(Account, backref="daily_perf")
    ticker = CharField(max_length=20)
    date = DateField()
    shares = FloatField(default=0.0)
    price_close = FloatField(default=0.0)
    value_local = FloatField(default=0.0)
    value_cad = FloatField(default=0.0)
    cost_basis_cad = FloatField(default=0.0)
    unrealized_pnl_cad = FloatField(default=0.0)
    realized_pnl_cad = FloatField(default=0.0)
    day_pnl_cad = FloatField(default=0.0)
    weight = FloatField(default=0.0)

    class Meta:
        indexes = ((("account", "ticker", "date"), True),)


class NetWorthSnapshot(BaseModel):
    date = DateField(unique=True)
    value = FloatField()
    invested = FloatField(default=0.0)
    cash = FloatField(default=0.0)
    cost_basis = FloatField(default=0.0)
    day_change = FloatField(default=0.0)
    return_pct = FloatField(default=0.0)
    all_time_pct = FloatField(default=0.0)
    fx_usd_cad = FloatField(default=1.35)


class FxRate(BaseModel):
    date = DateField()
    pair = CharField(max_length=10)
    rate = FloatField(default=1.0)

    class Meta:
        indexes = ((("date", "pair"), True),)


class Recommendation(BaseModel):
    ticker = CharField(max_length=20)
    run_date = DateField()
    action = CharField(max_length=20)
    score = FloatField(default=0.0)
    rationale = TextField()
    constraints = TextField(null=True)
    executed = BooleanField(default=False)
    outcome_1d = TextField(null=True)
    outcome_7d = TextField(null=True)
    outcome_30d = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)


class Setting(BaseModel):
    key = CharField(unique=True, max_length=50)
    value = TextField()


class Watchlist(BaseModel):
    ticker = CharField(unique=True, max_length=20)
    added_at = DateTimeField(default=datetime.now)


class CacheEntry(BaseModel):
    key = CharField(unique=True, max_length=256)
    value = TextField()
    fetched_at = DateTimeField(default=datetime.now)


def init_db(skip_migration: bool = False):
    if db.is_closed():
        db.connect()
    db.create_tables([
        Account, Holding, Transaction, PositionLot, PriceBar,
        CorporateAction, Dividend, RealizedPnL, PositionPerformanceDaily,
        NetWorthSnapshot, FxRate, Recommendation, Setting, Watchlist, CacheEntry
    ], safe=True)
    if not skip_migration:
        _migrate_from_json()
        _migrate_holdings_to_lots()


def _migrate_from_json():
    """One-time migration of existing JSON data into SQLite."""
    if db.database != str(DATA_DIR / "portfolio.db"):
        return
    if Account.select().count() > 0:
        return

    portfolio_file = DATA_DIR / "portfolio.json"
    settings_file = DATA_DIR / "settings.json"

    if portfolio_file.exists():
        with open(portfolio_file) as f:
            data = json.load(f)

        for acc_name, acc_data in data.get("accounts", {}).items():
            account = Account.get_or_create(
                name=acc_name.upper(),
                defaults={
                    "cash": acc_data.get("cash", 0.0),
                    "initial_cash": acc_data.get("initial_cash", 0.0),
                },
            )[0]
            if account.cash == 0.0 and account.initial_cash == 0.0:
                account.cash = acc_data.get("cash", 0.0)
                account.initial_cash = acc_data.get("initial_cash", 0.0)
                account.save()

            for ticker, h in acc_data.get("holdings", {}).items():
                Holding.get_or_create(
                    account=account,
                    ticker=ticker.upper(),
                    defaults={"shares": h["shares"], "avg_price": h["avg_price"]},
                )

        # Migrate net worth history
        history_file = DATA_DIR / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
            for date_str, val in history.items():
                NetWorthSnapshot.get_or_create(
                    date=date.fromisoformat(date_str),
                    defaults={"value": val},
                )

    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
        for key, val in settings.items():
            if isinstance(val, dict):
                Setting.get_or_create(key=key, defaults={"value": json.dumps(val)})
            else:
                Setting.get_or_create(key=key, defaults={"value": str(val)})


def _migrate_holdings_to_lots():
    """Create lots from existing holdings if none exist."""
    try:
        if PositionLot.select().count() > 0:
            return
        for acc in Account.select():
            for h in acc.holdings:
                if h.shares <= 0:
                    continue
                lot, created = PositionLot.get_or_create(
                    account=acc,
                    holding=h,
                    ticker=h.ticker,
                    quantity=h.shares,
                    defaults={
                        "remaining_quantity": h.shares,
                        "cost_price": h.avg_price,
                        "cost_currency": "USD" if acc.name == "USD" else "CAD",
                        "acquired_at": datetime.now(),
                        "lot_source": "migration",
                    },
                )
                if not created and lot.remaining_quantity == 0:
                    lot.remaining_quantity = h.shares
                    lot.quantity = h.shares
                    lot.cost_price = h.avg_price
                    lot.save()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cache helpers – generic key/value cache backed by SQLite
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Watchlist helpers
# ---------------------------------------------------------------------------

def get_watchlist() -> list[str]:
    """Return the list of tickers on the watchlist."""
    return [w.ticker for w in Watchlist.select().order_by(Watchlist.added_at)]


def add_to_watchlist(ticker: str) -> bool:
    """Add a ticker to the watchlist. Returns True if added, False if already present."""
    _, created = Watchlist.get_or_create(ticker=ticker.upper().strip())
    return created


def remove_from_watchlist(ticker: str) -> bool:
    """Remove a ticker from the watchlist. Returns True if removed."""
    try:
        w = Watchlist.get(Watchlist.ticker == ticker.upper().strip())
        w.delete_instance()
        return True
    except Watchlist.DoesNotExist:
        return False


# ---------------------------------------------------------------------------
# Cache helpers – generic key/value cache backed by SQLite
# ---------------------------------------------------------------------------

def cache_get(key: str, ttl_seconds: int) -> Optional[Any]:
    """Return cached value if fresh, else None."""
    try:
        entry = CacheEntry.get(CacheEntry.key == key)
        age = (datetime.now() - entry.fetched_at).total_seconds()
        if age < ttl_seconds:
            return json.loads(entry.value)
    except (CacheEntry.DoesNotExist, json.JSONDecodeError):
        pass
    return None


def cache_set(key: str, value: Any) -> None:
    """Store a JSON-serializable value in the cache (upsert)."""
    serialized = json.dumps(value)
    entry, created = CacheEntry.get_or_create(
        key=key,
        defaults={"value": serialized, "fetched_at": datetime.now()},
    )
    if not created:
        entry.value = serialized
        entry.fetched_at = datetime.now()
        entry.save()


def _migrate_holdings_to_lots():
    """Create lots from existing holdings if none exist."""
    try:
        if PositionLot.select().count() > 0:
            return
        for acc in Account.select():
            for h in acc.holdings:
                if h.shares <= 0:
                    continue
                PositionLot.get_or_create(
                    account=acc,
                    holding=h,
                    ticker=h.ticker,
                    quantity=h.shares,
                    defaults={
                        "remaining_quantity": h.shares,
                        "cost_price": h.avg_price,
                        "cost_currency": getattr(acc, "currency", "USD") if acc.name == "USD" else "CAD",
                        "acquired_at": datetime.now(),
                        "lot_source": "migration",
                    },
                )
    except Exception:
        pass
