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
    timestamp = DateTimeField(default=datetime.now)


class NetWorthSnapshot(BaseModel):
    date = DateField(unique=True)
    value = FloatField()


class Setting(BaseModel):
    key = CharField(unique=True, max_length=50)
    value = TextField()


class CacheEntry(BaseModel):
    key = CharField(unique=True, max_length=256)
    value = TextField()
    fetched_at = DateTimeField(default=datetime.now)


def init_db(skip_migration: bool = False):
    if db.is_closed():
        db.connect()
    db.create_tables([Account, Holding, Transaction, NetWorthSnapshot, Setting, CacheEntry], safe=True)
    if not skip_migration:
        _migrate_from_json()


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
