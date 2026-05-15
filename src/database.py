from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

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


def init_db():
    if db.is_closed():
        db.connect()
    db.create_tables([Account, Holding, Transaction, NetWorthSnapshot, Setting], safe=True)
    _migrate_from_json()


def _migrate_from_json():
    """One-time migration of existing JSON data into SQLite."""
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
