from __future__ import annotations

from datetime import date, datetime
from typing import Any

from src.database import (
    init_db,
    Account,
    Holding,
    NetWorthSnapshot,
    PriceBar,
    PositionLot,
    PositionPerformanceDaily,
    Recommendation,
)


class SqlitePortfolioRepo:
    def get_accounts(self) -> list[dict[str, Any]]:
        init_db()
        return [
            {"name": a.name, "cash": a.cash, "initial_cash": a.initial_cash, "currency": getattr(a, "currency", "CAD")}
            for a in Account.select()
        ]

    def get_holdings(self, account: str | None = None) -> list[dict[str, Any]]:
        init_db()
        q = Holding.select()
        if account:
            q = q.join(Account).where(Account.name == account.upper())
        return [
            {"account": h.account.name, "ticker": h.ticker, "shares": h.shares, "avg_price": h.avg_price}
            for h in q
        ]

    def get_cash(self, account: str) -> float:
        init_db()
        try:
            a = Account.get(Account.name == account.upper())
            return float(a.cash)
        except Account.DoesNotExist:
            return 0.0

    def get_net_worth_history(self) -> dict[str, float]:
        init_db()
        return {str(s.date): s.value for s in NetWorthSnapshot.select().order_by(NetWorthSnapshot.date)}


class SqliteMarketRepo:
    def get_price_bars(self, ticker: str, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
        init_db()
        q = PriceBar.select().where(PriceBar.ticker == ticker.upper()).order_by(PriceBar.date)
        if start:
            try:
                from datetime import date as _date
                q = q.where(PriceBar.date >= _date.fromisoformat(start))
            except Exception:
                pass
        if end:
            try:
                from datetime import date as _date
                q = q.where(PriceBar.date <= _date.fromisoformat(end))
            except Exception:
                pass
        return [
            {
                "ticker": b.ticker,
                "date": str(b.date),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "adj_close": b.adj_close,
                "volume": b.volume,
                "source": b.source,
            }
            for b in q
        ]

    def save_price_bars(self, bars: list[dict[str, Any]]) -> int:
        init_db()
        count = 0
        for b in bars:
            try:
                ticker = (b.get("ticker") or "").upper()
                d = b.get("date")
                if isinstance(d, str):
                    d = date.fromisoformat(d)
                elif isinstance(d, datetime):
                    d = d.date()
                if not ticker or not d:
                    continue
                PriceBar.get_or_create(
                    ticker=ticker,
                    date=d,
                    defaults={
                        "open": b.get("open"),
                        "high": b.get("high"),
                        "low": b.get("low"),
                        "close": b.get("close"),
                        "adj_close": b.get("adj_close") or b.get("close"),
                        "volume": b.get("volume"),
                        "source": b.get("source", "yfinance"),
                    },
                )
                count += 1
            except Exception:
                continue
        return count
