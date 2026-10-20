from __future__ import annotations

from datetime import date
from typing import Any

from src import data_client
from src.repositories.sqlite_repo import SqlitePortfolioRepo, SqliteMarketRepo


class PortfolioService:
    def __init__(self, portfolio_repo: SqlitePortfolioRepo | None = None, market_repo: SqliteMarketRepo | None = None):
        self.portfolio_repo = portfolio_repo or SqlitePortfolioRepo()
        self.market_repo = market_repo or SqliteMarketRepo()

    def snapshot(self) -> dict[str, Any]:
        from src import portfolio
        return portfolio.snapshot()

    def get_accounts(self) -> list[dict[str, Any]]:
        return self.portfolio_repo.get_accounts()

    def get_holdings(self, account: str | None = None) -> list[dict[str, Any]]:
        return self.portfolio_repo.get_holdings(account)

    def ensure_price_bars(self, tickers: list[str], period: str = "1y") -> int:
        from src import data_client
        total = 0
        for t in set([x.upper() for x in tickers if x]):
            try:
                hist = data_client.get_price_history(t, period=period)
                if hist.empty:
                    continue
                bars = []
                for idx, row in hist.iterrows():
                    try:
                        d = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])
                    except Exception:
                        continue
                    bars.append({
                        "ticker": t,
                        "date": d,
                        "open": float(row.get("Open")) if row.get("Open") is not None else None,
                        "high": float(row.get("High")) if row.get("High") is not None else None,
                        "low": float(row.get("Low")) if row.get("Low") is not None else None,
                        "close": float(row.get("Close")) if row.get("Close") is not None else None,
                        "adj_close": float(row.get("Close")) if row.get("Close") is not None else None,
                        "volume": float(row.get("Volume")) if row.get("Volume") is not None else None,
                        "source": "yfinance",
                    })
                total += self.market_repo.save_price_bars(bars)
            except Exception:
                continue
        return total
