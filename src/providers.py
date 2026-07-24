"""
DataProvider protocol and adapters for data sources.

Both ``data_client`` (yfinance/FMP) and ``alpha_vantage`` provide overlapping
market data.  This module defines a common interface so code can accept either
source without knowing which one it's using.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class DataProvider(Protocol):
    """Common interface satisfied by both yfinance and Alpha Vantage adapters."""

    def get_price(self, ticker: str) -> Tuple[float, float]:
        """Return (current_price, previous_close) for *ticker*."""
        ...

    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Return a dict of fundamental info for *ticker*."""
        ...

    def get_macro_news(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return a list of macro-level news headline dicts."""
        ...

    def get_ticker_news(self, ticker: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Return a list of ticker-specific news headline dicts."""
        ...


class YFinanceAdapter:
    """Wraps ``src.data_client`` behind the DataProvider protocol."""

    def get_price(self, ticker: str) -> Tuple[float, float]:
        from src import data_client
        return data_client.get_current_price(ticker)

    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        from src import data_client
        return data_client.get_ticker_info(ticker)

    def get_macro_news(self, limit: int = 5) -> List[Dict[str, Any]]:
        from src import data_client
        return data_client.get_macro_news()

    def get_ticker_news(self, ticker: str, limit: int = 3) -> List[Dict[str, Any]]:
        from src import data_client
        return data_client.get_ticker_news(ticker, limit)


class AlphaVantageAdapter:
    """Wraps ``src.alpha_vantage`` behind the DataProvider protocol."""

    def get_price(self, ticker: str) -> Tuple[float, float]:
        from src import alpha_vantage
        return alpha_vantage.get_daily_price(ticker)

    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        from src import alpha_vantage
        return alpha_vantage.get_company_overview(ticker)

    def get_macro_news(self, limit: int = 5) -> List[Dict[str, Any]]:
        from src import alpha_vantage
        return alpha_vantage.get_macro_news(limit)

    def get_ticker_news(self, ticker: str, limit: int = 3) -> List[Dict[str, Any]]:
        from src import alpha_vantage
        return alpha_vantage.get_ticker_news(ticker, limit)
