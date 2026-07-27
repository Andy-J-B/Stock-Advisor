"""
src/ticker_map.py

Maps Canadian-listed tickers (.NE CDRs, .TO TSX) to their US equivalents
so that Alpha Vantage API calls (news, overview, pricing) return real data
instead of zeros or empty results.

yfinance works natively with Canadian tickers, but Alpha Vantage only
supports US-listed symbols.
"""

from __future__ import annotations

import logging
from typing import Dict

import yfinance as yf

from src.database import cache_get, cache_set

log = logging.getLogger(__name__)

# Alpha Vantage / yfinance US exchange codes
_US_EXCHANGES = frozenset({"NMS", "NYQ", "PCX", "BTS", "PSE", "NCM", "NGM"})

# Known edge cases where the base ticker doesn't map to the US listing.
# These are CDRs whose base ticker either doesn't exist or resolves to
# the wrong security on US exchanges.
_CDR_TO_US: Dict[str, str] = {
    "VISA.NE": "V",
    "BRK.TO": "BRK-B",
}

# Cache ticker mappings for 30 days — they rarely change.
TTL_TICKER_MAP = 86400 * 30

_CANADIAN_SUFFIXES = (".NE", ".TO")


def _is_canadian(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in _CANADIAN_SUFFIXES)


def _is_us_ticker(ticker: str) -> bool:
    """Heuristic: a ticker is US if it has no .NE/.TO suffix."""
    return not _is_canadian(ticker)


def resolve_us_ticker(ticker: str) -> str:
    """
    Return the US-listed equivalent of *ticker* for Alpha Vantage calls.

    Resolution order:
      1. US tickers (no .NE/.TO suffix) pass through unchanged.
      2. Static lookup for known edge cases (VISA.NE → V, BRK.TO → BRK-B).
      3. Strip suffix and check if the base ticker trades on a US exchange.
      4. If no US equivalent is found, return the original ticker unchanged.

    Results are cached in the database so repeated calls are instant.
    """
    ticker = ticker.upper().strip()

    # Fast path — already a US ticker
    if _is_us_ticker(ticker):
        return ticker

    # Check cache (including cached negative results)
    cache_key = f"ticker_map:{ticker}"
    cached = cache_get(cache_key, TTL_TICKER_MAP)
    if cached is not None:
        return cached

    # Step 1: static map for known CDRs
    if ticker in _CDR_TO_US:
        us = _CDR_TO_US[ticker]
        cache_set(cache_key, us)
        log.debug("ticker_map: %s → %s (static)", ticker, us)
        return us

    # Step 2: strip suffix, try base ticker on US exchanges
    base = ticker.split(".")[0]
    try:
        info = yf.Ticker(base).info
        exchange = info.get("exchange", "")
        if exchange in _US_EXCHANGES:
            cache_set(cache_key, base)
            log.debug("ticker_map: %s → %s (base, exchange=%s)", ticker, base, exchange)
            return base
    except Exception as exc:
        log.debug("ticker_map: yfinance lookup failed for %s: %s", base, exc)

    # No US equivalent — cache and return original
    cache_set(cache_key, ticker)
    log.debug("ticker_map: %s → %s (no US equivalent)", ticker, ticker)
    return ticker


def resolve_tickers(tickers: list[str]) -> Dict[str, str]:
    """
    Batch-resolve a list of tickers to their US equivalents.

    Returns a dict mapping each input ticker to its resolved US ticker.
    Tickrs without a .NE/.TO suffix map to themselves.
    """
    result: Dict[str, str] = {}
    for t in tickers:
        t_upper = t.upper().strip()
        result[t_upper] = resolve_us_ticker(t_upper)
    return result
