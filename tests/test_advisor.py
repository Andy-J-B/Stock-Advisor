"""Tests for src.advisor.evaluate_portfolio portfolio valuation."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def portfolio():
    return {
        "accounts": {
            "CAD": {
                "holdings": {"MSFT.NE": {"shares": 10, "avg_price": 30.0}},
                "cash": 1000.0,
            },
            "USD": {
                "holdings": {},
                "cash": 2000.0,
            },
        }
    }


def test_evaluate_portfolio_values_holdings_in_cad(portfolio):
    """CDR holdings must be valued at the actual CDR price, not the US price."""
    from src import advisor

    settings = {"risk_allocation": {"conservative": 30, "moderate": 30, "aggressive": 40}}
    captured: dict = {}

    def fake_gemini(prompt):
        captured["prompt"] = prompt
        return None

    with patch("src.advisor.av_get_macro_news", return_value=[]), \
         patch("src.advisor.resolve_tickers", return_value={"MSFT.NE": "MSFT"}), \
         patch("src.advisor.av_get_ticker_news", return_value=[]), \
         patch("src.advisor.get_company_overview", return_value={"Sector": "TECHNOLOGY"}), \
         patch("src.advisor.data_client.get_usd_to_cad", return_value=1.4), \
         patch("src.advisor.data_client.get_current_prices_batch",
               return_value={"MSFT.NE": (32.0, 31.0)}), \
         patch("src.advisor._gemini_generate", side_effect=fake_gemini):
        advisor.evaluate_portfolio(portfolio, settings)

    prompt = captured["prompt"]
    # 10 CDR shares * 32.00 CAD each = 320.00 (NOT 10 * MSFT US price ~465 USD)
    assert "**TOTAL HOLDINGS VALUE:** $320.00" in prompt
    # cash = 1000 CAD + 2000 USD * 1.4 FX = 3800 CAD
    assert "**TOTAL CASH (Buying Power):** $3,800.00" in prompt
    assert "TECHNOLOGY" in prompt


def test_evaluate_portfolio_falls_back_without_gemini(portfolio):
    """Without Gemini the command still returns a structural Panel, not a crash."""
    from src import advisor

    settings = {"risk_allocation": {}}
    with patch("src.advisor.av_get_macro_news", return_value=[]), \
         patch("src.advisor.resolve_tickers", return_value={"MSFT.NE": "MSFT"}), \
         patch("src.advisor.av_get_ticker_news", return_value=[]), \
         patch("src.advisor.get_company_overview", return_value={}), \
         patch("src.advisor.data_client.get_usd_to_cad", return_value=1.4), \
         patch("src.advisor.data_client.get_current_prices_batch",
               return_value={"MSFT.NE": (32.0, 31.0)}), \
         patch("src.advisor._gemini_generate", return_value=None):
        result = advisor.evaluate_portfolio(portfolio, settings)

    assert getattr(result, "title", None) == "Portfolio Analysis"
