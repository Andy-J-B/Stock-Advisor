"""Tests for pure helper functions in main.py."""

from main import _price_color, _pct_str, _dol_str, _colorize, _build_holding_row


# ---------------------------------------------------------------------------
# _price_color
# ---------------------------------------------------------------------------

def test_price_color_positive():
    assert _price_color(5.0) == "green"


def test_price_color_zero():
    assert _price_color(0.0) == "green"


def test_price_color_negative():
    assert _price_color(-1.0) == "red"


# ---------------------------------------------------------------------------
# _pct_str
# ---------------------------------------------------------------------------

def test_pct_str_positive():
    assert _pct_str(5.5) == "+5.50%"


def test_pct_str_negative():
    assert _pct_str(-3.2) == "-3.20%"


def test_pct_str_zero():
    assert _pct_str(0.0) == "+0.00%"


# ---------------------------------------------------------------------------
# _dol_str
# ---------------------------------------------------------------------------

def test_dol_str_positive():
    assert _dol_str(1234.5) == "$1,234.50"


def test_dol_str_zero():
    assert _dol_str(0.0) == "$0.00"


def test_dol_str_large():
    assert _dol_str(1_000_000) == "$1,000,000.00"


def test_dol_str_negative():
    assert _dol_str(-500) == "$-500.00"


# ---------------------------------------------------------------------------
# _colorize
# ---------------------------------------------------------------------------

def test_colorize_positive():
    result = _colorize(5.0, "{:+.2f}%")
    assert result == "[green]+5.00%[/green]"


def test_colorize_negative():
    result = _colorize(-3.2, "{:+.2f}")
    assert result == "[red]-3.20[/red]"


def test_colorize_zero():
    result = _colorize(0.0, "{:.1f}")
    assert result == "[green]0.0[/green]"


# ---------------------------------------------------------------------------
# _build_holding_row
# ---------------------------------------------------------------------------

def test_build_holding_row_normal():
    """Normal case with live_price > 0 and prev_close available."""
    row, metrics = _build_holding_row("AAPL", 10, 150.0, 160.0, 155.0)
    assert row[0] == "AAPL"
    assert row[1] == "10"
    assert row[2] == "$150.00"
    assert metrics["cost"] == 1500.0
    assert metrics["value"] == 1600.0
    assert metrics["day_chg"] == 50.0  # (160 - 155) * 10


def test_build_holding_row_zero_price():
    """When live_price == 0, the row shows N/A and metrics report 0."""
    row, metrics = _build_holding_row("AAPL", 10, 150.0, 0.0, 155.0)
    assert "[yellow]N/A[/yellow]" in row[3] or row[3] == "[yellow]N/A[/yellow]"
    assert metrics["value"] == 0
    assert metrics["day_chg"] == 0


def test_build_holding_row_no_prev_close():
    """When prev_close == 0, day change defaults to 0."""
    row, metrics = _build_holding_row("AAPL", 10, 150.0, 155.0, 0.0)
    assert metrics["day_chg"] == 0.0


def test_build_holding_row_loss():
    """When current price is below avg price, return diff is negative."""
    row, metrics = _build_holding_row("AAPL", 5, 200.0, 180.0, 175.0)
    assert metrics["cost"] == 1000.0
    assert metrics["value"] == 900.0
    assert metrics["day_chg"] == 25.0


def test_build_holding_row_fractional_shares():
    row, metrics = _build_holding_row("BTC", 0.5, 30000.0, 31000.0, 30500.0)
    assert metrics["cost"] == 15000.0
    assert metrics["value"] == 15500.0
    assert metrics["day_chg"] == 250.0
