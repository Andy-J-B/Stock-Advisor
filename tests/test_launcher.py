"""Tests for the interactive launcher metadata helpers."""

from launcher import (
    NEW_COMMANDS,
    _build_entries,
    _command_name,
    _filter_entries,
)


def test_recent_commands_are_in_the_menu():
    keys = {entry.key for entry in _build_entries()}
    assert {"top-buys", "stock-status", "brief", "newsletter", "advisor-chat"} <= keys
    assert "dashboard" in keys


def test_tui_is_gone():
    keys = {entry.key for entry in _build_entries()}
    assert "tui" not in keys


def test_dashboard_is_first():
    entries = _build_entries()
    assert entries[0].key == "dashboard"
    assert entries[0].kind == "action"


def test_portfolio_commands_are_grouped():
    by_key = {entry.key: entry for entry in _build_entries()}
    assert by_key["add-stock"].group == "Portfolio"
    assert by_key["top-buys"].group == "Screens & Brief"
    assert by_key["settings"].group == "Configuration"


def test_new_commands_have_badges():
    keys = {entry.key for entry in _build_entries()}
    assert NEW_COMMANDS <= keys


def test_filter_matches_key_group_and_description():
    entries = _build_entries()
    assert [entry.key for _, entry in _filter_entries(entries, "top-buys")] == [
        "top-buys"
    ]
    assert [entry.key for _, entry in _filter_entries(entries, "screens")] == [
        "top-buys",
        "stock-status",
        "brief",
        "brief-weights",
        "newsletter",
    ]
    assert any(
        entry.key == "dividends"
        for _, entry in _filter_entries(entries, "dividend income")
    )


def test_filter_empty_returns_everything():
    entries = _build_entries()
    assert len(_filter_entries(entries, "")) == len(entries)


def test_filter_is_case_insensitive():
    entries = _build_entries()
    upper = [entry.key for _, entry in _filter_entries(entries, "BRIEF")]
    lower = [entry.key for _, entry in _filter_entries(entries, "brief")]
    assert upper == lower


def test_command_name_prefers_explicit_typer_name():
    class Dummy:
        name = "custom-name"
        callback = type("Callback", (), {"__name__": "python_name"})()

    assert _command_name(Dummy()) == "custom-name"


def test_command_name_hyphenates_callback_name():
    class Dummy:
        name = None
        callback = type("Callback", (), {"__name__": "add_stock"})()

    assert _command_name(Dummy()) == "add-stock"
