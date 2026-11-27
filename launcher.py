#!/usr/bin/env python
"""Interactive command launcher for Stock Advisor.

Lists every CLI command with a one-line description and lets you pick one
(or type a full command line) to run — like a mini Jupyter menu for the CLI.

Recent workflows (top-buys, stock-status, brief, newsletter and the Streamlit
dashboard) are grouped and highlighted so they are easy to find.

Usage:
    .venv/bin/python launcher.py
    .venv/bin/python launcher.py --list      # non-interactive command index
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from main import app

PROJECT_DIR = Path(__file__).resolve().parent
console = Console()

# Commands are grouped in this order. Anything not listed falls into "Other".
GROUP_ORDER = [
    "Dashboard",
    "Portfolio",
    "Analysis & Advice",
    "Screens & Brief",
    "Watchlist & History",
    "Configuration",
    "Other",
]

GROUPS: dict[str, list[str]] = {
    "Portfolio": [
        "view-portfolio",
        "add-stock",
        "sell-stock",
        "remove-stock",
        "deposit",
        "update-cash",
        "set-initial",
        "export",
        "dividends",
    ],
    "Analysis & Advice": [
        "analyze",
        "research",
        "portfolio-news",
        "market-update",
        "rebalance",
        "optimize-portfolio",
        "predict",
        "advisor-chat",
    ],
    "Screens & Brief": [
        "top-buys",
        "stock-status",
        "brief",
        "brief-weights",
        "newsletter",
    ],
    "Watchlist & History": ["watchlist"],
    "Configuration": ["settings"],
}

# Recently added commands get a NEW badge in the menu.
NEW_COMMANDS = {
    "top-buys",
    "stock-status",
    "brief",
    "newsletter",
    "advisor-chat",
    "dashboard",
}

EXAMPLES: dict[str, str] = {
    "dashboard": (
        "streamlit run dashboard.py\n"
        "(needs DATABASE_URL in .env or .streamlit/secrets.toml)"
    ),
    "top-buys": (
        "top-buys --limit 10\n"
        "top-buys --tickers MSFT,NVDA,UNH,JPM --limit 4\n"
        "top-buys --universe sp500,tsx60 --pool-size 80 --persist"
    ),
    "stock-status": (
        "stock-status\n"
        "stock-status --us 20 --ca 10\n"
        "stock-status --tickers MSFT,NVDA,RY.TO --sort ath\n"
        "stock-status --persist"
    ),
    "brief": (
        "brief\n"
        "brief --persist --notify\n"
        "brief --tickers MSFT,NVDA"
    ),
    "newsletter": "newsletter\nnewsletter --no-send\nnewsletter --preview",
    "watchlist": (
        "watchlist show\n"
        "watchlist add --ticker NVDA,MSFT\n"
        "watchlist remove --ticker NVDA"
    ),
    "add-stock": (
        "add-stock AAPL 10 185.50 --account USD\n"
        "add-stock VFV.TO 5 120 --account CAD --currency CAD"
    ),
    "sell-stock": "sell-stock AAPL 5 200 --account USD",
    "deposit": (
        "deposit 1000 --account USD --currency USD\n"
        "deposit 500 --account CAD --currency CAD"
    ),
    "predict": "predict AAPL --horizon 5\npredict MSFT --retrain",
    "optimize-portfolio": (
        "optimize-portfolio\n"
        "optimize-portfolio --objective min-volatility\n"
        "optimize-portfolio -o efficient-risk -t 0.15"
    ),
    "advisor-chat": (
        'advisor-chat AAPL\nadvisor-chat VISA.NE --question "Should I trim?"'
    ),
}

# Non-Typer shortcuts that still belong in the launcher menu.
EXTRA_ACTIONS: list[tuple[str, str, str]] = [
    (
        "dashboard",
        "Dashboard",
        "Open the Streamlit history dashboard (reads Supabase).",
    ),
]


@dataclass(frozen=True)
class Entry:
    """One selectable launcher item."""

    key: str
    group: str
    description: str
    kind: str = "command"  # "command" | "action"


def _command_name(cmd) -> str:
    """Return the CLI name Typer exposes for a registered command."""
    return cmd.name or cmd.callback.__name__.replace("_", "-")


def _build_entries() -> list[Entry]:
    """Return every command plus launcher-only actions in display order."""
    entries: list[Entry] = [
        Entry(key=key, group=group, description=description, kind="action")
        for key, group, description in EXTRA_ACTIONS
    ]

    for cmd in app.registered_commands:
        key = _command_name(cmd)
        doc = (cmd.callback.__doc__ or "").strip()
        description = doc.splitlines()[0] if doc else "(no description)"
        group = next(
            (name for name, members in GROUPS.items() if key in members),
            "Other",
        )
        entries.append(
            Entry(key=key, group=group, description=description, kind="command")
        )

    def sort_key(entry: Entry) -> tuple[int, int, str]:
        group_rank = (
            GROUP_ORDER.index(entry.group)
            if entry.group in GROUP_ORDER
            else len(GROUP_ORDER)
        )
        members = GROUPS.get(entry.group, [])
        position = members.index(entry.key) if entry.key in members else 999
        return (group_rank, position, entry.key)

    return sorted(entries, key=sort_key)


def _filter_entries(
    entries: list[Entry], query: str
) -> list[tuple[int, Entry]]:
    """Return (menu_number, entry) pairs matching a case-insensitive query."""
    query = query.strip().lower()
    numbered = list(enumerate(entries, 1))
    if not query:
        return numbered
    return [
        (number, entry)
        for number, entry in numbered
        if query in entry.key.lower()
        or query in entry.group.lower()
        or query in entry.description.lower()
    ]


def _render_menu(entries: list[Entry], query: str = "") -> None:
    """Print the grouped command table."""
    matches = _filter_entries(entries, query)

    table = Table(
        title="Stock Advisor — Command Launcher",
        border_style="bright_blue",
        title_style="bold",
    )
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Group", style="dim", no_wrap=True)
    table.add_column("Command", style="bold green", no_wrap=True)
    table.add_column("What it does", style="white")

    for number, entry in matches:
        label = entry.key
        if entry.key in NEW_COMMANDS:
            label += " [cyan]NEW[/cyan]"
        table.add_row(str(number), entry.group, label, entry.description)

    if not matches:
        table.add_row("", "", "[yellow]no matches[/yellow]", f"filter: {query!r}")
    console.print(table)


def _print_hint() -> None:
    console.print(
        Panel.fit(
            "Enter a [cyan]#[/cyan] to run an item, type a full command "
            "(e.g. [green]predict AAPL --horizon 5[/green]), "
            "[cyan]/text[/cyan] to filter, [cyan]r[/cyan] to re-run the last "
            "command, or [yellow]q[/yellow] to quit.",
            border_style="dim",
        )
    )


def _show_help(entry: Entry) -> None:
    """Show the CLI's own help page (or the action description)."""
    if entry.kind == "action":
        console.print(f"[bold yellow]--- {entry.key} ---[/bold yellow]")
        console.print(entry.description)
        return

    result = subprocess.run(
        [sys.executable, "main.py", entry.key, "--help"],
        cwd=PROJECT_DIR,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    out = result.stdout or result.stderr
    console.print(f"[bold yellow]--- {entry.key} --help ---[/bold yellow]")
    console.print(out)


def _show_examples(entry: Entry) -> None:
    examples = EXAMPLES.get(entry.key)
    if examples:
        console.print(
            Panel(
                examples,
                title=f"[bold]Examples: {entry.key}[/bold]",
                border_style="dim",
                expand=False,
            )
        )


def _run_action(entry: Entry) -> None:
    """Run a launcher-only action (currently the Streamlit dashboard)."""
    if entry.key != "dashboard":
        console.print(f"[red]Unknown action: {entry.key}[/red]")
        return

    if importlib.util.find_spec("streamlit") is None:
        console.print(
            "[red]Streamlit is not installed.[/red] "
            "Run [green]pip install -r requirements.txt[/green] first."
        )
        return

    console.print("\n[bold blue]▶ streamlit run dashboard.py[/bold blue]\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            cwd=PROJECT_DIR,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


def _run(argv: list[str]) -> None:
    console.print(f"\n[bold blue]▶ python main.py {' '.join(argv)}[/bold blue]\n")
    try:
        result = subprocess.run([sys.executable, "main.py", *argv], cwd=PROJECT_DIR)
        if result.returncode != 0:
            console.print(f"[dim]Command exited with status {result.returncode}.[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Command cancelled.[/yellow]")


def _ask(prompt: str, default: str = "") -> str | None:
    """Ask a question; return None if stdin is closed (EOF)."""
    try:
        return Prompt.ask(prompt, default=default).strip()
    except EOFError:
        return None


def _run_command(entry: Entry) -> list[str] | None:
    """Show help/examples, collect arguments, and run a CLI command."""
    _show_help(entry)
    _show_examples(entry)
    args = _ask(f"Arguments for [green]{entry.key}[/green] (empty = no args)")
    if args is None:
        return None
    argv = [entry.key] + shlex.split(args)
    _run(argv)
    return argv


def _list_commands() -> None:
    """Non-interactive command index (useful for scripts and docs)."""
    entries = _build_entries()
    for number, entry in _filter_entries(entries, ""):
        badge = " [cyan](new)[/cyan]" if entry.key in NEW_COMMANDS else ""
        console.print(
            f"[cyan]{number:>2}[/cyan]  [dim]{entry.group:<20}[/dim] "
            f"[bold green]{entry.key}[/bold green]{badge}  {entry.description}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive menu for the Stock Advisor CLI."
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Print the command index and exit.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the screen between menu renders.",
    )
    options = parser.parse_args(argv)

    entries = _build_entries()

    if options.list:
        _list_commands()
        return

    query = ""
    last_argv: list[str] | None = None

    while True:
        if not options.no_clear:
            console.clear()

        _render_menu(entries, query)
        _print_hint()

        choice = _ask("Choice", "q")
        if choice is None or choice.lower() in ("q", "quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break

        # "/" filters the menu without running anything.
        if choice.startswith("/"):
            query = choice[1:].strip()
            continue

        # "r" re-runs the last command (handy while iterating on flags).
        if choice.lower() == "r":
            if last_argv:
                _run(last_argv)
            else:
                console.print("[yellow]Nothing to re-run yet.[/yellow]")
            if _ask("[dim]Press Enter to return to the menu[/dim]") is None:
                break
            continue

        if choice.isdigit():
            number = int(choice)
            matches = dict(_filter_entries(entries, query))
            entry = matches.get(number)
            if entry is None:
                console.print(f"[red]Invalid choice: {choice}[/red]")
                if _ask("[dim]Press Enter to continue[/dim]") is None:
                    break
                continue

            if entry.kind == "action":
                _run_action(entry)
            else:
                argv = _run_command(entry)
                if argv is None:
                    break
                last_argv = argv
            query = ""
        else:
            argv = shlex.split(choice)
            known = {entry.key for entry in entries}
            key = argv[0].lstrip("-") if argv else ""
            if key in known:
                _run(argv)
                last_argv = argv
                query = ""
            else:
                console.print(
                    f"[red]Unknown command: {argv[0] if argv else ''}[/red]"
                )

        if _ask("[dim]Press Enter to return to the menu[/dim]") is None:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
