#!/usr/bin/env python
"""Interactive command launcher for Stock Advisor.

Lists every CLI command with a one-line description and lets you pick one
(or type a full command line) to run — like a mini Jupyter menu for the CLI.

Usage:
    .venv/bin/python launcher.py
"""

import os
import shlex
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from main import app

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
console = Console()


def _commands():
    """Return [(name, first_line_of_docstring)] for every registered command."""
    out = []
    for cmd in app.registered_commands:
        name = cmd.name or cmd.callback.__name__.replace("_", "-")
        doc = (cmd.callback.__doc__ or "").strip()
        description = doc.splitlines()[0] if doc else "(no description)"
        out.append((name, description))
    return out


def _run(argv: list[str]):
    console.print(f"\n[bold blue]▶ python main.py {' '.join(argv)}[/bold blue]\n")
    try:
        subprocess.run(
            [sys.executable, "main.py", *argv],
            cwd=PROJECT_DIR,
            stdin=subprocess.DEVNULL,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Command cancelled.[/yellow]")


def _ask(prompt: str, default: str = "") -> str | None:
    """Ask a question; return None if stdin is closed (EOF)."""
    try:
        return Prompt.ask(prompt, default=default).strip()
    except EOFError:
        return None


def main():
    commands = _commands()

    while True:
        console.clear()

        table = Table(
            title="Stock Advisor — Command Launcher",
            border_style="bright_blue",
            title_style="bold",
        )
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Command", style="bold green", no_wrap=True)
        table.add_column("What it does", style="white")
        for i, (name, description) in enumerate(commands, 1):
            table.add_row(str(i), name, description)
        console.print(table)
        console.print(
            Panel.fit(
                "Enter a [cyan]#[/cyan] to run a command, type a full command "
                "(e.g. [green]predict AAPL --horizon 5[/green]), or [yellow]q[/yellow] to quit.",
                border_style="dim",
            )
        )

        choice = _ask("Choice", "q")
        if choice is None or choice.lower() in ("q", "quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if choice.isdigit():
            idx = int(choice) - 1
            if not 0 <= idx < len(commands):
                console.print(f"[red]Invalid choice: {choice}[/red]")
                if _ask("[dim]Press Enter to continue[/dim]") is None:
                    break
                continue
            name, _ = commands[idx]
            args = _ask(
                f"Arguments for [green]{name}[/green] (empty = no args)"
            )
            if args is None:
                break
            argv = [name] + shlex.split(args)
            _run(argv)
        else:
            argv = shlex.split(choice)
            if argv and argv[0].lstrip("-") in {c for c, _ in commands}:
                _run(argv)
            else:
                console.print(f"[red]Unknown command: {argv[0] if argv else ''}[/red]")

        if _ask("[dim]Press Enter to return to the menu[/dim]") is None:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
