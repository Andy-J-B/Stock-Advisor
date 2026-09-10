import json
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import IntPrompt

console = Console()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"


def initialize_app():
    if not DATA_DIR.exists():
        console.print(
            "[bold yellow]First time setup: Creating data directories...[/bold yellow]"
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump({"holdings": {}}, f, indent=4)
        console.print("[green]Created empty portfolio file.[/green]")

    if not SETTINGS_FILE.exists():
        # In CI or any non-interactive shell there is no one to answer the
        # risk-profile prompts, so fall back to a sensible default instead of
        # blocking on stdin (which aborts with EOF in GitHub Actions).
        if sys.stdin.isatty() and not os.environ.get("CI"):
            console.print(
                "\n[bold cyan]Let's configure your advisor risk profile![/bold cyan]"
            )
            console.print(
                "Allocate your portfolio across three categories (must total 100%). \n[Convervative (Safe Assets) /Moderate (Blue chip)/Aggressive (Growth/Tech)]"
            )

            while True:
                cons = IntPrompt.ask(
                    "Percentage for [bold green]Conservative[/bold green] (Safe assets)",
                    default=0,
                )
                mod = IntPrompt.ask(
                    "Percentage for [bold yellow]Moderate[/bold yellow] (Blue-chip stocks)",
                    default=100,
                )

                total_so_far = cons + mod
                if total_so_far > 100:
                    console.print(
                        f"[red]Error: Total is {total_so_far}%. It cannot exceed 100%. Let's try again.[/red]\n"
                    )
                    continue

                agg = 100 - total_so_far
                console.print(
                    f"Remaining allocated to [bold red]Aggressive[/bold red] (Growth/Tech): {agg}%\n"
                )
                break
            risk_allocation = {
                "conservative": cons,
                "moderate": mod,
                "aggressive": agg,
            }
        else:
            risk_allocation = {"conservative": 30, "moderate": 40, "aggressive": 30}
            console.print(
                "[yellow]Non-interactive environment: using default risk allocation "
                "(30/40/30). Edit data/settings.json to customize.[/yellow]"
            )

        default_settings = {
            "risk_allocation": risk_allocation,
            "currency": "USD",
        }

        with open(SETTINGS_FILE, "w") as f:
            json.dump(default_settings, f, indent=4)

        console.print("[bold green]Profile saved successfully![/bold green]\n")
        return True

    return False
