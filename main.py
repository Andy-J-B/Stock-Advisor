import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import our custom modules from the src directory
# Note: These will show errors in your editor until you create the actual files in src/
from src import setup, portfolio, advisor, config, data_client

app = typer.Typer(
    name="Stock Advisor",
    help="A CLI tool to manage your portfolio and get AI-driven stock advice.",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main_setup(ctx: typer.Context):
    """
    Runs before every command to ensure the app is initialized.
    """
    just_initialized = setup.initialize_app()

    # If no command was passed (e.g., user just typed `python main.py`), show the help menu
    if ctx.invoked_subcommand is None and not just_initialized:
        console.print(
            Panel.fit(
                "[bold blue]Welcome to your Terminal Stock Advisor[/bold blue]\n"
                "Run [bold cyan]python main.py --help[/bold cyan] to see available commands."
            )
        )


@app.command()
def view_portfolio(
    account: str = typer.Option(
        "ALL", "--account", "-a", help="Specific account to view (e.g., USD, CAD)"
    )
):
    """View your current stock holdings with live market prices."""
    current_portfolio = portfolio.load()
    accounts = current_portfolio.get("accounts", {})

    if not accounts:
        console.print(
            "[yellow]No accounts found. Use 'add-stock' to get started.[/yellow]"
        )
        return

    accounts_to_show = (
        [account.upper()] if account.upper() != "ALL" else accounts.keys()
    )

    for acc in accounts_to_show:
        if acc not in accounts:
            console.print(f"[red]Account '{acc}' not found.[/red]")
            continue

        holdings = accounts[acc].get("holdings", {})

        if not holdings:
            console.print(f"\n[bold yellow]--- {acc} Account ---[/bold yellow]")
            console.print("Empty.")
            continue

        # Upgraded table with Live Price and Return columns
        table = Table(title=f"{acc} Portfolio")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Shares", justify="right", style="magenta")
        table.add_column("Avg Price", justify="right")
        table.add_column("Live Price", justify="right", style="blue")
        table.add_column("Total Return", justify="right")

        with console.status(
            f"[bold green]Fetching live market data for {acc}...[/bold green]"
        ):
            for ticker, data in holdings.items():
                shares = data["shares"]
                avg_price = data["avg_price"]

                # Fetch live price from our new data_client!
                live_price = data_client.get_current_price(ticker)

                # Calculate profit/loss
                if live_price > 0:
                    total_return_pct = ((live_price - avg_price) / avg_price) * 100
                    return_str = (
                        f"[green]+{total_return_pct:.2f}%[/green]"
                        if total_return_pct >= 0
                        else f"[red]{total_return_pct:.2f}%[/red]"
                    )
                    live_price_str = f"${live_price:.2f}"
                else:
                    return_str = "[yellow]N/A[/yellow]"
                    live_price_str = "[yellow]Error[/yellow]"

                table.add_row(
                    ticker, str(shares), f"${avg_price:.2f}", live_price_str, return_str
                )

        console.print("\n")
        console.print(table)


@app.command()
def add_stock(
    ticker: str,
    shares: float,
    price: float,
    account: str = typer.Option(
        "USD", "--account", "-a", help="Account to add to (e.g., USD, CAD)"
    ),
):
    """Add or update a stock in a specific account portfolio."""
    portfolio.add_position(account, ticker, shares, price)
    console.print(
        f"[bold green]Successfully added {shares} shares of {ticker.upper()} at ${price:.2f} to your {account.upper()} account.[/bold green]"
    )


@app.command()
def analyze():
    """Feature 1: Analyze current holdings against your risk profile."""
    console.print("[bold blue]Analyzing portfolio...[/bold blue]")

    user_settings = config.load_settings()
    current_portfolio = portfolio.load()

    if not current_portfolio.get("holdings"):
        console.print("[yellow]Please add stocks to your portfolio first![/yellow]")
        return

    # Pass the data to the brain (advisor.py)
    advice = advisor.evaluate_portfolio(current_portfolio, user_settings)
    console.print(advice)


@app.command()
def market_update():
    """Feature 2: Get general market recommendations based on today's news."""
    with console.status("[bold green]Fetching latest market news...[/bold green]"):
        news = data_client.get_macro_news()

    with console.status("[bold cyan]Analyzing sentiment...[/bold cyan]"):
        recommendations = advisor.generate_market_advice(news)

    console.print(Panel(recommendations, title="Market Update", expand=False))


@app.command()
def portfolio_news():
    """Feature 3: Get recommendations for your specific stocks based on recent news."""
    current_portfolio = portfolio.load()
    tickers = list(current_portfolio.get("holdings", {}).keys())

    if not tickers:
        console.print("[yellow]Your portfolio is empty. Nothing to analyze.[/yellow]")
        return

    console.print(f"Fetching news for: [bold cyan]{', '.join(tickers)}[/bold cyan]\n")

    for ticker in tickers:
        news = data_client.get_ticker_news(ticker)
        advice = advisor.analyze_ticker_sentiment(ticker, news)
        console.print(f"[bold]{ticker} Update:[/bold] {advice}")
        console.print("-" * 40)


@app.command()
def settings(
    conservative: int = typer.Option(
        None, "--conservative", "-c", help="Percentage of safe assets"
    ),
    moderate: int = typer.Option(
        None, "--moderate", "-m", help="Percentage of standard assets"
    ),
    aggressive: int = typer.Option(
        None, "--aggressive", "-a", help="Percentage of high-growth assets"
    ),
):
    """View or update your advisor risk allocation."""
    # If the user provided ANY of the options, they are trying to update
    if any(x is not None for x in [conservative, moderate, aggressive]):
        # Default to 0 if an option wasn't provided, to handle the math
        c = conservative or 0
        m = moderate or 0
        a = aggressive or 0

        total = c + m + a
        if total != 100:
            console.print(
                f"[red]Error: Your allocation totals {total}%. It must equal exactly 100%.[/red]"
            )
            raise typer.Exit(code=1)

        config.update_allocation(c, m, a)
        console.print("[bold green]Risk allocation successfully updated![/bold green]")

    # If they just ran `python main.py settings` without options, just view current
    current = config.load_settings()
    alloc = current.get("risk_allocation", {})

    table = Table(title="Current Risk Allocation")
    table.add_column("Category", style="cyan")
    table.add_column("Target Allocation", justify="right", style="green")

    table.add_row("Conservative", f"{alloc.get('conservative', 0)}%")
    table.add_row("Moderate", f"{alloc.get('moderate', 100)}%")
    table.add_row("Aggressive", f"{alloc.get('aggressive', 0)}%")

    console.print(table)


if __name__ == "__main__":
    app()
