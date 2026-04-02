import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv
from rich.markdown import Markdown

load_dotenv()

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
def deposit(
    amount: float,
    currency: str = typer.Option("USD", "--currency", "-c", help="USD or CAD"),
):
    """Deposit cash into your master CAD wallet (auto-converts USD)."""
    with console.status("[bold green]Processing deposit...[/bold green]"):
        final_amt, rate = portfolio.deposit_cash(amount, currency)

    if currency.upper() == "USD":
        console.print(
            f"[green]Converted ${amount} USD to [bold]${final_amt:,.2f} CAD[/bold] (Rate: {rate:.4f})[/green]"
        )
    else:
        console.print(f"[green]Deposited ${final_amt:,.2f} CAD.[/green]")


@app.command()
def sell_stock(
    ticker: str,
    shares: float,
    price: float,
    account: str = typer.Option("USD", "--account", "-a"),
):
    """Sell stock and convert proceeds to CAD cash."""
    try:
        with console.status("[bold red]Processing sale...[/bold red]"):
            proceeds, rate = portfolio.sell_position(account, ticker, shares, price)

        console.print(f"[bold green]Sold {shares} {ticker}![/bold green]")
        console.print(
            f"Proceeds added to CAD cash: [bold]${proceeds:,.2f}[/bold] (FX Rate: {rate:.4f})"
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def view_portfolio(
    account: str = typer.Option(
        "ALL", "--account", "-a", help="Specific account to view (e.g., USD, CAD)"
    )
):
    """View portfolio with live market prices, currency conversion, and total returns."""
    current_portfolio = portfolio.load()
    accounts = current_portfolio.get("accounts", {})

    if not accounts:
        console.print(
            "[yellow]No accounts found. Use 'add-stock' to get started.[/yellow]"
        )
        return

    # Fetch live FX rate for normalization
    with console.status("[bold green]Fetching live exchange rates...[/bold green]"):
        fx_rate = data_client.get_usd_to_cad()

    accounts_to_show = (
        [account.upper()] if account.upper() != "ALL" else accounts.keys()
    )

    # Grand totals (Normalized to CAD)
    grand_total_value_cad = 0.0
    grand_total_cost_cad = 0.0
    grand_total_cash_cad = 0.0

    for acc_name in accounts_to_show:
        if acc_name not in accounts:
            console.print(f"[red]Account '{acc_name}' not found.[/red]")
            continue

        holdings = accounts[acc_name].get("holdings", {})
        cash = accounts[acc_name].get("cash", 0.0)

        # Determine multiplier for this account's contribution to global CAD totals
        multiplier = fx_rate if acc_name == "USD" else 1.0
        grand_total_cash_cad += cash * multiplier

        # Build Table for the specific account
        table = Table(
            title=f"[bold cyan]{acc_name} Portfolio[/bold cyan] (FX: {multiplier:.4f})"
        )
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Shares", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("Live Price", justify="right", style="blue")
        table.add_column("Return %", justify="right")
        table.add_column("Return $", justify="right")

        acc_cost_basis = 0.0
        acc_market_value = 0.0

        if not holdings:
            console.print(f"\n[bold yellow]--- {acc_name} Account ---[/bold yellow]")
            console.print(f"No holdings. Cash: ${cash:,.2f}")
        else:
            with console.status(
                f"[bold green]Pricing {acc_name} holdings...[/bold green]"
            ):
                for ticker, data in holdings.items():
                    shares = data["shares"]
                    avg_price = data["avg_price"]
                    live_price = data_client.get_current_price(ticker)

                    # Logic for individual account display
                    cost = shares * avg_price
                    acc_cost_basis += cost

                    if live_price > 0:
                        value = shares * live_price
                        acc_market_value += value

                        diff = value - cost
                        pct = (diff / cost) * 100 if cost > 0 else 0
                        color = "green" if diff >= 0 else "red"

                        ret_pct_str = f"[{color}]{pct:+.2f}%[/{color}]"
                        ret_dol_str = f"[{color}]{diff:+.2f}[/{color}]"
                        live_price_str = f"${live_price:.2f}"

                        # Add to Grand Totals (Converted to CAD)
                        grand_total_value_cad += value * multiplier
                        grand_total_cost_cad += cost * multiplier
                    else:
                        ret_pct_str = "[yellow]N/A[/yellow]"
                        ret_dol_str = "[yellow]N/A[/yellow]"
                        live_price_str = "[yellow]Error[/yellow]"

                    table.add_row(
                        ticker,
                        str(shares),
                        f"${avg_price:.2f}",
                        live_price_str,
                        ret_pct_str,
                        ret_dol_str,
                    )

            console.print(table)

            # Account-specific summary (Local Currency)
            acc_ret_dol = acc_market_value - acc_cost_basis
            acc_ret_pct = (
                (acc_ret_dol / acc_cost_basis * 100) if acc_cost_basis > 0 else 0
            )
            ret_color = "green" if acc_ret_dol >= 0 else "red"

            console.print(
                f"  [bold]Cash Balance:[/bold]        [white]${cash:,.2f}[/white]"
            )
            console.print(
                f"  [bold]Account Return:[/bold]      [{ret_color}]${acc_ret_dol:,.2f} ({acc_ret_pct:+.2f}%)[/{ret_color}]"
            )
            console.print(
                f"  [bold]Total Account Value:[/bold] [cyan]${(acc_market_value + cash):,.2f}[/cyan]\n"
            )

    # Global Summary Panel (Always normalized to CAD)
    if len(accounts_to_show) > 0:
        global_ret_dol = grand_total_value_cad - grand_total_cost_cad
        global_ret_pct = (
            (global_ret_dol / grand_total_cost_cad * 100)
            if grand_total_cost_cad > 0
            else 0
        )
        global_color = "green" if global_ret_dol >= 0 else "red"
        net_worth = grand_total_value_cad + grand_total_cash_cad

        summary_table = Table(
            show_header=False,
            border_style="bright_blue",
            title="[bold blue]GLOBAL PORTFOLIO SUMMARY (CAD)[/bold blue]",
        )
        summary_table.add_row("Total Combined Cash", f"${grand_total_cash_cad:,.2f}")
        summary_table.add_row(
            "Total Combined Return",
            f"[{global_color}]${global_ret_dol:,.2f} ({global_ret_pct:+.2f}%)[/{global_color}]",
        )
        summary_table.add_row("NET WORTH", f"[bold cyan]${net_worth:,.2f}[/bold cyan]")

        console.print(Panel(summary_table, expand=False))


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


@app.command(name="update-cash")
def update_cash_cmd(
    amount: float,
    account: str = typer.Option(
        "USD", "--account", "-a", help="Account to update (e.g., USD, CAD)"
    ),
):
    """Update the available buying power (cash) in an account."""
    portfolio.update_cash(account, amount)
    console.print(
        f"[bold green]Successfully updated {account.upper()} buying power to ${amount:,.2f}[/bold green]"
    )


@app.command()
def analyze():
    """Feature 1: Analyze current holdings against your risk profile."""
    console.print("[bold blue]Analyzing portfolio...[/bold blue]")

    user_settings = config.load_settings()
    current_portfolio = portfolio.load()

    # --- BUG FIX: Check the new multi-account structure ---
    accounts = current_portfolio.get("accounts", {})
    has_assets = False
    for acc_name, acc_data in accounts.items():
        if acc_data.get("holdings") or acc_data.get("cash", 0) > 0:
            has_assets = True
            break

    if not has_assets:
        console.print(
            "[yellow]Please add stocks or cash to your portfolio first![/yellow]"
        )
        return

    # Pass the data to the brain (advisor.py)
    with console.status("[bold cyan]Consulting AI Advisor...[/bold cyan]"):
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


@app.command()
def research(ticker: str):
    """Feature 4: Get a deep-dive analyst report and action plan for a specific stock."""
    current_portfolio = portfolio.load()

    with console.status(
        f"[bold cyan]Compiling Senior Analyst Report for {ticker.upper()}... This may take a few seconds.[/bold cyan]"
    ):
        report_md = advisor.generate_stock_report(ticker, current_portfolio)

    if "Error" in report_md or "not found" in report_md:
        console.print(report_md)
    else:
        # Wrap the markdown in a nice panel
        console.print(
            Panel(
                Markdown(report_md),
                title=f"📈 Investment Thesis: {ticker.upper()}",
                border_style="cyan",
            )
        )
