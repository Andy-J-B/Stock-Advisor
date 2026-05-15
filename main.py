import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv
from rich.markdown import Markdown
from src import setup, portfolio, advisor, config, data_client, alpha_vantage

load_dotenv()

app = typer.Typer(
    name="Stock Advisor",
    help="A CLI tool to manage your portfolio and get AI-driven stock advice.",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main_setup(ctx: typer.Context):
    """
    If no subcommand is provided, treat the first argument as a stock ticker
    for an immediate deep-dive research report.
    """
    just_initialized = setup.initialize_app()

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


@app.command(name="set-initial")
def set_initial_cmd(
    amount: float,
    account: str = typer.Option(
        "USD", "--account", "-a", help="Account to update (e.g., USD, CAD)"
    ),
):
    """Set the initial cash investment amount for an account to track all-time returns."""
    portfolio.set_initial_cash(account, amount)
    console.print(
        f"[bold green]Successfully set {account.upper()} initial investment to ${amount:,.2f}[/bold green]"
    )


@app.command()
def view_portfolio(
    account: str = typer.Option(
        "ALL", "--account", "-a", help="Specific account to view (e.g., USD, CAD)"
    )
):
    """View portfolio with live market prices, currency conversion, and total returns."""
    # Run ensure_account_exists indirectly via loading to patch old JSON files
    current_portfolio = portfolio.load()
    for acc in ["USD", "CAD"]:
        portfolio.ensure_account_exists(current_portfolio, acc)

    current_portfolio = portfolio.load()  # Reload patched data
    accounts = current_portfolio.get("accounts", {})

    if not accounts:
        console.print(
            "[yellow]No accounts found. Use 'add-stock' to get started.[/yellow]"
        )
        return

    with console.status("[bold green]Fetching live exchange rates...[/bold green]"):
        fx_rate = data_client.get_usd_to_cad()

    accounts_to_show = (
        [account.upper()] if account.upper() != "ALL" else accounts.keys()
    )

    # Grand totals (Normalized to CAD)
    grand_total_value_cad = 0.0
    grand_total_cost_cad = 0.0
    grand_total_cash_cad = 0.0
    grand_total_initial_cad = 0.0
    grand_total_day_chg_cad = 0.0

    for acc_name in accounts_to_show:
        if acc_name not in accounts:
            console.print(f"[red]Account '{acc_name}' not found.[/red]")
            continue

        holdings = accounts[acc_name].get("holdings", {})
        cash = accounts[acc_name].get("cash", 0.0)
        initial_cash = accounts[acc_name].get("initial_cash", 0.0)

        multiplier = fx_rate if acc_name == "USD" else 1.0
        grand_total_cash_cad += cash * multiplier
        grand_total_initial_cad += initial_cash * multiplier

        table = Table(
            title=f"[bold cyan]{acc_name} Portfolio[/bold cyan] (FX: {multiplier:.4f})"
        )
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Shares", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("Live Price", justify="right", style="blue")
        table.add_column("Day Change", justify="right")
        table.add_column("Day Chg ($)", justify="right")
        table.add_column("Total Value", justify="right", style="magenta")
        table.add_column("Return %", justify="right")
        table.add_column("Return $", justify="right")

        acc_cost_basis = 0.0
        acc_market_value = 0.0
        acc_day_chg_dol = 0.0

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
                    live_price, prev_close = data_client.get_current_price(ticker)

                    cost = shares * avg_price
                    acc_cost_basis += cost

                    if live_price > 0:
                        value = shares * live_price
                        acc_market_value += value

                        # All-time return calculations
                        diff = value - cost
                        pct = (diff / cost) * 100 if cost > 0 else 0
                        color = "green" if diff >= 0 else "red"

                        # Intraday calculations
                        if prev_close > 0:
                            day_diff = live_price - prev_close
                            day_pct = (day_diff / prev_close) * 100
                        else:
                            day_diff, day_pct = 0.0, 0.0

                        total_day_diff = day_diff * shares
                        acc_day_chg_dol += total_day_diff

                        day_color = "green" if day_diff >= 0 else "red"

                        ret_pct_str = f"[{color}]{pct:+.2f}%[/{color}]"
                        ret_dol_str = f"[{color}]{diff:+.2f}[/{color}]"
                        live_price_str = f"${live_price:.2f}"
                        day_change_str = f"[{day_color}]{day_diff:+.2f} ({day_pct:+.2f}%)[/{day_color}]"
                        day_chg_dol_str = (
                            f"[{day_color}]{total_day_diff:+.2f}[/{day_color}]"
                        )
                        total_val_str = f"${value:,.2f}"

                        grand_total_value_cad += value * multiplier
                        grand_total_cost_cad += cost * multiplier
                        grand_total_day_chg_cad += total_day_diff * multiplier
                    else:
                        ret_pct_str = "[yellow]N/A[/yellow]"
                        ret_dol_str = "[yellow]N/A[/yellow]"
                        live_price_str = "[yellow]Error[/yellow]"
                        day_change_str = "[yellow]N/A[/yellow]"
                        day_chg_dol_str = "[yellow]N/A[/yellow]"
                        total_val_str = "[yellow]N/A[/yellow]"

                    table.add_row(
                        ticker,
                        str(shares),
                        f"${avg_price:.2f}",
                        live_price_str,
                        day_change_str,
                        day_chg_dol_str,
                        total_val_str,
                        ret_pct_str,
                        ret_dol_str,
                    )

            console.print(table)

            # Account-specific summary
            acc_ret_dol = acc_market_value - acc_cost_basis
            acc_ret_pct = (
                (acc_ret_dol / acc_cost_basis * 100) if acc_cost_basis > 0 else 0
            )
            ret_color = "green" if acc_ret_dol >= 0 else "red"

            # Calculate today's return % (based on yesterday's total value)
            acc_prev_value = acc_market_value - acc_day_chg_dol
            acc_day_chg_pct = (
                (acc_day_chg_dol / acc_prev_value * 100) if acc_prev_value > 0 else 0
            )
            day_ret_color = "green" if acc_day_chg_dol >= 0 else "red"

            total_acc_value = acc_market_value + cash
            all_time_ret_dol = total_acc_value - initial_cash
            all_time_ret_pct = (
                (all_time_ret_dol / initial_cash * 100) if initial_cash > 0 else 0
            )
            all_time_color = "green" if all_time_ret_dol >= 0 else "red"

            console.print(
                f"  [bold]Initial Investment:[/bold]  [white]${initial_cash:,.2f}[/white]"
            )
            console.print(
                f"  [bold]Cash Balance:[/bold]        [white]${cash:,.2f}[/white]"
            )
            console.print(
                f"  [bold]Today's Return:[/bold]      [{day_ret_color}]${acc_day_chg_dol:,.2f} ({acc_day_chg_pct:+.2f}%)[/{day_ret_color}]"
            )
            console.print(
                f"  [bold]Holdings Return:[/bold]     [{ret_color}]${acc_ret_dol:,.2f} ({acc_ret_pct:+.2f}%)[/{ret_color}]"
            )
            console.print(
                f"  [bold]All-Time Return:[/bold]     [{all_time_color}]${all_time_ret_dol:,.2f} ({all_time_ret_pct:+.2f}%)[/{all_time_color}]"
            )
            console.print(
                f"  [bold]Total Account Value:[/bold] [cyan]${total_acc_value:,.2f}[/cyan]\n"
            )

    # Global Summary Panel
    if len(accounts_to_show) > 0:
        global_ret_dol = grand_total_value_cad - grand_total_cost_cad
        global_ret_pct = (
            (global_ret_dol / grand_total_cost_cad * 100)
            if grand_total_cost_cad > 0
            else 0
        )
        global_color = "green" if global_ret_dol >= 0 else "red"

        # Calculate global day change
        global_prev_value = grand_total_value_cad - grand_total_day_chg_cad
        global_day_chg_pct = (
            (grand_total_day_chg_cad / global_prev_value * 100)
            if global_prev_value > 0
            else 0
        )
        global_day_color = "green" if grand_total_day_chg_cad >= 0 else "red"

        net_worth = grand_total_value_cad + grand_total_cash_cad

        global_all_time_dol = net_worth - grand_total_initial_cad
        global_all_time_pct = (
            (global_all_time_dol / grand_total_initial_cad * 100)
            if grand_total_initial_cad > 0
            else 0
        )
        global_all_time_color = "green" if global_all_time_dol >= 0 else "red"

        summary_table = Table(
            show_header=False,
            border_style="bright_blue",
            title="[bold blue]GLOBAL PORTFOLIO SUMMARY (CAD)[/bold blue]",
        )
        summary_table.add_row(
            "Total Initial Invested", f"${grand_total_initial_cad:,.2f}"
        )
        summary_table.add_row("Total Combined Cash", f"${grand_total_cash_cad:,.2f}")
        summary_table.add_row(
            "Today's Return",
            f"[{global_day_color}]${grand_total_day_chg_cad:,.2f} ({global_day_chg_pct:+.2f}%)[/{global_day_color}]",
        )
        summary_table.add_row(
            "Active Holdings Return",
            f"[{global_color}]${global_ret_dol:,.2f} ({global_ret_pct:+.2f}%)[/{global_color}]",
        )
        summary_table.add_row(
            "All-Time Global Return",
            f"[{global_all_time_color}]${global_all_time_dol:,.2f} ({global_all_time_pct:+.2f}%)[/{global_all_time_color}]",
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
def remove_stock(
    ticker: str,
    shares: float = typer.Option(None, "--shares", "-s", help="Number of shares to remove (omit to remove entire position)"),
    account: str = typer.Option("USD", "--account", "-a", help="Account to remove from (e.g., USD, CAD)"),
):
    """Remove a stock position (or partial shares) from your portfolio."""
    try:
        portfolio.remove_position(account, ticker, shares)
        if shares:
            console.print(f"[bold green]Removed {shares} shares of {ticker.upper()} from {account.upper()}.[/bold green]")
        else:
            console.print(f"[bold green]Removed entire position of {ticker.upper()} from {account.upper()}.[/bold green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def analyze():
    """Feature 1: Analyze current holdings against your risk profile."""
    console.print("[bold blue]Analyzing portfolio...[/bold blue]")

    user_settings = config.load_settings()
    current_portfolio = portfolio.load()

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
    tickers = []
    for acc in current_portfolio.get("accounts", {}).values():
        tickers.extend(acc.get("holdings", {}).keys())

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
    current = config.load_settings()
    current_alloc = current.get("risk_allocation", {})

    if any(x is not None for x in [conservative, moderate, aggressive]):
        c = conservative if conservative is not None else current_alloc.get("conservative", 0)
        m = moderate if moderate is not None else current_alloc.get("moderate", 100)
        a = aggressive if aggressive is not None else current_alloc.get("aggressive", 0)

        specified = sum(1 for x in [conservative, moderate, aggressive] if x is not None)
        if specified == 2:
            if conservative is None:
                c = 100 - m - a
            elif moderate is None:
                m = 100 - c - a
            else:
                a = 100 - c - m
        elif specified == 1:
            user_val = (
                conservative if conservative is not None
                else moderate if moderate is not None
                else aggressive
            )
            remaining = 100 - user_val
            split = remaining // 2
            if conservative is None:
                c = split
            if moderate is None:
                m = split
            if aggressive is None:
                a = remaining - split

        total = c + m + a
        if total != 100:
            console.print(
                f"[red]Error: Your allocation totals {total}%. It must equal exactly 100%.[/red]"
            )
            raise typer.Exit(code=1)

        config.update_allocation(c, m, a)
        console.print("[bold green]Risk allocation successfully updated![/bold green]")

    alloc = current_alloc

    table = Table(title="Current Risk Allocation")
    table.add_column("Category", style="cyan")
    table.add_column("Target Allocation", justify="right", style="green")

    table.add_row("Conservative", f"{alloc.get('conservative', 0)}%")
    table.add_row("Moderate", f"{alloc.get('moderate', 100)}%")
    table.add_row("Aggressive", f"{alloc.get('aggressive', 0)}%")

    console.print(table)


# Ensure the existing research command is also available explicitly
@app.command()
def research(ticker: str):
    """Get a deep-dive analyst report and action plan for a specific stock."""
    current_portfolio = portfolio.load()

    if ticker:
        # Route to the research logic
        with console.status(
            f"[bold cyan]Performing Ultimate Deep-Dive for {ticker.upper()}...[/bold cyan]"
        ):
            report_md = advisor.generate_stock_report(ticker, current_portfolio)

        console.print(
            Panel(
                Markdown(report_md),
                title=f"📈 Senior Analyst Report: {ticker.upper()}",
                border_style="bright_magenta",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold blue]Welcome to your Terminal Stock Advisor[/bold blue]\n"
                "Run [bold cyan]python main.py <TICKER>[/bold cyan] for a Deep-Dive.\n"
                "Run [bold cyan]python main.py --help[/bold cyan] for all commands."
            )
        )


if __name__ == "__main__":
    app()
