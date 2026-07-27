import csv
import io
import time
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv
from rich.markdown import Markdown
from src import setup, portfolio, advisor, config, data_client, __version__
from src import risk, indicators, optimizer, features, ml_model, anomaly

load_dotenv()

app = typer.Typer(
    name="Stock Advisor",
    help="A CLI tool to manage your portfolio and get AI-driven stock advice.",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main_setup(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit.", is_eager=True),
):
    if version:
        console.print(f"Stock Advisor v{__version__}")
        raise typer.Exit()

    just_initialized = setup.initialize_app()

    if ctx.invoked_subcommand is None and not just_initialized:
        console.print(
            Panel.fit(
                "[bold blue]Welcome to your Terminal Stock Advisor[/bold blue]\n"
                "Run [bold cyan]python main.py --help[/bold cyan] to see available commands."
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirm(msg: str) -> bool:
    return typer.confirm(f"[yellow]{msg}[/yellow]", default=False)


def _price_color(diff: float) -> str:
    return "green" if diff >= 0 else "red"


def _pct_str(pct: float) -> str:
    return f"{pct:+.2f}%"


def _dol_str(val: float) -> str:
    return f"${val:,.2f}"


def _colorize(val: float, fmt: str) -> str:
    color = _price_color(val)
    return f"[{color}]{fmt.format(val)}[/{color}]"


def _build_holding_row(ticker: str, shares: float, avg_price: float, live_price: float, prev_close: float):
    cost = shares * avg_price
    value = shares * live_price
    diff = value - cost

    if prev_close > 0:
        day_diff = live_price - prev_close
        day_pct = (day_diff / prev_close) * 100
    else:
        day_diff = day_pct = 0.0

    total_day_diff = day_diff * shares
    day_color = _price_color(day_diff)
    ret_pct = (diff / cost * 100) if cost > 0 else 0.0
    ret_color = _price_color(diff)
    ret_pct_str = f"[{ret_color}]{ret_pct:+.2f}%[/{ret_color}]"
    ret_dol_str = _colorize(diff, "{:+.2f}")

    row = [
        ticker,
        str(shares),
        f"${avg_price:.2f}",
        f"${live_price:.2f}" if live_price > 0 else "[yellow]N/A[/yellow]",
        f"[{day_color}]{day_diff:+.2f} ({_pct_str(day_pct)})[/{day_color}]",
        f"[{day_color}]{total_day_diff:+.2f}[/{day_color}]",
        f"${value:,.2f}" if live_price > 0 else "[yellow]N/A[/yellow]",
        ret_pct_str if live_price > 0 else "[yellow]N/A[/yellow]",
        ret_dol_str if live_price > 0 else "[yellow]N/A[/yellow]",
    ]
    if live_price > 0:
        metrics = {"cost": cost, "value": value, "day_chg": total_day_diff}
    else:
        metrics = {"cost": cost, "value": cost, "day_chg": 0.0}
    return row, metrics


def _build_account_table(acc_name: str, multiplier: float):
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
    return table


def _print_account_summary(cash: float, initial_cash: float, cost: float, value: float, day_chg: float):
    ret_dol = value - cost
    ret_pct = (ret_dol / cost * 100) if cost > 0 else 0
    prev_value = value - day_chg
    day_pct = (day_chg / prev_value * 100) if prev_value > 0 else 0
    total_val = value + cash
    all_time_dol = total_val - initial_cash
    all_time_pct = (all_time_dol / initial_cash * 100) if initial_cash > 0 else 0

    console.print(f"  [bold]Initial Investment:[/bold]  [white]{_dol_str(initial_cash)}[/white]")
    console.print(f"  [bold]Invested in Stocks:[/bold]  [white]{_dol_str(cost)}[/white]")
    console.print(f"  [bold]Cash Balance:[/bold]        [white]{_dol_str(cash)}[/white]")
    console.print(f"  [bold]Today's Return:[/bold]      {_colorize(day_chg, '{:,.2f}')} ({_colorize(day_pct, '{:+.2f}%')})")
    console.print(f"  [bold]Holdings P/L:[/bold]        {_colorize(ret_dol, '{:,.2f}')} ({_colorize(ret_pct, '{:+.2f}%')})")
    console.print(f"  [bold]All-Time Return:[/bold]     {_colorize(all_time_dol, '{:,.2f}')} ({_colorize(all_time_pct, '{:+.2f}%')})")
    console.print(f"  [bold]Total Account Value:[/bold] [cyan]{_dol_str(total_val)}[/cyan]\n")

    return {
        "value_cad": value,
        "cost_cad": cost,
        "day_chg_cad": day_chg,
        "cash_cad": cash,
        "initial_cad": initial_cash,
    }


def _print_global_summary(grand: dict):
    ret_dol = grand["value"] - grand["cost"]
    ret_pct = (ret_dol / grand["cost"] * 100) if grand["cost"] > 0 else 0
    prev = grand["value"] - grand["day_chg"]
    day_pct = (grand["day_chg"] / prev * 100) if prev > 0 else 0
    net_worth = grand["value"] + grand["cash"]
    all_time_dol = net_worth - grand["initial"]
    all_time_pct = (all_time_dol / grand["initial"] * 100) if grand["initial"] > 0 else 0

    t = Table(show_header=False, border_style="bright_blue", title="[bold blue]GLOBAL PORTFOLIO SUMMARY (CAD)[/bold blue]")
    t.add_row("Total Initial Invested", _dol_str(grand["initial"]))
    t.add_row("Total Invested in Stocks", f"[bold white]{_dol_str(grand['cost'])}[/bold white]")
    t.add_row("Total Combined Cash", _dol_str(grand["cash"]))
    t.add_row("Today's Return", _colorize(grand["day_chg"], "{:,.2f}") + f" ({_colorize(day_pct, '{:+.2f}%')})")
    t.add_row("Stocks P/L", _colorize(ret_dol, "{:,.2f}") + f" ({_colorize(ret_pct, '{:+.2f}%')})")
    t.add_row("All-Time Global Return", _colorize(all_time_dol, "{:,.2f}") + f" ({_colorize(all_time_pct, '{:+.2f}%')})")
    t.add_row("NET WORTH", f"[bold cyan]{_dol_str(net_worth)}[/bold cyan]")
    console.print(Panel(t, expand=False))


# ---------------------------------------------------------------------------
# Commands – Portfolio Management
# ---------------------------------------------------------------------------

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
    if not _confirm(f"Sell {shares} shares of {ticker.upper()} at ${price:.2f}?"):
        raise typer.Exit()
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
    if not _confirm(f"Set initial investment for {account.upper()} to ${amount:,.2f}?"):
        raise typer.Exit()
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
    current_portfolio = portfolio.load()
    for acc in ["USD", "CAD"]:
        portfolio.ensure_account_exists(current_portfolio, acc)
    portfolio.save(current_portfolio)
    accounts = current_portfolio.get("accounts", {})

    if not accounts:
        console.print("[yellow]No accounts found. Use 'add-stock' to get started.[/yellow]")
        return

    with console.status("[bold green]Fetching live exchange rates...[/bold green]"):
        fx_rate = data_client.get_usd_to_cad()

    accounts_to_show = [account.upper()] if account.upper() != "ALL" else list(accounts.keys())

    # Collect all tickers across accounts we'll show
    all_tickers: list[str] = []
    for acc_name in accounts_to_show:
        if acc_name in accounts:
            all_tickers.extend(accounts[acc_name].get("holdings", {}).keys())
    all_tickers = list(dict.fromkeys(all_tickers))  # dedupe, preserve order

    # Batch-fetch all live prices in parallel
    prices: dict[str, tuple[float, float]] = {}
    if all_tickers:
        with console.status("[bold green]Fetching live prices...[/bold green]"):
            prices = data_client.get_current_prices_batch(all_tickers)

    grand = {"value": 0.0, "cost": 0.0, "cash": 0.0, "initial": 0.0, "day_chg": 0.0}

    for acc_name in accounts_to_show:
        if acc_name not in accounts:
            console.print(f"[red]Account '{acc_name}' not found.[/red]")
            continue

        acc_data = accounts[acc_name]
        holdings = acc_data.get("holdings", {})
        cash = acc_data.get("cash", 0.0)
        initial_cash = acc_data.get("initial_cash", 0.0)
        multiplier = fx_rate if acc_name == "USD" else 1.0

        grand["cash"] += cash * multiplier
        grand["initial"] += initial_cash * multiplier

        if not holdings:
            console.print(f"\n[bold yellow]--- {acc_name} Account ---[/bold yellow]")
            console.print(f"No holdings. Cash: ${cash:,.2f}")
            continue

        table = _build_account_table(acc_name, multiplier)
        acc_cost = acc_market = acc_day = 0.0

        for ticker, data in holdings.items():
            live_price, prev_close = prices.get(ticker, (0.0, 0.0))
            row, metrics = _build_holding_row(
                ticker, data["shares"], data["avg_price"], live_price, prev_close
            )
            table.add_row(*row)
            acc_cost += metrics["cost"]
            acc_market += metrics["value"]
            acc_day += metrics["day_chg"]
            grand["value"] += metrics["value"] * multiplier
            grand["cost"] += metrics["cost"] * multiplier
            grand["day_chg"] += metrics["day_chg"] * multiplier

        console.print(table)
        _print_account_summary(cash, initial_cash, acc_cost, acc_market, acc_day)

    if accounts_to_show:
        _print_global_summary(grand)


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
    if not _confirm(f"Update {account.upper()} cash balance to ${amount:,.2f}?"):
        raise typer.Exit()
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
    label = f"Remove entire position of {ticker.upper()}" if not shares else f"Remove {shares} shares of {ticker.upper()}"
    if not _confirm(f"{label} from {account.upper()}?"):
        raise typer.Exit()
    try:
        portfolio.remove_position(account, ticker, shares)
        if shares:
            console.print(f"[bold green]Removed {shares} shares of {ticker.upper()} from {account.upper()}.[/bold green]")
        else:
            console.print(f"[bold green]Removed entire position of {ticker.upper()} from {account.upper()}.[/bold green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


# ---------------------------------------------------------------------------
# Commands – Analysis & Advice
# ---------------------------------------------------------------------------

@app.command()
def analyze():
    """Analyze current holdings against your risk profile."""
    console.print("[bold blue]Analyzing portfolio...[/bold blue]")

    user_settings = config.load_settings()
    current_portfolio = portfolio.load()

    accounts = current_portfolio.get("accounts", {})
    has_assets = any(
        acc_data.get("holdings") or acc_data.get("cash", 0) > 0
        for acc_data in accounts.values()
    )

    if not has_assets:
        console.print("[yellow]Please add stocks or cash to your portfolio first![/yellow]")
        return

    with console.status("[bold cyan]Consulting AI Advisor...[/bold cyan]"):
        advice = advisor.evaluate_portfolio(current_portfolio, user_settings)

    console.print(advice)

    # -- Risk & Signals panel ------------------------------------------------
    tickers = []
    for acc in accounts.values():
        tickers.extend(acc.get("holdings", {}).keys())
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return

    with console.status("[bold green]Fetching historical prices...[/bold green]"):
        close_df = data_client.get_close_prices(tickers, period="1y")

    if close_df.empty or close_df.shape[0] < 5:
        console.print("[yellow]Not enough price history for risk analysis.[/yellow]")
        return

    # Portfolio-level risk metrics (equal-weight proxy)
    port_ret = close_df.pct_change().dropna()
    port_daily = port_ret.mean(axis=1)

    risk_table = Table(
        title="[bold]Portfolio Risk Metrics (1Y)[/bold]",
        border_style="bright_magenta",
    )
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", justify="right")

    risk_table.add_row("Sharpe Ratio", f"{risk.sharpe_ratio(port_daily):.2f}")
    risk_table.add_row("Sortino Ratio", f"{risk.sortino_ratio(port_daily):.2f}")
    risk_table.add_row("Max Drawdown", f"{risk.max_drawdown(close_df.mean(axis=1)):.2%}")
    risk_table.add_row("VaR (95%)", f"{risk.historical_var(port_daily):.2%}")
    risk_table.add_row("CVaR (95%)", f"{risk.cvar(port_daily):.2%}")
    console.print(risk_table)

    # Per-ticker technical signals
    for ticker in tickers:
        if ticker not in close_df.columns:
            continue
        hist = data_client.get_price_history(ticker, period="1y")
        if hist.empty or hist.shape[0] < 50:
            continue
        indicators.compute_indicators(hist)
        sigs = indicators.interpret_signals(hist)
        if not sigs:
            continue
        sig_table = Table(
            title=f"[bold]{ticker} Technical Signals[/bold]",
            border_style="bright_cyan",
        )
        sig_table.add_column("Indicator", style="cyan")
        sig_table.add_column("Value", justify="right")
        sig_table.add_column("Signal", justify="right")
        for s in sigs:
            sig_table.add_row(s["name"], s["value"], s["signal"])
        console.print(sig_table)


@app.command("optimize-portfolio")
def optimize_portfolio(
    objective: str = typer.Option(
        "max-sharpe", "--objective", "-o",
        help="Optimization objective: max-sharpe, min-volatility, efficient-risk",
    ),
    target_volatility: float = typer.Option(
        None, "--target-volatility", "-t",
        help="Target annual volatility (for efficient-risk objective).",
    ),
):
    """Suggest optimal portfolio weights using mean-variance optimization."""
    current_portfolio = portfolio.load()
    accounts = current_portfolio.get("accounts", {})

    tickers = []
    for acc in accounts.values():
        tickers.extend(acc.get("holdings", {}).keys())
    tickers = list(dict.fromkeys(tickers))

    if len(tickers) < 2:
        console.print("[yellow]Need at least 2 unique tickers to optimize.[/yellow]")
        return

    with console.status("[bold green]Fetching price history...[/bold green]"):
        close_df = data_client.get_close_prices(tickers, period="1y")

    if close_df.empty or close_df.shape[0] < 30:
        console.print("[yellow]Not enough price history (need ~30+ trading days).[/yellow]")
        return

    try:
        result = optimizer.optimize(close_df, objective, target_volatility)
    except (ValueError, Exception) as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        raise typer.Exit()

    # Weights table
    wt = Table(title=f"[bold]Optimal Allocation ({objective})[/bold]", border_style="bright_green")
    wt.add_column("Ticker", style="cyan")
    wt.add_column("Weight", justify="right")
    for t, w in sorted(result["weights"].items(), key=lambda x: -x[1]):
        if w > 0:
            wt.add_row(t, f"{w:.1%}")
    console.print(wt)

    console.print(f"  Expected annual return: [green]{result['expected_return']:.1%}[/green]")
    console.print(f"  Annual volatility:      {result['volatility']:.1%}")
    console.print(f"  Sharpe ratio:           [bold]{result['sharpe']:.2f}[/bold]")

    # Discrete allocation
    total_cash = sum(acc.get("cash", 0.0) for acc in accounts.values())
    if total_cash <= 0:
        return

    with console.status("[bold green]Computing share allocation...[/bold green]"):
        try:
            alloc = optimizer.discrete_allocation(result["weights"], close_df, total_cash)
        except Exception:
            return

    if alloc["allocations"]:
        at = Table(title=f"[bold]Discrete Allocation (${total_cash:,.0f} available)[/bold]")
        at.add_column("Ticker", style="cyan")
        at.add_column("Shares", justify="right")
        at.add_column("Approx Cost", justify="right")
        for t, shares in alloc["allocations"].items():
            latest = close_df[t].iloc[-1] if t in close_df.columns else 0
            at.add_row(t, str(shares), f"${shares * latest:,.0f}")
        at.add_row("[dim]Leftover[/dim]", "", f"${alloc['leftover']:,.0f}")
        console.print(at)


@app.command()
def predict(
    ticker: str = typer.Argument(..., help="Ticker symbol to predict"),
    horizon: int = typer.Option(
        1, "--horizon", "-h", help="Prediction horizon in trading days (1 or 5)"
    ),
    retrain: bool = typer.Option(
        False, "--retrain", "-r", help="Force model retraining"
    ),
    max_age: int = typer.Option(
        7, "--max-age", help="Max model age in days before retrain"
    ),
):
    """Predict next-day/week directional movement for a ticker."""
    ticker = ticker.upper()
    console.print(f"[bold blue]Predicting {ticker} ({horizon}-day horizon)...[/bold blue]")

    # Fetch price history
    with console.status("[bold green]Fetching price history...[/bold green]"):
        ohlcv = data_client.get_price_history(ticker, period="2y")

    if ohlcv.empty or ohlcv.shape[0] < 60:
        console.print("[yellow]Not enough price history (need ~60+ days).[/yellow]")
        raise typer.Exit()

    # Build features
    with console.status("[bold cyan]Building features...[/bold cyan]"):
        feat = features.build_features(ohlcv)
        target = features.build_target(ohlcv["Close"], horizon=horizon)

    # Check if we need to train
    stale = ml_model.is_stale(ticker, horizon, max_age_days=max_age)
    if retrain or stale:
        reason = "forced retrain" if retrain else "model missing or stale"
        console.print(f"[yellow]Training model ({reason})...[/yellow]")

        mask = target.notna() & feat.notna().all(axis=1)
        X_train = feat.loc[mask]
        y_train = target.loc[mask]

        if len(X_train) < 30:
            console.print("[red]Not enough labeled data to train (need 30+ rows).[/red]")
            raise typer.Exit()

        with console.status("[bold green]Training LightGBM...[/bold green]"):
            result = ml_model.train(X_train, y_train)

        ml_model.save_model(result["model"], ticker, horizon, metadata={
            "cv_accuracy": result["cv_accuracy"],
            "n_train": result["n_train"],
        })
        console.print(
            f"  CV accuracy: [bold]{result['cv_accuracy']:.1%}[/bold] "
            f"(walk-forward, {len(result['fold_accuracies'])} folds)"
        )
        model = result["model"]
    else:
        payload = ml_model.load_model(ticker, horizon)
        model = payload["model"]
        age_days = (time.time() - payload["trained_at"]) / 86400
        console.print(f"  Using cached model ({age_days:.0f} days old)")

    # Predict on latest row
    latest_feat = feat.iloc[[-1]].dropna(axis=1)
    if latest_feat.empty:
        console.print("[yellow]Latest row has no valid features.[/yellow]")
        raise typer.Exit()

    pred = ml_model.predict(model, latest_feat.iloc[0])

    # Display result
    prob = pred["probability_up"]
    label = pred["label"]
    conf = pred["confidence"]

    if label == "up":
        console.print(
            f"\n[bold green]Signal: UP[/bold green]  "
            f"(probability {prob:.1%}, confidence {conf:.0%})"
        )
    else:
        console.print(
            f"\n[bold red]Signal: DOWN[/bold red]  "
            f"(probability {1 - prob:.1%}, confidence {conf:.0%})"
        )

    # Load saved accuracy for caveat
    payload = ml_model.load_model(ticker, horizon)
    saved_acc = (payload or {}).get("metadata", {}).get("cv_accuracy")
    if saved_acc is not None:
        console.print(
            f"\n[dim]Model holdout accuracy: {saved_acc:.1%} — "
            f"treat as a weak signal, not a recommendation.[/dim]"
        )


@app.command()
def market_update():
    """Get general market recommendations based on today's news."""
    with console.status("[bold green]Fetching latest market news...[/bold green]"):
        news = data_client.get_macro_news()

    with console.status("[bold cyan]Analyzing sentiment...[/bold cyan]"):
        recommendations = advisor.generate_market_advice(news)

    console.print(Panel(recommendations, title="Market Update", expand=False))


@app.command()
def portfolio_news():
    """Get recommendations for your specific stocks based on recent news."""
    current_portfolio = portfolio.load()
    tickers = []
    for acc in current_portfolio.get("accounts", {}).values():
        tickers.extend(acc.get("holdings", {}).keys())

    if not tickers:
        console.print("[yellow]Your portfolio is empty. Nothing to analyze.[/yellow]")
        return

    console.print(f"Fetching news for: [bold cyan]{', '.join(tickers)}[/bold cyan]\n")

    with console.status("[bold green]Fetching news...[/bold green]"):
        news_batch = data_client.get_ticker_news_batch(tickers)

    for ticker in tickers:
        news = news_batch.get(ticker, [])
        advice, headline_scores = advisor.analyze_ticker_sentiment(ticker, news)
        console.print(f"[bold]{ticker} Update:[/bold] {advice}")

        if headline_scores:
            st = Table(show_header=True, border_style="bright_cyan")
            st.add_column("Headline", max_width=50)
            st.add_column("Label", justify="center")
            st.add_column("Score", justify="right")
            for article, sc in zip(news[:10], headline_scores):
                label_color = {"positive": "green", "negative": "red"}.get(
                    sc["label"], "yellow"
                )
                st.add_row(
                    article.get("title", "")[:50],
                    f"[{label_color}]{sc['label']}[/{label_color}]",
                    f"{sc['compound']:+.2f}",
                )
            if len(headline_scores) > 1:
                avg = sum(s["compound"] for s in headline_scores) / len(
                    headline_scores
                )
                st.add_row(
                    "[dim]Aggregate[/dim]", "", f"[bold]{avg:+.2f}[/bold]"
                )
            console.print(st)

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
            console.print(f"[red]Error: Your allocation totals {total}%. It must equal exactly 100%.[/red]")
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


@app.command()
def research(ticker: str):
    """Get a deep-dive analyst report and action plan for a specific stock."""
    current_portfolio = portfolio.load()
    with console.status(f"[bold cyan]Performing Ultimate Deep-Dive for {ticker.upper()}...[/bold cyan]"):
        report_md = advisor.generate_stock_report(ticker, current_portfolio)

    console.print(
        Panel(
            Markdown(report_md),
            title=f"📈 Senior Analyst Report: {ticker.upper()}",
            border_style="bright_magenta",
        )
    )


# ---------------------------------------------------------------------------
# Commands – New Features (P3)
# ---------------------------------------------------------------------------

@app.command()
def export(
    account: str = typer.Option("ALL", "--account", "-a", help="Account to export"),
):
    """Export portfolio holdings to CSV."""
    current_portfolio = portfolio.load()
    accounts = current_portfolio.get("accounts", {})
    accs = [account.upper()] if account.upper() != "ALL" else list(accounts.keys())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Account", "Ticker", "Shares", "Avg Price"])

    for acc_name in accs:
        for ticker, data in accounts.get(acc_name, {}).get("holdings", {}).items():
            writer.writerow([acc_name, ticker, data["shares"], data["avg_price"]])

    csv_content = output.getvalue()
    console.print(csv_content)
    console.print(f"[green]Exported {len(accs)} account(s) to CSV above.[/green]")


@app.command()
def dividends():
    """Show projected annual dividend income from your holdings."""
    current_portfolio = portfolio.load()
    total_projected = 0.0
    table = Table(title="Projected Annual Dividend Income")
    table.add_column("Ticker", style="cyan")
    table.add_column("Shares", justify="right")
    table.add_column("Div/Yield", justify="right")
    table.add_column("Annual Income", justify="right")

    with console.status("[bold green]Fetching dividend data...[/bold green]"):
        for acc in current_portfolio.get("accounts", {}).values():
            for ticker, data in acc.get("holdings", {}).items():
                info = data_client.get_ticker_info(ticker)
                div_yield = info.get("dividendYield", 0) or 0
                div = info.get("dividendRate", 0) or 0

                if div > 0:
                    annual = div * data["shares"]
                    total_projected += annual
                    table.add_row(
                        ticker, str(data["shares"]),
                        f"{div_yield*100:.2f}%" if div_yield else "N/A",
                        f"${annual:,.2f}",
                    )

    console.print(table)
    console.print(f"\n[bold]Total Projected Annual Income:[/bold] [green]${total_projected:,.2f}[/green]")


@app.command()
def rebalance():
    """Get AI-powered rebalancing suggestions for your portfolio."""
    current_portfolio = portfolio.load()
    user_settings = config.load_settings()

    accounts = current_portfolio.get("accounts", {})
    has_assets = any(
        acc_data.get("holdings") or acc_data.get("cash", 0) > 0
        for acc_data in accounts.values()
    )
    if not has_assets:
        console.print("[yellow]Your portfolio is empty. Nothing to rebalance.[/yellow]")
        return

    with console.status("[bold cyan]Generating rebalancing suggestions...[/bold cyan]"):
        suggestions = advisor.evaluate_portfolio(current_portfolio, user_settings)

    console.print(Panel(suggestions, title="🧹 Rebalancing Suggestions", border_style="green"))


@app.command()
def tui():
    """Launch the Textual Terminal UI dashboard."""
    from tui import StockDashboard
    StockDashboard().run()


if __name__ == "__main__":
    app()
