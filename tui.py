import plotext as plt
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Markdown, Static
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from src import portfolio, advisor


class PlotextChart(Static):
    """A custom widget to render Plotext charts in Textual."""

    def on_mount(self) -> None:
        self.render_chart()

    def render_chart(self):
        history = portfolio.get_history()
        if not history:
            self.update("No history data yet. Run the CLI tool to log net worth.")
            return

        dates = list(history.keys())
        values = list(history.values())

        plt.clf()  # Clear previous
        plt.theme("dark")

        plt.date_form("Y-m-d")
        plt.plot(dates, values, marker="dot", color="cyan")
        plt.title("Net Worth History (CAD)")
        plt.plotsize(self.size.width or 60, self.size.height or 15)

        # Build ANSI string and update widget
        ansi_chart = plt.build()
        self.update(ansi_chart)

    def on_resize(self, event) -> None:
        self.render_chart()


class StockDashboard(App):
    CSS = """
    DataTable { width: 1fr; height: 100%; border-right: solid cyan; }
    #right_panel { width: 2fr; height: 100%; padding: 1; }
    #chart_area { height: 40%; border-bottom: solid green; margin-bottom: 1; }
    #ai_report { height: 60%; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield DataTable(id="portfolio_table", cursor_type="row")
            with VerticalScroll(id="right_panel"):
                yield PlotextChart(id="chart_area")
                yield Markdown(
                    "# Select a stock to generate an AI Report...", id="ai_report"
                )
        yield Footer()

    def on_mount(self) -> None:
        # 1. Log today's net worth upon opening the dashboard
        portfolio.log_net_worth()

        # 2. Populate the DataTable
        table = self.query_one(DataTable)
        table.add_columns("Ticker", "Shares", "Avg Price")

        current_portfolio = portfolio.load()
        for acc_name, acc_data in current_portfolio.get("accounts", {}).items():
            for ticker, data in acc_data.get("holdings", {}).items():
                table.add_row(
                    ticker, str(data["shares"]), f"${data['avg_price']:.2f}", key=ticker
                )

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Fired when a user clicks a row in the table."""
        ticker = event.row_key.value
        report_widget = self.query_one("#ai_report", Markdown)

        # Show loading state
        await report_widget.update(
            f"## Fetching data and generating AI report for {ticker}...\n*(This may take a few seconds)*"
        )

        # Run AI generation in a background thread so the UI doesn't freeze
        self.run_worker(self.fetch_report(ticker))

    async def fetch_report(self, ticker: str):
        current_portfolio = portfolio.load()
        report_md = advisor.generate_stock_report(ticker, current_portfolio)
        report_widget = self.query_one("#ai_report", Markdown)
        self.call_from_thread(report_widget.update, report_md)


if __name__ == "__main__":
    app = StockDashboard()
    app.run()
