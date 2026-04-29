from dotenv import load_dotenv


import plotext as plt
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Markdown, Static, Label
from textual.containers import Horizontal, Vertical, VerticalScroll

from src import portfolio, advisor

load_dotenv()


class PlotextChart(Static):
    """A custom widget to render Plotext charts safely in Textual."""

    def on_mount(self) -> None:
        self.render_chart()

    def render_chart(self):
        history = portfolio.get_history()

        # Safe dimensions fallback to prevent plotext crashes
        w = max(10, self.size.width or 60)
        h = max(5, self.size.height or 15)

        plt.clf()
        plt.theme("dark")

        if not history:
            plt.title("No Net Worth History")
            plt.plotsize(w, h)
            self.update(plt.build())
            return

        dates = list(history.keys())
        values = list(history.values())

        plt.date_form("Y-m-d")
        plt.plot(dates, values, marker="dot", color="cyan")
        plt.title("Net Worth History (CAD)")
        plt.plotsize(w, h)

        self.update(plt.build())

    def on_resize(self, event) -> None:
        self.render_chart()


class StockDashboard(App):
    CSS = """
    #main_container { height: 100%; }
    
    #left_panel { 
        width: 35%; 
        height: 100%; 
        border-right: solid cyan; 
    }
    
    #right_panel { 
        width: 65%; 
        height: 100%; 
    }
    
    #chart_area { 
        height: 40%; 
        border-bottom: solid green; 
        padding: 1;
    }
    
    #ai_report_container { 
        height: 60%; 
        padding: 1 2; 
    }
    
    #portfolio_summary {
        padding: 1;
        text-align: center;
        text-style: bold;
        color: lime;
        border-bottom: dashed cyan;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_container"):
            # Left side: Holdings
            with Vertical(id="left_panel"):
                yield Label("Loading Portfolio...", id="portfolio_summary")
                yield DataTable(id="portfolio_table", cursor_type="row")

            # Right side: Chart and AI Report
            with Vertical(id="right_panel"):
                yield PlotextChart(id="chart_area")
                with VerticalScroll(id="ai_report_container"):
                    yield Markdown(
                        "# 👈 Select a stock to generate an AI Report...",
                        id="ai_report",
                    )
        yield Footer()

    def on_mount(self) -> None:
        # Log today's net worth upon opening
        portfolio.log_net_worth()

        # Update the summary label
        current_portfolio = portfolio.load()
        history = portfolio.get_history()
        if history:
            latest_val = list(history.values())[-1]
            self.query_one("#portfolio_summary", Label).update(
                f"Total Net Worth: ${latest_val:,.2f} CAD"
            )

        # Populate the DataTable
        table = self.query_one(DataTable)
        table.add_columns("Ticker", "Shares", "Avg Price")

        for acc_name, acc_data in current_portfolio.get("accounts", {}).items():
            for ticker, data in acc_data.get("holdings", {}).items():
                # Add a visual indicator of the account type
                display_ticker = f"{ticker} ({acc_name})"
                table.add_row(
                    display_ticker,
                    str(data["shares"]),
                    f"${data['avg_price']:.2f}",
                    key=ticker,  # Keep the raw ticker as the internal key
                )

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Fired when a user clicks a row in the table."""
        ticker = event.row_key.value
        report_widget = self.query_one("#ai_report", Markdown)

        # Show loading state
        report_widget.update(
            f"## ⏳ Fetching data and generating AI report for **{ticker}**...\n*(This may take a few seconds)*"
        )

        # Fire off the background worker
        self.fetch_report_worker(ticker)

    @work(thread=True)
    def fetch_report_worker(self, ticker: str):
        """Runs in a background thread to prevent UI freezing."""
        current_portfolio = portfolio.load()
        try:
            report_md = advisor.generate_stock_report(ticker, current_portfolio)
        except Exception as e:
            report_md = f"❌ **Analysis Failed:** {str(e)}"

        # Safely update the UI from the background thread
        self.call_from_thread(self.update_report_ui, report_md)

    def update_report_ui(self, report_md: str):
        """Called by the worker to update the UI on the main thread."""
        report_widget = self.query_one("#ai_report", Markdown)
        report_widget.update(report_md)


if __name__ == "__main__":
    app = StockDashboard()
    app.run()
