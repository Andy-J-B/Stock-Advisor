import yfinance as yf
import requests

import requests


def get_usd_to_cad() -> float:
    """Fetches the latest USD to CAD rate from the Free Currency API."""
    primary_url = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
    fallback_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"

    try:
        try:
            response = requests.get(primary_url, timeout=5)
            response.raise_for_status()
        except:
            response = requests.get(fallback_url, timeout=5)
            response.raise_for_status()

        data = response.json()
        return data["usd"]["cad"]
    except Exception as e:
        # Fallback to a sensible default if the internet is down
        return 1.35


def get_current_price(ticker: str) -> float:
    """
    Fetches the current live price (or latest close) for a given ticker.
    """
    try:
        # yfinance allows us to pull data easily
        stock = yf.Ticker(ticker)

        # .fast_info is significantly faster than downloading historical tables
        current_price = stock.fast_info["lastPrice"]
        return round(current_price, 2)

    except Exception as e:
        # If the ticker is invalid or offline, return 0.0 as a safe fallback
        return 0.0


def get_ticker_news(ticker: str, limit: int = 3) -> list:
    """
    Fetches recent news articles for a specific ticker using yfinance's built-in scraper.
    """
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news

        formatted_news = []
        # Grab only the top few articles based on the limit
        for article in raw_news[:limit]:
            formatted_news.append(
                {
                    "title": article.get("title", "No Title Available"),
                    "publisher": article.get("publisher", "Unknown Publisher"),
                    "link": article.get("link", ""),
                }
            )

        return formatted_news

    except Exception as e:
        return [{"title": f"Could not fetch news for {ticker}.", "publisher": "System"}]


def get_macro_news() -> list:
    """
    Fetches general financial market news.
    Currently returns mock data. You can plug in NewsAPI or Finnhub here later.
    """
    response = requests.get(
        f"https://newsapi.org/v2/top-headlines?category=business&apiKey=70d39de976cc4625bc3929766d7a6720"
    )

    return response.json()["articles"][:5]
