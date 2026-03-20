import yfinance as yf
import requests


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
