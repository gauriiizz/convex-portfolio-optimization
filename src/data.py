from typing import List
import pandas as pd
import yfinance as yf

# Replaced TATAMOTORS with HCLTECH for stable Yahoo Finance price feeds
DEFAULT_NIFTY_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "TITAN.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "HCLTECH.NS", "TATASTEEL.NS", "NTPC.NS"
]

def fetch_historical_prices(
    tickers: List[str] = DEFAULT_NIFTY_UNIVERSE,
    start_date: str = "2021-01-01",
    end_date: str = "2026-06-01"
) -> pd.DataFrame:
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Handle multi-index columns returned by yfinance
    if "Adj Close" in raw.columns:
        prices = raw["Adj Close"]
    elif "Close" in raw.columns:
        prices = raw["Close"]
    else:
        prices = raw

    # Drop only columns that failed completely, forward fill small intraday holidays
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna()
    return prices

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()