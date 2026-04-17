import yfinance as yf
import numpy as np
import pandas as pd

def fetch_prices(ticker: str, period: str = "2y") -> pd.Series:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return df["Close"].squeeze().dropna()

def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()

def get_data(ticker: str, period: str = "2y"):
    prices = fetch_prices(ticker, period)
    returns = compute_log_returns(prices)
    return prices, returns