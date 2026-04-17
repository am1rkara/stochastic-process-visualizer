import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class GBMParams:
    mu: float
    sigma: float
    dt: float = 1/252

@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float
    dt: float = 1/252

def fit_gbm(returns: pd.Series) -> GBMParams:
    dt = 1/252
    mu_log = returns.mean()
    sigma = returns.std()
    mu = mu_log / dt + 0.5 * sigma**2 / dt
    sigma_ann = sigma / np.sqrt(dt)
    return GBMParams(mu=mu, sigma=sigma_ann)

def fit_ou(prices: pd.Series) -> OUParams:
    dt = 1/252
    S = prices.values
    x = S[:-1]
    y = S[1:]
    beta = np.polyfit(x, y, 1)
    alpha, b = beta[1], beta[0]
    theta = -np.log(b) / dt
    mu = alpha / (1 - b)
    residuals = y - (alpha + b * x)
    sigma = residuals.std() / np.sqrt(dt * (1 - b**2) / (2 * theta)) ** 0.5
    return OUParams(theta=theta, mu=mu, sigma=sigma)