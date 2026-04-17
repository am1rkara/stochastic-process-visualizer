import numpy as np
from fitting.fitting import GBMParams, OUParams

def simulate_gbm(params: GBMParams, S0: float, n_paths: int = 1000, n_steps: int = 60):
    dt = params.dt
    mu, sigma = params.mu, params.sigma
    Z = np.random.randn(n_paths, n_steps)
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(log_paths)
    return np.hstack([np.full((n_paths, 1), S0), paths])

def simulate_ou(params: OUParams, S0: float, n_paths: int = 1000, n_steps: int = 60):
    dt = params.dt
    theta, mu, sigma = params.theta, params.mu, params.sigma
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    Z = np.random.randn(n_paths, n_steps)
    for t in range(n_steps):
        paths[:, t+1] = (paths[:, t]
                         + theta * (mu - paths[:, t]) * dt
                         + sigma * np.sqrt(dt) * Z[:, t])
    return paths

def compute_cone(paths: np.ndarray, percentiles: list = [10, 25, 50, 75, 90]):
    return {p: np.percentile(paths, p, axis=0) for p in percentiles}