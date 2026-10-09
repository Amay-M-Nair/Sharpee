"""Synthetic markets with known structure, for tests and pipeline sanity checks.

Returns follow R = beta f + epsilon. With `ar` set, each stock's cumulative
residual is an AR(1) (a discretized OU process), so residual reversal is a
real, learnable edge. With `ar=None` the cumulative residual is a random walk
and no strategy should beat zero before costs.
"""

import numpy as np
import pandas as pd

from .preprocess import MarketData


def make_market(n_stocks: int = 60, n_days: int = 1500, n_factors: int = 3,
                ar: float | None = 0.9, resid_vol: float = 0.01, seed: int = 0,
                start: str = "2008-01-01") -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days + 1)

    f = rng.normal(0.0003, 0.01, size=(n_days, n_factors))
    beta = rng.normal(0.0, 0.5, size=(n_stocks, n_factors))
    beta[:, 0] += 1.0  # first factor acts like the market

    if ar is None:
        eps = rng.normal(0.0, resid_vol, size=(n_days, n_stocks))
    else:
        # residual "price" X follows X(t+1) = ar X(t) + noise; its increments are the residual returns
        x = np.zeros((n_days + 1, n_stocks))
        noise = rng.normal(0.0, resid_vol, size=(n_days, n_stocks))
        for t in range(n_days):
            x[t + 1] = ar * x[t] + noise[t]
        eps = np.diff(x, axis=0)

    r = f @ beta.T + eps
    close = 100.0 * np.vstack([np.ones(n_stocks), np.cumprod(1.0 + r, axis=0)])
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    close = pd.DataFrame(close, index=dates, columns=tickers)
    volume = pd.DataFrame(1e6, index=dates, columns=tickers)
    member = pd.DataFrame(True, index=dates, columns=tickers)
    market = pd.Series(np.r_[np.nan, f[:, 0]], index=dates)
    return MarketData(close, volume, member, market)
