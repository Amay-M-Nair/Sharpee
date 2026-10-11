"""Non-neural baselines: ridge regression and gradient boosting on the same 30-day residual paths.

Neither can be trained on portfolio Sharpe, so they do what standard machine
learning does: forecast each stock's next-day residual payoff (the return its
hedged bet earns), z-scored across stocks each day, then trade the forecasts
through the same portfolio layer, costs and backtester as every other strategy.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .dataset import path_features

TABULAR = ("ridge", "gbm")


def payoffs(panel, idx: np.ndarray) -> np.ndarray:
    """(len(idx), N) next-day payoff of each slot's hedged bet, (Phi_t R_{t+1})_i, with the Phi in force on t.

    Holding residual weights w on day t earns exactly w . payoff, which is what
    the backtester books for stock weights x = Phi^T w.
    """
    out = np.zeros((len(idx), panel.n_slots), dtype=np.float32)
    months = panel.month[idx]
    for m in np.unique(months):
        rows = np.flatnonzero(months == m)
        uni = panel.universe[m]
        filled = uni >= 0
        r = np.zeros((len(rows), panel.n_slots), dtype=np.float32)
        r[:, filled] = panel.r_next[idx[rows]][:, uni[filled]]
        out[rows] = r @ panel.phi[m].T
    return out


def zscore_rows(a: np.ndarray, mask: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Cross-sectional z-score per day over masked entries, clipped; NaN elsewhere."""
    a = np.where(mask, a.astype(np.float64), np.nan)
    sd = np.nanstd(a, axis=1, keepdims=True)
    z = (a - np.nanmean(a, axis=1, keepdims=True)) / np.where(sd > 0, sd, np.nan)
    return np.clip(z, -clip, clip)


def make_xy(panel, idx: np.ndarray, lookback: int):
    """Rows for every tradable stock-day: X = scaled 30-day path (n, L), y = z-scored next-day payoff (n,)."""
    x = path_features(panel.windows(idx, lookback))
    mask = panel.tradable[idx]
    y = zscore_rows(payoffs(panel, idx), mask)
    keep = mask & np.isfinite(y)
    return x[keep], y[keep]


def fit(name: str, params: dict, x: np.ndarray, y: np.ndarray):
    if name == "ridge":
        model = Ridge(**params)
    elif name == "gbm":
        model = HistGradientBoostingRegressor(**params, early_stopping=False, random_state=0)
    else:
        raise ValueError(f"unknown tabular model {name}")
    return model.fit(x, y)


def predict_scores(model, panel, idx: np.ndarray, lookback: int) -> np.ndarray:
    """(len(idx), N) forecasts as scores; untradable slots get 0 (the portfolio layer ignores them)."""
    x = path_features(panel.windows(idx, lookback))
    d, n, length = x.shape
    s = model.predict(x.reshape(d * n, length)).reshape(d, n)
    return np.where(panel.tradable[idx], s, 0.0).astype(np.float32)
