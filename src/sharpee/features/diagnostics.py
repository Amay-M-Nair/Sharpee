"""Residual diagnostics: did factor removal work, and is there mean reversion to trade?"""

import numpy as np
import pandas as pd

from ..strategies.ou_strategy import ou_fit


def daily_residuals(panel, idx) -> np.ndarray:
    """(len(idx), N) residual return on each date, NaN for empty or untradable slots."""
    eps = panel.windows(idx, 1)[..., 0]
    return np.where(panel.tradable[idx], eps, np.nan)


def raw_returns(panel, idx) -> np.ndarray:
    """(len(idx), N) raw return R(t) of the universe stocks; idx must start after the first date."""
    out = np.full((len(idx), panel.n_slots), np.nan, dtype=np.float32)
    for j, i in enumerate(idx):
        uni = panel.universe[panel.month[i]]
        ok = (uni >= 0) & panel.tradable[i]
        out[j, ok] = panel.r_next[i - 1, uni[ok]]
    return out


def _corr_with(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column correlation of a (D, N) with b (D,), ignoring NaNs."""
    ok = ~np.isnan(a)
    out = np.full(a.shape[1], np.nan)
    for j in range(a.shape[1]):
        m = ok[:, j]
        if m.sum() > 20 and a[m, j].std() > 0:
            out[j] = np.corrcoef(a[m, j], b[m])[0, 1]
    return out


def autocorr(a: np.ndarray, lag: int) -> float:
    """Average across stocks of the lag-k autocorrelation of each column."""
    x, y = a[:-lag], a[lag:]
    vals = []
    for j in range(a.shape[1]):
        m = ~np.isnan(x[:, j]) & ~np.isnan(y[:, j])
        if m.sum() > 20 and x[m, j].std() > 0 and y[m, j].std() > 0:
            vals.append(np.corrcoef(x[m, j], y[m, j])[0, 1])
    return float(np.mean(vals)) if vals else np.nan


def residual_report(panel, idx, ou_window: int = 60, kappa_min: float = 252 / 30) -> pd.Series:
    """Headline numbers comparing raw returns with residuals over the dates in idx."""
    idx = idx[idx > 0]
    eps, raw = daily_residuals(panel, idx), raw_returns(panel, idx)
    market = panel.market_next[idx - 1]  # R_market(t)
    s, kappa, valid = ou_fit(panel.windows(idx[::21], ou_window))
    tradable = panel.tradable[idx[::21]]
    half_life = np.log(2) / kappa[valid & tradable] * 252
    return pd.Series({
        "raw: mean |corr| with market": np.nanmean(np.abs(_corr_with(raw, market))),
        "residual: mean |corr| with market": np.nanmean(np.abs(_corr_with(eps, market))),
        "residual / raw variance": np.nanmean(np.nanvar(eps, 0) / np.nanvar(raw, 0)),
        "raw: lag-1 autocorrelation": autocorr(raw, 1),
        "residual: lag-1 autocorrelation": autocorr(eps, 1),
        "OU fits that mean-revert (0<b<1)": valid[tradable].mean(),
        "OU fits passing the kappa filter": (valid & (kappa > kappa_min))[tradable].mean(),
        "median OU half-life (days)": float(np.median(half_life)) if len(half_life) else np.nan,
    })
