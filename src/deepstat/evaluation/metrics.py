"""Performance metrics on daily backtest frames. Annualization is arithmetic (x252, x sqrt 252)."""

import numpy as np
import pandas as pd

PERIODS = 252


def sharpe(r) -> float:
    """Annualized mean / std of daily returns. R_net is already an excess return."""
    r = np.asarray(r, dtype=np.float64)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(PERIODS)) if sd > 0 else 0.0


def max_drawdown(r) -> float:
    """Largest peak-to-trough fall of the compounded equity curve (negative number)."""
    equity = np.cumprod(1.0 + np.asarray(r, dtype=np.float64))
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float((equity / peak - 1.0).min())


def summarize(frame: pd.DataFrame) -> dict:
    """One row of the results table for a backtest frame from engine.run_backtest."""
    net, gross, mkt = frame["net"], frame["gross"], frame["market"]
    var_m = mkt.var()
    beta = float(net.cov(mkt) / var_m) if var_m > 0 else 0.0
    return {
        "ann_return": net.mean() * PERIODS,
        "ann_vol": net.std() * np.sqrt(PERIODS),
        "sharpe": sharpe(net),
        "gross_sharpe": sharpe(gross),
        "max_drawdown": max_drawdown(net),
        "avg_turnover": frame["turnover"].mean(),
        "cost_drag": (gross.mean() - net.mean()) * PERIODS,
        "beta": beta,
        "corr_market": float(net.corr(mkt)),
        "win_rate": float((net > 0).mean()),
        "net_exposure_mean": frame["net_exposure"].mean(),
        "net_exposure_min": frame["net_exposure"].min(),
        "net_exposure_max": frame["net_exposure"].max(),
        "n_days": len(frame),
    }
