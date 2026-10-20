"""Chronological backtest: scores -> constrained weights -> daily gross/net returns."""

import numpy as np
import pandas as pd
import torch

from ..portfolio.construction import build_weights, to_global
from ..portfolio.risk import exposures
from .costs import portfolio_returns
from .realistic import holding_costs


@torch.no_grad()
def weights_for(panel, idx: np.ndarray, scores: np.ndarray, cap: float | None,
                center: bool = True) -> np.ndarray:
    """Slot scores (len(idx), N) -> ticker-space weights (len(idx), G), one month at a time."""
    x_glob = np.zeros((len(idx), len(panel.tickers)), dtype=np.float32)
    months = panel.month[idx]
    for m in np.unique(months):
        rows = np.flatnonzero(months == m)
        x = build_weights(torch.from_numpy(scores[rows].astype(np.float32)),
                          torch.from_numpy(panel.phi[m]),
                          torch.from_numpy(panel.tradable[idx[rows]]), cap=cap, center=center)
        uni = torch.from_numpy(np.broadcast_to(panel.universe[m], x.shape).copy())
        x_glob[rows] = to_global(x, uni, len(panel.tickers)).numpy()
    return x_glob


def run_backtest(panel, idx: np.ndarray, scores: np.ndarray, cost_bps: float = 5.0,
                 delay: int = 0, cap: float | None = 0.02, center: bool = True,
                 spread: np.ndarray | None = None, borrow_rate: float = 0.0) -> pd.DataFrame:
    """One row per decision date t in idx (contiguous), holding the return realized over t -> t+1.

    delay=1 is the execution-lag check: weights formed at t are only traded at
    t+1 and so earn R(t+2). center=False for rule-based positions (OU), see
    portfolio.construction.

    By default every trade costs cost_bps. Passing spread, a (T, G) cost per unit
    traded on the panel's dates (backtest.realistic), replaces it with per-stock
    costs, and borrow_rate adds an annual fee on short notional.
    """
    if len(idx) > 1 and np.any(np.diff(idx) != 1):
        raise ValueError("backtest dates must be contiguous")
    x = weights_for(panel, idx, scores, cap, center)
    if delay:
        x = np.vstack([np.zeros((delay, x.shape[1]), dtype=x.dtype), x[:-delay]])

    gross, traded, net = portfolio_returns(torch.from_numpy(x), torch.from_numpy(panel.r_next[idx]),
                                           cost_bps * 1e-4)
    gross, traded = gross.numpy(), traded.numpy()
    if spread is None and borrow_rate == 0:
        spread_cost, borrow_cost = (gross - net.numpy()), np.zeros_like(gross)
    else:
        per_unit = np.full_like(x, cost_bps * 1e-4) if spread is None else spread[idx]
        spread_cost, borrow_cost = holding_costs(x, per_unit, borrow_rate)
    cost = spread_cost + borrow_cost
    frame = pd.DataFrame({
        "gross": gross, "cost": cost, "net": gross - cost, "spread_cost": spread_cost,
        "borrow_cost": borrow_cost, "turnover": traded, "market": panel.market_next[idx],
        **exposures(x),
    }, index=panel.dates[idx])
    frame.index.name = "date"
    # the last panel date has no realized next-day return
    if idx[-1] == len(panel.dates) - 1:
        frame = frame.iloc[:-1]
    return frame
