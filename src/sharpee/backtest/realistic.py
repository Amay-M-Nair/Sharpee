"""Realistic costs, evaluation only (docs/realistic_costs.md): per-stock spread tiers and borrow fees.

Spread: each day, every index member is ranked by trailing median dollar volume
(the eligibility filter's own point-in-time quantity), and the rank sets its
cost per unit traded. Borrow: a daily fee on short notional. Both stay linear,
so a scenario is just a spread multiplier and a borrow rate.
"""

import numpy as np
import pandas as pd

from ..config import repo_path
from ..data.preprocess import MarketData


def liquidity_rank(md: MarketData, window: int) -> pd.DataFrame:
    """(dates x tickers) rank by trailing `window`-day median dollar volume among index members; NaN off-index."""
    dollar_vol = (md.close * md.volume).rolling(window, min_periods=window).median()
    return dollar_vol.where(md.member).rank(axis=1, ascending=False, method="first")


def spread_from_rank(rank: np.ndarray, tiers, floor_bps: float) -> np.ndarray:
    """Cost per unit traded (a fraction): tiers [(max_rank, bps), ...] in ascending order; floor below them."""
    cost = np.full(rank.shape, floor_bps, dtype=np.float64)
    for max_rank, bps in reversed(tiers):
        cost = np.where(rank <= max_rank, bps, cost)  # NaN ranks compare False and keep the floor
    return (cost * 1e-4).astype(np.float32)


def spread_matrix(panel, cfg: dict) -> np.ndarray:
    """(T, G) cost per unit traded aligned to panel.dates and panel.tickers."""
    md = MarketData.load(repo_path(cfg["processed_dir"]))
    rank = liquidity_rank(md, cfg["dollar_volume_window"]).reindex(index=panel.dates, columns=panel.tickers)
    rc = cfg["realistic_costs"]
    return spread_from_rank(rank.to_numpy(), rc["tiers"], rc["tier_floor_bps"])


def holding_costs(x: np.ndarray, spread: np.ndarray, borrow_rate: float,
                  x_prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-day spread cost sum_i c_i |dx_i| and borrow fee rate/252 * short notional, for weights x (T, G)."""
    prev = np.vstack([np.zeros_like(x[:1]) if x_prev is None else x_prev[None], x[:-1]])
    spread_cost = (spread * np.abs(x - prev)).sum(1)
    borrow_cost = borrow_rate / 252 * np.clip(-x, 0, None).sum(1)
    return spread_cost, borrow_cost
