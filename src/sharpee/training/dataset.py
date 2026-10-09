"""Inputs for the portfolio-level loss: contiguous blocks of days covering every stock.

A Sharpe ratio is a property of the portfolio's daily return series, and
turnover links consecutive days, so a training example is a block of
(T_days, N_stocks, L) paths, not an independent (L,) sequence.
"""

import numpy as np
import torch


def path_features(windows: np.ndarray) -> np.ndarray:
    """(..., L) residual returns -> cumulative residual path in units of its own volatility.

    Dimensionless like an s-score: sqrt(L) * std is the typical size of the
    path's endpoint. Empty slots (all zeros) stay zero.
    """
    cum = np.cumsum(windows, axis=-1)
    scale = windows.std(axis=-1, keepdims=True) * np.sqrt(windows.shape[-1])
    return (cum / np.maximum(scale, 1e-8)).astype(np.float32)


class DayBlocks:
    """Everything needed to score and account a contiguous run of decision dates, on one device."""

    def __init__(self, panel, idx: np.ndarray, lookback: int, device="cpu"):
        if len(idx) > 1 and np.any(np.diff(idx) != 1):
            raise ValueError("DayBlocks needs contiguous dates")
        self.panel, self.idx, self.lookback = panel, idx, lookback
        self.n_tickers = len(panel.tickers)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.X = t(path_features(panel.windows(idx, lookback)))   # (D, N, L)
        self.month = t(panel.month[idx].astype(np.int64))           # (D,)
        self.phi = t(panel.phi)                                     # (M, N, N)
        self.universe = t(panel.universe)                           # (M, N)
        self.tradable = t(panel.tradable[idx])                      # (D, N)
        self.r_next = t(panel.r_next[idx])                          # (D, G)

    def __len__(self):
        return len(self.idx)

    def slices(self, block: int, rng: np.random.Generator | None = None) -> list[slice]:
        """Cut the dates into contiguous blocks; with rng, the cut points and order vary per epoch.

        A short first or last piece is merged into its neighbour, so every block
        spans at least block // 2 days: a Sharpe ratio over a handful of days is
        noise, and over one day it is undefined.
        """
        n = len(self)
        off = int(rng.integers(block)) if rng is not None else 0
        cuts = [0] + list(range(block - off if off else block, n, block)) + [n]
        min_len = max(2, block // 2)
        if len(cuts) > 2 and cuts[1] - cuts[0] < min_len:
            del cuts[1]
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < min_len:
            del cuts[-2]
        out = [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:])]
        if rng is not None:
            rng.shuffle(out)
        return out

    def batch(self, sl: slice):
        """X (Tb, N, L), phi (Tb, N, N), tradable (Tb, N), universe (Tb, N), r_next (Tb, G)."""
        m = self.month[sl]
        return self.X[sl], self.phi[m], self.tradable[sl], self.universe[m], self.r_next[sl]
