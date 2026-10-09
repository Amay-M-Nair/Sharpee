"""Rolling-PCA factor model and the residual panel every strategy reads.

Each month, on its first trading day k, a factor model is fitted on returns
from days [k-W, k-1] for the stocks eligible on day k-1. It defines

    Q    (n, K)  eigenportfolio weights, v_k / sigma
    B    (n, K)  OLS loadings of each stock on the eigenportfolio returns
    Phi  (n, n)  I - B Q^T, so that epsilon(t) = Phi R(t)

Row i of Phi is a tradable portfolio (long stock i, short its factor hedge),
which is what lets the portfolio layer turn residual positions into stock
weights with x = Phi^T w.

On every decision date t, the trailing residual history is computed with the
Phi in force on t, so a window never mixes two factor models and nothing
after t is used.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def factor_model(r_win: np.ndarray, n_factors: int):
    """r_win: (W, n) simple returns -> Phi (n, n), Q (n, K), B (n, K)."""
    mu = r_win.mean(0)
    sd = r_win.std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = (r_win - mu) / sd
    corr = z.T @ z / (len(z) - 1)

    # eigh returns ascending eigenvalues; keep the top K
    _, vecs = np.linalg.eigh(corr)
    v = vecs[:, ::-1][:, :n_factors]
    v = v * np.where(v.sum(0) < 0, -1.0, 1.0)  # cosmetic: positive-sum eigenvectors

    q = v / sd[:, None]
    f = r_win @ q                               # (W, K) eigenportfolio returns
    x = np.column_stack([np.ones(len(f)), f])
    coef, *_ = np.linalg.lstsq(x, r_win, rcond=None)
    b = coef[1:].T                              # intercept stays inside the residual
    phi = np.eye(len(q)) - b @ q.T
    return phi, q, b


@dataclass
class ResidualPanel:
    """Everything a strategy may see on each decision date, in fixed universe slots.

    T decision dates, M monthly refits, N universe slots, G tickers overall.
    Slots holding -1 in `universe` are empty (fewer than N eligible stocks).
    """

    dates: pd.DatetimeIndex   # (T,)
    tickers: np.ndarray       # (G,)
    month: np.ndarray         # (T,) refit in force on each date
    month_start: np.ndarray   # (M,) first date index of each refit
    universe: np.ndarray      # (M, N) global ticker ids, -1 = empty slot
    phi: np.ndarray           # (M, N, N)
    eps: np.ndarray           # (sum of rows, N) residual returns, one block per refit
    eps_offset: np.ndarray    # (M,) first row of each refit's block
    tradable: np.ndarray      # (T, N) slot filled and the stock has a price on t
    r_next: np.ndarray        # (T, G) R(t+1), NaN -> 0. Realized P&L only, never a feature
    market_next: np.ndarray   # (T,) benchmark R(t+1)
    max_lookback: int

    @property
    def n_slots(self) -> int:
        return self.universe.shape[1]

    def windows(self, idx, length: int) -> np.ndarray:
        """Residual returns over the `length` days ending at each date: (len(idx), N, length)."""
        if length > self.max_lookback:
            raise ValueError(f"length {length} > max_lookback {self.max_lookback}")
        out = np.empty((len(idx), self.n_slots, length), dtype=np.float32)
        for j, i in enumerate(idx):
            m = self.month[i]
            row = self.eps_offset[m] + self.max_lookback - 1 + (i - self.month_start[m])
            out[j] = self.eps[row - length + 1: row + 1].T
        return out

    def index_of(self, start, end) -> np.ndarray:
        """Date indices with start <= date <= end."""
        d = self.dates
        return np.flatnonzero((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)))

    def save(self, path: Path):
        np.savez_compressed(
            path, dates=self.dates.values.astype("datetime64[ns]"), tickers=self.tickers.astype(str),
            month=self.month, month_start=self.month_start, universe=self.universe, phi=self.phi,
            eps=self.eps, eps_offset=self.eps_offset, tradable=self.tradable, r_next=self.r_next,
            market_next=self.market_next, max_lookback=self.max_lookback)

    @classmethod
    def load(cls, path: Path) -> "ResidualPanel":
        z = np.load(path)
        return cls(dates=pd.DatetimeIndex(z["dates"]), tickers=z["tickers"], month=z["month"],
                   month_start=z["month_start"], universe=z["universe"], phi=z["phi"], eps=z["eps"],
                   eps_offset=z["eps_offset"], tradable=z["tradable"], r_next=z["r_next"],
                   market_next=z["market_next"], max_lookback=int(z["max_lookback"]))


def build_panel(md, elig: pd.DataFrame, n_factors: int, pca_window: int, max_lookback: int,
                universe_size: int, min_universe: int = 30) -> ResidualPanel:
    """Fit one factor model per month and lay out the residual panel."""
    returns = md.returns
    r = returns.to_numpy(np.float64)
    has_price = md.close.notna().to_numpy()
    e = elig.reindex(index=returns.index, columns=returns.columns, fill_value=False).to_numpy(bool)
    dates = returns.index
    n = universe_size

    key = dates.year * 12 + dates.month
    firsts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    firsts = firsts[firsts >= max(pca_window, max_lookback) + 1]
    # start once the eligible set is large enough to fit K factors sensibly
    big = np.array([e[k - 1].sum() >= min_universe for k in firsts])
    if not big.any():
        raise ValueError("no month has enough eligible stocks")
    firsts = firsts[np.argmax(big):]
    t0 = firsts[0]

    universe, phis, blocks, offsets, month_start = [], [], [], [], []
    tradable = np.zeros((len(dates) - t0, n), dtype=bool)
    month = np.empty(len(dates) - t0, dtype=np.int32)
    row = 0
    for m, k in enumerate(firsts):
        k_next = firsts[m + 1] if m + 1 < len(firsts) else len(dates)
        ids = np.flatnonzero(e[k - 1])[:n]
        if len(ids) <= n_factors + 1:
            raise ValueError(f"{dates[k].date()}: only {len(ids)} eligible stocks")
        phi_small, _, _ = factor_model(np.nan_to_num(r[k - pca_window:k, ids]), n_factors)

        slots = np.full(n, -1, dtype=np.int64)
        slots[:len(ids)] = ids
        phi = np.zeros((n, n), dtype=np.float32)
        phi[:len(ids), :len(ids)] = phi_small

        # residuals for days k-L+1 .. k_next-1, all with this month's Phi
        r_ext = np.nan_to_num(r[k - max_lookback + 1:k_next, ids])
        block = np.zeros((len(r_ext), n), dtype=np.float32)
        block[:, :len(ids)] = r_ext @ phi_small.T

        universe.append(slots)
        phis.append(phi)
        blocks.append(block)
        offsets.append(row)
        row += len(block)
        month_start.append(k - t0)
        month[k - t0:k_next - t0] = m
        tradable[k - t0:k_next - t0, :len(ids)] = has_price[k:k_next][:, ids]

    r_next = np.zeros((len(dates) - t0, r.shape[1]), dtype=np.float32)
    r_next[:-1] = np.nan_to_num(r[t0 + 1:])
    market = md.market.reindex(dates).to_numpy(np.float64)
    market_next = np.zeros(len(dates) - t0, dtype=np.float32)
    market_next[:-1] = np.nan_to_num(market[t0 + 1:])

    return ResidualPanel(
        dates=dates[t0:], tickers=np.asarray(returns.columns, dtype=str), month=month,
        month_start=np.asarray(month_start), universe=np.stack(universe), phi=np.stack(phis),
        eps=np.concatenate(blocks), eps_offset=np.asarray(offsets), tradable=tradable,
        r_next=r_next, market_next=market_next, max_lookback=max_lookback)
