"""Avellaneda-Lee style OU mean reversion on cumulative residuals.

Reproduced from Avellaneda & Lee (2010): the AR(1) fit of the cumulative
residual over a 60-day window, the kappa filter, and the s-score open/close
thresholds. Simplified: PCA eigenportfolios refit monthly instead of daily,
no drift (alpha) adjustment to the s-score, and positions pass through the
shared portfolio layer (centering, Phi^T w, gross 1, cap) like every other model.
"""

import numpy as np


def ou_fit(windows: np.ndarray, periods: int = 252):
    """
    Args:
        windows: (..., W) residual returns, oldest first

    Returns:
        s, kappa, valid: each (...). s = (x_last - mu) / sigma_eq on the
        cumulative residual x; valid where 0 < b < 1 (an AR(1) that mean-reverts)
    """
    x = np.cumsum(windows.astype(np.float64), axis=-1)
    x0, x1 = x[..., :-1], x[..., 1:]
    m0, m1 = x0.mean(-1, keepdims=True), x1.mean(-1, keepdims=True)
    var0 = ((x0 - m0) ** 2).mean(-1)
    cov = ((x0 - m0) * (x1 - m1)).mean(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = cov / var0
        a = m1[..., 0] - b * m0[..., 0]
        zeta = x1 - (a[..., None] + b[..., None] * x0)
        valid = (b > 0) & (b < 1) & (var0 > 0)
        b_ok = np.where(valid, b, 0.5)
        kappa = -np.log(b_ok) * periods
        mu = a / (1 - b_ok)
        sigma_eq = np.sqrt(zeta.var(-1) / (1 - b_ok ** 2))
        s = (x[..., -1] - mu) / sigma_eq
    valid &= np.isfinite(s) & (sigma_eq > 0)
    return np.where(valid, s, 0.0), np.where(valid, kappa, 0.0), valid


def ou_positions(panel, idx: np.ndarray, window: int = 60, entry: float = 1.25,
                 exit_short: float = 0.75, exit_long: float = -0.50,
                 kappa_min: float = 252 / 30, chunk: int = 256) -> np.ndarray:
    """Residual positions in {-1, 0, +1} for each slot on each date in idx (contiguous).

    Open long when s < -entry, open short when s > entry; close a long when
    s > exit_long, close a short when s < exit_short. Only stocks with kappa >
    kappa_min (half-life under ~21 trading days by default) may hold a
    position. Positions persist across days by ticker, so a stock that stays
    in the universe across a monthly refit keeps its position.
    """
    held = np.zeros(len(panel.tickers), dtype=np.int8)
    out = np.zeros((len(idx), panel.n_slots), dtype=np.float32)
    for c in range(0, len(idx), chunk):
        part = idx[c:c + chunk]
        s_all, kappa_all, valid_all = ou_fit(panel.windows(part, window))
        for j, i in enumerate(part):
            uni = panel.universe[panel.month[i]]
            filled = uni >= 0
            ok = valid_all[j] & (kappa_all[j] > kappa_min) & panel.tradable[i]
            s = s_all[j]

            cur = np.where(filled, held[np.where(filled, uni, 0)], 0)
            new = cur.copy()
            new[(cur == 0) & (s < -entry)] = 1
            new[(cur == 0) & (s > entry)] = -1
            new[(cur == 1) & (s > exit_long)] = 0
            new[(cur == -1) & (s < exit_short)] = 0
            new[~ok] = 0

            held[:] = 0
            held[uni[filled]] = new[filled]
            out[c + j] = new
    return out
