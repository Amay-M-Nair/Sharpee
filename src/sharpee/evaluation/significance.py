"""Is the Sharpe ratio real? PSR, DSR (Bailey & Lopez de Prado) and stationary-bootstrap intervals.

Sharpe ratios passed to PSR/DSR are per period (daily); bootstrap intervals are
reported annualized. See docs/phase3_protocol.md for how each is used.
"""

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329
PERIODS = 252


def daily_sharpe(r) -> float:
    r = np.asarray(r, dtype=np.float64)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def psr(r, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark), allowing for sample length, skewness and fat tails.

    PSR = Phi( (SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) ), g4 non-excess kurtosis.
    """
    r = np.asarray(r, dtype=np.float64)
    sr = daily_sharpe(r)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(len(r) - 1) / np.sqrt(max(denom, 1e-12))))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Daily Sharpe the best of n_trials skill-less strategies would reach by luck alone.

    SR* = sqrt(V) ((1-g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))), g = Euler-Mascheroni.
    """
    if n_trials < 2:
        return 0.0
    n = float(n_trials)
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                    + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def dsr(r, n_trials: int, var_sr: float) -> float:
    """PSR with the benchmark raised to the expected best-of-N luck Sharpe."""
    return psr(r, expected_max_sharpe(n_trials, var_sr))


def _stationary_indices(t: int, n_boot: int, mean_block: float, rng) -> np.ndarray:
    """(n_boot, t) resampling indices: blocks of geometric length (Politis & Romano), wrapping around."""
    idx = np.empty((n_boot, t), dtype=np.int64)
    idx[:, 0] = rng.integers(t, size=n_boot)
    restart = rng.random((n_boot, t)) < 1.0 / mean_block
    jumps = rng.integers(t, size=(n_boot, t))
    for k in range(1, t):
        idx[:, k] = np.where(restart[:, k], jumps[:, k], (idx[:, k - 1] + 1) % t)
    return idx


def _annual_sharpes(samples: np.ndarray) -> np.ndarray:
    sd = samples.std(axis=1, ddof=1)
    return np.where(sd > 0, samples.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0) * np.sqrt(PERIODS)


def stationary_bootstrap_ci(r, n_boot: int = 2000, mean_block: float = 10.0,
                            alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Annualized-Sharpe confidence interval that keeps the serial correlation of daily returns."""
    r = np.asarray(r, dtype=np.float64)
    srs = _annual_sharpes(r[_stationary_indices(len(r), n_boot, mean_block, np.random.default_rng(seed))])
    lo, hi = np.quantile(srs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_sharpe_difference(a, b, n_boot: int = 2000, mean_block: float = 10.0,
                             alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Annualized Sharpe(a) - Sharpe(b), with a bootstrap interval that resamples the same days for both.

    Pairing keeps the correlation between the two strategies, which a separate
    interval for each would ignore. Returns (point estimate, low, high).
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("series must cover the same days")
    idx = _stationary_indices(len(a), n_boot, mean_block, np.random.default_rng(seed))
    diffs = _annual_sharpes(a[idx]) - _annual_sharpes(b[idx])
    point = (daily_sharpe(a) - daily_sharpe(b)) * np.sqrt(PERIODS)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
