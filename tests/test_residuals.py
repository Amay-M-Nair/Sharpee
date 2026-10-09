import numpy as np

from deepstat.features.pca_residuals import factor_model
from deepstat.strategies.ou_strategy import ou_fit, ou_positions


def returns_window(seed=0, w=200, n=25, k=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, (w, k))
    beta = rng.normal(1, 0.5, (n, k))
    return f @ beta.T + rng.normal(0, 0.01, (w, n))


def test_phi_matches_definition():
    r = returns_window()
    phi, q, b = factor_model(r, 3)
    f = r @ q
    assert np.allclose(r @ phi.T, r - f @ b.T)


def test_residuals_uncorrelated_with_factors_in_window():
    r = returns_window()
    phi, q, _ = factor_model(r, 3)
    eps, f = r @ phi.T, r @ q
    cov = (eps - eps.mean(0)).T @ (f - f.mean(0)) / len(r)
    assert np.abs(cov).max() < 1e-12


def test_residual_identity():
    """Holding residual weights w is the same P&L as holding stock weights Phi^T w."""
    r = returns_window()
    phi, _, _ = factor_model(r[:-1], 3)
    w = np.random.default_rng(1).normal(size=r.shape[1])
    r_next = r[-1]
    assert np.isclose(w @ (phi @ r_next), (phi.T @ w) @ r_next)


def test_panel_windows_use_the_phi_in_force(market, panel):
    i = len(panel.dates) - 5
    m = panel.month[i]
    ids = panel.universe[m][panel.universe[m] >= 0]
    n = len(ids)
    r = market.returns.reindex(panel.dates).to_numpy()
    date_pos = market.returns.index.get_loc(panel.dates[i])
    raw = np.nan_to_num(market.returns.to_numpy()[date_pos - 19:date_pos + 1][:, ids])
    expected = raw @ panel.phi[m][:n, :n].T.astype(np.float64)
    got = panel.windows(np.array([i]), 20)[0][:n].T
    assert np.allclose(got, expected, atol=1e-6)
    assert r.shape[0] == len(panel.dates)


def test_ou_fit_recovers_mean_reversion():
    rng = np.random.default_rng(0)
    b_true, x = 0.8, np.zeros(2001)
    for t in range(2000):
        x[t + 1] = b_true * x[t] + rng.normal(0, 0.01)
    s, kappa, valid = ou_fit(np.diff(x)[None, :])
    assert valid[0]
    assert abs(np.exp(-kappa[0] / 252) - b_true) < 0.05


def test_ou_score_sign():
    # a path that has just dropped far below its mean should score strongly negative
    w = np.r_[np.random.default_rng(0).normal(0, 0.002, 59), -0.08]
    s, _, valid = ou_fit(w[None, :])
    assert valid[0] and s[0] < -1.25


def test_ou_positions_are_valid(panel):
    idx = np.arange(len(panel.dates))
    pos = ou_positions(panel, idx, window=40)
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}
    assert np.all(pos[~panel.tradable] == 0)
    assert np.abs(pos).sum() > 0
