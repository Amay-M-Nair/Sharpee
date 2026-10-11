import numpy as np
import pytest

from sharpee.backtest.engine import weights_for
from sharpee.training.tabular import fit, make_xy, payoffs, predict_scores, zscore_rows


def test_payoff_is_what_the_hedged_bet_earns(panel):
    """Booked P&L of x = Phi^T w (through the real backtester path) equals w . payoff, scaled the same way."""
    idx = np.arange(100, 160)                                        # spans a monthly refit
    w = np.random.default_rng(0).normal(size=(len(idx), panel.n_slots)).astype(np.float32)
    w[~panel.tradable[idx]] = 0
    x = weights_for(panel, idx, w, cap=None, center=False)
    booked = (x * panel.r_next[idx]).sum(1)

    w_norm = w / np.abs(w).sum(1, keepdims=True)
    gross_x = np.array([np.abs(w_norm[j] @ panel.phi[panel.month[i]]).sum() for j, i in enumerate(idx)])
    expected = (w_norm * payoffs(panel, idx)).sum(1) / gross_x       # weights_for rescales x to gross 1
    assert np.allclose(booked, expected, atol=1e-6)


def test_zscore_rows():
    a = np.array([[1.0, 2.0, 3.0, 100.0]])
    z = zscore_rows(a, np.array([[True, True, True, False]]))
    assert np.isnan(z[0, 3]) and np.isclose(np.nanmean(z), 0) and np.isclose(np.nanstd(z), 1)


def test_features_never_see_the_future(panel):
    """X on day t uses returns up to t only; y on day t is R(t+1), the realized payoff."""
    idx = np.arange(100, 160)
    x_full, _ = make_xy(panel, idx, 30)
    x_part, _ = make_xy(panel, idx[:30], 30)
    assert np.array_equal(x_full[:len(x_part)], x_part)


@pytest.mark.parametrize("name,params", [("ridge", {"alpha": 10.0}),
                                         ("gbm", {"max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 20})])
def test_fit_and_score(panel, name, params):
    x, y = make_xy(panel, np.arange(60, 200), 30)
    model = fit(name, params, x, y)
    s = predict_scores(model, panel, np.arange(200, 220), 30)
    assert s.shape == (20, panel.n_slots) and np.all(np.isfinite(s))
    assert np.all(s[~panel.tradable[200:220]] == 0)
