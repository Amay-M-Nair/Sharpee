import numpy as np
import pandas as pd

from sharpee.evaluation.metrics import max_drawdown, sharpe, summarize


def test_sharpe_hand_value():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    mean, sd = 0.005, np.sqrt(((0.005**2) + (0.015**2) + (0.015**2) + (0.005**2)) / 3)
    assert np.isclose(sharpe(r), mean / sd * np.sqrt(252))


def test_sharpe_of_constant_series_is_zero():
    assert sharpe(np.zeros(10)) == 0.0


def test_sharpe_of_broken_series_is_nan():
    assert np.isnan(sharpe([0.01, np.nan, 0.02]))


def test_max_drawdown_hand_value():
    # equity 1.1, 0.55, 0.66: worst fall is from 1.1 to 0.55
    assert np.isclose(max_drawdown([0.1, -0.5, 0.2]), -0.5)
    assert max_drawdown([0.01, 0.02]) == 0.0


def test_summarize_beta_and_costs():
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, 500)
    gross = 0.5 * mkt + rng.normal(0.001, 0.005, 500)
    frame = pd.DataFrame({"gross": gross, "net": gross - 0.0002, "market": mkt,
                          "turnover": 0.4, "net_exposure": 0.0})
    s = summarize(frame)
    assert abs(s["beta"] - 0.5) < 0.1
    assert np.isclose(s["cost_drag"], 0.0002 * 252)
    assert s["sharpe"] < s["gross_sharpe"]
