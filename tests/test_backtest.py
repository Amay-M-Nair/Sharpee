import numpy as np
import pytest
import torch

from deepstat.backtest.costs import portfolio_returns, turnover
from deepstat.backtest.engine import run_backtest, weights_for
from deepstat.evaluation.walk_forward import make_fold, make_folds, split


def test_hand_example_timing_and_costs():
    """Weights set at t earn R(t+1); the trade at t is charged in that same row."""
    x = torch.tensor([[0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
    r_next = torch.tensor([[0.02, 0.01], [0.00, 0.04], [0.03, -0.01]])
    gross, traded, net = portfolio_returns(x, r_next, cost=0.001)
    assert torch.allclose(gross, torch.tensor([0.005, -0.02, -0.02]))
    assert torch.allclose(traded, torch.tensor([1.0, 0.0, 2.0]))
    assert torch.allclose(net, torch.tensor([0.004, -0.02, -0.022]))


def test_turnover_from_previous_position():
    x = torch.tensor([[0.5, -0.5]])
    assert turnover(x, x_prev=torch.tensor([0.5, -0.5])).item() == 0.0


def test_delay_shifts_positions(panel):
    idx = np.arange(100, 160)
    scores = np.random.default_rng(0).normal(size=(len(idx), panel.n_slots)).astype(np.float32)
    now = run_backtest(panel, idx, scores, cost_bps=0, delay=0)
    late = run_backtest(panel, idx, scores, cost_bps=0, delay=1)
    x = weights_for(panel, idx, scores, cap=0.02)
    # with delay 1, row t holds x(t-1) and earns R(t+1)
    expected = (x[:-1] * panel.r_next[idx[1:]]).sum(1)
    assert np.allclose(late["gross"].to_numpy()[1:], expected, atol=1e-6)
    assert late["gross"].iloc[0] == 0.0
    assert not np.allclose(now["gross"], late["gross"])


def test_backtest_drops_last_unrealized_day(panel):
    idx = np.arange(len(panel.dates) - 10, len(panel.dates))
    frame = run_backtest(panel, idx, np.ones((10, panel.n_slots), dtype=np.float32))
    assert len(frame) == 9


def test_backtest_rejects_gaps(panel):
    with pytest.raises(ValueError):
        run_backtest(panel, np.array([10, 11, 13]), np.zeros((3, panel.n_slots), dtype=np.float32))


def test_folds_are_ordered_with_embargo(panel):
    fold = make_folds([2020])[0]
    assert fold.train == ("2010-01-01", "2017-12-31") and fold.val == ("2018-01-01", "2019-12-31")
    holdout = make_fold(2024, 2025)
    assert holdout.train[1] == "2021-12-31" and holdout.test == ("2024-01-01", "2025-12-31")

    d = panel.dates
    small = make_fold(d[-60].year, d[-60].year, train_start=d[0].year, val_years=0)
    small.val = (str(d[200].date()), str(d[300].date()))
    small.train = (str(d[0].date()), str(d[150].date()))
    small.test = (str(d[350].date()), str(d[-1].date()))
    tr, va, te = split(panel, small, embargo=30)
    assert tr[-1] == 150 - 30 and va[-1] == 300 - 30 and te[0] == 350
