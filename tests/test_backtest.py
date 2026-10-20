import numpy as np
import pytest
import torch

from sharpee.backtest.costs import portfolio_returns, turnover
from sharpee.backtest.engine import run_backtest, weights_for
from sharpee.backtest.realistic import holding_costs, spread_from_rank
from sharpee.evaluation.walk_forward import make_fold, make_folds, split


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


def test_spread_tiers_hand_values():
    rank = np.array([[1, 100, 101, 250, 301, np.nan]])
    cost = spread_from_rank(rank, [(100, 3.0), (200, 5.0), (300, 7.0)], 10.0)
    assert np.allclose(cost, np.array([[3, 3, 5, 7, 10, 10]]) * 1e-4)


def test_holding_costs_hand_example():
    """Spread is charged per stock on |dx|; borrow on short notional, hedge legs included."""
    x = np.array([[0.5, -0.3, -0.2], [0.2, -0.3, 0.1]])
    spread = np.array([[0.0003, 0.0010, 0.0005]] * 2)
    spread_cost, borrow_cost = holding_costs(x, spread, borrow_rate=0.0252)
    # day 0 trades from flat: .5*3 + .3*10 + .2*5 = 5.5 bps; day 1: .3*3 + 0 + .3*5 = 2.4 bps
    assert np.allclose(spread_cost, [5.5e-4, 2.4e-4])
    # 2.52% / 252 = 1 bp a day per unit short: shorts 0.5 then 0.3
    assert np.allclose(borrow_cost, [0.5e-4, 0.3e-4])


def test_flat_spread_matches_flat_cost(panel):
    idx = np.arange(100, 160)
    scores = np.random.default_rng(0).normal(size=(len(idx), panel.n_slots)).astype(np.float32)
    flat = run_backtest(panel, idx, scores, cost_bps=5)
    spread = np.full((len(panel.dates), len(panel.tickers)), 5e-4, dtype=np.float32)
    tiered = run_backtest(panel, idx, scores, spread=spread, borrow_rate=0.01)
    assert np.allclose(flat["spread_cost"], tiered["spread_cost"], atol=1e-7)
    assert np.allclose(tiered["borrow_cost"], 0.01 / 252 * -tiered["short_exposure"], atol=1e-9)
    assert np.allclose(tiered["net"], tiered["gross"] - tiered["spread_cost"] - tiered["borrow_cost"])
