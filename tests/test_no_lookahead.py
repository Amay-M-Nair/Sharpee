"""Perturbing returns after date t must leave everything decided at or before t unchanged."""

import numpy as np
import pytest

from sharpee.backtest.engine import weights_for
from sharpee.data.preprocess import MarketData
from sharpee.pipeline import panel_from_market
from sharpee.strategies.ou_strategy import ou_positions

from .conftest import SMALL


@pytest.fixture(scope="module")
def pair(market, panel):
    """The fixture panel, and one rebuilt after scrambling every price after a cutoff."""
    cut = panel.dates[len(panel.dates) // 2]
    rng = np.random.default_rng(7)
    close = market.close.copy()
    after = close.index > cut
    close.loc[after] = close.loc[after] * rng.uniform(0.5, 1.5, size=close.loc[after].shape)
    volume = market.volume.copy()
    volume.loc[after] = volume.loc[after] * rng.uniform(0.1, 10, size=volume.loc[after].shape)
    changed = MarketData(close, volume, market.member, market.market)
    return panel, panel_from_market(changed, SMALL), int(np.flatnonzero(panel.dates == cut)[0])


def test_residual_panel_unchanged(pair):
    a, b, t = pair
    idx = np.arange(t + 1)
    assert (a.dates == b.dates).all()
    assert np.array_equal(a.month[idx], b.month[idx])
    months = np.unique(a.month[idx])
    assert np.array_equal(a.universe[months], b.universe[months])
    assert np.array_equal(a.phi[months], b.phi[months])
    assert np.array_equal(a.tradable[idx], b.tradable[idx])
    assert np.array_equal(a.windows(idx, a.max_lookback), b.windows(idx, b.max_lookback))


def test_future_changes_do_reach_the_future(pair):
    """Guards against a vacuous pass: the scramble must change something after the cutoff."""
    a, b, t = pair
    idx = np.arange(t + 1, len(a.dates))
    assert not np.array_equal(a.windows(idx, 20), b.windows(idx, 20))


def test_ou_positions_and_weights_unchanged(pair):
    a, b, t = pair
    idx = np.arange(len(a.dates))
    pa = ou_positions(a, idx, window=40)
    pb = ou_positions(b, idx, window=40)
    assert np.array_equal(pa[:t + 1], pb[:t + 1])
    past = idx[:t + 1]
    assert np.array_equal(weights_for(a, past, pa[:t + 1], cap=0.02),
                          weights_for(b, past, pb[:t + 1], cap=0.02))


def test_realized_return_is_the_only_thing_that_moves(pair):
    """r_next on the cutoff date is R(t+1): realized after the decision, so it may differ."""
    a, b, t = pair
    assert np.array_equal(a.r_next[:t], b.r_next[:t])
    assert not np.array_equal(a.r_next[t], b.r_next[t])
