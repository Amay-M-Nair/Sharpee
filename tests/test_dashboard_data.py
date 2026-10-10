"""The dashboard recomputes results from small exported files; they must match the Phase 3 outputs."""

import numpy as np
import pandas as pd
import pytest

import dash_data as dd


@pytest.mark.parametrize("period", ["test", "holdout"])
def test_dashboard_sharpes_match_phase3_results(period):
    returns = dd.load_returns(period)
    official = pd.read_csv(dd.PHASE3 / f"{period}_summary.csv")
    for row in official.itertuples():
        mine = dd.sharpe(dd.net_returns(returns, row.strategy, row.cost_bps, row.delay))
        assert abs(mine - row.sharpe) < 1e-3, (row.strategy, row.cost_bps, row.delay)


def test_net_returns_are_linear_in_cost():
    returns = dd.load_returns("test")
    r0, r5, r10 = (dd.net_returns(returns, "transformer", c) for c in (0, 5, 10))
    assert np.allclose(r5 - r0, (r10 - r0) / 2)


def test_signals_cover_the_universe():
    signals = dd.load_signals("test")
    per_day = signals.groupby("date").size()
    assert per_day.max() <= 150 and per_day.median() >= 140
    assert set(signals["ou"].dropna().unique()) <= {-1.0, 0.0, 1.0}


def test_verdict_line():
    assert dd.verdict_line("test") == "no reliable edge after costs"


def test_learned_rule_reverts_moderate_moves():
    rule = dd.learned_rule(dd.load_signals("test"))
    assert list(rule.columns) == ["mlp", "temporal_cnn", "transformer"]
    # long when the residual fell (s < 0), short when it rose: a downward slope across the middle
    assert (rule.loc[-1.25] > 0).all() and (rule.loc[1.25] < 0).all()


def test_attention_is_a_distribution_over_the_window():
    att = dd.load_attention()
    assert len(att) == 30 and att.index.max() == 0
    assert abs(att["moderate"].sum() - 1) < 1e-3 and abs(att["extreme"].sum() - 1) < 1e-3
