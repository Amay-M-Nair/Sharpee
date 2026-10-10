import numpy as np
from scipy.stats import norm

from sharpee.evaluation.significance import (dsr, expected_max_sharpe, paired_sharpe_difference, psr,
                                             stationary_bootstrap_ci)


def test_psr_hand_value():
    # mean 0.01, sd 0.0158114 -> daily SR 0.632456; deviations are symmetric so skew 0;
    # biased m2 = 2e-4, m4 = 6.8e-8 -> kurtosis 1.7
    r = [0.01, 0.02, -0.01, 0.03, 0.00]
    z = 0.6324555 * np.sqrt(4) / np.sqrt(1 + (1.7 - 1) / 4 * 0.6324555 ** 2)
    assert np.isclose(psr(r), norm.cdf(z), atol=1e-6)


def test_psr_is_one_half_at_its_own_sharpe():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    sr = r.mean() / r.std(ddof=1)
    assert np.isclose(psr(r, sr), 0.5)


def test_expected_max_sharpe_hand_value():
    # (1 - 0.5772) * Phi^-1(0.9) + 0.5772 * Phi^-1(1 - 1/(10e)) = 0.4228 * 1.2816 + 0.5772 * 1.7896
    assert abs(expected_max_sharpe(10, 1.0) - 1.5748) < 1e-3
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_more_trials_raise_the_bar():
    r = np.random.default_rng(1).normal(0.001, 0.01, 1000)
    assert dsr(r, 1, 1e-3) == psr(r)
    assert dsr(r, 100, 1e-3) < dsr(r, 10, 1e-3) < psr(r)


def test_bootstrap_interval_contains_the_estimate_and_shrinks_with_data():
    rng = np.random.default_rng(2)
    short, long = rng.normal(0.0005, 0.01, 250), rng.normal(0.0005, 0.01, 2500)
    lo_s, hi_s = stationary_bootstrap_ci(short, n_boot=500)
    lo_l, hi_l = stationary_bootstrap_ci(long, n_boot=500)
    point = long.mean() / long.std(ddof=1) * np.sqrt(252)
    assert lo_l < point < hi_l
    assert hi_l - lo_l < hi_s - lo_s


def test_paired_difference():
    rng = np.random.default_rng(3)
    b = rng.normal(0.0, 0.01, 1000)
    same = paired_sharpe_difference(b, b, n_boot=300)
    assert same == (0.0, 0.0, 0.0)
    better = b + 0.002  # same noise, clearly higher mean
    point, lo, hi = paired_sharpe_difference(better, b, n_boot=300)
    assert lo > 0 and lo < point < hi
