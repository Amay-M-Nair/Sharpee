import numpy as np
import torch

from sharpee.portfolio.construction import build_weights, to_global
from sharpee.portfolio.risk import exposures


def test_gross_one_and_untradable_zero():
    torch.manual_seed(0)
    scores = torch.randn(4, 10)
    phi = torch.eye(10) - 0.05 * torch.randn(10, 10)
    tradable = torch.ones(4, 10, dtype=torch.bool)
    tradable[:, 3] = False
    x = build_weights(scores, phi, tradable, cap=None)
    assert torch.allclose(x.abs().sum(1), torch.ones(4), atol=1e-6)
    assert torch.all(x[:, 3] == 0)


def test_identity_phi_gives_normalized_scores():
    scores = torch.tensor([[3.0, 1.0, -1.0, 1.0]])
    ones = torch.ones(1, 4, dtype=torch.bool)
    centered = scores - scores.mean()
    assert torch.allclose(build_weights(scores, torch.eye(4), ones, cap=None), centered / centered.abs().sum())
    assert torch.allclose(build_weights(scores, torch.eye(4), ones, cap=None, center=False),
                          scores / scores.abs().sum())


def test_constant_model_output_holds_nothing():
    """Without centering, a constant score is an equal-weight basket of every residual; a model
    collapsed onto that bet during training. Centered, a constant score must mean no position."""
    x = build_weights(torch.full((2, 10), 0.7), torch.eye(10), torch.ones(2, 10, dtype=torch.bool))
    assert torch.all(x == 0)


def test_x_is_phi_transpose_w():
    torch.manual_seed(1)
    scores, phi = torch.randn(1, 6), torch.randn(6, 6)
    x = build_weights(scores, phi, torch.ones(1, 6, dtype=torch.bool), cap=None)
    w = scores - scores.mean()
    w = w / w.abs().sum()
    expected = phi.T @ w[0]
    assert torch.allclose(x[0], expected / expected.abs().sum(), atol=1e-6)


def test_cap_limits_each_bet_without_reinflating_the_rest():
    scores = torch.tensor([[10.0, 1.0, -1.0, 0.5] + [0.0] * 96])
    x = build_weights(scores, torch.eye(100), torch.ones(1, 100, dtype=torch.bool), cap=0.02, center=False)
    w = scores / scores.abs().sum()
    assert torch.allclose(x, w.clamp(-0.02, 0.02))  # nothing is pushed into the zero-score names
    assert x.abs().sum() < 1.0


def test_cap_keeps_the_factor_hedge():
    """Regression: clipping stock weights after Phi^T w left the hedge legs in place and the
    book long the market. Capping residual positions first must keep it factor-neutral."""
    from sharpee.features.pca_residuals import factor_model

    rng = np.random.default_rng(0)
    f = rng.normal(0.0005, 0.01, (252, 3))
    r = f @ rng.normal(1.0, 0.4, (60, 3)).T + rng.normal(0, 0.01, (252, 60))
    phi, q, _ = factor_model(r, 3)
    scores = -np.abs(rng.standard_t(1.5, size=60))  # lopsided, like the MLP's
    x = build_weights(torch.from_numpy(scores[None]).float(), torch.from_numpy(phi).float(),
                      torch.ones(1, 60, dtype=torch.bool), cap=0.02)[0].double().numpy()

    def max_factor_corr(weights):
        port, factors = r @ weights, r @ q
        return max(abs(np.corrcoef(port, factors[:, k])[0, 1]) for k in range(3))

    assert max_factor_corr(x) < 1e-3

    # the old order (hedge, then clip stock weights) fails the same check
    old = phi.T @ (scores / np.abs(scores).sum())
    for _ in range(10):
        old = np.clip(old / np.abs(old).sum(), -0.02, 0.02)
    assert max_factor_corr(old) > 0.1


def test_batched_phi_matches_shared_phi():
    torch.manual_seed(3)
    scores, phi = torch.randn(5, 8), torch.randn(8, 8)
    tradable = torch.ones(5, 8, dtype=torch.bool)
    a = build_weights(scores, phi, tradable, cap=None)
    b = build_weights(scores, phi.expand(5, 8, 8), tradable, cap=None)
    assert torch.allclose(a, b, atol=1e-6)


def test_to_global_drops_empty_slots():
    x = torch.tensor([[0.5, -0.3, 0.2]])
    uni = torch.tensor([[4, 1, -1]])
    g = to_global(x, uni, 6)
    assert g.shape == (1, 6)
    assert g[0, 4] == 0.5 and g[0, 1] == -0.3 and g.abs().sum() == 0.8


def test_exposures():
    e = exposures(np.array([[0.3, -0.5, 0.2]]))
    assert np.isclose(e["net_exposure"][0], 0.0) and np.isclose(e["gross_exposure"][0], 1.0)
    assert e["n_positions"][0] == 3 and np.isclose(e["max_abs_weight"][0], 0.5)
