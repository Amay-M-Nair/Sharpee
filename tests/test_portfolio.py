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


def test_identity_phi_gives_centered_scores():
    scores = torch.tensor([[3.0, 1.0, -1.0, 1.0]])
    x = build_weights(scores, torch.eye(4), torch.ones(1, 4, dtype=torch.bool), cap=None)
    centered = scores - scores.mean()
    assert torch.allclose(x, centered / centered.abs().sum())


def test_x_is_phi_transpose_w():
    torch.manual_seed(1)
    scores, phi = torch.randn(1, 6), torch.randn(6, 6)
    x = build_weights(scores, phi, torch.ones(1, 6, dtype=torch.bool), cap=None)
    w = scores - scores.mean()
    w = w / w.abs().sum()
    expected = (phi.T @ w[0])
    assert torch.allclose(x[0], expected / expected.abs().sum(), atol=1e-6)


def test_cap_holds():
    torch.manual_seed(2)
    scores = torch.randn(3, 100) ** 3   # heavy-tailed, so the cap binds
    x = build_weights(scores, torch.eye(100), torch.ones(3, 100, dtype=torch.bool), cap=0.02)
    assert x.abs().max() <= 0.02 + 1e-7
    assert torch.all(x.abs().sum(1) > 0.99)


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
