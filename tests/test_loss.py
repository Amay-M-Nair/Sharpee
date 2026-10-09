import numpy as np
import torch

from sharpee.models import build_model
from sharpee.training.dataset import DayBlocks, path_features
from sharpee.training.loss import sharpe_loss
from sharpee.training.train import portfolio_forward
from sharpee.training.tuning import expand


def test_loss_is_negative_sharpe():
    r = torch.tensor([0.01, -0.005, 0.02, 0.003])
    assert torch.isclose(sharpe_loss(r), -r.mean() / r.std())


def test_exposure_penalty():
    r = torch.tensor([0.01, -0.005, 0.02])
    exposure = torch.tensor([0.1, -0.2, 0.0])
    expected = -r.mean() / r.std() + 2.0 * (exposure ** 2).mean()
    assert torch.isclose(sharpe_loss(r, exposure, lam=2.0), expected)


def test_gradients_reach_the_model_through_weights_and_costs(panel):
    """The loss is computed after Phi^T w, the cap and trading costs; training needs gradients through all of it."""
    torch.manual_seed(0)
    data = DayBlocks(panel, np.arange(100, 160), lookback=30)
    model = build_model("mlp", lookback=30)
    net, x_glob = portfolio_forward(model, data, slice(0, 40), cap=0.02, cost=0.0005)
    sharpe_loss(net[1:], x_glob[1:].sum(-1)).backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_costs_lower_net_returns(panel):
    torch.manual_seed(0)
    data = DayBlocks(panel, np.arange(100, 160), lookback=30)
    model = build_model("mlp", lookback=30).eval()
    with torch.no_grad():
        free, _ = portfolio_forward(model, data, slice(0, 40), cap=0.02, cost=0.0)
        costly, _ = portfolio_forward(model, data, slice(0, 40), cap=0.02, cost=0.001)
    assert torch.all(costly <= free + 1e-9) and torch.any(costly < free)


def test_path_features_are_scale_free():
    w = np.random.default_rng(0).normal(0, 0.01, (3, 5, 30)).astype(np.float32)
    assert np.allclose(path_features(w), path_features(4 * w), atol=1e-5)
    assert np.all(path_features(np.zeros((2, 30), dtype=np.float32)) == 0)


def test_day_blocks_cover_every_date_once_with_no_tiny_blocks(panel):
    """A 1-day block makes the Sharpe loss NaN and silently poisons the weights."""
    data = DayBlocks(panel, np.arange(50, 250), lookback=30)
    rng = np.random.default_rng(3)
    for _ in range(50):
        slices = data.slices(64, rng)
        covered = sorted(i for s in slices for i in range(s.start, s.stop))
        assert covered == list(range(200))
        assert min(s.stop - s.start for s in slices) >= 32


def test_grid_expansion():
    cfg = {"model": "mlp", "params": {"hidden": 8}, "train": {"lr": 0.1},
           "tune": {"params.hidden": [8, 16], "train.lr": [0.1, 0.01, 0.001]}}
    grid = expand(cfg)
    assert len(grid) == 6 and all("tune" not in g for g in grid)
    assert {(g["params"]["hidden"], g["train"]["lr"]) for g in grid} == {
        (h, lr) for h in (8, 16) for lr in (0.1, 0.01, 0.001)}
    assert expand({"model": "mlp", "params": {}, "train": {}}) == [{"model": "mlp", "params": {}, "train": {}}]
