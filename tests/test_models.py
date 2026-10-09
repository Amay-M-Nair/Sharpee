import pytest
import torch

from sharpee.models import MODELS, build_model


@pytest.mark.parametrize("name", sorted(MODELS))
def test_one_score_per_path(name):
    model = build_model(name, lookback=30).eval()
    out = model(torch.randn(7, 30, 1))
    assert out.shape == (7,)
    assert torch.isfinite(out).all()


def test_transformer_returns_attention():
    model = build_model("transformer", lookback=30, num_layers=2, num_heads=4).eval()
    score, attn = model(torch.randn(5, 30, 1), return_attn=True)
    assert score.shape == (5,)
    assert len(attn) == 2 and attn[0].shape == (5, 4, 30, 30)
    assert torch.allclose(attn[0].sum(-1), torch.ones(5, 4, 30), atol=1e-5)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_paths_are_scored_independently(name):
    """A stock's score must not depend on which other stocks share the batch."""
    torch.manual_seed(0)
    model = build_model(name, lookback=30).eval()
    x = torch.randn(6, 30, 1)
    with torch.no_grad():
        assert torch.allclose(model(x)[:2], model(x[:2]), atol=1e-5)
