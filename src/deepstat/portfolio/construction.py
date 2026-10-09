"""Scores -> factor-hedged stock weights. Training and backtesting both call this.

    1. residual space:  center scores over tradable stocks, L1-normalize -> w
    2. stock space:     x = Phi^T w   (holding residual i = stock i minus its factor hedge)
    3. constraints:     zero untradable names, gross exposure 1, single-name cap
"""

import torch


def build_weights(scores: torch.Tensor, phi: torch.Tensor, tradable: torch.Tensor,
                  cap: float | None = 0.02, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        scores:   (T, N) model outputs or rule-based residual positions
        phi:      (T, N, N) or (N, N) residual composition matrix in force each day
        tradable: (T, N) bool

    Returns:
        x: (T, N) stock weights; sum |x| = 1 unless nothing is tradable, |x| <= cap
    """
    m = tradable.to(scores.dtype)
    s = scores * m
    s = s - m * s.sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
    w = s / s.abs().sum(-1, keepdim=True).clamp_min(eps)

    # x_j = sum_i Phi_ij w_i
    if phi.dim() == 2:
        x = w @ phi
    else:
        x = torch.einsum("tij,ti->tj", phi, w)
    x = x * m
    x = x / x.abs().sum(-1, keepdim=True).clamp_min(eps)

    if cap is not None:
        # clip and renormalize until the cap holds; ends on a clip so |x| <= cap exactly
        for _ in range(10):
            x = x.clamp(-cap, cap)
            x = x / x.abs().sum(-1, keepdim=True).clamp_min(eps)
        x = x.clamp(-cap, cap)
    return x


def to_global(x: torch.Tensor, universe: torch.Tensor, n_tickers: int) -> torch.Tensor:
    """Scatter slot weights (T, N) into ticker space (T, G). Slots with id -1 are dropped.

    Turnover needs this: the universe is re-drawn every month, so slot j on
    one day is not necessarily the same stock as slot j the next day.
    """
    ids = torch.where(universe >= 0, universe, torch.full_like(universe, n_tickers))
    out = x.new_zeros(x.shape[0], n_tickers + 1)
    out = out.scatter(1, ids, x)
    return out[:, :n_tickers]
