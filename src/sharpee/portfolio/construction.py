"""Scores -> factor-hedged stock weights. Training and backtesting both call this.

    1. residual positions:  s demeaned across tradable stocks (center=True),
                            w = s / sum|s|, each clipped to |w_i| <= cap
    2. stock weights:       x = Phi^T w, rescaled so that sum|x| = sum|w|

Each w_i is a hedged bet (long stock i, short its factor exposure), so the
book is a sum of hedged bets. The position limit is applied to w, before the
mapping: clipping x afterwards would cut a stock leg but keep its hedge legs,
leaving the portfolio exposed to the market (a trained model found and used
exactly that gap). Clipped weight is not redistributed, so a concentrated
signal gives a smaller book rather than inflating every other position.

Model scores are centered, so a constant output means no position. Without it
a network can collapse to an equal-weight basket of all residuals, a bet on
the average residual rather than on relative value. Rule-based positions
(OU's sparse +1/-1) use center=False and are taken as they are, as in
Avellaneda-Lee; centering them would spread every position's sign onto all
the flat stocks.
"""

import torch


def build_weights(scores: torch.Tensor, phi: torch.Tensor, tradable: torch.Tensor,
                  cap: float | None = 0.02, center: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        scores:   (T, N) model outputs or rule-based residual positions
        phi:      (T, N, N) or (N, N) residual composition matrix in force each day
        tradable: (T, N) bool
        center:   demean scores across tradable stocks (models: True, OU: False)

    Returns:
        x: (T, N) stock weights; gross sum|x| <= 1 (exactly 1 when no position hits the cap)
    """
    m = tradable.to(scores.dtype)
    raw = scores * m
    s = raw
    if center:
        s = s - m * s.sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
    size = s.abs().sum(-1, keepdim=True)
    # no cross-sectional variation (up to rounding) means no position, whatever the score units
    flat = size <= 1e-6 * raw.abs().sum(-1, keepdim=True) + eps
    w = torch.where(flat, torch.zeros_like(s), s / size.clamp_min(eps))
    if cap is not None:
        w = w.clamp(-cap, cap)

    # x_j = sum_i Phi_ij w_i
    if phi.dim() == 2:
        x = w @ phi
    else:
        x = torch.einsum("tij,ti->tj", phi, w)
    x = x * m
    # rescaling keeps the hedge; the stock book is as large as the residual book
    return x * (w.abs().sum(-1, keepdim=True) / x.abs().sum(-1, keepdim=True).clamp_min(eps))


def to_global(x: torch.Tensor, universe: torch.Tensor, n_tickers: int) -> torch.Tensor:
    """Scatter slot weights (T, N) into ticker space (T, G). Slots with id -1 are dropped.

    Turnover needs this: the universe is re-drawn every month, so slot j on
    one day is not necessarily the same stock as slot j the next day.
    """
    ids = torch.where(universe >= 0, universe, torch.full_like(universe, n_tickers))
    out = x.new_zeros(x.shape[0], n_tickers + 1)
    out = out.scatter(1, ids, x)
    return out[:, :n_tickers]
