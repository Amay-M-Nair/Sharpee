"""Differentiable portfolio objective."""

import torch


def sharpe_loss(net: torch.Tensor, net_exposure: torch.Tensor | None = None,
                lam: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    """-mean(R_net) / (std(R_net) + eps) + lam * mean_t((sum_i x(i,t))^2).

    Per-period Sharpe (not annualized; that is only a constant factor).
    Costs are already inside R_net, so there is no separate turnover term.
    """
    loss = -net.mean() / (net.std() + eps)
    if lam:
        loss = loss + lam * (net_exposure ** 2).mean()
    return loss
