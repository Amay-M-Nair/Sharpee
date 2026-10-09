"""Portfolio accounting shared by the training loss and the backtester.

Timing: weights x(t) are set at the close of day t and earn R(t+1).

    R_gross(t+1) = sum_i x(i,t) R(i,t+1)
    TO(t)        = sum_i |x(i,t) - x(i,t-1)|      total traded notional, both sides
    R_net(t+1)   = R_gross(t+1) - c TO(t)          the trade at t is charged to the
                                                   return of the position it creates

Weight drift between daily rebalances is ignored. R_net of this self-financing
long/short book is already an excess return; no risk-free rate is subtracted.
"""

import torch


def turnover(x: torch.Tensor, x_prev: torch.Tensor | None = None) -> torch.Tensor:
    """x: (T, G) -> (T,). The first day trades from x_prev (default: flat)."""
    if x_prev is None:
        x_prev = torch.zeros_like(x[0])
    prev = torch.cat([x_prev.unsqueeze(0), x[:-1]], dim=0)
    return (x - prev).abs().sum(-1)


def portfolio_returns(x: torch.Tensor, r_next: torch.Tensor, cost: float,
                      x_prev: torch.Tensor | None = None):
    """
    Args:
        x:      (T, G) weights held from the close of t
        r_next: (T, G) R(t+1)
        cost:   proportional cost per unit of traded notional (5 bps = 0.0005)

    Returns:
        gross, traded, net: each (T,), row t realized over t -> t+1
    """
    gross = (x * r_next).sum(-1)
    traded = turnover(x, x_prev)
    return gross, traded, gross - cost * traded
