"""Linear control: one weight per day of the path, no hidden layer.

Trained with the same Sharpe-after-costs loss as the networks, so comparing it
with them isolates what depth and nonlinearity add.
"""

import torch
import torch.nn as nn


class ResidualLinear(nn.Module):
    """(batch, L, 1) -> (batch,) score, a weighted sum of the path."""

    def __init__(self, lookback: int = 30):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(lookback, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
