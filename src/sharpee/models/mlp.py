"""Feed-forward baseline on the same residual path."""

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """(batch, L, 1) -> (batch,) score. Sees the path as a flat vector, no notion of order."""

    def __init__(self, lookback: int = 30, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
