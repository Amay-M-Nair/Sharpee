"""1D-convolution baseline: local patterns (reversals, persistence, volatility bursts)."""

import torch
import torch.nn as nn


class ResidualCNN(nn.Module):
    """(batch, L, 1) -> (batch,) score. Dilated convolutions, then average over time."""

    def __init__(self, lookback: int = 30, channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))     # (batch, channels, L)
        return self.head(h.mean(-1)).squeeze(-1)
