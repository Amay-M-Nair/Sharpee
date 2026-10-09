"""Compact Transformer encoder that scores one residual path at a time."""

import torch
import torch.nn as nn

from .blocks import EncoderLayer, PositionalEncoding


class ResidualTransformer(nn.Module):
    """(batch, L, 1) cumulative-residual path -> (batch,) score.

    Each day is a token; the encoder sees the whole window (no causal mask is
    needed because the window already ends at the decision date).
    """

    def __init__(self, lookback: int = 30, d_model: int = 64, num_heads: int = 4,
                 num_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, max_len=lookback, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h = self.pos(self.embed(x))
        attn = []
        for layer in self.layers:
            h, a = layer(h)
            attn.append(a)
        score = self.head(self.norm(h).mean(1)).squeeze(-1)
        return (score, attn) if return_attn else score
