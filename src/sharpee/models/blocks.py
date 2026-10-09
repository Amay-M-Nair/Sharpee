"""Transformer building blocks, vendored from ../2.transformer-from-scratch/src
(attention.py, layers.py, positional.py) so this repo runs on its own.
Only the encoder side is needed here."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention across several parallel heads."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)"""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)"""
        batch, _, seq_len, _ = x.shape
        # contiguous() because transpose only changes indexing, and view needs
        # a contiguous buffer
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, q_len, d_model)
            key:   (batch, kv_len, d_model)
            value: (batch, kv_len, d_model)
            mask:  bool, broadcastable to (batch, heads, q_len, kv_len)

        Returns:
            output:       (batch, q_len, d_model)
            attn_weights: (batch, heads, q_len, kv_len)
        """
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        # head_dim, not d_model: after the split that is the d_k
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # finfo.min not -inf: a fully masked row would otherwise give NaN
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = self.dropout(attn_weights) @ V

        # weights returned undropped - they are for inspection
        return self.W_o(self._merge_heads(attn_out)), attn_weights


class FeedForward(nn.Module):
    """Position-wise MLP: d_model -> d_ff -> d_model."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Self-attention, then feed-forward. Two sublayers. Pre-norm: x = x + sublayer(norm(x))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Args:
            x:        (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len)

        Returns:
            x:            unchanged shape
            attn_weights: (batch, heads, src_len, src_len)
        """
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed, normed, normed, src_mask)
        x = x + self.dropout(attn_out)

        x = x + self.dropout(self.feed_forward(self.norm2(x)))

        return x, attn_weights


class PositionalEncoding(nn.Module):
    """Adds fixed position signals to token embeddings. Not learned."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(-2i/d_model) computed in log space to stay well-conditioned
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        # slice div_term in case d_model is odd
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        # buffer, not parameter: moves with .to(device), never optimised
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> same shape."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
