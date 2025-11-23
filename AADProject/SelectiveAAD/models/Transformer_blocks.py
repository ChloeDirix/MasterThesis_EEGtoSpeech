import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=2000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]


class AttentionPool(nn.Module):
    """Single-head attention pooling over time."""
    def __init__(self, d_model):
        super().__init__()
        self.att = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, D]
        weights = torch.softmax(self.att(x), dim=1)  # [B, T, 1]
        pooled = (weights * x).sum(dim=1)            # [B, D]
        return pooled
