"""
Shared positional encoding utilities for transformer-style models.

Why this file exists
--------------------
Previously, the positional encoding lived in a broad utility module
(`Transformer_blocks.py`). For a cleaner structure, we move the shared
sequence-encoding utilities into the `encoders/common` package so that
all encoder-related building blocks are grouped together.

All tensors in this project follow the convention:
    B = batch size
    T = number of time steps / samples in the window
    D = feature dimension / embedding dimension

Therefore, most sequence tensors are shaped:
    [B, T, D]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence models.

    Purpose
    -------
    A transformer does not inherently know the order of time samples.
    Positional encoding injects time-position information into the input
    embeddings so that token 0 and token 100 are distinguishable.

    Input
    -----
    x : torch.Tensor
        Shape [B, T, D], where:
        - B = batch size
        - T = number of time samples
        - D = embedding dimension

    Output
    ------
    torch.Tensor
        Same shape as input: [B, T, D]

    Notes
    -----
    - This uses the fixed sinusoidal encoding introduced in the original
      Transformer paper.
    - The encoding is registered as a buffer, which means:
        * it moves with the model to CPU/GPU,
        * it is saved in checkpoints,
        * but it is not trainable.
    """

    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        """
        Parameters
        ----------
        d_model : int
            Embedding dimension D.
        max_len : int, default=2000
            Maximum supported sequence length.
        """
        super().__init__()

        # Create a matrix of shape [T, D] that will hold one positional vector
        # per time step from 0 to max_len - 1.
        pe = torch.zeros(max_len, d_model)

        # Column vector with positions [0, 1, 2, ..., max_len-1]^T
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Frequency scaling term for alternating sine/cosine dimensions.
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # Even dimensions use sine; odd dimensions use cosine.
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        # Add a batch dimension so it can broadcast against [B, T, D].
        # Final stored shape: [1, T, D]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, T, D].

        Returns
        -------
        torch.Tensor
            Positional-encoded tensor with shape [B, T, D].
        """
        T = x.size(1)
        return x + self.pe[:, :T]
