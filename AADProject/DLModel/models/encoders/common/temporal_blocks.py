"""
Reusable temporal building blocks for EEG / stimulus encoders.

Why this file exists
--------------------
Several models in the project use the same structural ideas:
- 1D temporal convolutions
- residual connections
- normalization + activation + dropout

Instead of defining these blocks inside one specific model file, we place
them in `encoders/common` so multiple encoder families can reuse them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTemporalConvBlock(nn.Module):
    """
    Generic residual temporal convolution block.

    This block assumes an input shaped as [B, T, D]:
    - B = batch size
    - T = time dimension
    - D = feature dimension / channels

    Internally, PyTorch Conv1d expects [B, D, T], so we transpose before
    and after the convolution.

    Block structure
    ---------------
    input
      -> Conv1d over time
      -> LayerNorm
      -> GELU
      -> Dropout
      -> Residual add with original input

    Input
    -----
    x : torch.Tensor
        Shape [B, T, D]

    Output
    ------
    torch.Tensor
        Same shape [B, T, D]
    """

    def __init__(self, d_model: int, conv_kernel: int = 31, dropout: float = 0.0) -> None:
        """
        Parameters
        ----------
        d_model : int
            Number of channels / features D.
        conv_kernel : int, default=31
            Temporal kernel size of the 1D convolution.
        dropout : float, default=0.0
            Dropout probability after activation.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=int(conv_kernel),
            padding=int(conv_kernel) // 2,
            bias=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual temporal convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, T, D].

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape [B, T, D].
        """
        residual = x

        # Conv1d operates on [B, channels, time]
        y = x.transpose(1, 2)   # [B, D, T]
        y = self.conv(y)        # [B, D, T]
        y = y.transpose(1, 2)   # [B, T, D]

        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)

        # Residual connection: keep the original information path open.
        return residual + y
