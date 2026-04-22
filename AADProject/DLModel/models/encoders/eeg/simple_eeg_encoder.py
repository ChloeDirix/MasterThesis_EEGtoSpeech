"""
Simple EEG encoder used by Type 3 style models.

This encoder maps EEG windows directly into a stimulus-like feature space.
It is intentionally lightweight and easy to optimize.

Shape convention
----------------
Input:
    [B, T, C_eeg]
Output:
    [B, T, C_out]

where:
- B = batch size
- T = number of time samples in the window
- C_eeg = number of EEG channels
- C_out = output feature dimension (typically the stimulus feature dimension)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleEEGEncoder(nn.Module):
    """
    EEG -> projected feature sequence.

    Two operating modes are supported:

    1) use_conv = True
       A 1D convolution is applied across time, jointly mixing channels and
       local temporal context.

    2) use_conv = False
       A simple linear layer is applied independently at each time step.

    This is the clean, simple encoder family used by Type 3.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        conv_kernel: int = 31,
        dropout: float = 0.0,
        use_conv: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of EEG channels.
        output_dim : int
            Output feature dimension. In Type 3 this is usually aligned with
            the stimulus feature dimension.
        conv_kernel : int, default=31
            Temporal kernel size if `use_conv=True`.
        dropout : float, default=0.0
            Dropout probability applied to the output features.
        use_conv : bool, default=True
            Whether to use a temporal convolution instead of a linear layer.
        """
        super().__init__()
        self.use_conv = bool(use_conv)

        if self.use_conv:
            # Conv1d expects [B, channels, time], so the forward method will
            # transpose the input from [B, T, C] -> [B, C, T].
            self.proj = nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=int(conv_kernel),
                padding=int(conv_kernel) // 2,
                bias=True,
            )
        else:
            # Linear projection applies independently at each time step.
            self.proj = nn.Linear(input_dim, output_dim, bias=True)

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project EEG into the target feature space.

        Parameters
        ----------
        x : torch.Tensor
            EEG tensor of shape [B, T, C_eeg].

        Returns
        -------
        torch.Tensor
            Projected tensor of shape [B, T, C_out].
        """
        if self.use_conv:
            x = x.transpose(1, 2)   # [B, C_eeg, T]
            x = self.proj(x)        # [B, C_out, T]
            x = x.transpose(1, 2)   # [B, T, C_out]
        else:
            x = self.proj(x)        # [B, T, C_out]

        x = self.dropout(x)
        return x
