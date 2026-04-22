"""
Simple audio / stimulus encoder used by Type 3 style models.

This module optionally transforms the candidate stimulus features before
they are compared to the EEG-derived representation.

Shape convention
----------------
Input:
    [B, T, C_in]
Output:
    [B, T, C_out]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleAudioEncoder(nn.Module):
    """
    Lightweight stimulus encoder.

    Supported modes
    ---------------
    - "identity": leave the features unchanged
    - "linear":   linear projection at each time step
    - "conv":     temporal convolution over the stimulus sequence
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        mode: str = "identity",
        conv_kernel: int = 31,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of input stimulus features.
        output_dim : int | None, default=None
            Output feature dimension. If None, the input dimension is reused.
        dropout : float, default=0.0
            Dropout probability applied after projection.
        mode : str, default="identity"
            One of {"identity", "linear", "conv"}.
        conv_kernel : int, default=31
            Kernel size used when mode="conv".
        """
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.mode = str(mode).lower()
        self.output_dim = int(output_dim)

        if self.mode == "identity":
            if output_dim != input_dim:
                raise ValueError(
                    "SimpleAudioEncoder mode='identity' requires output_dim == input_dim "
                    f"(got input_dim={input_dim}, output_dim={output_dim})"
                )
            self.proj = nn.Identity()

        elif self.mode == "linear":
            self.proj = nn.Linear(input_dim, output_dim, bias=True)

        elif self.mode == "conv":
            self.proj = nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=int(conv_kernel),
                padding=int(conv_kernel) // 2,
                bias=True,
            )

        else:
            raise ValueError(f"Unknown SimpleAudioEncoder mode: {mode}")

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the stimulus sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input stimulus tensor of shape [B, T, C_in].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, T, C_out].
        """
        if self.mode == "conv":
            x = x.transpose(1, 2)   # [B, C_in, T]
            x = self.proj(x)        # [B, C_out, T]
            x = x.transpose(1, 2)   # [B, T, C_out]
        else:
            x = self.proj(x)        # [B, T, C_out]

        x = self.dropout(x)
        return x
