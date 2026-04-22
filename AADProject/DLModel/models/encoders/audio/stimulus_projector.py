"""
Stimulus projector wrapper for Type 4.

Why this file exists
--------------------
Type 4 can use multiple stimulus encoder families:
- identity
- linear
- conv
- Bollens-inspired LSTM encoder

To keep `Type4_model.py` readable, we isolate that selection logic here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from DLModel.models.encoders.audio.bollens_speech_encoder import BollensSpeechEncoder


class StimulusProjector(nn.Module):
    """
    Flexible stimulus encoder / projector for Type 4.

    Supported modes
    ---------------
    - "identity"
    - "linear"
    - "conv"
    - "bollens_lstm"

    Input
    -----
    x : torch.Tensor
        Shape [B, T, C_in]

    Output
    ------
    torch.Tensor
        Shape [B, T, D_out]
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        mode: str = "linear",
        conv_kernel: int = 31,
        dropout: float = 0.0,
        base_dim: int = 64,
        lstm_hidden_1: int = 64,
        lstm_hidden_2: int = 4,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of input stimulus features.
        out_dim : int
            Desired output latent dimension.
        mode : str, default="linear"
            Stimulus encoder mode.
        conv_kernel : int, default=31
            Kernel size when mode="conv".
        dropout : float, default=0.0
            Dropout probability.
        base_dim : int, default=64
            Internal feature dimension for the Bollens-style speech encoder.
        lstm_hidden_1 : int, default=64
            Hidden size of first BiLSTM in the Bollens-style encoder.
        lstm_hidden_2 : int, default=4
            Hidden size of second BiLSTM in the Bollens-style encoder.
        """
        super().__init__()

        mode = str(mode).lower()
        self.mode = mode
        self.dropout = nn.Dropout(float(dropout))

        if mode == "identity":
            if input_dim != out_dim:
                raise ValueError(
                    f"StimulusProjector mode='identity' requires input_dim == out_dim, "
                    f"got input_dim={input_dim}, out_dim={out_dim}"
                )
            self.proj = nn.Identity()

        elif mode == "linear":
            self.proj = nn.Linear(input_dim, out_dim)

        elif mode == "conv":
            self.proj = nn.Conv1d(
                in_channels=input_dim,
                out_channels=out_dim,
                kernel_size=int(conv_kernel),
                padding=int(conv_kernel) // 2,
                bias=True,
            )

        elif mode == "bollens_lstm":
            self.proj = BollensSpeechEncoder(
                input_dim=input_dim,
                out_dim=out_dim,
                base_dim=int(base_dim),
                conv_kernel=int(conv_kernel),
                lstm_hidden_1=int(lstm_hidden_1),
                lstm_hidden_2=int(lstm_hidden_2),
                dropout=float(dropout),
            )

        else:
            raise ValueError(f"Unknown stimulus projector mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input stimulus sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, T, C_in].

        Returns
        -------
        torch.Tensor
            Encoded tensor with shape [B, T, D_out].
        """
        if self.mode == "conv":
            x = x.transpose(1, 2)    # [B, C_in, T]
            x = self.proj(x)         # [B, D_out, T]
            x = x.transpose(1, 2)    # [B, T, D_out]
            x = self.dropout(x)
            return x

        if self.mode in {"identity", "linear"}:
            x = self.proj(x)         # [B, T, D_out]
            x = self.dropout(x)
            return x

        if self.mode == "bollens_lstm":
            x = self.proj(x)         # [B, T, D_out]
            return x

        raise RuntimeError(f"Unhandled stimulus mode: {self.mode}")
