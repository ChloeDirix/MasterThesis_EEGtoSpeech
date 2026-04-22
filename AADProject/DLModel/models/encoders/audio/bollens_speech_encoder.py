"""
Bollens-inspired speech encoder for Type 4 style models.

This is a richer stimulus encoder than the simple linear/identity/conv
projector. It follows the spirit of the Bollens paper:
- input projection
- residual temporal convolution
- BiLSTM
- BiLSTM
- output projection

Shape convention
----------------
Input:
    [B, T, C_in]
Output:
    [B, T, D_out]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from DLModel.models.encoders.common.temporal_blocks import ResidualTemporalConvBlock


class BollensSpeechEncoder(nn.Module):
    """
    Bollens-inspired speech encoder.

    Forward path
    ------------
    [B, T, C_in]
        -> Linear input projection to base_dim
        -> Residual temporal conv block
        -> BiLSTM(hidden=lstm_hidden_1)
        -> BiLSTM(hidden=lstm_hidden_2)
        -> Output projection to out_dim
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        base_dim: int = 64,
        conv_kernel: int = 64,
        lstm_hidden_1: int = 64,
        lstm_hidden_2: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of stimulus input features.
        out_dim : int
            Desired output latent dimension.
        base_dim : int, default=64
            Intermediate feature dimension after the input projection.
        conv_kernel : int, default=64
            Kernel size for the residual temporal conv block.
        lstm_hidden_1 : int, default=64
            Hidden size of the first bidirectional LSTM.
        lstm_hidden_2 : int, default=4
            Hidden size of the second bidirectional LSTM.
        dropout : float, default=0.0
            Dropout probability applied between recurrent blocks and at output.
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, int(base_dim))

        self.conv_block = ResidualTemporalConvBlock(
            d_model=int(base_dim),
            conv_kernel=int(conv_kernel),
            dropout=float(dropout),
        )

        self.lstm1 = nn.LSTM(
            input_size=int(base_dim),
            hidden_size=int(lstm_hidden_1),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=2 * int(lstm_hidden_1),
            hidden_size=int(lstm_hidden_2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Because the second LSTM is bidirectional, its output size is:
        # 2 * lstm_hidden_2
        self.output_proj = nn.Linear(2 * int(lstm_hidden_2), int(out_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a stimulus sequence into a latent feature sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, T, C_in].

        Returns
        -------
        torch.Tensor
            Output tensor with shape [B, T, D_out].
        """
        x = self.input_proj(x)      # [B, T, base_dim]
        x = self.conv_block(x)      # [B, T, base_dim]

        x, _ = self.lstm1(x)        # [B, T, 2*lstm_hidden_1]
        x = self.dropout(x)

        x, _ = self.lstm2(x)        # [B, T, 2*lstm_hidden_2]
        x = self.dropout(x)

        x = self.output_proj(x)     # [B, T, out_dim]
        x = self.dropout(x)
        return x
