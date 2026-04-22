"""
Transformer-based EEG encoder used by Type 4 style models.

This module is the "richer" EEG encoder family:
- input projection
- residual temporal convolution frontend
- positional encoding
- transformer encoder stack
- output projection

Shape convention
----------------
Input:
    [B, T, C_eeg]
Output:
    [B, T, D_out]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from DLModel.models.encoders.common.positional_encoding import PositionalEncoding
from DLModel.models.encoders.common.temporal_blocks import ResidualTemporalConvBlock


class TransformerEEGEncoder(nn.Module):
    """
    Transformer-based EEG encoder for Type 4.

    Conceptually
    ------------
    Instead of directly predicting a stimulus-like signal as in Type 3,
    this encoder first builds a latent EEG representation that can use:
    - local temporal structure (via convolution)
    - global temporal context (via self-attention)

    Forward path
    ------------
    [B, T, C_eeg]
        -> Linear input projection to d_model
        -> Residual temporal conv frontend
        -> Positional encoding
        -> Transformer encoder stack
        -> Output projection to out_dim
        -> Dropout
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        out_dim: int,
        n_heads: int,
        n_layers: int,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ff_mult: int = 4,
        max_len: int = 4000,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of EEG channels.
        d_model : int
            Internal transformer dimension.
        out_dim : int
            Final latent output dimension.
        n_heads : int
            Number of self-attention heads.
        n_layers : int
            Number of transformer encoder layers.
        conv_kernel : int, default=31
            Kernel size for the residual temporal conv frontend.
        dropout : float, default=0.1
            Dropout probability throughout the encoder.
        ff_mult : int, default=4
            Expansion multiplier for the transformer feedforward block.
            Feedforward size = ff_mult * d_model.
        max_len : int, default=4000
            Maximum supported sequence length for positional encoding.
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, got d_model={d_model}, n_heads={n_heads}"
            )

        # Step 1: map raw EEG channels into the transformer feature space.
        self.input_proj = nn.Linear(input_dim, d_model)

        # Step 2: local temporal feature extraction before attention.
        self.conv_frontend = ResidualTemporalConvBlock(
            d_model=d_model,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )

        # Step 3: inject time-position information.
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Step 4: build the actual transformer stack.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(ff_mult * d_model),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(n_layers),
        )

        # Step 5: project the contextualized hidden states into the final
        # latent dimension used for scoring against encoded stimuli.
        self.output_proj = nn.Linear(d_model, out_dim)
        self.output_dropout = nn.Dropout(float(dropout))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Encode an EEG window into a latent feature sequence.

        Parameters
        ----------
        eeg : torch.Tensor
            Input EEG tensor of shape [B, T, C_eeg].

        Returns
        -------
        torch.Tensor
            Output latent tensor of shape [B, T, D_out].
        """
        x = self.input_proj(eeg)         # [B, T, d_model]
        x = self.conv_frontend(x)        # [B, T, d_model]
        x = self.pos_enc(x)              # [B, T, d_model]
        x = self.transformer(x)          # [B, T, d_model]
        x = self.output_proj(x)          # [B, T, out_dim]
        x = self.output_dropout(x)       # [B, T, out_dim]
        return x
