"""
EEG
input [batch, t, c]

learn linear projection d_model
adds positional encoding
transformer encoder layers
pooling over time

output [batch, d_model]

AUDIO
input [batch, t, n_audio_features]

learn linear projection d_model
adds positional encoding
transformer encoder layers
pooling over time

output [batch, d_model]
"""

import torch
import torch.nn as nn
from SelectiveAAD.models.Transformer_blocks import PositionalEncoding, AttentionPool


class EEGEncoder(nn.Module):
    """
    EEG Encoder:
        Input:  [B, T, C_eeg]
        Output: [B, D]
    """

    def __init__(self, input_dim, d_model=64, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()

        # Project EEG channels -> model dimension
        self.in_proj = nn.Linear(input_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Attention pooling
        self.pool = AttentionPool(d_model)

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x : [B, T, C_eeg]
        """
        x = self.in_proj(x)       # [B, T, D]
        x = self.pos_enc(x)       # [B, T, D]
        x = self.transformer(x)   # [B, T, D]
        x = self.pool(x)          # [B, D]
        x = self.norm(x)
        return x


class AudioEncoder(nn.Module):
    """
    Audio Encoder:
        Input:  [B, T, C_stim]  (C_stim = #bands, e.g., 17 or 1)
        Output: [B, D]
    """

    def __init__(self, input_dim, d_model=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()

        # Same architecture as EEGEncoder, but typically smaller (fewer layers)
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.pool = AttentionPool(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, C_stim]
        """
        x = self.in_proj(x)       # [B, T, D]
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.pool(x)          # [B, D]
        x = self.norm(x)
        return x



if __name__ == "__main__":
    B = 4
    T = 160
    C_eeg = 48
    C_stim = 17

    eeg = torch.randn(B, T, C_eeg)
    stim = torch.randn(B, T, C_stim)

    eeg_enc = EEGEncoder(input_dim=C_eeg)
    audio_enc = AudioEncoder(input_dim=C_stim)

    z_eeg = eeg_enc(eeg)
    z_stim = audio_enc(stim)

    print(z_eeg.shape)   # [4, 64]
    print(z_stim.shape)  # [4, 64]
