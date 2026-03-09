import torch
import torch.nn as nn
import torch.nn.functional as F

from DLModel.models.Transformer_blocks import PositionalEncoding, AttentionPool


# -----------------------------
# Convolution block
# --> pythorch module
# Input  : [B, T, C]
# Output : [B, T, d_model]
# -----------------------------
class Conv(nn.Module):
    def __init__(self, in_dim: int, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.kernel_size = int(kernel_size)

        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            padding=0,  #put zero because we do manual padding
            bias=True,
        )
        self.norm = nn.LayerNorm(d_model)  #normalizes features per time step (stabilizes training)
        self.act = nn.GELU()               #smooth acitvation 
        self.drop = nn.Dropout(dropout)    #randomly zeros some activation during training (reduces overfitting)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x_c = x.transpose(1, 2)  # [B, C, T]

        #do the manual padding
        k = self.kernel_size
        pad_left = (k - 1) // 2
        pad_right = (k - 1) - pad_left
        x_pad = F.pad(x_c, (pad_left, pad_right))  # pad time dim

        y = self.conv(x_pad)        # [B, d_model, T]
        y = y.transpose(1, 2)       # [B, T, d_model]
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        return y


# -----------------------------
# Shared Transformer backbone
# -----------------------------
def make_transformer(d_model: int, n_layers: int, n_heads: int, dropout: float) -> nn.TransformerEncoder:
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=4 * d_model,    #size of MLP inside each layer
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,   # often more stable
    )
    return nn.TransformerEncoder(enc_layer, num_layers=n_layers)


# ============================================================
# EEG Encoder
# ============================================================
class EEGEncoder(nn.Module):
    """
    EEG Encoder (hybrid):
      Input : [B, T, C_eeg]
      Output: [B, out_dim]

    Pipeline:
      (optional) Conv1D front-end over time -> [B,T,d_model]
      (optional) Linear projection if conv not used
      PositionalEncoding
      TransformerEncoder
      AttentionPool over time
      LayerNorm
      (optional) out_proj to out_dim
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        conv_kernel: int = 31,
        out_dim: int | None = None,
        pool: str = "attn",  # "attn" (recommended) or "mean"
        use_conv: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.out_dim = int(d_model if out_dim is None else out_dim)
        self.pool_mode = str(pool).lower()
        self.use_conv = bool(use_conv)

        # Either conv-front-end OR direct linear projection to d_model
        if self.use_conv:
            self.frontend = Conv(input_dim, self.d_model, conv_kernel, dropout)
            self.in_proj = nn.Identity()
        else:
            self.frontend = nn.Identity()
            self.in_proj = nn.Sequential(
                nn.Linear(input_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.pos_enc = PositionalEncoding(self.d_model)
        self.transformer = make_transformer(self.d_model, n_layers, n_heads, dropout)

        # Pooling
        self.attn_pool = AttentionPool(self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

        # Output projection if needed
        self.out_proj = nn.Identity() if self.out_dim == self.d_model else nn.Linear(self.d_model, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C_eeg]
        x = self.frontend(x)    # [B, T, d_model] if use_conv else identity
        x = self.in_proj(x)     # [B, T, d_model]

        x = self.pos_enc(x)     # [B, T, d_model]
        x = self.transformer(x) # [B, T, d_model]

        if self.pool_mode in ("attn", "attention"):
            x = self.attn_pool(x)        # [B, d_model]
        elif self.pool_mode == "mean":
            x = x.mean(dim=1)            # [B, d_model]
        else:
            raise ValueError(f"Unknown pool='{self.pool_mode}'. Use 'attn' or 'mean'.")

        x = self.norm(x)                 # [B, d_model]
        x = self.out_proj(x)             # [B, out_dim]
        return x


# ============================================================
# Audio Encoder
# ============================================================
class AudioEncoder(nn.Module):
    """
    Audio Encoder (hybrid, symmetric to EEG):
      Input : [B, T, C_stim]
      Output: [B, out_dim]

    Pipeline:
      (optional) Conv1D front-end OR Linear projection
      PositionalEncoding
      TransformerEncoder
      AttentionPool
      LayerNorm
      (optional) out_proj
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        dropout: float = 0.1,
        conv_kernel: int = 31,
        out_dim: int | None = None,
        use_conv: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.out_dim = int(d_model if out_dim is None else out_dim)
        self.use_conv = bool(use_conv)

        if self.use_conv:
            self.frontend = Conv(input_dim, self.d_model, conv_kernel, dropout)
            proj_in=self.d_model
        else:
            self.frontend = nn.Identity()
            proj_in = input_dim

        self.proj = nn.Sequential(
            nn.Linear(proj_in, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Identity() if self.out_dim == self.d_model else nn.Linear(self.d_model, self.out_dim)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C_stim]
        x = self.frontend(x)  # [B,T,d_model] or identity
        x = self.proj(x)      # [B,T,d_model]
        x = x.mean(dim=1)     # [B,d_model]  (simple + robust)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


if __name__ == "__main__":
    B, T = 4, 160
    C_eeg, C_stim = 64, 17

    eeg = torch.randn(B, T, C_eeg)
    stim = torch.randn(B, T, C_stim)

    eeg_enc = EEGEncoder(input_dim=C_eeg, d_model=128, n_layers=4, n_heads=8, dropout=0.1, conv_kernel=31, out_dim=128, pool="attn")
    aud_enc = AudioEncoder(input_dim=C_stim, d_model=128, n_layers=2, n_heads=8, dropout=0.1, conv_kernel=31, out_dim=128, pool="attn")

    z_eeg = eeg_enc(eeg)
    z_stim = aud_enc(stim)

    print("EEG:", z_eeg.shape)   # [B, 128]
    print("AUD:", z_stim.shape)  # [B, 128]