import torch
import torch.nn as nn

from DLModel.models.Transformer_blocks import PositionalEncoding


class ConvFrontendBlock(nn.Module):
    """
    Small temporal conv frontend with residual connection.

    Input:  [B, T, D]
    Output: [B, T, D]
    """
    def __init__(self, d_model: int, conv_kernel: int = 31, dropout: float = 0.0):
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
        residual = x
        y = x.transpose(1, 2)            # [B, D, T]
        y = self.conv(y)                 # [B, D, T]
        y = y.transpose(1, 2)            # [B, T, D]
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return residual + y


class EEGTransformerEncoder(nn.Module):
    """
    EEG encoder that is more Bollens-like than the current Type3 encoder,
    but still returns framewise latent features for ranking.

    Input:  [B, T, C_eeg]
    Output: [B, T, D_out]
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
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, got d_model={d_model}, n_heads={n_heads}"
            )

        self.input_proj = nn.Linear(input_dim, d_model)

        self.conv_frontend = ConvFrontendBlock(
            d_model=d_model,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )

        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

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

        self.output_proj = nn.Linear(d_model, out_dim)
        self.output_dropout = nn.Dropout(float(dropout))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(eeg)         # [B, T, d_model]
        x = self.conv_frontend(x)        # [B, T, d_model]
        x = self.pos_enc(x)              # [B, T, d_model]
        x = self.transformer(x)          # [B, T, d_model]
        x = self.output_proj(x)          # [B, T, out_dim]
        x = self.output_dropout(x)
        return x


class StimulusProjector(nn.Module):
    """
    Stimulus encoder / projector.

    Supported modes:
      - "identity"
      - "linear"
      - "conv"
      - "bollens_lstm"

    Input:  [B, T, C_stim]
    Output: [B, T, D_out]
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
    ):
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

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "conv":
            x = x.transpose(1, 2)    # [B, C, T]
            x = self.proj(x)         # [B, D, T]
            x = x.transpose(1, 2)    # [B, T, D]
            x = self.dropout(x)
            return x

        elif self.mode in {"identity", "linear"}:
            x = self.proj(x)         # [B, T, D]
            x = self.dropout(x)
            return x

        elif self.mode == "bollens_lstm":
            x = self.proj(x)         # [B, T, D]
            return x

        else:
            raise RuntimeError(f"Unhandled stimulus mode: {self.mode}")


class TransformerRankAADModel(nn.Module):
    """
    Type4:
    - EEG path: conv + transformer encoder
    - Stimulus path: lightweight projector
    - Output: candidate scores [B, K]

    eeg:     [B, T, C_eeg]
    stimuli: [B, K, T, C_stim]
    logits:  [B, K]
    """

    def __init__(self, dl_cfg, eeg_input_dim: int, stim_input_dim: int):
        super().__init__()
        self.dl_cfg = dl_cfg

        eeg_cfg = self.dl_cfg["model"]["eeg_encoder"]
        aud_cfg = self.dl_cfg["model"].get("audio_encoder", {})

        d_model = int(eeg_cfg.get("d_model", 128))
        out_dim = int(eeg_cfg.get("out_dim", stim_input_dim))
        n_heads = int(eeg_cfg.get("n_heads", 8))
        n_layers = int(eeg_cfg.get("n_layers", 2))
        dropout = float(eeg_cfg.get("dropout", 0.1))
        conv_kernel = int(eeg_cfg.get("conv_kernel", 31))
        ff_mult = int(eeg_cfg.get("ff_mult", 4))

        self.eeg_encoder = EEGTransformerEncoder(
            input_dim=eeg_input_dim,
            d_model=d_model,
            out_dim=out_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            conv_kernel=conv_kernel,
            dropout=dropout,
            ff_mult=ff_mult,
            max_len=int(eeg_cfg.get("max_len", 4000)),
        )

        self.audio_encoder = StimulusProjector(
            input_dim=stim_input_dim,
            out_dim=out_dim,
            mode=str(aud_cfg.get("mode", "linear")),
            conv_kernel=int(aud_cfg.get("conv_kernel", 31)),
            dropout=float(aud_cfg.get("dropout", 0.0)),
            base_dim=int(aud_cfg.get("base_dim", 64)),
            lstm_hidden_1=int(aud_cfg.get("lstm_hidden_1", 64)),
            lstm_hidden_2=int(aud_cfg.get("lstm_hidden_2", 4)),
        )

    def predict_latent(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.eeg_encoder(eeg)

    def encode_stimuli(self, stimuli: torch.Tensor) -> torch.Tensor:
        B, K, T, C = stimuli.shape
        flat = stimuli.reshape(B * K, T, C)
        enc = self.audio_encoder(flat)
        enc = enc.reshape(B, K, T, -1)
        return enc

    @staticmethod
    def score_candidates(
        eeg_latent: torch.Tensor,
        stim_latent: torch.Tensor,
        eps: float = 1e-8,
        normalize: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        eeg_latent:  [B, T, D]
        stim_latent: [B, K, T, D]
        returns:     [B, K]
        """
        x = eeg_latent
        y = stim_latent

        x = x.unsqueeze(1)  # [B, 1, T, D]

        if normalize:
            x = x / (torch.norm(x, dim=-1, keepdim=True) + eps)
            y = y / (torch.norm(y, dim=-1, keepdim=True) + eps)

        sim_t = (x * y).sum(dim=-1)      # [B, K, T]
        logits = sim_t.mean(dim=-1)      # [B, K]

        temperature = max(float(temperature), eps)
        logits = logits / temperature
        return logits

    def forward(self, eeg: torch.Tensor, stimuli: torch.Tensor) -> torch.Tensor:
        eeg_latent = self.predict_latent(eeg)        # [B, T, D]
        stim_latent = self.encode_stimuli(stimuli)  # [B, K, T, D]

        contrastive_cfg = self.dl_cfg["model"].get("contrastive", {})
        logits = self.score_candidates(
            eeg_latent=eeg_latent,
            stim_latent=stim_latent,
            normalize=bool(self.dl_cfg["loss"].get("normalize", True)),
            temperature=float(
                contrastive_cfg.get(
                    "temperature",
                    self.dl_cfg["loss"].get("temperature", 1.0),
                )
            ),
        )
        return logits

    
import torch
import torch.nn as nn


class SpeechConvResidualBlock(nn.Module):
    """
    Residual temporal conv block used in the speech path.

    Input:  [B, T, D]
    Output: [B, T, D]
    """
    def __init__(self, d_model: int, conv_kernel: int = 64, dropout: float = 0.0):
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
        residual = x
        y = x.transpose(1, 2)   # [B, D, T]
        y = self.conv(y)        # [B, D, T]
        y = y.transpose(1, 2)   # [B, T, D]
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return residual + y


class BollensSpeechEncoder(nn.Module):
    """
    Bollens-inspired speech encoder:
      input -> FC(64) -> conv residual block -> BiLSTM(64) -> BiLSTM(4) -> output projection

    Input:  [B, T, C_in]
    Output: [B, T, D_out]
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
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, int(base_dim))

        self.conv_block = SpeechConvResidualBlock(
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

        # BiLSTM with hidden_size=4 -> output dim = 8, exactly like the paper
        self.output_proj = nn.Linear(2 * int(lstm_hidden_2), int(out_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)      # [B, T, 64]
        x = self.conv_block(x)      # [B, T, 64]

        x, _ = self.lstm1(x)        # [B, T, 128]
        x = self.dropout(x)

        x, _ = self.lstm2(x)        # [B, T, 8]
        x = self.dropout(x)

        x = self.output_proj(x)     # [B, T, out_dim]
        return x