import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    """
    EEG -> stimulus-space prediction
    Input:  [B, T, C_eeg]
    Output: [B, T, C_stim]
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        conv_kernel: int = 31,
        dropout: float = 0.0,
        use_conv: bool = True,
    ):
        super().__init__()
        self.use_conv = bool(use_conv)

        if self.use_conv:
            self.proj = nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=int(conv_kernel),
                padding=int(conv_kernel) // 2,
                bias=True,
            )
        else:
            self.proj = nn.Linear(input_dim, output_dim, bias=True)

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = x.transpose(1, 2)   # [B, C_eeg, T]
            x = self.proj(x)        # [B, C_stim, T]
            x = x.transpose(1, 2)   # [B, T, C_stim]
        else:
            x = self.proj(x)        # [B, T, C_stim]

        x = self.dropout(x)
        return x


class AudioEncoder(nn.Module):
    """
    Stimulus-space audio transform
    Input:  [B, T, C_stim]
    Output: [B, T, C_stim]
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        mode: str = "identity",   # "identity", "linear", "conv"
        conv_kernel: int = 31,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.mode = str(mode).lower()
        self.output_dim = int(output_dim)

        if self.mode == "identity":
            if output_dim != input_dim:
                raise ValueError(
                    f"AudioEncoder mode='identity' requires output_dim == input_dim "
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
            raise ValueError(f"Unknown AudioEncoder mode: {mode}")

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "conv":
            x = x.transpose(1, 2)   # [B, C, T]
            x = self.proj(x)        # [B, C_out, T]
            x = x.transpose(1, 2)   # [B, T, C_out]
        else:
            x = self.proj(x)        # [B, T, C_out]

        x = self.dropout(x)
        return x