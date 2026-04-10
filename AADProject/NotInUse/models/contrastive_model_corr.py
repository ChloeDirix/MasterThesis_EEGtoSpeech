import torch
import torch.nn as nn


class ContrastiveAADModel(nn.Module):
    """
    Backward-style decoder model.

    EEG input:   [B, T, C_eeg]
    Stim input:  [B, K, T, C_stim]   (used only for scoring outside forward)
    Output:      predicted stimulus [B, T, C_stim]

    Training:
      EEG -> predicted attended stimulus

    Validation:
      score predicted stimulus against both candidate stimuli via correlation-over-time
    """

    def __init__(self, dl_cfg, eeg_input_dim, stim_input_dim):
        super().__init__()
        self.dl_cfg = dl_cfg

        eeg_cfg = self.dl_cfg["model"]["eeg_encoder"]
        k = int(eeg_cfg["conv_kernel"])
        dropout = float(eeg_cfg.get("dropout", 0.0))

        self.decoder = nn.Conv1d(
            in_channels=eeg_input_dim,
            out_channels=stim_input_dim,
            kernel_size=k,
            padding=k // 2,
            bias=True,
        )

        self.dropout = nn.Dropout(dropout)

    def predict_stimulus(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, T, C_eeg]
        returns: [B, T, C_stim]
        """
        x = eeg.transpose(1, 2)   # [B, C_eeg, T]
        x = self.decoder(x)       # [B, C_stim, T]
        x = x.transpose(1, 2)     # [B, T, C_stim]
        x = self.dropout(x)
        return x

    @staticmethod
    def score_candidates(pred_stim: torch.Tensor, stimuli: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        pred_stim: [B, T, C]
        stimuli:   [B, K, T, C]

        Returns:
            logits [B, K], where each logit is mean Pearson correlation over TIME
            averaged across stimulus channels/bands.
        """
        if pred_stim.ndim != 3:
            raise ValueError(f"pred_stim must be [B,T,C], got {pred_stim.shape}")
        if stimuli.ndim != 4:
            raise ValueError(f"stimuli must be [B,K,T,C], got {stimuli.shape}")

        # center over time
        pred_c = pred_stim - pred_stim.mean(dim=1, keepdim=True)         # [B,T,C]
        stim_c = stimuli - stimuli.mean(dim=2, keepdim=True)             # [B,K,T,C]

        pred_c = pred_c.unsqueeze(1)                                     # [B,1,T,C]

        cov = (pred_c * stim_c).mean(dim=2)                              # [B,K,C]

        pred_std = torch.sqrt((pred_c ** 2).mean(dim=2) + eps)           # [B,1,C]
        stim_std = torch.sqrt((stim_c ** 2).mean(dim=2) + eps)           # [B,K,C]

        corr = cov / (pred_std * stim_std + eps)                         # [B,K,C]
        logits = corr.mean(dim=-1)                                       # [B,K]

        return logits

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, T, C_eeg]
        returns predicted stimulus: [B, T, C_stim]
        """
        return self.predict_stimulus(eeg)