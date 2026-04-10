import torch
import torch.nn as nn

from DLModel.models.encoder import EEGEncoder, AudioEncoder


class DirectCorrAADModel(nn.Module):
    """
    eeg:     [B, T, C_eeg]
    stimuli: [B, K, T, C_stim]

    returns dict with:
      pred:       [B, T, C_stim]
      candidates: [B, K, T, C_stim]
      scores:     [B, K]   (correlation scores, useful for monitoring)
    """

    def __init__(self, dl_cfg, eeg_input_dim, stim_input_dim):
        super().__init__()
        self.dl_cfg = dl_cfg

        eeg_cfg = self.dl_cfg["model"]["eeg_encoder"]
        aud_cfg = self.dl_cfg["model"].get("audio_encoder", {})

        self.eeg_encoder = EEGEncoder(
            input_dim=eeg_input_dim,
            output_dim=stim_input_dim,
            conv_kernel=int(eeg_cfg.get("conv_kernel", 31)),
            dropout=float(eeg_cfg.get("dropout", 0.0)),
            use_conv=bool(eeg_cfg.get("use_conv", True)),
        )

        audio_out_dim = int(aud_cfg.get("out_dim", stim_input_dim))
        self.audio_encoder = AudioEncoder(
            input_dim=stim_input_dim,
            output_dim=audio_out_dim,
            dropout=float(aud_cfg.get("dropout", 0.0)),
            mode=str(aud_cfg.get("mode", "identity")),
            conv_kernel=int(aud_cfg.get("conv_kernel", 31)),
        )

        if audio_out_dim != stim_input_dim:
            raise ValueError(
                "DirectCorrAADModel requires audio encoder output_dim == stim_input_dim."
            )

    def predict_stimulus(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.eeg_encoder(eeg)

    def encode_stimuli(self, stimuli: torch.Tensor) -> torch.Tensor:
        B, K, T, C = stimuli.shape
        stim_flat = stimuli.reshape(B * K, T, C)
        stim_enc = self.audio_encoder(stim_flat)
        stim_enc = stim_enc.reshape(B, K, T, -1)
        return stim_enc

    @staticmethod
    def score_candidates(pred_stim: torch.Tensor, stimuli: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if pred_stim.ndim != 3:
            raise ValueError(f"pred_stim must be [B,T,C], got {pred_stim.shape}")
        if stimuli.ndim != 4:
            raise ValueError(f"stimuli must be [B,K,T,C], got {stimuli.shape}")

        pred_c = pred_stim - pred_stim.mean(dim=1, keepdim=True)     # [B,T,C]
        stim_c = stimuli - stimuli.mean(dim=2, keepdim=True)         # [B,K,T,C]

        pred_c = pred_c.unsqueeze(1)                                 # [B,1,T,C]

        cov = (pred_c * stim_c).mean(dim=2)                          # [B,K,C]
        pred_std = torch.sqrt((pred_c ** 2).mean(dim=2) + eps)       # [B,1,C]
        stim_std = torch.sqrt((stim_c ** 2).mean(dim=2) + eps)       # [B,K,C]

        corr = cov / (pred_std * stim_std + eps)                     # [B,K,C]
        scores = corr.mean(dim=-1)                                   # [B,K]
        return scores

    def forward(self, eeg: torch.Tensor, stimuli: torch.Tensor) -> dict:
        pred = self.predict_stimulus(eeg)          # [B,T,C]
        candidates = self.encode_stimuli(stimuli)  # [B,K,T,C]
        scores = self.score_candidates(pred, candidates)

        return {
            "pred": pred,
            "candidates": candidates,
            "scores": scores,
        }