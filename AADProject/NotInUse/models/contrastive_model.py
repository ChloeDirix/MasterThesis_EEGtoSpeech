import torch
import torch.nn as nn

from DLModel.models.encoder import EEGEncoder, AudioEncoder


class ContrastiveAADModel(nn.Module):
    """
    EEG input:   [B, T, C_eeg]
    Stim input:  [B, K, T, C_stim]
    Output:      logits [B, K]
    """

    def __init__(self, dl_cfg, eeg_input_dim, stim_input_dim):
        super().__init__()
        self.dl_cfg = dl_cfg

        eeg_cfg = self.dl_cfg["model"]["eeg_encoder"]
        aud_cfg = self.dl_cfg["model"].get("audio_encoder", {})

        # EEG side: keep behavior as close as possible to your working model
        self.eeg_encoder = EEGEncoder(
            input_dim=eeg_input_dim,
            output_dim=stim_input_dim,
            conv_kernel=int(eeg_cfg["conv_kernel"]),
            dropout=float(eeg_cfg["dropout"]),
            use_conv=bool(eeg_cfg["use_conv"]),
        )

        # Audio side: default to identity for the middle step
        audio_mode = str(aud_cfg["mode"])
        audio_out_dim = int(aud_cfg["out_dim"])

        self.audio_encoder = AudioEncoder(
            input_dim=stim_input_dim,
            output_dim=audio_out_dim,
            dropout=float(aud_cfg["dropout"]),
            mode=audio_mode,
            conv_kernel=int(aud_cfg["conv_kernel"]),
        )

        if audio_out_dim != stim_input_dim:
            raise ValueError(
                "For this middle step, audio encoder output_dim must match stim_input_dim, "
                "because the EEG encoder predicts stimulus space directly."
            )

    def predict_stimulus(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, T, C_eeg]
        returns: [B, T, C_stim]
        """
        return self.eeg_encoder(eeg)

    def encode_stimuli(self, stimuli: torch.Tensor) -> torch.Tensor:
        """
        stimuli: [B, K, T, C_stim]
        returns: [B, K, T, C_stim]
        """
        B, K, T, C = stimuli.shape
        stim_flat = stimuli.reshape(B * K, T, C)
        stim_enc = self.audio_encoder(stim_flat)
        stim_enc = stim_enc.reshape(B, K, T, -1)
        return stim_enc

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
        pred_c = pred_stim - pred_stim.mean(dim=1, keepdim=True)     # [B,T,C]
        stim_c = stimuli - stimuli.mean(dim=2, keepdim=True)         # [B,K,T,C]

        pred_c = pred_c.unsqueeze(1)                                 # [B,1,T,C]

        cov = (pred_c * stim_c).mean(dim=2)                          # [B,K,C]

        pred_std = torch.sqrt((pred_c ** 2).mean(dim=2) + eps)       # [B,1,C]
        stim_std = torch.sqrt((stim_c ** 2).mean(dim=2) + eps)       # [B,K,C]

        corr = cov / (pred_std * stim_std + eps)                     # [B,K,C]
        logits = corr.mean(dim=-1)                                   # [B,K]

        return logits

    def forward(self, eeg: torch.Tensor, stimuli: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, T, C_eeg]
        stimuli: [B, K, T, C_stim]
        returns logits: [B, K]
        """
        pred_stim = self.predict_stimulus(eeg)   # [B,T,C_stim]
        stimuli_enc = self.encode_stimuli(stimuli)  # [B,K,T,C_stim]
        logits = self.score_candidates(pred_stim, stimuli_enc)
        return logits