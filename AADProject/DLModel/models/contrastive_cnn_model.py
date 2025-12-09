import torch
import torch.nn as nn
import torch.nn.functional as F

from DLModel.models.ccn_encoders import EEGEncoder_CNN, AudioEncoder_CNN


class ContrastiveAADModel_CNN(nn.Module):
    """
    Fast contrastive AAD with CNN encoders.
    """
    def __init__(self, eeg_input_dim, stim_input_dim, d_model=32, temperature=0.1):
        super().__init__()

        self.temperature = temperature
        self.scale = 1.0 / temperature

        self.eeg_encoder = EEGEncoder_CNN(
            input_dim=eeg_input_dim,
            d_model=d_model,
            dropout=0.2
        )

        self.audio_encoder = AudioEncoder_CNN(
            input_dim=stim_input_dim,
            d_model=d_model,
            dropout=0.2
        )

    def get_embeddings(self, eeg, stim):
        z_eeg = self.eeg_encoder(eeg)
        z_stim = self.audio_encoder(stim)
        return z_eeg, z_stim

    def forward(self, eeg, stimuli):
        """
        eeg: [B, T, C_eeg]
        stimuli: [B, K, T, C_stim]
        """
        B, K, T, C = stimuli.shape

        z_eeg = self.eeg_encoder(eeg)                 # [B, D]

        stim_flat = stimuli.reshape(B * K, T, C)
        z_stim = self.audio_encoder(stim_flat)        # [B*K, D]
        z_stim = z_stim.reshape(B, K, -1)             # [B, K, D]

        # Normalize
        z_eeg = F.normalize(z_eeg, dim=-1)
        z_stim = F.normalize(z_stim, dim=-1)

        # Compute logits
        logits = (z_eeg.unsqueeze(1) * z_stim).sum(dim=-1)  # [B, K]
        logits = logits * self.scale

        return logits
