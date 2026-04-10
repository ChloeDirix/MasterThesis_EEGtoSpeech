import torch.nn as nn
import torch.nn.functional as F
import torch

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
        self.temperature = self.dl_cfg["model"]["contrastive"]["temperature"]
        self.scale = 1.0 / self.temperature

        self.eeg_encoder = EEGEncoder(
            input_dim=eeg_input_dim,
            d_model=self.dl_cfg["model"]["eeg_encoder"]["d_model"],
            n_layers=self.dl_cfg["model"]["eeg_encoder"]["n_layers"],
            n_heads=self.dl_cfg["model"]["eeg_encoder"]["n_heads"],
            dropout=self.dl_cfg["model"]["eeg_encoder"]["dropout"],
            conv_kernel=self.dl_cfg["model"]["eeg_encoder"]["conv_kernel"],
            out_dim=self.dl_cfg["model"]["eeg_encoder"]["out_dim"],
            pool=self.dl_cfg["model"]["eeg_encoder"]["pool"],
            use_conv=self.dl_cfg["model"]["eeg_encoder"].get("use_conv", True),
        )

        self.audio_encoder = AudioEncoder(
            input_dim=stim_input_dim,
            d_model=self.dl_cfg["model"]["audio_encoder"]["d_model"],
            dropout=self.dl_cfg["model"]["audio_encoder"]["dropout"],
            conv_kernel=self.dl_cfg["model"]["audio_encoder"]["conv_kernel"],
            out_dim=self.dl_cfg["model"]["audio_encoder"]["out_dim"],
            use_conv=self.dl_cfg["model"]["audio_encoder"].get("use_conv", True),
        )

    def get_pair_embeddings(self, eeg, stimuli):
        """
        eeg:     [B, T, C_eeg]
        stimuli: [B, K, T, C_stim] or [B, T, C_stim]

        returns:
        z_eeg:  [B, T, D]
        z_stim: [B, K, T, D] or [B, T, D]
        """
        z_eeg = self.eeg_encoder(eeg)   # [B, T, D]

        if stimuli.dim() == 4:
            B, K, T, C = stimuli.shape
            stimuli_flat = stimuli.view(B * K, T, C)
            z_stim = self.audio_encoder(stimuli_flat)     # [B*K, T, D]
            z_stim = z_stim.view(B, K, T, -1)             # [B, K, T, D]

        elif stimuli.dim() == 3:
            z_stim = self.audio_encoder(stimuli)          # [B, T, D]

        else:
            raise ValueError(f"Expected stimuli with 3 or 4 dims, got shape {stimuli.shape}")

        return z_eeg, z_stim

    def forward(self, eeg, stimuli):
        """
        eeg:     [B, T, C_eeg]
        stimuli: [B, K, T, C_stim]
        returns logits: [B, K]
        """
        B, K, T, C = stimuli.shape

        z_eeg = self.eeg_encoder(eeg)                      # [B, T, D]

        stim_flat = stimuli.reshape(B * K, T, C)
        z_stim = self.audio_encoder(stim_flat)            # [B*K, T, D]
        z_stim = z_stim.reshape(B, K, T, -1)              # [B, K, T, D]

        z_eeg = F.normalize(z_eeg, dim=-1)                # [B, T, D]
        z_stim = F.normalize(z_stim, dim=-1)              # [B, K, T, D]

        # timewise cosine similarity
        sim_t = (z_eeg.unsqueeze(1) * z_stim).sum(dim=-1)   # [B, K, T]

        # average over time only at the end
        logits = sim_t.mean(dim=-1)                         # [B, K]
        logits = logits * self.scale

        return logits

    def contrastive_loss(self, logits, att):
        return F.cross_entropy(logits, att)

    def clip_loss(self, z_eeg, z_stim):
        """
        z_eeg:  [B, T, D]
        z_stim: [B, T, D]
        """
        z_eeg = F.normalize(z_eeg, dim=-1)
        z_stim = F.normalize(z_stim, dim=-1)

        z_eeg = z_eeg.mean(dim=1)   # only for CLIP-style batch loss
        z_stim = z_stim.mean(dim=1)

        logits = (z_eeg @ z_stim.T) / self.temperature
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)

        loss_e2s = F.cross_entropy(logits, labels)
        loss_s2e = F.cross_entropy(logits.T, labels)

        return (loss_e2s + loss_s2e) / 2