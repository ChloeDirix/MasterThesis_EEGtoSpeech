import torch.nn as nn
import torch.nn.functional as F
import torch

from DLModel.models.encoder import EEGEncoder, AudioEncoder


class ContrastiveAADModel(nn.Module):
    """
    Full model for selective AAD contrastive learning.
    EEG input:    [B, T, C_eeg]
    Stim input:   [B, K, T, C_stim]
    Output:       logits [B, K]
    """
    def __init__(self, dl_cfg,eeg_input_dim, stim_input_dim):
        super().__init__()
        self.dl_cfg=dl_cfg
        self.temperature = self.dl_cfg["model"]["contrastive"]["temperature"]
        self.scale = 1.0 / self.temperature

       
        # Create encoders
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
        
    
    # ----------------
    # --- encoders ---
    # ----------------

    def get_pair_embeddings(self, eeg, stimuli):
        """
        eeg:     [B, T, C_eeg]
        stimuli: [B, K, T, C_stim]  (K=2)
        returns:
        z_eeg:  [B, D]
        z_stim: [B, K, D]
        """
        B, K, T, C = stimuli.shape
        z_eeg = self.eeg_encoder(eeg)                 # [B,D]
        stim_flat = stimuli.reshape(B * K, T, C)
        z_stim = self.audio_encoder(stim_flat).reshape(B, K, -1)  # [B,K,D]
        return z_eeg, z_stim

    
    def forward(self, eeg, stimuli):
        """
        eeg:      [B, T, C_eeg]
        stimuli:  [B, K, T, C_stim]
        """

        B, K, T, C = stimuli.shape

        # ---- Encode EEG ----
        z_eeg = self.eeg_encoder(eeg)            # [B, d_model]

        # ---- Encode stimuli ----
        stim_flat = stimuli.reshape(B * K, T, C)  # [B*K, T, C]
        z_stim = self.audio_encoder(stim_flat)    # [B*K, d_model]
        z_stim = z_stim.reshape(B, K, -1)         # [B, K, d_model]
                                                  # z_stim[b][0] = embedding of matched stimulus
                                                  # z_stim[b][1] = embedding of mismatched stimulus


        # ---- Cosine similarities ----
        z_eeg_norm = F.normalize(z_eeg, dim=-1)
        z_stim_norm = F.normalize(z_stim, dim=-1)

        logits = (z_eeg_norm.unsqueeze(1) * z_stim_norm).sum(dim=-1)  # dot product
                    # z_eeg_norm.unsqueeeze(1) : dim becomes: [B, 1, D]
                    # * is an elementwise multiplication: [B, K, D]
                    # sum over features: [B,K]

        logits = logits * self.scale    # scale=1/temperature: improves contrast

        return logits









    # ----------------
    # ---- losses ----
    # ----------------
   
    # contrastive loss function
    def contrastive_loss(self, logits, att):
        """
        logits: [B, K]
        att:    [B]  (0 or 1)
        Implements 2-way InfoNCE = CE over similarities
        """
        return F.cross_entropy(logits, att)

    #to check add symmetric CLIP and add negatives
    #calssifying AAD Types

    def clip_loss(self, z_eeg, z_stim):
        """
        z_eeg:  [B, D]
        z_stim: [B, D]

        Computes symmetric InfoNCE (CLIP) loss:
          L = (L_eeg2stim + L_stim2eeg) / 2
        """

        # Normalize embeddings
        z_eeg = F.normalize(z_eeg, dim=-1)  # [B, D]
        z_stim = F.normalize(z_stim, dim=-1)  # [B, D]

        # Similarity matrix: [B, B]
        logits = (z_eeg @ z_stim.T) / self.temperature

        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)

        # EEG → Stim loss: row i should match column i
        loss_e2s = F.cross_entropy(logits, labels)

        # Stim → EEG loss: match inverse direction
        loss_s2e = F.cross_entropy(logits.T, labels)

        # Symmetric CLIP loss
        loss = (loss_e2s + loss_s2e) / 2
        return loss

