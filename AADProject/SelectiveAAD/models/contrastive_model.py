"""

Wraps both encoders and defines forward(eeg, stimuli)

Inputs (batched):
- EEG: [B, t, c_eeg]
- Stimuli: [B, K=2, t, c_stim]

Process:
- Encode EEG: z_eeg = eeg_encoder(eeg) → [B, D]
- Flatten stimuli: [B*K, T, C_stim] → audio encoder → [B*K, D]
- Reshape back → [B, K, D]
- Compute similarities: sim[b, k] = similarity(z_eeg[b], z_audio[b, k]) → [B, K]


"""


import torch.nn as nn
import torch.nn.functional as F
from SelectiveAAD.models.encoder import EEGEncoder, AudioEncoder


class ContrastiveAADModel(nn.Module):
    """
    Full model for selective AAD contrastive learning.
    EEG input:    [B, T, C_eeg]
    Stim input:   [B, K, T, C_stim]
    Output:       logits [B, K]
    """
    def __init__(self, eeg_input_dim, stim_input_dim, d_model=64, n_eeg_layers=4, n_stim_layers=2, n_heads=4, temperature=0.07):
        super().__init__()

        self.temperature = temperature
        self.scale = 1.0 / temperature

        # Create encoders
        self.eeg_encoder = EEGEncoder(
            input_dim=eeg_input_dim,
            d_model=d_model,
            n_layers=n_eeg_layers,
            n_heads=n_heads
        )

        self.audio_encoder = AudioEncoder(
            input_dim=stim_input_dim,
            d_model=d_model,
            n_layers=n_stim_layers,
            n_heads=n_heads
        )



    def get_embeddings(self, eeg, stim):
        z_eeg = self.eeg_encoder(eeg)
        z_stim = self.audio_encoder(stim)
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


if __name__ == "__main__":
    from SelectiveAAD import datasets
    from paths import paths
    cfg = paths.load_config()
    nwb_paths = [paths.subject_eegPP("S1")]

    ds = datasets.AADDataset(nwb_paths, cfg, multiband=True)
    loader = datasets.DataLoader(ds, batch_size=4, shuffle=True)

    eeg, stim, att = next(iter(loader))
    print(eeg.shape, stim.shape, att)

    model = ContrastiveAADModel(
        eeg_input_dim=eeg.shape[-1],
        stim_input_dim=stim.shape[-1],
        d_model=64
    )

    logits = model(eeg, stim)
    print("Logits shape:", logits.shape)
    print(logits)
