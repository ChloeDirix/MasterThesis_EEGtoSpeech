"""
Type 4 auditory attention decoding model.

This version refactors the encoder logic into dedicated encoder modules:
- EEG transformer encoder -> encoders/eeg/transformer_eeg_encoder.py
- flexible stimulus encoder -> encoders/audio/stimulus_projector.py

Conceptually
------------
Type 4 is a latent-space ranking model:
1. EEG is encoded into a contextualized latent sequence.
2. Each candidate stimulus is encoded into the same latent space.
3. Similarity is computed between EEG latent sequence and each stimulus latent
   sequence.
4. The candidate with the highest score is predicted as attended.

Tensor shapes
-------------
eeg:
    [B, T, C_eeg]

stimuli:
    [B, K, T, C_stim]

logits:
    [B, K]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from DLModel.models.encoders.eeg.transformer_eeg_encoder import TransformerEEGEncoder
from DLModel.models.encoders.audio.stimulus_projector import StimulusProjector


class TransformerRankAADModel(nn.Module):
    """
    Type 4 transformer-based ranking model.

    Parameters
    ----------
    dl_cfg : dict
        Deep learning configuration dictionary.
    eeg_input_dim : int
        Number of EEG input channels.
    stim_input_dim : int
        Number of stimulus input features.
    """

    def __init__(self, dl_cfg, eeg_input_dim: int, stim_input_dim: int) -> None:
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

        # EEG encoder: richer path with temporal conv + transformer context.
        self.eeg_encoder = TransformerEEGEncoder(
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

        # Stimulus encoder: flexible projector into the same latent dimension.
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
        """
        Encode EEG into latent sequence features.

        Parameters
        ----------
        eeg : torch.Tensor
            EEG input of shape [B, T, C_eeg].

        Returns
        -------
        torch.Tensor
            EEG latent sequence of shape [B, T, D].
        """
        return self.eeg_encoder(eeg)

    def encode_stimuli(self, stimuli: torch.Tensor) -> torch.Tensor:
        """
        Encode all candidate stimuli.

        Parameters
        ----------
        stimuli : torch.Tensor
            Input candidate tensor with shape [B, K, T, C_stim].

        Returns
        -------
        torch.Tensor
            Encoded candidate tensor with shape [B, K, T, D].
        """
        B, K, T, C = stimuli.shape

        # Flatten candidates into the batch dimension so one encoder call can
        # process all candidates.
        flat = stimuli.reshape(B * K, T, C)   # [B*K, T, C_stim]
        enc = self.audio_encoder(flat)        # [B*K, T, D]
        enc = enc.reshape(B, K, T, -1)        # [B, K, T, D]
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
        Compute similarity scores between EEG latents and candidate stimulus latents.

        Parameters
        ----------
        eeg_latent : torch.Tensor
            Tensor of shape [B, T, D].
        stim_latent : torch.Tensor
            Tensor of shape [B, K, T, D].
        eps : float, default=1e-8
            Numerical stability constant.
        normalize : bool, default=True
            Whether to L2-normalize latent vectors across the feature dimension.
        temperature : float, default=1.0
            Temperature scaling applied to the final logits.

        Returns
        -------
        torch.Tensor
            Candidate logits with shape [B, K].

        Explanation
        -----------
        1. Expand EEG latent to [B, 1, T, D] so it can broadcast against K candidates.
        2. Optionally normalize feature vectors.
        3. Compute dot-product similarity per time step.
        4. Average over time to get one score per candidate.
        5. Divide by temperature to control score sharpness.
        """
        x = eeg_latent                   # [B, T, D]
        y = stim_latent                  # [B, K, T, D]

        x = x.unsqueeze(1)               # [B, 1, T, D]

        if normalize:
            x = x / (torch.norm(x, dim=-1, keepdim=True) + eps)
            y = y / (torch.norm(y, dim=-1, keepdim=True) + eps)

        # Dot-product similarity for each time sample.
        sim_t = (x * y).sum(dim=-1)      # [B, K, T]

        # Average similarity over time to produce one logit per candidate.
        logits = sim_t.mean(dim=-1)      # [B, K]

        # Avoid division by zero while still allowing user-controlled temperature.
        temperature = max(float(temperature), eps)
        logits = logits / temperature
        return logits

    def forward(self, eeg: torch.Tensor, stimuli: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Type 4.

        Parameters
        ----------
        eeg : torch.Tensor
            EEG tensor of shape [B, T, C_eeg].
        stimuli : torch.Tensor
            Candidate stimuli tensor of shape [B, K, T, C_stim].

        Returns
        -------
        torch.Tensor
            Candidate logits with shape [B, K].
        """
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
