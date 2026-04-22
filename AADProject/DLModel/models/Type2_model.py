"""
Type 2 auditory attention decoding model.

This version keeps the original model logic, but its internal encoder
imports are now cleaner:
- EEG encoder lives under encoders/eeg/
- audio encoder lives under encoders/audio/

Conceptually
------------
Type 2 is the most direct reconstruction-style model of the three:
1. EEG is projected into a stimulus-like representation.
2. Each candidate stimulus is optionally encoded.
3. Correlation-style scores are computed between the EEG-derived
   prediction and each candidate stimulus.
4. During training, the attended candidate itself is used as the target
   for the reconstruction-style loss in the LightningModule.

Tensor shapes
-------------
eeg:
    [B, T, C_eeg]

stimuli:
    [B, K, T, C_stim]

returns:
    {
      "pred":       [B, T, C_stim],
      "candidates": [B, K, T, C_stim],
      "scores":     [B, K]
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn

from DLModel.models.encoders.eeg.simple_eeg_encoder import SimpleEEGEncoder
from DLModel.models.encoders.audio.simple_audio_encoder import SimpleAudioEncoder


class DirectCorrAADModel(nn.Module):
    """
    Type 2 direct-correlation auditory attention decoding model.

    Parameters
    ----------
    dl_cfg : dict
        Deep learning section of the configuration.
    eeg_input_dim : int
        Number of EEG input channels.
    stim_input_dim : int
        Number of stimulus input features.

    Notes
    -----
    This class preserves the behavior of the original Type 2 model:
    - EEG is mapped directly into a stimulus-like prediction.
    - Candidate stimuli are optionally transformed.
    - Correlation-style scores are computed for monitoring / inference.

    The main structural change is only that the encoder classes are now
    imported from the new encoder folder layout.
    """

    def __init__(self, dl_cfg, eeg_input_dim: int, stim_input_dim: int) -> None:
        super().__init__()
        self.dl_cfg = dl_cfg

        eeg_cfg = self.dl_cfg["model"]["eeg_encoder"]
        aud_cfg = self.dl_cfg["model"].get("audio_encoder", {})

        # EEG path:
        # Map EEG [B, T, C_eeg] directly into a stimulus-like signal
        # [B, T, C_stim]. This is the reconstruction-style core of Type 2.
        self.eeg_encoder = SimpleEEGEncoder(
            input_dim=eeg_input_dim,
            output_dim=stim_input_dim,
            conv_kernel=int(eeg_cfg.get("conv_kernel", 31)),
            dropout=float(eeg_cfg.get("dropout", 0.0)),
            use_conv=bool(eeg_cfg.get("use_conv", True)),
        )

        # Stimulus path:
        # Optionally transform candidate stimuli before correlation scoring.
        audio_out_dim = int(aud_cfg.get("out_dim", stim_input_dim))
        self.audio_encoder = SimpleAudioEncoder(
            input_dim=stim_input_dim,
            output_dim=audio_out_dim,
            dropout=float(aud_cfg.get("dropout", 0.0)),
            mode=str(aud_cfg.get("mode", "identity")),
            conv_kernel=int(aud_cfg.get("conv_kernel", 31)),
        )

        # Type 2 compares the EEG-derived prediction against candidate
        # stimuli in the same feature space, so dimensions must match.
        if audio_out_dim != stim_input_dim:
            raise ValueError(
                "DirectCorrAADModel requires audio encoder output_dim == stim_input_dim."
            )

    def predict_stimulus(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Encode EEG into a stimulus-like prediction.

        Parameters
        ----------
        eeg : torch.Tensor
            EEG tensor of shape [B, T, C_eeg].

        Returns
        -------
        torch.Tensor
            Predicted stimulus-like tensor of shape [B, T, C_stim].
        """
        return self.eeg_encoder(eeg)

    def encode_stimuli(self, stimuli: torch.Tensor) -> torch.Tensor:
        """
        Encode each candidate stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Candidate stimuli tensor of shape [B, K, T, C_stim].

        Returns
        -------
        torch.Tensor
            Encoded candidates tensor of shape [B, K, T, C_stim].
        """
        B, K, T, C = stimuli.shape

        # Flatten the candidate dimension into the batch dimension so the
        # same audio encoder can process all candidates in one forward pass.
        stim_flat = stimuli.reshape(B * K, T, C)   # [B*K, T, C_stim]
        stim_enc = self.audio_encoder(stim_flat)   # [B*K, T, C_stim]
        stim_enc = stim_enc.reshape(B, K, T, -1)   # [B, K, T, C_stim]
        return stim_enc

    @staticmethod
    def score_candidates(
        pred_stim: torch.Tensor,
        stimuli: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute correlation-style scores for each stimulus candidate.

        Parameters
        ----------
        pred_stim : torch.Tensor
            EEG-derived predicted stimulus representation, shape [B, T, C].
        stimuli : torch.Tensor
            Candidate stimuli representations, shape [B, K, T, C].
        eps : float, default=1e-8
            Small constant for numerical stability.

        Returns
        -------
        torch.Tensor
            Candidate scores with shape [B, K].

        Explanation
        -----------
        1. Center the EEG-derived prediction over time.
        2. Center each candidate stimulus over time.
        3. Compute covariance over the time dimension.
        4. Normalize by the standard deviations to get a Pearson-like
           correlation per feature channel.
        5. Average over feature channels to get one score per candidate.
        """
        if pred_stim.ndim != 3:
            raise ValueError(f"pred_stim must be [B,T,C], got {pred_stim.shape}")
        if stimuli.ndim != 4:
            raise ValueError(f"stimuli must be [B,K,T,C], got {stimuli.shape}")

        # Center the EEG-derived prediction over time.
        pred_c = pred_stim - pred_stim.mean(dim=1, keepdim=True)     # [B, T, C]

        # Center each candidate stimulus over time.
        stim_c = stimuli - stimuli.mean(dim=2, keepdim=True)         # [B, K, T, C]

        # Add the candidate axis to enable broadcasting with [B, K, T, C].
        pred_c = pred_c.unsqueeze(1)                                 # [B, 1, T, C]

        # Covariance over time.
        cov = (pred_c * stim_c).mean(dim=2)                          # [B, K, C]

        # Standard deviations over time.
        pred_std = torch.sqrt((pred_c ** 2).mean(dim=2) + eps)       # [B, 1, C]
        stim_std = torch.sqrt((stim_c ** 2).mean(dim=2) + eps)       # [B, K, C]

        # Pearson-like correlation per feature channel.
        corr = cov / (pred_std * stim_std + eps)                     # [B, K, C]

        # Average over feature channels to get one candidate score.
        scores = corr.mean(dim=-1)                                   # [B, K]
        return scores

    def forward(self, eeg: torch.Tensor, stimuli: torch.Tensor) -> dict:
        """
        Forward pass for Type 2.

        Parameters
        ----------
        eeg : torch.Tensor
            EEG tensor with shape [B, T, C_eeg].
        stimuli : torch.Tensor
            Candidate stimuli tensor with shape [B, K, T, C_stim].

        Returns
        -------
        dict
            Dictionary containing:
            - "pred": EEG-derived predicted stimulus, [B, T, C_stim]
            - "candidates": encoded candidate stimuli, [B, K, T, C_stim]
            - "scores": correlation-style candidate scores, [B, K]

        Important
        ---------
        In the LightningModule, Type 2 uses the attended candidate as the
        reconstruction target for the loss. The returned scores are mainly
        useful for monitoring and final attended-vs-unattended decisions.
        """
        pred = self.predict_stimulus(eeg)          # [B, T, C_stim]
        candidates = self.encode_stimuli(stimuli)  # [B, K, T, C_stim]
        scores = self.score_candidates(pred, candidates)  # [B, K]

        return {
            "pred": pred,
            "candidates": candidates,
            "scores": scores,
        }
