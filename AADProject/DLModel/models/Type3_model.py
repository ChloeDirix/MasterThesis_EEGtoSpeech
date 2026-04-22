"""
Type 3 auditory attention decoding model.

This version keeps the original model logic, but its internal encoder
imports are now cleaner:
- EEG encoder lives under encoders/eeg/
- audio encoder lives under encoders/audio/

Conceptually
------------
Type 3 is a relatively simple ranking model:
1. EEG is projected into a stimulus-like representation.
2. Each candidate stimulus is optionally encoded.
3. A correlation-style similarity score is computed for each candidate.
4. The attended candidate should receive the higher score.

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

from DLModel.models.encoders.eeg.simple_eeg_encoder import SimpleEEGEncoder
from DLModel.models.encoders.audio.simple_audio_encoder import SimpleAudioEncoder


class CorrRankAADModel(nn.Module):
    """
    Type 3 correlation-ranking model.

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
    This class preserves the previous behavior of the original Type 3
    implementation. The refactor mainly improves structure and readability.
    """

    def __init__(self, dl_cfg, eeg_input_dim: int, stim_input_dim: int) -> None:
        super().__init__()
        self.dl_cfg = dl_cfg

        eeg_cfg = self.dl_cfg["model"]["eeg_encoder"]
        aud_cfg = self.dl_cfg["model"].get("audio_encoder", {})

        # EEG path:
        # Map EEG [B, T, C_eeg] directly into the stimulus comparison space.
        self.eeg_encoder = SimpleEEGEncoder(
            input_dim=eeg_input_dim,
            output_dim=stim_input_dim,
            conv_kernel=int(eeg_cfg.get("conv_kernel", 31)),
            dropout=float(eeg_cfg.get("dropout", 0.0)),
            use_conv=bool(eeg_cfg.get("use_conv", True)),
        )

        # Stimulus path:
        # Optionally transform the candidate stimuli before comparison.
        audio_out_dim = int(aud_cfg.get("out_dim", stim_input_dim))
        self.audio_encoder = SimpleAudioEncoder(
            input_dim=stim_input_dim,
            output_dim=audio_out_dim,
            dropout=float(aud_cfg.get("dropout", 0.0)),
            mode=str(aud_cfg.get("mode", "identity")),
            conv_kernel=int(aud_cfg.get("conv_kernel", 31)),
        )

        # This specific Type 3 scoring path expects EEG prediction and encoded
        # stimulus candidates to have matching feature dimensions.
        if audio_out_dim != stim_input_dim:
            raise ValueError(
                "CorrRankAADModel requires audio encoder output_dim == stim_input_dim."
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

        # Flatten the candidate dimension into the batch dimension so we can
        # run all candidates through the same audio encoder in one call.
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
            Candidate logits with shape [B, K].

        Explanation
        -----------
        1. Center both the predicted stimulus and each candidate.
        2. Compute covariance over time.
        3. Compute per-feature standard deviations.
        4. Convert covariance to correlation.
        5. Average over the feature dimension to get one score per candidate.
        """
        # Center the EEG-derived prediction over time.
        pred_c = pred_stim - pred_stim.mean(dim=1, keepdim=True)   # [B, T, C]

        # Center the candidate stimuli over time.
        stim_c = stimuli - stimuli.mean(dim=2, keepdim=True)       # [B, K, T, C]

        # Add candidate dimension so broadcasting works during elementwise
        # multiplication with [B, K, T, C].
        pred_c = pred_c.unsqueeze(1)                               # [B, 1, T, C]

        # Covariance over the time dimension.
        cov = (pred_c * stim_c).mean(dim=2)                        # [B, K, C]

        # Standard deviations over the time dimension.
        pred_std = torch.sqrt((pred_c ** 2).mean(dim=2) + eps)     # [B, 1, C]
        stim_std = torch.sqrt((stim_c ** 2).mean(dim=2) + eps)     # [B, K, C]

        # Pearson-like correlation per feature channel.
        corr = cov / (pred_std * stim_std + eps)                   # [B, K, C]

        # Average over feature channels to obtain one score per candidate.
        logits = corr.mean(dim=-1)                                 # [B, K]
        return logits

    def forward(self, eeg: torch.Tensor, stimuli: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Type 3.

        Parameters
        ----------
        eeg : torch.Tensor
            EEG tensor with shape [B, T, C_eeg].
        stimuli : torch.Tensor
            Candidate stimuli tensor with shape [B, K, T, C_stim].

        Returns
        -------
        torch.Tensor
            Candidate logits with shape [B, K].
        """
        pred = self.predict_stimulus(eeg)          # [B, T, C_stim]
        candidates = self.encode_stimuli(stimuli)  # [B, K, T, C_stim]
        logits = self.score_candidates(pred, candidates)
        return logits
