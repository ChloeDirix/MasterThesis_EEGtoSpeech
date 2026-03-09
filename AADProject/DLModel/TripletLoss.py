import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    anchor: EEG embedding z_eeg       [B, D]
    pos:    attended stim embedding   [B, D]
    neg:    unattended stim embedding [B, D]

    distance = 1 - cosine_similarity
    loss = max(0, d(anchor,pos) - d(anchor,neg) + margin)
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1),
            margin=self.margin,
            reduction="mean",
        )

    def forward(self, z_eeg: torch.Tensor, z_stim: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        """
        z_eeg:  [B, D]
        z_stim: [B, K, D]  (K=2)
        att:    [B] (0/1)
        """
        z_eeg = F.normalize(z_eeg, dim=-1)
        z_stim = F.normalize(z_stim, dim=-1)
        
        # dim checks
        if z_eeg.ndim != 2:
            raise ValueError(f"Expected z_eeg [B,D], got {tuple(z_eeg.shape)}")
        if z_stim.ndim != 3 or z_stim.size(1) != 2:
            raise ValueError(f"Expected z_stim [B,2,D], got {tuple(z_stim.shape)}")
        if att.ndim != 1:
            att = att.view(-1)

        b = torch.arange(z_eeg.size(0), device=z_eeg.device)       # b = [0,1,2,...,B-1]
                                                                   # att=[1,0,1,0...]
        pos = z_stim[b, att]         #   --> pos[0]=:z_stim[0,1], pos[2]=z_stim[2,1]   == select attended
        neg = z_stim[b, 1 - att]     #   ===select unattended
        return self.loss_fn(z_eeg, pos, neg)