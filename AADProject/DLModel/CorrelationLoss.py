import torch
import torch.nn as nn


class CorrelationLoss(nn.Module):
    """
    Negative Pearson-correlation loss over TIME, averaged over stimulus channels/bands.

    pred:   [B, T, C]
    target: [B, T, C]

    Returns:
        loss = 1 - mean_correlation
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have same shape, got {pred.shape} vs {target.shape}"
            )
        if pred.ndim != 3:
            raise ValueError(f"Expected pred/target shape [B,T,C], got {pred.shape}")

        # center over time
        pred_c = pred - pred.mean(dim=1, keepdim=True)         # [B,T,C]
        target_c = target - target.mean(dim=1, keepdim=True)   # [B,T,C]

        # covariance over time
        cov = (pred_c * target_c).mean(dim=1)                  # [B,C]

        # std over time
        pred_std = torch.sqrt((pred_c ** 2).mean(dim=1) + self.eps)        # [B,C]
        target_std = torch.sqrt((target_c ** 2).mean(dim=1) + self.eps)    # [B,C]

        corr = cov / (pred_std * target_std + self.eps)        # [B,C]
        mean_corr = corr.mean()                                # scalar over batch and channels

        return 1.0 - mean_corr