import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRankLoss(nn.Module):
    """
    loss = softplus(-(pos - neg)) = log(1 + exp(-(pos-neg)))
    --> encourages logits[b, att] > logits[b, 1-att].
    (For K=2 logits [B,2] and target att in {0,1})
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = float(margin)  # not used right now

    def forward(self, logits: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        # logits: [B,2], att: [B]
        # att says: att[i] = 0  → stimulus 0 is attended (positive)
        #           att[i] = 1  → stimulus 1 is attended (positive)
        
        if logits.ndim != 2 or logits.size(1) != 2:
            raise ValueError(f"Expected logits [B,2], got {tuple(logits.shape)}")
        if att.ndim != 1:
            att = att.view(-1)

        b = torch.arange(logits.size(0), device=logits.device)
        pos = logits[b, att]                  
        neg = logits[b, 1 - att]              
        diff = pos - neg                      
        
        return F.softplus(-(diff - self.margin)).mean()