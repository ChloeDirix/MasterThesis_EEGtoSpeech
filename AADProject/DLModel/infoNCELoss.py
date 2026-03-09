import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE with all negatives in the denominator

    Positive:
        - time-aligned attended embedding sa[i]

    Negatives:
        - all time-misaligned attended embeddings sa[j!=i] (in-batch)
        - unattended embedding su[i]
    """

    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.tau = float(temperature)
        self.normalize = normalize

    def forward(self, e: torch.Tensor, sa: torch.Tensor, su: torch.Tensor):
        """
        e  : EEG embeddings            [B, D]
        sa : attended audio embeddings [B, D]
        su : unattended audio emb      [B, D]
        """

        
        if e.ndim != 2 or sa.ndim != 2 or su.ndim != 2:
            raise ValueError("All inputs must be [B, D]")

        if not (e.shape == sa.shape == su.shape):
            raise ValueError("Shapes must match for e, sa, su")

        Batch_size = e.size(0)
        device = e.device       #check CPU/GPU the tensor is on so we can create targets on the same device

        if self.normalize:
            e = F.normalize(e, dim=-1)
            sa = F.normalize(sa, dim=-1)
            su = F.normalize(su, dim=-1)

  
        # 1) Similarity to all attended 
        sim_att = e @ sa.T                              # dot product over matrix of form: sim_att =[[e1·sa1, e1·sa2],[e2·sa1, e2·sa2]]

                                                        # [B,B]

        # 2) Similarity to unattended                  
        sim_unatt = torch.sum(e * su, dim=-1, keepdim=True)  # element wise multiplication: result: [[e1·su1][e2·su2]]
                                                             # then sum over D
                                                             # [B, 1]


        # 3) Combine                             
        logits = torch.cat([sim_att, sim_unatt], dim=1)     # concatenate 
                                                            # [B, B+1]
        targets = torch.arange(Batch_size, device=device)   # creates: tensor([0, 1, 2, B-1]) --> For row i, the correct class is column i.

        loss = F.cross_entropy(                     # apply softmax to the row, look at probability of the correct column and takes neg log 
            logits / self.tau,
            targets,
            reduction="mean",                       # take mean over a batch
        )

        return loss