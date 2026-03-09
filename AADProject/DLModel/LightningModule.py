import torch
import pytorch_lightning as pl
import torch.nn as nn
from DLModel.LogisticRankLoss import LogisticRankLoss
from DLModel.TripletLoss import TripletLoss
from DLModel.infoNCELoss import InfoNCELoss
from DLModel.models.contrastive_model import ContrastiveAADModel


class AADLightningModule(pl.LightningModule):
    """
    wrapper around model that organizes training
    -- define model architecture
    -- define loss function
    -- define what happens in training and validation step
    -- optimizer
    """

    def __init__(self, cfg, input_dims):
        super().__init__()
        C_eeg, C_stim, K = input_dims
        self.dl_cfg = cfg["DeepLearning"]

        # architecture
        self.model = ContrastiveAADModel(
            self.dl_cfg,
            eeg_input_dim=C_eeg,
            stim_input_dim=C_stim        
        )

        # Loss selction
        loss_cfg = self.dl_cfg.get("loss", {})
        self.loss_name = str(loss_cfg.get("name", "logistic_rank")).lower()
        margin = float(loss_cfg.get("margin", 0.2))

        if self.loss_name in ("logistic_rank", "rank"):
            # logistic rank acts on logits (similarities)
            self.criterion = LogisticRankLoss(margin=float(loss_cfg.get("margin", 0.0)))

        elif self.loss_name in ("triplet", "triplet_cosine"):
            self.criterion = TripletLoss(margin=margin)

        elif self.loss_name in ("infoNCE","infonce"):
            tau = float(loss_cfg.get("temperature", 0.07))
            norm = bool(loss_cfg.get("normalize", True))
            self.criterion = InfoNCELoss(temperature=tau, normalize=norm)

        else:
            raise ValueError(
                f"Unknown DeepLearning.loss.name='{self.loss_name}'. "
                "Use: logistic_rank | triplet | infoNCE"
            )



    def forward(self, eeg, stim):  
        return self.model(eeg, stim)   # activate model
    
    def _compute_loss_and_logits(self, eeg, stim, att):
        # For accuracy, we can always use the model logits (scaled cosine sims)
        logits = self(eeg, stim)  # [B,2]

        if self.loss_name in ("logistic_rank", "rank"):
            loss = self.criterion(logits, att)

        elif self.loss_name in ("triplet", "tripletLoss"):
            z_eeg, z_stim = self.model.get_pair_embeddings(eeg, stim)  # [B,D], [B,2,D]
            loss = self.criterion(z_eeg, z_stim, att)
        elif self.loss_name in ("infoNCE", "infonce"):
            z_eeg, z_stim = self.model.get_pair_embeddings(eeg, stim)  # [B,D], [B,2,D]
            
            # make sure att is [B] long
            if att.ndim != 1:
                att = att.view(-1)
            att = att.long()

            # split z_stim into attended (sa) and unattended (su)
            b = torch.arange(z_eeg.size(0), device=z_eeg.device)
            sa = z_stim[b, att]          # [B,D]
            su = z_stim[b, 1 - att]      # [B,D]

            # now your InfoNCELoss works unchanged
            loss = self.criterion(z_eeg, sa, su)

        else:
            raise RuntimeError("Unhandled loss_name")

        return loss, logits

    def training_step(self, batch, batch_idx):
        eeg, stim, att = batch
        loss, logits = self._compute_loss_and_logits(eeg, stim, att)
        acc = (logits.argmax(dim=1) == att).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
            # acc is not return but are logged and used for checkpointing, early stopping, etc.
            # acc is for monitoring, not for training

    def on_validation_epoch_end(self):
        if not hasattr(self, "_val_preds"):
            return
        preds = torch.cat(self._val_preds)
        atts  = torch.cat(self._val_atts)

        p_mean = preds.float().mean().item()
        a_mean = atts.float().mean().item()

        self.log("val_att_mean", a_mean, prog_bar=True)
        self.log("val_pred_mean", p_mean, prog_bar=True)

        # reset
        self._val_preds.clear()
        self._val_atts.clear()

    def validation_step(self, batch, batch_idx):
        eeg, stim, att = batch
        loss, logits = self._compute_loss_and_logits(eeg, stim, att)
        acc = (logits.argmax(dim=1) == att).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # -- define optimizer = algorithm that updates the model's weights based on the loss function
    # ADAMw= adaptive moment estimation with weight decay --> L2 regularization
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.dl_cfg["train"]["learning_rate"]),
            weight_decay=float(self.dl_cfg["train"]["weight_decay"])
        )

