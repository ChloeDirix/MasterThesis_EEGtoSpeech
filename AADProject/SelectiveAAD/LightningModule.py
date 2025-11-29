

import torch
import torch.nn as nn
import pytorch_lightning as pl

from SelectiveAAD.models.contrastive_model import ContrastiveAADModel
#from SelectiveAAD.models.contrastive_cnn_model import ContrastiveAADModel_CNN


class AADLightningModule(pl.LightningModule):
    def __init__(self, cfg, input_dims):
        super().__init__()

        self.cfg = cfg
        C_eeg, C_stim, K = input_dims

        self.dl_cfg = cfg["DeepLearning"]
        d_model=self.dl_cfg["eeg_encoder"]["d_model"]
        temperature=self.dl_cfg["contrastive"]["temperature"]


        self.model = ContrastiveAADModel(
            eeg_input_dim=C_eeg,
            stim_input_dim=C_stim,
            d_model=d_model,
            temperature=temperature,
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, eeg, stim):
        return self.model(eeg, stim)

    def training_step(self, batch, batch_idx):

        eeg, stim, att = batch
        logits = self(eeg, stim)
        loss = self.criterion(logits, att)
        acc = (logits.argmax(dim=1) == att).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss
            # àacc is not return but are logged and used for checkpointing, early stopping, etc.
            # acc is for monitoring, NOT for training

    def validation_step(self, batch, batch_idx):
        eeg, stim, att = batch
        logits = self(eeg, stim)
        loss = self.criterion(logits, att)
        acc = (logits.argmax(dim=1) == att).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.dl_cfg["learning_rate"]),
            weight_decay=float(self.dl_cfg.get("weight_decay", 0.0))
        )

