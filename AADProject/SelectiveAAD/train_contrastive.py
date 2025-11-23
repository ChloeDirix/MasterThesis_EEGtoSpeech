"""
PyTorch training loop

1. Parse config.
2. Create train / val SelectiveAADDataset → DataLoaders.
3. Initialize model, optimizer, scheduler.
4. Loop epochs:
    For each batch:
        Move tensors to GPU.
        logits = model(eeg, stimuli)
        loss = criterion(logits, attended_index)
        loss.backward(), optimizer.step().

    Periodically compute validation accuracy:
        pred = logits.argmax(dim=1)
        compare to attended_index.

5. Save:
    Best model weights (by val accuracy).
    Training curves.

"""


#important design choices:
# - how to split train/val/test
# usually subject-wise splits --> LOSO
# suggestion:
#   Config contains list of subjects per split.
#   Dataset builds index only from NWB files of the relevant subjects.

import os

from SelectiveAAD.DatasetsB.precomputed_dataset import PrecomputedAADDataset
from paths import paths
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from datasets import SelectiveAADDataset
#from SelectiveAAD.models.contrastive_model import ContrastiveAADModel
from SelectiveAAD.models.contrastive_cnn_model import ContrastiveAADModel_CNN

class AADLightningModule(pl.LightningModule):
    def __init__(self, cfg, input_dims):
        super().__init__()
        self.cfg = cfg

        C_eeg, C_stim, K = input_dims

        self.model = ContrastiveAADModel_CNN(
            eeg_input_dim=C_eeg,
            stim_input_dim=C_stim,
            d_model=32,
            temperature=0.1
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, eeg, stim):
        return self.model(eeg, stim)

    def training_step(self, batch, batch_idx):
        eeg, stim, att = batch
        logits = self(eeg, stim)
        loss = self.criterion(logits, att)
        acc = (logits.argmax(dim=1) == att).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eeg, stim, att = batch
        logits = self(eeg, stim)
        loss = self.criterion(logits, att)
        acc = (logits.argmax(dim=1) == att).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        dl_cfg = self.cfg["DeepLearning"]
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(dl_cfg["learning_rate"]),
            weight_decay=float(dl_cfg.get("weight_decay", 0.0))
        )

def main():

    # -----------------------------------
    # Load config
    # -----------------------------------

    from paths import paths

    cfg = paths.load_config()
    dl_cfg = cfg["DeepLearning"]
    train_paths = paths.subject_eegPP_list(cfg["train_subjects"])
    val_paths = paths.subject_eegPP_list(cfg["val_subjects"])

    # -----------------------------------
    # Generate datasets (datasets.py)
    # -----------------------------------
    train_ds = SelectiveAADDataset(
        nwb_paths=train_paths,
        cfg=cfg,
        multiband=True,
        split="train"
    )
    val_ds = SelectiveAADDataset(
        nwb_paths=val_paths,
        cfg=cfg,
        multiband=True,
        split="val"
    )

    # train_ds = PrecomputedAADDataset("DatasetsB/train_windows.pt")
    # val_ds = PrecomputedAADDataset("DatasetsB/val_windows.pt")
    #
    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Input dims
    eeg_sample, stim_sample, _ = train_ds[0]
    C_eeg = eeg_sample.shape[-1]
    C_stim = stim_sample.shape[-1]
    K = 2

    # ----------------------------
    # Lightning module
    # ----------------------------
    module = AADLightningModule(cfg, (C_eeg, C_stim, K))

    # ----------------------------
    # Callbacks
    # ----------------------------
    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best_model",
        save_top_k=1,
    )

    earlystop = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=dl_cfg["patience"]
    )

    # ----------------------------
    # Lightning trainer
    # ----------------------------
    trainer = pl.Trainer(
        max_epochs=dl_cfg["num_epochs"],
        accelerator="auto",
        devices="auto",
        precision=32,
        callbacks=[checkpoint, earlystop],
        log_every_n_steps=5
    )

    # Run training
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
