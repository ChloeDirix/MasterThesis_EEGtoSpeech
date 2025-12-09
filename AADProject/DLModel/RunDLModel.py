import os
import numpy as np
import pytorch_lightning as pl


from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader

from paths import paths
from DLModel.datasets import AADDataset
from DLModel.plot_checks import PlotCallback
from DLModel.LightningModule import AADLightningModule



def build_dataloaders(cfg, dl_cfg, train_subjects, val_subjects):
    """Create train / val datasets & loaders for a given subject split."""
    train_paths = paths.subject_eegPP_list(train_subjects)
    val_paths = paths.subject_eegPP_list(val_subjects)

    # Datasets
    train_ds = AADDataset(
        nwb_paths=train_paths,
        cfg=cfg,
        multiband=True,
        split="train",
    )
    val_ds = AADDataset(
        nwb_paths=val_paths,
        cfg=cfg,
        multiband=True,
        split="val",
    )

    # Quick-test mode: shrink dataset + epochs
    cfg_quicktest=dl_cfg["quick_test"]
    if cfg_quicktest["enable"]:
        print("quick mode test enabled")
        num_epochs = cfg_quicktest["max_epochs"]
        batch_size = 2
        train_ds = torch.utils.data.Subset(train_ds, range(min(32, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(32, len(val_ds))))
    else:
        num_epochs = dl_cfg["train"]["num_epochs"]
        batch_size = dl_cfg["train"]["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Input dims
    eeg_sample, stim_sample, _ = train_ds[0]
    C_eeg = eeg_sample.shape[-1]
    C_stim = stim_sample.shape[-1]
    K = 2

    return train_loader, val_loader, (C_eeg, C_stim, K), num_epochs

def train_single_fold(cfg, dl_cfg, train_subjects, val_subjects, fold_name="single"):

    # Dataloaders
    train_loader, val_loader, input_dims, num_epochs = build_dataloaders(
        cfg=cfg,
        dl_cfg=dl_cfg,
        train_subjects=train_subjects,
        val_subjects=val_subjects
    )

    # Lightning module
    module = AADLightningModule(cfg=cfg, input_dims=input_dims)

    # Results directory (per fold)
    os.makedirs(paths.RESULTS_DL, exist_ok=True)
    fold_dir = os.path.join(paths.result_file_DL(cfg), f"fold_{fold_name}_{len(train_subjects)}")
    print(fold_dir)
    os.makedirs(fold_dir, exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best",
        dirpath=fold_dir,
        save_top_k=1,
    )

    earlystop = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=dl_cfg["train"]["patience"],
    )

    plot_cb = PlotCallback(save_dir=os.path.join(fold_dir, "plots"))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,                      # max nr of epochs for training, unless earlyStopping
        accelerator="auto",                         # let lighting automatically chose device
        devices="auto",                             # use all available GPUs or 1 CPU
        precision=32,                               # numperic precision
        callbacks=[checkpoint, earlystop, plot_cb], # activate callbacks at correct moment
        log_every_n_steps=5,                        # nr of batches between logs printed
        num_sanity_val_steps=0,                     # Run validation on n batches before training to make sure validation loop works
    )

    # Train
    trainer.fit(module, train_loader, val_loader)

    # Best validation accuracy for this fold
    best_val = checkpoint.best_model_score
    best_val = float(best_val.item()) if best_val is not None else float("nan")
    print(f"[Fold {fold_name}] best val_acc = {best_val:.4f}")
    return best_val



def main():

    # -----------------------------------
    # Load config
    # -----------------------------------
    from paths import paths

    cfg = paths.load_config()
    dl_cfg = cfg["DeepLearning"]

    cfg_splits=cfg["splits"]
    if cfg_splits["mode"]=="loso":
        # --------------------------
        # LOSO: loop over subjects
        # --------------------------
        subjects = cfg["subjects"]
        print(f"Running LOSO over subjects: {subjects}")

        fold_scores = []
        for test_subj in subjects:

            fold_dir = os.path.join(paths.RESULTS_DL, f"fold_{test_subj}")
            best_ckpt = os.path.join(fold_dir, "best.ckpt")
            # --- skip if fold already trained ---
            if os.path.exists(best_ckpt):
                print(f"Skipping {test_subj} (already completed)")
                continue

            train_subjs = [s for s in subjects if s != test_subj]
            print(f"\n=== LOSO Fold: test={test_subj}, train={train_subjs} ===")

            best_val = train_single_fold(
                cfg,
                dl_cfg,
                train_subjects=train_subjs,
                val_subjects=[test_subj],
                fold_name=test_subj,
            )
            fold_scores.append((test_subj, best_val))

        # Summary
        print("\n=== LOSO summary ===")
        for subj, score in fold_scores:
            print(f"Subject {subj}: best val_acc = {score:.4f}")

        scores = [s for (_, s) in fold_scores if not np.isnan(s)]
        if len(scores) > 0:
            print(f"\nMean val_acc over folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        else:
            print("No valid scores recorded.")

    elif cfg_splits["mode"]=="single":
        # --------------------------
        # Single train/val split (original behavior)
        # --------------------------

        val_subjs = cfg_splits["single"]["val"]
        train_subjs =[s for s in cfg_splits["single"]["train"] if s not in val_subjs]

        print(f"Single split: train={train_subjs}, val={val_subjs}")

        train_single_fold(
            cfg,
            dl_cfg,
            train_subjects=train_subjs,
            val_subjects=val_subjs,
            fold_name=val_subjs[0],
        )

if __name__ == "__main__":
    main()

