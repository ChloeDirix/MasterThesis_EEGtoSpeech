from paths import paths
from DLModel.datasets import AADDataset
from DLModel.plot_checks import PlotCallback
from DLModel.LightningModule import AADLightningModule
from pytorch_lightning.loggers import CSVLogger
from DLModel.save_outputs import SaveValOutputsCallback

import argparse
import torch
import os
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler


# --------------------------
# ---- Helper functions ----
# --------------------------

def get_subject_list(cfg):
    """
    get the subjects from the cfg file and put them into dict
    """
    subj_cfg = cfg["subjects"]
    key = cfg.get("use_subjects", "all")  # "all" or "small"
    if isinstance(subj_cfg, dict):
        if key not in subj_cfg:
            raise KeyError(f"use_subjects='{key}' not found in cfg['subjects'] keys={list(subj_cfg.keys())}")
        return list(subj_cfg[key])
    return list(subj_cfg)


def _subject_to_paths(subjects):
    """
    Input: ["S1_DAS", "S12_DTU", ...]
    Output: list of NWB paths:
      Data_InputModel/EEG_PP/S1_DAS.nwb
      Data_InputModel/EEG_PP/S12_DTU.nwb
    """
    all_paths = []
    for s in subjects:
        try:
            subj_id, ds_key = s.rsplit("_", 1)
        except ValueError:
            raise ValueError(f"Subject '{s}' must look like '<SUBJ>_<DATASET>' e.g. S1_DAS")

        ds_key = ds_key.upper()
        if ds_key not in ("DAS", "DTU"):
            raise ValueError(f"Unknown dataset key '{ds_key}' in subject '{s}'")

        p = paths.subject_eegPP(subj_id, ds_key)
        all_paths.append(str(p) if isinstance(p, Path) else p)

    return all_paths

def _get_sample_subject_keys(ds):
    """
    Try common attribute names that AADDataset might expose for per-sample subject IDs.
    You need ONE of these on the dataset object (per item / per trial):
      - ds.sample_subject_keys
      - ds.sample_subject_ids
      - ds.sample_subjects
    """
    for attr in ("sample_subject_keys", "sample_subject_ids", "sample_subjects"):
        if hasattr(ds, attr):
            keys = list(getattr(ds, attr))
            return keys
    raise AttributeError(
        "AADDataset must expose per-sample subject IDs for subject-weighted sampling. "
        "Add e.g. `self.sample_subject_keys = [...]` (len == len(dataset))."
    )


def build_weighted_sampler_from_subject(ds):
    """
    Build an inverse frequency sampler to balance subjects.
    """
    keys = _get_sample_subject_keys(ds)

    unique = sorted(set(keys))
    counts = {k: 0 for k in unique}
    for k in keys:
        counts[k] += 1

    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)
    return sampler, counts


def build_weighted_sampler_combined_dataset_subject(ds):
    """
    Combined balancing: each sample gets weight = 1/(count_dataset[key] * count_subject[subj]).
    This prevents:
      - DAS dominating DTU
      - large subjects dominating small ones
    """
    if not hasattr(ds, "sample_dataset_keys"):
        raise AttributeError("AADDataset must expose dataset key to use weighted sampling")

    dkeys = list(ds.sample_dataset_keys)
    skeys = _get_sample_subject_keys(ds)

    # dataset counts
    d_unique = sorted(set(dkeys))
    d_counts = {k: 0 for k in d_unique}
    for k in dkeys:
        d_counts[k] += 1

    # subject counts
    s_unique = sorted(set(skeys))
    s_counts = {k: 0 for k in s_unique}
    for k in skeys:
        s_counts[k] += 1

    weights = torch.tensor(
        [1.0 / (d_counts[dk] * s_counts[sk]) for dk, sk in zip(dkeys, skeys)],
        dtype=torch.double
    )
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)

    return sampler, {"dataset": d_counts, "subject": s_counts}

def build_weighted_sampler_from_dataset(ds):
    """
    Build an inverse frequency sampler to balance datasets (DTU vs DAS).
    """
    if hasattr(ds, "sample_dataset_keys"):
        keys = list(ds.sample_dataset_keys)
    else:
        raise AttributeError("AADDataset must expose dataset key to use weighted sampling")

    unique = sorted(set(keys))
    counts = {k: 0 for k in unique}
    for k in keys:
        counts[k] += 1

    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)
    return sampler, counts


def get_fold_dir(cfg, results_dir: str, fold_name: str) -> str:
    """
    Single source of truth for fold directory.
    This MUST match where you save checkpoints/logs so that skip logic is consistent.
    """
    base = paths.result_file_DL(cfg, results_dir)
    return os.path.join(base, f"fold_{fold_name}")


# ------------------------
# ----- Dataloaders ------
# ------------------------

def build_dataloaders(cfg, dl_cfg, train_subjects, val_subjects):
    """
    Create train / val datasets using the dataset module
    """
    train_paths = _subject_to_paths(train_subjects)
    val_paths = _subject_to_paths(val_subjects)

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

    # Quick-test mode
    if dl_cfg["quick_test"]["enable"]:
        print("quick mode test enabled")
        num_epochs = dl_cfg["quick_test"]["max_epochs"]
        batch_size = 2
        train_ds = torch.utils.data.Subset(train_ds, range(min(32, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(32, len(val_ds))))
        sampler = None
    else:
        num_epochs = dl_cfg["train"]["num_epochs"]
        batch_size = dl_cfg["train"]["batch_size"]
        # sampler_mode can be: "none", "dataset", "subject", "both"
        sampler_mode = dl_cfg["train"].get("sampler_mode", "dataset").lower()

        if sampler_mode == "none":
            sampler = None

        elif sampler_mode == "dataset":
            sampler, counts = build_weighted_sampler_from_dataset(train_ds)
            print(f"Train sample counts by dataset: {counts}")

        elif sampler_mode == "subject":
            sampler, counts = build_weighted_sampler_from_subject(train_ds)
            print(f"Train sample counts by subject: {counts}")

        elif sampler_mode in ("both", "combined"):
            sampler, counts = build_weighted_sampler_combined_dataset_subject(train_ds)
            print(f"Train sample counts by dataset: {counts['dataset']}")
            print(f"Train sample counts by subject: {counts['subject']}")
        else:
            raise ValueError(f"Unknown sampler_mode='{sampler_mode}'. Use none/dataset/subject/both.")

    num_workers = dl_cfg["train"].get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    eeg_sample, stim_sample, _ = train_ds[0]
    C_eeg = eeg_sample.shape[-1]
    C_stim = stim_sample.shape[-1]
    K = 2

    return train_loader, val_loader, (C_eeg, C_stim, K), num_epochs


# ---------------------------------
# --------- Train phase -----------
# ---------------------------------

def train_single_fold(cfg, dl_cfg, train_subjects, val_subjects, results_dir, fold_name="single"):
    train_loader, val_loader, input_dims, num_epochs = build_dataloaders(
        cfg=cfg,
        dl_cfg=dl_cfg,
        train_subjects=train_subjects,
        val_subjects=val_subjects
    )

    module = AADLightningModule(cfg=cfg, input_dims=input_dims)

    # Results directory (per fold)
    fold_dir = get_fold_dir(cfg, results_dir, fold_name)
    print("Fold dir:", fold_dir)
    os.makedirs(fold_dir, exist_ok=True)

    logger = CSVLogger(save_dir=fold_dir, name="csv_logs")

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

    save_outputs_cb = SaveValOutputsCallback(
        save_dir=os.path.join(fold_dir, "posthoc"),
        save_embeddings=True,
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        precision=32,
        logger=logger,
        callbacks=[checkpoint, earlystop, plot_cb, save_outputs_cb],
        log_every_n_steps=5,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, train_loader, val_loader)

    best_val = checkpoint.best_model_score
    best_val = float(best_val.item()) if best_val is not None else float("nan")
    print(f"[Fold {fold_name}] best val_acc = {best_val:.4f}")
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-subj", type=str, default=None,
                        help="Run a single LOSO fold with this subject as validation/test.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="output directory")
    args = parser.parse_args()

    cfg = paths.load_config()
    dl_cfg = cfg["DeepLearning"]

    cfg_splits = cfg["splits"]
    subjects_cfg = cfg["subjects"]
    use_subjects = cfg.get("use_subjects", "all")
    subjects = subjects_cfg[use_subjects]

    # -- run 1 fold only (for SLURM array)
    if args.test_subj is not None:
        test_subj = args.test_subj
        train_subjs = [s for s in subjects if s != test_subj]
        print(f"Array/Single fold: test={test_subj}, train={train_subjs}")

        train_single_fold(
            cfg,
            dl_cfg,
            train_subjects=train_subjs,
            val_subjects=[test_subj],
            results_dir=args.results_dir,
            fold_name=test_subj,
        )
        return

    # -- run all subjects in loop
    if cfg_splits["mode"] == "loso":
        print(f"Running LOSO over subjects: {subjects}")

        fold_scores = []
        for test_subj in subjects:
            fold_dir = get_fold_dir(cfg, args.results_dir, test_subj)
            best_ckpt = os.path.join(fold_dir, "best.ckpt")

            if os.path.exists(best_ckpt):
                print(f"Skipping {test_subj} (already completed: {best_ckpt})")
                continue

            train_subjs = [s for s in subjects if s != test_subj]
            print(f"\n=== LOSO Fold: test={test_subj}, train={train_subjs} ===")

            best_val = train_single_fold(
                cfg,
                dl_cfg,
                train_subjects=train_subjs,
                val_subjects=[test_subj],
                results_dir=args.results_dir,
                fold_name=test_subj,
            )
            fold_scores.append((test_subj, best_val))

        print("\n=== LOSO summary ===")
        for subj, score in fold_scores:
            print(f"Subject {subj}: best val_acc = {score:.4f}")

        scores = [s for (_, s) in fold_scores if not np.isnan(s)]
        if len(scores) > 0:
            print(f"\nMean val_acc over folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        else:
            print("No valid scores recorded.")

    elif cfg_splits["mode"] == "single":
        val_subjs = cfg_splits["single"]["val"]
        train_subjs = [s for s in cfg_splits["single"]["train"] if s not in val_subjs]

        print(f"Single split: train={train_subjs}, val={val_subjs}")

        train_single_fold(
            cfg,
            dl_cfg,
            train_subjects=train_subjs,
            val_subjects=val_subjs,
            results_dir=args.results_dir,
            fold_name=val_subjs[0],
        )


if __name__ == "__main__":
    main()