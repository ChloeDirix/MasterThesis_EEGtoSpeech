import argparse
import os
import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from paths import paths
from DLModel.datasets import build_dataloaders
from DLModel.plot_checks import PlotCallback
from DLModel.LightningModule import AADLightningModule
from DLModel.checkpoint_utils import save_training_checkpoint


# =============================================================================
# Fold/result helpers
# =============================================================================
def get_fold_dir(cfg, results_dir: str, fold_name: str) -> str:
    """
    Single source of truth for fold directory.
    This must match where checkpoints/logs are saved so skip logic is consistent.
    """
    return os.path.join(results_dir, "folds", f"fold_{fold_name}")


# =============================================================================
# Final clean evaluation
# =============================================================================
def evaluate_best_checkpoint_on_val(module, checkpoint_path, val_loader, save_dir):
    """
    Reload best checkpoint into the current LightningModule, run one clean pass
    on the real validation loader only, and save window- and trial-level outputs.

    Saves:
      final_eval_best.npz with:
        preds_window, labels_window, trial_uids,
        preds_trial, labels_trial,
        acc_window, acc_trial
    """
    os.makedirs(save_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    module.load_state_dict(state_dict, strict=True)
    module.eval()

    device = module.device
    all_scores = []
    all_preds = []
    all_labels = []
    all_trial_uids = []

    with torch.no_grad():
        for batch in val_loader:
            eeg, stim, att, meta = batch

            eeg = eeg.to(device)
            stim = stim.to(device)
            att = att.long().view(-1).to(device)

            _, scores = module._compute_loss_and_scores(eeg, stim, att)
            preds = torch.argmax(scores, dim=1)

            all_scores.append(scores.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_labels.append(att.detach().cpu())

            batch_trial_uids = meta["trial_uid"]
            all_trial_uids.extend([str(x) for x in batch_trial_uids])

    if len(all_scores) == 0:
        raise RuntimeError("No batches found in val_loader during final evaluation.")

    scores_window = torch.cat(all_scores, dim=0)     # [Nw, 2]
    preds_window = torch.cat(all_preds, dim=0)       # [Nw]
    labels_window = torch.cat(all_labels, dim=0)     # [Nw]

    acc_window = float((preds_window == labels_window).float().mean().item())

    # aggregate to trial level
    trial_score_sums = {}
    trial_counts = {}
    trial_targets = {}

    for i in range(scores_window.size(0)):
        tid = all_trial_uids[i]
        sc = scores_window[i]
        y = int(labels_window[i].item())

        if tid not in trial_score_sums:
            trial_score_sums[tid] = sc.clone()
            trial_counts[tid] = 1
            trial_targets[tid] = y
        else:
            trial_score_sums[tid] += sc
            trial_counts[tid] += 1

    preds_trial = []
    labels_trial = []
    trial_uid_unique = []

    for tid in trial_score_sums.keys():
        mean_scores = trial_score_sums[tid] / float(trial_counts[tid])
        pred_t = int(torch.argmax(mean_scores).item())
        y = int(trial_targets[tid])

        preds_trial.append(pred_t)
        labels_trial.append(y)
        trial_uid_unique.append(tid)

    preds_trial = np.asarray(preds_trial, dtype=np.int64)
    labels_trial = np.asarray(labels_trial, dtype=np.int64)
    acc_trial = float((preds_trial == labels_trial).mean()) if len(labels_trial) > 0 else float("nan")

    np.savez(
        os.path.join(save_dir, "final_eval_best.npz"),
        preds_window=preds_window.numpy(),
        labels_window=labels_window.numpy(),
        trial_uids=np.asarray(all_trial_uids, dtype=object),
        preds_trial=preds_trial,
        labels_trial=labels_trial,
        trial_uids_unique=np.asarray(trial_uid_unique, dtype=object),
        acc_window=np.asarray(acc_window, dtype=np.float32),
        acc_trial=np.asarray(acc_trial, dtype=np.float32),
    )

    print(f"[Final eval] acc_window={acc_window:.4f}, acc_trial={acc_trial:.4f}")
    print(f"[Final eval] Saved to: {os.path.join(save_dir, 'final_eval_best.npz')}")

    return {
        "acc_window": acc_window,
        "acc_trial": acc_trial,
    }


# =============================================================================
# Training
# =============================================================================
def train_single_fold(cfg, dl_cfg, train_subjects, val_subjects, results_dir, fold_name="single"):
    train_loader, train_eval_loader, val_loader, input_dims, num_epochs, normalization_bundle = build_dataloaders(
        cfg, dl_cfg, train_subjects, val_subjects
    )

    module = AADLightningModule(cfg=cfg, input_dims=input_dims)

    fold_dir = get_fold_dir(cfg, results_dir, fold_name)
    print("Fold dir:", fold_dir)
    os.makedirs(fold_dir, exist_ok=True)

    logger = CSVLogger(save_dir=fold_dir, name="csv_logs")

    checkpoint = ModelCheckpoint(
        monitor="val_acc_window",
        mode="max",
        filename="best",
        dirpath=fold_dir,
        save_top_k=1,
    )

    earlystop = EarlyStopping(
        monitor="val_acc_window",
        mode="max",
        patience=int(dl_cfg["train"]["patience"]),
    )

    subject_label = None
    dataset_label = None
    if len(val_subjects) == 1:
        subject_label = str(val_subjects[0])
        if "_" in subject_label:
            dataset_label = subject_label.split("_")[-1].upper()

    window_len_s = float(cfg["DeepLearning"]["data_windows"]["val"]["window_len_s"])

    plot_cb = PlotCallback(
        plot_dir=os.path.join(fold_dir, "plots"),
        history_dir=os.path.join(fold_dir, "posthoc"),
        zoom_n_epochs=5.0,
        subject_label=subject_label,
        dataset_label=dataset_label,
        window_len_s=window_len_s,
        debug_keys=False,
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        val_check_interval=0.2,
        accelerator="auto",
        devices="auto",
        precision=32,
        logger=logger,
        callbacks=[checkpoint, earlystop, plot_cb],
        log_every_n_steps=5,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=[train_eval_loader, val_loader],
    )

    ckpt_path = os.path.join(fold_dir, "reusable_checkpoint.pt")
    best_model_path = checkpoint.best_model_path

    save_training_checkpoint(
        path=ckpt_path,
        model=module.model,
        cfg=cfg,
        input_dims=input_dims,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        eeg_std=normalization_bundle.get("eeg_std"),
        envL_std=normalization_bundle.get("envL_std"),
        envR_std=normalization_bundle.get("envR_std"),
        subject_stds=normalization_bundle.get("subject_stds"),
        dataset_stds=normalization_bundle.get("dataset_stds"),
    )
    print(f"Saved reusable checkpoint to: {ckpt_path}")

    best_val = checkpoint.best_model_score
    best_val = float(best_val.item()) if best_val is not None else float("nan")
    print(f"[Fold {fold_name}] best val_acc_window = {best_val:.4f}")
    print(f"[Fold {fold_name}] best_model_path = {best_model_path}")

    if best_model_path and os.path.exists(best_model_path):
        evaluate_best_checkpoint_on_val(
            module=module,
            checkpoint_path=best_model_path,
            val_loader=val_loader,
            save_dir=os.path.join(fold_dir, "posthoc"),
        )
    else:
        raise FileNotFoundError(f"No best checkpoint found for fold {fold_name}: {best_model_path}")

    return best_val

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-subj",
        type=str,
        default=None,
        help="Run a single LOSO fold with this subject as validation/test.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    args = parser.parse_args()

    cfg = paths.load_config()
    dl_cfg = cfg["DeepLearning"]

    # -------------------------------
    # Tiny overfit sanity test
    # -------------------------------
    overfit_cfg = dl_cfg["overfit_tiny"]
    if overfit_cfg["enable"]:
        subject = overfit_cfg["subject"]
        print(f"Running tiny overfit sanity test on subject: {subject}")

        train_single_fold(
            cfg=cfg,
            dl_cfg=dl_cfg,
            train_subjects=[subject],
            val_subjects=[subject],
            results_dir=args.results_dir,
            fold_name=f"overfit_{subject}",
        )
        return

    cfg_splits = cfg["splits"]
    subjects_cfg = cfg["subjects"]
    use_subjects = cfg["use_subjects"]
    subjects = subjects_cfg[use_subjects]

    # -------------------------------
    # Run one fold only
    # -------------------------------
    if args.test_subj is not None:
        test_subj = args.test_subj
        train_subjs = [s for s in subjects if s != test_subj]
        print(f"Array/Single fold: test={test_subj}, train={train_subjs}")

        train_single_fold(
            cfg=cfg,
            dl_cfg=dl_cfg,
            train_subjects=train_subjs,
            val_subjects=[test_subj],
            results_dir=args.results_dir,
            fold_name=test_subj,
        )
        return

    # -------------------------------
    # Run all folds
    # -------------------------------
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
                cfg=cfg,
                dl_cfg=dl_cfg,
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
            cfg=cfg,
            dl_cfg=dl_cfg,
            train_subjects=train_subjs,
            val_subjects=val_subjs,
            results_dir=args.results_dir,
            fold_name=val_subjs[0],
        )

    else:
        raise ValueError(f"Unknown split mode: {cfg_splits['mode']}")


if __name__ == "__main__":
    main()