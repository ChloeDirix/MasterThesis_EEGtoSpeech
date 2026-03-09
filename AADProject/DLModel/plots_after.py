# DLModel/plots_after.py
import os
import glob
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Patch


# -----------------------------
# Coloring (requested)
# -----------------------------
DATASET_COLORS = {
    "DAS": "lightblue",
    "DTU": "lightgreen",
}

def dataset_color(ds: str) -> str:
    return DATASET_COLORS.get(str(ds).upper(), "lightgray")


# -----------------------------
# Parsing / sorting helpers
# -----------------------------
def fold_to_subject(fold_name: str) -> str:
    # fold_S10_DAS_33 -> S10_DAS
    s = str(fold_name)
    if s.startswith("fold_"):
        s = s[len("fold_"):]
    parts = s.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])  # S10_DAS
    return s

def subject_to_dataset(subject: str) -> str:
    # S10_DAS -> DAS
    subject = str(subject)
    if "_" in subject:
        return subject.split("_")[-1].upper()
    return ""

def parse_subject_num(s: str) -> int:
    # Works for "S10_DAS", "fold_S10_DAS_33", etc.
    m = re.search(r"S(\d+)", str(s))
    return int(m.group(1)) if m else 10**9

def dataset_rank(ds: str) -> int:
    ds = str(ds).upper()
    return {"DAS": 0, "DTU": 1}.get(ds, 99)

def natural_subject_sort_cols(dfp: pd.DataFrame) -> pd.DataFrame:
    """
    Adds standardized columns:
      subject, dataset, subj_num, ds_rank
    """
    dfp = dfp.copy()
    dfp["subject"] = dfp["fold"].astype(str).apply(fold_to_subject)
    dfp["dataset"] = dfp["subject"].astype(str).apply(subject_to_dataset)
    dfp["subj_num"] = dfp["subject"].astype(str).apply(parse_subject_num)
    dfp["ds_rank"] = dfp["dataset"].astype(str).apply(dataset_rank)
    return dfp


# -----------------------------
# Plotting
# -----------------------------
def plot_acc_per_subject(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    dfp = df.dropna(subset=["best_val_acc"]).copy()
    if len(dfp) == 0:
        print("[WARN] No best_val_acc values; skipping subject accuracy plot.")
        return

    dfp = natural_subject_sort_cols(dfp)
    dfp = dfp.sort_values(["ds_rank", "subj_num", "subject"]).copy()

    colors = [dataset_color(ds) for ds in dfp["dataset"]]

    plt.figure(figsize=(12, 4))
    plt.bar(dfp["subject"], dfp["best_val_acc"].astype(float), color=colors)
    plt.xticks(rotation=90)
    plt.ylabel("Best val_acc")
    plt.title("Best validation accuracy per subject (LOSO folds)")
    plt.grid(True, axis="y")
    plt.tight_layout()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DATASET_COLORS["DAS"], label="DAS"),
        plt.Rectangle((0, 0), 1, 1, color=DATASET_COLORS["DTU"], label="DTU"),
    ]
    plt.legend(handles=handles, loc="best")

    plt.savefig(os.path.join(out_dir, "best_val_acc_per_subject.png"))
    plt.close()

    # Dataset-level mean ± std
    stats = dfp.groupby("dataset")["best_val_acc"].agg(["count", "mean", "std"]).reset_index()
    stats.to_csv(os.path.join(out_dir, "dataset_best_val_acc_stats.csv"), index=False)

    stats["ds_rank"] = stats["dataset"].astype(str).apply(dataset_rank)
    stats = stats.sort_values(["ds_rank", "dataset"]).drop(columns=["ds_rank"])

    plt.figure()
    plt.bar(stats["dataset"], stats["mean"].astype(float),
            color=[dataset_color(ds) for ds in stats["dataset"]])
    plt.errorbar(stats["dataset"], stats["mean"].astype(float), yerr=stats["std"].astype(float),
                 fmt="none", capsize=4)
    plt.ylabel("Best val_acc")
    plt.title("Best val_acc by dataset (mean ± std across subjects)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_val_acc_by_dataset_mean_std.png"))
    plt.close()


def plot_bar_best_acc(df: pd.DataFrame, out_dir: str):
    """
    Best acc per fold, sorted by accuracy (keeps your original behavior),
    but colored by dataset.
    """
    os.makedirs(out_dir, exist_ok=True)
    dfp = df.dropna(subset=["best_val_acc"]).copy()
    if len(dfp) == 0:
        print("[WARN] No folds with best_val_acc found; skipping bar plot.")
        return

    dfp = dfp.sort_values("best_val_acc", ascending=False)
    dfp = natural_subject_sort_cols(dfp)

    colors = [dataset_color(ds) for ds in dfp["dataset"]]

    plt.figure(figsize=(10, 4))
    plt.bar(dfp["fold"].astype(str), dfp["best_val_acc"].astype(float), color=colors)
    plt.xticks(rotation=90)
    plt.ylabel("Best val_acc")
    plt.title("Best validation accuracy per fold")
    plt.grid(True, axis="y")
    plt.tight_layout()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DATASET_COLORS["DAS"], label="DAS"),
        plt.Rectangle((0, 0), 1, 1, color=DATASET_COLORS["DTU"], label="DTU"),
    ]
    plt.legend(handles=handles, loc="best")

    plt.savefig(os.path.join(out_dir, "best_val_acc_per_fold.png"))
    plt.close()


def plot_agg_confusion(all_labels: np.ndarray, all_preds: np.ndarray, out_dir: str):
    """
    Lighter, more readable confusion matrix.
    Uses a colorful-but-readable colormap + dynamic text color.
    """
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    plt.figure()
    # "YlGnBu" tends to be readable and not too dark; keeps contrast
    im = plt.imshow(cm, cmap="YlGnBu", vmin=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    maxv = cm.max() if cm.size else 1
    thresh = maxv * 0.5

    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > thresh else "black"
        plt.text(j, i, str(v), ha="center", va="center", color=color)

    plt.xticks([0, 1], ["Att Left", "Att Right"])
    plt.yticks([0, 1], ["Att Left", "Att Right"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Aggregated Confusion Matrix (all folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_all_folds.png"))
    plt.close()


# -----------------------------
# Metrics reading
# -----------------------------
def find_metrics_csv_anywhere(fold_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(fold_dir, "**", "metrics.csv"), recursive=True)
    candidates = sorted(candidates, key=lambda p: (p.count(os.sep), p))
    return candidates[-1] if candidates else None

def load_metrics(metrics_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if "epoch" not in df.columns:
        df["epoch"] = np.nan
    if "step" not in df.columns:
        df["step"] = np.arange(len(df))
    return df.sort_values(["epoch", "step"])

def pick_metric_column(df: pd.DataFrame, preferred: list[str]) -> str | None:
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        for p in preferred:
            if p in c:
                return c
    return None

def epoch_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col is None or col not in df.columns:
        return None
    s = df[["epoch", col]].dropna()
    if len(s) == 0:
        return None
    return s.groupby("epoch")[col].last()

def best_val_from_metrics(metrics_csv: str) -> dict:
    df = load_metrics(metrics_csv)
    val_acc_col = pick_metric_column(df, ["val_acc", "val_accuracy", "val_acc_epoch"])
    val_loss_col = pick_metric_column(df, ["val_loss", "val_loss_epoch"])

    out = {"best_val_acc": np.nan, "best_epoch": np.nan, "best_val_loss": np.nan}

    va = epoch_series(df, val_acc_col)
    if va is not None and len(va) > 0:
        best_epoch = int(va.idxmax())
        out["best_val_acc"] = float(va.loc[best_epoch])
        out["best_epoch"] = best_epoch

        vl = epoch_series(df, val_loss_col)
        if vl is not None and best_epoch in vl.index:
            out["best_val_loss"] = float(vl.loc[best_epoch])

    return out


# -----------------------------
# Posthoc outputs
# -----------------------------
def read_val_outputs_npz(fold_dir: str):
    npz_path = os.path.join(fold_dir, "posthoc", "val_outputs.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path)
    if "preds" not in d.files or "labels" not in d.files:
        return None
    return d["preds"], d["labels"]


# ---------------------------
# Baseline comparison
# ---------------------------
def load_linear_baseline(baseline_csv: str) -> pd.DataFrame:
    """
    Expects columns:
      Subject_ID, Dataset, Windowed_Accuracy
    Produces:
      subject, dataset, linear_acc, subj_num, ds_rank
    """
    dfb = pd.read_csv(baseline_csv)

    needed = {"Subject_ID", "Dataset", "Windowed_Accuracy"}
    missing = needed - set(dfb.columns)
    if missing:
        raise ValueError(f"Baseline CSV missing columns: {missing}. Expected {needed}")

    dfb = dfb.rename(columns={
        "Subject_ID": "subject",
        "Dataset": "dataset",
        "Windowed_Accuracy": "linear_acc",
    }).copy()

    dfb["dataset"] = dfb["dataset"].astype(str).str.upper()
    dfb["subject"] = dfb["subject"].astype(str)

    # If % scale, convert to 0..1
    if dfb["linear_acc"].dropna().max() > 1.5:
        dfb["linear_acc"] = dfb["linear_acc"] / 100.0

    dfb["subj_num"] = dfb["subject"].apply(parse_subject_num)
    dfb["ds_rank"] = dfb["dataset"].apply(dataset_rank)

    return dfb[["subject", "dataset", "linear_acc", "subj_num", "ds_rank"]].copy()


def plot_compare_per_subject(df_dl: pd.DataFrame, df_lin: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    dfd = df_dl.dropna(subset=["best_val_acc"]).copy()
    if len(dfd) == 0:
        print("[WARN] No DL best_val_acc available; skipping baseline comparison plots.")
        return

    dfd = natural_subject_sort_cols(dfd)
    dfd = dfd.rename(columns={"best_val_acc": "dl_acc"})[
        ["subject", "dataset", "subj_num", "ds_rank", "dl_acc"]
    ].copy()

    dfm = pd.merge(dfd, df_lin, on=["subject", "dataset", "subj_num", "ds_rank"], how="left")
    dfm = dfm.sort_values(["ds_rank", "subj_num", "subject"]).copy()

    dfm.to_csv(os.path.join(out_dir, "dl_vs_linear_per_subject.csv"), index=False)

    missing = int(dfm["linear_acc"].isna().sum())
    if missing:
        print(f"[WARN] Baseline missing for {missing}/{len(dfm)} subjects after merge (check naming).")

    x = np.arange(len(dfm))
    width = 0.42

    plt.figure(figsize=(13, 4))
    plt.bar(
        x - width / 2,
        dfm["dl_acc"].astype(float),
        width,
        color=[dataset_color(ds) for ds in dfm["dataset"]],
    )

    if dfm["linear_acc"].notna().any():
        plt.bar(
            x + width / 2,
            dfm["linear_acc"].astype(float),
            width,
            label="Linear (SI)",
            color="#bdbdbd",
        )

    plt.xticks(x, dfm["subject"].astype(str), rotation=90)
    plt.ylabel("Accuracy")
    plt.title("DL vs Linear baseline per subject")
    plt.grid(True, axis="y")

    legend_elements = [
        Patch(facecolor="#bdbdbd", label="Linear (SI)"),
        Patch(facecolor=DATASET_COLORS["DAS"], label="DAS"),
        Patch(facecolor=DATASET_COLORS["DTU"], label="DTU"),
    ]
    plt.legend(handles=legend_elements,loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dl_vs_linear_per_subject.png"))
    plt.close()


def plot_compare_dataset_means(df_dl: pd.DataFrame, df_lin: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    dfd = df_dl.dropna(subset=["best_val_acc"]).copy()
    dfd = natural_subject_sort_cols(dfd)
    dfd = dfd.rename(columns={"best_val_acc": "dl_acc"})[
        ["subject", "dataset", "dl_acc", "ds_rank"]
    ].copy()

    dfm = pd.merge(
        dfd,
        df_lin[["subject", "dataset", "linear_acc"]],
        on=["subject", "dataset"],
        how="left",
    )

    stats = dfm.groupby("dataset").agg(
        n=("subject", "count"),
        dl_mean=("dl_acc", "mean"),
        dl_std=("dl_acc", "std"),
        lin_mean=("linear_acc", "mean"),
        lin_std=("linear_acc", "std"),
    ).reset_index()

    stats["ds_rank"] = stats["dataset"].apply(dataset_rank)
    stats = stats.sort_values(["ds_rank", "dataset"]).drop(columns=["ds_rank"])
    stats.to_csv(os.path.join(out_dir, "dl_vs_linear_dataset_stats.csv"), index=False)

    x = np.arange(len(stats))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(
        x - width / 2,
        stats["dl_mean"].astype(float),
        width,
        yerr=stats["dl_std"].astype(float),
        capsize=4,
        color=[dataset_color(ds) for ds in stats["dataset"]],
    )

    if stats["lin_mean"].notna().any():
        plt.bar(
            x + width / 2,
            stats["lin_mean"].astype(float),
            width,
            yerr=stats["lin_std"].astype(float),
            capsize=4,
            color="#bdbdbd",
        )

    plt.xticks(x, stats["dataset"].astype(str))
    plt.ylabel("Accuracy")
    plt.title("DL vs Linear baseline (mean ± std)")
    plt.grid(True, axis="y")
    legend_elements = [
        Patch(facecolor="#bdbdbd", label="Linear (SI)"),
        Patch(facecolor=DATASET_COLORS["DAS"], label="DAS"),
        Patch(facecolor=DATASET_COLORS["DTU"], label="DTU"),
    ]
    plt.legend(handles=legend_elements,loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dl_vs_linear_dataset_mean_std.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True,
                    help="Directory containing fold_* subfolders.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Where to write summary + plots. Default: <results-dir>/summary_posthoc")
    ap.add_argument("--baseline-csv", type=str, default=None,
                    help="Optional linear baseline CSV to compare against.")
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or os.path.join(results_dir, "summary_posthoc")
    os.makedirs(out_dir, exist_ok=True)

    fold_dirs = sorted([d for d in glob.glob(os.path.join(results_dir, "fold_*")) if os.path.isdir(d)])
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No fold_* directories found in: {results_dir}")

    rows = []
    all_preds, all_labels = [], []

    print(f"Found {len(fold_dirs)} folds under {results_dir}")

    for fd in fold_dirs:
        fold_name = os.path.basename(fd)
        metrics_csv = find_metrics_csv_anywhere(fd)

        row = {
            "fold": fold_name,
            "fold_dir": fd,
            "metrics_csv": metrics_csv or "",
            "best_val_acc": np.nan,
            "best_epoch": np.nan,
            "best_val_loss": np.nan,
        }

        if metrics_csv:
            row.update(best_val_from_metrics(metrics_csv))
            print(f"[OK] {fold_name}: metrics.csv -> {metrics_csv}")
        else:
            print(f"[WARN] {fold_name}: no metrics.csv found")

        out_npz = read_val_outputs_npz(fd)
        if out_npz is not None:
            preds, labels = out_npz
            all_preds.append(preds)
            all_labels.append(labels)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save summary table (sorted by accuracy, as before)
    df_sorted = df.sort_values("best_val_acc", ascending=False, na_position="last")
    df_sorted.to_csv(os.path.join(out_dir, "loso_summary.csv"), index=False)

    valid = df["best_val_acc"].dropna().astype(float)
    mean_acc = float(valid.mean()) if len(valid) else float("nan")
    std_acc = float(valid.std()) if len(valid) else float("nan")

    with open(os.path.join(out_dir, "loso_summary.txt"), "w") as f:
        f.write(f"Folds found: {len(df)}\n")
        f.write(f"Folds with best_val_acc: {len(valid)}\n")
        f.write(f"Mean best_val_acc: {mean_acc:.4f}\n")
        f.write(f"Std  best_val_acc: {std_acc:.4f}\n")

    # Plots
    plot_bar_best_acc(df_sorted, out_dir)
    plot_acc_per_subject(df_sorted, out_dir)

    if len(all_preds) > 0 and len(all_labels) > 0:
        all_preds_cat = np.concatenate(all_preds)
        all_labels_cat = np.concatenate(all_labels)
        plot_agg_confusion(all_labels_cat, all_preds_cat, out_dir)
    else:
        print("[INFO] No posthoc/val_outputs.npz found in folds; skipping aggregated confusion matrix.")

    # Baseline compare (optional)
    if args.baseline_csv:
        print(f"\nLoading linear baseline: {args.baseline_csv}")
        df_lin = load_linear_baseline(args.baseline_csv)
        plot_compare_per_subject(df_sorted, df_lin, out_dir)
        plot_compare_dataset_means(df_sorted, df_lin, out_dir)

    print(f"\nWrote summary + plots to: {out_dir}")
    if len(valid):
        print(f"Mean best_val_acc = {mean_acc:.4f} ± {std_acc:.4f}  (N={len(valid)})")
    else:
        print("No valid best_val_acc values found (likely no val_acc logged).")


if __name__ == "__main__":
    main()