import os
import glob
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Patch


DATASET_COLORS = {
    "DAS": "lightblue",
    "DTU": "lightgreen",
}


def dataset_color(ds: str) -> str:
    return DATASET_COLORS.get(str(ds).upper(), "lightgray")


def fold_to_subject(fold_name: str) -> str:
    s = str(fold_name)
    if s.startswith("fold_"):
        s = s[len("fold_"):]
    parts = s.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return s


def subject_to_dataset(subject: str) -> str:
    subject = str(subject)
    if "_" in subject:
        return subject.split("_")[-1].upper()
    return ""


def parse_subject_num(s: str) -> int:
    m = re.search(r"S(\d+)", str(s))
    return int(m.group(1)) if m else 10**9


def dataset_rank(ds: str) -> int:
    ds = str(ds).upper()
    return {"DAS": 0, "DTU": 1}.get(ds, 99)


def add_subject_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject" not in df.columns:
        df["subject"] = df["fold"].astype(str).apply(fold_to_subject)
    if "dataset" not in df.columns:
        df["dataset"] = df["subject"].astype(str).apply(subject_to_dataset)
    df["subj_num"] = df["subject"].astype(str).apply(parse_subject_num)
    df["ds_rank"] = df["dataset"].astype(str).apply(dataset_rank)
    return df


def read_metric_history_npz(fold_dir: str):
    npz_path = os.path.join(fold_dir, "posthoc", "metric_history.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        d = np.load(npz_path)
        return {k: d[k] for k in d.files}
    except Exception as e:
        print(f"[WARN] Could not read {npz_path}: {e}")
        return None


def interp_to_common_grid(x, y, x_common):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_common = np.asarray(x_common, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.full_like(x_common, np.nan, dtype=float)

    m = min(len(x), len(y))
    x = x[:m]
    y = y[:m]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.full_like(x_common, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    if len(x_unique) < 2:
        return np.full_like(x_common, np.nan, dtype=float)

    out = np.full_like(x_common, np.nan, dtype=float)
    inside = (x_common >= x_unique.min()) & (x_common <= x_unique.max())
    out[inside] = np.interp(x_common[inside], x_unique, y_unique)
    return out


def read_final_eval_best_npz(fold_dir: str):
    npz_path = os.path.join(fold_dir, "posthoc", "final_eval_best.npz")
    if not os.path.exists(npz_path):
        return None

    d = np.load(npz_path, allow_pickle=True)
    required = {"preds_window", "labels_window", "acc_window"}
    missing = required - set(d.files)
    if missing:
        print(f"[WARN] {npz_path} missing keys: {missing}")
        return None

    return {
        "preds_window": d["preds_window"],
        "labels_window": d["labels_window"],
        "acc_window": float(d["acc_window"]),
        "preds_trial": d["preds_trial"] if "preds_trial" in d.files else None,
        "labels_trial": d["labels_trial"] if "labels_trial" in d.files else None,
        "acc_trial": float(d["acc_trial"]) if "acc_trial" in d.files else np.nan,
    }


def find_metrics_csv_anywhere(fold_dir: str):
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


def pick_metric_column(df: pd.DataFrame, preferred: list[str]):
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        for p in preferred:
            if p in c:
                return c
    return None


def epoch_series(df: pd.DataFrame, col: str):
    if col is None or col not in df.columns:
        return None
    s = df[["epoch", col]].dropna()
    if len(s) == 0:
        return None
    return s.groupby("epoch")[col].last()


def best_val_from_metrics(metrics_csv: str) -> dict:
    df = load_metrics(metrics_csv)

    val_acc_window_col = pick_metric_column(df, ["val_acc_window", "val_acc_window_epoch"])
    val_loss_col = pick_metric_column(df, ["val_loss", "val_loss_epoch"])

    out = {
        "best_val_acc_window": np.nan,
        "best_epoch_window": np.nan,
        "best_val_loss_at_best_window": np.nan,
    }

    vaw = epoch_series(df, val_acc_window_col)
    if vaw is not None and len(vaw) > 0:
        best_epoch_window = int(vaw.idxmax())
        out["best_val_acc_window"] = float(vaw.loc[best_epoch_window])
        out["best_epoch_window"] = best_epoch_window

        vl = epoch_series(df, val_loss_col)
        if vl is not None and best_epoch_window in vl.index:
            out["best_val_loss_at_best_window"] = float(vl.loc[best_epoch_window])

    return out


def plot_confusion_matrix_window(labels, preds, mean_acc_text, out_path):
    labels = np.asarray(labels).astype(int).reshape(-1)
    preds = np.asarray(preds).astype(int).reshape(-1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    plt.figure(figsize=(5.5, 5))
    im = plt.imshow(cm, cmap="YlGnBu", vmin=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    maxv = cm.max() if cm.size else 1
    thresh = maxv * 0.5

    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > thresh else "black"
        plt.text(j, i, str(v), ha="center", va="center", color=color, fontsize=11)

    plt.xticks([0, 1], ["Att Left", "Att Right"])
    plt.yticks([0, 1], ["Att Left", "Att Right"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Window Confusion Matrix — Best Checkpoint\nMean accuracy across folds = {mean_acc_text}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_per_subject(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    d = df.dropna(subset=["final_eval_acc_window"]).copy()
    if len(d) == 0:
        print("[WARN] No final_eval_acc_window values for per-subject plot.")
        return

    d = add_subject_dataset_columns(d)
    d = d.sort_values(["ds_rank", "subj_num", "subject"]).copy()

    plt.figure(figsize=(12, 4.5))
    plt.bar(
        d["subject"],
        d["final_eval_acc_window"].astype(float),
        color=[dataset_color(ds) for ds in d["dataset"]],
    )
    plt.xticks(rotation=90)
    plt.ylabel("Window accuracy")
    plt.title("Window Accuracy per Subject — Best Checkpoint")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "window_accuracy_per_subject.png"))
    plt.close()


def plot_per_subject_reordered(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    d = df.dropna(subset=["final_eval_acc_window"]).copy()
    if len(d) == 0:
        print("[WARN] No final_eval_acc_window values for reordered per-subject plot.")
        return

    d = add_subject_dataset_columns(d)
    d = d.sort_values(["final_eval_acc_window", "ds_rank", "subj_num"], ascending=[False, True, True]).copy()

    plt.figure(figsize=(12, 4.5))
    plt.bar(
        d["subject"],
        d["final_eval_acc_window"].astype(float),
        color=[dataset_color(ds) for ds in d["dataset"]],
    )
    plt.xticks(rotation=90)
    plt.ylabel("Window accuracy")
    plt.title("Window Accuracy per Subject — Best Checkpoint — Reordered")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "window_accuracy_per_subject_reordered.png"))
    plt.close()


def plot_dataset_mean_accuracy(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    d = df.dropna(subset=["final_eval_acc_window"]).copy()
    if len(d) == 0:
        print("[WARN] No final_eval_acc_window values for dataset mean plot.")
        return

    d = add_subject_dataset_columns(d)

    stats = (
        d.groupby("dataset")["final_eval_acc_window"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    stats["ds_rank"] = stats["dataset"].apply(dataset_rank)
    stats = stats.sort_values(["ds_rank", "dataset"]).drop(columns=["ds_rank"])

    stats.to_csv(os.path.join(out_dir, "dataset_window_accuracy_summary.csv"), index=False)

    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(
        stats["dataset"],
        stats["mean"].astype(float),
        yerr=stats["std"].fillna(0).astype(float),
        capsize=4,
        color=[dataset_color(ds) for ds in stats["dataset"]],
    )

    for bar, mean_val in zip(bars, stats["mean"].astype(float)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    overall_mean = float(d["final_eval_acc_window"].mean())

    plt.ylabel("Mean window accuracy")
    plt.title(f"Mean Window Accuracy by Dataset — Overall mean = {overall_mean:.3f}")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "window_accuracy_by_dataset_mean_std.png"))
    plt.close()


def plot_dense_mean_curves_from_histories(
    df_summary: pd.DataFrame,
    out_dir: str,
    suffix: str,
    zoom_n_epochs: float = 5.0,
    n_grid: int = 400,
):
    os.makedirs(out_dir, exist_ok=True)

    histories = []
    max_x = 0.0

    for _, row in df_summary.iterrows():
        fold_dir = str(row.get("fold_dir", "")).strip()
        if not fold_dir or not os.path.isdir(fold_dir):
            continue

        hist = read_metric_history_npz(fold_dir)
        if hist is None:
            continue

        x = np.asarray(hist.get("val_event_x", []), dtype=float)
        if len(x) == 0:
            continue

        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue

        max_x = max(max_x, float(np.nanmax(x)))
        histories.append(hist)

    if len(histories) == 0:
        print(f"[WARN] No histories found for {suffix}")
        return

    x_common = np.linspace(0.0, max_x, int(n_grid))

    def stack_metric(metric_name: str):
        mats = []
        for hist in histories:
            x = hist.get("val_event_x", np.asarray([]))
            y = hist.get(metric_name, np.asarray([]))
            mats.append(interp_to_common_grid(x, y, x_common))
        return np.vstack(mats)

    def nanmeanstd(x: np.ndarray):
        if x.size == 0:
            ncols = len(x_common)
            return (
                np.full(ncols, np.nan, dtype=float),
                np.full(ncols, np.nan, dtype=float),
                np.zeros(ncols, dtype=int),
            )

        n = np.sum(~np.isnan(x), axis=0)

        mean = np.full(x.shape[1], np.nan, dtype=float)
        std = np.full(x.shape[1], np.nan, dtype=float)

        valid_cols = n > 0
        if np.any(valid_cols):
            mean[valid_cols] = np.nanmean(x[:, valid_cols], axis=0)

        valid_std_cols = n > 1
        if np.any(valid_std_cols):
            std[valid_std_cols] = np.nanstd(x[:, valid_std_cols], axis=0)

        return mean, std, n

    def safe_fill_between(x, mean, std, alpha=0.2):
        mask = np.isfinite(x) & np.isfinite(mean) & np.isfinite(std)
        if np.any(mask):
            plt.fill_between(
                x[mask],
                (mean - std)[mask],
                (mean + std)[mask],
                alpha=alpha,
            )

    te_acc = stack_metric("train_eval_acc_window")
    v_acc = stack_metric("val_acc_window")
    te_loss = stack_metric("train_eval_loss")
    v_loss = stack_metric("val_loss")

    te_acc_mean, te_acc_std, te_acc_n = nanmeanstd(te_acc)
    v_acc_mean, v_acc_std, v_acc_n = nanmeanstd(v_acc)
    te_loss_mean, te_loss_std, te_loss_n = nanmeanstd(te_loss)
    v_loss_mean, v_loss_std, v_loss_n = nanmeanstd(v_loss)

    dense_df = pd.DataFrame({
        "x": x_common,
        "train_eval_acc_window_mean": te_acc_mean,
        "train_eval_acc_window_std": te_acc_std,
        "n_train_eval_acc_window": te_acc_n,
        "val_acc_window_mean": v_acc_mean,
        "val_acc_window_std": v_acc_std,
        "n_val_acc_window": v_acc_n,
        "train_eval_loss_mean": te_loss_mean,
        "train_eval_loss_std": te_loss_std,
        "n_train_eval_loss": te_loss_n,
        "val_loss_mean": v_loss_mean,
        "val_loss_std": v_loss_std,
        "n_val_loss": v_loss_n,
    })
    dense_df.to_csv(os.path.join(out_dir, f"mean_dense_curve_table_{suffix}.csv"), index=False)

    # Accuracy
    if np.isfinite(te_acc_mean).any() or np.isfinite(v_acc_mean).any():
        plt.figure(figsize=(8.5, 5))
        if np.isfinite(te_acc_mean).any():
            plt.plot(x_common, te_acc_mean, label="Train eval window accuracy")
            safe_fill_between(x_common, te_acc_mean, te_acc_std, alpha=0.2)
        if np.isfinite(v_acc_mean).any():
            plt.plot(x_common, v_acc_mean, label="Validation window accuracy")
            safe_fill_between(x_common, v_acc_mean, v_acc_std, alpha=0.2)
        plt.xlabel("Epoch progress")
        plt.ylabel("Accuracy")
        plt.title(f"Mean Dense Window Accuracy — {suffix.replace('_', ' ').upper()}")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_dense_acc_curve_{suffix}.png"))
        plt.close()

    # Loss
    if np.isfinite(te_loss_mean).any() or np.isfinite(v_loss_mean).any():
        plt.figure(figsize=(8.5, 5))
        if np.isfinite(te_loss_mean).any():
            plt.plot(x_common, te_loss_mean, label="Train eval loss")
            safe_fill_between(x_common, te_loss_mean, te_loss_std, alpha=0.2)
        if np.isfinite(v_loss_mean).any():
            plt.plot(x_common, v_loss_mean, label="Validation loss")
            safe_fill_between(x_common, v_loss_mean, v_loss_std, alpha=0.2)
        plt.xlabel("Epoch progress")
        plt.ylabel("Loss")
        plt.title(f"Mean Dense Window Loss — {suffix.replace('_', ' ').upper()}")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_dense_loss_curve_{suffix}.png"))
        plt.close()

    # Zoom accuracy
    zoom_mask = x_common <= float(zoom_n_epochs)
    x_zoom = x_common[zoom_mask]

    if len(x_zoom) > 0 and (np.isfinite(te_acc_mean[zoom_mask]).any() or np.isfinite(v_acc_mean[zoom_mask]).any()):
        plt.figure(figsize=(8.5, 5))
        if np.isfinite(te_acc_mean[zoom_mask]).any():
            plt.plot(x_zoom, te_acc_mean[zoom_mask], label="Train eval window accuracy")
            safe_fill_between(x_zoom, te_acc_mean[zoom_mask], te_acc_std[zoom_mask], alpha=0.2)
        if np.isfinite(v_acc_mean[zoom_mask]).any():
            plt.plot(x_zoom, v_acc_mean[zoom_mask], label="Validation window accuracy")
            safe_fill_between(x_zoom, v_acc_mean[zoom_mask], v_acc_std[zoom_mask], alpha=0.2)
        plt.xlabel("Epoch progress")
        plt.ylabel("Accuracy")
        plt.title(f"Mean Dense Window Accuracy — First {zoom_n_epochs:g} Epochs — {suffix.replace('_', ' ').upper()}")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_dense_acc_zoom_{suffix}.png"))
        plt.close()

    # Zoom loss
    if len(x_zoom) > 0 and (np.isfinite(te_loss_mean[zoom_mask]).any() or np.isfinite(v_loss_mean[zoom_mask]).any()):
        plt.figure(figsize=(8.5, 5))
        if np.isfinite(te_loss_mean[zoom_mask]).any():
            plt.plot(x_zoom, te_loss_mean[zoom_mask], label="Train eval loss")
            safe_fill_between(x_zoom, te_loss_mean[zoom_mask], te_loss_std[zoom_mask], alpha=0.2)
        if np.isfinite(v_loss_mean[zoom_mask]).any():
            plt.plot(x_zoom, v_loss_mean[zoom_mask], label="Validation loss")
            safe_fill_between(x_zoom, v_loss_mean[zoom_mask], v_loss_std[zoom_mask], alpha=0.2)
        plt.xlabel("Epoch progress")
        plt.ylabel("Loss")
        plt.title(f"Mean Dense Window Loss — First {zoom_n_epochs:g} Epochs — {suffix.replace('_', ' ').upper()}")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mean_dense_loss_zoom_{suffix}.png"))
        plt.close()


def write_clean_summary_txt(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    d = df.dropna(subset=["final_eval_acc_window"]).copy()
    if len(d) == 0:
        return

    d = add_subject_dataset_columns(d)

    overall_mean = float(d["final_eval_acc_window"].mean())
    overall_std = float(d["final_eval_acc_window"].std())

    ds_stats = (
        d.groupby("dataset")["final_eval_acc_window"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    ds_stats["ds_rank"] = ds_stats["dataset"].apply(dataset_rank)
    ds_stats = ds_stats.sort_values(["ds_rank", "dataset"]).drop(columns=["ds_rank"])

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("Clean LOSO summary\n")
        f.write("==================\n\n")
        f.write(f"Number of folds with final window accuracy: {len(d)}\n")
        f.write(f"Overall mean final window accuracy: {overall_mean:.4f}\n")
        f.write(f"Overall std final window accuracy: {overall_std:.4f}\n\n")

        for _, row in ds_stats.iterrows():
            f.write(
                f"{row['dataset']}: "
                f"N={int(row['count'])}, "
                f"mean={float(row['mean']):.4f}, "
                f"std={float(row['std']) if pd.notna(row['std']) else float('nan'):.4f}\n"
            )


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

    dfb["subject"] = dfb["subject"].astype(str)
    dfb["dataset"] = dfb["dataset"].astype(str).str.upper()

    if dfb["linear_acc"].dropna().max() > 1.5:
        dfb["linear_acc"] = dfb["linear_acc"] / 100.0

    dfb["subj_num"] = dfb["subject"].apply(parse_subject_num)
    dfb["ds_rank"] = dfb["dataset"].apply(dataset_rank)

    return dfb[["subject", "dataset", "linear_acc", "subj_num", "ds_rank"]].copy()


def plot_compare_per_subject(df_dl: pd.DataFrame, df_lin: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    dfd = df_dl.dropna(subset=["final_eval_acc_window"]).copy()
    if len(dfd) == 0:
        print("[WARN] No DL final_eval_acc_window available; skipping baseline comparison.")
        return

    dfd = add_subject_dataset_columns(dfd)
    dfd = dfd.rename(columns={"final_eval_acc_window": "dl_acc"})[
        ["subject", "dataset", "subj_num", "ds_rank", "dl_acc"]
    ].copy()

    dfm = pd.merge(
        dfd,
        df_lin,
        on=["subject", "dataset", "subj_num", "ds_rank"],
        how="left",
    )
    dfm = dfm.sort_values(["ds_rank", "subj_num", "subject"]).copy()
    dfm.to_csv(os.path.join(out_dir, "dl_vs_linear_per_subject.csv"), index=False)

    missing = int(dfm["linear_acc"].isna().sum())
    if missing:
        print(f"[WARN] Baseline missing for {missing}/{len(dfm)} subjects after merge.")

    x = np.arange(len(dfm))
    width = 0.42

    plt.figure(figsize=(13, 4.5))
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
            color="#bdbdbd",
        )

    plt.xticks(x, dfm["subject"].astype(str), rotation=90)
    plt.ylabel("Window accuracy")
    plt.title("DL vs Linear Baseline per Subject")
    plt.grid(True, axis="y")

    legend_elements = [
        Patch(facecolor="#bdbdbd", label="Linear"),
        Patch(facecolor=DATASET_COLORS["DAS"], label="DAS"),
        Patch(facecolor=DATASET_COLORS["DTU"], label="DTU"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dl_vs_linear_per_subject.png"))
    plt.close()


def plot_compare_dataset_means(df_dl: pd.DataFrame, df_lin: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    dfd = df_dl.dropna(subset=["final_eval_acc_window"]).copy()
    if len(dfd) == 0:
        print("[WARN] No DL final_eval_acc_window available; skipping dataset baseline comparison.")
        return

    dfd = add_subject_dataset_columns(dfd)
    dfd = dfd.rename(columns={"final_eval_acc_window": "dl_acc"})[
        ["subject", "dataset", "subj_num", "ds_rank", "dl_acc"]
    ].copy()

    dfm = pd.merge(
        dfd,
        df_lin,
        on=["subject", "dataset", "subj_num", "ds_rank"],
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

    plt.figure(figsize=(6.5, 4.5))
    bars1 = plt.bar(
        x - width / 2,
        stats["dl_mean"].astype(float),
        width,
        yerr=stats["dl_std"].fillna(0).astype(float),
        capsize=4,
        color=[dataset_color(ds) for ds in stats["dataset"]],
    )

    if stats["lin_mean"].notna().any():
        bars2 = plt.bar(
            x + width / 2,
            stats["lin_mean"].astype(float),
            width,
            yerr=stats["lin_std"].fillna(0).astype(float),
            capsize=4,
            color="#bdbdbd",
        )
        for bar, val in zip(bars2, stats["lin_mean"].astype(float)):
            if np.isfinite(val):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    for bar, val in zip(bars1, stats["dl_mean"].astype(float)):
        if np.isfinite(val):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.xticks(x, stats["dataset"].astype(str))
    plt.ylabel("Window accuracy")
    plt.title("DL vs Linear Baseline — Mean Window Accuracy")
    plt.grid(True, axis="y")

    legend_elements = [
        Patch(facecolor="#bdbdbd", label="Linear"),
        Patch(facecolor=DATASET_COLORS["DAS"], label="DAS"),
        Patch(facecolor=DATASET_COLORS["DTU"], label="DTU"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dl_vs_linear_dataset_mean_std.png"))
    plt.close()


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing fold_* subfolders or a folds/ subfolder.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output root. Default: same as results-dir",
    )
    ap.add_argument(
        "--zoom-epochs",
        type=float,
        default=5.0,
        help="Number of early epochs for zoom plots.",
    )
    ap.add_argument(
        "--baseline-csv",
        type=str,
        default=None,
        help="Optional linear baseline CSV for DL-vs-linear comparison.",
    )
    args = ap.parse_args()

    results_dir = args.results_dir
    out_root = args.out_dir or results_dir

    summary_dir = os.path.join(results_dir, "summary")
    comparison_dir = os.path.join(results_dir, "dl_vs_linear")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    folds_root = os.path.join(results_dir, "folds")
    fold_dirs = sorted([d for d in glob.glob(os.path.join(folds_root, "fold_*")) if os.path.isdir(d)])

    if len(fold_dirs) == 0:
        raise FileNotFoundError(
            f"No fold_* directories found in:\n  {folds_root}"
        )


    print(f"Found {len(fold_dirs)} folds under {folds_root}")

    rows = []
    all_preds_window = []
    all_labels_window = []

    for fd in fold_dirs:
        fold_name = os.path.basename(fd)
        metrics_csv = find_metrics_csv_anywhere(fd)

        row = {
            "fold": fold_name,
            "fold_dir": fd,
            "metrics_csv": metrics_csv or "",
            "best_val_acc_window": np.nan,
            "best_epoch_window": np.nan,
            "best_val_loss_at_best_window": np.nan,
            "final_eval_acc_window": np.nan,
            "final_eval_acc_trial": np.nan,
        }

        if metrics_csv:
            row.update(best_val_from_metrics(metrics_csv))

        final_eval = read_final_eval_best_npz(fd)
        if final_eval is not None:
            row["final_eval_acc_window"] = final_eval["acc_window"]
            row["final_eval_acc_trial"] = final_eval["acc_trial"]

            all_preds_window.append(final_eval["preds_window"])
            all_labels_window.append(final_eval["labels_window"])

        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_subject_dataset_columns(df)
    df.to_csv(os.path.join(summary_dir, "loso_summary.csv"), index=False)

    plot_dense_mean_curves_from_histories(
        df_summary=df,
        out_dir=summary_dir,
        suffix="all_folds",
        zoom_n_epochs=args.zoom_epochs,
    )

    for ds in ["DAS", "DTU"]:
        dsub = df[df["dataset"].astype(str).str.upper() == ds].copy()
        if len(dsub) == 0:
            continue
        plot_dense_mean_curves_from_histories(
            df_summary=dsub,
            out_dir=summary_dir,
            suffix=f"{ds.lower()}_folds",
            zoom_n_epochs=args.zoom_epochs,
        )

    plot_per_subject(df, summary_dir)
    plot_per_subject_reordered(df, summary_dir)
    plot_dataset_mean_accuracy(df, summary_dir)
    write_clean_summary_txt(df, summary_dir)

    if len(all_preds_window) > 0:
        preds_w = np.concatenate(all_preds_window)
        labels_w = np.concatenate(all_labels_window)
        acc_from_cm = float(np.mean(preds_w == labels_w))

        mean_fold_acc = df["final_eval_acc_window"].dropna().mean()
        mean_text = f"{float(mean_fold_acc):.3f}" if pd.notna(mean_fold_acc) else f"{acc_from_cm:.3f}"

        plot_confusion_matrix_window(
            labels=labels_w,
            preds=preds_w,
            mean_acc_text=mean_text,
            out_path=os.path.join(summary_dir, "confusion_matrix_window_all_folds.png"),
        )

        with open(os.path.join(summary_dir, "confusion_matrix_window_accuracy.txt"), "w") as f:
            f.write(f"Accuracy from concatenated window predictions: {acc_from_cm:.6f}\n")
            f.write(f"Number of window samples: {len(labels_w)}\n")
    else:
        print("[WARN] No final_eval_best.npz files found for window confusion matrix.")

    if args.baseline_csv:
        print(f"Loading linear baseline from: {args.baseline_csv}")
        df_lin = load_linear_baseline(args.baseline_csv)
        plot_compare_per_subject(df, df_lin, comparison_dir)
        plot_compare_dataset_means(df, df_lin, comparison_dir)
    else:
        print("[INFO] No --baseline-csv provided; skipping DL-vs-linear comparison.")

    print("\nWrote outputs to:")
    print(f"  Summary:      {summary_dir}")
    print(f"  Comparison:   {comparison_dir}")
    print("")


if __name__ == "__main__":
    main()