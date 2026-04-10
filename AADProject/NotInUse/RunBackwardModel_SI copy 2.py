#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import gzip
import os
from pathlib import Path

import numpy as np

import DataPreparation
from BackwardModel import summaryStats
from paths import paths


from BackwardModel.plots import (
    plot_subject_bars,
    plot_correlation_distributions,
    plot_window_length_curve,
    plot_trf_weights,
    plot_window_heatmap,
)

# =========================
# Helpers: normalization
# =========================

class Standardizer:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X):
        self.mu = np.mean(X, axis=0, keepdims=True)
        self.sigma = np.std(X, axis=0, keepdims=True)
        self.sigma[self.sigma < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sigma


# =========================
# Lag matrix + ridge
# =========================

def create_lag_matrix(EEG, lags_t, fs):
    """
    input:
        EEG: (t,channels)
        lags_t: (lag_low_ms, lag_high_ms)
        fs: Hz
    output: X_lagged: (t, n_lags*channels)
    """
    lag_low, lag_high = lags_t

    lags = np.arange(
        int(np.floor(lag_low / 1000 * fs)),
        int(np.ceil(lag_high / 1000 * fs)) + 1
    ).astype(int)

    n_samples_t, n_ch = EEG.shape
    n_lags = len(lags)

    X_lagged = np.zeros((n_samples_t, n_lags * n_ch), dtype=EEG.dtype)

    for i, lag in enumerate(lags):
        c0 = i * n_ch
        c1 = c0 + n_ch

        if lag > 0:
            X_lagged[:-lag, c0:c1] = EEG[lag:, :]
        elif lag < 0:
            d = -lag
            X_lagged[d:, c0:c1] = EEG[:-d, :]
        else:
            X_lagged[:, c0:c1] = EEG

    return X_lagged


def ridge_predict(X, w):
    return X @ w


# =========================
# Fast windowed correlation
# =========================

def _prefix_sums_1d(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(x), np.cumsum(x * x)


def _seg_sum(p, start, end):
    # sum on [start, end)
    if end <= 0:
        return 0.0
    if start <= 0:
        return float(p[end - 1])
    return float(p[end - 1] - p[start - 1])


def _window_corr_1d_from_prefix(px, pxx, py, pyy, pxy, start, end):
    n = end - start
    if n <= 1:
        return 0.0

    sx = _seg_sum(px, start, end)
    sy = _seg_sum(py, start, end)
    sxx = _seg_sum(pxx, start, end)
    syy = _seg_sum(pyy, start, end)
    sxy = _seg_sum(pxy, start, end)

    mx = sx / n
    my = sy / n

    vx = (sxx / n) - mx * mx
    vy = (syy / n) - my * my
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0

    cov = (sxy / n) - mx * my
    return cov / np.sqrt(vx * vy)


def windowed_corr_fast(y_pred, y_true, window_len, step):
    """
    y_pred, y_true: (T,) or (T,B)
    returns: correlations per window (len = n_windows), averaged over bands if multiband
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    T = y_pred.shape[0]
    starts = np.arange(0, T - window_len + 1, step, dtype=int)

    if y_pred.ndim == 1:
        yp = y_pred.astype(float, copy=False)
        yt = y_true.astype(float, copy=False)
        px, pxx = _prefix_sums_1d(yp)
        py, pyy = _prefix_sums_1d(yt)
        pxy = np.cumsum(yp * yt)

        cors = np.empty(len(starts), dtype=float)
        for i, st in enumerate(starts):
            cors[i] = _window_corr_1d_from_prefix(px, pxx, py, pyy, pxy, st, st + window_len)
        return cors

    # multiband
    B = y_pred.shape[1]
    cors = np.empty((len(starts), B), dtype=float)
    for b in range(B):
        yp = y_pred[:, b].astype(float, copy=False)
        yt = y_true[:, b].astype(float, copy=False)
        px, pxx = _prefix_sums_1d(yp)
        py, pyy = _prefix_sums_1d(yt)
        pxy = np.cumsum(yp * yt)
        for i, st in enumerate(starts):
            cors[i, b] = _window_corr_1d_from_prefix(px, pxx, py, pyy, pxy, st, st + window_len)
    return cors.mean(axis=1)


# =========================
# Data loading
# =========================

def load_subject_block(nwb_path, lags, fs, multiband, eeg_std=None, y_std=None):
    """
    Loads all trials for 1 subject and returns stacked arrays:
      X: (T_total, n_features)
      y: (T_total, Ydim) or (T_total,1)
    """
    data_subj, _ = DataPreparation.Load_data(nwb_path, merged=True, multiband=multiband)

    X_list = []
    y_list = []

    for eeg, env_att, _ in data_subj:
        if eeg_std is not None:
            eeg = eeg_std.transform(eeg)

        y = env_att if multiband else env_att.reshape(-1, 1)
        if y_std is not None:
            y = y_std.transform(y)

        X_list.append(create_lag_matrix(eeg, lags, fs))
        y_list.append(y)

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y


# =========================
# Core: run one LOSO fold
# =========================

def run_mTRF_subject(train_paths, test_path, cfg):
    
    # load config parameters
    fs = cfg["preprocessing"]["target_fs"]
    window_len = int(cfg["backward_model"]["window_s"] * fs)
    step = int(cfg["backward_model"]["step_s"] * fs)
    lags = cfg["backward_model"]["lag_ms"]
    multiband = cfg["backward_model"]["multiband"]

    print(f"Loading {len(train_paths)} training subjects...")

    # ----- Split train/val for lambda tuning -----
    val_fraction = 0.2
    n_subjects = len(train_paths)
    n_val = max(1, int(n_subjects * val_fraction))
    val_paths = train_paths[-n_val:]
    tr_paths = train_paths[:-n_val]

    # ----- Fit standardizers on TRAIN subjects only -----
    eeg_accum = []
    y_accum = []

    for p in tr_paths:
        data_subj, _ = DataPreparation.Load_data(p, merged=True, multiband=multiband)
        for eeg, env_att, _ in data_subj:
            eeg_accum.append(eeg)
            y = env_att if multiband else env_att.reshape(-1, 1)
            y_accum.append(y)

    eeg_std = Standardizer().fit(np.vstack(eeg_accum))
    y_std = Standardizer().fit(np.vstack(y_accum))

    # ----- Determine feature dimension -----
    X_first, y_first = load_subject_block(tr_paths[0], lags, fs, multiband, eeg_std, y_std)
    n_features = X_first.shape[1]
    Ydim = y_first.shape[1] if multiband else 1

    # ----- Precompute XtX, XtY for training set (excluding val) -----
    XtX = X_first.T @ X_first
    XtY = (X_first.T @ y_first) if multiband else (X_first.T @ y_first).ravel()

    for p in tr_paths[1:]:
        Xs, ys = load_subject_block(p, lags, fs, multiband, eeg_std, y_std)
        XtX += Xs.T @ Xs
        XtY += (Xs.T @ ys) if multiband else (Xs.T @ ys).ravel()

    # ----- Preload validation blocks -----
    print("Tuning λ ...")
    val_blocks = [
        load_subject_block(p, lags, fs, multiband, eeg_std, y_std)
        for p in val_paths
    ]

    # ----- Lambda tuning using eigendecomposition (fast per-lambda) -----
    lambdas = np.logspace(0, 6, 7)
    best_lambda = None
    best_score = -np.inf

    eigvals, Q = np.linalg.eigh(XtX)
    QtY = Q.T @ XtY

    for lam in lambdas:
        #train on val
        if multiband:
            w = Q @ (QtY / (eigvals[:, None] + lam))
        else:
            w = Q @ (QtY / (eigvals + lam))
        
        #test on val
        scores = []
        for Xv, yv in val_blocks:
            y_pred = ridge_predict(Xv, w)
            if multiband:
                r_att = float(np.mean([np.corrcoef(y_pred[:, b], yv[:, b])[0, 1] for b in range(y_pred.shape[1])]))
            else:
                r_att = float(np.corrcoef(y_pred.ravel(), yv.ravel())[0, 1])
            scores.append(r_att)

        mean_score = float(np.mean(scores))
        print(f"λ={lam:.1e}  | mean r={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam

    print(f"Best λ = {best_lambda}")

    # ----- Train on all training subjects -----
    XtX = np.zeros((n_features, n_features), dtype=float)
    XtY = np.zeros((n_features, Ydim), dtype=float) if multiband else np.zeros((n_features,), dtype=float)

    for p in train_paths:
        Xs, ys = load_subject_block(p, lags, fs, multiband, eeg_std, y_std)
        XtX += Xs.T @ Xs
        XtY += (Xs.T @ ys) if multiband else (Xs.T @ ys).ravel()

    w = np.linalg.solve(XtX + best_lambda * np.eye(n_features), XtY)
    decoder_w = w.copy()

    # ----- Evaluate on test subject -----
    test_data, test_nwb = DataPreparation.Load_data(test_path, merged=True, multiband=multiband)

    results = []
    for ti, (eeg, env_att, env_unatt) in enumerate(test_data):
        eeg = eeg_std.transform(eeg)

        y_att = env_att if multiband else env_att.reshape(-1, 1)
        y_un = env_unatt if multiband else env_unatt.reshape(-1, 1)

        y_att = y_std.transform(y_att)
        y_un = y_std.transform(y_un)

        X = create_lag_matrix(eeg, lags, fs)
        y_pred = X @ w

        # Full-trial correlation
        if multiband:
            corr_att = float(np.mean([np.corrcoef(y_pred[:, b], y_att[:, b])[0, 1] for b in range(Ydim)]))
            corr_unatt = float(np.mean([np.corrcoef(y_pred[:, b], y_un[:, b])[0, 1] for b in range(Ydim)]))
        else:
            corr_att = float(np.corrcoef(y_pred.ravel(), y_att.ravel())[0, 1])
            corr_unatt = float(np.corrcoef(y_pred.ravel(), y_un.ravel())[0, 1])

        correct_full = bool(corr_att > corr_unatt)

        # Window evaluation
        if multiband:
            cors_att = windowed_corr_fast(y_pred, y_att, window_len, step)
            cors_un = windowed_corr_fast(y_pred, y_un, window_len, step)
        else:
            cors_att = windowed_corr_fast(y_pred.ravel(), y_att.ravel(), window_len, step)
            cors_un = windowed_corr_fast(y_pred.ravel(), y_un.ravel(), window_len, step)

        correct_vec = cors_att > cors_un
        window_acc = float(correct_vec.mean())

        starts = np.arange(0, X.shape[0] - window_len + 1, step, dtype=int)
        win_list = [
            {"start": float(st / fs), "end": float((st + window_len) / fs), "correct": bool(c)}
            for st, c in zip(starts, correct_vec)
        ]

        results.append({
            "trial": ti + 1,
            "corr_att": corr_att,
            "corr_unatt": corr_unatt,
            "correct_full": correct_full,
            "window_accuracy": window_acc,
            "windows": win_list
        })

        print(
            f"Trial {ti + 1}/{len(test_data)}: r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | "
            f"{correct_full} | win_acc={window_acc:.2f}"
        )

    full_acc = float(np.mean([r["correct_full"] for r in results]))
    win_acc = float(np.mean([r["window_accuracy"] for r in results]))

    dataset_label = "unknown"
    p = str(test_path).lower()
    if "dtu" in p:
        dataset_label = "DTU"
    elif "das" in p:
        dataset_label = "DAS"

    subject_label = os.path.basename(str(test_path)).replace(".nwb", "")

    print(f"Subject {test_nwb.identifier}: Full acc = {full_acc:.2f} | Window acc = {win_acc:.2f}")

    return {
        "subject_id": test_nwb.identifier,
        "subject_label": subject_label,
        "dataset": dataset_label,
        "results": results,
        "full_accuracy": full_acc,
        "window_accuracy": win_acc,
        "decoder_w": decoder_w,
        "lags": lags,
        "fs": fs,
        "n_channels": test_data[0][0].shape[1],
    }


# =========================
# JSON serialization
# =========================
def write_json_gz(path, obj, indent=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


# =========================
# Summaries and plots
# =========================

def summarize_results(results, cfg, run_dir):
    if not results:
        print("No usable results, skipping summary.")
        return

    def save_group(group_results, group_name, group_dir):
        os.makedirs(group_dir, exist_ok=True)

        full_accs = [r["full_accuracy"] for r in group_results]
        win_accs = [r["window_accuracy"] for r in group_results]

        print(f"\n=== Results: {group_name} ===")
        print(f"N subjects: {len(group_results)}")
        print(f"Mean full-trial accuracy: {np.mean(full_accs):.2f}")
        print(f"Mean windowed accuracy:  {np.mean(win_accs):.2f}")

        json_path = os.path.join(group_dir, f"mTRF_results_{group_name}.json")
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(group_results), f, indent=4)

        csv_path = os.path.join(group_dir, f"mTRF_summary_{group_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Subject_ID", "Dataset", "Full_Trial_Accuracy", "Windowed_Accuracy"])

            for subj in group_results:
                writer.writerow([
                    subj.get("subject_label", subj["subject_id"]),
                    subj.get("dataset", "unknown"),
                    subj["full_accuracy"],
                    subj["window_accuracy"],
                ])

            writer.writerow([
                f"MEAN_{group_name}",
                group_name,
                float(np.mean(full_accs)),
                float(np.mean(win_accs)),
            ])

            all_full = [r["full_accuracy"] for r in results]
            all_win = [r["window_accuracy"] for r in results]
            writer.writerow([
                "MEAN_ALL",
                "ALL",
                float(np.mean(all_full)),
                float(np.mean(all_win)),
            ])

        all_att = np.concatenate([np.array([t["corr_att"] for t in subj["results"]]) for subj in group_results])
        all_unatt = np.concatenate([np.array([t["corr_unatt"] for t in subj["results"]]) for subj in group_results])
        stats = summaryStats.SummaryStats(all_att, all_unatt)

        stats_path = os.path.join(group_dir, f"stats_{group_name}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

        summaryStats.plot_histograms(all_att, all_unatt, os.path.join(group_dir, "Correlations"))

        subjects_lbl = [subj.get("subject_label", subj["subject_id"]) for subj in group_results]
        window_s = cfg["backward_model"]["window_s"]
        plot_subject_bars(full_accs, win_accs, subjects_lbl, os.path.join(group_dir, "Accuracy"), mode="Subject-independent",window_s=window_s)


    save_group(results, "ALL", run_dir)

    grouped = {}
    for r in results:
        ds = str(r.get("dataset", "unknown"))
        grouped.setdefault(ds, []).append(r)

    if len(grouped) > 1:
        for ds, group_results in grouped.items():
            group_dir = os.path.join(run_dir, ds)
            save_group(group_results, ds, group_dir)


# =========================
# Merge helper for array mode
# =========================

def merge_subject_jsons(run_dir):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("json/*.json"))
    results = []
    for fp in files:
        import gzip
        with open(fp, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results



# =========================
# Main entrypoint
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-subject", type=str, default=None,
                        help="Run LOSO for exactly this subject name (must match config subjects list).")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Existing run directory. If not provided, a new one is created.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge per-subject JSONs in --run-dir and run summarize_results.")
    args = parser.parse_args()

    cfg = paths.load_config()
    mode = cfg["SI_mode"]["mode"]
    subjects = cfg["subjects"]["all"]

    print("\n========== RUNNING LOSO BACKWARD MODEL ==========")
    print(f"Subjects: {subjects}")
    print("=================================================\n")

    SI_base = paths.RESULTS_LIN / "SI"
    if args.run_dir is None:
        run_dir = paths.get_next_run_dir(SI_base)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Saving all SI results to: {run_dir}")
        paths.save_config_copy(cfg, run_dir)
    else:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
        print(f"Using existing run dir: {run_dir}")

    if args.merge_only:
        all_results = merge_subject_jsons(run_dir)
        print(f"Merged {len(all_results)} subject JSON files from {run_dir}")
        summarize_results(all_results, cfg, run_dir)
        print("\n========== DONE ==========\n")
        return

    if args.single_subject is not None:
        test_subj = args.single_subject
        if test_subj not in subjects:
            raise ValueError(f"--single-subject {test_subj} not found in config subjects list.")

        test_path = paths.subject_eegPP(test_subj)
        train_paths = [paths.subject_eegPP(s) for s in subjects if s != test_subj]

        res = run_mTRF_subject(train_paths, test_path, cfg)

        JSON_dir=os.path.join(run_dir,"json")
        os.makedirs(JSON_dir, exist_ok=True)
        subject_json = os.path.join(JSON_dir, f"{test_subj}.json")
        with open(subject_json, "w") as f:
                json.dump(make_json_serializable(res), f, indent=2)
        

        print(f"✓ Finished subject {test_subj}")
        print(f"✓ Saved: {subject_json}")
        return

    # Sequential fallback
    all_results = []

    def LOSO_loop(subj_list):
        for i, test_subj in enumerate(subj_list, start=1):
            print(f"\n===============================================")
            print(f"LOSO {i}/{len(subj_list)} — Test subject: {test_subj}")
            print("===============================================")

            test_path = paths.subject_eegPP(test_subj)
            train_paths = [paths.subject_eegPP(s) for s in subjects if s != test_subj]

            res = run_mTRF_subject(train_paths, test_path, cfg)
            all_results.append(res)

            tmp_json = os.path.join(run_dir, f"mTRF_partial_{i}.json")
            with open(tmp_json, "w") as f:
                json.dump(make_json_serializable(all_results), f, indent=4)

            print(f"✓ Finished subject {test_subj}")
            print(f"✓ Partial results saved to: {tmp_json}\n")

    if mode == "separate":
        LOSO_loop(subjects)
    elif mode == "mixed":
        LOSO_loop([s for s in subjects if "DTU" in s])
        LOSO_loop([s for s in subjects if "DAS" in s])

    print("\n========== LOSO COMPLETED — SUMMARIZING ==========\n")
    summarize_results(all_results, cfg, run_dir)
    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()
