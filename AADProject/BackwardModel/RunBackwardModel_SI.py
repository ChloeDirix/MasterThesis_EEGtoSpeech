#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
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
from scipy.stats import zscore

# =========================
# Helpers: normalization
# =========================

class Standardizer:
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.n = 0
        self._sum = None
        self._sumsq = None

    def fit(self, X):
        self.mu = np.mean(X, axis=0, keepdims=True)
        self.sigma = np.std(X, axis=0, keepdims=True)
        return self

    def partial_fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]

        sx = np.sum(X, axis=0, keepdims=True)
        sxx = np.sum(X * X, axis=0, keepdims=True)

        if self._sum is None:
            self._sum = sx
            self._sumsq = sxx
        else:
            self._sum += sx
            self._sumsq += sxx

        self.n += X.shape[0]
        return self

    def finalize(self):
        if self.n == 0:
            raise ValueError("Standardizer.finalize() called with no data.")
        self.mu = self._sum / self.n
        var = self._sumsq / self.n - self.mu * self.mu
        var[var < 1e-8] = 1.0
        self.sigma = np.sqrt(var)
        return self

    def transform(self, X):
        return (X - self.mu) / self.sigma

#other option zscore per trial
def zscore_trial(X):
    """
    Per-trial z-scoring along time axis.
    X: (T, D) or (T,)
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 1:
        X = X[:, None]
        squeeze_back = True
    else:
        squeeze_back = False

    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)

    Xz = (X - mu) / sigma
    return Xz[:, 0] if squeeze_back else Xz

# =========================
# Lag matrix + ridge
# =========================

def create_lag_matrix(EEG, lags_t, fs):
    """
    EEG: (T, channels)
    lags_t: (lag_low_ms, lag_high_ms)
    fs: Hz
    returns X_lagged: (T, n_lags*channels)
    """
    EEG = np.asarray(EEG, dtype=np.float32)

    lag_low, lag_high = lags_t

    lags = np.arange(
        int(np.floor(lag_low / 1000 * fs)),
        int(np.ceil(lag_high / 1000 * fs)) + 1
    ).astype(int)

    n_samples_t, n_ch = EEG.shape
    n_lags = len(lags)

    X_lagged = np.zeros((n_samples_t, n_lags * n_ch), dtype=np.float32)

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
# Data loading helper
# =========================
def load_subject_block(nwb_path, lags, fs, sum_subbands, eeg_std=None, y_std=None, norm_mode="global"):
    """
    Loads all trials for 1 subject and returns stacked arrays:
      X: (T_total, n_features)
      y: (T_total, Ydim)

    norm_mode:
      - "global": use fitted Standardizer objects
      - "per_trial": z-score each trial independently
      - "none": no normalization
    """

    data_subj, _ = DataPreparation.Load_data(
        nwb_path,
        merged=False,
        sum_subbands=sum_subbands,
    )

    X_list = []
    y_list = []

    for eeg, env_att, _ in data_subj:
        y = np.asarray(env_att, dtype=np.float32)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if norm_mode == "global":
            if eeg_std is not None:
                eeg = eeg_std.transform(eeg)
            if y_std is not None:
                y = y_std.transform(y)

        elif norm_mode == "per_trial":
            eeg = zscore_trial(eeg)
            y = zscore_trial(y)

        elif norm_mode == "none":
            pass

        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        X_list.append(create_lag_matrix(eeg, lags, fs))
        y_list.append(y.astype(np.float32))

    return np.vstack(X_list), np.vstack(y_list)

def prepare_target(env, sum_subbands=True):
    """
    Convert target envelope to shape (T, 1) if summing subbands,
    otherwise keep multiband shape (T, B).

    env: (T,), (T,1), or (T,B)
    """
    env = np.asarray(env, dtype=np.float32)

    if env.ndim == 1:
        return env.reshape(-1, 1)

    if sum_subbands:
        return np.sum(env, axis=1, keepdims=True)

    return env
 
def subject_stats_or_block(nwb_path, lags, fs, sum_subbands, eeg_std, y_std, keep_block=False, norm_mode="global"):
    Xs, ys = load_subject_block(
        nwb_path,
        lags,
        fs,
        sum_subbands,
        eeg_std,
        y_std,
        norm_mode,
    )

    XtX = (Xs.T @ Xs).astype(np.float64)
    XtY = (Xs.T @ ys).astype(np.float64)

    if keep_block:
        return XtX, XtY, Xs, ys
    return XtX, XtY, None, None


def mean_corr(y_pred, y_true):
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        r = np.mean([np.corrcoef(y_pred[:, b], y_true[:, b])[0, 1] for b in range(y_true.shape[1])])
    else:
        r = np.corrcoef(y_pred.ravel(), y_true.ravel())[0, 1]

    if np.isnan(r):
        r = 0.0
    return float(r)


# =========================
# Core: one LOSO fold
# =========================

def run_mTRF_subject(train_paths, test_path, cfg, window_s_override=None, step_s_override=None, norm_mode="global"):
    fs = float(cfg["preprocessing"]["target_fs"])


    window_s = float(window_s_override) if window_s_override is not None else float(cfg["backward_model"]["window_s"])
    step_s   = float(step_s_override)   if step_s_override   is not None else float(cfg["backward_model"]["step_s"])

    window_len = max(1, int(round(window_s * fs)))
    step = max(1, int(round(step_s * fs)))

    lags = cfg["backward_model"]["lag_ms"]

    sum_subbands = cfg["backward_model"].get("sum_subbands", True)

    print(f"[NORMALIZATION] norm_mode={norm_mode}")
    print(f"[TARGET] sum_subbands={sum_subbands}")
    print(f"[WINDOW] window_s={window_s} step_s={step_s} -> window_len={window_len} step={step} samples @ fs={fs}")
    print(f"Loading {len(train_paths)} training subjects...")

    # ----- train/val split for λ tuning -----
    val_fraction = 0.2
    n_subjects = len(train_paths)
    n_val = max(1, int(n_subjects * val_fraction))
    val_paths = train_paths[-n_val:]
    tr_paths = train_paths[:-n_val]
    if len(tr_paths) == 0:
        raise ValueError("Not enough training subjects after train/val split.")

    # ----- fit standardizers on TRAIN subjects only -----
    eeg_std = None
    y_std = None
    if norm_mode == "global":
        eeg_std = Standardizer()
        y_std = Standardizer()

        for p in tr_paths:
            data_subj, _ = DataPreparation.Load_data(
                p,
                merged=False,
                sum_subbands=sum_subbands,
            )

            for eeg, env_att, _ in data_subj:
                eeg_std.partial_fit(eeg)

                y = np.asarray(env_att, dtype=np.float32)
                if y.ndim == 1:
                    y = y.reshape(-1, 1)

                y_std.partial_fit(y)

        eeg_std.finalize()
        y_std.finalize()

    # ----- precompute XtX, XtY on training (excluding val) -----
    train_stats = []
    n_features = None
    Ydim = None

    for p in tr_paths:
        XtX_sub, XtY_sub, _, _ = subject_stats_or_block(
            p, lags, int(fs), sum_subbands, eeg_std, y_std, keep_block=False, norm_mode=norm_mode
        )
        train_stats.append((XtX_sub, XtY_sub))

        if n_features is None:
            n_features = XtX_sub.shape[0]
            Ydim = XtY_sub.shape[1]

    XtX = np.zeros((n_features, n_features), dtype=np.float64)
    XtY = np.zeros((n_features, Ydim), dtype=np.float64)
    for XtX_sub, XtY_sub in train_stats:
        XtX += XtX_sub
        XtY += XtY_sub

    # ----- preload validation blocks -----
    print("Tuning λ ...")
    val_blocks = []
    for p in val_paths:
        _, _, Xv, yv = subject_stats_or_block(
            p, lags, int(fs), sum_subbands, eeg_std, y_std, keep_block=True, norm_mode=norm_mode
        )
        val_blocks.append((Xv, yv))

    # ----- λ tuning via eigendecomposition -----
    lambdas = np.logspace(0, 6, 7)
    best_lambda = None
    best_score = -np.inf

    eigvals, Q = np.linalg.eigh(XtX)
    QtY = Q.T @ XtY

    for lam in lambdas:
        w = Q @ (QtY / (eigvals[:, None] + lam))

        scores = []
        for Xv, yv in val_blocks:
            y_pred = ridge_predict(Xv, w)
            scores.append(mean_corr(y_pred, yv))

        mean_score = float(np.mean(scores))
        print(f"λ={lam:.1e}  | mean r={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam

    print(f"Best λ = {best_lambda}")

    # ----- train on ALL training subjects -----
    XtX = np.zeros((n_features, n_features), dtype=np.float64)
    XtY = np.zeros((n_features, Ydim), dtype=np.float64)

    for p in train_paths:
        XtX_sub, XtY_sub, _, _ = subject_stats_or_block(
            p, lags, int(fs), sum_subbands, eeg_std, y_std, keep_block=False, norm_mode=norm_mode
        )
        XtX += XtX_sub
        XtY += XtY_sub

    w = np.linalg.solve(XtX + best_lambda * np.eye(n_features), XtY)
    decoder_w = w.copy()

    # ----- dataset label -----
    dataset_label = "unknown"
    pth = str(test_path).lower()
    if "dtu" in pth:
        dataset_label = "DTU"
    elif "das" in pth:
        dataset_label = "DAS"

    # ----- evaluate on test subject -----
    test_data, test_meta, test_nwb = DataPreparation.Load_data(
        test_path,
        merged=False,
        sum_subbands=sum_subbands,
        return_meta=True,
    )

    results = []
    for ti, (eeg, env_att, env_unatt) in enumerate(test_data):

        y_att = np.asarray(env_att, dtype=np.float32)
        y_un  = np.asarray(env_unatt, dtype=np.float32)

        if y_att.ndim == 1:
            y_att = y_att.reshape(-1, 1)
        if y_un.ndim == 1:
            y_un = y_un.reshape(-1, 1)

        if norm_mode == "global":
            eeg = eeg_std.transform(eeg)
            y_att = y_std.transform(y_att)
            y_un  = y_std.transform(y_un)

        elif norm_mode == "per_trial":
            eeg = zscore_trial(eeg)
            y_att = zscore_trial(y_att)
            y_un  = zscore_trial(y_un)

        elif norm_mode == "none":
            pass

        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        X = create_lag_matrix(eeg, lags, int(fs))
        y_pred = X @ w

        cond = int(test_meta[ti].get("acoustic_condition", -1)) if isinstance(test_meta, list) else -1

        if Ydim > 1:
            corr_att = float(np.mean([np.corrcoef(y_pred[:, b], y_att[:, b])[0, 1] for b in range(Ydim)]))
            corr_un  = float(np.mean([np.corrcoef(y_pred[:, b], y_un[:, b])[0, 1] for b in range(Ydim)]))
        else:
            corr_att = float(np.corrcoef(y_pred.ravel(), y_att.ravel())[0, 1])
            corr_un  = float(np.corrcoef(y_pred.ravel(), y_un.ravel())[0, 1])

        if np.isnan(corr_att):
            corr_att = 0.0
        if np.isnan(corr_un):
            corr_un = 0.0

        correct_full = bool(corr_att > corr_un)

        if Ydim > 1:
            cors_att = windowed_corr_fast(y_pred, y_att, window_len, step)
            cors_un  = windowed_corr_fast(y_pred, y_un,  window_len, step)
        else:
            cors_att = windowed_corr_fast(y_pred.ravel(), y_att.ravel(), window_len, step)
            cors_un  = windowed_corr_fast(y_pred.ravel(), y_un.ravel(),  window_len, step)

        correct_vec = cors_att > cors_un
        window_acc = float(np.mean(correct_vec)) if correct_vec.size > 0 else 0.0

        starts = np.arange(0, X.shape[0] - window_len + 1, step, dtype=int)
        n_windows = int(len(starts))

        win_list = [
            {"start": float(st / fs), "end": float((st + window_len) / fs), "correct": bool(c)}
            for st, c in zip(starts, correct_vec)
        ]

        results.append({
            "trial": ti + 1,
            "acoustic_condition": cond,
            "corr_att": corr_att,
            "corr_unatt": corr_un,
            "correct_full": correct_full,
            "window_accuracy": window_acc,
            "n_windows": n_windows,
            "windows": win_list,
        })

        print(
            f"Trial {ti + 1}/{len(test_data)} | cond={cond}: "
            f"r_att={corr_att:.3f} | r_unatt={corr_un:.3f} | {correct_full} | "
            f"win_acc={window_acc:.2f} | n_windows={n_windows}"
        )

    full_acc = float(np.mean([r["correct_full"] for r in results]))
    win_acc  = float(np.mean([r["window_accuracy"] for r in results]))

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
        "window_s": window_s,
        "step_s": step_s,
        "window_len_samples": int(window_len),
        "step_samples": int(step),
        "norm_mode": norm_mode,
        "sum_subbands": sum_subbands,
    }

# =========================
# JSON serialization
# =========================

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
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(group_results), f, indent=2)

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

            writer.writerow([f"MEAN_{group_name}", group_name, float(np.mean(full_accs)), float(np.mean(win_accs))])

            all_full = [r["full_accuracy"] for r in results]
            all_win = [r["window_accuracy"] for r in results]
            writer.writerow(["MEAN_ALL", "ALL", float(np.mean(all_full)), float(np.mean(all_win))])

        all_att = np.concatenate([np.array([t["corr_att"] for t in subj["results"]]) for subj in group_results])
        all_unatt = np.concatenate([np.array([t["corr_unatt"] for t in subj["results"]]) for subj in group_results])
        stats = summaryStats.SummaryStats(all_att, all_unatt)

        stats_path = os.path.join(group_dir, f"stats_{group_name}.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        summaryStats.plot_histograms(all_att, all_unatt, os.path.join(group_dir, "Correlations"))

        subjects_lbl = [subj.get("subject_label", subj["subject_id"]) for subj in group_results]
        window_s = group_results[0].get("window_s", cfg["backward_model"]["window_s"])

        plot_subject_bars(
            full_accs, win_accs, subjects_lbl,
            os.path.join(group_dir, "Accuracy"),
            mode="Subject-independent",
            window_s=window_s,
        )

    save_group(results, "ALL", run_dir)

    grouped = {}
    for r in results:
        ds = str(r.get("dataset", "unknown"))
        grouped.setdefault(ds, []).append(r)

    if len(grouped) > 1:
        for ds, group_results in grouped.items():
            save_group(group_results, ds, os.path.join(run_dir, ds))


# =========================
# Merge helper for array mode
# =========================

def merge_subject_jsons(run_dir):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("json/*.json"))
    results = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


# =========================
# Mode helpers
# =========================

def _dataset_of(subj_name: str) -> str:
    s = subj_name.lower()
    if "dtu" in s:
        return "DTU"
    if "das" in s:
        return "DAS"
    return "unknown"


def _select_subjects_by_mode(all_subjects, test_subj, mode):
    test_ds = _dataset_of(test_subj)

    if mode == "separate":
        train_subjects = [s for s in all_subjects if _dataset_of(s) == test_ds and s != test_subj]
    elif mode == "mixed":
        train_subjects = [s for s in all_subjects if s != test_subj]
    else:
        raise ValueError(f"Unknown SI mode: {mode}")

    if len(train_subjects) == 0:
        raise ValueError(f"No training subjects selected for test subject {test_subj} (mode={mode}).")

    return train_subjects


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-subject", type=str, default=None,
                        help="Run LOSO for exactly this subject name (must match config subjects list).")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Existing run directory. If not provided, a new one is created.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge per-subject JSONs in --run-dir and run summarize_results.")

    parser.add_argument("--mode", type=str, default=None, choices=["mixed", "separate"],
                        help="mixed=all except test, separate=same dataset only. If omitted, cfg SI_mode or 'mixed'.")

    # window overrides
    parser.add_argument("--window-s", type=float, default=None,
                        help="Override backward_model.window_s (seconds).")
    parser.add_argument("--step-s", type=float, default=None,
                        help="Override backward_model.step_s (seconds).")

    args = parser.parse_args()

    cfg = paths.load_config()

    use_subjects = cfg.get("use_subjects", "all")
    if use_subjects not in cfg["subjects"]:
        raise ValueError(
            f"Unknown use_subjects='{use_subjects}'. "
            f"Available: {list(cfg['subjects'].keys())}"
        )

    subjects = cfg["subjects"][use_subjects]

    mode = args.mode
    if mode is None:
        mode = cfg.get("SI_mode", {}).get("mode", "mixed")
    mode = mode.lower()

    norm_mode=cfg["data"]["norm_mode"]

    if mode not in ["mixed", "separate"]:
        raise ValueError(f"Unknown mode '{mode}'. Use 'mixed' or 'separate'.")
    print("\n========== RUNNING LOSO BACKWARD MODEL ==========")
    print(f"use_subjects: {use_subjects}")
    print(f"Mode: {mode}")
    print(f"N selected subjects: {len(subjects)}")
    print(f"Subjects: {subjects}")
    print(f"Normalization: {norm_mode}")
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

    def run_one_subject(test_subj):
        test_path = paths.subject_eegPP(test_subj)
        train_subjects = _select_subjects_by_mode(subjects, test_subj, mode)
        train_paths = [paths.subject_eegPP(s) for s in train_subjects]

        print(f"Test subject: {test_subj} ({_dataset_of(test_subj)})")
        print(f"Training on {len(train_subjects)} subjects: "
              f"{train_subjects[:5]}{' ...' if len(train_subjects) > 5 else ''}")

        return run_mTRF_subject(
            train_paths, test_path, cfg,
            window_s_override=args.window_s,
            step_s_override=args.step_s,
            norm_mode=norm_mode,
        )

    if args.single_subject is not None:
        res = run_one_subject(args.single_subject)

        json_dir = os.path.join(run_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        subject_json = os.path.join(json_dir, f"{args.single_subject}.json")
        with open(subject_json, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(res), f, indent=2)

        print(f"✓ Finished subject {args.single_subject}")
        print(f"✓ Saved: {subject_json}")
        return

    all_results = []
    for i, test_subj in enumerate(subjects, start=1):
        print(f"\n===============================================")
        print(f"LOSO {i}/{len(subjects)} — Test subject: {test_subj}")
        print("===============================================")

        res = run_one_subject(test_subj)
        all_results.append(res)

        tmp_json = os.path.join(run_dir, f"mTRF_partial_{i}.json")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(all_results), f, indent=2)

        print(f"✓ Finished subject {test_subj}")
        print(f"✓ Partial results saved to: {tmp_json}\n")

    print("\n========== LOSO COMPLETED — SUMMARIZING ==========\n")
    summarize_results(all_results, cfg, run_dir)
    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()