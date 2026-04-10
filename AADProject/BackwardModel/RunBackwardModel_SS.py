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
    plot_window_heatmap,
    plot_trf_weights,
)

# ============================================================
# Helpers
# ============================================================

def ensure_2d_env(env):
    env = np.asarray(env, dtype=np.float32)
    if env.ndim == 1:
        return env[:, None]
    return env


def create_lag_matrix(EEG, lags_t, fs):
    """
    EEG: (T, C)
    lags_t: (lag_low_ms, lag_high_ms)
    fs: Hz
    returns X_lagged: (T, n_lags*C)
    """
    EEG = np.asarray(EEG, dtype=np.float32)

    lag_low, lag_high = lags_t
    lags = np.arange(
        int(np.floor(lag_low / 1000 * fs)),
        int(np.ceil(lag_high / 1000 * fs)) + 1
    ).astype(int)

    T, C = EEG.shape
    L = len(lags)

    X = np.zeros((T, L * C), dtype=np.float32)

    for i, lag in enumerate(lags):
        c0 = i * C
        c1 = c0 + C
        if lag > 0:
            X[:-lag, c0:c1] = EEG[lag:, :]
        elif lag < 0:
            d = -lag
            X[d:, c0:c1] = EEG[:-d, :]
        else:
            X[:, c0:c1] = EEG

    return X


def ridge_predict(X, w):
    return X @ w


def mean_corr(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_true.ndim > 1 and y_true.shape[1] > 1:
        vals = []
        for b in range(y_true.shape[1]):
            r = np.corrcoef(y_pred[:, b], y_true[:, b])[0, 1]
            if np.isnan(r):
                r = 0.0
            vals.append(r)
        return float(np.mean(vals))

    r = np.corrcoef(y_pred.ravel(), y_true.ravel())[0, 1]
    if np.isnan(r):
        r = 0.0
    return float(r)


# ============================================================
# Fast windowed correlation
# ============================================================

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
    returns: correlation per window (len = n_windows),
             averaged over bands if multiband
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


# ============================================================
# JSON serialization
# ============================================================

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


# ============================================================
# Subject-specific core (leave-one-trial-out)
# ============================================================

def run_mTRF_subject_specific(nwb_path: str, cfg, window_s_override=None, step_s_override=None):
    fs = float(cfg["preprocessing"]["target_fs"])

    # CLI overrides (if provided), otherwise config defaults
    window_s = float(window_s_override) if window_s_override is not None else float(cfg["backward_model"]["window_s"])
    step_s   = float(step_s_override)   if step_s_override   is not None else float(cfg["backward_model"]["step_s"])

    # robust rounding + avoid zero
    window_len = max(1, int(round(window_s * fs)))
    step = max(1, int(round(step_s * fs)))

    lags = cfg["backward_model"]["lag_ms"]
    sum_subbands = cfg["backward_model"].get("sum_subbands", True)

    print(f"[TARGET] sum_subbands={sum_subbands}")
    print(f"[WINDOW] window_s={window_s} step_s={step_s} -> window_len={window_len} step={step} samples @ fs={fs}")

    # Load data
    data, nwbfile = DataPreparation.Load_data(
        nwb_path,
        merged=True,
        sum_subbands=sum_subbands,
    )

    # Dataset label fallback
    p = str(nwb_path).lower()
    if "dtu" in p:
        dataset_label = "DTU"
    elif "das" in p:
        dataset_label = "DAS"
    else:
        dataset_label = "unknown"

    precomputed = []
    XtX_trials = []
    XtY_trials = []

    for eeg, env_att, env_unatt in data:
        env_att = ensure_2d_env(env_att)
        env_unatt = ensure_2d_env(env_unatt)

        X = create_lag_matrix(eeg, lags, int(fs))
        y = env_att

        XtX = (X.T @ X).astype(np.float64)
        XtY = (X.T @ y).astype(np.float64)

        precomputed.append({
            "X": X,
            "env_att": env_att,
            "env_unatt": env_unatt,
        })
        XtX_trials.append(XtX)
        XtY_trials.append(XtY)

    n_trials = len(precomputed)
    n_features = XtX_trials[0].shape[0]
    Ydim = XtY_trials[0].shape[1]

    XtX_total = np.zeros((n_features, n_features), dtype=np.float64)
    XtY_total = np.zeros((n_features, Ydim), dtype=np.float64)

    for i in range(n_trials):
        XtX_total += XtX_trials[i]
        XtY_total += XtY_trials[i]

    # ========================================================
    # Lambda tuning
    # ========================================================
    val_fraction = 0.2
    n_val = max(1, int(n_trials * val_fraction))
    val_indices = np.arange(n_trials - n_val, n_trials, dtype=int)
    train_indices = np.arange(0, n_trials - n_val, dtype=int)

    print(f"Tuning λ using {len(train_indices)} train and {len(val_indices)} validation trials...")

    XtX_train = np.zeros_like(XtX_total)
    XtY_train = np.zeros_like(XtY_total)
    for i in train_indices:
        XtX_train += XtX_trials[i]
        XtY_train += XtY_trials[i]

    lambdas = np.logspace(0, 6, 7)
    best_lambda = None
    best_score = -np.inf

    eigvals, Q = np.linalg.eigh(XtX_train)
    QtY = Q.T @ XtY_train

    for lam in lambdas:
        w = Q @ (QtY / (eigvals[:, None] + lam))

        scores = []
        for vi in val_indices:
            test = precomputed[vi]
            y_pred = ridge_predict(test["X"], w)

            corr_att = mean_corr(y_pred, test["env_att"])
            corr_un = mean_corr(y_pred, test["env_unatt"])
            scores.append(corr_att - corr_un)

        mean_score = float(np.mean(scores))
        print(f"λ={lam:.1e} | mean Δr={mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam

    print(f"Best λ from validation = {best_lambda:.1e}")

    # ========================================================
    # LOOCV
    # ========================================================
    subject_results = []
    decoder_w = None

    for test_idx in range(n_trials):
        XtX_train_i = XtX_total - XtX_trials[test_idx]
        XtY_train_i = XtY_total - XtY_trials[test_idx]

        w = np.linalg.solve(XtX_train_i + best_lambda * np.eye(n_features), XtY_train_i)
        if decoder_w is None:
            decoder_w = w.copy()

        test = precomputed[test_idx]
        y_pred = ridge_predict(test["X"], w)

        # Full-trial correlation
        corr_att = mean_corr(y_pred, test["env_att"])
        corr_un = mean_corr(y_pred, test["env_unatt"])
        correct_full = bool(corr_att > corr_un)

        # Windowed evaluation
        if Ydim > 1:
            cors_att = windowed_corr_fast(y_pred, test["env_att"], window_len, step)
            cors_un = windowed_corr_fast(y_pred, test["env_unatt"], window_len, step)
        else:
            cors_att = windowed_corr_fast(y_pred.ravel(), test["env_att"].ravel(), window_len, step)
            cors_un = windowed_corr_fast(y_pred.ravel(), test["env_unatt"].ravel(), window_len, step)

        correct_vec = cors_att > cors_un
        window_acc = float(correct_vec.mean()) if correct_vec.size > 0 else 0.0

        starts = np.arange(0, test["X"].shape[0] - window_len + 1, step, dtype=int)
        n_windows = int(len(starts))

        windows = [
            {"start": float(st / fs), "end": float((st + window_len) / fs), "correct": bool(c)}
            for st, c in zip(starts, correct_vec)
        ]

        subject_results.append({
            "trial": int(test_idx + 1),
            "corr_att": corr_att,
            "corr_unatt": corr_un,
            "correct_full": bool(correct_full),
            "window_accuracy": window_acc,
            "n_windows": n_windows,
            "windows": windows,
        })

        print(
            f"Trial {test_idx + 1}/{n_trials}: r_att={corr_att:.3f} | r_unatt={corr_un:.3f} | "
            f"full_correct={correct_full} | win_acc={window_acc:.2f} | n_windows={n_windows}"
        )

    subj_acc = float(np.mean([r["correct_full"] for r in subject_results]))
    subj_win_acc = float(np.mean([r["window_accuracy"] for r in subject_results]))
    subject_label = os.path.basename(str(nwb_path)).replace(".nwb", "")

    print(f"\nSubject — Full-trial acc: {subj_acc:.2f}, Windowed acc: {subj_win_acc:.2f}")

    return {
        "subject_id": getattr(nwbfile, "identifier", subject_label),
        "subject_label": subject_label,
        "dataset": dataset_label,
        "results": subject_results,
        "full_accuracy": subj_acc,
        "window_accuracy": subj_win_acc,
        "decoder_w": decoder_w,
        "lags": lags,
        "fs": fs,
        "n_channels": data[0][0].shape[1],

        # record actual window used
        "window_s": window_s,
        "step_s": step_s,
        "window_len_samples": int(window_len),
        "step_samples": int(step),

        # record target format used
        "sum_subbands": sum_subbands,
    }


# ============================================================
# Merge + summarize
# ============================================================

def merge_subject_jsons(run_dir):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("json/*.json"))
    results = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


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
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
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

        # use window_s from results, fallback to cfg
        window_s = group_results[0].get("window_s", cfg["backward_model"]["window_s"])

        plot_subject_bars(
            full_accs,
            win_accs,
            subjects_lbl,
            os.path.join(group_dir, "Accuracy"),
            mode="Subject-specific",
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-subject", type=str, default=None,
                        help="Run SS for exactly this subject name (must match config subjects list).")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Existing run directory. If not provided, a new one is created.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge per-subject JSONs in --run-dir and run summarize_results.")

    parser.add_argument("--window-s", type=float, default=None,
                        help="Override backward_model.window_s (seconds).")
    parser.add_argument("--step-s", type=float, default=None,
                        help="Override backward_model.step_s (seconds).")

    args = parser.parse_args()

    cfg = paths.load_config()
    subjects = cfg["subjects"]["all"]

    print("\n========== RUNNING SS BACKWARD MODEL ==========")
    print(f"Subjects: {subjects}")
    print(f"sum_subbands: {cfg['backward_model'].get('sum_subbands', True)}")
    print("==============================================\n")

    SS_base = paths.RESULTS_LIN / "SS"
    if args.run_dir is None:
        run_dir = paths.get_next_run_dir(SS_base)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Saving SS results to: {run_dir}")
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

    # Single-subject mode
    if args.single_subject is not None:
        test_subj = args.single_subject
        if test_subj not in subjects:
            raise ValueError(f"--single-subject {test_subj} not found in config subjects list.")

        subj_path = paths.subject_eegPP(test_subj)
        res = run_mTRF_subject_specific(
            subj_path, cfg,
            window_s_override=args.window_s,
            step_s_override=args.step_s,
        )

        json_dir = os.path.join(run_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        subject_json = os.path.join(json_dir, f"{test_subj}.json")

        with open(subject_json, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(res), f, indent=2)

        print(f"✓ Finished subject {test_subj}")
        print(f"✓ Saved: {subject_json}")
        return

    # Sequential fallback
    all_results = []
    for test_subj in subjects:
        subj_path = paths.subject_eegPP(test_subj)
        res = run_mTRF_subject_specific(
            subj_path, cfg,
            window_s_override=args.window_s,
            step_s_override=args.step_s,
        )
        all_results.append(res)

    summarize_results(all_results, cfg, run_dir)
    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()