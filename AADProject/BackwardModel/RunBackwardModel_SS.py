import csv
import json
import os

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


def create_lag_matrix(EEG, lags_t, fs):
    """
    input:
        EEG: (t,channels)
        lags_t: parameter in config file
        fs: sample frequency Hz
    output: X_lagged: (t,n_features)=(t,len(lags_t)*fs*channels)
    """

    lag_low,lag_high=lags_t

    lags = np.arange(  # make a vector of lags going from lag_low to lag_high
        int(np.floor(lag_low / 1000 * fs)),
        int(np.ceil(lag_high / 1000 * fs)) + 1
    )

    n_samples_t, n_ch = EEG.shape
    lags = np.asarray(lags, dtype=int)
    n_lags = len(lags)

    X_lagged = np.zeros((n_samples_t, n_lags * n_ch), dtype=EEG.dtype)

    for i, lag in enumerate(lags):
        col_start = i * n_ch
        col_end = col_start + n_ch

        if lag > 0:
            # X_lagged[t] = EEG[t + lag], last 'lag' rows stay 0
            X_lagged[:-lag, col_start:col_end] = EEG[lag:, :]

        elif lag < 0:
            d = -lag
            # X_lagged[t] = EEG[t - d], first 'd' rows stay 0
            X_lagged[d:, col_start:col_end] = EEG[:-d, :]

        else:  # lag == 0
            X_lagged[:, col_start:col_end] = EEG

    return X_lagged


def ridge_fit(X, y, alpha):
    """Closed-form ridge regression"""
    XtX = X.T @ X
    XtY = X.T @ y
    n_features = XtX.shape[0]
    w = np.linalg.solve(XtX + alpha * np.eye(n_features), XtY)
    return w


def ridge_predict(X, w):
    """Apply learned weights."""
    return X @ w


def run_mTRF(nwb_path: str, cfg):

    # == load config ==
    fs = cfg["preprocessing"]["target_fs"]
    window_len = int(cfg["backward_model"]["window_s"]* fs)
    step = int(cfg["backward_model"]["step_s"] * fs)
    lags = cfg["backward_model"]["lag_ms"]
    multiband=cfg["backward_model"]["multiband"]


    # == prepare data for model ==
    data,nwbfile =DataPreparation.Load_data(nwb_path,merged=True,multiband=True)


    # == Precompute lagged matrices ==
    precomputed = []
    for eeg, env_att, env_unatt in data:
        X_lagged = create_lag_matrix(eeg, lags,fs)
        precomputed.append({
            "X": X_lagged,
            "env_att": env_att,
            "env_unatt": env_unatt,
        })


    # == Tune λ ==
    val_fraction = 0.2
    n_trials = len(data)
    n_val = max(1, int(n_trials * val_fraction))

    lambdas = np.logspace(0, 6, 7)     #generates 7 values between 0 and 10^7, spaced logarithmically.
    best_lambda = None
    best_score = -np.inf    #negative infinity, so the first value is always better

    val_indices = np.arange(n_trials - n_val, n_trials)
    train_indices = np.arange(0, n_trials - n_val)
    print(f"Tuning λ using {len(train_indices)} train and {len(val_indices)} validation trials...")

    # Build training data
    X_train = np.vstack([precomputed[i]["X"] for i in train_indices])
    if multiband:
        y_train = np.vstack([precomputed[i]["env_att"] for i in train_indices])
    else:
        y_train = np.hstack([precomputed[i]["env_att"] for i in train_indices])

    # Precompute for reuse across λ's
    XtX = X_train.T @ X_train
    XtY = X_train.T @ y_train
    n_features = XtX.shape[0]

    # Iterate over lambda
    for lam in lambdas:
        # train
        w = np.linalg.solve(XtX + lam * np.eye(n_features), XtY)

        # test
        scores = []
        for vi in val_indices:
            test = precomputed[vi]
            y_pred_att = ridge_predict(test["X"], w)
            if multiband:
                corrs = [np.corrcoef(y_pred_att[:, b], test["env_att"][:, b])[0, 1]
                         for b in range(y_pred_att.shape[1])]
                corr_att = np.mean(corrs)
            else:
                corr_att = np.corrcoef(y_pred_att, test["env_att"])[0, 1]

            scores.append(corr_att)

        mean_score = np.mean(scores)
        print(f"λ={lam:.1e} | mean Δr={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam

    print(f" Best λ from validation = {best_lambda:.1e}")


    # == Leave-one-trial-out cross-validation ==
    subject_results = []

    for test_idx in range(len(precomputed)):

        # Train on all trials except the test one
        X_train = np.vstack([precomputed[i]["X"]
                             for i in range(len(precomputed)) if i != test_idx])
        if multiband:
            y_train = np.vstack([precomputed[i]["env_att"] for i in range(len(precomputed)) if i != test_idx])
        else:
            y_train = np.hstack([precomputed[i]["env_att"]
                             for i in range(len(precomputed)) if i != test_idx])

        w = ridge_fit(X_train, y_train, best_lambda or 0.0)   # <-- uses helper

        # Test trial
        test = precomputed[test_idx]

        if test_idx == 0:  # save TRF from first fold (they barely vary)
            decoder_w = w.copy()

        # Full-trial evaluation
        y_pred_att = ridge_predict(test["X"], w)
        if multiband:
            corrs = [np.corrcoef(y_pred_att[:, b], test["env_att"][:, b])[0, 1]
                     for b in range(y_pred_att.shape[1])]
            corr_att = np.mean(corrs)
            corrs = [np.corrcoef(y_pred_att[:, b], test["env_unatt"][:, b])[0, 1]
                     for b in range(y_pred_att.shape[1])]
            corr_unatt = np.mean(corrs)
        else:
            corr_att = np.corrcoef(y_pred_att, test["env_att"])[0, 1]
            corr_unatt = np.corrcoef(y_pred_att, test["env_unatt"])[0, 1]

        correct_full = bool(corr_att > corr_unatt)

        # Windowed evaluation
        n_samples = test["X"].shape[0]
        window_results = []
        for start in range(0, n_samples - window_len + 1, step):
            end = start + window_len
            X_win = test["X"][start:end]
            env_att_win = test["env_att"][start:end]
            env_unatt_win = test["env_unatt"][start:end]

            y_pred_win = ridge_predict(X_win, w)
            if multiband:
                w_corrs = [np.corrcoef(y_pred_win[:, b], env_att_win[:, b])[0, 1]
                         for b in range(y_pred_win.shape[1])]
                w_corr_att = np.mean(w_corrs)
                w_corrs = [np.corrcoef(y_pred_win[:, b], env_unatt_win[:, b])[0, 1]
                         for b in range(y_pred_win.shape[1])]
                w_corr_unatt = np.mean(w_corrs)
            else:
                w_corr_att = np.corrcoef(y_pred_win, env_att_win)[0, 1]
                w_corr_unatt = np.corrcoef(y_pred_win, env_unatt_win)[0, 1]

            window_results.append({
                "start": start / fs,
                "end": end / fs,
                "corr_att": w_corr_att,
                "corr_unatt": w_corr_unatt,
                "correct": bool(w_corr_att > w_corr_unatt)
            })


        window_acc = np.mean([w["correct"] for w in window_results])

        subject_results.append({
            "trial": test_idx + 1,
            "corr_att": corr_att,
            "corr_unatt": corr_unatt,
            "correct_full": bool(correct_full),
            "window_accuracy": window_acc,
            "windows": window_results
        })


        print(f"Trial {test_idx+1}/{len(data)}: "
              f"r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | "
              f"full_correct={correct_full} | win_acc={window_acc:.2f}")

    # --- summary ---
    subj_acc = np.mean([r["correct_full"] for r in subject_results])
    subj_win_acc = np.mean([r["window_accuracy"] for r in subject_results])

    print(f"\n Subject — Full-trial acc: {subj_acc:.2f}, Windowed acc: {subj_win_acc:.2f}")

    return {
        "subject_id": nwbfile.identifier,
        "results": subject_results,
        "full_accuracy": subj_acc,
        "window_accuracy": subj_win_acc,
        "decoder_w": decoder_w,
        "lags": lags,
        "fs": fs,
        "n_channels": data[0][0].shape[1],
    }


def main():
    cfg = paths.load_config()
    subjects = cfg["subjects"]["all"]

    # Create an automatic unique run directory
    SS_base = paths.RESULTS_LIN / "SS"
    run_dir = paths.get_next_run_dir(SS_base)
    print(f"Saving SS results to: {run_dir}")

    # Save config snapshot
    paths.save_config_copy(cfg, run_dir)

    all_results = []

    for subject_id in subjects:
        subject_file = paths.subject_eegPP(subject_id)
        os.makedirs(paths.RESULTS_LIN, exist_ok=True)

        # == Run backward model ==
        subj_results = run_mTRF(subject_file, cfg)
        all_results.append(subj_results)

    # === Summarize results ===
    summarize_results(all_results, cfg, run_dir)

def make_json_serializable(obj):
    """Recursively convert numpy arrays to lists for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


def summarize_results(results, cfg, run_dir):
    """Aggregate and save all results."""
    if not results:
        print("No usable results, skipping summary.")
        return


    # Aggregate accuracies
    full_accs = [r["full_accuracy"] for r in results]
    win_accs = [r["window_accuracy"] for r in results]

    print("\n=== Group Results ===")
    print(f"Mean full-trial accuracy: {np.mean(full_accs):.2f}")
    print(f"Mean windowed accuracy:  {np.mean(win_accs):.2f}")

    # JSON export
    json_path = os.path.join(run_dir, "mTRF_results.json")
    serializable_results = make_json_serializable(results)
    with open(json_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Saved results to {json_path}")

    # CSV summary
    csv_path = os.path.join(run_dir, "mTRF_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject_ID", "Full_Trial_Accuracy", "Windowed_Accuracy"])
        for subj in results:
            writer.writerow([subj["subject_id"], subj["full_accuracy"], subj["window_accuracy"]])
    print(f"Per-subject summary saved to {csv_path}")

    # plotting histograms
    all_att = np.concatenate([np.array([t["corr_att"] for t in subj["results"]]) for subj in results])
    all_unatt = np.concatenate([np.array([t["corr_unatt"] for t in subj["results"]]) for subj in results])
    stats = summaryStats.SummaryStats(all_att, all_unatt)

    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    summaryStats.plot_histograms(all_att, all_unatt, os.path.join(run_dir, "Correlations"))

    subjects = [r["subject_id"] for r in results]
    full_accs = [r["full_accuracy"] for r in results]
    win_accs = [r["window_accuracy"] for r in results]


    # == extra plots ==
    # 1. Accuracy bars
    plot_subject_bars(full_accs, win_accs, subjects, os.path.join(run_dir, "Accuracy"))


    # 2. Accuracy vs. window length
    plot_window_length_curve(results,
                             cfg["backward_model"]["window_s"],
                             os.path.join(run_dir, "WindowLengthCurve"))

    # 3. TRF temporal profile (use first subject)
    first = results[0]
    lags_low, lags_high = first["lags"]
    lag_samples = np.arange(lags_low, lags_high + 1)
    plot_trf_weights(
        first["decoder_w"],
        first["n_channels"],
        lag_samples,
        os.path.join(run_dir, "TRF"))

    # 4. Per-subject heatmaps
    for subj in results:
        plot_window_heatmap(subj, subj["subject_id"],os.path.join(run_dir, f"Heatmap_{subj['subject_id']}"))

if __name__ == '__main__':
    main()
