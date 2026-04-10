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

def ensure_env_shape(env, multiband: bool):
    env = np.asarray(env)
    if multiband:
        return env[:, None] if env.ndim == 1 else env
    else:
        return env[:, 0] if env.ndim == 2 else env


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
    data,nwbfile =DataPreparation.Load_data(nwb_path,merged=True,multiband=multiband)

    # -------- determine dataset label--------
    dataset_label = "unknown"
    try:
        trials_df = nwbfile.trials.to_dataframe()
        if "dataset" in trials_df.columns and len(trials_df) > 0:
            ds_set = set(trials_df["dataset"].astype(str).tolist())
            if len(ds_set) == 1:
                dataset_label = next(iter(ds_set))
            elif len(ds_set) > 1:
                dataset_label = "mixed"
    except Exception:
        # fallback: infer from filename
        p = str(nwb_path).lower()
        if "dtu" in p:
            dataset_label = "DTU"
        elif "das" in p:
            dataset_label = "DAS"


    # == Precompute lagged matrices ==
    precomputed = []
    for eeg, env_att, env_unatt in data:
        env_att = ensure_env_shape(env_att, multiband)
        env_unatt = ensure_env_shape(env_unatt, multiband)

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

                corrs = [np.corrcoef(y_pred_att[:, b], test["env_unatt"][:, b])[0, 1]
                        for b in range(y_pred_att.shape[1])]
                corr_unatt = np.mean(corrs)
            else:
                corr_att = np.corrcoef(y_pred_att, test["env_att"])[0, 1]
                corr_unatt = np.corrcoef(y_pred_att, test["env_unatt"])[0, 1]

            scores.append(corr_att - corr_unatt)

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

    # --- subject label (full ID, e.g. S1_DAS) ---
    subject_label = os.path.basename(str(nwb_path))
    subject_label = subject_label.replace(".nwb", "")

    return {
        "subject_id": nwbfile.identifier,  #eg S1
        "subject_label": subject_label,    #eg S1_DAS
        "dataset": dataset_label,
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
    print(cfg["subjects"])
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

    def save_group(group_results, group_name, group_dir):
        os.makedirs(group_dir, exist_ok=True)

        # Aggregate accuracies
        full_accs = [r["full_accuracy"] for r in group_results]
        win_accs = [r["window_accuracy"] for r in group_results]

        print(f"\n=== Results: {group_name} ===")
        print(f"N subjects: {len(group_results)}")
        print(f"Mean full-trial accuracy: {np.mean(full_accs):.2f}")
        print(f"Mean windowed accuracy:  {np.mean(win_accs):.2f}")

        # JSON export
        json_path = os.path.join(group_dir, f"mTRF_results_{group_name}.json")
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(group_results), f, indent=4)
        print(f"Saved results to {json_path}")

        # CSV summary
        csv_path = os.path.join(group_dir, f"mTRF_summary_{group_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Subject_ID", "Dataset", "Full_Trial_Accuracy", "Windowed_Accuracy"])
            
            #per subject
            for subj in group_results:
                writer.writerow([subj.get("subject_label", subj["subject_id"]), subj.get("dataset", "unknown"),
                                 subj["full_accuracy"], subj["window_accuracy"]])
            #average
            writer.writerow([
                f"MEAN_{group_name}",
                group_name,
                float(np.mean(full_accs)),
                float(np.mean(win_accs)),
            ])

            #average all datasets
            all_full = [r["full_accuracy"] for r in results]
            all_win = [r["window_accuracy"] for r in results]
            writer.writerow([
                "MEAN_ALL",
                "ALL",
                float(np.mean(all_full)),
                float(np.mean(all_win)),
            ])

        print(f"Per-subject summary saved to {csv_path}")

        # Histograms/stats
        all_att = np.concatenate([np.array([t["corr_att"] for t in subj["results"]]) for subj in group_results])
        all_unatt = np.concatenate([np.array([t["corr_unatt"] for t in subj["results"]]) for subj in group_results])
        stats = summaryStats.SummaryStats(all_att, all_unatt)

        stats_path = os.path.join(group_dir, f"stats_{group_name}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

        summaryStats.plot_histograms(all_att, all_unatt, os.path.join(group_dir, "Correlations"))

        subjects = [subj.get("subject_label", subj["subject_id"]) for subj in group_results]
        plot_subject_bars(full_accs, win_accs, subjects, os.path.join(group_dir, "Accuracy"))

        
        
    save_group(results, "ALL", run_dir)
    # -------------------
    # Per-dataset groups
    # -------------------
    grouped = {}
    for r in results:
        ds = str(r.get("dataset", "unknown"))
        grouped.setdefault(ds, []).append(r)

     

    # Only write separate groups if there is more than one dataset present
    if len(grouped) > 1:
        for ds, group_results in grouped.items():
            group_dir = os.path.join(run_dir, f"{ds}")
            save_group(group_results, ds, group_dir)

    
    
if __name__ == '__main__':
    main()
