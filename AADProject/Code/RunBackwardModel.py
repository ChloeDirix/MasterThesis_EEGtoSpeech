import os
import numpy as np
from pynwb import NWBHDF5IO
from scipy.stats import zscore
from sklearn.linear_model import Ridge, LinearRegression

from Code.BackwardModel import evaluate_model, create_lag_matrix
from Code import DataPrep

def ridge_fit(X, y, alpha):
    """Closed-form ridge regression (fast)."""
    XtX = X.T @ X
    XtY = X.T @ y
    n_features = XtX.shape[0]
    w = np.linalg.solve(XtX + alpha * np.eye(n_features), XtY)
    return w


def ridge_predict(X, w):
    """Apply learned weights."""
    return X @ w

def run_mTRF(nwb_path: str, cfg):
    fs = cfg["target_fs"]

    # == Window parameters ==
    window_len = int(cfg["decision_window"] * fs)
    step = int(cfg["decision_step"] * fs)

    # == Load NWB file ==
    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        trials = nwbfile.trials.to_dataframe()
        print(f"Loaded {len(trials)} trials from {nwb_path}")

        # get eeg and envelopes ready
        data = []
        for _, trial_row in trials.iterrows():
            trial_id = trial_row.name+1  # index in NWB table

            # EEG preprocessed
            preproc_key = f"trial_{trial_id}_EEG_preprocessed"
            eeg = nwbfile.processing["eeg_preprocessed"].data_interfaces[preproc_key].data[:]

            # Stimuli
            stimL_name = trial_row.get("stim_L_name", None)
            stimR_name = trial_row.get("stim_R_name", None)
            envL = np.load(os.path.join(cfg["Env_dir"], f"{stimL_name}_env.npy"))
            envR = np.load(os.path.join(cfg["Env_dir"], f"{stimR_name}_env.npy"))

            # Attended ear
            att_ear = trial_row.get("attended_ear", None)

            # align everything
            eeg, env_att, env_unatt = DataPrep.PrepareInputs(eeg, envL, envR, att_ear)
            data.append((eeg, env_att, env_unatt))


    # == Precompute lagged matrices ==
    lag_samp = np.arange(int(cfg["lag_ms"][0] / 1000 * fs),
                         int(cfg["lag_ms"][1] / 1000 * fs))

    precomputed = []
    for eeg, env_att, env_unatt in data:
        eeg_z = zscore(eeg, axis=1)
        env_att_z = zscore(env_att)
        env_unatt_z = zscore(env_unatt)

        X_lagged = create_lag_matrix(eeg_z, lag_samp)
        precomputed.append({
            "X_lagged": X_lagged,
            "env_att": env_att_z,
            "env_unatt": env_unatt_z
        })


    # == Optional: tune λ ==
    val_fraction = 0.2
    n_trials = len(data)
    n_val = max(1, int(n_trials * val_fraction))
    val_indices = np.arange(n_trials - n_val, n_trials)
    train_indices = np.arange(0, n_trials - n_val)

    lambdas = np.logspace(0, 6, 7)
    best_lambda = None
    best_score = -np.inf

    print(f"Tuning λ using {len(train_indices)} train and {len(val_indices)} validation trials...")

    # Build training data (all train trials)
    X_train = np.vstack([precomputed[i]["X_lagged"] for i in train_indices])
    y_train = np.hstack([precomputed[i]["env_att"] for i in train_indices])

    # Precompute for reuse across λ

    XtX = X_train.T @ X_train
    XtY = X_train.T @ y_train
    n_features = XtX.shape[0]

    # Validate on held-out trials
    for lam in lambdas:
        w = np.linalg.solve(XtX + lam * np.eye(n_features), XtY)

        scores = []
        for vi in val_indices:
            test = precomputed[vi]
            y_pred_att = ridge_predict(test["X_lagged"], w)

            corr_att = np.corrcoef(y_pred_att, test["env_att"])[0, 1]
            corr_unatt = np.corrcoef(y_pred_att, test["env_unatt"])[0, 1]
            scores.append(corr_att - corr_unatt)

        mean_score = np.mean(scores)
        print(f"λ={lam:.1e} | mean Δr={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam

    print(f"✅ Best λ from validation = {best_lambda:.1e}")


    # == Leave-one-trial-out cross-validation ==
    subject_results = []

    for test_idx in range(len(precomputed)):

        # Train on all trials except the test one
        X_train = np.vstack([precomputed[i]["X_lagged"]
                             for i in range(len(precomputed)) if i != test_idx])
        y_train = np.hstack([precomputed[i]["env_att"]
                             for i in range(len(precomputed)) if i != test_idx])

        w = ridge_fit(X_train, y_train, best_lambda or 0.0)   # <-- uses helper

        # Test trial
        test = precomputed[test_idx]

        # Full-trial evaluation
        y_pred_att = ridge_predict(test["X_lagged"], w)
        corr_att = np.corrcoef(y_pred_att, test["env_att"])[0, 1]
        corr_unatt = np.corrcoef(y_pred_att, test["env_unatt"])[0, 1]
        correct_full = bool(corr_att > corr_unatt)

        # Windowed evaluation
        n_samples = test["X_lagged"].shape[0]
        window_results = []
        for start in range(0, n_samples - window_len + 1, window_len):
            end = start + window_len
            X_win = test["X_lagged"][start:end]
            env_att_win = test["env_att"][start:end]
            env_unatt_win = test["env_unatt"][start:end]

            y_pred_win = ridge_predict(X_win, w)
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

        #print output
        print(f"Trial {test_idx+1}/{len(data)}: "
              f"r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | "
              f"full_correct={correct_full} | win_acc={window_acc:.2f}")

    # --- summary ---
    subj_acc = np.mean([r["correct_full"] for r in subject_results])
    subj_win_acc = np.mean([r["window_accuracy"] for r in subject_results])

    print(f"\n✅ Subject — Full-trial acc: {subj_acc:.2f}, Windowed acc: {subj_win_acc:.2f}")

    return {
        "subject_id": nwbfile.identifier,
        "results": subject_results,
        "full_accuracy": subj_acc,
        "window_accuracy": subj_win_acc
    }
