import numpy as np

from Code import DataPreparation


def create_lag_matrix(EEG, lags_t, fs):
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

    # == load config ==
    fs = cfg["target_fs"]
    window_len = int(cfg["decision_window"] * fs)
    step = int(cfg["decision_step"] * fs)
    lags = cfg["lag_ms"]
    multiband=cfg["multiband"]


    # == prepare data for model ==
    data,nwbfile =DataPreparation.getData(nwb_path,cfg,multiband)


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

    # Precompute for reuse across λ
    XtX = X_train.T @ X_train
    XtY = X_train.T @ y_train
    n_features = XtX.shape[0]   # n_features=channels×number of lags

    # Iterate over lambda
    for lam in lambdas:
        # train
        w = np.linalg.solve(XtX + lam * np.eye(n_features), XtY)

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

    print(f"✅ Best λ from validation = {best_lambda:.1e}")


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
