import csv
import json
import os

import numpy as np

import DataPreparation
from BackwardModel import summaryStats
from paths import paths


def create_lag_matrix(EEG, lags_t, fs):
    """
    input:
        EEG: (t,channels)
        lags_t: parameter in config file
        fs: sample frequency Hz
    output: X_lagged: (t,len(lags_t)*fs*channels)
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

def load_subject_block(nwb_path, lags, fs, multiband):
    """
    input: path to preprocessed subject data
    output: precomputed X_lagged ad Y_lagged for 1 subject (Xi,yi)
    """

    # == prepare data for model ==
    data_subj, _ = DataPreparation.Load_data(nwb_path, merged=True, multiband=True)

    X_list = []
    y_list = []

    for eeg, env_att, env_unatt in data_subj:
        X = create_lag_matrix(eeg, lags, fs)
        X_list.append(X)
        if multiband:
            y_list.append(env_att)
        else:
            y_list.append(env_att.reshape(-1))

    X_subj = np.vstack(X_list)
    y_subj = np.vstack(y_list) if multiband else np.hstack(y_list)
    return X_subj, y_subj

def run_mTRF_subject(train_paths, test_path, cfg):

    # == load config ==
    fs = cfg["preprocessing"]["target_fs"]
    window_len = int(cfg["backward_model"]["window_s"] * fs)
    step = int(cfg["backward_model"]["step_s"] * fs)
    lags = cfg["backward_model"]["lag_ms"]
    multiband = cfg["backward_model"]["multiband"]


    print(f"Loading {len(train_paths)} training subjects...")


    # == Tune λ ==
    # define validation and train subjects
    val_fraction = 0.2
    n_subjects = len(train_paths)
    n_val = max(1, int(n_subjects * val_fraction))

    val_paths = train_paths[-n_val:]
    tr_paths = train_paths[:-n_val]

    # Load 1 subject to determine feature dimension
    X_first, y_first = load_subject_block(tr_paths[0])
    n_features = X_first.shape[1]

    XtX = np.zeros((n_features, n_features))
    if multiband:
        Ydim = y_first.shape[1]
        XtY = np.zeros((n_features, Ydim))
    else:
        XtY = np.zeros((n_features,))

    # Use first subject
    XtX += X_first.T @ X_first
    XtY += X_first.T @ y_first

    # Add Remaining subjects
    for p in tr_paths[1:]:
        Xs, ys = load_subject_block(p)
        XtX += Xs.T @ Xs
        XtY += Xs.T @ ys

    # validation loop
    print("Tuning λ ...")

    lambdas = np.logspace(0, 6, 7)
    best_lambda = None
    best_score = -np.inf        #negative infinity, so the first value is always better

    # Preload validation subjects
    val_blocks = [load_subject_block(p) for p in val_paths]

    for lam in lambdas:
        # train
        w = np.linalg.solve(XtX + lam * np.eye(n_features), XtY)

        # test
        scores = []
        for Xv, yv in val_blocks:
            y_pred = ridge_predict(Xv,w)

            if multiband:
                r_att = np.mean([np.corrcoef(y_pred[:, b], yv[:, b])[0, 1]
                            for b in range(y_pred.shape[1])])
            else:
                r_att = np.corrcoef(y_pred, yv)[0, 1]

            scores.append(r_att)

        mean_score = np.mean(scores)
        print(f"λ={lam:.1e}  | mean r={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam

    print(f"Best λ = {best_lambda}")

    # == Leave one subject out cross (LOSO) validation ==
    XtX = np.zeros((n_features, n_features))
    XtY = np.zeros((n_features, Ydim)) if multiband else np.zeros((n_features,))

    for p in train_paths:
        Xs, ys = load_subject_block(p)
        XtX += Xs.T @ Xs
        XtY += Xs.T @ ys

    w = np.linalg.solve(XtX + best_lambda * np.eye(n_features), XtY)

    # == evaluate on test subject ==
    test_data, test_nwb = DataPreparation.Load_data(test_path, merged=True, multiband=True)

    results = []
    for ti, (eeg, env_att, env_unatt) in enumerate(test_data):
        X = create_lag_matrix(eeg, lags, fs)
        y_pred = X @ w

        # Full trial correlation
        if multiband:
            corr_att = np.mean([np.corrcoef(y_pred[:, b], env_att[:, b])[0, 1] for b in range(y_pred.shape[1])])
            corr_unatt = np.mean([np.corrcoef(y_pred[:, b], env_unatt[:, b])[0, 1] for b in range(y_pred.shape[1])])
        else:
            corr_att = np.corrcoef(y_pred, env_att)[0, 1]
            corr_unatt = np.corrcoef(y_pred, env_unatt)[0, 1]

        correct_full = bool(corr_att > corr_unatt)

        # Window evaluation
        n_samples = X.shape[0]
        win_list = []
        for start in range(0, n_samples - window_len + 1, step):
            end = start + window_len
            yp = X[start:end] @ w

            if multiband:
                wa = np.mean([np.corrcoef(yp[:, b], env_att[start:end, b])[0, 1] for b in range(yp.shape[1])])
                wu = np.mean([np.corrcoef(yp[:, b], env_unatt[start:end, b])[0, 1] for b in range(yp.shape[1])])
            else:
                wa = np.corrcoef(yp, env_att[start:end])[0, 1]
                wu = np.corrcoef(yp, env_unatt[start:end])[0, 1]

            win_list.append({"correct": bool(wa > wu)})

        window_acc = np.mean([x["correct"] for x in win_list])

        results.append({
            "trial": ti + 1,
            "corr_att": corr_att,
            "corr_unatt": corr_unatt,
            "correct_full": correct_full,
            "window_accuracy": window_acc,
            "windows": win_list
        })

        print(
            f"Trial {ti + 1}/{len(test_data)}: r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | {correct_full} | win_acc={window_acc:.2f}")

    full_acc = np.mean([r["correct_full"] for r in results])
    win_acc = np.mean([r["window_accuracy"] for r in results])

    print(f"Subject {test_nwb.identifier}: Full acc = {full_acc:.2f} | Window acc = {win_acc:.2f}")

    return {
        "subject_id": test_nwb.identifier,
        "results": results,
        "full_accuracy": full_acc,
        "window_accuracy": win_acc,
    }



def main():
    cfg = paths.load_config()
    subjects = cfg["subjects"]["all"]

    print("\n========== RUNNING LOSO BACKWARD MODEL ==========")
    print(f"Subjects: {subjects}")
    print("=================================================\n")

    all_results = []

    # Make results directory
    os.makedirs(paths.RESULTS_LIN, exist_ok=True)

    # == LOSO LOOP ==
    for i, test_subj in enumerate(subjects, start=1):

        print(f"\n===============================================")
        print(f"LOSO {i}/{len(subjects)} — Test subject: {test_subj}")
        print("===============================================")

        # Paths
        test_path = paths.subject_eegPP(test_subj)
        train_paths = [paths.subject_eegPP(s) for s in subjects if s != test_subj]

        # Run backward model
        res = run_mTRF_subject(train_paths, test_path, cfg)

        all_results.append(res)

        # Save intermediate results after each subject
        tmp_json = paths.result_file_lin(f"mTRF_partial_{i}.json")
        with open(tmp_json, "w") as f:
            json.dump(all_results, f, indent=4)

        print(f"✓ Finished subject {test_subj}")
        print(f"✓ Partial results saved to: {tmp_json}\n")

    print("\n========== LOSO COMPLETED — SUMMARIZING ==========\n")

    summarize_results(all_results, cfg)

    print("\n========== DONE ==========\n")


def summarize_results(results, cfg):
    """Aggregate and save all subject-level results."""
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
    json_path = paths.result_file_lin(f"mTRF_results_LOSO.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")

    # CSV summary
    csv_path = paths.result_file_lin(f"mTRF_summary_LOSO.csv")
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

    with open(paths.result_file_lin(f"stats_LOSO.json"), "w") as f:
        json.dump(stats, f, indent=4)

    summaryStats.plot_histograms(all_att, all_unatt, paths.result_file_lin(f"Histogram_LOSO"))


if __name__ == '__main__':
    main()
