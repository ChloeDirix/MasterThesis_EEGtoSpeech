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

def load_subject_block(nwb_path, lags, fs, multiband, eeg_std=None, y_std=None):
    """
    input: path to preprocessed subject data
    output: precomputed X_lagged ad Y_lagged for 1 subject (Xi,yi)
    """

    # == prepare data for model ==
    data_subj, nwb_path = DataPreparation.Load_data(nwb_path, merged=True, multiband=multiband)
    

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

    return np.vstack(X_list), np.vstack(y_list)

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

    # normalization: fit standardizers on TRAIN subjects only
    eeg_accum = []
    y_accum = []

    for p in tr_paths:
        data_subj, _ = DataPreparation.Load_data(p, merged=True, multiband=multiband)
        for eeg, env_att, _ in data_subj:
            eeg_accum.append(eeg)
            y = env_att if multiband else env_att.reshape(-1, 1)
            y_accum.append(y)

    eeg_std = Standardizer().fit(np.vstack(eeg_accum))
    y_std   = Standardizer().fit(np.vstack(y_accum))

    # Load 1 subject to determine feature dimension
    X_first, y_first = load_subject_block(tr_paths[0],lags, fs, multiband,eeg_std,y_std)
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
        Xs, ys = load_subject_block(p,lags, fs, multiband, eeg_std, y_std)
        XtX += Xs.T @ Xs
        XtY += Xs.T @ ys

    

    # validation loop
    print("Tuning λ ...")

    lambdas = np.logspace(0, 6, 7)
    best_lambda = None
    best_score = -np.inf        #negative infinity, so the first value is always better

    # Preload validation subjects
    val_blocks = [load_subject_block(p,lags, fs, multiband,eeg_std,y_std) for p in val_paths]

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
        Xs, ys = load_subject_block(p,lags, fs, multiband,eeg_std, y_std)
        XtX += Xs.T @ Xs
        XtY += Xs.T @ ys

    w = np.linalg.solve(XtX + best_lambda * np.eye(n_features), XtY)
    decoder_w = w.copy()
    # == evaluate on test subject ==
    test_data, test_nwb = DataPreparation.Load_data(test_path, merged=True, multiband=multiband)

    results = []
    for ti, (eeg, env_att, env_unatt) in enumerate(test_data):
        eeg=eeg_std.transform(eeg)
        y_att = env_att if multiband else env_att.reshape(-1,1)
        y_un  = env_unatt if multiband else env_unatt.reshape(-1,1)

        y_att = y_std.transform(y_att)
        y_un  = y_std.transform(y_un)

        X = create_lag_matrix(eeg, lags, fs)
        y_pred = X @ w

        # Full trial correlation
        corr_att = np.mean([np.corrcoef(y_pred[:, b], y_att[:, b])[0, 1] for b in range(Ydim)])
        corr_unatt = np.mean([np.corrcoef(y_pred[:, b], y_un[:, b])[0, 1] for b in range(Ydim)])

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

            win_list.append({
                "start": start / fs,
                "end": end / fs,
                "correct": bool(wa > wu)
            })

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

    # dataset label
    dataset_label = "unknown"
    p = str(test_path).lower()
    if "dtu" in p:
        dataset_label = "DTU"
    elif "das" in p:
        dataset_label = "DAS"

    print(f"Subject {test_nwb.identifier}: Full acc = {full_acc:.2f} | Window acc = {win_acc:.2f}")

    # --- subject label (full ID, e.g. S1_DAS) ---
    subject_label = os.path.basename(str(test_path))
    subject_label = subject_label.replace(".nwb", "")

    

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



def main():
    cfg = paths.load_config()
    mode=cfg["SI_mode"]["mode"]

    subjects = cfg["subjects"]["all"]

    print("\n========== RUNNING LOSO BACKWARD MODEL ==========")
    print(f"Subjects: {subjects}")
    print("=================================================\n")

    # --------------------------------------------------
    # Create automatic run-numbered directory for SI run
    # --------------------------------------------------
    SI_base = paths.RESULTS_LIN / "SI"
    run_dir = paths.get_next_run_dir(SI_base)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Saving all SI results to: {run_dir}")

    # Save config snapshot
    paths.save_config_copy(cfg, run_dir)

    all_results = []

    
    
    # == LOSO LOOP ==
    def LOSO_loop(subjects, run_dir):
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
            tmp_json = os.path.join(run_dir, f"mTRF_partial_{i}.json")
            serializable_partial=make_json_serializable(all_results)
            with open(tmp_json, "w") as f:
                json.dump(serializable_partial, f, indent=4)

            print(f"✓ Finished subject {test_subj}")
            print(f"✓ Partial results saved to: {tmp_json}\n")

    if mode == "separate":
        LOSO_loop(subjects, run_dir)
    elif mode == "mixed":
        LOSO_loop([s for s in subjects if "DTU" in s], run_dir)
        LOSO_loop([s for s in subjects if "DAS" in s], run_dir)

        print("\n========== LOSO COMPLETED — SUMMARIZING ==========\n")

    summarize_results(all_results, cfg, run_dir)

    print("\n========== DONE ==========\n")

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
    """Aggregate and save all subject-level results."""
    
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

        # JSON export
        json_path = os.path.join(group_dir, f"mTRF_results_{group_name}.json")
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(group_results), f, indent=4)

        # CSV export 
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

            # mean of this group
            writer.writerow([
                f"MEAN_{group_name}",
                group_name,
                float(np.mean(full_accs)),
                float(np.mean(win_accs)),
            ])

            # mean across ALL results (not just this group)
            all_full = [r["full_accuracy"] for r in results]
            all_win = [r["window_accuracy"] for r in results]
            writer.writerow([
                "MEAN_ALL",
                "ALL",
                float(np.mean(all_full)),
                float(np.mean(all_win)),
            ])

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

                       

    # ---------- ALL ----------
    save_group(results, "ALL", run_dir)

    # ---------- per-dataset ----------
    grouped = {}
    for r in results:
        ds = str(r.get("dataset", "unknown"))
        grouped.setdefault(ds, []).append(r)

    if len(grouped) > 1:
        for ds, group_results in grouped.items():
            group_dir = os.path.join(run_dir, ds)
            save_group(group_results, ds, group_dir)

if __name__ == '__main__':
    main()
