import os
import numpy as np
from BackwardModel import evaluate_model, train_backward_model


def run_mTRF(subjects_list, cfg):

    pp_dir = os.path.join(cfg["base_dir"], cfg["PP_dir"])
    env_dir = os.path.join(cfg["base_dir"], cfg["Env_dir"])
    fs = cfg["target_fs"]

    # Window parameters for windowed evaluation
    window_s = cfg["decision_window"]
    step_s = cfg["decision_step"]
    window_len = int(window_s * fs)  #in samples
    step = int(step_s * fs)          #in samples

    all_results = []

    #iterate over subjects
    for subject_id, subject_data in subjects_list:
        print(f"\n=== Running mTRF for Subject {subject_id} ===")
        data = []

        # iterate over trials to load files
        for trial_index in range(1, subject_data.get_trial_count() + 1):
            eeg_file = os.path.join(pp_dir, f"{subject_id}_trial{trial_index:02d}_preprocessed.npy")
            env_att_file = os.path.join(env_dir, f"{subject_id}_trial{trial_index:02d}_env_att.npy")
            env_unatt_file = os.path.join(env_dir, f"{subject_id}_trial{trial_index:02d}_env_unatt.npy")

            if not os.path.exists(eeg_file) or not os.path.exists(env_att_file) or not os.path.exists(env_unatt_file):
                print(f"  Missing data for trial {trial_index}, skipping…")
                continue

            eeg = np.load(eeg_file)
            env_att = np.load(env_att_file)
            env_unatt = np.load(env_unatt_file)
            data.append((eeg, env_att, env_unatt))

        valid_trials = len(data)
        if valid_trials == 0:
            print("  No usable trials — skipping subject.")
            continue

        subject_results = []

        # Leave-one-trial-out cross-validation
        for test_idx in range(valid_trials):
            train_eeg = [data[i][0] for i in range(valid_trials) if i != test_idx]
            train_env = [data[i][1] for i in range(valid_trials) if i != test_idx]

            if not all(e.shape[0] == t.shape[0] for e, t in zip(train_eeg, train_env)):
                raise ValueError("EEG and envelope trial lengths must match before concatenation.")

            train_eeg = np.vstack(train_eeg)
            train_env = np.hstack(train_env)

            # Train backward model
            model, lags, mean_std_list = train_backward_model(
                train_eeg, train_env, fs,
                lambda_val=cfg["lambda_val"],
                lag_ms=cfg["lag_ms"],
            )

            # Test trial
            eeg_test, env_att_test, env_unatt_test = data[test_idx]

            # Full-trial evaluation
            corr_att, _ = evaluate_model(model, eeg_test, env_att_test, lags, mean_std_list)
            corr_unatt, _ = evaluate_model(model, eeg_test, env_unatt_test, lags, mean_std_list)
            correct_full = corr_att > corr_unatt

            # Windowed evaluation
            n_samples = eeg_test.shape[0]
            window_results = []
            for start in range(0, n_samples - window_len + 1, step):
                end = start + window_len
                eeg_win = eeg_test[start:end]
                env_att_win = env_att_test[start:end]
                env_unatt_win = env_unatt_test[start:end]

                w_corr_att, _ = evaluate_model(model, eeg_win, env_att_win, lags, mean_std_list)
                w_corr_unatt, _ = evaluate_model(model, eeg_win, env_unatt_win, lags, mean_std_list)
                window_results.append({
                    "start": start / fs,
                    "end": end / fs,
                    "corr_att": w_corr_att,
                    "corr_unatt": w_corr_unatt,
                    "correct": w_corr_att > w_corr_unatt
                })

            # Calculate window accuracy
            window_acc = np.mean([w["correct"] for w in window_results])

            subject_results.append({
                "trial": test_idx + 1,
                "corr_att": corr_att,
                "corr_unatt": corr_unatt,
                "correct_full": correct_full,
                "window_accuracy": window_acc,
                "windows": window_results
            })

            print(f"Trial {test_idx+1}/{valid_trials}: "
                  f"r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | "
                  f"full_correct={correct_full} | win_acc={window_acc:.2f}")

        # --- Subject summary ---
        subj_acc = np.mean([r["correct_full"] for r in subject_results])
        subj_win_acc = np.mean([r["window_accuracy"] for r in subject_results])
        print(f"Subject {subject_id} — Full-trial acc: {subj_acc:.2f}, Windowed acc: {subj_win_acc:.2f}")

        all_results.append({
            "subject_id": subject_id,
            "results": subject_results,
            "full_accuracy": subj_acc,
            "window_accuracy": subj_win_acc
        })

    return all_results
