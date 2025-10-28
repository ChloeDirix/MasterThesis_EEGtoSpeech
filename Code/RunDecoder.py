import os
import numpy as np

from BackwardModel import train_backward_model, evaluate_model

def run_mTRF(subject_id, subject_data, cfg):
    pp_dir = os.path.join(cfg["base_dir"], cfg["PP_dir"])
    env_dir = os.path.join(cfg["base_dir"], cfg["Env_dir"])

    print(pp_dir)
    results = []
    data = []

    # Gather valid trials
    for trial_index in range(1, subject_data.get_trial_count() + 1):
        eeg_file = os.path.join(pp_dir, f"{subject_id}_trial{trial_index:02d}_preprocessed.npy")
        env_att_file = os.path.join(env_dir, f"{subject_id}_trial{trial_index:02d}_env_att.npy")
        env_unatt_file = os.path.join(env_dir, f"{subject_id}_trial{trial_index:02d}_env_unatt.npy")

        if not os.path.exists(eeg_file) or not os.path.exists(env_att_file) or not os.path.exists(env_unatt_file):
            print(f"Missing data for trial {trial_index}, skipping…")
            continue

        eeg = np.load(eeg_file)
        env_att = np.load(env_att_file)
        env_unatt = np.load(env_unatt_file)
        data.append((eeg, env_att, env_unatt))

    valid_trials = len(data)
    if valid_trials == 0:
        print("No usable trials — skipping subject")
        return []

    # Leave-one-trial-out cross-validation
    for test_idx in range(valid_trials):
        train_eeg = [data[i][0] for i in range(valid_trials) if i != test_idx]
        train_env = [data[i][1] for i in range(valid_trials) if i != test_idx]

        train_eeg = np.vstack(train_eeg)
        train_env = np.hstack(train_env)

        model, lags = train_backward_model(
            train_eeg, train_env, fs=cfg["target_fs"],
            lambda_val=1.0, lag_ms=(0, 250)
        )

        eeg_test, env_att_test, env_unatt_test = data[test_idx]

        corr_att, _ = evaluate_model(model, eeg_test, env_att_test, lags)
        corr_unatt, _ = evaluate_model(model, eeg_test, env_unatt_test, lags)

        correct = corr_att > corr_unatt
        results.append({
            "trial": test_idx + 1,
            "corr_att": corr_att,
            "corr_unatt": corr_unatt,
            "correct": correct
        })

        print(f"Trial {test_idx+1}/{valid_trials}: "
              f"r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | correct={correct}")

    accuracy = np.mean([r["correct"] for r in results])
    print(f"\n✅ Subject Decoding Accuracy: {accuracy:.2f}")

    return results
