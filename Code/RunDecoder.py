import os
import numpy as np
import yaml

from BackwardModel import train_backward_model, evaluate_model

def run_mTRF(subject_id, subject_data, cfg):
    pp_dir = os.path.join(cfg["base_dir"], cfg["PP_dir"])
    env_dir = os.path.join(cfg["base_dir"], cfg["Env_dir"])

    print(pp_dir)
    results = []

    print(subject_data.getTrials())
    for trial_index in range(1,subject_data.get_trial_count()+1):
        print(f"{subject_id}_trial{trial_index:02d}_preprocessed.npy")
        eeg_file = os.path.join(pp_dir, f"{subject_id}_trial{trial_index:02d}_preprocessed.npy")
        env_att_file = os.path.join(env_dir, f"{subject_id}_trial{trial_index:02d}_env_att.npy")
        env_unatt_file = os.path.join(env_dir, f"{subject_id}_trial{trial_index:02d}_env_unatt.npy")

        if not os.path.exists(eeg_file) or not os.path.exists(env_att_file):
            print(f"Missing data for trial {trial_index}, skipping…")
            continue

        if not os.path.exists(env_unatt_file):
            print(f"No unattended env found — skipping trial {trial_index}")
            continue

        env_unatt = np.load(env_unatt_file)


        eeg = np.load(eeg_file)
        env_att = np.load(env_att_file)
        env_unatt = np.load(env_unatt_file)

        model, lags = train_backward_model(
            eeg, env_att, fs=cfg["target_fs"],
            lambda_val=1.0, lag_ms=(-100,400)
        )

        corr_att, pred_att = evaluate_model(model, eeg, env_att, lags)

        corr_unatt, pred_unatt = evaluate_model(model, eeg, env_unatt, lags)
        decoded_correct = corr_att > corr_unatt

        results.append({
            "trial": trial_index,
            "corr_att": corr_att,
            "corr_unatt": corr_unatt,
            "correct": decoded_correct
        })

        print(f"Trial {trial_index}: r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | correct={decoded_correct}")

        accuracy = np.mean([r["correct"] for r in results])
        print("\nSubject Accuracy:", accuracy)

    return results


