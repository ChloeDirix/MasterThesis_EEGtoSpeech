import os
import numpy as np

from AADProject.Code.BackwardModel import evaluate_model, train_backward_model


def run_mTRF(subjects_list, cfg):
    pp_dir = os.path.join(cfg["base_dir"], cfg["PP_dir"])
    env_dir = os.path.join(cfg["base_dir"], cfg["Env_dir"])
    fs=cfg[("target_fs")]

    window_s = cfg.get("decision_window_s", 10)
    step_s = cfg.get("decision_step_s", 1)


    for subject_id,subject_data in subjects_list:
        data = []
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
        results = []
        for test_idx in range(valid_trials):
            train_eeg = [data[i][0] for i in range(valid_trials) if i != test_idx]
            train_env = [data[i][1] for i in range(valid_trials) if i != test_idx]
            if not all(e.shape[0] == t.shape[0] for e, t in zip(train_eeg, train_env)):
                raise ValueError("EEG and envelope trial lengths must match before concatenation.")
            train_eeg = np.vstack(train_eeg)
            train_env = np.hstack(train_env)


            model, lags, mean_std_list = train_backward_model(
                train_eeg, train_env, fs,
                lambda_val=1000, lag_ms=(-100, 400)
            )

            eeg_test, env_att_test, env_unatt_test = data[test_idx]

            corr_att, _ = evaluate_model(model, eeg_test, env_att_test, lags, mean_std_list)
            corr_unatt, _ = evaluate_model(model, eeg_test, env_unatt_test, lags, mean_std_list)
            correct = corr_att > corr_unatt

            results.append({
                "trial": test_idx + 1,
                "corr_att": corr_att,
                "corr_unatt": corr_unatt,
                "correct": correct
            })

            print(f"Trial {test_idx+1}/{valid_trials}: "
                  f"r_att={corr_att:.3f} | r_unatt={corr_unatt:.3f} | correct={correct}")



    return results
