import os

import numpy as np


def PrepareInputs(eeg, envL, envR, attended_ear):

    # --- Align EEG and envelopes ---
    eeg_trim, env_left, env_right = align_lengths(eeg, envL, envR)

    # --- Assign attended/unattended envelopes ---
    env_att, env_unatt = get_attended(attended_ear, env_left, env_right)



    return eeg_trim, env_att, env_unatt




def align_lengths(eeg, env_left, env_right):
    """Trim or pad so EEG and envelopes have equal length."""
    #print(len(eeg), len(env_left), len(env_right))
    min_len = min(len(eeg), len(env_left), len(env_right))
    eeg=eeg[:min_len]
    env_left = env_left[:min_len]
    env_right = env_right[:min_len]
    return eeg, env_left, env_right


def get_attended(attended_ear, env_left, env_right):
    #print("attended_ear should be",str(attended_ear).upper())
    if str(attended_ear).upper().endswith("L"):

        #print("attended ear = L")
        return env_left, env_right
    else:
        #print("attended ear = R")
        return env_right, env_left


def merge_repetition_trials(trials_df, data):
    stim_names = trials_df["stim_L_name"].apply(lambda x: os.path.splitext(x)[0]).tolist()

    long_indices = [i for i, s in enumerate(stim_names) if not s.startswith("rep_")]

    # Build mapping: long_stim_name → indices of repetition trials
    rep_groups = {}
    for i, s in enumerate(stim_names):
        if s.startswith("rep_"):
            original = s[4:]  # remove 'rep_'
            rep_groups.setdefault(original, []).append(i)

    merged_data = []

    # Merge only if exactly 3 repetitions exist
    for original_stim, idx_list in rep_groups.items():
        if len(idx_list) != 3:
            print(f"⚠ Warning: {original_stim} has {len(idx_list)} repetitions, skipping merge.")
            continue

        eeg_concat = np.concatenate([data[i][0] for i in idx_list], axis=0)
        att_concat = np.concatenate([data[i][1] for i in idx_list], axis=0)
        unatt_concat = np.concatenate([data[i][2] for i in idx_list], axis=0)

        merged_data.append((eeg_concat, att_concat, unatt_concat))

    # Long trials (Exp1 and Exp2)
    long_data = [data[i] for i in long_indices]

    return long_data, merged_data

