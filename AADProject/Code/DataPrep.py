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


def merge_repetition_trials(trials_df, eeg_all, env_att_all, env_unatt_all):

    merged_eeg = []
    merged_att = []
    merged_unatt = []
    merged_meta = []

    # ---- 1) Identify unique long-trial stimuli (those without "rep_") ----
    stim_names = trials_df['stimulus'].tolist()

    long_stims = sorted({name for name in stim_names if not name.startswith("rep_")})

    # ---- 2) For each long stimulus, find its 3 repetition trials ----
    for stim in long_stims:

        # Find indices of repetition trials "rep_<stim>"
        rep_name = f"rep_{stim}"
        rep_idx = [i for i, s in enumerate(stim_names) if s == rep_name]

        # Only merge if we have repetitions (Exp1 has, Exp2 does NOT)
        if len(rep_idx) == 3:
            # Extract arrays
            eeg_parts = [eeg_all[i] for i in rep_idx]
            att_parts = [env_att_all[i] for i in rep_idx]
            unatt_parts = [env_unatt_all[i] for i in rep_idx]

            # Concatenate in time
            eeg_concat = np.concatenate(eeg_parts, axis=0)
            att_concat = np.concatenate(att_parts, axis=0)
            unatt_concat = np.concatenate(unatt_parts, axis=0)

            merged_eeg.append(eeg_concat)
            merged_att.append(att_concat)
            merged_unatt.append(unatt_concat)
            merged_meta.append({
                "stimulus": stim,
                "source_trials": rep_idx
            })

    return merged_eeg, merged_att, merged_unatt, merged_meta

