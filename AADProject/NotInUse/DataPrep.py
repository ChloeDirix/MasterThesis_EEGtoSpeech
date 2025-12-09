import os

import numpy as np
from pynwb import NWBHDF5IO
from scipy.stats import zscore

from paths import paths


def getData(nwb_path, multiband):

    # == Load NWB file ==
    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        trials = nwbfile.trials.to_dataframe()
        print(f"Loaded {len(trials)} trials from {nwb_path}")

        # get eeg and envelopes
        data = []
        for _, trial_row in trials.iterrows():
            trial_id = trial_row.name + 1  # index in NWB table

            # --- EEG preprocessed ---
            preproc_key = f"trial_{trial_id}_EEG_preprocessed"
            eeg = nwbfile.processing["eeg_preprocessed"].data_interfaces[preproc_key].data[:]

            # --- Stimuli ---
            stimL_name = trial_row.get("stim_L_name", None)
            stimR_name = trial_row.get("stim_R_name", None)

            stimL_base = os.path.splitext(stimL_name)[0]   # Remove .wav or any extension
            stimR_base = os.path.splitext(stimR_name)[0]

            npzL = np.load(paths.envelope(f"{stimL_base}_env.npz"))
            npzR = np.load(paths.envelope(f"{stimR_base}_env.npz"))

            envL_sub = npzL["envelope"]  # envelopes (samples, bands)
            envR_sub = npzR["envelope"]
            wL = npzL["subband_weights"]  # weights (bands,)
            wR = npzR["subband_weights"]

            if multiband:
                envL = envL_sub  # shape (samples, bands)
                envR = envR_sub  # shape (samples, bands)
            else:
                envL = envL_sub @ (wL / np.sum(wL))  # Weighted broadband envelopes (1D)
                envR = envR_sub @ (wR / np.sum(wR))


            # --- Attended ear ---
            att_ear = trial_row.get("attended_ear", None)

            # drop channels with constant variance
            ch_std = np.std(eeg, axis=0)
            keep_mask = ch_std > 0
            eeg = eeg[:, keep_mask]

            # -- Make envelopes and eeg aligned ---
            eeg_trim, env_left_trim, env_right_trim = align_lengths(eeg, envL, envR)

            # determine attended/unattended
            env_att, env_unatt = get_attended(att_ear,env_left_trim, env_right_trim)

            eeg = zscore(eeg, axis=0)
            env_att = zscore(env_att, axis=0)
            env_unatt = zscore(env_unatt, axis=0)

            data.append((eeg, env_att, env_unatt))

    long_data, merged_data = merge_repetition_trials(trials, data)
    data = long_data + merged_data

    normalized_data = []
    for eeg, env_att, env_unatt in data:
        eeg_norm = zscore(eeg, axis=0)
        env_att_norm = zscore(env_att, axis=0)
        env_unatt_norm = zscore(env_unatt, axis=0)
        normalized_data.append((eeg_norm, env_att_norm, env_unatt_norm))

    print(f"Final number of trials: {len(normalized_data)}")

    return normalized_data, nwbfile


def align_lengths(eeg, env_left, env_right):

    min_len = min(len(eeg), len(env_left), len(env_right))
    eeg=eeg[:min_len]
    env_left = env_left[:min_len]
    env_right = env_right[:min_len]
    return eeg, env_left, env_right


def get_attended(attended_ear, env_left, env_right):
    if str(attended_ear).upper().endswith("L"):
        return env_left, env_right
    else:
        return env_right, env_left


