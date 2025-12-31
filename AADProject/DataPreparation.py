import os
import numpy as np
from scipy.stats import zscore
from pynwb import NWBHDF5IO

from paths import paths
from Preprocessing import EEGPreprocessing, stimulusPreprocessing
from Loaders.matlab_loader import MatlabSubjectLoader
from NWB.NWB_Manager import NWBManager


# -------------------------------------------------------------------------------
#                             DATA PREPROCESSING
# -------------------------------------------------------------------------------
def Preprocess_Data():

    cfg= paths.load_config()

    # what steps and what subjects
    do_env = cfg["Do_envelope_extraction"]
    do_eeg = cfg["Do_preprocessing"]
    subjects = cfg["subjects"]["all"]

    # make output dir
    os.makedirs(paths.EEG_PP, exist_ok=True)
    os.makedirs(paths.ENVELOPES, exist_ok=True)

    # ---------------- Envelope extraction ----------------
    if do_env:
        stimulusPreprocessing.PreprocessAudioFiles(cfg)

    # ---------------- EEG preprocessing ------------------
    if do_eeg:
        nwb_mgr = NWBManager()

        for subject_id in subjects:
            print(f"\n=== EEG preprocessing subject {subject_id} ===")

            in_file = paths.subject_raw(subject_id)
            out_file = paths.subject_eegPP(subject_id)

            loader = MatlabSubjectLoader(in_file, subject_id)
            subject = loader.load()

            for trial in subject.trials:
                eeg, fs=EEGPreprocessing.preprocess_trial(trial, cfg)

                trial.eeg_PP = eeg
                trial.fs_eeg = fs
                trial.metadata["preprocessed"] = True

            nwb_mgr.save_subject(subject, out_file)
            print(f"Saved preprocessed NWB for {subject_id} → {out_file}")



# -------------------------------------------------------------------------------
#                             Load Data
# -------------------------------------------------------------------------------

def Load_data(nwb_path, merged=True, multiband=True):

    data=[]
    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        trials = nwbfile.trials.to_dataframe()


        for _, tr in trials.iterrows():
            eeg, env_att, env_unatt = load_single_trial(nwbfile, tr, multiband)
            data.append((eeg, env_att, env_unatt))

    # ------------- Merge repetition trials for mTRF ----------
    if merged:
        data = merge_repetition_trials(trials, data)
    else:
        print(f"Using raw (unmerged) trials: {len(data)}")

    return data, nwbfile  # keep NWB object for metadata later


# ==============================================================================
#                           INTERNAL TRIAL PROCESSING
# ==============================================================================


def load_single_trial(nwbfile, tr, multiband):
    """Load EEG + attended/unattended envelopes for one trial."""

    tid = tr.name + 1
    pre_key = f"trial_{tid}_EEG_preprocessed"

    # ---------------- EEG ----------------
    eeg = nwbfile.processing["eeg_preprocessed"].data_interfaces[pre_key].data[:]

    # Drop channels with zero-variance (flat channels)
    eeg = eeg[:, np.std(eeg, axis=0) > 0]

    # ---------------- Envelopes ----------------
    stimL = os.path.splitext(tr["stim_L_name"])[0]
    stimR = os.path.splitext(tr["stim_R_name"])[0]

    envL, _ = get_envelope(stimL, multiband)
    envR, _ = get_envelope(stimR, multiband)

    # ---------------- Attended ear ----------------
    env_att, env_unatt = get_attended(tr["attended_ear"], envL, envR)

    # ---------------- Align lengths ---------------
    eeg, env_att, env_unatt = align_lengths(eeg, env_att, env_unatt)

    # ---------------- Z-score ---------------------
    eeg = zscore(eeg, axis=0)
    env_att = zscore(env_att, axis=0)
    env_unatt = zscore(env_unatt, axis=0)

    return eeg, env_att, env_unatt


def merge_repetition_trials(trials_df, data):
    """
    Merge trials where the left stimulus name starts with rep_.
    Expected format: rep_originalname
    """
    stim_bases = trials_df["stim_L_name"].apply(lambda x: os.path.splitext(x)[0])

    long_trials_idx = [i for i, s in enumerate(stim_bases) if not s.startswith("rep_")]

    # Group repetitions
    rep_groups = {}
    for idx, name in enumerate(stim_bases):
        if name.startswith("rep_"):
            orig = name[4:]
            rep_groups.setdefault(orig, []).append(idx)

    merged = []

    for orig_stim, idxs in rep_groups.items():
        if len(idxs) != 3:
            print(f"Warning: {orig_stim} has {len(idxs)} reps (expected 3). Skipping.")
            continue

        eeg = np.concatenate([data[i][0] for i in idxs])
        att = np.concatenate([data[i][1] for i in idxs])
        unatt = np.concatenate([data[i][2] for i in idxs])

        merged.append((eeg, att, unatt))

    long_trials = [data[i] for i in long_trials_idx]
    return long_trials + merged


# -------------------------------------------------------------------------------
#                             Get Envelopes
# -------------------------------------------------------------------------------

def get_envelope(stim_base, multiband=True):
    npz = np.load(paths.envelope(f"{stim_base}_env.npz"))
    env = npz["envelope"]
    w = npz["subband_weights"]

    if multiband:
        return env, w
    else:
        w_norm = w / np.sum(w)
        return env @ w_norm, w_norm

def get_attended(att_ear, env_left, env_right):
    if str(att_ear).upper().endswith("L"):
        return env_left, env_right
    else:
        return env_right, env_left


def align_lengths(*arrays):
    m = min(len(a) for a in arrays)
    return tuple(a[:m] for a in arrays)


if __name__ == "__main__":
    Preprocess_Data()


