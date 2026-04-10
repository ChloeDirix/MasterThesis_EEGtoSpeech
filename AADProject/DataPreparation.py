# datapreparation.py
import os
import numpy as np
from pynwb import NWBHDF5IO

from paths import paths
from Preprocessing import EEGPreprocessing, stimulusPreprocessing
from Loaders.matlab_loader import MatlabSubjectLoader
from NWB.NWB_Manager import NWBManager


# -------------------------------------------------------------------------------
#                             DATA PREPROCESSING
# -------------------------------------------------------------------------------
def Preprocess_Data():
    cfg = paths.load_config()

    do_env = cfg["Do_envelope_extraction"]
    do_eeg = cfg["Do_preprocessing"]

    output_dir=paths.data_input_model_dir()
    print(f"Saving preprocessed data to: {output_dir}")

    os.makedirs(os.path.join(output_dir, "EEG_PP"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Envelopes"), exist_ok=True)

    for dataset, use in cfg["Datasets"].items():
        if not use:
            print(f"Skipping dataset {dataset}")
            continue

        print(f"Using dataset {dataset}")

        # ---------------- Envelope extraction ----------------
        if do_env:
            stimulusPreprocessing.PreprocessAudioFiles(cfg, dataset)

        # ---------------- EEG preprocessing ------------------
        if do_eeg:
            nwb_mgr = NWBManager()

            for subject_id in cfg[f"subjects_{dataset}"]["all"]:
                print(f"\n=== EEG preprocessing subject {subject_id} ({dataset}) ===")

                in_file = paths.subject_raw(subject_id, dataset)
                out_file = paths.subject_eegPP(subject_id, dataset)
                print(out_file, in_file)

                loader = MatlabSubjectLoader(in_file, subject_id)
                subject = loader.load()

                for trial in subject.trials:
                    eeg, fs = EEGPreprocessing.preprocess_trial(trial, cfg)
                    trial.eeg_PP = eeg
                    trial.fs_eeg = fs
                    trial.metadata["preprocessed"] = True

                nwb_mgr.save_subject(subject, out_file)
                print(f"Saved preprocessed NWB for {subject_id}_{dataset} → {out_file}")


# -------------------------------------------------------------------------------
#                             Load Data
# -------------------------------------------------------------------------------
def Load_data(
    nwb_path,
    merged=False,
    sum_subbands=True,
    multiband=None,
    return_meta=False,
):
    """
    Main data loader.

    Preferred behavior:
        sum_subbands=True  -> return single summed envelope, shape (T, 1)
        sum_subbands=False -> return separate subbands, shape (T, B)
    """

    data = []
    meta = []

    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        trials_df = nwbfile.trials.to_dataframe()

        for _, tr in trials_df.iterrows():
            eeg, env_att, env_unatt = load_single_trial(
                nwbfile,
                tr,
                sum_subbands=sum_subbands,
            )
            data.append((eeg, env_att, env_unatt))

            if return_meta:
                meta.append({
                    "trial_index": int(tr["trial_index"]),
                    "dataset": str(tr.get("dataset", "")),
                    "acoustic_condition": int(tr["acoustic_condition"]) if "acoustic_condition" in tr else -1,
                    "n_speakers": int(tr["n_speakers"]) if "n_speakers" in tr else -1,
                })

    if merged:
        data = merge_repetition_trials(trials_df, data)
        
    return (data, meta, nwbfile) if return_meta else (data, nwbfile)


# ==============================================================================
#                           INTERNAL TRIAL PROCESSING
# ==============================================================================
def load_single_trial(nwbfile, tr, sum_subbands=True):
    """Load EEG + attended/unattended envelopes for one trial."""
    tid = int(tr["trial_index"])
    pre_key = f"trial_{tid}_EEG_preprocessed"

    # ---------------- EEG ----------------
    eeg = nwbfile.processing["eeg_preprocessed"].data_interfaces[pre_key].data[:]
    eeg = np.asarray(eeg, dtype=np.float32)

    # Keep only scalp EEG
    cfg = paths.load_config()
    target_n_channels = cfg["preprocessing"]["target_n_channels"]

    if eeg.shape[1] >= target_n_channels:
        eeg = eeg[:, :target_n_channels]
    else:
        raise ValueError(
            f"Trial {tid}: EEG has only {eeg.shape[1]} channels, "
            f"expected >= {target_n_channels}"
        )

    # ---------------- Envelopes ----------------
    stimL = os.path.splitext(str(tr["stim_L_name"]))[0]
    stimR = os.path.splitext(str(tr["stim_R_name"]))[0]

    envL, _ = get_envelope(stimL, sum_subbands=sum_subbands)
    envR, _ = get_envelope(stimR, sum_subbands=sum_subbands)

    # ---------------- Attended ear ----------------
    env_att, env_unatt = get_attended(tr["attended_ear"], envL, envR)

    # ---------------- Align lengths ----------------
    eeg, env_att, env_unatt = align_lengths(eeg, env_att, env_unatt)

    return eeg, env_att, env_unatt


def merge_repetition_trials(trials_df, data):
    """
    Merge trials where the left stimulus name starts with rep_.
    Expected format: rep_originalname
    """
    stim_bases = trials_df["stim_L_name"].apply(lambda x: os.path.splitext(str(x))[0])

    long_trials_idx = [i for i, s in enumerate(stim_bases) if not str(s).startswith("rep_")]

    # Group repetitions
    rep_groups = {}
    for idx, name in enumerate(stim_bases):
        name = str(name)
        if name.startswith("rep_"):
            orig = name[4:]
            rep_groups.setdefault(orig, []).append(idx)

    merged = []

    for orig_stim, idxs in rep_groups.items():
        if len(idxs) != 3:
            print(f"Warning: {orig_stim} has {len(idxs)} reps (expected 3). Skipping.")
            continue

        eeg = np.concatenate([data[i][0] for i in idxs], axis=0)
        att = np.concatenate([data[i][1] for i in idxs], axis=0)
        unatt = np.concatenate([data[i][2] for i in idxs], axis=0)

        merged.append((eeg, att, unatt))

    long_trials = [data[i] for i in long_trials_idx]
    return long_trials + merged


# -------------------------------------------------------------------------------
#                             Get Envelopes
# -------------------------------------------------------------------------------
def get_envelope(stim_base, sum_subbands=True):
    """
    Returns:
        env_out, weights

    If sum_subbands=True:
        env_out has shape (T, 1)

    If sum_subbands=False:
        env_out has shape (T, B)
    """
    npz = np.load(paths.envelope(f"{stim_base}_env.npz"))
    env = np.asarray(npz["envelope"], dtype=np.float32)
    w = np.asarray(npz["subband_weights"], dtype=np.float32)

    if env.ndim == 1:
        env = env[:, None]

    if sum_subbands:
        env_out = np.sum(env, axis=1, keepdims=True).astype(np.float32)

        # z-score AFTER summing
        mu = np.mean(env_out, axis=0, keepdims=True)
        sigma = np.std(env_out, axis=0, keepdims=True)
        env_out = (env_out - mu) / (sigma + 1e-8)

        return env_out.astype(np.float32), w

    # normalize each subband only when returning multiband
    mu = np.mean(env, axis=0, keepdims=True)
    sigma = np.std(env, axis=0, keepdims=True)
    env = (env - mu) / (sigma + 1e-8)

    return env.astype(np.float32), w


def get_attended(att_ear, env_left, env_right):
    s = str(att_ear).lower()
    if "left" in s or s.startswith("l") or s.endswith("l"):
        return env_left, env_right
    if "right" in s or s.startswith("r") or s.endswith("r"):
        return env_right, env_left
    raise ValueError(f"Unknown attended_ear value: {att_ear}")


def align_lengths(*arrays):
    m = min(len(a) for a in arrays)
    return tuple(a[:m] for a in arrays)


if __name__ == "__main__":
    Preprocess_Data()