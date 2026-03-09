import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pynwb import NWBHDF5IO
from scipy.stats import zscore
from paths import paths

import os
import numpy as np
from paths import paths

# =============================================================================
# Helper functions (datapreparation)
# =============================================================================
def _get_envelope(stim_base: str, multiband: bool = True):
    npz = np.load(paths.envelope(f"{stim_base}_env.npz"))
    env = npz["envelope"]
    w = npz["subband_weights"]

    if multiband:
        return env, w
    else:
        w_norm = w / np.sum(w)
        env_mono = env @ w_norm
        return env_mono[:, None], w_norm  # keep shape (T,1)

def _align_lengths(*arrays):
    m = min(len(a) for a in arrays)
    return tuple(a[:m] for a in arrays)

def _attended_index(att_ear):
    """returns index for (L,R)"""
    s = str(att_ear).lower()
    if "left" in s or s.startswith("l") or s.endswith("l"):
        return 0
    if "right" in s or s.startswith("r") or s.endswith("r"):
        return 1
    else: 
        print("no attended index found")

def _load_data_from_trial(nwbfile, tr_row, cfg, multiband: bool):
    """
    Loads EEG and L/R envelopes for a trial.
    Returns: eeg(T,C), envL(T,B), envR(T,B), att_idx(0/1), dataset_key(str)
    """

    # ---------------- EEG ----------------
    if "trial_index" not in tr_row:
        raise KeyError("trials_df does not contain 'trial_index' column, cannot map to EEG interfaces.")

    tid = int(tr_row["trial_index"])

    # Try a few common index conventions (some files store 0-based, some 1-based)
    candidates = [
        f"trial_{tid}_EEG_preprocessed",
        f"trial_{tid+1}_EEG_preprocessed",
        f"trial_{tid-1}_EEG_preprocessed",
    ]

    di = nwbfile.processing["eeg_preprocessed"].data_interfaces

    eeg_key = None
    for k in candidates:
        if k in di:
            eeg_key = k
            break

    if eeg_key is None:
        # Give a helpful error message
        available = list(di.keys())
        raise KeyError(
            f"None of {candidates} found in eeg_preprocessed.data_interfaces.\n"
            f"Example available keys: {available[:20]}"
        )

    eeg = di[eeg_key].data[:]
    eeg = np.asarray(eeg)


    # remove extra channels
    nC = cfg["preprocessing"]["target_n_channels"]   
    if eeg.shape[1] < nC:
        raise ValueError(f"Trial {tid}: EEG has {eeg.shape[1]} channels, expected >= {nC}")
    eeg = eeg[:, :nC]

    # envelopes
    stimL = os.path.splitext(str(tr_row["stim_L_name"]))[0]
    stimR = os.path.splitext(str(tr_row["stim_R_name"]))[0]

    envL, _ = _get_envelope(stimL, multiband=multiband)
    envR, _ = _get_envelope(stimR, multiband=multiband)

    eeg, envL, envR = _align_lengths(eeg, envL, envR)

    # attended stimulus
    att = _attended_index(tr_row["attended_ear"])
    ds_key = str(tr_row.get("dataset", "")).upper()

    return eeg, envL, envR, att, ds_key



class AADDataset(Dataset):
    """
    PyTorch Dataset for auditory attention decoding (AAD) from EEG.
    Loads preprocessed EEG and stimulus envelopes from NWB files, and creates windows from them.

    Input sample:
      EEG window:     (win_len, C_eeg)
      ENV window:     (2, win_len, B)  where 0=left, 1=right
      attended index: scalar (0 or 1) --> 0=unattended, 1=attended
    """

    def __init__(self, nwb_paths, cfg, multiband=True, split="train"):
        self.cfg = cfg
        self.multiband = multiband
        self.split = split

        self.fs = cfg["preprocessing"]["target_fs"]
        self.win_len = int(cfg["DeepLearning"]["data_windows"]["window_len_s"] * self.fs)
        self.win_step = int(cfg["DeepLearning"]["data_windows"]["window_step_s"] * self.fs)

        
        self.trials = []
        self.index = []
        self.sample_dataset_keys = []   #DTU/DAS
        self.sample_subject_keys = []   #S1_DTU
        # ---------------- Load all trials ----------------
        for nwb_path in nwb_paths:
            with NWBHDF5IO(nwb_path, "r") as io:
                nwb = io.read()
                trials_df = nwb.trials.to_dataframe()
                subj_key = os.path.splitext(os.path.basename(str(nwb_path)))[0]

                # Use iterrows() to align with your linear pipeline style
                for idx, tr_row in trials_df.iterrows():
                    eeg, envL, envR, att, ds_key = _load_data_from_trial(
                        nwbfile=nwb,
                        tr_row=tr_row,
                        cfg=self.cfg,
                        multiband=self.multiband,
                    )

                    # Z-score per trial
                    eeg = zscore(eeg, axis=0)
                    envL = zscore(envL, axis=0)
                    envR = zscore(envR, axis=0)

                    # Pack envelopes: (2, T, B)
                    envelopes = np.stack([envL, envR], axis=0)
                    T = len(eeg)

                    # Store trial in a dict
                    tdict = {
                        "eeg": eeg,
                        "env": envelopes,
                        "att": int(att),
                        "T": int(T),
                        "dataset": ds_key,
                    }
                    trial_idx = len(self.trials)
                    self.trials.append(tdict)

                    # Build window indices for this trial
                    if T > self.win_len:
                        starts = np.arange(0, T - self.win_len, self.win_step)
                        for s in starts:
                            self.index.append((trial_idx, int(s)))
                            self.sample_dataset_keys.append(ds_key)
                            self.sample_subject_keys.append(subj_key)
                            
        print(f"{split} dataset: {len(self.index)} windows")
        print(split, "unique dataset keys:", sorted(set(self.sample_dataset_keys))[:10])
        
    # ----------------------------------------------------------
    # PyTorch Dataset
    # ----------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        trial_idx, start = self.index[idx]
        tinfo = self.trials[trial_idx]

        end = start + self.win_len

        eeg_win = tinfo["eeg"][start:end]           # (T, C)
        env_win = tinfo["env"][:, start:end, :]     # (2, T, B)
        att = tinfo["att"]

        return (
            torch.tensor(eeg_win, dtype=torch.float32),
            torch.tensor(env_win, dtype=torch.float32),
            torch.tensor(att, dtype=torch.long)
        )
