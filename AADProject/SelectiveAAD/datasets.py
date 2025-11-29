import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pynwb import NWBHDF5IO
from scipy.stats import zscore
from paths import paths


class AADDataset(Dataset):

    def __init__(self, nwb_paths, cfg, multiband=True, split="train"):
        self.cfg = cfg
        self.multiband = multiband
        self.split = split

        self.fs = cfg["target_fs"]
        self.win_len = int(cfg["DeepLearning"]["window_len_s"] * self.fs)
        self.win_step = int(cfg["DeepLearning"]["window_step_s"] * self.fs)

        # ---------------------------------------------------------
        # STORAGE
        # Each element: dict with keys:
        #   "eeg", "env", "att", "T"
        # ---------------------------------------------------------
        self.trials = []
        self.index = []  # list of (trial_idx, start_sample)

        # ---------------------------------------------------------
        # LOAD AND PREPROCESS ALL TRIALS
        # ---------------------------------------------------------
        for nwb_path in nwb_paths:
            with NWBHDF5IO(nwb_path, "r") as io:
                nwb = io.read()
                trials_df = nwb.trials.to_dataframe()

                for row in trials_df.itertuples():
                    trial_id = row.Index + 1

                    # ---- 1. Load EEG ----
                    key = f"trial_{trial_id}_EEG_preprocessed"
                    eeg = nwb.processing["eeg_preprocessed"].data_interfaces[key].data[:]
                    # Shape: (T, C)

                    # Drop constant channels once
                    std = np.std(eeg, axis=0)
                    keep = std > 0
                    eeg = eeg[:, keep]

                    # ---- 2. Load envelopes ----
                    stimL = row.stim_L_name
                    stimR = row.stim_R_name

                    L_base = os.path.splitext(stimL)[0]
                    R_base = os.path.splitext(stimR)[0]

                    L_npz = np.load(paths.envelope(f"{L_base}_env.npz"))
                    R_npz = np.load(paths.envelope(f"{R_base}_env.npz"))

                    envL = L_npz["envelope"]
                    envR = R_npz["envelope"]

                    if not multiband:
                        wL = L_npz["subband_weights"]
                        wR = R_npz["subband_weights"]
                        envL = envL @ (wL / np.sum(wL))
                        envR = envR @ (wR / np.sum(wR))
                        envL = envL[:, None]  # (T,1)
                        envR = envR[:, None]  # (T,1)

                    # ---- 3. Align lengths ----
                    Lmin = min(len(eeg), len(envL), len(envR))
                    eeg = eeg[:Lmin]
                    envL = envL[:Lmin]
                    envR = envR[:Lmin]

                    # ---- 4. Z-score once ----
                    eeg = zscore(eeg, axis=0)
                    envL = zscore(envL, axis=0)
                    envR = zscore(envR, axis=0)

                    # ---- 5. Pack envelopes ----
                    envelopes = np.stack([envL, envR], axis=0)  # (2, T, bands)

                    # ---- 6. Attended index ----
                    ear = str(row.attended_ear).upper()
                    att = 0 if ear.endswith("L") else 1

                    # ---- 7. Store preprocessed trial ----
                    tdict = {
                        "eeg": eeg,            # (T, C)
                        "env": envelopes,      # (2, T, B)
                        "att": att,
                        "T": Lmin
                    }
                    trial_idx = len(self.trials)
                    self.trials.append(tdict)

                    # ---- 8. Build window indices for this trial ----
                    starts = np.arange(0, Lmin - self.win_len, self.win_step)
                    for s in starts:
                        self.index.append((trial_idx, int(s)))

        print(f"{split} dataset: {len(self.index)} windows")

    # ----------------------------------------------------------
    # PyTorch Dataset API
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
