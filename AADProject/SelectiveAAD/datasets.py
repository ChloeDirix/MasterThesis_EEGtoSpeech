import os
import numpy as np
import torch

from pynwb import NWBHDF5IO
from scipy.stats import zscore
from torch.utils.data import Dataset, DataLoader
from paths import paths


class SelectiveAADDataset(Dataset):
    """
    SAFE Windows version:
    - Does NOT store NWB file handles anywhere.
    - Opens + closes NWB inside __getitem__.
    - Uses num_workers=0 in DataLoader.

    Minimal modifications for stability.
    """

    def __init__(self, nwb_paths, cfg, multiband=True, split="train"):
        self.nwb_paths = nwb_paths
        self.cfg = cfg
        self.multiband = multiband
        self.split = split

        # Window params
        self.fs = cfg["target_fs"]
        self.win_len = int(cfg["DeepLearning"]["window_len_s"] * self.fs)
        self.win_step = int(cfg["DeepLearning"]["window_step_s"] * self.fs)

        # Build the window index: (path, trial_id, start_sample)
        self.index = []

        for nwb_path in nwb_paths:
            with NWBHDF5IO(nwb_path, "r") as io:
                nwb = io.read()
                trials = nwb.trials.to_dataframe()

                for trial_row in trials.itertuples():
                    trial_id = trial_row.Index + 1

                    # Load EEG length only — cheap
                    eeg = self._load_eeg(nwb, trial_id)
                    T = eeg.shape[0]

                    # Window start indices
                    starts = np.arange(0, T - self.win_len, self.win_step)
                    for s in starts:
                        self.index.append((nwb_path, trial_id, int(s)))

        print(f"{split} dataset: {len(self.index)} windows")

    # -------------------------------------------------
    # Low-level helpers
    # -------------------------------------------------

    def _load_eeg(self, nwb, trial_id):
        key = f"trial_{trial_id}_EEG_preprocessed"
        eeg = nwb.processing["eeg_preprocessed"].data_interfaces[key].data[:]
        return eeg  # (T, C)

    def _load_envelopes(self, trial_row):
        stimL_name = trial_row["stim_L_name"]
        stimR_name = trial_row["stim_R_name"]

        L_base = os.path.splitext(stimL_name)[0]
        R_base = os.path.splitext(stimR_name)[0]

        npzL = np.load(paths.envelope(f"{L_base}_env.npz"))
        npzR = np.load(paths.envelope(f"{R_base}_env.npz"))

        envL_sub = npzL["envelope"]
        envR_sub = npzR["envelope"]

        wL = npzL["subband_weights"]
        wR = npzR["subband_weights"]

        if self.multiband:
            return envL_sub, envR_sub
        else:
            return (
                envL_sub @ (wL / np.sum(wL)),
                envR_sub @ (wR / np.sum(wR)),
            )

    def _align_lengths(self, eeg, envL, envR):
        L = min(len(eeg), len(envL), len(envR))
        return eeg[:L], envL[:L], envR[:L]

    def _drop_constant_channels(self, eeg):
        ch_std = np.std(eeg, axis=0)
        keep_mask = ch_std > 0
        return eeg[:, keep_mask]

    def _compute_attended_index(self, attended_ear):
        ear = str(attended_ear).upper()
        return 0 if ear.endswith("L") else 1

    # -------------------------------------------------
    # PyTorch dataset API
    # -------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        nwb_path, trial_id, start = self.index[idx]

        # LOAD EVERYTHING FRESH (no caching of NWB objects)
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            trials_df = nwb.trials.to_dataframe()
            trial_row = trials_df.loc[trial_id - 1]

            # Load EEG
            eeg = self._load_eeg(nwb, trial_id)

            # Load envelopes
            envL, envR = self._load_envelopes(trial_row)


            # Clean + align
            eeg = self._drop_constant_channels(eeg)
            eeg, envL, envR = self._align_lengths(eeg, envL, envR)

            # Attended ear
            att_index = self._compute_attended_index(trial_row["attended_ear"])

            # Z-score
            eeg = zscore(eeg, axis=0)
            envL = zscore(envL, axis=0)
            envR = zscore(envR, axis=0)

            # Pack stimuli
            envelopes = np.stack([envL, envR], axis=0)  # [2, T, bands]

        # Window slicing
        end = start + self.win_len
        eeg_win = eeg[start:end]
        env_win = envelopes[:, start:end, :]

        # Convert to torch
        eeg_win = torch.tensor(eeg_win, dtype=torch.float32)
        env_win = torch.tensor(env_win, dtype=torch.float32)
        att_index = torch.tensor(att_index, dtype=torch.long)

        return eeg_win, env_win, att_index

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from paths import paths
    cfg=paths.load_config()
    nwb_paths=[paths.subject_eegPP("S1")]
    ds = SelectiveAADDataset(nwb_paths, cfg, multiband=True, split="debug")

    print(f"Dataset size (#windows): {len(ds)}\n")
    eeg, stim, att = ds[0]
    print("=== Single sample ===")
    print("EEG shape: ", eeg.shape) # [T, C_eeg]
    print("Stim shape: ", stim.shape) # [2, T, n_bands]
    print("Attended index: ", att.item()) # 0=L, 1=R

    # ------------------------- # Visualize first 5 seconds # -------------------------
    T = eeg.shape[0]
    t = torch.arange(T) / cfg["target_fs"]
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, eeg[:, 0].numpy())
    plt.title("EEG (channel 0)")
    plt.subplot(3, 1, 2)
    plt.plot(t, stim[0, :, 0].numpy())
    plt.title("Envelope LEFT (band 0)")
    plt.subplot(3, 1, 3)
    plt.plot(t, stim[1, :, 0].numpy())
    plt.title("Envelope RIGHT (band 0)")
    plt.tight_layout()
    plt.show()

    # ------------------------- # Test DataLoader batch # -------------------------
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    eeg_b, stim_b, att_b = next(iter(loader))
    print("\n=== Batch ===")
    print("EEG batch: ", eeg_b.shape) # [B, T, C_eeg]
    print("Stim batch: ", stim_b.shape) # [B, 2, T, n_bands]
    print("Att batch: ", att_b.shape, att_b.tolist())