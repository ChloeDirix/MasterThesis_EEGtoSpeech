"""
EEG–Speech Preprocessing Pipeline for KU Leuven AAD Dataset
-----------------------------------------------------------
"""
import os

import numpy as np             # For numerical arrays and math
import mne                     # EEG processing utilities (filtering, resampling)
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt # For signal processing (filtering, envelope)

from Code import Trial


def preprocess_trial(trial, cfg):
    #load parameters
    fs=trial.fs
    target_fs = cfg["target_fs"]
    band = tuple(cfg["band"])
    plot_seconds = cfg["plot_seconds"]
    plot_steps = cfg["plot_steps"]
    eeg=trial.eeg_orient

    out_dir=os.path.join(cfg["base_dir"], cfg["PP_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # --- plotting parameters ---
    nplot = min(int(plot_seconds * fs), eeg.shape[0])
    t = np.arange(nplot) / fs
    ch = 0  # first channel to visualize

    if plot_steps:
        fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=False)
        fig.suptitle("EEG Preprocessing Pipeline", fontsize=14)

        axs[0].plot(t, eeg[:nplot, ch])
        axs[0].set_title("Raw EEG")


    # --- Step 1: Rereference ---
    eeg = rereference(eeg)

    if plot_steps:
        axs[1].plot(t, eeg[:nplot, ch])
        axs[1].set_title("After rereferencing")

    # --- Step 2: Band-pass filter ---
    eeg = bandpass_filter(eeg, fs, band[0], band[1])

    if plot_steps:
        axs[2].plot(t, eeg[:nplot, ch])
        axs[2].set_title(f"After bandpass ({band[0]}–{band[1]} Hz)")

    # --- Step 3: Downsample ---
    if fs != target_fs:
        eeg = downsample_eeg(eeg, fs, target_fs)
        fs = target_fs
        nplot = min(int(plot_seconds * fs), eeg.shape[0])
        t = np.arange(nplot) / fs
        if plot_steps:
            axs[3].plot(t, eeg[:nplot, ch])
            axs[3].set_title(f"After downsampling ({fs} Hz)")

    # --- Step 4: Z-score normalization ---
    eeg = zscore_normalize(eeg)

    if plot_steps:
        axs[4].plot(t, eeg[:nplot, ch])
        axs[4].set_title("After normalization (z-score)")

        for ax in axs:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    #save preprocessed version in dir and in trial class
    np.save(os.path.join(out_dir, f"{trial.subject_id}_trial{trial.index:02d}_preprocessed.npy"), eeg)
    return eeg




# --------------------------------------------------------------------------------------------

# --- Band-pass filter (zero-phase Butterworth) ---
def bandpass_filter(eeg, fs, low=1, high=9, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, eeg, axis=0)


# --- Common-average rereferencing ---
def rereference(eeg):
    return eeg - np.mean(eeg, axis=1, keepdims=True)


# --- Downsampling (original fs --> target fs) ---
def downsample_eeg(eeg, orig_fs, target_fs):
    if orig_fs == target_fs:
        return eeg
    factor = int(round(orig_fs / target_fs))
    if factor < 1:
        raise ValueError("orig_fs must be >= target_fs")
    return mne.filter.resample(eeg, down=factor, npad="auto")


# --- Z-score normalization per channel ---
def zscore_normalize(eeg):
    mean = np.mean(eeg, axis=0)
    std = np.std(eeg, axis=0)
    std[std == 0] = 1.0
    return (eeg - mean) / std

