import numpy as np
from matplotlib import pyplot as plt
from mne.filter import filter_data
from scipy.signal import resample_poly
from scipy.stats import zscore

from DataModels import Trial


# --------------------------
# Filter functions
# --------------------------

def rereference(data):
    """Common average re-reference. eeg shape = (samples, channels)."""
    return data - np.mean(data, axis=1, keepdims=True)


# --------------------------
# EEG preprocessing
# --------------------------

def preprocess_trial(trial, cfg):

    eeg = trial.eeg_raw
    fs = trial.fs_eeg
    target_fs = cfg["target_fs"]
    band = cfg["band"]
    lpbe, upbe = band
    plot_steps = cfg["plot_steps"]
    plot_seconds = cfg["plot_seconds"]

    # For plotting
    if plot_steps:
        nplot = min(int(plot_seconds * fs), eeg.shape[0])
        t = np.arange(nplot) / fs
        ch = 0
        fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=False)
        fig.suptitle(f"EEG Preprocessing Pipeline — Trial {trial.index}", fontsize=14)
        axs[0].plot(t, eeg[:nplot, ch])
        axs[0].set_title("Raw EEG")

    # 1️⃣ Re-reference
    eeg = rereference(eeg)
    if plot_steps:
        axs[1].plot(t, eeg[:nplot, ch])
        axs[1].set_title("After rereferencing")

    # 2️⃣ High-pass filter
    eeg = filter_data(eeg.astype(np.float64), fs, l_freq=lpbe, h_freq=None, verbose='CRITICAL')
    if plot_steps:
        axs[2].plot(t, eeg[:nplot, ch])
        axs[2].set_title(f"After high-pass ({lpbe} Hz)")

    # 3️⃣ Low-pass filter
    eeg = filter_data(eeg, fs, l_freq=None, h_freq=upbe, verbose='CRITICAL')
    if plot_steps:
        axs[3].plot(t, eeg[:nplot, ch])
        axs[3].set_title(f"After low-pass ({upbe} Hz)")

    # 4️⃣ Resample
    if int(fs) != target_fs:
        eeg = resample_poly(eeg, target_fs, fs)
        fs = target_fs
        if plot_steps:
            nplot = min(int(plot_seconds * fs), eeg.shape[0])
            t = np.arange(nplot) / fs
            axs[4].plot(t, eeg[:nplot, ch])
            axs[4].set_title(f"After resampling ({fs} Hz)")
    elif plot_steps:
        axs[4].plot(t, eeg[:nplot, ch])
        axs[4].set_title("No resampling needed")

    # 5️⃣ (Optional) Z-score normalization
    #eeg = zscore(eeg, axis=0)

    trial.eeg_PP = eeg
    trial.fs_eeg = fs
    trial.metadata["preprocessed"] = True

    return trial





