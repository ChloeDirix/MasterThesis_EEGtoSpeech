import numpy as np
from matplotlib import pyplot as plt
from mne.filter import filter_data
from scipy.signal import resample_poly, remez, filtfilt, firwin
from scipy.stats import zscore

from DataModels import Trial


# --------------------------
# Filter functions
# --------------------------

def rereference(data, cz_index=47):
    return data - data[:, [cz_index]]

def design_bandpass(fs, low=1.0, high=9.0, numtaps=513):
    """FIR bandpass filter similar to MATLAB equiripple but simpler + stable."""
    nyq = fs / 2
    # Use firwin with Hamming (MATLAB defaults similar for simple FIR)
    return firwin(numtaps, [low/nyq, high/nyq], pass_zero=False)


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
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
        fig.suptitle(f"EEG Preprocessing Pipeline — Trial {trial.index}", fontsize=14)
        axs[0].plot(t, eeg[:nplot, ch])
        axs[0].set_title("Raw EEG")

    # 1️⃣ Re-reference
    eeg = rereference(eeg)
    if plot_steps:
        axs[1].plot(t, eeg[:nplot, ch])
        axs[1].set_title("After rereferencing")

    # 2️⃣ band-pass filter
    b = design_bandpass(fs, low=band[0], high=band[1])
    eeg = filtfilt(b, [1], eeg, axis=0)
    if plot_steps:
        axs[2].plot(t, eeg[:nplot, ch])
        axs[2].set_title(f"After high-pass ({lpbe} Hz)")


    # 4️⃣ Resample
    if int(fs) != target_fs:
        factor = int(round(fs / target_fs))
        eeg = eeg[::factor]
        fs = target_fs
        if plot_steps:
            nplot = min(int(plot_seconds * fs), eeg.shape[0])
            t = np.arange(nplot) / fs
            axs[3].plot(t, eeg[:nplot, ch])
            axs[3].set_title(f"After resampling ({fs} Hz)")

    elif plot_steps:
        axs[3].plot(t, eeg[:nplot, ch])
        axs[3].set_title("No resampling needed")

    if plot_steps:
        plt.show()

    # 5️⃣ (Optional) Z-score normalization
    #eeg = zscore(eeg, axis=0)

    trial.eeg_PP = eeg
    trial.fs_eeg = fs
    trial.metadata["preprocessed"] = True

    return eeg, fs





