import numpy as np
from math import gcd

from matplotlib import pyplot as plt
from mne.filter import filter_data, resample
from scipy.signal import butter, filtfilt, resample_poly
from scipy.stats import zscore


# --------------------------
# Filter functions
# --------------------------

def rereference(data):
    """Common average re-reference. eeg shape = (samples, channels)."""
    return data - np.mean(data, axis=1, keepdims=True)

def bandpass_filter(data, fs, low=1, high=9, order=4):
    nyq = 0.5 * fs

    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)
def anti_alias_filter(x, fs, cutoff):
    nyq = fs / 2
    if cutoff >= nyq:
        return x

    b, a = butter(5, cutoff / nyq, btype='low')
    return filtfilt(b, a, x)
def fast_resample(data, orig_fs, target_fs, axis=0):
    """Fast polyphase resampling that supports 1D or ND arrays (resamples along `axis`)."""
    if orig_fs == target_fs:
        return data
    cutoff = target_fs / 2 * 0.9
    data = anti_alias_filter(data, orig_fs, cutoff)
    g = gcd(int(orig_fs), int(target_fs))
    up = int(target_fs // g)
    down = int(orig_fs // g)
    # resample_poly expects data shaped (...,) or (N,...) - we use axis parameter
    return resample_poly(data, up, down, axis=axis)

# --------------------------
# EEG preprocessing
# --------------------------

def preprocess_trial(eeg,fs,cfg):

    target_fs = int(cfg["target_fs"])
    plot_seconds = cfg["plot_seconds"]
    plot_steps = cfg["plot_steps"]
    lpbe = cfg.get("lpbe", 1.0)  # high-pass cutoff (low border)
    upbe = cfg.get("upbe", 9.0)  # low-pass cutoff (high border)

    nplot = min(int(plot_seconds * fs), eeg.shape[0])
    t = np.arange(nplot) / fs
    ch = 0  # channel to plot

    if plot_steps:
        fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=False)
        fig.suptitle("EEG Preprocessing Pipeline (Reference-Matched)", fontsize=14)
        axs[0].plot(t, eeg[:nplot, ch])
        axs[0].set_title("Raw EEG")

    # 1️⃣ Re-reference
    eeg = rereference(eeg)
    if plot_steps:
        axs[1].plot(t, eeg[:nplot, ch])
        axs[1].set_title("After rereferencing")

    # 2️⃣ High-pass filter (remove slow drifts)
    eeg = filter_data(eeg.astype(np.float64), fs, l_freq=lpbe, h_freq=None, verbose='CRITICAL')
    if plot_steps:
        axs[2].plot(t, eeg[:nplot, ch])
        axs[2].set_title(f"After high-pass ({lpbe} Hz)")

    # 3️⃣ Low-pass filter (remove high-frequency noise)
    eeg = filter_data(eeg, fs, l_freq=None, h_freq=upbe, verbose='CRITICAL')
    if plot_steps:
        axs[3].plot(t, eeg[:nplot, ch])
        axs[3].set_title(f"After low-pass ({upbe} Hz)")

    # 4️⃣ Resample to target sampling rate
    if int(fs) != target_fs:
        eeg = resample(eeg, down=fs / target_fs)
        fs = target_fs
        nplot = min(int(plot_seconds * fs), eeg.shape[0])
        t = np.arange(nplot) / fs
        if plot_steps:
            axs[4].plot(t, eeg[:nplot, ch])
            axs[4].set_title(f"After resampling ({fs} Hz)")
    else:
        if plot_steps:
            axs[4].plot(t, eeg[:nplot, ch])
            axs[4].set_title("No resampling needed")

    # 5️⃣ Z-score per channel
    eeg = zscore(eeg, axis=0)

    return eeg
def preprocess_trial_old(eeg, fs, cfg):

    target_fs = int(cfg["target_fs"])
    band = tuple(cfg["band"])
    plot_seconds = cfg["plot_seconds"]
    plot_steps = cfg["plot_steps"]


    nplot = min(int(plot_seconds * fs), eeg.shape[0])
    t = np.arange(nplot) / fs
    ch = 0  # first channel to visualize

    if plot_steps:
        fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=False)
        fig.suptitle("EEG Preprocessing Pipeline", fontsize=14)

        axs[0].plot(t, eeg[:nplot, ch])
        axs[0].set_title("Raw EEG")


    # 1) reref
    eeg = rereference(eeg)
    if plot_steps:
        axs[1].plot(t, eeg[:nplot, ch])
        axs[1].set_title("After rereferencing")

    # 2) bandpass
    eeg = bandpass_filter(eeg, fs, low=band[0], high=band[1], order=3)

    if plot_steps:
        axs[2].plot(t, eeg[:nplot, ch])
        axs[2].set_title(f"After bandpass ({band[0]}–{band[1]} Hz)")
    # resample if needed
    if int(fs) != target_fs:
        eeg = fast_resample(eeg, int(fs), target_fs, axis=0)
        fs = target_fs
        nplot = min(int(plot_seconds * fs), eeg.shape[0])
        t = np.arange(nplot) / fs
        if plot_steps:
            axs[3].plot(t, eeg[:nplot, ch])
            axs[3].set_title(f"After downsampling ({fs} Hz)")

    eeg=zscore(eeg)
    return eeg


