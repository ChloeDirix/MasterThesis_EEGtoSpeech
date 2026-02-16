import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, firwin, remez, freqz, resample_poly
from scipy.stats import zscore



# --------------------------
# Filter functions
# --------------------------

def rereference(eeg, method="Cz"):

    if method.lower() == "cz":
        CZ_INDEX = 47
        ref = eeg[:, [CZ_INDEX]]
        eeg_reref = eeg - ref
        return eeg_reref

    elif method.lower() == "mean":
        ref = np.mean(eeg, axis=1, keepdims=True)   # shape (samples, 1)
        eeg_reref = eeg - ref
        return eeg_reref

    # ---------- remove zero-variance channels --
    #eeg = eeg[:, np.std(eeg, axis=0) > 0]

    return eeg




def design_equiripple_bandpass(fs, HP, LP):
    Fst1 = HP - 0.45  # 0.55 Hz
    Fp1 = HP + 0.45  # 1.45 Hz
    Fp2 = LP - 0.45  # 8.55 Hz
    Fst2 = LP + 0.45  # 9.45 Hz
    
    bands = [
        0, Fst1,
        Fp1, Fp2,
        Fst2, fs / 2
    ]

    # Desired gain in each band:
    #   0 → stopband
    #   1 → passband
    desired = [0, 1, 0]

    # MATLAB filter lengths are usually ~500 taps
    numtaps = 513

    b = remez(
        numtaps,
        bands,
        desired,
        fs=fs
    )
    return b



# --------------------------
# EEG preprocessing
# --------------------------

def preprocess_trial(trial, cfg):

    eeg = trial.eeg_raw
    fs = trial.fs_eeg
    target_fs = cfg["preprocessing"]["target_fs"]

    band = cfg["preprocessing"]["band"]
    HP, LP = band
    plot_steps = cfg["preprocessing"]["plotting"]["show_preprocessing_steps"]
    plot_seconds = cfg["preprocessing"]["plotting"]["seconds"]

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
    eeg = rereference(eeg, cfg["preprocessing"]["rereference_method"])
    if plot_steps:
        axs[1].plot(t, eeg[:nplot, ch])
        axs[1].set_title("After rereferencing")

    # 2️⃣ band-pass filter
    b = design_equiripple_bandpass(fs,HP,LP )
    #plot_fir_filter_properties(b, fs)
    eeg = filtfilt(b, [1], eeg, axis=0)
    if plot_steps:
        axs[2].plot(t, eeg[:nplot, ch])
        axs[2].set_title(f"After high-pass ({LP} Hz)")


    # 4️⃣ Resample
    if int(fs) != target_fs:
        eeg = resample_poly(eeg, target_fs, fs, axis=0)
        fs = target_fs
        # factor = int(round(fs / target_fs))
        # eeg = eeg[::factor]
        # fs = target_fs

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

    # Z-score normalization
    #eeg = zscore(eeg, axis=0)

    return eeg, fs



def plot_fir_filter_properties(b, fs, title="Bandpass FIR Filter"):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Frequency response
    w, h = freqz(b, worN=4096)
    freqs = w * fs / (2 * np.pi)

    zeros = np.roots(b)
    poles = np.array([0])

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    # Magnitude
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(freqs, 20 * np.log10(np.abs(h) + 1e-12))
    ax1.set_title("Magnitude Response (dB)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)
    ax1.set_xlim(0, fs / 4)

    # Phase
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(freqs, np.unwrap(np.angle(h)))
    ax2.set_title("Phase Response")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (radians)")
    ax2.grid(True)
    ax2.set_xlim(0, fs / 2)

    # Impulse Response (fixed)
    ax3 = plt.subplot(2, 2, 3)
    markerline, stemlines, baseline = ax3.stem(b, basefmt=" ")
    plt.setp(stemlines, 'linewidth', 1)
    plt.setp(markerline, 'marker', '.')
    ax3.set_title("Impulse Response")
    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Amplitude")

    # Pole-Zero plot
    ax4 = plt.subplot(2, 2, 4)
    unit_circle = np.exp(1j * np.linspace(0, 2*np.pi, 400))
    ax4.plot(np.real(unit_circle), np.imag(unit_circle), 'k--', label="Unit circle")
    ax4.scatter(np.real(zeros), np.imag(zeros), color='blue', label="Zeros")
    ax4.scatter(np.real(poles), np.imag(poles), color='red', marker='x', s=80, label="Pole")
    ax4.set_aspect("equal", "box")
    ax4.set_title("Pole-Zero Plot")
    ax4.set_xlabel("Real")
    ax4.set_ylabel("Imag")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()
