from pathlib import Path

import librosa
import numpy as np
from gammatone.filters import erb_filterbank, make_erb_filters
from matplotlib import pyplot as plt
from mne.filter import filter_data
from scipy.signal import hilbert, resample_poly, filtfilt, remez

from paths import paths


# ============================================================
# Filter helpers
# ============================================================
def construct_bpfilter_equiripple(fs, hp, lp, numtaps=513):
    """
    Equiripple FIR band-pass
    """
    if not (0 <= hp < lp < fs / 2):
        raise ValueError(f"Invalid band: hp={hp}, lp={lp}, fs={fs}")

    # Transition edges
    Fst1 = max(hp - 0.45, 0.0)
    Fp1 = hp + 0.45
    Fp2 = lp - 0.45
    Fst2 = min(lp + 0.45, fs / 2.0)

    if not (Fst1 <= Fp1 < Fp2 <= Fst2):
        raise ValueError(
            f"Invalid transition bands after clamping: "
            f"Fst1={Fst1}, Fp1={Fp1}, Fp2={Fp2}, Fst2={Fst2}"
        )

    bands = [0.0, Fst1, Fp1, Fp2, Fst2, fs / 2.0]
    desired = [0, 1, 0]

    b = remez(numtaps, bands, desired, fs=fs)
    a = 1.0
    return b, a


def freq2erb(f):
    return 21.4 * np.log10(4.37e-3 * f + 1.0)


def erb2freq(erb):
    return (10 ** (erb / 21.4) - 1.0) / 4.37e-3


def erbspacebw(f_low, f_high, spacing):
    erb_low = freq2erb(f_low)
    erb_high = freq2erb(f_high)
    erb_points = np.arange(erb_low, erb_high, spacing)
    return erb2freq(erb_points)


# ============================================================
# Main preprocessing
# ============================================================
from scipy.signal import butter, sosfiltfilt, resample_poly

def design_butter_bandpass(fs, HP, LP, order=4):
    return butter(order, [HP, LP], btype="bandpass", fs=fs, output="sos")


def PreprocessAudioFiles(cfg, dataset):
    """
    Saves multiband envelopes to .npz.
    """
    target_fs = int(cfg["preprocessing"]["target_fs"])

    hp_cutoff = cfg["preprocessing"]["band"][0]
    lp_cutoff = cfg["preprocessing"]["band"][1]

    stim_dir = Path(paths.STIM_DAS) if dataset == "DAS" else Path(paths.STIM_DTU)

    for stimulus in stim_dir.iterdir():
        print(stimulus.name)

        is_wav = stimulus.is_file() and stimulus.suffix.lower() == ".wav"
        keep = is_wav and (("_dry" in stimulus.stem) if dataset == "DAS" else True)

        if not keep:
            continue

        print(f"  - processing: {stimulus.name}")
        audio, fs_audio = librosa.load(stimulus, sr=None, mono=True)

        env, fs_env, cf, weights = extract_envelope_das2019(
            audio=audio,
            fs_audio=fs_audio,
            target_fs=target_fs,
            hp_cutoff=hp_cutoff,
            lp_cutoff=lp_cutoff,
            plot=False,
        )

        out_path = paths.envelope(f"{stimulus.stem}_env.npz")
        np.savez(
            out_path,
            envelope=env.astype(np.float32),              # multiband, NOT z-scored
            fs_env=np.array([fs_env], dtype=np.int32),
            cf=cf.astype(np.float32),
            subband_weights=weights.astype(np.float32),
        )
        print(
            f"  saved: {out_path} | shape={env.shape} | fs={fs_env} Hz | bands={env.shape[1]}"
        )


def extract_envelope_das2019(
    audio,
    fs_audio,
    target_fs=32,
    hp_cutoff=1,
    lp_cutoff=9,
    plot=False,
):
    """
    DAS-style multiband envelope extraction.

    Returns:
        env : shape (T, B), multiband envelopes, NOT normalized
        fs_env : envelope sampling rate
        cf : center frequencies
        subband_weights : all-ones for now
    """
    fs_intermediate_audio = 8000   # Hz
    fs_intermediate_env = 128      # Hz
    power = 0.6
    spacing = 1.5
    f_low, f_high = 150, 4000

    audio = np.asarray(audio, dtype=np.float64)

    # 1) Resample audio to 8 kHz
    audio_8k = resample_poly(audio, fs_intermediate_audio, fs_audio)
    fs_audio_8k = fs_intermediate_audio

    # 2) Gammatone filterbank -> subbands
    cf = erbspacebw(f_low, f_high, spacing)
    erb_filt = make_erb_filters(fs_audio_8k, cf)
    subbands = erb_filterbank(audio_8k, erb_filt)   # (bands, samples)
    subbands = subbands.T                           # (samples, bands)

    # 3) Envelope per subband
    env = np.abs(subbands) ** power

    # 4) Subband weights
    subband_weights = np.ones(env.shape[1], dtype=np.float32)

    # 5) Resample envelopes to intermediate env fs
    env = resample_poly(env, fs_intermediate_env, fs_audio_8k, axis=0)
    fs_env = fs_intermediate_env

    # 6) Band-pass filter subband envelopes
    #b_bp, a_bp = construct_bpfilter_equiripple(fs_env, hp_cutoff, lp_cutoff)
    #env = filtfilt(b_bp, a_bp, env, axis=0)
    sos = design_butter_bandpass(fs_env, hp_cutoff, lp_cutoff, order=4)
    env = sosfiltfilt(sos, env, axis=0)

    # 7) Downsample to target fs
    if fs_env != target_fs:
        env = resample_poly(env, target_fs, fs_env, axis=0)
        fs_env = target_fs

    if plot:
        diagnose_envelope(audio_8k, fs_audio_8k, env, fs_env, cf, seconds=3)

    return env.astype(np.float32), fs_env, cf, subband_weights


def extract_envelope_hilbert(audio, fs_audio, target_fs, hp_cutoff=1, lp_cutoff=9, plot=False):
    """
    Single-band Hilbert envelope alternative.
    """
    audio = np.asarray(audio, dtype=np.float64)

    env = np.abs(hilbert(audio))

    env = filter_data(
        env, fs_audio, hp_cutoff, None,
        method="fir", phase="zero", verbose="CRITICAL"
    )
    env = filter_data(
        env, fs_audio, None, lp_cutoff,
        method="fir", phase="zero", verbose="CRITICAL"
    )

    env = resample_poly(env, target_fs, fs_audio)
    env = env[:, None].astype(np.float32)

    if plot:
        t_audio = np.arange(len(audio)) / fs_audio
        t_env = np.arange(len(env)) / target_fs

        plt.figure(figsize=(10, 4))
        plt.plot(t_audio, audio / (np.max(np.abs(audio)) + 1e-12), alpha=0.5, label="Audio")
        plt.plot(t_env, env[:, 0] / (np.max(np.abs(env[:, 0])) + 1e-12), label="Envelope")
        plt.legend()
        plt.title("Audio waveform vs extracted envelope")
        plt.tight_layout()
        plt.show()

    return env


# ============================================================
# Diagnostics
# ============================================================
def diagnose_envelope(audio, fs_audio, env, fs_env, cf, seconds=3):
    print(f"Audio length: {len(audio) / fs_audio:.2f} sec")
    print(f"Envelope shape: {env.shape} (samples x bands)")
    print(f"Envelope fs: {fs_env} Hz")
    print(f"Num subbands: {env.shape[1]}")
    print(f"Center freqs: {cf}")

    plot_envelope_vs_audio(audio, fs_audio, env[:, 0], fs_env, seconds)
    plot_subband_envelopes(env, fs_env, num_bands=5, seconds=seconds)
    plot_envelope_spectrum(env, fs_env, band_index=0)


def plot_envelope_vs_audio(audio, fs_audio, env, fs_env, seconds=3):
    from scipy.signal import resample_poly

    N_audio = int(seconds * fs_audio)
    N_env = int(seconds * fs_env)

    audio_seg = audio[:N_audio]
    env_seg = env[:N_env]

    env_up = resample_poly(env_seg, fs_audio, fs_env)

    audio_plot = audio_seg / (np.max(np.abs(audio_seg)) + 1e-12)
    env_plot = env_up / (np.max(np.abs(env_up)) + 1e-12)

    t = np.arange(len(audio_plot)) / fs_audio

    plt.figure(figsize=(12, 4))
    plt.plot(t, audio_plot, label="Normalized Audio", alpha=0.5)
    plt.plot(t, env_plot, label="Envelope (up-sampled)", linewidth=2)
    plt.title(f"Audio vs Envelope (first {seconds} seconds)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_envelope_spectrum(env, fs_env, band_index=0):
    from numpy.fft import rfft, rfftfreq

    y = env[:, band_index]
    Y = np.abs(rfft(y))
    f = rfftfreq(len(y), 1 / fs_env)

    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Y)
    plt.title(f"Envelope Spectrum (band {band_index})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 20)
    plt.tight_layout()
    plt.show()


def plot_subband_envelopes(env, fs_env, num_bands=5, seconds=3):
    samples = int(seconds * fs_env)
    env_seg = env[:samples, :]
    n_bands = env_seg.shape[1]

    idx = np.linspace(0, n_bands - 1, num_bands).astype(int)

    plt.figure(figsize=(12, 8))
    for i, b in enumerate(idx):
        plt.subplot(num_bands, 1, i + 1)
        plt.plot(env_seg[:, b])
        plt.title(f"Subband {b}")
    plt.tight_layout()
    plt.show()