from pathlib import Path

import librosa
import numpy as np
from gammatone.filters import erb_filterbank, make_erb_filters
from matplotlib import pyplot as plt
from mne.filter import filter_data
from scipy.signal import hilbert, resample_poly, filtfilt, remez
from scipy.stats import zscore

from paths import paths


## filter functions ------------------------
def construct_bpfilter_equiripple(fs, hp, lp, numtaps=513):

    # Transition edges exactly like MATLAB
    Fst1 = hp - 0.45
    Fp1  = hp + 0.45
    Fp2  = lp - 0.45
    Fst2 = lp + 0.45

    # Safety clamp
    Fst1 = max(Fst1, 0.0)
    Fst2 = min(Fst2, fs / 2.0)

    # Bands in Hz (remez can take Hz directly if Hz=fs is passed)
    bands = [0.0, Fst1, Fp1, Fp2, Fst2, fs / 2.0]
    desired = [0, 1, 0]  # stop, pass, stop

    # Equal weights (not exactly Ast1/Ap/Ast2, but close enough)
    b = remez(numtaps, bands, desired, fs=fs)
    a = 1.0
    return b, a


def freq2erb(f):
    return 21.4 * np.log10(4.37e-3 * f + 1)

def erb2freq(erb):
    return (10**(erb / 21.4) - 1) / 4.37e-3

def erbspacebw(f_low, f_high, spacing):
    erb_low = freq2erb(f_low)
    erb_high = freq2erb(f_high)
    erb_points = np.arange(erb_low, erb_high, spacing)
    return erb2freq(erb_points)

## actual code --------------------
def PreprocessAudioFiles(cfg,dataset):

    target_fs = int(cfg["preprocessing"]["target_fs"])
    print(f"[Audio] Saving subband envelopes (.npz) to: {paths.ENVELOPES}")

    
    for stimulus in (Path(paths.STIM_DAS).iterdir() if dataset == "DAS" else Path(paths.STIM_DTU).iterdir()):
        print(stimulus.name)

        is_wav = stimulus.is_file() and stimulus.suffix.lower() == ".wav"
        keep = is_wav and (("_dry" in stimulus.stem) if dataset == "DAS" else True)

        if keep:
            print(f"  - processing: {stimulus.name}")
            audio, fs_audio = librosa.load(stimulus, sr=None, mono=True)
            env, fs_env, cf, weights = extract_envelope_das2019(
                audio,
                fs_audio,
                target_fs=target_fs,
                hp_cutoff=cfg["preprocessing"]["band"][0],
                lp_cutoff=cfg["preprocessing"]["band"][1],
                plot=False,
            )

            out_path = paths.envelope(f"{stimulus.stem}_env.npz")
            np.savez(
                out_path,
                envelope=env,
                fs_env=np.array([fs_env], dtype=np.int32),
                cf=cf,
                subband_weights=weights,
            )
            print(f"  saved: {out_path} | shape={env.shape} | fs={fs_env}Hz | bands={env.shape[1]}")





def extract_envelope_das2019(audio, fs_audio, target_fs=32, hp_cutoff=1, lp_cutoff=9, plot=True):
    fs_intermediate_audio = 8000  # Hz
    fs_intermediate_env = 128  # Hz
    power = 0.6
    spacing = 1.5
    f_low, f_high = 150, 4000

    # 1) Resample audio to 8 kHz
    audio = resample_poly(audio, fs_intermediate_audio, fs_audio)
    fs_audio = fs_intermediate_audio

    # 2) Gammatone filterbank -> subbands
    cf = erbspacebw(f_low, f_high, spacing)  # center freqs
    erb_filt = make_erb_filters(fs_audio, cf)
    subbands = erb_filterbank(audio, erb_filt)  # (bands, samples)
    subbands = subbands.T  # (samples, bands)


    # 3) Powerlaw "envelopes"
    env = np.maximum(subbands, 0) ** power

    # 7) Subband weights: all ones (Das2019)
    subband_weights = np.ones(env.shape[1], dtype=np.float32)

    # 4) Resample envelopes to 128 Hz
    env = resample_poly(env, fs_intermediate_env, fs_intermediate_audio, axis=0)
    fs_env = fs_intermediate_env

    # 5) Band-pass 1–9 Hz using equiripple FIR (zero-phase)
    b_bp, a_bp = construct_bpfilter_equiripple(fs_env, hp_cutoff, lp_cutoff)
    env = filtfilt(b_bp, a_bp, env, axis=0)

    # 6) Downsample to target fs
    env=resample_poly(env, target_fs, fs_env)
    fs_env=target_fs

    # 7) zscore
    env=zscore(env, axis=0)




    if plot:
        diagnose_envelope(audio, fs_audio, env, fs_env, cf, seconds=3)

    return env, fs_env, cf, subband_weights


def extract_envelope_hilbert(audio, fs_audio, target_fs, hp_cutoff=1, lp_cutoff=9, plot=False):
    audio=audio.astype(np.float64)


    # --- Envelope extraction via Hilbert ---
    env = np.abs(hilbert(audio))

    # --- High-pass filter (remove DC) ---
    env = filter_data(env, fs_audio, hp_cutoff, None, method='fir', phase='zero', verbose='CRITICAL')

    # --- Low-pass filter envelope ---
    env = filter_data(env, fs_audio, None, lp_cutoff, method='fir', phase='zero', verbose='CRITICAL')

    # --- Resample to target_fs ---
    env = resample_poly(env, target_fs, fs_audio)

    # --- Normalize ---
    #env = zscore(env)

    if plot:
        t_audio = np.arange(len(audio)) / fs_audio
        t_env = np.arange(len(env)) / target_fs
        plt.figure(figsize=(10, 4))
        plt.plot(t_audio, audio / np.max(np.abs(audio)), 'r', alpha=0.5, label="Audio")
        plt.plot(t_env, env / np.max(env), 'b', label="Envelope")
        plt.legend()
        plt.title("Audio waveform vs. extracted envelope")
        plt.tight_layout()
        plt.show()

    return env


# diagnose plots ------------------------------------------
def diagnose_envelope(audio, fs_audio, env, fs_env, cf, seconds=3):
    print(f"Audio length: {len(audio)/fs_audio:.2f} sec")
    print(f"Envelope shape: {env.shape} (samples x bands)")
    print(f"Envelope fs: {fs_env} Hz")
    print(f"Num subbands: {env.shape[1]}")
    print(f"Center freqs: {cf}")

    plot_envelope_vs_audio(audio, fs_audio, env[:,0], fs_env, seconds)
    plot_subband_envelopes(env, fs_env, num_bands=5, seconds=seconds)
    plot_envelope_spectrum(env, fs_env, band_index=0)

def plot_envelope_vs_audio(audio, fs_audio, env, fs_env, seconds=3):
    """
    Plots raw audio and extracted envelope (resampled to audio rate)
    for the first 'seconds'.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import resample_poly

    # restrict window
    N_audio = int(seconds * fs_audio)
    N_env = int(seconds * fs_env)

    audio_seg = audio[:N_audio]
    env_seg = env[:N_env]

    # resample envelope to audio rate for visual overlay
    env_up = resample_poly(env_seg, fs_audio, fs_env)

    # normalize for plotting
    audio_plot = audio_seg / np.max(np.abs(audio_seg))
    env_plot = env_up / np.max(np.abs(env_up))

    # time axis
    t = np.arange(len(audio_plot)) / fs_audio

    plt.figure(figsize=(12, 4))
    plt.plot(t, audio_plot, label="Normalized Audio", alpha=0.5)
    plt.plot(t, env_plot, label="Envelope (up-sampled)", linewidth=2)
    plt.title("Audio vs Envelope (first {} seconds)".format(seconds))
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_envelope_spectrum(env, fs_env, band_index=0):
    import matplotlib.pyplot as plt
    from numpy.fft import rfft, rfftfreq

    y = env[:, band_index]
    Y = np.abs(rfft(y))
    f = rfftfreq(len(y), 1/fs_env)

    plt.figure(figsize=(8,4))
    plt.semilogy(f, Y)
    plt.title(f"Envelope Spectrum (band {band_index})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 20)
    plt.tight_layout()
    plt.show()

def plot_subband_envelopes(env, fs_env, num_bands=5, seconds=3):
    import matplotlib.pyplot as plt

    samples = int(seconds * fs_env)
    env_seg = env[:samples, :]
    n_bands = env_seg.shape[1]

    # select a few bands across the spectrum
    idx = np.linspace(0, n_bands - 1, num_bands).astype(int)

    plt.figure(figsize=(12, 8))
    for i, b in enumerate(idx):
        plt.subplot(num_bands, 1, i + 1)
        plt.plot(env_seg[:, b])
        plt.title(f"Subband {b}")
    plt.tight_layout()
    plt.show()









