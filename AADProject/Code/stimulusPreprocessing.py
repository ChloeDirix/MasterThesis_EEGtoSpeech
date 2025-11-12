import os

import librosa
import numpy as np
import yaml
from gammatone.filters import erb_filterbank, make_erb_filters, centre_freqs
from gammatone.gtgram import gtgram
from matplotlib import pyplot as plt


from scipy.signal import hilbert, resample_poly, gammatone, lfilter, filtfilt, butter
from scipy.stats import zscore
from mne.filter import resample, filter_data
from pathlib import Path



def PreprocessAudioFiles(cfg):
    target_fs = int(cfg["target_fs"])
    path=os.path.join(cfg["stim_dir"])
    out_dir_env = os.path.join(cfg["Env_dir"])
    os.makedirs(out_dir_env, exist_ok=True)

    for stimulus in Path(path).iterdir():
        if stimulus.is_file() and stimulus.suffix == ".wav" and "_dry" in stimulus.stem:
            print(stimulus.name)
            audio, fs_audio = librosa.load(stimulus, sr=None, mono=True)
            envelope,fs_env, cf=extract_envelope_das2019(audio,fs_audio, target_fs, cfg["band"][0],cfg["band"][1])
            print(len(envelope))
            np.save(os.path.join(out_dir_env,f"{stimulus.name}_env.npy"), envelope)

def construct_bpfilter(lowcut, highcut, fs):
    """Construct the bandpass filter equivalent to MATLAB’s equiripple filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='bandpass')
    return b, a

def extract_envelope_das2019(audio, fs_audio, target_fs=32, hp_cutoff=1, lp_cutoff=9,  plot=False):
    """
    Implements Das et al. (2019) gammatone envelope extraction.
    Equivalent to the MATLAB function preprocess_data() (audio part).
    """
    # === Parameters ===
    fs_intermediate_audio = 8000          # Hz
    fs_intermediate_env = 128             # Hz
    power = 0.6                           # power-law compression
    spacing = 1.5
    f_low, f_high = 150, 4000
    lowpass, highpass = lp_cutoff,hp_cutoff
    subband_envelopes = True              # multi-band
    bp_b, bp_a = construct_bpfilter(highpass, lowpass, fs_intermediate_env)

    # === 1. Resample audio to 8 kHz ===
    audio = librosa.resample(audio, orig_sr=fs_audio, target_sr=fs_intermediate_audio)
    fs_audio = fs_intermediate_audio

    # === 2. Apply gammatone filterbank ===
    cf = centre_freqs(fs_audio, spacing, f_low, f_high)
    erb_filters = make_erb_filters(fs_audio, cf)
    filtered_audio = erb_filterbank(audio, erb_filters)  # shape: (n_bands, n_samples)



    if isinstance(filtered_audio, list):
        min_len = min(len(band) for band in filtered_audio)
        filtered_audio = np.stack([band[:min_len] for band in filtered_audio], axis=0)
    elif filtered_audio.ndim == 1:
        filtered_audio = filtered_audio[np.newaxis, :]

    # === 3. Power-law envelope ===
    envelope = np.abs(filtered_audio) ** power

    # === 4. Downsample envelope to 128 Hz ===
    envelope = resample_poly(envelope, fs_intermediate_env, int(fs_audio), axis=-1)
    fs_env = fs_intermediate_env

    # === 5. Bandpass filter 1–9 Hz ===
    envelope = filtfilt(bp_b, bp_a, envelope, axis=-1)

    # === 6. Downsample to 32 Hz target ===
    envelope = resample_poly(envelope, target_fs, fs_env, axis=-1)
    fs_env = target_fs

    # === 7. Optional normalization ===
    #envelope = zscore(envelope, axis=-1)
    print(f"Envelope type: {type(envelope)}")
    if isinstance(envelope, (list, tuple)):
        print(f"List length: {len(envelope)}")
        print([np.shape(e) for e in envelope])
    else:
        print(f"Envelope shape: {np.shape(envelope)}")

    if plot:
        t_audio = np.arange(len(audio)) / fs_audio
        t_env = np.arange(envelope.shape[-1]) / fs_env
        plt.figure(figsize=(10, 4))
        plt.plot(t_audio, audio / np.max(np.abs(audio)), 'r', alpha=0.4, label="Audio")
        plt.plot(t_env, np.mean(envelope, axis=0) / np.max(np.mean(envelope, axis=0)),
                 'b', label="Mean envelope")
        plt.legend()
        plt.title("Gammatone Power-Law Envelope (Das et al., 2019)")
        plt.tight_layout()
        plt.show()

    band_energy = np.sqrt(np.mean(envelope ** 2, axis=1))
    weights = band_energy / np.sum(band_energy)
    broadband_env = np.average(envelope, axis=0, weights=weights)

    return broadband_env, fs_env, cf

def extract_envelope_gammatone(audio, fs_audio, target_fs, f_low=50, f_high=8000, n_filters=32, plot=False):
    audio = audio.astype(np.float64)
    audio = audio / np.max(np.abs(audio))  # Normalize amplitude

    # --- Compute gammatone spectrogram ---
    # gtgram returns an array of shape (n_filters, n_time_frames)
    win_time = 0.025  # 25 ms window
    hop_time = 0.010  # 10 ms hop (100 Hz temporal resolution)
    gt_env = gtgram(audio, fs_audio, win_time, hop_time, n_filters, f_low, f_high)
    # -> shape (n_filters, n_frames)

    # --- Envelope as mean across frequency bands ---
    env = np.mean(gt_env, axis=0)

    # --- Resample to target_fs ---
    current_fs = int(1 / hop_time)
    env = resample_poly(env, target_fs, current_fs)

    # --- Band-limit envelope (optional smoothing) ---
    env = filter_data(env, target_fs, None, 9, method='fir', phase='zero', verbose='CRITICAL')
    env = filter_data(env, target_fs, 1, None, method='fir', phase='zero', verbose='CRITICAL')

    # --- Normalize ---
    env = zscore(env)

    # --- Plot for inspection ---
    if plot:
        t_audio = np.arange(len(audio)) / fs_audio
        t_env = np.arange(len(env)) / target_fs
        plt.figure(figsize=(10, 4))
        plt.plot(t_audio, audio / np.max(np.abs(audio)), 'r', alpha=0.4, label="Audio")
        plt.plot(t_env, env / np.max(env), 'b', label="Gammatone Envelope")
        plt.legend()
        plt.title("Gammatone Envelope Extraction")
        plt.tight_layout()
        plt.show()

    return env

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

if __name__ == "__main__":
    cfg = yaml.safe_load(open("../config.yaml", "r"))
    PreprocessAudioFiles(cfg)





