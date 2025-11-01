import os
import numpy as np
from matplotlib import pyplot as plt

import PlotUtils as PU
from scipy.signal import butter, filtfilt, hilbert, resample_poly, gammatone, lfilter
from scipy.stats import zscore
from math import gcd
from mne.filter import resample, filter_data


# --------------------------
# Filter functions
# --------------------------
def low_pass_filter(x, fs, cutoff=8.0, order=3):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)

def fast_resample(data, orig_fs, target_fs, axis=0):
    if orig_fs == target_fs:
        return data

    # Anti-alias before downsampling
    #cutoff = min(target_fs / 2 * 0.9, orig_fs / 2 * 0.9)
    #data = anti_alias_filter(data, orig_fs, cutoff)

    # Direct ratio (no integer rounding!)
    up = target_fs
    down = orig_fs

    data_resampled = resample_poly(data, up, down, axis=axis)

    # Correct amplitude scaling (resample_poly divides by down/up)
    data_resampled *= (down / up)

    return data_resampled

# --------------------------
# Preprocessing Audio
# --------------------------
def PreprocessAudioFiles(trial, eeg_PP, cfg, save=True):
    print(f"{trial.index} : envelope extraction started")

    plot_checks = cfg.get("plot_Envelope_Checks", False)
    target_fs = int(cfg["target_fs"])

    stimulusL = trial.stimulusL
    stimulusR = trial.stimulusR
    fs_right = trial.fs_right
    fs_left = trial.fs_left
    attended_ear = trial.attended_ear

    # --- Extract envelopes (no trimming here) ---
    env_left = extract_envelope_hilbert(stimulusL, fs_left, target_fs)
    env_right = extract_envelope_hilbert(stimulusR, fs_right, target_fs)

    # --- Align EEG and envelopes ---
    eeg_trim, env_left, env_right = align_lengths(eeg_PP, env_left, env_right)


    # --- Assign attended/unattended envelopes ---
    env_att, env_unatt = get_attended(attended_ear, env_left, env_right)

    # --- Plot checks ---
    if plot_checks:
        PU.plot_trial_diagnostics(trial.index, eeg_trim, cfg["target_fs"],
                                  env_att, env_unatt, cfg["plot_seconds"])

    # --- Save ---
    if save:
        save_files(eeg_trim, env_att, env_unatt, trial, cfg)

    print(f"{trial.index} : envelope extraction done")
    return eeg_trim, env_att, env_unatt


def extract_envelope_gammatone(audio, fs_audio, target_fs, bp=(0.5, 32.0), plot=False):
    audio = np.asarray(audio, dtype=np.float64)
    # --- Convert to mono if needed ---
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # --- Normalize PCM range if necessary ---
    if np.max(np.abs(audio)) > 100:
        audio = audio / 32768.0

    # --- Define intermediate sampling rates ---
    srInt1 = 8000.0  # for gammatone filterbank
    srInt2 = 128.0  # intermediate for post-filtering
    spacing = 1.5  # not used directly, but defines filter spacing
    freqs = np.array([
        178.7, 250.3, 334.5, 433.5, 549.9, 686.8, 847.7,
        1036.9, 1259.3, 1520.9, 1828.4, 2190.0, 2615.1,
        3114.9, 3702.6
    ])

    print(f"[DEBUG] fs_audio={fs_audio} → target_fs={target_fs}")

    # --- Band-limit audio before downsampling ---
    audio = filter_data(audio, fs_audio, None, srInt1 / 2, verbose='CRITICAL')

    # --- Resample to 8 kHz for gammatone ---
    audio = resample(audio, srInt1, fs_audio, verbose='CRITICAL')

    # --- Apply gammatone filterbank ---
    envs = []
    for f in freqs:
        b, a = gammatone(freq=f, ftype='fir', order=4, fs=srInt1)
        env = np.real(lfilter(b, a, audio))
        envs.append(np.abs(env) ** 0.6)  # compressive nonlinearity

    envs = np.array(envs)
    env_sum = np.sum(envs, axis=0)

    # --- Downsample to 128 Hz and band-pass filter (0.5–32 Hz) ---
    env_bp = resample(env_sum, srInt2, srInt1, verbose='CRITICAL')
    env_bp = filter_data(env_bp, srInt2, bp[0], bp[1], verbose='CRITICAL')

    # --- Final resampling to match EEG envelope sampling rate ---
    env_final = resample(env_bp, target_fs, srInt2, verbose='CRITICAL')

    # --- Z-score normalization ---
    env_z = zscore(env_final)

    print(f"[DEBUG] envelope mean={np.mean(env_z):.3f}, std={np.std(env_z):.3f}, len={len(env_z)}")

    # --- Optional plotting ---

    if plot:
        t_audio = np.arange(len(audio)) / srInt1
        t_env = np.arange(len(env_final)) / target_fs
        plt.figure(figsize=(10, 4))
        plt.plot(t_audio, audio / np.max(np.abs(audio)), 'r', alpha=0.5, label='Audio')
        plt.plot(t_env, env_final / np.max(env_final), 'b', label='Envelope')
        plt.legend()
        plt.title("Audio waveform vs. extracted envelope")
        plt.tight_layout()
        plt.show()

    return env_z

def extract_envelope_hilbert(audio, fs_audio, target_fs, lp_cutoff=9, plot=False):
    audio=audio.astype(np.float64)

    # --- High-pass filter (remove DC) ---
    audio=filter_data(audio.astype(float), fs_audio, 1, None, verbose='CRITICAL')

    # --- Envelope extraction via Hilbert ---
    env = np.abs(hilbert(audio))

    # --- Low-pass filter envelope ---
    env = filter_data(env, fs_audio, None, lp_cutoff, verbose='CRITICAL')

    # --- Resample to target_fs ---
    #env = resample(env, up=target_fs, down=fs_audio, verbose='CRITICAL')
    env = resample(env, down=fs_audio / target_fs)

    # --- Normalize ---
    env = zscore(env)

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

def extract_envelope_old(audio, fs_audio, target_fs):
    #audio = audio.astype(np.float32)

    # Detect and correct scaling
    if np.max(np.abs(audio)) > 100:  # likely int16 PCM
        audio /= 32768.0
    elif np.max(np.abs(audio)) == 0:
        print("Warning: silent audio detected!")
    # Resample
    print(f"[DEBUG] fs_audio={fs_audio}, target_fs={target_fs}")
    audio_resampled = fast_resample(audio, fs_audio, target_fs)
    fs_env = target_fs

    # Extract envelope
    print(f"[DEBUG] audio range before scaling: min={audio.min():.3f}, max={audio.max():.3f}")
    print(f"[DEBUG] after resample: min={audio_resampled.min():.3f}, max={audio_resampled.max():.3f}")
    env = np.abs(hilbert(audio_resampled))
    print(f"[DEBUG] envelope mean={np.mean(env):.6f}, std={np.std(env):.6f}")

    env = low_pass_filter(env, fs_env)
    t_audio = np.arange(len(audio)) / fs_audio
    t_env = np.arange(len(env)) / target_fs

    plt.figure(figsize=(10, 4))
    plt.plot(t_audio, audio / np.max(np.abs(audio)), 'r', alpha=0.5, label="Audio (norm.)")
    plt.plot(t_env, env / np.max(env), 'b', label="Envelope (norm.)")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.title("Audio vs Envelope (normalized)")
    plt.tight_layout()
    plt.show()

    # Normalize
    env = zscore(env)

    return env


def align_lengths(eeg, env_left, env_right):
    """Trim or pad so EEG and envelopes have equal length."""

    min_len = min(len(eeg), len(env_left), len(env_right))
    eeg=eeg[:min_len]
    env_left = env_left[:min_len]
    env_right = env_right[:min_len]
    print(len(eeg), len(env_left), len(env_right))
    return eeg, env_left, env_right


def get_attended(attended_ear, env_left, env_right):
    if attended_ear and env_left is not None and env_right is not None:
        if str(attended_ear).upper().startswith("L"):
            print(attended_ear," attended ear = L")
            return env_left, env_right
        else:
            print(attended_ear," attended ear = R")
            return env_right, env_left
    return env_left, env_right


def save_files(eeg, env_att, env_unatt, trial, cfg):
    out_dir_env = os.path.join(cfg["base_dir"], cfg["Env_dir"])
    os.makedirs(out_dir_env, exist_ok=True)
    out_dir_PP = os.path.join(cfg["base_dir"], cfg["PP_dir"])
    os.makedirs(out_dir_PP, exist_ok=True)

    np.save(os.path.join(out_dir_PP, f"{trial.subject_id}_trial{trial.index:02d}_preprocessed.npy"), eeg)
    np.save(os.path.join(out_dir_env, f"{trial.subject_id}_trial{trial.index:02d}_env_att.npy"), env_att)
    np.save(os.path.join(out_dir_env, f"{trial.subject_id}_trial{trial.index:02d}_env_unatt.npy"), env_unatt)
