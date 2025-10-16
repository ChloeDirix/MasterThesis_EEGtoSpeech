"""
EEG–Speech Preprocessing Pipeline for KU Leuven AAD Dataset
-----------------------------------------------------------
This script:
- Loads MATLAB (.mat) EEG data from the KU Leuven AAD dataset.
- Handles multiple MATLAB struct formats robustly.
- Preprocesses EEG (rereference, band-pass 1–9 Hz, downsample, z-score).
- Extracts Hilbert speech envelopes from corresponding .wav files.
- Aligns EEG and speech data, produces quick diagnostic plots.
- Saves preprocessed numpy arrays for each trial.
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import loadmat


# ============================================================
# CONFIGURATION
# ============================================================

base_dir = r"C:\Users\lilou\OneDrive\Documenten"
data_dir = os.path.join(base_dir, "4004271")
stim_dir = os.path.join(data_dir, "stimuli")
subject_file = os.path.join(data_dir, "S1.mat")

target_fs = 128        # Final EEG sampling rate (Hz)
band = (2, 9)          # Band-pass filter range (Hz)
plot_seconds = 30      # Duration to visualize per trial (s)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

# --- Band-pass filter (zero-phase Butterworth) ---
def bandpass_filter(eeg, fs, low=1, high=9, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, eeg, axis=0)

# --- Common-average rereferencing ---
def rereference(eeg):
    return eeg - np.mean(eeg, axis=1, keepdims=True)

# --- Downsampling (originlal fs --> target fs) ---
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

# --- Speech envelope extraction using Hilbert + resample---
def extract_envelope(audio, fs_audio, target_fs):

    # Ensure correct type and shape
    audio = np.asarray(audio, dtype=np.float64).squeeze()

    # Get analytic signal -> envelope magnitude
    env = np.abs(hilbert(audio))

    # Low-pass (<9 Hz) to match EEG envelope dynamics
    env = mne.filter.filter_data(env, sfreq=fs_audio, l_freq=None, h_freq=9.0,
                                 method="fir", verbose=False)

    # Resample to EEG rate
    env = librosa.resample(y=env, orig_sr=fs_audio, target_sr=target_fs)

    # Normalize amplitude to [−1, 1]
    env /= np.max(np.abs(env))
    return env.astype(np.float32)

# --- Flexible field getter for MATLAB structs ---
def get_field(obj, name):
    """
    Retrieve a field `name` from a MATLAB struct or nested object.

    Works for:
      - scipy.io.loadmat object trees
      - MATLAB structs stored as object arrays or numpy.void types
      - Attributes (.RawData) or dict-like access
    """
    import numpy as _np
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    if isinstance(obj, _np.ndarray) and obj.dtype.names:
        val = obj[name]
        if isinstance(val, _np.ndarray) and val.size == 1:
            return val.item()
        return val
    if isinstance(obj, _np.ndarray):
        for el in obj.flatten():
            try:
                v = get_field(el, name)
                if v is not None:
                    return v
            except Exception:
                continue
    try:
        it = obj.item()
        return get_field(it, name)
    except Exception:
        pass
    raise KeyError(f"Field '{name}' not found in object of type {type(obj)}")

# --- Identify all valid trials in a MATLAB file ---
def find_trials(mat):
    """
    Detect the array of trials in a MATLAB dataset.

    Searches for keys like 'data' or selects the largest
    non-metadata ndarray. Then checks for RawData/FileHeader fields.
    """
    if "data" in mat:
        cand = mat["data"]
    else:
        cand = None
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray):
                if cand is None or v.size > getattr(cand, "size", 0):
                    cand = v
        if cand is None:
            raise ValueError("No candidate trial array found in .mat")

    arr = np.array(cand, copy=False).flatten()
    trials = []
    for el in arr:
        if isinstance(el, (int, float, np.integer, np.floating, str)):
            continue
        for key in ("RawData", "FileHeader"):
            try:
                _ = get_field(el, key)
                trials.append(el)
                break
            except Exception:
                continue
    return trials


# ============================================================
# MAIN PIPELINE
# ============================================================

# --- Load MATLAB dataset ---
mat = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
print("Top-level keys:", [k for k in mat.keys() if not k.startswith("__")])

# --- Identify trial entries ---
trials = find_trials(mat)
print(f"Found {len(trials)} trial-like entries")

# --- Prepare output directory ---
out_dir = os.path.join(base_dir, "preprocessed")
os.makedirs(out_dir, exist_ok=True)

# --- Process each trial individually ---
for i, trial in enumerate(trials, start=1):
    # Extract EEG and metadata fields
    try:
        rawdata = get_field(trial, "RawData")
    except KeyError:
        print(f"Skipping trial {i}: no RawData field")
        continue

    # Determine number of channels if available
    try:
        channels_info = get_field(rawdata, "Channels")
        n_ch = int(np.asarray(channels_info).size)
    except Exception:
        n_ch = None

    # Load EEG array
    try:
        eeg_raw = get_field(rawdata, "EegData")
    except KeyError:
        print(f"Skipping trial {i}: no EegData found")
        continue

    eeg_arr = np.squeeze(np.array(eeg_raw))
    if eeg_arr.ndim != 2:
        print(f"Skipping trial {i}: unexpected ndim {eeg_arr.ndim}, shape {eeg_arr.shape}")
        continue

    # Orient array correctly (samples × channels)
    if n_ch is not None:
        if eeg_arr.shape[0] == n_ch:
            eeg = eeg_arr.T
        elif eeg_arr.shape[1] == n_ch:
            eeg = eeg_arr
        else:
            eeg = eeg_arr.T if eeg_arr.shape[0] > eeg_arr.shape[1] else eeg_arr
    else:
        eeg = eeg_arr.T if eeg_arr.shape[0] <= 256 and eeg_arr.shape[1] > eeg_arr.shape[0] else eeg_arr

    # Sampling rate
    try:
        fs = float(get_field(get_field(trial, "FileHeader"), "SampleRate"))
    except Exception:
        try:
            fs = float(get_field(rawdata, "SampleRate"))
        except Exception:
            fs = 128.0
            print(f"Warning: sample rate not found for trial {i}, assuming {fs} Hz")

    print(f"\nTrial {i}: EEG shape {eeg.shape}, fs={fs}")

    # Attended ear and stimuli names
    try:
        attended_ear = get_field(trial, "attended_ear")
    except Exception:
        attended_ear = None
    try:
        stimuli = get_field(trial, "stimuli")
        stim_names = [str(s) for s in np.atleast_1d(stimuli)]
    except Exception:
        stim_names = []

    # --------------------------------------------------------
    # EEG PREPROCESSING
    # --------------------------------------------------------
    eeg = rereference(eeg)
    eeg = bandpass_filter(eeg, fs, *band)
    if fs > target_fs:
        eeg = downsample_eeg(eeg, orig_fs=fs, target_fs=target_fs)
        fs = target_fs
    eeg = zscore_normalize(eeg)

    # --------------------------------------------------------
    # SPEECH ENVELOPE PROCESSING
    # --------------------------------------------------------
    env_left = env_right = None
    if len(stim_names) >= 2:
        left_path = os.path.join(stim_dir, stim_names[0])
        right_path = os.path.join(stim_dir, stim_names[1])
        try:
            audio_left, fs_audio = librosa.load(left_path, sr=None, mono=True)
            audio_right, _ = librosa.load(right_path, sr=None, mono=True)
            env_left = extract_envelope(audio_left, fs_audio, target_fs)
            env_right = extract_envelope(audio_right, fs_audio, target_fs)
        except Exception as e:
            print(f"Could not load stimuli for trial {i}: {e}")

    # Match lengths between EEG and envelopes
    lens = [len(eeg)] + [len(env_left) for env_left in [env_left] if env_left is not None] + \
           [len(env_right) for env_right in [env_right] if env_right is not None]
    min_len = min(lens)
    eeg = eeg[:min_len]
    if env_left is not None:
        env_left = env_left[:min_len]
    if env_right is not None:
        env_right = env_right[:min_len]

    # Determine attended vs. unattended
    env_att = env_unatt = None
    if attended_ear and env_left is not None and env_right is not None:
        if str(attended_ear).upper().startswith("L"):
            env_att, env_unatt = env_left, env_right
        else:
            env_att, env_unatt = env_right, env_left
    else:
        env_att, env_unatt = env_left, env_right

    # --------------------------------------------------------
    # DIAGNOSTIC PLOTS
    # --------------------------------------------------------
    nplot = min(int(plot_seconds * fs), len(eeg))
    t = np.arange(nplot) / fs
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, eeg[:nplot, 0])
    plt.title(f"Trial {i} – EEG (channel 0)")
    plt.subplot(3, 1, 2)
    plt.plot(t, env_att[:nplot] if env_att is not None else [])
    plt.title("Attended envelope" if env_att is not None else "Attended envelope (missing)")
    plt.subplot(3, 1, 3)
    plt.plot(t, env_unatt[:nplot] if env_unatt is not None else [])
    plt.title("Unattended envelope" if env_unatt is not None else "Unattended envelope (missing)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------
    np.save(os.path.join(out_dir, f"S1_trial{i:02d}_EEG.npy"), eeg)
    if env_att is not None:
        np.save(os.path.join(out_dir, f"S1_trial{i:02d}_env_att.npy"), env_att)
    if env_unatt is not None:
        np.save(os.path.join(out_dir, f"S1_trial{i:02d}_env_unatt.npy"), env_unatt)

    print(f"Saved trial {i}: EEG {eeg.shape}, env_att {None if env_att is None else len(env_att)}")

