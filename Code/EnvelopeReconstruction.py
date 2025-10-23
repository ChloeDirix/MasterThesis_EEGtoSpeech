import numpy as np             # For numerical arrays and math
import mne                     # EEG processing utilities (filtering, resampling)
import librosa                 # For loading audio (.wav) files and resampling
from scipy.signal import hilbert  # For signal processing (filtering, envelope)




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

