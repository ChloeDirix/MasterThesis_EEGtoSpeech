import os                                   #filesystem paths (cross-platform)
import numpy as np
import mne                                  #EEG helper functions (resampling, plotting)
from scipy.signal import butter, filtfilt   #design + apply a zero-phase Butterworth filter
from scipy.io import loadmat                #read matlab files into python objects


# -----------------------------
# filepaths
# -----------------------------
base_dir = r"C:\Users\lilou\Onedrive\Documenten\thesis"
eeg_dir = os.path.join(base_dir, "4004271")
subject_file = os.path.join(eeg_dir, "S1.mat")



# -----------------------------
# Load MATLAB file
# -----------------------------

#read .mat file into a pyhton dict
## ---- squeeze_me: compresses singleton dimensions
## ---- struct_as_record: makes dict records
subject = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
print("Loaded keys:", subject.keys())

# Find the data
try:
    #data = subject["data"]
    data=subject["trials"]
except KeyError:
    # fallback: search for numpy array
    for k in subject.keys():
        if isinstance(subject[k], np.ndarray):
            data = subject[k]
            print(k)
            break

#print(f"Data type: {type(data)}")

print("trials",dir(data))
print("trial1",data[0])
print(data[:])



# -----------------------------
# Helper functions
# -----------------------------

#zero-phase Butterworth band-pass filter
def bandpass_filter(eeg, fs, low=1, high=9, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, eeg, axis=0)

# Re-reference to average of all channels
def rereference(eeg):
    return eeg - np.mean(eeg, axis=1, keepdims=True)

#downsample EEG
def downsample_eeg(eeg, orig_fs, target_fs):
    return mne.filter.resample(eeg, down=orig_fs/target_fs, npad="auto")

#normalize channels
def zscore_normalize(eeg):
    """Z-score normalization per channel."""
    return (eeg - np.mean(eeg, axis=0)) / np.std(eeg, axis=0)

"""
# -----------------------------
# Example: process first trial
# -----------------------------
# The structure might be something like subject['data'][0].RawData.EegData
trial = data[0] if isinstance(data, (list, np.ndarray)) else data
print(trial.RawData.EegData.T)
eeg = trial.RawData.EegData.T  # transpose to shape (samples, channels)
fs = trial.FileHeader.SampleRate
print(f"EEG shape: {eeg.shape}, fs: {fs}")

# 1. Re-reference
eeg = rereference(eeg)

# 2. Band-pass filter 1–9 Hz
eeg = bandpass_filter(eeg, fs, low=1, high=9)

# 3. Downsample (if >128 Hz)
target_fs = 128
if fs > target_fs:
    eeg = mne.filter.resample(eeg, down=fs/target_fs, npad="auto")
    fs = target_fs

# 4. Normalize
eeg = zscore_normalize(eeg)

print(f"Preprocessed EEG shape: {eeg.shape}, final fs: {fs} Hz")

# -----------------------------
# Save or continue with speech alignment
# -----------------------------
out_dir = os.path.join(base_dir, "preprocessed")
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "S1_trial1_eeg.npy"), eeg)

print("✅ Preprocessing complete. Saved preprocessed EEG for first trial.")
"""