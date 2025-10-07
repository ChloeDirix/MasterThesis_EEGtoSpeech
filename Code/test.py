import numpy as np

# Load the .npy file
data = np.load("sub-009_ses-shortstories01_task-listeningActive_run-05_desc-preproc-audio-audiobook_4_eeg.npy")

# Check what’s inside
print(type(data))   # Usually <class 'numpy.ndarray'>
print(data.shape)   # Shape of the array
print(data.dtype)   # Data type
print(data)         # Print array contents (be careful if it’s large)