# loaders/matlab_loader.py
from scipy.io import loadmat
from Loaders.DataModels import Subject, Trial
import numpy as np

class MatlabSubjectLoader:
    def __init__(self, mat_path: str, subject_id: str):
        self.mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        self.subject_id = subject_id

    def load(self) -> Subject:
        """Load .mat file and return a Subject object with Trial instances."""
        raw_trials = self.find_trials(self.mat)
        trials = []
        for i, raw_trial in enumerate(raw_trials, start=1):
            print(f"\n--- Trial {i} ---")
            trial = self._parse_trial(raw_trial, i)
            if trial.validate():
                trials.append(trial)
        return Subject(self.subject_id, trials)

    def _parse_trial(self, raw_trial, idx: int) -> Trial:
        """Extract EEG, stimuli, metadata, etc."""

        eeg_raw = self.get_field(raw_trial, "RawData").EegData
        eeg_arr = np.squeeze(np.array(eeg_raw))
        channels= self.get_field(self.get_field(raw_trial, "RawData"), "Channels")
        channel_names=channels
        #channel_names= [str(ch) for ch in np.atleast_1d(channels)]
        n_ch = int(np.asarray(channels).size)
        eeg_data = self.orient_eeg(eeg_arr, n_ch)
        fs_eeg = float(self.get_field(raw_trial.FileHeader, "SampleRate"))

        # Stimuli
        stim_names_orig = [str(s) for s in np.atleast_1d(self.get_field(raw_trial, "stimuli"))]
        stim_names = [
            stim_names_orig[0].replace("hrtf", "dry"),
            stim_names_orig[1].replace("hrtf", "dry"),
        ]
        attended_ear = self.get_field(raw_trial, "attended_ear")
        meta = {
            "attended_ear": attended_ear,
            "stim_names": stim_names,
            "trial_id": self.get_field(raw_trial.FileHeader, "TrialID"),
        }

        return Trial(index=idx, eeg_raw=eeg_data, fs_eeg=fs_eeg, channels=channel_names, metadata=meta)

    # --- Identify all valid trials in a MATLAB file ---
    def find_trials(self,mat):
        # Searches for keys like 'data' or selects the largest
        # non-metadata ndarray. Then checks for RawData/FileHeader fields.

        if "trials" in mat:
            cand = mat["trials"]   #candidate key
        elif "data" in mat:
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

        # loop through each element of the array.
        # skip any element that’s just a scalar, number, or string
        for el in arr:
            if isinstance(el, (int, float, np.integer, np.floating, str)):
                continue
            for key in ("RawData", "FileHeader"):
                try:
                    _ = self.get_field(el, key)
                    trials.append(el)
                    break
                except Exception:
                    continue
        return trials

    # Orient array correctly (samples (t) × channels) (necessary for mne)
    def orient_eeg(self,eeg_arr, n_ch):
        if n_ch is not None:
            if eeg_arr.shape[0] == n_ch:  # nr of rows equals nr of channels
                eeg = eeg_arr.T  # transpose
            elif eeg_arr.shape[1] == n_ch:  # nr of columns equals nr of channels
                eeg = eeg_arr
            else:
                # if no not right, the biggest dim is probably time
                eeg = eeg_arr.T if eeg_arr.shape[0] > eeg_arr.shape[1] else eeg_arr
        else:
            # n_ch=None
            # <256 is a plausible nr of electrodes and there are more time samples
            eeg = eeg_arr.T if eeg_arr.shape[0] <= 256 and eeg_arr.shape[1] > eeg_arr.shape[0] else eeg_arr
        return eeg

    def get_field(self,obj, name):
        """
        Retrieve a field `name` from a MATLAB struct or nested object.

        Works for:
          - scipy.io.loadmat object trees
          - MATLAB structs stored as object arrays or numpy.void types
          - Attributes (.RawData) or dict-like access
        """

        if hasattr(obj, name):  # if stored as attribute
            # print("obj is attribute")
            return getattr(obj, name)

        if isinstance(obj, dict) and name in obj:  # if stored as dict
            print("obj is dict")
            return obj[name]

        if isinstance(obj, np.ndarray) and obj.dtype.names:  # if stored as structured NumPy arrays
            val = obj[name]
            if isinstance(val, np.ndarray) and val.size == 1:  # if stored inside array
                print("obj is NumpyArray")
                return val.item()
            return val
        if isinstance(obj, np.ndarray):  # if stored in single python object wrapped in another array
            print("obj is wrapped in NumpyArray")
            for el in obj.flatten():
                try:
                    v = self.get_field(el, name)
                    if v is not None:
                        return v
                except Exception:
                    continue
        try:
            it = obj.item()  # extract it
            return self.get_field(it, name)  # recursion
        except Exception:
            pass
        raise KeyError(f"Field '{name}' not found in object of type {type(obj)}")
