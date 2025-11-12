# loaders/matlab_loader.py
from scipy.io import loadmat
from DataModels import Subject, Trial
import Code.MatlabHelper as MH
import numpy as np

class MatlabSubjectLoader:
    def __init__(self, mat_path: str, subject_id: str):
        self.mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        self.subject_id = subject_id

    def load(self) -> Subject:
        """Load .mat file and return a Subject object with Trial instances."""
        raw_trials = MH.find_trials(self.mat)
        trials = []
        for i, raw_trial in enumerate(raw_trials, start=1):
            print(f"\n--- Trial {i} ---")
            trial = self._parse_trial(raw_trial, i)
            if trial.validate():
                trials.append(trial)
        return Subject(self.subject_id, trials)

    def _parse_trial(self, raw_trial, idx: int) -> Trial:
        """Extract EEG, stimuli, metadata, etc."""

        eeg_raw = MH.get_field(raw_trial, "RawData").EegData
        eeg_arr = np.squeeze(np.array(eeg_raw))

        n_ch = int(np.asarray(MH.get_field(MH.get_field(raw_trial, "RawData"), "Channels")).size)
        eeg_data = MH.orient_eeg(eeg_arr, n_ch)
        fs_eeg = float(MH.get_field(raw_trial.FileHeader, "SampleRate"))

        # Stimuli
        stim_names_orig = [str(s) for s in np.atleast_1d(MH.get_field(raw_trial, "stimuli"))]
        stim_names = [
            stim_names_orig[0].replace("hrtf", "dry"),
            stim_names_orig[1].replace("hrtf", "dry"),
        ]
        attended_ear = MH.get_field(raw_trial, "attended_ear")
        meta = {
            "attended_ear": attended_ear,
            "stim_names": stim_names,
            "trial_id": MH.get_field(raw_trial.FileHeader, "TrialID"),
        }

        return Trial(index=idx, eeg_raw=eeg_data, fs_eeg=fs_eeg, metadata=meta)
