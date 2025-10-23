import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import loadmat

import PreprocessingFull as PP
import EnvelopeReconstruction as ER
import FindData as Find
import PlotUtils as PU


class SubjectProcessor:
    """
    Class to handle loading, preprocessing, and saving EEG + envelope data for one subject.
    """

    def __init__(self, subject_id, cfg):
        #get everything out of the config file
        self.subject_id = subject_id
        self.base_dir = cfg["base_dir"]
        self.data_dir = os.path.join(self.base_dir, cfg["data_dir"])
        self.stim_dir = os.path.join(self.data_dir, cfg["stim_dir"])
        self.subject_file = os.path.join(self.data_dir, f"{subject_id}.mat")
        self.target_fs = cfg["target_fs"]
        self.band = tuple(cfg["band"])
        self.plot_seconds = cfg["plot_seconds"]
        self.plot_steps = cfg["plot_steps"]
        self.plot_Envelope_Checks=cfg["plot_Envelope_Checks"]

        # Prepare output directory
        self.out_dir = os.path.join(self.base_dir, "preprocessed")
        os.makedirs(self.out_dir, exist_ok=True)

        print(f"\n Loading subject {subject_id}...")

        #load the matlab file
        self.mat = loadmat(self.subject_file, squeeze_me=True, struct_as_record=False)
        print("Top-level keys:", [k for k in self.mat.keys() if not k.startswith("__")])

        #identify trial entries
        # --- Identify trial entries ---
        self.trials = Find.find_trials(self.mat)
        print(f" Found {len(self.trials)} trials")


    # Process all trials for this subject by looping over them
    def process_all(self):
        for i, trial in enumerate(self.trials, start=1):
            print(f"\n--- Trial {i} ---")
            self.process_trial(i, trial)

    # Run full preprocessing for one trial.
    def process_trial(self, i, trial):
        # Extract EEG and metadata fields
        try:
            rawdata = Find.get_field(trial, "RawData")
            print(f"-- raw data found for trial {i} --")
        except KeyError:
            print(f" Skipping trial {i}: no RawData field")
            return

        # Determine nr of channels if available
        try:
            channels_info = Find.get_field(rawdata, "Channels")
            n_ch = int(np.asarray(channels_info).size)
            print(f"-- nr of channels found for trial {i}: ", n_ch)
        except Exception:
            n_ch = None

        # EEG data
        try:
            eeg_raw = Find.get_field(rawdata, "EegData")
            print(f"-- eeg data found for trial {i} ")
        except KeyError:
            print(f"Skipping trial {i}: no EegData found")
            return

        eeg_arr = np.squeeze(np.array(eeg_raw))
        if eeg_arr.ndim != 2:
            print(f" Skipping trial {i}: unexpected shape {eeg_arr.shape}")
            return

        eeg = Find.orient_eeg(eeg_arr, n_ch)

        # Sampling rate
        try:
            fs = float(Find.get_field(Find.get_field(trial, "FileHeader"), "SampleRate"))
            print(f"-- fs found for trial {i}: ", fs)
        except Exception:
            try:
                fs = float(Find.get_field(rawdata, "SampleRate"))
            except Exception:
                fs = 128.0
                print(f" No sample rate found — assuming {fs} Hz")


        # Attended ear + stimuli names
        try:
            attended_ear = Find.get_field(trial, "attended_ear")
            print(f"-- attended ear for trial {i}: ", attended_ear)
        except Exception:
            attended_ear = None
        try:
            stimuli = Find.get_field(trial, "stimuli")
            stim_names = [str(s) for s in np.atleast_1d(stimuli)]
            print(f"-- stimuli for trial {i}: ", stimuli)
        except Exception:
            stim_names = []


        # ------------------ EEG preprocessing ------------------

        eeg_pre, fs = PP.preprocess_eeg(eeg, fs, self.target_fs, self.band, self.plot_steps, self.plot_seconds)


        # ------------------ Speech envelope extraction ------------------
        env_left, env_right = None, None
        if len(stim_names) >= 2:
            # each trail has metadata telling the audio files that were played
            left_path = os.path.join(self.stim_dir, stim_names[0])
            right_path = os.path.join(self.stim_dir, stim_names[1])
            try:
                # load both stimuli using librosa
                audio_left, fs_audio = librosa.load(left_path, sr=None, mono=True)
                audio_right, _ = librosa.load(right_path, sr=None, mono=True)
                # compute speech envelopes
                env_left = ER.extract_envelope(audio_left, fs_audio, self.target_fs)
                env_right = ER.extract_envelope(audio_right, fs_audio, self.target_fs)
            except Exception as e:
                print(f"Could not load stimuli for trial {i}: {e}")

        # Match lengths
        eeg_pre, env_left, env_right = Find.align_lengths(eeg_pre, env_left, env_right)

        # Attended vs unattended
        env_att, env_unatt = Find.get_attended(attended_ear, env_left, env_right)

        # Diagnostic plot
        if self.plot_Envelope_Checks:
            PU.plot_trial_diagnostics(i, eeg_pre, fs, env_att, env_unatt, self.plot_seconds)

        # Save output
        np.save(os.path.join(self.out_dir, f"{self.subject_id}_trial{i:02d}_EEG.npy"), eeg_pre)
        if env_att is not None:
            np.save(os.path.join(self.out_dir, f"{self.subject_id}_trial{i:02d}_env_att.npy"), env_att)
        if env_unatt is not None:
            np.save(os.path.join(self.out_dir, f"{self.subject_id}_trial{i:02d}_env_unatt.npy"), env_unatt)

        print(f" Saved trial {i}: EEG {eeg_pre.shape}, env_att {None if env_att is None else len(env_att)}")
