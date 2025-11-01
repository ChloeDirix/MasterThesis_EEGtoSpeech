import os
import librosa
import numpy as np
import MatlabHelper as MH



class Trial:
    def __init__(self, index, cfg, subject_id):
        self.index = index
        self.cfg = cfg
        self.subject_id = subject_id

        base_dir = cfg["base_dir"]
        self.stim_dir = os.path.join(base_dir, cfg["stim_dir"])

        # --- EEG data ---
        self.eegRaw = None
        self.eeg_data = None

        # --- Metadata: EEG ---
        self.fs_eeg = None
        self.channels_info = None
        self.n_ch = None

        # --- Filter info (RawData) ---
        self.filter_impl = None
        self.filter_order = None
        self.highpass = None

        # --- Stimuli setup ---
        self.attended_ear = None
        self.stim_names = []
        self.stimulusL = None
        self.fs_left = None
        self.stimulusR = None
        self.fs_right = None

        # --- Preprocessing ---
        self.eeg_PP = None
        self.env_att = None
        self.env_unatt = None

        # --- FileHeader metadata ---
        self.channel_count = None
        self.data_format = None
        self.electrode_cap = None
        self.file_type = None
        self.file_type_id = None
        self.file_type_numeric = None
        self.gains = None
        self.recording_info = None
        self.trial_id = None
        self.subject = subject_id  # redundant but stored for convenience

    # -------------------------------------------------------------------------
    # Setters
    # -------------------------------------------------------------------------
    def setEEG_PP(self, EEG):
        self.eeg_PP = EEG

    def set_envelopes(self, env_att, env_unatt):
        self.env_att = env_att
        self.env_unatt = env_unatt

    # -------------------------------------------------------------------------
    # Metadata extractor
    # -------------------------------------------------------------------------
    def fill_trial_metadata(self, RawTrial):

        # --- RAWDATA ---
        try:
            rawdata = MH.get_field(RawTrial, "RawData")
            print(f"-- RawData found for trial {self.index} --")
        except KeyError:
            print(f"Skipping trial {self.index}: no RawData field")
            return

        # Channels
        try:
            self.channels_info = MH.get_field(rawdata, "Channels")
            self.n_ch = int(np.asarray(self.channels_info).size)
            print(f"-- Channels found: {self.n_ch} --")
        except Exception:
            print(f"Trial {self.index}: no channel info")
            self.n_ch = None

        # EEG data
        try:
            self.eegRaw = MH.get_field(rawdata, "EegData")
            eeg_arr = np.squeeze(np.array(self.eegRaw))
            if eeg_arr.ndim != 2:
                print(f"Skipping trial {self.index}: unexpected EEG shape {eeg_arr.shape}")
                return
            self.eeg_data = MH.orient_eeg(eeg_arr, self.n_ch)
            print(f"-- EEG data loaded: {self.eeg_data.shape} --")
        except Exception as e:
            print(f"Skipping trial {self.index}: no EegData found ({e})")
            return

        # Filter info (optional)
        try:
            self.filter_impl = MH.get_field(rawdata, "FilterImplementation")
            print("filterImpl ", self.filter_impl)
        except Exception:
            self.filter_impl = None
        try:
            self.filter_order = MH.get_field(rawdata, "FilterOrder")
            print("filterorder :",self.filter_order)
        except Exception:
            self.filter_order = None
        try:
            self.highpass = MH.get_field(rawdata, "HighPass")
            print("HP ",self.highpass)
        except Exception:
            self.highpass = None

        print(f"-- Filter info: impl={self.filter_impl}, order={self.filter_order}, HP={self.highpass}")

        # --- STIMULI ---
        try:
            stimuli = MH.get_field(RawTrial, "stimuli")
            stim_names_orig = [str(s) for s in np.atleast_1d(stimuli)]
            self.stim_names = [
                stim_names_orig[0].replace("hrtf", "dry"),
                stim_names_orig[1].replace("hrtf", "dry"),
            ]
            self.stimulusL, self.fs_left = self.loadstimulus(self.stim_names[0])
            self.stimulusR, self.fs_right = self.loadstimulus(self.stim_names[1])
            print(f"-- Stimuli: {self.stim_names}, fs_left={self.fs_left}, fs_right={self.fs_right}")
        except Exception as e:
            print(f"Trial {self.index}: no stimulus info ({e})")
            self.stim_names = []

        # --- ATTENDED EAR ---
        try:
            self.attended_ear = MH.get_field(RawTrial, "attended_ear")
            print(f"-- attended ear for trial {self.index}: ", self.attended_ear)
        except Exception:
            print(f" Trial {self.index}: no attended ear info")
            self.attended_ear = None
        print(f"-- Attended ear: {self.attended_ear}")

        # --- FILE HEADER ---
        try:
            FileHeader = MH.get_field(RawTrial, "FileHeader")
        except Exception:
            FileHeader = None

        if FileHeader is not None:
            self._extract_file_header(FileHeader)
        else:
            print(f"Trial {self.index}: No FileHeader found.")
            self.fs_eeg = 128.0


    def _extract_file_header(self, fh):
        try:
            self.channel_count = MH.get_field(fh, "ChannelCount")
            print("channel_count: ", self.channel_count)
        except Exception:
            self.channel_count = None

        try:
            self.data_format = MH.get_field(fh, "DataFormat")
            print("data_format: ", self.data_format)
        except Exception:
            self.data_format = None

        try:
            self.electrode_cap = MH.get_field(fh, "ElectrodeCap")
            print("electrodeCap: ", self.electrode_cap)
        except Exception:
            self.electrode_cap = None

        try:
            self.file_type = MH.get_field(fh, "FileType")
            print("filetype: ", self.file_type)
        except Exception:
            self.file_type = None

        try:
            self.file_type_id = MH.get_field(fh, "FileTypeId")
            print("fileTypeID: ", self.file_type_id)
        except Exception:
            self.file_type_id = None

        try:
            self.file_type_numeric = MH.get_field(fh, "FileTypeNumeric")
            print("fileTypenum: ", self.file_type_numeric)
        except Exception:
            self.file_type_numeric = None

        try:
            self.gains = MH.get_field(fh, "Gains")
            print("gains: ", self.gains)
        except Exception:
            self.gains = None

        try:
            self.recording_info = MH.get_field(fh, "Recording")
            print("recording: ", self.recording_info)
        except Exception:
            self.recording_info = None

        try:
            self.fs_eeg = float(MH.get_field(fh, "SampleRate"))
            print("fs_header: ", self.fs_eeg)
        except Exception:
            self.fs_eeg = 128.0

        try:
            self.subject = str(MH.get_field(fh, "Subject"))
            print("subject: ", self.subject)
        except Exception:
            pass

        try:
            self.trial_id = str(MH.get_field(fh, "TrialID"))
            print("trialID: ", self.trialID)
        except Exception:
            self.trial_id = f"trial_{self.index}"



    # -------------------------------------------------------------------------
    # Load stimulus audio
    # -------------------------------------------------------------------------
    def loadstimulus(self, stimName):
        path = os.path.join(self.stim_dir, stimName)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stimulus file not found: {path}")
        audio, fs_audio = librosa.load(path, sr=None, mono=True)
        return [audio, fs_audio]

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def validate(self):
        """Check if required data for trial processing is available."""
        valid = True
        required = {
            "eeg_data": self.eeg_data,
            "fs_eeg": self.fs_eeg,
            "n_ch": self.n_ch,
            "attended_ear": self.attended_ear,
            "stim_names": self.stim_names if len(self.stim_names) == 2 else None,
        }

        for name, value in required.items():
            if value is None:
                print(f"Trial {self.index}: Missing field → {name}")
                valid = False

        if self.eeg_data is not None and self.eeg_data.ndim != 2:
            print(f"Trial {self.index}: EEG shape {self.eeg_data.shape} invalid.")
            valid = False

        return valid

