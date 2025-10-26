import os
import MatlabHelper as MH
import numpy as np

class Trial:
    def __init__(self, index, cfg, subject_id ):
        self.index = index
        self.cfg = cfg
        self.subject_id=subject_id

        # data (eeg)
        self.eeg_raw = None
        self.eeg_arr = None  ##array version
        self.eeg_orient = None
        self.eeg_PP = None   ##preprocessed

        # setup info
        self.attended_ear = None
        self.stim_names = []
        self.fs = None
        self.channels_info= None

        #envelopes
        self.env_left = None
        self.env_right = None
        self.env_att = None
        self.env_unatt = None

    def setEEG_PP(self, EEG):
        self.eeg_PP=EEG
    def fill_trial_metadata(self, RawTrial):

        # --- Extract rawdata directory ---
        #     contains: EegData, channels, "Filter-info"
        try:
            rawdata = MH.get_field(RawTrial, "RawData")
            print(rawdata,RawTrial)
            print(f"-- raw data found for trial {self.index} --")
        except KeyError:
            print(f" Skipping trial {self.index}: no RawData field")
            return

        # nr of channels if available
        try:
            self.channels_info = MH.get_field(rawdata, "Channels")
            self.n_ch = int(np.asarray(self.channels_info).size)
            print(f"-- nr of channels found for trial {self.index}: ", self.n_ch)
        except Exception:
            print(f" Trial {self.index}: no channel Info")
            self.n_ch = None

        # EEG data
        try:
            self.eeg_raw = MH.get_field(rawdata, "EegData")
            print(f"-- eeg data found for trial {self.index} ")
        except KeyError:
            print(f"Skipping trial {self.index}: no EegData found")
            return

        self.eeg_arr = np.squeeze(np.array(self.eeg_raw))  # make data into array
        if self.eeg_arr.ndim != 2:     # check shape ok
            print(f" Skipping trial {self.index}: unexpected shape {self.eeg_arr.shape}")
            return

        self.eeg_orient = MH.orient_eeg(self.eeg_arr, self.n_ch)  #check the orientation


        # --- Extract stimuli ---
        try:
            self.stimuli = MH.get_field(RawTrial, "stimuli")
            self.stim_names = [str(s) for s in np.atleast_1d(self.stimuli)]
            print(f"-- stimuli for trial {self.index}: ", self.stimuli)
        except Exception:
            print(f" Trial {self.index}: no stimulus info")
            self.stim_names = []

        # --- Extract Attended ear ---
        try:
            self.attended_ear = MH.get_field(RawTrial, "attended_ear")
            print(f"-- attended ear for trial {self.index}: ", self.attended_ear)
        except Exception:
            print(f" Trial {self.index}: no attended ear info")
            self.attended_ear = None


        # --- extract fileheader ---
        # contains channelcount info, dataformat, electrodeCap, filetype, gains, recording, samplerate, trialID
        try:
            rawdata = MH.get_field(RawTrial, "RawData")
            FileHeader = MH.get_field(RawTrial, "FileHeader")
        except Exception:
            FileHeader = None



        # Sampling rate
        try:
            self.fs = float(MH.get_field(FileHeader, "SampleRate"))
        except Exception:
            self.fs = 128.0
            print(f" No sample rate found — assuming {self.fs} Hz")



    def validate(self):
        """
        Check if required data for trial processing is available.
        Returns True if trial is valid, False if anything crucial is missing.
        """
        valid = True

        required_fields = {
            "eeg_orient": self.eeg_orient,
            "fs": self.fs,
            "n_ch": getattr(self, "n_ch", None),
            "attended_ear": self.attended_ear,
            "stim_names": self.stim_names if len(self.stim_names) == 2 else None,
        }

        for field_name, value in required_fields.items():
            if value is None:
                print(f" Trial {self.index}: Missing required field → {field_name}")
                valid = False

        # Extra shape check for EEG: (channels × samples)
        if self.eeg_orient is not None and self.eeg_orient.ndim != 2:
            print(f" Trial {self.index}: EEG has wrong shape {self.eeg_orient.shape}")
            valid = False

        return valid


