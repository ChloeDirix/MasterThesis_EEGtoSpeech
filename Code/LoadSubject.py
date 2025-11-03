import os

import numpy as np
from scipy.io import loadmat

from Code.Trial import Trial
from MatlabHelper import get_field, orient_eeg, find_trials

class LoadSubject:
    def __init__(self,subject_id, cfg):

        self.subject_id=subject_id
        self.cfg=cfg

        #filepaths
        self.base_dir = cfg["base_dir"]
        self.data_dir = os.path.join(self.base_dir, cfg["data_dir"])

        # self.PP_dir = os.path.join(self.base_dir, cfg["PP_dir"])
        # self.Env_dir= os.path.join(self.base_dir, cfg["Env_dir"])

        #Find subject data with the subject ID
        subject_file = os.path.join(self.data_dir, f"{subject_id}.mat")
        self.mat = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
        #self.dataExploration()
        self.Rawtrials = find_trials(self.mat)
        self.trials=[]
        self.LoadTrials()


    def dataExploration(self):
        for i, t in enumerate(self.mat['trials'][:1], start=1):
            print(f"\nTRIAL {i}")
            print("- RawData keys:", dir(t.RawData))
            print("- stimuli keys:", dir(t.stimuli))
            print("- attendedEar keys:", dir(t.attended_ear))
            print("- fileheader keys:", dir(t.FileHeader))

            print("- SampleRate:", get_field(t.FileHeader, "SampleRate"))
            print("- Stimuli:", get_field(t, "stimuli"))
            print("- Attended:", get_field(t, "attended_ear"))

    def LoadTrials(self):
        for i, RawTrial in enumerate(self.Rawtrials, start=1):
            print(f"\n--- Trial {i} ---")
            trialObject = Trial(i, self.cfg, self.subject_id)
            trialObject.fill_trial_metadata(RawTrial)
            self.trials.append(trialObject)

    def getSubject(self, subject_id):
         return self.mat
    def get_trial_count(self):
        return len(self.Rawtrials)

    def getTrials(self):
        return self.trials





