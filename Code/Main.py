import numpy as np
import yaml
import EEGPreprocessing
import AudioPreprocessing
from Code import summaryStats, trialcheck
from Code.LoadSubject import LoadSubject
from Code.RunDecoder import run_mTRF


# Load configuration file
cfg = yaml.safe_load(open("config.yaml", "r"))
preprocessing=cfg["Do_preprocessing"]
subjects_List=[]

for subject_id in cfg["subjects"]:
    print(f"\n=== Loading Subject {subject_id} ===")
    Subject_object = LoadSubject(subject_id, cfg)

    for trial in Subject_object.getTrials():
        if not trial.validate():
            print(f" Skipping trial {trial.index} — missing important fields!")
            continue
        if preprocessing == True:
            #eegPreprocessing
            eeg_PP=EEGPreprocessing.preprocess_trial(trial.eeg_data,trial.fs_eeg, cfg)
            print(f"{trial.index}: eeg preprocessing done")

            eeg, env_att, env_unatt=AudioPreprocessing.PreprocessAudioFiles(trial, eeg_PP, cfg, True)
            #diag = trialcheck.trial_diagnostics(eeg, env_att, fs=cfg['target_fs'])
            #print(diag)
            trial.setEEG_PP(eeg)
            trial.set_envelopes(env_att, env_unatt)

    subjects_List.append((subject_id, Subject_object))

results=run_mTRF(subjects_List, cfg)

if len(results) == 0:
    print(f"No usable results, skipping summary.\n")

# Extract arrays
r_att = np.array([r["corr_att"] for r in results])
r_unatt = np.array([r["corr_unatt"] for r in results])

acc = np.mean([r["correct"] for r in results])
print("Decoding Accuracy = {acc:.2f}")

stats = summaryStats.SummaryStats(r_att, r_unatt)
summaryStats.plot_histograms(r_att, r_unatt)

