import yaml
import Preprocessing as PP
import EnvelopeExt as EE
from Code.LoadSubject import LoadSubject

# Load configuration file
cfg = yaml.safe_load(open("config.yaml", "r"))

# Process each subject
for subject_id in cfg["subjects"]:
    print(f"\n=== Loading Subject {subject_id} ===")
    Subject_data = LoadSubject(subject_id, cfg)

    for trial in Subject_data.getTrials():
        if not trial.validate():
            print(f" Skipping trial {trial.index} — missing important fields!")
            continue

        eeg_PP=PP.preprocess_trial(trial,cfg)
        trial.setEEG_PP(eeg_PP)
        EE.extract_envelopes(Subject_data,trial,cfg)




