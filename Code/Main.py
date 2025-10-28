import numpy as np
import yaml
import Preprocessing as PP
import EnvelopeExt as EE
from Code import summaryStats
from Code.LoadSubject import LoadSubject
from Code.RunDecoder import run_mTRF

# Load configuration file
cfg = yaml.safe_load(open("config.yaml", "r"))

preprocessing=False


for subject_id in cfg["subjects"]:
    print(f"\n=== Loading Subject {subject_id} ===")
    Subject_data = LoadSubject(subject_id, cfg)

    for trial in Subject_data.getTrials():
        if not trial.validate():
            print(f" Skipping trial {trial.index} — missing important fields!")
            continue
        if preprocessing == True:
            eeg_PP=PP.preprocess_trial(trial,cfg)
            trial.setEEG_PP(eeg_PP)
            EE.extract_envelopes(Subject_data,trial,cfg)

    results=run_mTRF(subject_id, Subject_data, cfg)

    if len(results) == 0:
        print(f"No usable results for {subject_id}, skipping summary.\n")
        continue

        # Extract arrays
    r_att = np.array([r["corr_att"] for r in results])
    r_unatt = np.array([r["corr_unatt"] for r in results])

    acc = np.mean([r["correct"] for r in results])
    print(f"Subject {subject_id} — Decoding Accuracy = {acc:.2f}")

    stats = summaryStats.SummaryStats(r_att, r_unatt)
    summaryStats.plot_histograms(r_att, r_unatt)

