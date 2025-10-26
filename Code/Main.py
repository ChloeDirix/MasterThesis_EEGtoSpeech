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

    print(results)
    r_att=results["corr_att"]
    r_unatt=results["uncorr_att"]
    stats = summaryStats.SummaryStats(r_att, r_unatt)
    summaryStats.plot_histograms(r_att, r_unatt)


