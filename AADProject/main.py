import os
import json
import csv
import numpy as np
import yaml
from pynwb import NWBHDF5IO

from Loaders.matlab_loader import MatlabSubjectLoader
from NWB.NWB_Manager import NWBManager
from Code import RunBackwardModel, summaryStats
from Preprocessing import stimulusPreprocessing, EEGPreprocessing


def main():
    # === 1. Load configuration ===
    cfg = yaml.safe_load(open("config.yaml", "r"))

    # What steps?
    envelope_extraction=cfg["Do_envelope_extraction"]
    EEG_preprocessing = cfg["Do_preprocessing"]
    BackwardModel = cfg["Do_BackwardModel"]

    # find paths to subject files
    subjects = cfg["subjects"]
    RawEEG_dir=cfg["RawEEG_dir"]

    # Define dir for preprocessed data
    EEG_PP_dir = cfg["EEG_PP_dir"]
    os.makedirs(EEG_PP_dir, exist_ok=True)
    nwb_mgr = NWBManager()
    all_results = []

    # Define results dir
    results_dir=cfg["Results_dir"]

    if envelope_extraction:
        stimulusPreprocessing.PreprocessAudioFiles(cfg)

    # === 2. Loop over subjects_id (in cfg) ===
    for subject_id in subjects:

        subject_file_in = os.path.join(RawEEG_dir, f"{subject_id}.mat")  # .mat data
        subject_file_out = os.path.join(EEG_PP_dir, f"{subject_id}.nwb")  # .nwb data

       # Preprocessing
        if EEG_preprocessing:
            loader = MatlabSubjectLoader(subject_file_in, subject_id)  # subject/trial objects -- intermediate storage
            subject = loader.load()
            print(f"\n=== Processing subject {subject_id} ===")

            for trial in subject.trials:
                EEGPreprocessing.preprocess_trial(trial,cfg)

            # Save to NWB for reproducibility
            nwb_mgr.save_subject(subject, subject_file_out)

            print(f"Saved subject {subject_id} to {subject_file_out}")


            io = NWBHDF5IO(subject_file_out, "r")
            nwbfile = io.read()
            print(nwbfile)
            print(nwbfile.acquisition)
            print(nwbfile.processing)
            print(list(nwbfile.trials.columns))
            print(nwbfile.trials[:5])
            elec_df = nwbfile.electrodes.to_dataframe()
            channel_names = list(elec_df['label'])
            print("cahnnelnames-- ",channel_names)

        if BackwardModel:
            # 2d. Run backward model (mTRF)
            subj_results = RunBackwardModel.run_mTRF(subject_file_out, cfg)
            all_results.append(subj_results)

    # === 3. Summarize results ===
    if BackwardModel:
        summarize_results(all_results, results_dir)


def summarize_results(results, output_dir):
    """Aggregate and save all subject-level results."""
    if not results:
        print("No usable results, skipping summary.")
        return

    # Aggregate accuracies
    full_accs = [r["full_accuracy"] for r in results]
    win_accs = [r["window_accuracy"] for r in results]

    print("\n=== Group Results ===")
    print(f"Mean full-trial accuracy: {np.mean(full_accs):.2f}")
    print(f"Mean windowed accuracy:  {np.mean(win_accs):.2f}")

    # JSON export
    json_path = os.path.join(output_dir, "mTRF_results_mean.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")

    # CSV summary
    csv_path = os.path.join(output_dir, "mTRF_summary_mean.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject_ID", "Full_Trial_Accuracy", "Windowed_Accuracy"])
        for subj in results:
            writer.writerow([subj["subject_id"], subj["full_accuracy"], subj["window_accuracy"]])
    print(f"Per-subject summary saved to {csv_path}")

    # plotting histograms
    all_att = np.concatenate([np.array([t["corr_att"] for t in subj["results"]]) for subj in results])
    all_unatt = np.concatenate([np.array([t["corr_unatt"] for t in subj["results"]]) for subj in results])
    stats=summaryStats.SummaryStats(all_att, all_unatt)

    with open(os.path.join(output_dir, "statsmean.json"), "w") as f:
        json.dump(stats, f, indent=4)

    summaryStats.plot_histograms(all_att, all_unatt, os.path.join(output_dir, "Histogram_mean"))


if __name__ == "__main__":
    main()
