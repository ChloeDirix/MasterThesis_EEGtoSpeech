import os
import json
import csv
import numpy as np
import yaml
from pynwb import NWBHDF5IO

# Import your new modular components
from Loaders.matlab_loader import MatlabSubjectLoader
from NWB.NWB_Manager import NWBManager
from Code import EEGPreprocessing, RunBackwardModel, summaryStats, DataPrep


def main():
    # === 1. Load configuration ===
    cfg = yaml.safe_load(open("config.yaml", "r"))
    preprocessing = cfg.get("Do_preprocessing", False)
    BackwardModel = cfg.get("Do_BackwardModel", False)

    #Define path to subject files
    subjects = cfg["subjects"]
    RawEEG_dir=cfg["RawEEG_dir"]

    #Define nwb data dir
    Data_output_dir = cfg["Data_output_dir"]
    os.makedirs(Data_output_dir, exist_ok=True)
    nwb_mgr = NWBManager()
    all_results = []

    #define results dir
    results_dir=cfg["Results_dir"]

    # === 2. Loop over subjects ===
    for subject_id in subjects:
        nwb_out = os.path.join(Data_output_dir, f"{subject_id}.nwb")

        if preprocessing:
            print(f"\n=== Processing subject {subject_id} ===")
            subject_file = os.path.join(RawEEG_dir, f"{subject_id}.mat")

            # Load MATLAB → Subject/Trial objects
            loader = MatlabSubjectLoader(subject_file, subject_id)
            subject = loader.load()

            # Preprocessing

            print(f"Preprocessing EEG for {subject_id}...")
            for trial in subject.trials:
                EEGPreprocessing.preprocess_trial(trial,cfg)


            # Save to NWB for reproducibility
            nwb_mgr.save_subject(subject, nwb_out)


            print(f"Saved subject {subject_id} to {nwb_out}")

            io = NWBHDF5IO(nwb_out, "r")
            nwbfile = io.read()
            print(nwbfile)
            print(nwbfile.acquisition)
            print(nwbfile.processing)
            print(list(nwbfile.trials.columns))
            print(nwbfile.trials[:5])

        if BackwardModel:
            # 2d. Run backward model (mTRF)
            subj_results = RunBackwardModel.run_mTRF(nwb_out, cfg)
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
    json_path = os.path.join(output_dir, "mTRF_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")

    # CSV summary
    csv_path = os.path.join(output_dir, "mTRF_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject_ID", "Full_Trial_Accuracy", "Windowed_Accuracy"])
        for subj in results:
            writer.writerow([subj["subject_id"], subj["full_accuracy"], subj["window_accuracy"]])
    print(f"Per-subject summary saved to {csv_path}")

    # Optional: plotting histograms
    all_att = np.concatenate([np.array([t["corr_att"] for t in subj["results"]]) for subj in results])
    all_unatt = np.concatenate([np.array([t["corr_unatt"] for t in subj["results"]]) for subj in results])
    stats = summaryStats.SummaryStats(all_att, all_unatt)
    summaryStats.plot_histograms(all_att, all_unatt)


if __name__ == "__main__":
    main()
