import csv
import json
import os

import numpy as np
import yaml
from Implementation.Code import EEGPreprocessing, summaryStats
from Implementation.NotInUse.LoadSubject import LoadSubject
from Implementation.Code.RunBackwardModel import run_mTRF


# Load configuration file
cfg = yaml.safe_load(open("../config.yaml", "r"))
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
            eeg_PP = EEGPreprocessing.preprocess_trial(trial, trial.eeg_data, trial.fs_eeg, cfg)

    subjects_List.append((subject_id, Subject_object))


results=run_mTRF(subjects_List,trial.env_att, trial.env_unatt, cfg)

if len(results) == 0:
    print(f"No usable results, skipping summary.\n")

#
# # Extract arrays
# r_att = np.array([r["corr_att"] for r in results])
# r_unatt = np.array([r["corr_unatt"] for r in results])
#
# acc = np.mean([r["correct"] for r in results])
# print("Decoding Accuracy = {acc:.2f}")
#
# stats = summaryStats.SummaryStats(r_att, r_unatt)
# summaryStats.plot_histograms(r_att, r_unatt)
#


# Summarize results
print("\n=== Summary Across Subjects ===")
all_full_acc = []
all_window_acc = []

for subj in results:
    sid = subj["subject_id"]
    full_acc = subj["full_accuracy"]
    win_acc = subj["window_accuracy"]

    print(f"Subject {sid}: Full-trial acc = {full_acc:.2f}, Windowed acc = {win_acc:.2f}")
    all_full_acc.append(full_acc)
    all_window_acc.append(win_acc)

# Overall summary
mean_full_acc = np.mean(all_full_acc)
mean_window_acc = np.mean(all_window_acc)

print("\n=== Group Results ===")
print(f"Average Full-trial Accuracy: {mean_full_acc:.2f}")
print(f"Average Windowed Accuracy:  {mean_window_acc:.2f}")

# Save results
output_dir = cfg.get("Results_dir", "Results")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "mTRF_results.json")

with open(output_file, "w") as f:
    json.dump(results, f, indent=4, default=lambda x: x.item() if isinstance(x, np.generic) else x)

# save as CSV ===
csv_file = os.path.join(output_dir, "mTRF_summary.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Subject_ID", "Full_Trial_Accuracy", "Windowed_Accuracy"])

    for subj in results:
        sid = subj["subject_id"]
        full_acc = float(subj["full_accuracy"])
        win_acc = float(subj["window_accuracy"])
        writer.writerow([sid, full_acc, win_acc])

print(f"Per-subject summary saved to {csv_file}")

print(f"\nResults saved to {output_file}")

# Summary plots
r_att_all = []
r_unatt_all = []

for subj in results:
    for trial in subj["results"]:
        r_att_all.append(trial["corr_att"])
        r_unatt_all.append(trial["corr_unatt"])

r_att_all = np.array(r_att_all)
r_unatt_all = np.array(r_unatt_all)

stats = summaryStats.SummaryStats(r_att_all, r_unatt_all)
summaryStats.plot_histograms(r_att_all, r_unatt_all)

