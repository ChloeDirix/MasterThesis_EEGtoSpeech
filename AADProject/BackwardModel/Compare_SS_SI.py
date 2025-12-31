import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from paths import paths


# ============================================================
# Helper functions to load SS and SI results
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path):
    out = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["Subject_ID"]] = {
                "full_accuracy": float(row["Full_Trial_Accuracy"]),
                "window_accuracy": float(row["Windowed_Accuracy"])
            }
    return out


# ============================================================
# Run selection utilities
# ============================================================

def list_available_runs(base_dir):
    """List all run_XXXX directories under base_dir."""
    if not os.path.exists(base_dir):
        return []

    runs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]
    return sorted(runs, key=lambda x: int(x.replace("run_", "")))


def get_latest_run(base_dir):
    runs = list_available_runs(base_dir)
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    return os.path.join(base_dir, runs[-1])


def get_run_by_number(base_dir, n):
    """Return run_XXXX folder by number (int or str)."""
    run_name = f"run_{int(n):04d}"
    full_path = os.path.join(base_dir, run_name)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Run {run_name} does not exist in {base_dir}")

    return full_path


# ============================================================
# Plot 1 — SS vs SI accuracy per subject
# ============================================================

def plot_accuracy_comparison(ss_data, si_data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    subjects = sorted(ss_data.keys(), key=lambda s: int(s.replace("S", "")))


    ss_acc = [ss_data[s]["window_accuracy"] for s in subjects]
    si_acc = [si_data[s]["window_accuracy"] for s in subjects]

    x = np.arange(len(subjects))

    plt.figure(figsize=(12, 6))
    plt.plot(x, ss_acc, label="SS", marker="o", linewidth=2)
    plt.plot(x, si_acc, label="SI", marker="s", linewidth=2)

    plt.xticks(x, subjects, rotation=45)
    plt.ylabel("Windowed Accuracy")
    plt.title("SS vs SI Accuracy per Subject")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Plot 2 — Correlation distributions
# ============================================================

def extract_corrs(result_list):
    att = []
    unatt = []
    for subj in result_list:
        for trial in subj["results"]:
            att.append(trial["corr_att"])
            unatt.append(trial["corr_unatt"])
    return np.array(att), np.array(unatt)


def plot_corr_distributions(ss_json, si_json, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ss_att, ss_unatt = extract_corrs(ss_json)
    si_att, si_unatt = extract_corrs(si_json)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(ss_att, bins=30, alpha=0.6, label="Attended")
    plt.hist(ss_unatt, bins=30, alpha=0.6, label="Unattended")
    plt.title("SS Correlations")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(si_att, bins=30, alpha=0.6, label="Attended")
    plt.hist(si_unatt, bins=30, alpha=0.6, label="Unattended")
    plt.title("SI Correlations")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Plot 3 — TRF comparison
# ============================================================

def extract_trf(subj):
    if "decoder_w" not in subj:
        return None, None, False

    w = np.array(subj["decoder_w"])
    n_channels = subj["n_channels"]
    lag_low, lag_high = subj["lags"]

    # Compute expected number of lags
    n_lags = w.shape[0] // n_channels

    # Case 1: single band → shape = (n_lags * n_channels,)
    if w.ndim == 1:
        try:
            W = w.reshape(n_lags, n_channels)
        except:
            return None, None, False

        trf = W.mean(axis=1)  # mean across channels

    # Case 2: multiband → shape = (n_lags * n_channels, n_bands)
    elif w.ndim == 2:
        try:
            W = w.reshape(n_lags, n_channels, w.shape[1])  # (lags, channels, bands)
        except:
            return None, None, False

        trf = W.mean(axis=(1, 2))  # mean across channels and bands

    else:
        return None, None, False

    # Lag axis
    lags = np.arange(lag_low, lag_high + 1)
    if len(lags) != n_lags:
        lags = np.linspace(lag_low, lag_high, n_lags)

    return trf, lags, True



def plot_trf_comparison(ss_json, si_json, save_path):
    """
    Plot group-averaged TRFs for SS and SI.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ---- Compute group-average decoder weights ----
    ss_group_w = compute_group_average_trf(ss_json)
    si_group_w = compute_group_average_trf(si_json)

    # ---- Extract metadata (lags, channels) from first subject ----
    lag_low, lag_high = ss_json[0]["lags"]
    n_channels = ss_json[0]["n_channels"]

    # ---- Convert decoders into TRFs ----
    # reuse your extract_trf, but feed fake subj
    ss_fake = {
        "decoder_w": ss_group_w,
        "n_channels": n_channels,
        "lags": [lag_low, lag_high]
    }
    si_fake = {
        "decoder_w": si_group_w,
        "n_channels": n_channels,
        "lags": [lag_low, lag_high]
    }

    ss_trf, ss_lags, ok1 = extract_trf(ss_fake)
    si_trf, si_lags, ok2 = extract_trf(si_fake)

    if not ok1 or not ok2:
        print("TRF data not available — skipping TRF plot.")
        return

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.plot(ss_lags, ss_trf, label="SS (mean TRF)", linewidth=2)
    plt.plot(si_lags, si_trf, label="SI (mean TRF)", linestyle="--", linewidth=2)

    plt.xlabel("Lag (ms)")
    plt.ylabel("Mean Decoder Weight")
    plt.title("Group-Averaged TRF: SS vs SI")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    print(f"Saved group TRF comparison: {save_path}")


import yaml

def load_run_info(run_dir):
    """Load config + summary accuracy from a run folder."""
    config_path = os.path.join(run_dir, "config_used.yaml")
    csv_path = os.path.join(run_dir, "mTRF_summary.csv")

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    window_s = cfg["backward_model"]["window_s"]

    # Load accuracy CSV
    results = load_csv(csv_path)
    mean_acc = np.mean([v["window_accuracy"] for v in results.values()])

    return window_s, mean_acc

def compare_window_lengths(base_dir, save_path):
    """
    Plot mean accuracy vs window length for a given model (SS or SI),
    adding value labels and starting y-axis at 0.5.
    """

    run_dirs = list_available_runs(base_dir)
    if not run_dirs:
        print(f"No runs found in {base_dir}")
        return

    windows = []
    accuracies = []

    for run in run_dirs:
        run_path = os.path.join(base_dir, run)
        window_s, acc = load_run_info(run_path)
        windows.append(window_s)
        accuracies.append(acc)

    # Sort
    windows = np.array(windows)
    accuracies = np.array(accuracies)
    idx = np.argsort(windows)
    windows = windows[idx]
    accuracies = accuracies[idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(windows, accuracies, marker="o", linewidth=2)

    annotate_points(windows, accuracies)  # ← add labels

    plt.xlabel("Window Length (s)")
    plt.ylabel("Mean Accuracy")
    plt.title("Accuracy vs Window Length")
    plt.grid(alpha=0.3)
    plt.ylim(0.5, 1)       # ← chance level baseline
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")

def load_run_subject_accuracies(run_dir):
    """
    Load mTRF_summary.csv and return:
    - window length (from config)
    - subjects (list)
    - accuracies (list)
    """

    # Load config for window length
    config_path = os.path.join(run_dir, "config_used.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    window_s = cfg["backward_model"]["window_s"]

    # Load subject accuracies
    csv_path = os.path.join(run_dir, "mTRF_summary.csv")
    data = load_csv(csv_path)
    subjects = sorted(data.keys(), key=lambda s: int(s.replace("S", "")))

    accuracies = [data[s]["window_accuracy"] for s in subjects]

    return window_s, subjects, accuracies

def compare_subject_accuracy_across_windows(base_dir, save_path):
    """
    Plot accuracy per subject across runs with different window lengths.
    """

    run_dirs = list_available_runs(base_dir)
    if not run_dirs:
        print(f"No runs in {base_dir}")
        return

    # Storage
    window_lengths = []
    subjects_global = None
    per_run_accuracies = {}

    # Load data for each run
    for run in run_dirs:
        run_path = os.path.join(base_dir, run)
        win, subjects, accs = load_run_subject_accuracies(run_path)

        # Save
        window_lengths.append(win)
        per_run_accuracies[win] = accs

        # keep global subject ordering
        if subjects_global is None:
            subjects_global = subjects

    # Sort by window length
    window_lengths = sorted(window_lengths)

    # Prepare plot
    x = np.arange(len(subjects_global))
    plt.figure(figsize=(14, 6))

    # Plot each window length
    for win in window_lengths:
        plt.plot(
            x,
            per_run_accuracies[win],
            marker="o",
            linewidth=2,
            label=f"{win}s"
        )

    plt.xticks(x, subjects_global, rotation=45)
    plt.ylabel("Windowed Accuracy")
    plt.title("Accuracy Per Subject Across Window Lengths")
    plt.ylim(0.5, 1)
    plt.grid(alpha=0.3)
    plt.legend(title="Window")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved subject-window comparison plot: {save_path}")

def compare_window_lengths_combined(SS_base, SI_base, save_path):
    """
    Combined SS + SI mean accuracy vs window length.
    """
    SS_runs = list_available_runs(SS_base)
    SI_runs = list_available_runs(SI_base)

    SS_w, SS_a = [], []
    SI_w, SI_a = [], []

    for r in SS_runs:
        window, acc = load_run_info(os.path.join(SS_base, r))
        SS_w.append(window)
        SS_a.append(acc)

    for r in SI_runs:
        window, acc = load_run_info(os.path.join(SI_base, r))
        SI_w.append(window)
        SI_a.append(acc)

    # Sort
    SS_w, SS_a = np.array(SS_w), np.array(SS_a)
    SI_w, SI_a = np.array(SI_w), np.array(SI_a)

    SS_idx = np.argsort(SS_w)
    SI_idx = np.argsort(SI_w)

    SS_w, SS_a = SS_w[SS_idx], SS_a[SS_idx]
    SI_w, SI_a = SI_w[SI_idx], SI_a[SI_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(SS_w, SS_a, marker="o", linewidth=2, label="SS")
    plt.plot(SI_w, SI_a, marker="s", linewidth=2, linestyle="--", label="SI")

    annotate_points(SS_w, SS_a)
    annotate_points(SI_w, SI_a)

    plt.xlabel("Window Length (s)")
    plt.ylabel("Mean Accuracy")
    plt.title("SS vs SI: Accuracy Across Window Lengths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0.5, 1)   # ← chance baseline
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved combined window-length plot: {save_path}")

def compare_accuracy_vs_subjects_all_windows(SS_base, SI_base, save_path):
    """
    Plot accuracy vs subjects, with multiple curves:
    - SS for all window lengths
    - SI for all window lengths

    X-axis: subjects
    Y-axis: accuracy
    Lines: one per (model, window_length)
    """

    SS_runs = list_available_runs(SS_base)
    SI_runs = list_available_runs(SI_base)

    if not SS_runs or not SI_runs:
        print("No SS/SI runs found — cannot plot combined accuracy vs subjects.")
        return

    # --- Load SS runs ---
    SS_data = {}   # window → accuracy list
    subjects_global = None

    for run in SS_runs:
        run_path = os.path.join(SS_base, run)
        window, subjects, accuracies = load_run_subject_accuracies(run_path)
        SS_data[window] = accuracies

        if subjects_global is None:
            subjects_global = subjects

    # --- Load SI runs ---
    SI_data = {}   # window → accuracy list

    for run in SI_runs:
        run_path = os.path.join(SI_base, run)
        window, subjects, accuracies = load_run_subject_accuracies(run_path)
        SI_data[window] = accuracies

    # --- Sort window lengths ---
    windows_SS = sorted(SS_data.keys())
    windows_SI = sorted(SI_data.keys())

    x = np.arange(len(subjects_global))
    plt.figure(figsize=(18, 7))

    # Same color for same window, but different linestyle per model
    cmap = plt.cm.get_cmap("tab10", max(len(windows_SS), len(windows_SI)))

    # Plot SS curves
    for i, w in enumerate(windows_SS):
        plt.plot(
            x, SS_data[w],
            marker="o",
            linewidth=2,
            color=cmap(i),
            label=f"SS {w}s"
        )

    # Plot SI curves
    for i, w in enumerate(windows_SI):
        plt.plot(
            x, SI_data[w],
            marker="s",
            linewidth=2,
            linestyle="--",
            color=cmap(i),
            label=f"SI {w}s"
        )

    plt.xticks(x, subjects_global, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Subjects Across Window Lengths (SS & SI)")
    plt.ylim(0.5, 1)
    plt.grid(alpha=0.3)
    plt.legend(ncol=3, title="Model / Window")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")

def annotate_points(x, y):
    """Add text labels next to points in a line plot."""
    for xi, yi in zip(x, y):
        plt.text(
            xi, yi + 0.005,          # slightly above
            f"{yi:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

def compute_group_average_trf(results):
    """
    Average decoder weights across all subjects in a run.
    Handles multi-band decoders automatically.
    """

    Ws = []
    for subj in results:
        w = np.array(subj["decoder_w"], dtype=float)
        Ws.append(w)

    # Stack into array (n_subjects, decoder_dim, bands?)
    Ws = np.stack(Ws, axis=0)

    # Mean across subjects
    W_avg = Ws.mean(axis=0)

    # Return averaged decoder
    return W_avg


# ============================================================
# MAIN SCRIPT — Now supports manual run selection + comparison runs
# ============================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # CHOOSE RUNS HERE
    # ---------------------------------------------------------
    # Options:
    #  - a number, e.g.: 3 → loads run_0003
    #  - "latest" → automatically picks the newest
    # ---------------------------------------------------------

    SS_run_choice = "5"   # or 1, 2, 3, "latest"
    SI_run_choice = "3"

    SS_base = paths.RESULTS_LIN / "SS"
    SI_base = paths.RESULTS_LIN / "SI"

    # Resolve SS run
    if SS_run_choice == "latest":
        SS_run = get_latest_run(SS_base)
    else:
        SS_run = get_run_by_number(SS_base, SS_run_choice)

    # Resolve SI run
    if SI_run_choice == "latest":
        SI_run = get_latest_run(SI_base)
    else:
        SI_run = get_run_by_number(SI_base, SI_run_choice)

    print(f"Using SS run: {SS_run}")
    print(f"Using SI run: {SI_run}")

    # ---------------------------------------------------------
    # Create comparison output folder (also run-numbered!)
    # ---------------------------------------------------------
    COMP_base = paths.RESULTS_LIN / "Comparisons"
    os.makedirs(COMP_base, exist_ok=True)

    # Automatic numbering for comparison runs
    COMPrun = paths.get_next_run_dir(COMP_base)
    print(f"Saving comparison results to: {COMPrun}")

    # ---------------------------------------------------------
    # Load results
    # ---------------------------------------------------------
    SS_JSON = os.path.join(SS_run, "mTRF_results.json")
    SI_JSON = os.path.join(SI_run, "mTRF_results.json")

    SS_CSV = os.path.join(SS_run, "mTRF_summary.csv")
    SI_CSV = os.path.join(SI_run, "mTRF_summary.csv")

    ss_csv = load_csv(SS_CSV)
    si_csv = load_csv(SI_CSV)

    ss_json = load_json(SS_JSON)
    si_json = load_json(SI_JSON)

    # ---------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------
    plot_accuracy_comparison(ss_csv, si_csv,
                             save_path=os.path.join(COMPrun, "accuracy.png"))

    plot_corr_distributions(ss_json, si_json,
                            save_path=os.path.join(COMPrun, "correlations.png"))

    plot_trf_comparison(ss_json, si_json,
                        save_path=os.path.join(COMPrun, "trf.png"))

    # ===============================
    # Compare window lengths (SS)
    # ===============================
    compare_window_lengths(
        base_dir=paths.RESULTS_LIN / "SS",
        save_path=os.path.join(COMPrun, "SS_window_length_comparison.png")
    )

    # ===============================
    # Compare window lengths (SI)
    # ===============================
    compare_window_lengths(
        base_dir=paths.RESULTS_LIN / "SI",
        save_path=os.path.join(COMPrun, "SI_window_length_comparison.png")
    )


    # Compare subject accuracy across window lengths for SS
    compare_subject_accuracy_across_windows(
        base_dir=paths.RESULTS_LIN / "SS",
        save_path=os.path.join(COMPrun, "SS_subject_vs_window.png")
    )

    # Compare subject accuracy across window lengths for SI
    compare_subject_accuracy_across_windows(
        base_dir=paths.RESULTS_LIN / "SI",
        save_path=os.path.join(COMPrun, "SI_subject_vs_window.png"))

    print("\nAll comparison plots saved.\n")

    # combined for SS/SI and for window lengths
    compare_window_lengths_combined(
        SS_base=paths.RESULTS_LIN / "SS",
        SI_base=paths.RESULTS_LIN / "SI",
        save_path=os.path.join(COMPrun, "SS_SI_window_length_comparison.png")
    )

    compare_accuracy_vs_subjects_all_windows(
        SS_base=paths.RESULTS_LIN / "SS",
        SI_base=paths.RESULTS_LIN / "SI",
        save_path=os.path.join(COMPrun, "accuracy_vs_subjects_all_windows.png")
    )

