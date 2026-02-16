#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

from paths import paths


# ============================================================
# Helpers: file resolving (supports old + new run layouts)
# ============================================================

def _first_existing(*candidates):
    for p in candidates:
        if p is None:
            continue
        if os.path.exists(p):
            return p
    return None


def resolve_results_json(run_dir):
    """
    Return the best matching results JSON for a run.
    Supports both old and new naming / subfolders.
    """
    return _first_existing(
        os.path.join(run_dir, "mTRF_results_ALL.json"),
        os.path.join(run_dir, "mTRF_results.json"),
        os.path.join(run_dir, "ALL", "mTRF_results_ALL.json"),
        os.path.join(run_dir, "ALL", "mTRF_results.json"),
    )


def resolve_summary_csv(run_dir):
    """
    Return the best matching summary CSV for a run.
    Supports both old and new naming / subfolders.
    """
    return _first_existing(
        os.path.join(run_dir, "mTRF_summary_ALL.csv"),
        os.path.join(run_dir, "mTRF_summary.csv"),
        os.path.join(run_dir, "ALL", "mTRF_summary_ALL.csv"),
        os.path.join(run_dir, "ALL", "mTRF_summary.csv"),
    )


def resolve_config_yaml(run_dir):
    """
    Return config_used.yaml (or config copy) if present.
    """
    return _first_existing(
        os.path.join(run_dir, "config_used.yaml"),
        os.path.join(run_dir, "config.yaml"),
        os.path.join(run_dir, "config_copy.yaml"),
        os.path.join(run_dir, "ALL", "config_used.yaml"),
        os.path.join(run_dir, "ALL", "config.yaml"),
        os.path.join(run_dir, "ALL", "config_copy.yaml"),
    )


# ============================================================
# Loading JSON/CSV
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    """
    Load summary CSV into dict keyed by Subject_ID.
    Skips MEAN rows automatically.
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("Subject_ID", "").strip()
            if not sid:
                continue
            if sid.upper().startswith("MEAN"):
                continue  # MEAN_ALL, MEAN_DTU, etc.

            out[sid] = {
                "full_accuracy": float(row["Full_Trial_Accuracy"]),
                "window_accuracy": float(row["Windowed_Accuracy"]),
            }
    return out


def load_run_csv_json(run_dir):
    csv_path = resolve_summary_csv(run_dir)
    json_path = resolve_results_json(run_dir)

    if csv_path is None:
        raise FileNotFoundError(f"No summary CSV found in {run_dir}")
    if json_path is None:
        raise FileNotFoundError(f"No results JSON found in {run_dir}")

    return load_csv(csv_path), load_json(json_path)


# ============================================================
# Run listing + selection
# ============================================================

def list_available_runs(base_dir):
    base_dir = str(base_dir)
    if not os.path.exists(base_dir):
        return []
    runs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]

    def _run_num(d):
        # run_58955009 -> 58955009
        # run_0005 -> 5
        try:
            return int(d.replace("run_", ""))
        except Exception:
            return 10**18

    return sorted(runs, key=_run_num)


def print_run_list(base_dir, label):
    runs = list_available_runs(base_dir)
    print(f"\n[{label}] base: {base_dir}")
    if not runs:
        print("  (no runs found)")
        return
    for i, r in enumerate(runs):
        print(f"  idx:{i:<3d}  {r}")


def get_latest_run(base_dir):
    runs = list_available_runs(base_dir)
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    return os.path.join(base_dir, runs[-1])


def _try_run_folder(base_dir, run_name):
    p = os.path.join(base_dir, run_name)
    return p if os.path.exists(p) else None


def pick_run(base_dir, selector):
    """
    selector can be:
      - "latest"
      - "idx:N"
      - "run_58955009"
      - "58955009"  (numeric; tries run_58955009 and run_0005)
      - an existing path (absolute or relative)
    """
    selector = str(selector).strip()

    # direct path
    if os.path.exists(selector) and os.path.isdir(selector):
        return selector

    if selector.lower() == "latest":
        return get_latest_run(base_dir)

    if selector.lower().startswith("idx:"):
        runs = list_available_runs(base_dir)
        if not runs:
            raise FileNotFoundError(f"No runs in {base_dir}")
        idx = int(selector.split(":", 1)[1])
        if idx < 0 or idx >= len(runs):
            raise IndexError(f"{selector} out of range (0..{len(runs)-1}) for {base_dir}")
        return os.path.join(base_dir, runs[idx])

    if selector.startswith("run_"):
        p = _try_run_folder(base_dir, selector)
        if p:
            return p
        raise FileNotFoundError(f"Run {selector} does not exist in {base_dir}")

    # numeric id
    if selector.isdigit():
        n = int(selector)

        # try run_{n} (VSC style)
        p = _try_run_folder(base_dir, f"run_{n}")
        if p:
            return p

        # try run_{n:04d} (old style)
        p = _try_run_folder(base_dir, f"run_{n:04d}")
        if p:
            return p

        raise FileNotFoundError(
            f"Run id {n} not found in {base_dir} (tried run_{n} and run_{n:04d})"
        )

    raise ValueError(f"Unrecognized run selector: {selector}")


def parse_run_list(base_dir, selectors):
    """
    selectors: list like ["idx:0","58955009","latest"]
    returns list of run_dir paths (deduplicated, stable order)
    """
    out = []
    seen = set()
    for s in selectors:
        r = pick_run(base_dir, s)
        if r not in seen:
            out.append(r)
            seen.add(r)
    if not out:
        raise ValueError("No runs selected.")
    return out


# ============================================================
# Subject sorting utils
# ============================================================

def _subj_num(s):
    # "S8" -> 8, "S8_DTU" -> 8
    s = s.split("_")[0]
    return int(s.replace("S", ""))


# ============================================================
# Plot 1 — SS vs SI accuracy per subject (pairwise)
# ============================================================

def plot_accuracy_comparison(ss_data, si_data, save_path, title_suffix=""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # keep only common subjects to avoid mismatched keys
    common = sorted(set(ss_data.keys()) & set(si_data.keys()), key=_subj_num)
    if not common:
        raise ValueError("No common subjects between SS and SI summaries.")

    ss_acc = [ss_data[s]["window_accuracy"] for s in common]
    si_acc = [si_data[s]["window_accuracy"] for s in common]

    x = np.arange(len(common))

    plt.figure(figsize=(12, 6))
    plt.plot(x, ss_acc, label="SS", marker="o", linewidth=2)
    plt.plot(x, si_acc, label="SI", marker="s", linewidth=2)

    plt.xticks(x, common, rotation=45)
    plt.ylabel("Windowed Accuracy")
    plt.title(f"SS vs SI Accuracy per Subject{title_suffix}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Plot 2 — Correlation distributions (pairwise)
# ============================================================

def extract_corrs(result_list):
    att = []
    unatt = []
    for subj in result_list:
        for trial in subj.get("results", []):
            att.append(trial.get("corr_att", np.nan))
            unatt.append(trial.get("corr_unatt", np.nan))
    att = np.array(att, dtype=float)
    unatt = np.array(unatt, dtype=float)
    att = att[~np.isnan(att)]
    unatt = unatt[~np.isnan(unatt)]
    return att, unatt


def plot_corr_distributions(ss_json, si_json, save_path, title_suffix=""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ss_att, ss_unatt = extract_corrs(ss_json)
    si_att, si_unatt = extract_corrs(si_json)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(ss_att, bins=30, alpha=0.6, label="Attended")
    plt.hist(ss_unatt, bins=30, alpha=0.6, label="Unattended")
    plt.title(f"SS Correlations{title_suffix}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(si_att, bins=30, alpha=0.6, label="Attended")
    plt.hist(si_unatt, bins=30, alpha=0.6, label="Unattended")
    plt.title(f"SI Correlations{title_suffix}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Plot 3 — TRF comparison (pairwise, group-averaged)
# ============================================================

def compute_group_average_trf(results):
    Ws = []
    for subj in results:
        if "decoder_w" not in subj:
            continue
        w = np.array(subj["decoder_w"], dtype=float)
        Ws.append(w)
    if not Ws:
        return None
    Ws = np.stack(Ws, axis=0)
    return Ws.mean(axis=0)


def extract_trf(subj):
    if "decoder_w" not in subj:
        return None, None, False

    w = np.array(subj["decoder_w"])
    n_channels = subj["n_channels"]
    lag_low, lag_high = subj["lags"]

    n_lags = w.shape[0] // n_channels

    if w.ndim == 1:
        try:
            W = w.reshape(n_lags, n_channels)
        except Exception:
            return None, None, False
        trf = W.mean(axis=1)

    elif w.ndim == 2:
        try:
            W = w.reshape(n_lags, n_channels, w.shape[1])
        except Exception:
            return None, None, False
        trf = W.mean(axis=(1, 2))
    else:
        return None, None, False

    lags = np.arange(lag_low, lag_high + 1)
    if len(lags) != n_lags:
        lags = np.linspace(lag_low, lag_high, n_lags)

    return trf, lags, True


def plot_trf_comparison(ss_json, si_json, save_path, title_suffix=""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ss_group_w = compute_group_average_trf(ss_json)
    si_group_w = compute_group_average_trf(si_json)
    if ss_group_w is None or si_group_w is None:
        print("TRF data not available — skipping TRF plot.")
        return

    lag_low, lag_high = ss_json[0]["lags"]
    n_channels = ss_json[0]["n_channels"]

    ss_fake = {"decoder_w": ss_group_w, "n_channels": n_channels, "lags": [lag_low, lag_high]}
    si_fake = {"decoder_w": si_group_w, "n_channels": n_channels, "lags": [lag_low, lag_high]}

    ss_trf, ss_lags, ok1 = extract_trf(ss_fake)
    si_trf, si_lags, ok2 = extract_trf(si_fake)
    if not ok1 or not ok2:
        print("TRF extraction failed — skipping TRF plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(ss_lags, ss_trf, label="SS (mean TRF)", linewidth=2)
    plt.plot(si_lags, si_trf, label="SI (mean TRF)", linestyle="--", linewidth=2)

    plt.xlabel("Lag (ms)")
    plt.ylabel("Mean Decoder Weight")
    plt.title(f"Group-Averaged TRF: SS vs SI{title_suffix}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Multi-run overlay plot (within one model)
# ============================================================

def annotate_points(x, y):
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 0.005, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)


def load_run_info(run_dir):
    config_path = resolve_config_yaml(run_dir)
    csv_path = resolve_summary_csv(run_dir)

    if config_path is None:
        raise FileNotFoundError(f"No config yaml found in {run_dir}")
    if csv_path is None:
        raise FileNotFoundError(f"No summary CSV found in {run_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    window_s = cfg["backward_model"]["window_s"]
    results = load_csv(csv_path)
    mean_acc = float(np.mean([v["window_accuracy"] for v in results.values()]))

    return window_s, mean_acc


def plot_accuracy_multi(runs_data, title, save_path):
    """
    runs_data: list of tuples (label, csv_dict)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    common = None
    for _, d in runs_data:
        keys = set(d.keys())
        common = keys if common is None else (common & keys)

    if not common:
        raise ValueError("No common subjects across selected runs.")

    common = sorted(common, key=_subj_num)
    x = np.arange(len(common))

    plt.figure(figsize=(14, 6))
    for label, d in runs_data:
        acc = [d[s]["window_accuracy"] for s in common]
        plt.plot(x, acc, marker="o", linewidth=2, label=label)

    plt.xticks(x, common, rotation=45)
    plt.ylabel("Windowed Accuracy")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Window-length plots across ALL runs in a base folder
# ============================================================

def compare_window_lengths(base_dir, save_path):
    run_dirs = list_available_runs(base_dir)
    if not run_dirs:
        print(f"No runs found in {base_dir}")
        return

    windows = []
    accuracies = []

    for run in run_dirs:
        run_path = os.path.join(base_dir, run)
        try:
            window_s, acc = load_run_info(run_path)
        except Exception as e:
            print(f"Skipping {run_path}: {e}")
            continue
        windows.append(window_s)
        accuracies.append(acc)

    if not windows:
        print(f"No usable runs for window-length plot in {base_dir}")
        return

    windows = np.array(windows)
    accuracies = np.array(accuracies)
    idx = np.argsort(windows)
    windows = windows[idx]
    accuracies = accuracies[idx]

    plt.figure(figsize=(10, 6))
    plt.plot(windows, accuracies, marker="o", linewidth=2)
    annotate_points(windows, accuracies)

    plt.xlabel("Window Length (s)")
    plt.ylabel("Mean Accuracy")
    plt.title("Accuracy vs Window Length")
    plt.grid(alpha=0.3)
    plt.ylim(0.5, 1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


def load_run_subject_accuracies(run_dir):
    config_path = resolve_config_yaml(run_dir)
    csv_path = resolve_summary_csv(run_dir)

    if config_path is None or csv_path is None:
        raise FileNotFoundError(f"Missing config/csv in {run_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    window_s = cfg["backward_model"]["window_s"]

    data = load_csv(csv_path)
    subjects = sorted(data.keys(), key=_subj_num)
    accuracies = [data[s]["window_accuracy"] for s in subjects]

    return window_s, subjects, accuracies


def compare_subject_accuracy_across_windows(base_dir, save_path):
    run_dirs = list_available_runs(base_dir)
    if not run_dirs:
        print(f"No runs in {base_dir}")
        return

    window_lengths = []
    subjects_global = None
    per_run_accuracies = {}

    for run in run_dirs:
        run_path = os.path.join(base_dir, run)
        try:
            win, subjects, accs = load_run_subject_accuracies(run_path)
        except Exception as e:
            print(f"Skipping {run_path}: {e}")
            continue

        window_lengths.append(win)
        per_run_accuracies[win] = accs
        if subjects_global is None:
            subjects_global = subjects

    if subjects_global is None or not window_lengths:
        print(f"No usable runs for subject-vs-window plot in {base_dir}")
        return

    window_lengths = sorted(window_lengths)

    x = np.arange(len(subjects_global))
    plt.figure(figsize=(14, 6))

    for win in window_lengths:
        plt.plot(x, per_run_accuracies[win], marker="o", linewidth=2, label=f"{win}s")

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

    print(f"Saved: {save_path}")


def compare_window_lengths_combined(SS_base, SI_base, save_path):
    SS_runs = list_available_runs(SS_base)
    SI_runs = list_available_runs(SI_base)

    SS_w, SS_a = [], []
    SI_w, SI_a = [], []

    for r in SS_runs:
        try:
            window, acc = load_run_info(os.path.join(SS_base, r))
        except Exception:
            continue
        SS_w.append(window)
        SS_a.append(acc)

    for r in SI_runs:
        try:
            window, acc = load_run_info(os.path.join(SI_base, r))
        except Exception:
            continue
        SI_w.append(window)
        SI_a.append(acc)

    if not SS_w or not SI_w:
        print("Not enough SS/SI runs for combined window-length plot.")
        return

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
    plt.ylim(0.5, 1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


def compare_accuracy_vs_subjects_all_windows(SS_base, SI_base, save_path):
    SS_runs = list_available_runs(SS_base)
    SI_runs = list_available_runs(SI_base)

    if not SS_runs or not SI_runs:
        print("No SS/SI runs found — cannot plot combined accuracy vs subjects.")
        return

    SS_data = {}
    subjects_global = None
    for run in SS_runs:
        run_path = os.path.join(SS_base, run)
        try:
            window, subjects, accuracies = load_run_subject_accuracies(run_path)
        except Exception:
            continue
        SS_data[window] = accuracies
        if subjects_global is None:
            subjects_global = subjects

    SI_data = {}
    for run in SI_runs:
        run_path = os.path.join(SI_base, run)
        try:
            window, subjects, accuracies = load_run_subject_accuracies(run_path)
        except Exception:
            continue
        SI_data[window] = accuracies

    if subjects_global is None or not SS_data or not SI_data:
        print("Not enough usable SS/SI runs for accuracy_vs_subjects_all_windows.")
        return

    windows_SS = sorted(SS_data.keys())
    windows_SI = sorted(SI_data.keys())

    x = np.arange(len(subjects_global))
    plt.figure(figsize=(18, 7))
    cmap = plt.cm.get_cmap("tab10", max(len(windows_SS), len(windows_SI)))

    for i, w in enumerate(windows_SS):
        plt.plot(x, SS_data[w], marker="o", linewidth=2, color=cmap(i), label=f"SS {w}s")

    for i, w in enumerate(windows_SI):
        plt.plot(x, SI_data[w], marker="s", linewidth=2, linestyle="--", color=cmap(i), label=f"SI {w}s")

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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--list", action="store_true",
                        help="List available SS/SI runs with idx selectors and exit.")

    # NOTE: nargs="+" means at least one selector after the flag.
    # You can pass one or many selectors.
    parser.add_argument("--ss", nargs="+", default=["latest"],
                        help='SS run selectors: latest | idx:N | run_XXXX | XXXX (numeric id) | /path/to/run. '
                             'You can pass multiple, e.g. --ss idx:0 idx:1')

    parser.add_argument("--si", nargs="+", default=["latest"],
                        help='SI run selectors: latest | idx:N | run_XXXX | XXXX (numeric id) | /path/to/run. '
                             'You can pass multiple, e.g. --si idx:0 idx:1')

    args = parser.parse_args()

    SS_base = paths.RESULTS_LIN / "SS"
    SI_base = paths.RESULTS_LIN / "SI"

    if args.list:
        print_run_list(SS_base, "SS")
        print_run_list(SI_base, "SI")
        return

    # Resolve selected runs (can be multiple)
    ss_runs = parse_run_list(SS_base, args.ss)
    si_runs = parse_run_list(SI_base, args.si)

    print("\nSelected SS runs:")
    for r in ss_runs:
        print(" ", r)

    print("\nSelected SI runs:")
    for r in si_runs:
        print(" ", r)

    # Create comparison output folder (run-numbered)
    COMP_base = paths.RESULTS_LIN / "Comparisons"
    os.makedirs(COMP_base, exist_ok=True)
    COMPrun = paths.get_next_run_dir(COMP_base)
    os.makedirs(COMPrun, exist_ok=True)
    print(f"\nSaving comparison plots to: {COMPrun}\n")

    # ---- Multi-run overlay plots (within model) ----
    if len(ss_runs) > 1:
        ss_runs_data = []
        for r in ss_runs:
            try:
                win_s, _ = load_run_info(r)
                label = f"{Path(r).name} ({win_s}s)"
            except Exception:
                label = Path(r).name
            ss_csv, _ = load_run_csv_json(r)
            ss_runs_data.append((label, ss_csv))

        plot_accuracy_multi(
            ss_runs_data,
            title="SS Windowed Accuracy (selected runs)",
            save_path=os.path.join(COMPrun, "SS_accuracy_multi_selected.png")
        )

    if len(si_runs) > 1:
        si_runs_data = []
        for r in si_runs:
            try:
                win_s, _ = load_run_info(r)
                label = f"{Path(r).name} ({win_s}s)"
            except Exception:
                label = Path(r).name
            si_csv, _ = load_run_csv_json(r)
            si_runs_data.append((label, si_csv))

        plot_accuracy_multi(
            si_runs_data,
            title="SI Windowed Accuracy (selected runs)",
            save_path=os.path.join(COMPrun, "SI_accuracy_multi_selected.png")
        )

    # ---- Decide SS-vs-SI pairing strategy ----
    pairs = []

    if len(ss_runs) == len(si_runs):
        # pair by index
        pairs = list(zip(ss_runs, si_runs))
    elif len(ss_runs) == 1 and len(si_runs) > 1:
        pairs = [(ss_runs[0], s) for s in si_runs]
    elif len(si_runs) == 1 and len(ss_runs) > 1:
        pairs = [(s, si_runs[0]) for s in ss_runs]
    else:
        # fallback: just compare first-first
        pairs = [(ss_runs[0], si_runs[0])]

    # ---- Pairwise comparison plots ----
    for k, (ss_run, si_run) in enumerate(pairs, start=1):
        ss_name = Path(ss_run).name
        si_name = Path(si_run).name
        suffix = f" ({ss_name} vs {si_name})"

        ss_csv, ss_json = load_run_csv_json(ss_run)
        si_csv, si_json = load_run_csv_json(si_run)

        # files per pair
        tag = f"pair_{k:02d}_{ss_name}_VS_{si_name}"
        tag = tag.replace("/", "_")

        plot_accuracy_comparison(
            ss_csv, si_csv,
            save_path=os.path.join(COMPrun, f"accuracy_{tag}.png"),
            title_suffix=suffix
        )

        plot_corr_distributions(
            ss_json, si_json,
            save_path=os.path.join(COMPrun, f"correlations_{tag}.png"),
            title_suffix=suffix
        )

        plot_trf_comparison(
            ss_json, si_json,
            save_path=os.path.join(COMPrun, f"trf_{tag}.png"),
            title_suffix=suffix
        )

    # ---- Global window-length plots (scan all runs) ----
    compare_window_lengths(
        base_dir=SS_base,
        save_path=os.path.join(COMPrun, "SS_window_length_comparison.png")
    )
    compare_window_lengths(
        base_dir=SI_base,
        save_path=os.path.join(COMPrun, "SI_window_length_comparison.png")
    )

    compare_subject_accuracy_across_windows(
        base_dir=SS_base,
        save_path=os.path.join(COMPrun, "SS_subject_vs_window.png")
    )
    compare_subject_accuracy_across_windows(
        base_dir=SI_base,
        save_path=os.path.join(COMPrun, "SI_subject_vs_window.png")
    )

    compare_window_lengths_combined(
        SS_base=SS_base,
        SI_base=SI_base,
        save_path=os.path.join(COMPrun, "SS_SI_window_length_comparison.png")
    )

    compare_accuracy_vs_subjects_all_windows(
        SS_base=SS_base,
        SI_base=SI_base,
        save_path=os.path.join(COMPrun, "accuracy_vs_subjects_all_windows.png")
    )

    print("\nDone. All comparison plots saved.\n")


if __name__ == "__main__":
    main()
