#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

from paths import paths


# ============================================================
# File resolving (supports old + new run layouts)
# ============================================================

def _first_existing(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def resolve_results_json(run_dir):
    return _first_existing(
        os.path.join(run_dir, "mTRF_results_ALL.json"),
        os.path.join(run_dir, "mTRF_results.json"),
        os.path.join(run_dir, "ALL", "mTRF_results_ALL.json"),
        os.path.join(run_dir, "ALL", "mTRF_results.json"),
    )


def resolve_summary_csv(run_dir):
    return _first_existing(
        os.path.join(run_dir, "mTRF_summary_ALL.csv"),
        os.path.join(run_dir, "mTRF_summary.csv"),
        os.path.join(run_dir, "ALL", "mTRF_summary_ALL.csv"),
        os.path.join(run_dir, "ALL", "mTRF_summary.csv"),
    )


def resolve_config_yaml(run_dir):
    return _first_existing(
        os.path.join(run_dir, "config_used.yaml"),
        os.path.join(run_dir, "config.yaml"),
        os.path.join(run_dir, "config_copy.yaml"),
        os.path.join(run_dir, "ALL", "config_used.yaml"),
        os.path.join(run_dir, "ALL", "config.yaml"),
        os.path.join(run_dir, "ALL", "config_copy.yaml"),
    )


# ============================================================
# Loaders
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_summary(path):
    """
    Dict keyed by Subject_ID; skips MEAN rows.
    Expects columns:
      - Full_Trial_Accuracy
      - Windowed_Accuracy
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("Subject_ID") or "").strip()
            if not sid:
                continue
            if sid.upper().startswith("MEAN"):
                continue
            out[sid] = {
                "full_accuracy": float(row["Full_Trial_Accuracy"]),
                "window_accuracy": float(row["Windowed_Accuracy"]),
            }
    return out


def load_run(run_dir):
    csv_path = resolve_summary_csv(run_dir)
    json_path = resolve_results_json(run_dir)
    if not csv_path:
        raise FileNotFoundError(f"No summary CSV found in {run_dir}")
    if not json_path:
        raise FileNotFoundError(f"No results JSON found in {run_dir}")
    return load_csv_summary(csv_path), load_json(json_path)


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


def pick_run(base_dir, selector):
    selector = str(selector).strip()

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
            raise IndexError(f"{selector} out of range (0..{len(runs)-1})")
        return os.path.join(base_dir, runs[idx])

    if selector.startswith("run_"):
        p = os.path.join(base_dir, selector)
        if os.path.isdir(p):
            return p
        raise FileNotFoundError(f"Run {selector} not found in {base_dir}")

    if selector.isdigit():
        n = int(selector)
        p1 = os.path.join(base_dir, f"run_{n}")
        p2 = os.path.join(base_dir, f"run_{n:04d}")
        if os.path.isdir(p1):
            return p1
        if os.path.isdir(p2):
            return p2
        raise FileNotFoundError(f"Run id {n} not found in {base_dir}")

    raise ValueError(f"Unrecognized selector: {selector}")


def parse_run_list(base_dir, selectors):
    out, seen = [], set()
    for s in selectors:
        r = pick_run(base_dir, s)
        if r not in seen:
            out.append(r)
            seen.add(r)
    if not out:
        raise ValueError("No runs selected.")
    return out


# ============================================================
# Subject helpers
# ============================================================

def dataset_of(subject_id: str) -> str:
    u = str(subject_id).upper()
    if "DAS" in u:
        return "DAS"
    if "DTU" in u:
        return "DTU"
    return "UNK"


def subj_num(subject_id: str) -> int:
    m = re.search(r"S(\d+)", str(subject_id).upper())
    return int(m.group(1)) if m else 9999


def sort_subjects(subjects):
    ds_order = {"DAS": 0, "DTU": 1, "UNK": 2}
    return sorted(subjects, key=lambda s: (ds_order.get(dataset_of(s), 99), subj_num(s), str(s)))


# ============================================================
# Window length inference (prefer JSON, fallback YAML)
# ============================================================

def infer_window_s_from_json(results_json):
    lens = []
    for subj in results_json:
        for trial in subj.get("results", []):
            for w in trial.get("windows", []):
                try:
                    lens.append(float(w["end"]) - float(w["start"]))
                except Exception:
                    pass
    if not lens:
        return None
    return float(np.median(lens))


def infer_window_s(run_dir, results_json):
    w = infer_window_s_from_json(results_json)
    if w is not None and np.isfinite(w) and w > 0:
        return w

    cfg_path = resolve_config_yaml(run_dir)
    if cfg_path:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return float(cfg["backward_model"]["window_s"])
        except Exception:
            pass
    return None


# ============================================================
# Mean accuracy split by dataset (from CSV)
# ============================================================

def mean_acc_by_dataset(summary_csv):
    buckets = {"DAS": [], "DTU": [], "UNK": []}
    for sid, v in summary_csv.items():
        buckets[dataset_of(sid)].append(v["window_accuracy"])
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in buckets.items()}


# ============================================================
# Plot: mean accuracy vs window length split DAS/DTU
# ============================================================

def plot_mean_acc_vs_window_split(ss_rows, si_rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    C_DAS = "steelblue"
    C_DTU = "forestgreen"

    def arr(rows, key):
        w = np.array([r["window_s"] for r in rows], float)
        y = np.array([r[key] for r in rows], float)
        m = np.isfinite(w) & np.isfinite(y)
        return w[m], y[m]

    plt.figure(figsize=(10, 6))

    w, y = arr(ss_rows, "mean_DAS")
    if w.size:
        plt.plot(w, y, marker="o", linewidth=2, label="SS DAS", color=C_DAS)

    w, y = arr(ss_rows, "mean_DTU")
    if w.size:
        plt.plot(w, y, marker="o", linewidth=2, label="SS DTU", color=C_DTU)

    w, y = arr(si_rows, "mean_DAS")
    if w.size:
        plt.plot(w, y, marker="s", linestyle="--", linewidth=2, label="SI DAS", color=C_DAS)

    w, y = arr(si_rows, "mean_DTU")
    if w.size:
        plt.plot(w, y, marker="s", linestyle="--", linewidth=2, label="SI DTU", color=C_DTU)

    plt.xlabel("Window length (s)")
    plt.ylabel("Mean subject accuracy")
    plt.title("Mean Accuracy vs Window Length")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mean_accuracy_vs_window_split.png"), dpi=200)
    plt.close()


# ============================================================
# Color helpers: keep dataset hue, make SS dark and SI light
# ============================================================

def _to_rgb(c):
    return np.array(plt.matplotlib.colors.to_rgb(c), dtype=float)

def _clamp01(x):
    return np.clip(x, 0.0, 1.0)

def darken(color, amount=0.25):
    rgb = _to_rgb(color)
    return tuple(_clamp01(rgb * (1.0 - amount)))

def lighten(color, amount=0.35):
    rgb = _to_rgb(color)
    return tuple(_clamp01(rgb + (1.0 - rgb) * amount))


# ============================================================
# SS vs SI per-subject plot: dataset hue + SS dark / SI light
#   - SS: filled circle (dark)
#   - SI: hollow square (light face + darker edge)
#   - no x-offset
# ============================================================

# ============================================================
# accuracy per subject SS vs SI
# ============================================================

def dataset_of(s: str) -> str:
    u = str(s).upper()
    if "DAS" in u: return "DAS"
    if "DTU" in u: return "DTU"
    return "UNK"

def subj_num(s: str) -> int:
    m = re.search(r"S(\d+)", str(s).upper())
    return int(m.group(1)) if m else 9999

def sort_subjects(subjects):
    ds_order = {"DAS": 0, "DTU": 1, "UNK": 2}
    return sorted(subjects, key=lambda s: (ds_order.get(dataset_of(s), 99), subj_num(s), str(s)))

import matplotlib.colors as mcolors

def lighten_color(color, amount=0.5):
    try:
        c = np.array(mcolors.to_rgb(color))
    except ValueError:
        c = np.array(mcolors.to_rgb("gray"))
    return tuple(c + (1.0 - c) * amount)

def darken_color(color, amount=0.5):
    try:
        c = np.array(mcolors.to_rgb(color))
    except ValueError:        
        c = np.array(mcolors.to_rgb("gray"))
    return tuple(c * (1.0 - amount))    

def plot_dumbbell_ss_vs_si(
    ss_csv, si_csv, save_dir, window_s=None,
    title="SS vs SI per subject",
):
    os.makedirs(save_dir, exist_ok=True)

    common = sorted(set(ss_csv) & set(si_csv))
    if not common:
        return False

    subjects = sort_subjects(common)
    ss = np.array([ss_csv[s]["window_accuracy"] for s in subjects], float)
    si = np.array([si_csv[s]["window_accuracy"] for s in subjects], float)
    ds = [dataset_of(s) for s in subjects]

    # Dataset colors (your vibe)
    C_DAS = "steelblue"
    C_DTU = "forestgreen"
    C_UNK = "darkgray"
    c_map = {"DAS": C_DAS, "DTU": C_DTU, "UNK": C_UNK}
    base_colors = [c_map.get(d, C_UNK) for d in ds]
    

    # Lighten colors for better visibility
    SI_colors = [lighten_color(c, 0.40) for c in base_colors]
    ss_colors = [darken_color(c, 0.25) for c in base_colors]

    x = np.arange(len(subjects))

    plt.figure(figsize=(max(12, 0.40 * len(subjects)), 5.3))

    # connector lines (colored by dataset)
    for i in range(len(subjects)):
        plt.plot([x[i], x[i]], [ss[i], si[i]], color=base_colors[i], alpha=0.20, linewidth=4, zorder=1)
        plt.plot([x[i], x[i]], [0, min(ss[i], si[i])], color="lightgrey", alpha=0.40, linewidth=1, zorder=1)

    # markers:
    # SS = circle, SI = square (swap if you prefer)
    plt.scatter(x, ss, s=45, marker="o", c=ss_colors, edgecolors="black", linewidths=0.2,label="SS", zorder=3)
    plt.scatter(x, si, s=45, marker="s", c=SI_colors, edgecolors="black", linewidths=0.2,label="SI", zorder=3)

    # separator between DAS and DTU (if both exist)
    if "DAS" in ds and "DTU" in ds:
        last_das = max(i for i, d in enumerate(ds) if d == "DAS")
        plt.axvline(last_das + 0.5, linestyle="--", linewidth=1, alpha=0.5)

    plt.grid(axis="y", alpha=0.25)
    plt.ylim(0, 1)
    plt.xticks(x, subjects, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    wtxt = f" — window={window_s:.3g}s" if window_s is not None else ""
    plt.title(f"{title}{wtxt}")

    from matplotlib.patches import Patch
    patches = []
    if any(d == "DAS" for d in ds):
        patches.append(Patch(facecolor=lighten_color(base_colors[ds.index("DAS")], 0.35), edgecolor=darken_color(base_colors[ds.index("DAS")], 0.25), label="DAS"))
    if any(d == "DTU" for d in ds):
        patches.append(Patch(facecolor=lighten_color(base_colors[ds.index("DTU")], 0.35), edgecolor=darken_color(base_colors[ds.index("DTU")], 0.25), label="DTU"))
    if any(d == "UNK" for d in ds):
        patches.append(Patch(facecolor=lighten_color(base_colors[ds.index("UNK")], 0.35), edgecolor=darken_color(base_colors[ds.index("UNK")], 0.25), label="UNK"))

    leg1 = plt.legend(frameon=True, loc="upper left")
    plt.gca().add_artist(leg1)
    if patches:
        plt.legend(handles=patches, frameon=True, loc="lower left", title="Dataset")


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "SS_vs_SI_dumbbell_vertical.png"), dpi=200)
    plt.close()
    return True




# ============================================================
# Pair SS and SI runs by inferred window length
# ============================================================

def pair_by_window(ss_rows, si_rows, ndigits=6):
    ss_map = {round(r["window_s"], ndigits): r for r in ss_rows}
    si_map = {round(r["window_s"], ndigits): r for r in si_rows}
    common = sorted(set(ss_map) & set(si_map))
    return [(ss_map[w], si_map[w]) for w in common]


# ============================================================
# MESD (Geirnaert) from JSON
#   - Build p(tau) from windows[].correct
#   - Optionally only use NON-OVERLAPPING windows (default)
# ============================================================
def snap_tau(tau, ndigits=3):
    """
    Turn floaty taus into stable keys.
    ndigits=3 means 4.999999 -> 5.0, 10.0000001 -> 10.0, etc.
    """
    if tau is None or not np.isfinite(tau):
        return None
    return float(np.round(float(tau), ndigits))


def aggregate_curve_by_tau(rows, ndigits=3):
    """
    rows: list of dicts with keys window_s and p_tau
    Returns sorted unique taus and mean p_tau per tau.
    Also returns std and counts (useful for debug/plotting).
    """
    buckets = {}  # tau -> list of p
    for r in rows:
        p = r.get("p_tau", np.nan)
        tau = r.get("window_s", None)
        if not np.isfinite(p):
            continue
        tau = snap_tau(tau, ndigits=ndigits)
        if tau is None:
            continue
        buckets.setdefault(tau, []).append(float(p))

    taus = np.array(sorted(buckets.keys()), float)
    ps_mean = np.array([np.mean(buckets[t]) for t in taus], float)
    ps_std  = np.array([np.std(buckets[t])  for t in taus], float)
    counts  = np.array([len(buckets[t])     for t in taus], int)
    return taus, ps_mean, ps_std, counts


def plot_mesd_curve_with_points(res, label, save_dir, tau_pts=None, esd_pts=None):
    """
    Same as your plot_mesd_curve, but also shows evaluated points.
    """
    if res.get("reason") != "ok":
        return
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(res["tau_grid"], res["esd_grid"], linewidth=2, label=f"{label} ESD(τ)")

    if tau_pts is not None and esd_pts is not None and len(tau_pts) > 0:
        plt.plot(tau_pts, esd_pts, "o", markersize=6, label=f"{label} evaluated τ")

    plt.scatter([res["tau_opt"]], [res["MESD"]], s=60, marker="o", label=f"{label} optimum")
    plt.xlabel("Decision window length τ (s)")
    plt.ylabel("Expected switch duration ESD (s)")
    plt.title(f"{label} — MESD={res['MESD']:.2f}s at τ={res['tau_opt']:.2f}s, N={res['N_opt']}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{label}_MESD_curve.png"), dpi=200)
    plt.close()

def infer_tau_from_subject_json(subject_json):
    lens = []
    for tr in subject_json.get("results", []):
        for w in tr.get("windows", []):
            try:
                lens.append(float(w["end"]) - float(w["start"]))
            except Exception:
                pass
    if not lens:
        return None
    return float(np.median(lens))


def _nonoverlap_correct_flags(windows, eps=1e-9):
    """
    Greedy non-overlap selection:
      - sort by start
      - keep w if w.start >= last_end (within eps)
    Returns list of 0/1 values for correct.
    """
    if not windows:
        return []

    ws = []
    for w in windows:
        try:
            s = float(w["start"])
            e = float(w["end"])
            c = 1.0 if bool(w.get("correct", False)) else 0.0
            ws.append((s, e, c))
        except Exception:
            pass

    if not ws:
        return []

    ws.sort(key=lambda t: (t[0], t[1]))
    picked = []
    last_end = -np.inf

    for s, e, c in ws:
        if s + eps >= last_end:
            picked.append(c)
            last_end = e

    return picked


def subject_accuracy_from_json(subject_json, nonoverlap=True):
    vals = []
    for tr in subject_json.get("results", []):
        windows = tr.get("windows", [])
        if nonoverlap:
            vals.extend(_nonoverlap_correct_flags(windows))
        else:
            for w in windows:
                if "correct" in w:
                    vals.append(1.0 if bool(w["correct"]) else 0.0)

    if not vals:
        return np.nan
    return float(np.mean(vals))


def p_of_tau_from_run_json(run_json, nonoverlap=True):
    tau = None
    for subj in run_json:
        tau = infer_tau_from_subject_json(subj)
        if tau is not None:
            break
    if tau is None:
        return None, np.nan

    subj_ps = []
    for subj in run_json:
        p = subject_accuracy_from_json(subj, nonoverlap=nonoverlap)
        if np.isfinite(p):
            subj_ps.append(p)

    if not subj_ps:
        return tau, np.nan

    return tau, float(np.mean(subj_ps))


def _kbar(P0, r, N):
    return int(np.floor(np.log((r**N)*(1.0 - P0) + P0) / np.log(r) + 1.0))


def find_min_N(p, P0=0.8, c=0.65, Nmin=5):
    if not (0.5 < p < 1.0):
        return None
    r = p / (1.0 - p)
    N = Nmin
    while True:
        kbar = _kbar(P0, r, N)
        xbar = (kbar - 1) / (N - 1)
        if xbar >= c:
            return N
        N += 1
        if N > 5000:
            return None


def h_j_i(i, j, p):
    r = p / (1.0 - p)
    num1 = (j - i) / (2.0*p - 1.0)
    num2 = p * (r**(-j) - r**(-i)) / ((2.0*p - 1.0)**2)
    return num1 + num2


def esd_seconds(p, tau, N, c=0.65):
    if N is None or not np.isfinite(p) or p <= 0.5 or p >= 1.0:
        return np.nan
    r = p / (1.0 - p)
    kc = int(np.ceil(c*(N-1) + 1))
    if kc <= 1:
        return 0.0

    weights = np.array([r**(-l) for l in range(1, kc)], dtype=float)
    denom = float(np.sum(weights))
    if denom <= 0:
        return np.nan

    hs = np.array([h_j_i(i, kc, p) for i in range(1, kc)], dtype=float)
    numer = float(np.sum(weights * hs))
    return float(tau * (numer / denom))


def compute_mesd_from_curve(tau_points, p_points, K=1000, P0=0.8, c=0.65, Nmin=5):
    tau_points = np.array(tau_points, float)
    p_points = np.array(p_points, float)

    m = np.isfinite(tau_points) & np.isfinite(p_points) & (p_points > 0.5) & (p_points < 1.0)
    tau_points = tau_points[m]
    p_points = p_points[m]

    if tau_points.size < 2:
        return {
            "MESD": np.nan, "tau_opt": np.nan, "p_opt": np.nan, "N_opt": None,
            "reason": "Need >=2 (tau,p) points with p>0.5"
        }

    idx = np.argsort(tau_points)
    tau_points = tau_points[idx]
    p_points = p_points[idx]

    tau_grid = np.linspace(float(tau_points.min()), float(tau_points.max()), K)
    p_grid = np.interp(tau_grid, tau_points, p_points)

    esds = np.full(K, np.nan, float)
    Ns = [None] * K

    for i, (tau, p) in enumerate(zip(tau_grid, p_grid)):
        if p <= 0.5 or p >= 1.0:
            continue
        N = find_min_N(p, P0=P0, c=c, Nmin=Nmin)
        Ns[i] = N
        esds[i] = esd_seconds(p, tau, N, c=c)

    if not np.any(np.isfinite(esds)):
        return {
            "MESD": np.nan, "tau_opt": np.nan, "p_opt": np.nan, "N_opt": None,
            "reason": "No finite ESD values"
        }

    j = int(np.nanargmin(esds))
    return {
        "MESD": float(esds[j]),
        "tau_opt": float(tau_grid[j]),
        "p_opt": float(p_grid[j]),
        "N_opt": Ns[j],
        "tau_grid": tau_grid,
        "p_grid": p_grid,
        "esd_grid": esds,
        "N_grid": Ns,
        "reason": "ok"
    }


def plot_mesd_curve(res, label, save_dir):
    if res.get("reason") != "ok":
        return
    os.makedirs(save_dir, exist_ok=True)

    tau = res["tau_grid"]
    esd = res["esd_grid"]

    plt.figure(figsize=(10, 6))
    plt.plot(tau, esd, linewidth=2, label=f"{label} ESD(τ)")
    plt.scatter([res["tau_opt"]], [res["MESD"]], s=60, marker="o", label=f"{label} optimum")
    plt.xlabel("Decision window length τ (s)")
    plt.ylabel("Expected switch duration ESD (s)")
    plt.title(f"{label} — MESD={res['MESD']:.2f}s at τ={res['tau_opt']:.2f}s, N={res['N_opt']}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{label}_MESD_curve.png"), dpi=200)
    plt.close()


# ============================================================
# Collect rows
# ============================================================

def collect_rows_for_run_dirs(run_dirs, debug=False, mesd_nonoverlap=True):
    rows = []
    for run_dir in run_dirs:
        try:
            s_csv, s_json = load_run(run_dir)
            w = infer_window_s(run_dir, s_json)
            if w is None:
                if debug:
                    print(f"[WARN] window_s missing: {run_dir}")
                continue

            means = mean_acc_by_dataset(s_csv)
            tau, p = p_of_tau_from_run_json(s_json, nonoverlap=mesd_nonoverlap)

            rows.append({
                "run_dir": run_dir,
                "run_name": Path(run_dir).name,
                "window_s": float(w),
                "mean_DAS": means["DAS"],
                "mean_DTU": means["DTU"],
                "p_tau": p,
                "tau_json": tau
            })
        except Exception as e:
            if debug:
                print(f"[WARN] skip {run_dir}: {e}")

    rows.sort(key=lambda r: r["window_s"])
    return rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--list", action="store_true",
                        help="List available SS/SI runs with idx selectors and exit.")

    parser.add_argument("--ss", nargs="+", default=None,
                        help="SS run selectors: latest | idx:N | run_XXXX | XXXX | /path/to/run. "
                             "If omitted, all SS runs are used.")

    parser.add_argument("--si", nargs="+", default=None,
                        help="SI run selectors: latest | idx:N | run_XXXX | XXXX | /path/to/run. "
                             "If omitted, all SI runs are used.")

    parser.add_argument("--debug", action="store_true",
                        help="Print debug warnings.")

    # MESD params (paper defaults)
    parser.add_argument("--mesd_P0", type=float, default=0.8)
    parser.add_argument("--mesd_c", type=float, default=0.65)
    parser.add_argument("--mesd_Nmin", type=int, default=5)
    parser.add_argument("--mesd_K", type=int, default=1000)

    # Default ON: only use non-overlapping windows
    parser.add_argument("--mesd_allow_overlap", action="store_true",
                        help="If set, MESD uses ALL windows (including overlapping). "
                             "Default is non-overlapping only.")

    args = parser.parse_args()

    SS_base = paths.RESULTS_LIN / "SS"
    SI_base = paths.RESULTS_LIN / "SI"

    if args.list:
        print_run_list(SS_base, "SS")
        print_run_list(SI_base, "SI")
        return

    COMP_base = paths.RESULTS_LIN / "Comparisons"
    os.makedirs(COMP_base, exist_ok=True)
    COMPrun = paths.get_next_run_dir(COMP_base)
    os.makedirs(COMPrun, exist_ok=True)
    print(f"\nSaving comparison report to: {COMPrun}\n")

    if args.ss is None:
        ss_run_dirs = [os.path.join(SS_base, r) for r in list_available_runs(SS_base)]
    else:
        ss_run_dirs = parse_run_list(SS_base, args.ss)

    if args.si is None:
        si_run_dirs = [os.path.join(SI_base, r) for r in list_available_runs(SI_base)]
    else:
        si_run_dirs = parse_run_list(SI_base, args.si)

    mesd_nonoverlap = (not args.mesd_allow_overlap)

    ss_rows = collect_rows_for_run_dirs(ss_run_dirs, debug=args.debug, mesd_nonoverlap=mesd_nonoverlap)
    si_rows = collect_rows_for_run_dirs(si_run_dirs, debug=args.debug, mesd_nonoverlap=mesd_nonoverlap)

    plot_mean_acc_vs_window_split(ss_rows, si_rows, save_dir=str(COMPrun))

    pairs = pair_by_window(ss_rows, si_rows)
    if not pairs:
        print("[WARN] No SS/SI pairs found by window length intersection.")

    for ss_r, si_r in pairs:
        w = ss_r["window_s"]
        subdir = os.path.join(COMPrun, f"window_{w:.3g}s".replace(".", "p"))
        os.makedirs(subdir, exist_ok=True)

        try:
            ss_csv, _ = load_run(ss_r["run_dir"])
            si_csv, _ = load_run(si_r["run_dir"])
        except Exception as e:
            if args.debug:
                print(f"[WARN] load failed for window={w}: {e}")
            continue

        plot_dumbbell_ss_vs_si(ss_csv, si_csv, save_dir=subdir, window_s=w)

    # MESD
        # =========================
    # MESD (FIXED)
    #   - snap tau to avoid float-noise
    #   - aggregate p(tau) if multiple runs share same tau
    #   - plot evaluated tau points too
    # =========================

    # aggregate SS curve
    ss_taus_u, ss_ps_u, ss_ps_std, ss_counts = aggregate_curve_by_tau(ss_rows, ndigits=3)
    # aggregate SI curve
    si_taus_u, si_ps_u, si_ps_std, si_counts = aggregate_curve_by_tau(si_rows, ndigits=3)

    if args.debug:
        print("\n[MESD DEBUG] SS taus:", ss_taus_u.tolist())
        print("[MESD DEBUG] SS counts per tau:", ss_counts.tolist())
        print("[MESD DEBUG] SS p(tau):", [round(x, 4) for x in ss_ps_u.tolist()])

        print("\n[MESD DEBUG] SI taus:", si_taus_u.tolist())
        print("[MESD DEBUG] SI counts per tau:", si_counts.tolist())
        print("[MESD DEBUG] SI p(tau):", [round(x, 4) for x in si_ps_u.tolist()])

    res_ss = compute_mesd_from_curve(
        ss_taus_u, ss_ps_u,
        K=args.mesd_K, P0=args.mesd_P0, c=args.mesd_c, Nmin=args.mesd_Nmin
    )
    res_si = compute_mesd_from_curve(
        si_taus_u, si_ps_u,
        K=args.mesd_K, P0=args.mesd_P0, c=args.mesd_c, Nmin=args.mesd_Nmin
    )

    mode_txt = "NON-overlapping windows" if mesd_nonoverlap else "ALL windows (overlap allowed)"
    print(f"\n[MESD] Mode: {mode_txt}")
    print("[MESD] Parameters:", f"P0={args.mesd_P0}, c={args.mesd_c}, Nmin={args.mesd_Nmin}, K={args.mesd_K}")

    print("[MESD] SS:", res_ss["reason"])
    if res_ss["reason"] == "ok":
        print(f"  MESD={res_ss['MESD']:.3f}s  tau_opt={res_ss['tau_opt']:.3f}s  p_opt={res_ss['p_opt']:.3f}  N_opt={res_ss['N_opt']}")

    print("[MESD] SI:", res_si["reason"])
    if res_si["reason"] == "ok":
        print(f"  MESD={res_si['MESD']:.3f}s  tau_opt={res_si['tau_opt']:.3f}s  p_opt={res_si['p_opt']:.3f}  N_opt={res_si['N_opt']}")

    # also compute ESD at the evaluated tau points (for plot markers)
    def esd_at_points(taus, ps, P0, c, Nmin):
        out = []
        for tau, p in zip(taus, ps):
            if not (np.isfinite(tau) and np.isfinite(p) and 0.5 < p < 1.0):
                out.append(np.nan)
                continue
            N = find_min_N(p, P0=P0, c=c, Nmin=Nmin)
            out.append(esd_seconds(p, tau, N, c=c))
        return np.array(out, float)

    ss_esd_pts = esd_at_points(ss_taus_u, ss_ps_u, args.mesd_P0, args.mesd_c, args.mesd_Nmin)
    si_esd_pts = esd_at_points(si_taus_u, si_ps_u, args.mesd_P0, args.mesd_c, args.mesd_Nmin)

    plot_mesd_curve_with_points(res_ss, "SS", save_dir=str(COMPrun), tau_pts=ss_taus_u, esd_pts=ss_esd_pts)
    plot_mesd_curve_with_points(res_si, "SI", save_dir=str(COMPrun), tau_pts=si_taus_u, esd_pts=si_esd_pts)

    print("\nDone.\n")
    print("Key outputs:")
    print(f"  - {os.path.join(COMPrun, 'mean_accuracy_vs_window_split.png')}")
    if res_ss.get("reason") == "ok":
        print(f"  - {os.path.join(COMPrun, 'SS_MESD_curve.png')}")
    if res_si.get("reason") == "ok":
        print(f"  - {os.path.join(COMPrun, 'SI_MESD_curve.png')}")
    print("  - per-window folders: window_*/*SS_vs_SI_per_subject.png")


if __name__ == "__main__":
    main()
