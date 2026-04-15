#!/usr/bin/env python3
"""
Standalone diagnostics for AAD preprocessing outputs.

What it does
------------
- Reads preprocessed NWB files from both coarse and fine data folders.
- Compares the same subject/trial across variants.
- Produces simple overlay plots and summary stats without rerunning training.
- Verifies that fine and coarse really refer to the same stimulus pairing.
- Adds raw LEFT/RIGHT envelope comparisons to catch pairing issues.
- Adds EEG sign-flip overlay to check whether differences are mainly polarity.

Typical usage
-------------
python aad_preprocessing_checks.py --list-subjects
python aad_preprocessing_checks.py --subject sub-001_DAS
python aad_preprocessing_checks.py --subject S01_DTU --trial 3
python aad_preprocessing_checks.py --subject sub-001_DAS --seconds 20
python aad_preprocessing_checks.py --subject sub-001_DAS --show

Outputs are saved under:
Results_DL/preprocessing_checks/<subject>_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO
from fractions import Fraction
from scipy.signal import correlate, resample_poly, butter, sosfiltfilt
from paths import paths


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def lowpass_for_visual_compare(x: np.ndarray, fs: float, cutoff: float = 9.0, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)

def resample_to_match(x: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    if abs(fs_from - fs_to) < 1e-9:
        return x

    ratio = Fraction(fs_to / fs_from).limit_denominator(1000)
    up = ratio.numerator
    down = ratio.denominator
    return resample_poly(x, up, down, axis=0)


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class CompareStats:
    subject: str
    trial_index: int
    eeg_shape_fine: Tuple[int, int]
    eeg_shape_coarse: Tuple[int, int]
    env_att_shape_fine: Tuple[int, int]
    env_att_shape_coarse: Tuple[int, int]
    eeg_corr_mean: float
    eeg_abs_corr_mean: float
    eeg_abs_diff_mean: float
    eeg_abs_diff_max: float
    env_left_corr_mean: float
    env_right_corr_mean: float
    env_att_corr_mean: float
    env_att_abs_diff_mean: float
    env_unatt_corr_mean: float
    env_unatt_abs_diff_mean: float
    stimulus_pair_match: bool
    attended_ear_match: bool


class VariantPaths:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.eeg_dir = self.base_dir / "EEG_PP"
        self.env_dir = self.base_dir / "Envelopes"

    def subject_nwb(self, subject: str) -> Path:
        return self.eeg_dir / f"{subject}.nwb"

    def exists(self) -> bool:
        return self.eeg_dir.exists() and self.env_dir.exists()


# -----------------------------------------------------------------------------
# Locate datasets
# -----------------------------------------------------------------------------

def get_variant_dirs() -> Dict[str, VariantPaths]:
    root = Path("/user/leuven/373/vsc37381/scratch")
    variants = {
        "coarse": VariantPaths(root / "Data_InputModel"),
        "fine": VariantPaths(root / "Data_InputModelFine"),
    }
    return variants


# -----------------------------------------------------------------------------
# NWB reading
# -----------------------------------------------------------------------------

def read_trials_table(nwb_path: Path):
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        trials_df = nwbfile.trials.to_dataframe()
    return trials_df


def get_target_n_channels() -> int:
    cfg = paths.load_config()
    return int(cfg["preprocessing"]["target_n_channels"])


def get_target_fs() -> float:
    cfg = paths.load_config()
    return float(cfg["preprocessing"]["target_fs"])


def load_trial_from_nwb(nwb_path: Path, trial_index: int) -> Tuple[np.ndarray, float, dict]:
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        trials_df = nwbfile.trials.to_dataframe()

        match = trials_df[trials_df["trial_index"] == trial_index]
        if len(match) != 1:
            available = trials_df["trial_index"].tolist()
            raise ValueError(
                f"Trial index {trial_index} not found exactly once in {nwb_path}\n"
                f"Available trial_index values: {available}"
            )
        tr = match.iloc[0]

        pre_key = f"trial_{int(trial_index)}_EEG_preprocessed"
        ts = nwbfile.processing["eeg_preprocessed"].data_interfaces[pre_key]

        eeg = np.asarray(ts.data[:], dtype=np.float32)
        eeg = eeg[:, : get_target_n_channels()]

        if hasattr(ts, "rate") and ts.rate is not None:
            fs_eeg = float(ts.rate)
        elif hasattr(ts, "starting_time_rate") and ts.starting_time_rate is not None:
            fs_eeg = float(ts.starting_time_rate)
        else:
            raise ValueError(f"Could not determine EEG sampling rate for {pre_key} in {nwb_path}")

        meta = {
            "trial_index": int(tr["trial_index"]),
            "stim_L_name": str(tr["stim_L_name"]),
            "stim_R_name": str(tr["stim_R_name"]),
            "attended_ear": str(tr["attended_ear"]),
            "dataset": str(tr.get("dataset", "")),
            "acoustic_condition": int(tr["acoustic_condition"]) if "acoustic_condition" in tr else -1,
        }
        return eeg, fs_eeg, meta

# -----------------------------------------------------------------------------
# Envelope reading
# -----------------------------------------------------------------------------

def zscore_time(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma = np.maximum(sigma, 1e-12)
    return (x - mu) / sigma


def get_attended(left: np.ndarray, right: np.ndarray, attended_ear: str) -> Tuple[np.ndarray, np.ndarray]:
    s = attended_ear.lower()
    if "left" in s or s.startswith("l") or s.endswith("l"):
        return left, right
    if "right" in s or s.startswith("r") or s.endswith("r"):
        return right, left
    raise ValueError(f"Unknown attended_ear value: {attended_ear}")


def get_attended_fs(fs_left: float, fs_right: float, attended_ear: str) -> Tuple[float, float]:
    s = attended_ear.lower()
    if "left" in s or s.startswith("l") or s.endswith("l"):
        return fs_left, fs_right
    if "right" in s or s.startswith("r") or s.endswith("r"):
        return fs_right, fs_left
    raise ValueError(f"Unknown attended_ear value: {attended_ear}")


def load_env_lr(
    env_dir: Path,
    stim_left_name: str,
    stim_right_name: str,
) -> Tuple[np.ndarray, np.ndarray, float, float, Path, Path]:
    stimL = Path(stim_left_name).stem
    stimR = Path(stim_right_name).stem

    pathL = env_dir / f"{stimL}_env.npz"
    pathR = env_dir / f"{stimR}_env.npz"

    npzL = np.load(pathL)
    npzR = np.load(pathR)

    envL = np.asarray(npzL["envelope"], dtype=np.float32)
    envR = np.asarray(npzR["envelope"], dtype=np.float32)

    fsL = float(np.asarray(npzL["fs_env"]).ravel()[0])
    fsR = float(np.asarray(npzR["fs_env"]).ravel()[0])

    if envL.ndim == 1:
        envL = envL[:, None]
    if envR.ndim == 1:
        envR = envR[:, None]

    envL = zscore_time(np.sum(envL, axis=1, keepdims=True)).astype(np.float32)
    envR = zscore_time(np.sum(envR, axis=1, keepdims=True)).astype(np.float32)

    return envL, envR, fsL, fsR, pathL, pathR


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def align2(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = align2(np.asarray(x).ravel(), np.asarray(y).ravel())
    if len(x) == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mean_channel_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = align2(a, b)
    cors = []
    for ch in range(min(a.shape[1], b.shape[1])):
        r = safe_corr(a[:, ch], b[:, ch])
        if not np.isnan(r):
            cors.append(r)
    return float(np.mean(cors)) if cors else float("nan")


def mean_abs_channel_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = align2(a, b)
    cors = []
    for ch in range(min(a.shape[1], b.shape[1])):
        r = safe_corr(a[:, ch], b[:, ch])
        if not np.isnan(r):
            cors.append(abs(r))
    return float(np.mean(cors)) if cors else float("nan")


def mean_band_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = align2(a, b)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    cors = []
    for i in range(min(a.shape[1], b.shape[1])):
        r = safe_corr(a[:, i], b[:, i])
        if not np.isnan(r):
            cors.append(r)
    return float(np.mean(cors)) if cors else float("nan")


def estimate_best_lag_seconds(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: float = 2.0) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = min(len(x), len(y))
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])

    c = correlate(x, y, mode="full")
    lags = np.arange(-n + 1, n)
    max_lag = int(round(max_lag_s * fs))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    best = lags[mask][np.argmax(np.abs(c[mask]))]
    return float(best / fs)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_overlay_only(
    x_fine: np.ndarray,
    x_coarse: np.ndarray,
    fs: float,
    title: str,
    out_path: Path,
    seconds: float = 15.0,
    fine_label: str = "fine",
    coarse_label: str = "coarse",
    show: bool = False,
) -> None:
    x_fine, x_coarse = align2(x_fine, x_coarse)
    n = min(len(x_fine), int(round(seconds * fs)))
    x_fine = x_fine[:n]
    x_coarse = x_coarse[:n]
    t = np.arange(n) / fs

    plt.figure(figsize=(14, 4))
    plt.plot(t, x_coarse, label=coarse_label, alpha=0.8)
    plt.plot(t, x_fine, label=fine_label, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_overlay_only_normalized(
    x_fine: np.ndarray,
    x_coarse: np.ndarray,
    fs: float,
    title: str,
    out_path: Path,
    seconds: float = 15.0,
    fine_label: str = "fine",
    coarse_label: str = "coarse",
    show: bool = False,
) -> None:
    x_fine, x_coarse = align2(x_fine, x_coarse)
    n = min(len(x_fine), int(round(seconds * fs)))
    x_fine = x_fine[:n]
    x_coarse = x_coarse[:n]

    x_fine = (x_fine - np.mean(x_fine)) / max(np.std(x_fine), 1e-12)
    x_coarse = (x_coarse - np.mean(x_coarse)) / max(np.std(x_coarse), 1e-12)

    t = np.arange(n) / fs

    plt.figure(figsize=(14, 4))
    plt.plot(t, x_coarse, label=f"{coarse_label} (z-scored)", alpha=0.8)
    plt.plot(t, x_fine, label=f"{fine_label} (z-scored)", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Main comparison logic
# -----------------------------------------------------------------------------

def compare_subject_trial(
    subject: str,
    trial_index: int,
    fine_paths: VariantPaths,
    coarse_paths: VariantPaths,
    out_dir: Path,
    seconds: float,
    show: bool = False,
    manual_channel: int | None = None,
    extra_channels: list[int] | None = None,
) -> CompareStats:
    fine_nwb = fine_paths.subject_nwb(subject)
    coarse_nwb = coarse_paths.subject_nwb(subject)

    eeg_fine, fs_eeg_fine, meta_fine = load_trial_from_nwb(fine_nwb, trial_index)
    eeg_coarse, fs_eeg_coarse, meta_coarse = load_trial_from_nwb(coarse_nwb, trial_index)
    trial_dir = mkdir(out_dir / f"trial_{trial_index:03d}")

    # Hard metadata checks
    stimL_fine = Path(meta_fine["stim_L_name"]).stem
    stimL_coarse = Path(meta_coarse["stim_L_name"]).stem
    stimR_fine = Path(meta_fine["stim_R_name"]).stem
    stimR_coarse = Path(meta_coarse["stim_R_name"]).stem
    att_fine = str(meta_fine["attended_ear"]).lower()
    att_coarse = str(meta_coarse["attended_ear"]).lower()

    stim_pair_match = (stimL_fine == stimL_coarse) and (stimR_fine == stimR_coarse)
    attended_match = att_fine == att_coarse

    print("\n--- METADATA CHECK ---")
    print("fine   stim_L:", meta_fine["stim_L_name"])
    print("coarse stim_L:", meta_coarse["stim_L_name"])
    print("fine   stim_R:", meta_fine["stim_R_name"])
    print("coarse stim_R:", meta_coarse["stim_R_name"])
    print("fine   attended:", meta_fine["attended_ear"])
    print("coarse attended:", meta_coarse["attended_ear"])
    print("fine   dataset:", meta_fine["dataset"])
    print("coarse dataset:", meta_coarse["dataset"])
    print("fine   acoustic_condition:", meta_fine["acoustic_condition"])
    print("coarse acoustic_condition:", meta_coarse["acoustic_condition"])

    if not stim_pair_match or not attended_match:
        with open(trial_dir / "pairing_check.txt", "w", encoding="utf-8") as f:
            f.write(f"fine_nwb: {fine_nwb}\n")
            f.write(f"coarse_nwb: {coarse_nwb}\n")
            f.write(f"trial_index: {trial_index}\n\n")
            f.write(f"fine stim_L: {meta_fine['stim_L_name']}\n")
            f.write(f"coarse stim_L: {meta_coarse['stim_L_name']}\n")
            f.write(f"fine stim_R: {meta_fine['stim_R_name']}\n")
            f.write(f"coarse stim_R: {meta_coarse['stim_R_name']}\n")
            f.write(f"fine attended: {meta_fine['attended_ear']}\n")
            f.write(f"coarse attended: {meta_coarse['attended_ear']}\n")
            f.write(f"fine dataset: {meta_fine['dataset']}\n")
            f.write(f"coarse dataset: {meta_coarse['dataset']}\n")
            f.write(f"fine acoustic_condition: {meta_fine['acoustic_condition']}\n")
            f.write(f"coarse acoustic_condition: {meta_coarse['acoustic_condition']}\n")

        raise AssertionError(
            f"Metadata mismatch for {subject} trial {trial_index}. "
            f"See {trial_dir / 'pairing_check.txt'}"
        )

    # Raw LEFT/RIGHT envelopes first
    envL_fine, envR_fine, fsL_fine, fsR_fine, fine_envL, fine_envR = load_env_lr(
        fine_paths.env_dir,
        meta_fine["stim_L_name"],
        meta_fine["stim_R_name"],
    )
    envL_coarse, envR_coarse, fsL_coarse, fsR_coarse, coarse_envL, coarse_envR = load_env_lr(
        coarse_paths.env_dir,
        meta_coarse["stim_L_name"],
        meta_coarse["stim_R_name"],
    )

    print("\n--- ENVELOPE FS CHECK ---")
    print("fine   left fs :", fsL_fine)
    print("coarse left fs :", fsL_coarse)
    print("fine   right fs:", fsR_fine)
    print("coarse right fs:", fsR_coarse)
    print("fine   left len :", len(envL_fine))
    print("coarse left len :", len(envL_coarse))
    print("fine   right len:", len(envR_fine))
    print("coarse right len:", len(envR_coarse))

    with open(trial_dir / "pairing_check.txt", "w", encoding="utf-8") as f:
        f.write(f"fine_nwb: {fine_nwb}\n")
        f.write(f"coarse_nwb: {coarse_nwb}\n")
        f.write(f"trial_index: {trial_index}\n\n")
        f.write(f"fine stim_L: {meta_fine['stim_L_name']}\n")
        f.write(f"coarse stim_L: {meta_coarse['stim_L_name']}\n")
        f.write(f"fine stim_R: {meta_fine['stim_R_name']}\n")
        f.write(f"coarse stim_R: {meta_coarse['stim_R_name']}\n")
        f.write(f"fine attended: {meta_fine['attended_ear']}\n")
        f.write(f"coarse attended: {meta_coarse['attended_ear']}\n")
        f.write(f"fine env left file: {fine_envL}\n")
        f.write(f"coarse env left file: {coarse_envL}\n")
        f.write(f"fine env right file: {fine_envR}\n")
        f.write(f"coarse env right file: {coarse_envR}\n")
        f.write(f"fine env left fs: {fsL_fine}\n")
        f.write(f"coarse env left fs: {fsL_coarse}\n")
        f.write(f"fine env right fs: {fsR_fine}\n")
        f.write(f"coarse env right fs: {fsR_coarse}\n")

    # EEG uses project target fs
    eeg_coarse_rs = resample_to_match(eeg_coarse, fs_eeg_coarse, fs_eeg_fine)
    eeg_fine_a, eeg_coarse_a = align2(eeg_fine, eeg_coarse_rs)

    fs_eeg = fs_eeg_fine

    eeg_coarse_lp_a = lowpass_for_visual_compare(eeg_coarse_a, fs_eeg, cutoff=9.0, order=4)
    eeg_fine_lp_a = lowpass_for_visual_compare(eeg_fine_a, fs_eeg, cutoff=9.0, order=4)
    # Resample coarse envelopes to fine envelope fs before comparing
    envL_coarse_rs = resample_to_match(envL_coarse, fsL_coarse, fsL_fine)
    envR_coarse_rs = resample_to_match(envR_coarse, fsR_coarse, fsR_fine)

    envL_fine_a, envL_coarse_a = align2(envL_fine, envL_coarse_rs)
    envR_fine_a, envR_coarse_a = align2(envR_fine, envR_coarse_rs)

    # Recompute attended/unattended after resampling
    env_att_fine_a, env_unatt_fine_a = get_attended(envL_fine_a, envR_fine_a, meta_fine["attended_ear"])
    env_att_coarse_a, env_unatt_coarse_a = get_attended(envL_coarse_a, envR_coarse_a, meta_coarse["attended_ear"])

    # Use the correct envelope fs per signal
    fs_env_left = fsL_fine
    fs_env_right = fsR_fine
    fs_env_att, fs_env_unatt = get_attended_fs(fs_env_left, fs_env_right, meta_fine["attended_ear"])

    # Compute per-channel correlations
    ch_corrs = []
    ch_abs_corrs = []
    for ch in range(eeg_fine_a.shape[1]):
        r = safe_corr(eeg_fine_a[:, ch], eeg_coarse_a[:, ch])
        ch_corrs.append(r)
        ch_abs_corrs.append(abs(r) if not np.isnan(r) else np.nan)

    ch_corrs = np.asarray(ch_corrs, dtype=float)
    ch_abs_corrs = np.asarray(ch_abs_corrs, dtype=float)

    valid = np.where(~np.isnan(ch_corrs))[0]
    if len(valid) == 0:
        auto_plot_ch = 0
        best_ch = 0
        worst_ch = 0
    else:
        ordered = valid[np.argsort(ch_abs_corrs[valid])]
        worst_ch = int(ordered[0])
        auto_plot_ch = int(ordered[len(ordered) // 2])
        best_ch = int(ordered[-1])

    # Main channel to use in summary metrics
    if manual_channel is not None:
        if manual_channel < 0 or manual_channel >= eeg_fine_a.shape[1]:
            raise ValueError(
                f"Requested --channel {manual_channel}, but valid EEG channels are "
                f"0 to {eeg_fine_a.shape[1] - 1}"
            )
        plot_ch = int(manual_channel)
    else:
        plot_ch = auto_plot_ch

    # Channels to actually plot
    channels_to_plot = [plot_ch, best_ch, worst_ch, auto_plot_ch]
    if extra_channels is not None:
        channels_to_plot.extend(extra_channels)

    # remove duplicates, keep order, keep only valid
    seen = set()
    channels_to_plot_clean = []
    for ch in channels_to_plot:
        ch = int(ch)
        if 0 <= ch < eeg_fine_a.shape[1] and ch not in seen:
            channels_to_plot_clean.append(ch)
            seen.add(ch)

    print("\n--- EEG CHANNEL SUMMARY ---")
    print(f"main plot channel   : {plot_ch}")
    print(f"auto median channel : {auto_plot_ch}")
    print(f"best channel        : {best_ch}")
    print(f"worst channel       : {worst_ch}")
    print(f"channels to plot    : {channels_to_plot_clean}")

    # EEG overlays
    # EEG overlays for multiple channels
    for ch in channels_to_plot_clean:
        tag = []
        if ch == plot_ch:
            tag.append("main")
        if ch == auto_plot_ch:
            tag.append("median")
        if ch == best_ch:
            tag.append("best")
        if ch == worst_ch:
            tag.append("worst")
        tag_str = "_".join(tag) if tag else "extra"

        plot_overlay_only(
            eeg_fine_a[:, ch],
            eeg_coarse_a[:, ch],
            fs_eeg,
            f"EEG overlay | channel {ch} | {subject} trial {trial_index}",
            trial_dir / f"eeg_overlay_ch{ch}_{tag_str}.png",
            seconds=seconds,
            show=show,
        )
        plot_overlay_only_normalized(
            eeg_fine_a[:, ch],
            eeg_coarse_a[:, ch],
            fs_eeg,
            f"EEG normalized overlay | channel {ch} | {subject} trial {trial_index}",
            trial_dir / f"eeg_overlay_norm_ch{ch}_{tag_str}.png",
            seconds=seconds,
            show=show,
        )
        plot_overlay_only_normalized(
            eeg_fine_a[:, ch],
            -eeg_coarse_a[:, ch],
            fs_eeg,
            f"EEG normalized overlay with coarse inverted | channel {ch} | {subject} trial {trial_index}",
            trial_dir / f"eeg_overlay_norm_inverted_ch{ch}_{tag_str}.png",
            seconds=seconds,
            coarse_label="-coarse",
            show=show,
        )
        plot_overlay_only_normalized(
            eeg_fine_a[:, ch],
            eeg_coarse_lp_a[:, ch],
            fs_eeg,
            f"EEG normalized overlay | fine vs low-passed coarse | channel {ch} | {subject} trial {trial_index}",
            trial_dir / f"eeg_overlay_norm_coarseLP9_ch{ch}_{tag_str}.png",
            seconds=seconds,
            coarse_label="coarse LP9",
            show=show,
        )
        plot_overlay_only_normalized(
            eeg_fine_lp_a[:, ch],
            eeg_coarse_lp_a[:, ch],
            fs_eeg,
            f"EEG normalized overlay | both low-passed to 9 Hz | channel {ch} | {subject} trial {trial_index}",
            trial_dir / f"eeg_overlay_norm_bothLP9_ch{ch}_{tag_str}.png",
            seconds=seconds,
            fine_label="fine LP9",
            coarse_label="coarse LP9",
            show=show,
        )

    # Raw LEFT/RIGHT envelope overlays with their own fs
    plot_overlay_only(
        envL_fine_a[:, 0],
        envL_coarse_a[:, 0],
        fs_env_left,
        f"LEFT envelope overlay | {subject} trial {trial_index}",
        trial_dir / "env_left_overlay.png",
        seconds=seconds,
        show=show,
    )
    plot_overlay_only_normalized(
        envL_fine_a[:, 0],
        envL_coarse_a[:, 0],
        fs_env_left,
        f"LEFT envelope normalized overlay | {subject} trial {trial_index}",
        trial_dir / "env_left_overlay_norm.png",
        seconds=seconds,
        show=show,
    )

    plot_overlay_only(
        envR_fine_a[:, 0],
        envR_coarse_a[:, 0],
        fs_env_right,
        f"RIGHT envelope overlay | {subject} trial {trial_index}",
        trial_dir / "env_right_overlay.png",
        seconds=seconds,
        show=show,
    )
    plot_overlay_only_normalized(
        envR_fine_a[:, 0],
        envR_coarse_a[:, 0],
        fs_env_right,
        f"RIGHT envelope normalized overlay | {subject} trial {trial_index}",
        trial_dir / "env_right_overlay_norm.png",
        seconds=seconds,
        show=show,
    )

    # Attended/unattended overlays with the correct attended/unattended fs
    plot_overlay_only(
        env_att_fine_a[:, 0],
        env_att_coarse_a[:, 0],
        fs_env_att,
        f"Attended envelope overlay | {subject} trial {trial_index}",
        trial_dir / "env_att_overlay.png",
        seconds=seconds,
        show=show,
    )
    plot_overlay_only_normalized(
        env_att_fine_a[:, 0],
        env_att_coarse_a[:, 0],
        fs_env_att,
        f"Attended envelope normalized overlay | {subject} trial {trial_index}",
        trial_dir / "env_att_overlay_norm.png",
        seconds=seconds,
        show=show,
    )

    plot_overlay_only(
        env_unatt_fine_a[:, 0],
        env_unatt_coarse_a[:, 0],
        fs_env_unatt,
        f"Unattended envelope overlay | {subject} trial {trial_index}",
        trial_dir / "env_unatt_overlay.png",
        seconds=seconds,
        show=show,
    )
    plot_overlay_only_normalized(
        env_unatt_fine_a[:, 0],
        env_unatt_coarse_a[:, 0],
        fs_env_unatt,
        f"Unattended envelope normalized overlay | {subject} trial {trial_index}",
        trial_dir / "env_unatt_overlay_norm.png",
        seconds=seconds,
        show=show,
    )




    metrics = {
        "subject": subject,
        "trial_index": trial_index,
        "meta_fine": meta_fine,
        "meta_coarse": meta_coarse,
        "stimulus_pair_match": stim_pair_match,
        "attended_ear_match": attended_match,
        "plot_channel": plot_ch,
        "plot_channel_corr": None if np.isnan(ch_corrs[plot_ch]) else float(ch_corrs[plot_ch]),
        "plot_channel_abs_corr": None if np.isnan(ch_abs_corrs[plot_ch]) else float(ch_abs_corrs[plot_ch]),
        "fs_eeg": fs_eeg,
        "fs_env_fine_left": fsL_fine,
        "fs_env_fine_right": fsR_fine,
        "fs_env_coarse_left": fsL_coarse,
        "fs_env_coarse_right": fsR_coarse,
        "fs_env_left_plot": fs_env_left,
        "fs_env_right_plot": fs_env_right,
        "fs_env_att_plot": fs_env_att,
        "fs_env_unatt_plot": fs_env_unatt,
        "eeg_best_lag_seconds_plot_channel": estimate_best_lag_seconds(
            eeg_fine_a[:, plot_ch], eeg_coarse_a[:, plot_ch], fs_eeg
        ),
        "eeg_best_lag_seconds_plot_channel_inverted": estimate_best_lag_seconds(
            eeg_fine_a[:, plot_ch], -eeg_coarse_a[:, plot_ch], fs_eeg
        ),
        "env_left_best_lag_seconds": estimate_best_lag_seconds(
            envL_fine_a[:, 0], envL_coarse_a[:, 0], fs_env_left
        ),
        "env_right_best_lag_seconds": estimate_best_lag_seconds(
            envR_fine_a[:, 0], envR_coarse_a[:, 0], fs_env_right
        ),
        "env_att_best_lag_seconds": estimate_best_lag_seconds(
            env_att_fine_a[:, 0], env_att_coarse_a[:, 0], fs_env_att
        ),
        "env_unatt_best_lag_seconds": estimate_best_lag_seconds(
            env_unatt_fine_a[:, 0], env_unatt_coarse_a[:, 0], fs_env_unatt
        ),
        "channel_correlations": np.nan_to_num(ch_corrs, nan=0.0).tolist(),
        "channel_abs_correlations": np.nan_to_num(ch_abs_corrs, nan=0.0).tolist(),
        "fine_env_left_file": str(fine_envL),
        "fine_env_right_file": str(fine_envR),
        "coarse_env_left_file": str(coarse_envL),
        "coarse_env_right_file": str(coarse_envR),
        "left_env_corr": safe_corr(envL_fine_a[:, 0], envL_coarse_a[:, 0]),
        "right_env_corr": safe_corr(envR_fine_a[:, 0], envR_coarse_a[:, 0]),
        "att_env_corr": safe_corr(env_att_fine_a[:, 0], env_att_coarse_a[:, 0]),
        "unatt_env_corr": safe_corr(env_unatt_fine_a[:, 0], env_unatt_coarse_a[:, 0]),
        "eeg_corr_plot_channel": safe_corr(eeg_fine_a[:, plot_ch], eeg_coarse_a[:, plot_ch]),
        "eeg_corr_plot_channel_inverted": safe_corr(eeg_fine_a[:, plot_ch], -eeg_coarse_a[:, plot_ch]),
        "eeg_corr_plot_channel_coarse_lp9": safe_corr(eeg_fine_a[:, plot_ch], eeg_coarse_lp_a[:, plot_ch]),
        "eeg_corr_plot_channel_both_lp9": safe_corr(eeg_fine_lp_a[:, plot_ch], eeg_coarse_lp_a[:, plot_ch]),
        "auto_plot_channel": auto_plot_ch,
        "best_channel": best_ch,
        "worst_channel": worst_ch,
        "channels_plotted": channels_to_plot_clean,
    
    }
    with open(trial_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return CompareStats(
        subject=subject,
        trial_index=trial_index,
        eeg_shape_fine=tuple(eeg_fine.shape),
        eeg_shape_coarse=tuple(eeg_coarse.shape),
        env_att_shape_fine=tuple(env_att_fine_a.shape),
        env_att_shape_coarse=tuple(env_att_coarse_a.shape),
        eeg_corr_mean=mean_channel_corr(eeg_fine_a, eeg_coarse_a),
        eeg_abs_corr_mean=mean_abs_channel_corr(eeg_fine_a, eeg_coarse_a),
        eeg_abs_diff_mean=float(np.mean(np.abs(eeg_fine_a - eeg_coarse_a))),
        eeg_abs_diff_max=float(np.max(np.abs(eeg_fine_a - eeg_coarse_a))),
        env_left_corr_mean=mean_band_corr(envL_fine_a, envL_coarse_a),
        env_right_corr_mean=mean_band_corr(envR_fine_a, envR_coarse_a),
        env_att_corr_mean=mean_band_corr(env_att_fine_a, env_att_coarse_a),
        env_att_abs_diff_mean=float(np.mean(np.abs(env_att_fine_a - env_att_coarse_a))),
        env_unatt_corr_mean=mean_band_corr(env_unatt_fine_a, env_unatt_coarse_a),
        env_unatt_abs_diff_mean=float(np.mean(np.abs(env_unatt_fine_a - env_unatt_coarse_a))),
        stimulus_pair_match=stim_pair_match,
        attended_ear_match=attended_match,
    )


# -----------------------------------------------------------------------------
# Subject discovery
# -----------------------------------------------------------------------------

def list_common_subjects(fine_paths: VariantPaths, coarse_paths: VariantPaths) -> List[str]:
    fine = {p.stem for p in fine_paths.eeg_dir.glob("*.nwb")}
    coarse = {p.stem for p in coarse_paths.eeg_dir.glob("*.nwb")}
    return sorted(fine & coarse)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone preprocessing checks for AAD project.")
    p.add_argument("--subject", type=str, default=None, help="Subject stem, e.g. S01_DTU or 001_DAS")
    p.add_argument("--trial", type=int, default=None, help="Single trial index to inspect. Defaults to all trials.")
    p.add_argument("--seconds", type=float, default=15.0, help="How many seconds to show in time-domain plots.")
    p.add_argument("--list-subjects", action="store_true", help="List subjects available in both fine and coarse.")
    p.add_argument("--show", action="store_true", help="Show plots interactively.")
    p.add_argument("--channel", type=int, default=None,
               help="Manually choose one EEG channel to plot.")
    p.add_argument("--extra-channels", type=int, nargs="*", default=None,
               help="Additional EEG channels to plot, e.g. --extra-channels 0 10 20")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    variant_dirs = get_variant_dirs()
    fine_paths = variant_dirs["fine"]
    coarse_paths = variant_dirs["coarse"]

    print(f"Fine dir:   {fine_paths.base_dir}")
    print(f"Coarse dir: {coarse_paths.base_dir}")
    print(f"Fine EEG exists:   {fine_paths.eeg_dir.exists()}")
    print(f"Fine ENV exists:   {fine_paths.env_dir.exists()}")
    print(f"Coarse EEG exists: {coarse_paths.eeg_dir.exists()}")
    print(f"Coarse ENV exists: {coarse_paths.env_dir.exists()}")

    if not fine_paths.exists():
        raise FileNotFoundError(f"Fine variant folders not found under {fine_paths.base_dir}")
    if not coarse_paths.exists():
        raise FileNotFoundError(f"Coarse variant folders not found under {coarse_paths.base_dir}")

    common_subjects = list_common_subjects(fine_paths, coarse_paths)

    if args.list_subjects:
        print("Subjects available in both fine and coarse:")
        for s in common_subjects:
            print(s)
        return

    if args.subject is None:
        raise ValueError("Please provide --subject, or use --list-subjects first.")
    if args.subject not in common_subjects:
        raise ValueError(f"Subject {args.subject} not found in both fine and coarse variants.")

    trial_table = read_trials_table(fine_paths.subject_nwb(args.subject))
    if args.trial is None:
        trial_indices = [int(v) for v in trial_table["trial_index"].tolist()]
    else:
        trial_indices = [args.trial]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = mkdir(paths.RESULTS_DL / "preprocessing_checks" / f"{args.subject}_{timestamp}")

    summary: List[CompareStats] = []
    for trial_index in trial_indices:
        print(f"\nRunning checks for {args.subject} trial {trial_index}...")
        summary.append(
            compare_subject_trial(
                subject=args.subject,
                trial_index=trial_index,
                fine_paths=fine_paths,
                coarse_paths=coarse_paths,
                out_dir=out_dir,
                seconds=args.seconds,
                show=args.show,
                manual_channel=args.channel,
                extra_channels=args.extra_channels,
            )
        )

    summary_json = [asdict(s) for s in summary]
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nSaved diagnostics to: {out_dir}")
    print("Key per-trial summary:")
    for row in summary:
        print(
            f"trial={row.trial_index} | "
            f"stim_match={row.stimulus_pair_match} | "
            f"att_match={row.attended_ear_match} | "
            f"EEG corr={row.eeg_corr_mean:.4f} | "
            f"EEG |corr|={row.eeg_abs_corr_mean:.4f} | "
            f"LEFT env corr={row.env_left_corr_mean:.4f} | "
            f"RIGHT env corr={row.env_right_corr_mean:.4f} | "
            f"ATT env corr={row.env_att_corr_mean:.4f} | "
            f"UNATT env corr={row.env_unatt_corr_mean:.4f}"
        )


if __name__ == "__main__":
    main()