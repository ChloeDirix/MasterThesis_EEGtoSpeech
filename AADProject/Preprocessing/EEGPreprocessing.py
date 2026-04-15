import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from scipy.signal import remez, resample_poly, butter, sosfiltfilt


def get_plot_dir(cfg, subject_id=None, dataset=None, trial_index=None):
    base_dir = cfg["preprocessing"]["plotting"].get("save_dir", "debug_plots")

    parts = [base_dir]
    if subject_id is not None:
        parts.append(str(subject_id))
    if dataset is not None:
        parts.append(str(dataset))
    if trial_index is not None:
        parts.append(f"trial_{int(trial_index):03d}")

    full_dir = os.path.join(*parts)
    os.makedirs(full_dir, exist_ok=True)
    return full_dir


def save_or_close(fig, out_path):
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.close(fig)


def rereference(eeg, method="Cz"):
    if method.lower() == "cz":
        CZ_INDEX = 47
        ref = eeg[:, [CZ_INDEX]]
        return eeg - ref

    elif method.lower() == "mean":
        ref = np.mean(eeg, axis=1, keepdims=True)
        return eeg - ref

    return eeg


def design_equiripple_bandpass(fs, HP, LP):
    Fst1 = HP - 0.45
    Fp1 = HP + 0.45
    Fp2 = LP - 0.45
    Fst2 = LP + 0.45

    bands = [0, Fst1, Fp1, Fp2, Fst2, fs / 2]
    desired = [0, 1, 0]
    numtaps = 513
    return remez(numtaps, bands, desired, fs=fs)


def design_butter_bandpass(fs, HP, LP, order=4):
    return butter(order, [HP, LP], btype="bandpass", fs=fs, output="sos")


def preprocess_trial(trial, cfg, subject_id=None, dataset=None):
    eeg_raw = np.asarray(trial.eeg_raw)
    eeg = eeg_raw.copy()

    fs_raw = trial.fs_eeg
    fs = fs_raw
    target_fs = cfg["preprocessing"]["target_fs"]

    HP, LP = cfg["preprocessing"]["band"]
    plot_steps = cfg["preprocessing"]["plotting"]["show_preprocessing_steps"]
    plot_seconds = cfg["preprocessing"]["plotting"]["seconds"]

    trial_index = int(trial.index)

    if plot_steps:
        plot_dir = get_plot_dir(
            cfg,
            subject_id=subject_id,
            dataset=dataset,
            trial_index=trial_index,
        )

        ch = 0
        nplot_raw = min(int(plot_seconds * fs_raw), eeg_raw.shape[0])
        t_raw = np.arange(nplot_raw) / fs_raw

        eeg_reref = rereference(eeg_raw, cfg["preprocessing"]["rereference_method"])

        sos = design_butter_bandpass(fs_raw, HP, LP, order=4)
        eeg_filt = sosfiltfilt(sos, eeg_reref, axis=0)

        if int(fs_raw) != target_fs:
            eeg_resamp = resample_poly(eeg_filt, target_fs, fs_raw, axis=0)
            fs_resamp = target_fs
        else:
            eeg_resamp = eeg_filt
            fs_resamp = fs_raw

        nplot_reref = min(int(plot_seconds * fs_raw), eeg_reref.shape[0])
        t_reref = np.arange(nplot_reref) / fs_raw

        nplot_filt = min(int(plot_seconds * fs_raw), eeg_filt.shape[0])
        t_filt = np.arange(nplot_filt) / fs_raw

        nplot_resamp = min(int(plot_seconds * fs_resamp), eeg_resamp.shape[0])
        t_resamp = np.arange(nplot_resamp) / fs_resamp

        fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
        fig.suptitle(
            f"EEG preprocessing | subj={subject_id} | ds={dataset} | trial={trial_index} | ch={ch} | band={HP}-{LP} Hz",
            fontsize=13,
        )

        axs[0].plot(t_raw, eeg_raw[:nplot_raw, ch])
        axs[0].set_title("Raw EEG")

        axs[1].plot(t_reref, eeg_reref[:nplot_reref, ch])
        axs[1].set_title(f"After rereference ({cfg['preprocessing']['rereference_method']})")

        axs[2].plot(t_filt, eeg_filt[:nplot_filt, ch])
        axs[2].set_title(f"After band-pass ({HP}-{LP} Hz)")

        axs[3].plot(t_resamp, eeg_resamp[:nplot_resamp, ch])
        axs[3].set_title(f"After resampling ({fs_resamp} Hz)")

        for ax in axs:
            ax.grid(True, alpha=0.3)

        fig_path = os.path.join(
            plot_dir,
            f"preproc_subj-{subject_id}_ds-{dataset}_trial-{trial_index:03d}_ch-{ch}.png"
        )
        save_or_close(fig, fig_path)

        # Also save a compact comparison of just filtered vs resampled
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(t_filt, eeg_filt[:nplot_filt, ch], label="filtered")
        ax2.plot(t_resamp, eeg_resamp[:nplot_resamp, ch], label="resampled", alpha=0.8)
        ax2.set_title(
            f"Filtered vs resampled | subj={subject_id} | ds={dataset} | trial={trial_index} | ch={ch}"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig2_path = os.path.join(
            plot_dir,
            f"compare_filtered_resampled_subj-{subject_id}_ds-{dataset}_trial-{trial_index:03d}_ch-{ch}.png"
        )
        save_or_close(fig2, fig2_path)

        # Use the precomputed signal so the processing path matches the plot
        eeg = eeg_resamp
        fs = fs_resamp

    else:
        eeg = rereference(eeg, cfg["preprocessing"]["rereference_method"])
        sos = design_butter_bandpass(fs, HP, LP, order=4)
        eeg = sosfiltfilt(sos, eeg, axis=0)

        if int(fs) != target_fs:
            eeg = resample_poly(eeg, target_fs, fs, axis=0)
            fs = target_fs

    return eeg, fs