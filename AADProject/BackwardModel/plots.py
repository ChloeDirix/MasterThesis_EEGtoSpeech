import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib.patches import Patch

def plot_subject_bars(full_accs, win_accs, subjects, save_path, mode, window_s=None):
    """
    Windowed accuracy bar plot, ordered DAS then DTU, with dataset colors + mean lines.
    full_accs is ignored (kept for backward compatibility with existing calls).
    """
    

    os.makedirs(save_path, exist_ok=True)

    def dataset_of(s: str) -> str:
        u = s.upper()
        if "DAS" in u:
            return "DAS"
        if "DTU" in u:
            return "DTU"
        return "UNK"

    def subj_num(s: str) -> int:
        m = re.search(r"S(\d+)", s.upper())
        return int(m.group(1)) if m else 9999

    # Pretty colors (CSS names are fine; hex is also ok)
    COLOR_DAS = "lightblue"
    COLOR_DTU = "lightgreen"
    COLOR_UNK = "lightgray"

    # Slightly darker for mean lines
    MEAN_DAS = "steelblue"  # darker teal-ish
    MEAN_DTU = "forestgreen"  # darker pink-ish
    MEAN_UNK = "darkgray"

    # Build + sort records
    records = []
    for subj, win in zip(subjects, win_accs):
        s = str(subj)
        ds = dataset_of(s)
        records.append((ds, s, float(win)))

    ds_order = {"DAS": 0, "DTU": 1, "UNK": 2}
    records.sort(key=lambda t: (ds_order.get(t[0], 99), subj_num(t[1])))

    ds_list   = [r[0] for r in records]
    subj_list = [r[1] for r in records]
    win_list  = np.array([r[2] for r in records], dtype=float)

    # Colors per bar
    colors = []
    for ds in ds_list:
        if ds == "DAS":
            colors.append(COLOR_DAS)
        elif ds == "DTU":
            colors.append(COLOR_DTU)
        else:
            colors.append(COLOR_UNK)

    x = np.arange(len(subj_list))

    plt.figure(figsize=(max(10, 0.35 * len(subj_list)), 5))
    plt.bar(x, win_list, color=colors, width=0.8)
    plt.grid(axis="y", alpha=0.25)
    plt.ylim(0, 1)

    plt.xticks(x, subj_list, rotation=45, ha="right")
    plt.ylabel("Windowed accuracy")
    plt.title(f"{mode} Model Accuracy per subject - window length={window_s} s")

    # Separator between DAS and DTU groups (if both exist)
    if "DAS" in ds_list and "DTU" in ds_list:
        last_das = max(i for i, ds in enumerate(ds_list) if ds == "DAS")
        plt.axvline(last_das + 0.5, linestyle="--", linewidth=1, alpha=0.7)

    # Mean lines per dataset spanning only their section
    def draw_mean_line(ds_name, mean_color):
        idx = np.array([i for i, ds in enumerate(ds_list) if ds == ds_name], dtype=int)
        if idx.size == 0:
            return None

        m = float(np.mean(win_list[idx]))

        # line spans only this dataset
        x0 = idx.min() - 0.4
        x1 = idx.max() + 0.4
        plt.hlines(m, x0, x1, colors=mean_color, linestyles="--", linewidth=2)

        # text slightly to the right of the last bar
        plt.text(
            x1 - 0.1,          # near the right edge of the dataset
            m + 0.015,         # slightly ABOVE the line
            f"avg = {m:.2f}",
            color=mean_color,
            va="bottom",
            ha="right",
            fontsize=9,
            fontweight="bold"
        )

        return m


    mean_das = draw_mean_line("DAS", MEAN_DAS)
    mean_dtu = draw_mean_line("DTU", MEAN_DTU)
    mean_unk = draw_mean_line("UNK", MEAN_UNK)

    # Legend
    handles = []
    if mean_das is not None:
        handles.append(Patch(facecolor=COLOR_DAS, label=f"DAS"))
        
    if mean_dtu is not None:
        handles.append(Patch(facecolor=COLOR_DTU, label=f"DTU"))
        
    if mean_unk is not None:
        handles.append(Patch(facecolor=COLOR_UNK, label=f"Unknown"))
        

    # A cleaner legend: show dataset colors + mean numbers as text rows
    if handles:
        plt.legend(handles=handles, loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "subject_bars_windowed.png"), dpi=200)
    plt.close()




def plot_correlation_distributions(all_att, all_unatt, save_path):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(all_att, bins=30, alpha=0.6, label="Attended")
    plt.hist(all_unatt, bins=30, alpha=0.6, label="Unattended")
    plt.xlabel("Correlation r")
    plt.ylabel("Count")
    plt.title("Distribution of Correlation Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "corr_distributions.png"))
    plt.close()


def plot_window_length_curve(results, window_len_s, save_path):
    os.makedirs(save_path, exist_ok=True)

    """
    results = list of subject dicts
    Each subject has "results" = trial list with variable #windows
    """

    # Extract all window lengths
    all_lengths = []
    all_accs = []
    for subj in results:
        for trial in subj["results"]:
            for w in trial["windows"]:
                length = w["end"] - w["start"]     # seconds
                all_lengths.append(length)
                all_accs.append(float(w["correct"]))

    # Aggregate mean acc per length
    lengths = np.array(all_lengths)
    accs = np.array(all_accs)

    unique_lengths = np.sort(np.unique(lengths))
    mean_acc = [accs[lengths == L].mean() for L in unique_lengths]

    plt.figure(figsize=(8, 5))
    plt.plot(unique_lengths, mean_acc, marker="o")
    plt.xlabel("Window length (s)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Window Length")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "window_length_curve.png"))
    plt.close()


def plot_trf_weights(w, n_channels, lags_ms, save_path):
    os.makedirs(save_path, exist_ok=True)

    """
    w : decoder weights, shape can be:
        - (n_features,)  for single-band envelope
        - (n_features, n_bands) for multiband envelope
    """

    w = np.array(w)

    # Total features
    n_features = w.shape[0]

    # Compute number of lags from known channels
    n_lags = n_features // n_channels

    # Reshape depending on multiband vs single-band
    if w.ndim == 1:
        # shape (n_features,)
        W = w.reshape(n_lags, n_channels)
        trf = W.mean(axis=1)

    elif w.ndim == 2:
        # shape (n_features, n_bands)
        n_bands = w.shape[1]
        W = w.reshape(n_lags, n_channels, n_bands)
        trf = W.mean(axis=(1, 2))

    else:
        raise ValueError("Unexpected decoder weight shape: {}".format(w.shape))

    # lags_ms must match shape
    if len(lags_ms) != n_lags:
        lags_ms = np.linspace(lags_ms[0], lags_ms[-1], n_lags)

    plt.figure(figsize=(8, 5))
    plt.plot(lags_ms, trf)
    plt.xlabel("Lag (ms)")
    plt.ylabel("Mean weight")
    plt.title("Decoder TRF (mean across channels)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "TRF_temporal.png"))
    plt.close()



def plot_window_heatmap(subject_result, subject_id, save_path):
    """
    Creates a heatmap of accuracy over windows for one subject
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Make a matrix: trials x windows
    max_w = max(len(trial["windows"]) for trial in subject_result["results"])
    mat = np.full((len(subject_result["results"]), max_w), np.nan)

    for ti, trial in enumerate(subject_result["results"]):
        for wi, win in enumerate(trial["windows"]):
            mat[ti, wi] = 1 if win["correct"] else 0

    plt.figure(figsize=(10, 6))
    plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Correct (1) / Incorrect (0)")
    plt.xlabel("Window #")
    plt.ylabel("Trial")
    plt.title(f"Window Accuracy Heatmap — Subject {subject_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"_window_heatmap_{subject_id}.png"))
    plt.close()
