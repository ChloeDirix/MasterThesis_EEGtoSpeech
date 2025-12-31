import numpy as np
import matplotlib.pyplot as plt
import os

def plot_subject_bars(full_accs, win_accs, subjects, save_path):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(subjects))
    w = 0.35
    plt.bar(x - w/2, full_accs, width=w, label="Full-trial")
    plt.bar(x + w/2, win_accs, width=w, label="Windowed")
    plt.xticks(x, subjects, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Backward Model Accuracy per Subject")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "subject_bars.png"))
    plt.close()

    plt.show()

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
