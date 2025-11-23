import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import welch
from scipy.stats import zscore, pearsonr


def run_trial_diagnostics(subject_id, trial_index, fs=64, max_lag_ms=500, cfg_file="config.yaml"):
    """
    Comprehensive EEG–envelope check for one trial.
    """

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    cfg = yaml.safe_load(open(cfg_file, "r"))
    base_dir = cfg["base_dir"]
    pp_file = os.path.join(base_dir, cfg["PP_dir"], f"{subject_id}_trial{trial_index:02d}_preprocessed.npy")
    env_file = os.path.join(base_dir, cfg["Env_dir"], f"{subject_id}_trial{trial_index:02d}_env_att.npy")

    eeg = np.load(pp_file)        # shape: (samples, channels)
    env = np.load(env_file)       # shape: (samples,)
    print(f"\n--- TRIAL {trial_index} DIAGNOSTIC ---")
    print(f"EEG shape: {eeg.shape}, Envelope shape: {env.shape}")

    assert eeg.shape[0] == env.shape[0], "EEG and envelope length mismatch"

    # ------------------------------------------------
    # Basic stats.json
    # ------------------------------------------------
    def quick_stats(name, x):
        print(f"{name:20s} mean={np.mean(x):8.3g}  std={np.std(x):8.3g}  min={np.min(x):8.3g}  max={np.max(x):8.3g}")

    quick_stats("EEG ch0 (raw)", eeg[:,0])
    quick_stats("Envelope (raw)", env)

    # ------------------------------------------------
    # Preprocess: trim + demean + zscore
    # ------------------------------------------------
    trim_sec = 0.5
    trim = int(trim_sec * fs)
    eeg = eeg[trim:]
    env = env[trim:]
    eeg = zscore(eeg, axis=0)
    env = zscore(env)
    T = len(env)
    t = np.arange(T) / fs

    # ------------------------------------------------
    # 1️⃣ Time-domain comparison
    # ------------------------------------------------
    plt.figure(figsize=(12,4))
    plt.plot(t[:fs*10], env[:fs*10], label="Envelope (z-scored)", lw=1.5)
    plt.plot(t[:fs*10], eeg[:fs*10,0], label="EEG channel 0 (z-scored)", lw=1)
    plt.title(f"Trial {trial_index}: First 10 s (EEG ch0 vs Envelope)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
    # 👉 You should see similar slow-varying fluctuations. EEG is noisier, but envelopes roughly follow speech rhythm.

    # ------------------------------------------------
    # 2️⃣ Power spectral density (Welch)
    # ------------------------------------------------
    f_env, P_env = welch(env, fs=fs, nperseg=1024)
    f_eeg, P_eeg = welch(eeg[:,0], fs=fs, nperseg=1024)

    plt.figure(figsize=(8,4))
    plt.semilogy(f_env, P_env, label="Envelope")
    plt.semilogy(f_eeg, P_eeg, label="EEG ch0")
    plt.xlim([0,20])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power spectral density")
    plt.legend()
    plt.show()
    # 👉 Expect envelope energy < 8 Hz (speech modulation band) and EEG broader with peaks around 8–12 Hz (alpha).

    # ------------------------------------------------
    # 3️⃣ Zero-lag Pearson correlations per channel
    # ------------------------------------------------
    corrs = [pearsonr(eeg[:,ch], env)[0] for ch in range(eeg.shape[1])]
    plt.figure(figsize=(6,3))
    plt.bar(np.arange(len(corrs)), corrs)
    plt.title("Zero-lag correlation (each channel)")
    plt.xlabel("Channel index")
    plt.ylabel("r")
    plt.show()
    # 👉 Most correlations near 0; a few channels may stand out slightly positive or negative.

    top_ch = int(np.argmax(np.abs(corrs)))
    print(f"Top correlated channel: {top_ch}, r = {corrs[top_ch]:.3f}")

    # ------------------------------------------------
    # 4️⃣ Lagged cross-correlation for top channel
    # ------------------------------------------------
    maxlag = int(max_lag_ms/1000 * fs)
    lags = np.arange(-maxlag, maxlag+1)
    r = np.empty(len(lags))
    x = eeg[:,top_ch]; y = env
    for i, lag in enumerate(lags):
        if lag < 0:
            r[i] = np.corrcoef(x[:lag], y[-lag:])[0,1]
        elif lag > 0:
            r[i] = np.corrcoef(x[lag:], y[:-lag])[0,1]
        else:
            r[i] = np.corrcoef(x, y)[0,1]
    best_idx = np.nanargmax(np.abs(r))
    best_lag = lags[best_idx] / fs * 1000
    best_r = r[best_idx]

    plt.figure(figsize=(8,4))
    plt.plot(lags * 1000 / fs, r, lw=1.5)
    plt.axvline(best_lag, color='k', ls='--', label=f"best lag {best_lag:.1f} ms")
    plt.xlabel("Lag (ms)")
    plt.ylabel("Correlation r")
    plt.title(f"Cross-correlation (EEG ch{top_ch} vs Envelope)")
    plt.legend()
    plt.show()
    # 👉 Expect a small positive peak (r≈0.02–0.05) around 100–200 ms if alignment is good (EEG lags speech).

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------
    print(f"Peak correlation = {best_r:.3f} at lag ≈ {best_lag:.1f} ms")
    print("Typical EEG–speech lag: 100–200 ms; values near that suggest good temporal alignment.")

    return {
        "corrs": corrs,
        "top_channel": top_ch,
        "best_r": best_r,
        "best_lag_ms": best_lag
    }

# Example call
if __name__ == "__main__":
    run_trial_diagnostics("S1", trial_index=5, fs=128)
    run_trial_diagnostics("S1", trial_index=18, fs=128)
    run_trial_diagnostics("S1", trial_index=4, fs=128)
