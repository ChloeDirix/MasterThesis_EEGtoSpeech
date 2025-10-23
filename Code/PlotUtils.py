import numpy as np
import matplotlib.pyplot as plt

def plot_trial_diagnostics(i, eeg, fs, env_att, env_unatt, plot_seconds=30):
    nplot = min(int(plot_seconds * fs), len(eeg))
    t = np.arange(nplot) / fs
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, eeg[:nplot, 0])
    plt.title(f"Trial {i} – EEG (channel 0)")
    plt.subplot(3, 1, 2)
    plt.plot(t, env_att[:nplot] if env_att is not None else [])
    plt.title("Attended envelope" if env_att is not None else "Attended envelope (missing)")
    plt.subplot(3, 1, 3)
    plt.plot(t, env_unatt[:nplot] if env_unatt is not None else [])
    plt.title("Unattended envelope" if env_unatt is not None else "Unattended envelope (missing)")
    plt.tight_layout()
    plt.show()
