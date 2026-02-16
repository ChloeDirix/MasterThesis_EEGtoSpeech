# evaluate_results.py
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import binomtest


def SummaryStats(r_att, r_unatt):
    r_att = np.array(r_att)
    r_unatt = np.array(r_unatt)
    valid = ~np.isnan(r_att)
    r_att = r_att[valid]; r_unatt = r_unatt[valid]
    acc = np.mean(r_att > r_unatt) * 100.0
    mean_att, sem_att = np.nanmean(r_att), stats.sem(r_att, nan_policy='omit')
    mean_un, sem_un = np.nanmean(r_unatt), stats.sem(r_unatt, nan_policy='omit')
    print("Trials evaluated:", len(r_att))
    print(f"Mean r_att = {mean_att:.4f} ± {sem_att:.4f}")
    print(f"Mean r_unatt = {mean_un:.4f} ± {sem_un:.4f}")
    print(f"Decoding accuracy (r_att>r_unatt): {acc:.1f}%")
    # paired test
    tstat, pval = stats.ttest_rel(r_att, r_unatt, nan_policy='omit')
    print(f"Paired t-test r_att vs r_unatt: t={tstat:.3f}, p={pval:.3e}")
    # binomial test vs chance (50%)
    n_correct = np.sum(r_att > r_unatt)
    n_total = len(r_att)
    p_binom = binomtest(n_correct, n_total, p=0.5, alternative='greater').pvalue
    print(f"Binomial test: {n_correct}/{n_total} correct, p = {p_binom:.3e}")
    return dict(mean_att=mean_att, mean_un=mean_un, acc=acc, t_p=pval, binom_p=p_binom)

def plot_histograms(r_att, r_unatt, save_path):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(7,4))
    plt.hist(r_att, bins=20, alpha=0.7, label='r_att')
    plt.hist(r_unatt, bins=20, alpha=0.7, label='r_unatt')
    plt.xlabel('Pearson r')
    plt.legend()
    plt.title('Distribution of correlations across trials')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "Histogram"), dpi=300)
        print(f"Histogram plot saved to: {os.path.join(save_path, 'Histogram')}")

    plt.show()


# ---------------------------
# Permutation (circular shift) test: compute null distribution per trial
# ---------------------------
def circular_shift(arr, shift):
    """circular shift 1D array"""
    return np.concatenate([arr[shift:], arr[:shift]])

def trial_level_permutation_test(eeg_test, y_hat, env_att, env_unatt=None, n_perm=200, min_shift_sec=1, fs=128):
    """
    For one trial: produce null distribution of correlations by circularly shifting the true envelope
    relative to prediction. Returns observed r_att and p-value (fraction of null >= observed).
    - eeg_test and y_hat are not needed here except for more advanced schemes; we use y_hat and env_att.
    """
    obs = pearson_r(y_hat, env_att)
    nulls = []
    N = len(env_att)
    min_shift = int(min_shift_sec * fs)
    for _ in range(n_perm):
        shift = random.randint(min_shift, N-min_shift-1)
        env_sh = circular_shift(env_att, shift)
        r = pearson_r(y_hat, env_sh)
        nulls.append(r)
    nulls = np.array(nulls)
    p = (np.sum(nulls >= obs) + 1) / (n_perm + 1)   # one-sided p
    return obs, p, nulls

def pearson_r(x, y):
    from scipy.stats import pearsonr
    if len(x)==0 or len(y)==0:
        return np.nan
    try:
        r = pearsonr(x, y)[0]
    except:
        r = np.nan
    if np.isnan(r):
        return 0.0
    return r
