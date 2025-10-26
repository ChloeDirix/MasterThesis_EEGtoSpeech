import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr


def create_lag_matrix(eeg, lags):
    """
    Creates a lagged feature matrix for backward model.
    EEG shape: (time, channels)
    Returns (time, channels * n_lags)
    """
    X = []

    for lag in lags:
        if lag < 0:
            X.append(np.pad(eeg[-lag:], ((0, -lag), (0,0)), 'constant'))
        else:
            X.append(np.pad(eeg[:-lag], ((lag,0), (0,0)), 'constant'))
    return np.hstack(X)

    print(X)

def train_backward_model(eeg, envelope, lags, alpha=1e3):
    """
    Train ridge regression model: EEG → envelope
    """
    X = create_lag_matrix(eeg, lags)
    y = envelope.reshape(-1, 1)
    w = Ridge(alpha=alpha)
    w.fit(X, y)
    return w


def reconstruct_envelope(model, eeg, lags):
    """
    Predict speech envelope from EEG
    """
    X = create_lag_matrix(eeg, lags)
    return model.predict(X).flatten()


def evaluate_reconstruction(pred, true_env):
    """
    Returns Pearson correlation (r)
    """
    min_len = min(len(pred), len(true_env))
    return pearsonr(pred[:min_len], true_env[:min_len])[0]
