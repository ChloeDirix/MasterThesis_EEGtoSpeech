import os
import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import toeplitz

def create_lag_matrix(X, lags):
    """
    X: EEG (samples × channels)
    lags: list/array of lags in samples (e.g., -32..32)
    """
    n_samples, n_ch = X.shape
    lagged = []

    for lag in lags:
        if lag < 0:
            lagged.append(np.vstack([X[-lag:], np.zeros((-lag, n_ch))]))
        elif lag > 0:
            lagged.append(np.vstack([np.zeros((lag, n_ch)), X[:-lag]]))
        else:
            lagged.append(X)

    return np.hstack(lagged)


def train_backward_model(eeg, envelope, fs, lambda_val=1.0, lag_ms=(-100,400)):
    """
    eeg: (samples × channels)
    envelope: (samples,)
    lag_ms: min and max lag in milliseconds
    """
    lag_samp = np.arange(int(lag_ms[0]/1000*fs),
                         int(lag_ms[1]/1000*fs))

    X_lagged = create_lag_matrix(eeg, lag_samp)
    y = envelope.reshape(-1,1)

    model = Ridge(alpha=lambda_val)
    model.fit(X_lagged, y)
    return model, lag_samp


def evaluate_model(model, eeg, envelope, lags):
    X_lagged = create_lag_matrix(eeg, lags)
    pred = model.predict(X_lagged).ravel()
    corr = np.corrcoef(pred, envelope)[0,1]
    return corr, pred
