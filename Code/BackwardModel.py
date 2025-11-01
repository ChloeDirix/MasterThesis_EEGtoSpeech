import os
import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import toeplitz

def create_lag_matrix(X, lags):
    n_samples, n_ch = X.shape
    lagged = []
    for lag in lags:
        if lag < 0:
            # EEG earlier (shift forward)
            rolled = np.roll(X, -lag, axis=0)
            rolled[-lag:, :] = 0
        elif lag > 0:
            # EEG later (shift backward)
            rolled = np.roll(X, -lag, axis=0)
            rolled[:lag, :] = 0
        else:
            rolled = X.copy()
        lagged.append(rolled)
    return np.hstack(lagged)



def train_backward_model(eeg, envelope, fs, lambda_val=0.01, lag_ms=(-100,400)):
    """
    eeg: (samples × channels)
    envelope: (samples,)
    lag_ms: min and max lag in milliseconds
    """
    lag_samp = np.arange(int(lag_ms[0]/1000*fs),
                         int(lag_ms[1]/1000*fs))

    X_lagged = create_lag_matrix(eeg, lag_samp)
    y = envelope.reshape(-1,1)
    X_lagged = (X_lagged - X_lagged.mean(axis=0)) / X_lagged.std(axis=0)
    y = (y - y.mean()) / y.std()

    model = Ridge(alpha=lambda_val)
    model.fit(X_lagged, y)
    return model, lag_samp


def evaluate_model(model, eeg, envelope, lags):

    X_lagged = create_lag_matrix(eeg, lags)
    pred = model.predict(X_lagged).ravel()
    corr = np.corrcoef(pred, envelope)[0,1]
    return corr, pred