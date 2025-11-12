import os
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import Ridge, LinearRegression


import numpy as np

def create_lag_matrix(X, lags):

    n_samples, n_ch = X.shape
    lags = np.asarray(lags)
    n_lags = len(lags)
    X_lagged = np.zeros((n_samples, n_lags * n_ch), dtype=X.dtype)

    for i, lag in enumerate(lags):
        rolled = np.roll(X, -lag, axis=0)
        if lag < 0:
            rolled[-lag:, :] = 0
        elif lag > 0:
            rolled[:lag, :] = 0
        X_lagged[:, i*n_ch:(i+1)*n_ch] = rolled

    return X_lagged


# def train_backward_model(eeg, envelope, fs, lambda_val=0, lag_ms=(-100,400)):
#     eeg=zscore(eeg,axis=1)
#     envelope=zscore(envelope)
#     lag_samp = np.arange(int(lag_ms[0]/1000*fs),
#                          int(lag_ms[1]/1000*fs))
#
#
#     X_lagged = create_lag_matrix(eeg, lag_samp)
#     y = envelope.reshape(-1, 1)
#
#
#     if lambda_val!=None:
#         model = Ridge(alpha=lambda_val)
#         model.fit(X_lagged, y)
#     else:
#         model = LinearRegression()
#         model.fit(X_lagged, y)
    return model, lag_samp


def evaluate_model(model, X_lagged, envelope):
    pred = model.predict(X_lagged).ravel()
    num = np.dot(pred, envelope)
    denom = np.linalg.norm(pred) * np.linalg.norm(envelope)
    corr = num / denom if denom != 0 else 0
    return corr, pred