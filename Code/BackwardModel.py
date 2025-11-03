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

    lag_samp = np.arange(int(lag_ms[0]/1000*fs),
                         int(lag_ms[1]/1000*fs))


    X_lagged = create_lag_matrix(eeg, lag_samp)
    X_mean=X_lagged.mean(axis=0)
    X_std=X_lagged.std(axis=0)
    X_lagged = (X_lagged - X_mean ) / X_std

    y = envelope.reshape(-1, 1)
    y_mean=y.mean()
    y_std=y.std()
    y = (y - y_mean) / y_std

    model = Ridge(alpha=lambda_val)
    model.fit(X_lagged, y)
    return model, lag_samp, [X_mean,X_std,y_mean, y_std]


def evaluate_model(model, eeg, envelope, lags, mean_std_list):
    X_mean_train=mean_std_list[0]
    X_std_train=mean_std_list[1]
    y_mean_train=mean_std_list[2]
    y_std_train=mean_std_list[3]

    X_lagged = create_lag_matrix(eeg, lags)
    X_lagged = (X_lagged - X_mean_train) / X_std_train

    envelope=(envelope-y_mean_train) / y_std_train

    pred = model.predict(X_lagged).ravel()
    corr = np.corrcoef(pred, envelope)[0,1]


    return corr, pred