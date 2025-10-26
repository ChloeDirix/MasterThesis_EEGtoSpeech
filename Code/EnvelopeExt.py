import os

import numpy as np             # For numerical arrays and math
import mne                     # EEG processing utilities (filtering, resampling)
import librosa                 # For loading audio (.wav) files and resampling
from scipy.signal import hilbert  # For signal processing (filtering, envelope)

import LoadSubject
import MatlabHelper as MH
import PlotUtils as PU

def extract_envelopes(subject,trial,cfg):

    stim_dir=subject.stim_dir
    out_dir=os.path.join(cfg["base_dir"], cfg["Env_dir"])
    os.makedirs(out_dir, exist_ok=True)

    plot_Envelope_Checks = cfg["plot_Envelope_Checks"]
    target_fs = cfg["target_fs"]
    plot_seconds = cfg["plot_seconds"]

    eeg_PP=trial.eeg_PP

    # ------------------ Speech envelope extraction ------------------
    env_left, env_right = None, None
    if len(trial.stim_names) >= 2:
        # each trail has metadata telling the audio files that were played
        left_path = os.path.join(stim_dir, trial.stim_names[0])
        right_path = os.path.join(stim_dir, trial.stim_names[1])

        # load both stimuli using librosa
        audio_left, fs_audio = librosa.load(left_path, sr=None, mono=True)
        audio_right, _ = librosa.load(right_path, sr=None, mono=True)
        # compute speech envelopes
        env_left = extract_envelope(audio_left, fs_audio, target_fs)
        env_right = extract_envelope(audio_right, fs_audio, target_fs)


    # Match lengths
    eeg_pre, env_left, env_right = MH.align_lengths(eeg_PP, env_left, env_right)

    # Attended vs unattended
    env_att, env_unatt = MH.get_attended(trial.attended_ear, env_left, env_right)

    # Diagnostic plot
    if plot_Envelope_Checks:
        PU.plot_trial_diagnostics(trial.index, eeg_pre, trial.fs, env_att, env_unatt, plot_seconds)


    #save
    if env_att is not None:
        np.save(os.path.join(out_dir, f"{trial.subject_id}_trial{trial.index:02d}_env_att.npy"), env_att)
    if env_unatt is not None:
        np.save(os.path.join(out_dir, f"{trial.subject_id}_trial{trial.index:02d}_env_unatt.npy"), env_unatt)

    print(f" Saved trial {trial.index}: EEG {eeg_pre.shape}, env_att {None if env_att is None else len(env_att)}")



# --- Speech envelope extraction using Hilbert + resample---
def extract_envelope(audio, fs_audio, target_fs):

    # Ensure correct type and shape
    audio = np.asarray(audio, dtype=np.float64).squeeze()

    # Get analytic signal -> envelope magnitude
    env = np.abs(hilbert(audio))

    # Low-pass (<9 Hz) to match EEG envelope dynamics
    env = mne.filter.filter_data(env, sfreq=fs_audio, l_freq=None, h_freq=9.0,
                                 method="fir", verbose=False)

    # Resample to EEG rate
    env = librosa.resample(y=env, orig_sr=fs_audio, target_sr=target_fs)

    # Normalize amplitude to [−1, 1]
    env /= np.max(np.abs(env))
    return env.astype(np.float32)

