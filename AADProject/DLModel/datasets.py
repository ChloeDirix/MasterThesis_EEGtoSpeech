import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pynwb import NWBHDF5IO

from paths import paths


# =============================================================================
# Shape / validation helpers
# =============================================================================
def _assert_2d_time_feature_array(X, name: str, min_features: int = 1):
    """
    Enforce shape convention (T, D):
      T = time
      D = feature dimension (EEG channels or envelope bands)
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (T, D), got {X.shape}")

    if X.shape[0] < 2:
        raise ValueError(f"{name} must have at least 2 time samples, got {X.shape}")

    if X.shape[1] < min_features:
        raise ValueError(
            f"{name} must have at least {min_features} features in axis=1, got {X.shape}"
        )

    return X


def _warn_if_shape_looks_transposed(X, name: str):
    """
    Soft warning only. Typical EEG/envelope arrays should have T >> D.
    """
    if X.shape[0] < X.shape[1]:
        print(
            f"Warning: {name} shape {X.shape} looks unusual for (T, D). "
            f"Expected time on axis 0 and features on axis 1."
        )


# =============================================================================
# Envelope loading helpers
# =============================================================================
def _get_envelope(stim_base: str, sum_subbands: bool = True):
    """
    Load envelope from npz.

    Returns:
      sum_subbands=True  -> env shape (T, 1)
      sum_subbands=False -> env shape (T, B)
    """
    npz = np.load(paths.envelope(f"{stim_base}_env.npz"))
    env = np.asarray(npz["envelope"], dtype=np.float32)
    w = np.asarray(npz["subband_weights"], dtype=np.float32)

    env = _assert_2d_time_feature_array(env, f"Envelope[{stim_base}]")
    _warn_if_shape_looks_transposed(env, f"Envelope[{stim_base}]")

    if sum_subbands:
        env_sum = np.sum(env, axis=1, keepdims=True).astype(np.float32)
        return env_sum, w

    return env, w


def _align_lengths(*arrays):
    """
    Truncate all arrays to the shortest length along axis 0.
    """
    m = min(len(a) for a in arrays)
    return tuple(a[:m] for a in arrays)


def _attended_index(att_ear):
    """
    returns index for (L,R): left->0, right->1
    """
    s = str(att_ear).lower()
    if "left" in s or s.startswith("l") or s.endswith("l"):
        return 0
    if "right" in s or s.startswith("r") or s.endswith("r"):
        return 1
    raise ValueError(f"Unknown attended_ear value: {att_ear}")


# =============================================================================
# Normalization helpers
# =============================================================================
def zscore_trial(X):
    """
    Per-trial z-scoring along time axis.
    X: (T, C) or (T,M)  with C channels and M mutiband
    """

    X = np.asarray(X, dtype=np.float32)

    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0

    Xz = (X - mu) / sigma
    return Xz


def zscore_env_pair(envL, envR):
    """
    Joint per-trial z-scoring for the two candidate envelopes.
    envL, envR: (T, B) or (T,)
    Uses shared mean/std across both streams, per feature/band.
    """
    envL = np.asarray(envL, dtype=np.float32)
    envR = np.asarray(envR, dtype=np.float32)

    if envL.shape != envR.shape:
        raise ValueError(f"envL/envR shape mismatch: {envL.shape} vs {envR.shape}")

    squeeze_back = False
    if envL.ndim == 1:
        envL = envL[:, None]
        envR = envR[:, None]
        squeeze_back = True
    elif envL.ndim != 2:
        raise ValueError(f"zscore_env_pair expects 1D or 2D arrays, got {envL.shape}")

    both = np.concatenate([envL, envR], axis=0)  # (2T, B)
    mu = np.mean(both, axis=0, keepdims=True)
    sigma = np.std(both, axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0

    envLz = (envL - mu) / sigma
    envRz = (envR - mu) / sigma

    if squeeze_back:
        return envLz[:, 0], envRz[:, 0]
    return envLz, envRz

def _normalize_trial(
    eeg,
    envL,
    envR,
    norm_mode,
    eeg_std=None,
    envL_std=None,
    envR_std=None,
):
    """
    Normalize one full trial before window extraction.
    """
    norm_mode = str(norm_mode).lower()

    if norm_mode in {
        "global",
        "per_subject",
        "per_subject_train_global_val",
        "global_per_dataset",
    }:
        if eeg_std is None or envL_std is None or envR_std is None:
            raise ValueError(
                f"{norm_mode} normalization requested, but eeg_std/envL_std/envR_std is None"
            )

        eeg = eeg_std.transform(eeg)
        envL = envL_std.transform(envL)
        envR = envR_std.transform(envR)

    elif norm_mode == "per_trial":
        eeg = zscore_trial(eeg)
        envL = zscore_trial(envL)
        envR = zscore_trial(envR)

    elif norm_mode == "none":
        pass

    else:
        raise ValueError(
            f"Unknown norm_mode='{norm_mode}'. Use one of: "
            f"'none', 'per_trial', 'per_subject', 'global', "
            f"'per_subject_train_global_val', 'global_per_dataset'."
        )

    return eeg.astype(np.float32), envL.astype(np.float32), envR.astype(np.float32)

# =============================================================================
# Trial loading
# =============================================================================
def _load_data_from_trial(nwbfile, tr_row, cfg, sum_subbands: bool = True):
    """
    Loads EEG and L/R envelopes for a trial.
    Returns dict with:
      eeg(T,C), envL(T,D), envR(T,D), att(0/1), dataset key, etc.
    """

    if "trial_index" not in tr_row:
        raise KeyError(
            "trials_df does not contain 'trial_index' column, cannot map to EEG interfaces."
        )

    tid = int(tr_row["trial_index"])

    candidates = [
        f"trial_{tid}_EEG_preprocessed",
        f"trial_{tid+1}_EEG_preprocessed",
        f"trial_{tid-1}_EEG_preprocessed",
    ]

    di = nwbfile.processing["eeg_preprocessed"].data_interfaces

    eeg_key = None
    for k in candidates:
        if k in di:
            eeg_key = k
            break

    if eeg_key is None:
        available = list(di.keys())
        raise KeyError(
            f"None of {candidates} found in eeg_preprocessed.data_interfaces.\n"
            f"Example available keys: {available[:20]}"
        )

    eeg = np.asarray(di[eeg_key].data[:], dtype=np.float32)

    if eeg.ndim != 2:
        raise ValueError(f"Trial {tid}: EEG must be 2D (T,C), got {eeg.shape}")

    nC = int(cfg["preprocessing"]["target_n_channels"])
    if eeg.shape[1] < nC:
        raise ValueError(f"Trial {tid}: EEG has {eeg.shape[1]} channels, expected >= {nC}")

    eeg = eeg[:, :nC]
    eeg = _assert_2d_time_feature_array(eeg, f"EEG[trial={tid}]", min_features=nC)
    _warn_if_shape_looks_transposed(eeg, f"EEG[trial={tid}]")

    stimL_name = str(tr_row["stim_L_name"])
    stimR_name = str(tr_row["stim_R_name"])

    stimL = os.path.splitext(stimL_name)[0]
    stimR = os.path.splitext(stimR_name)[0]

    envL, _ = _get_envelope(stimL, sum_subbands=sum_subbands)
    envR, _ = _get_envelope(stimR, sum_subbands=sum_subbands)

    envL = _assert_2d_time_feature_array(envL, f"envL[{stimL}]")
    envR = _assert_2d_time_feature_array(envR, f"envR[{stimR}]")

    if envL.shape[1] != envR.shape[1]:
        raise ValueError(f"Envelope band mismatch: envL {envL.shape}, envR {envR.shape}")

    eeg, envL, envR = _align_lengths(eeg, envL, envR)

    if len(eeg) < 2:
        raise ValueError(f"Trial {tid}: too short after alignment, length={len(eeg)}")

    att = _attended_index(tr_row["attended_ear"])
    ds_key = str(tr_row.get("dataset", "")).upper()

    return {
        "eeg": eeg,
        "envL": envL,
        "envR": envR,
        "att": int(att),
        "dataset": ds_key,
        "trial_index": tid,
        "stim_L_name": stimL_name,
        "stim_R_name": stimR_name,
    }

def _merge_repetition_trial_dicts(trial_dicts):
    """
    Kept for explicit use only.
    For window-based DL this is usually NOT recommended.
    """
    long_trials = []
    rep_groups = {}

    for td in trial_dicts:
        stimL_base = os.path.splitext(str(td["stim_L_name"]))[0]

        if stimL_base.startswith("rep_"):
            orig = stimL_base[4:]
            rep_groups.setdefault(orig, []).append(td)
        else:
            long_trials.append(td)

    merged = []
    for orig_stim, group in rep_groups.items():
        if len(group) != 3:
            print(
                f"Warning: repetition group '{orig_stim}' has {len(group)} reps "
                f"(expected 3). Skipping merge."
            )
            merged.extend(group)
            continue

        group = sorted(group, key=lambda x: int(x.get("trial_index", 0)))

        eeg = np.concatenate([g["eeg"] for g in group], axis=0)
        envL = np.concatenate([g["envL"] for g in group], axis=0)
        envR = np.concatenate([g["envR"] for g in group], axis=0)

        atts = [g["att"] for g in group]
        if len(set(atts)) != 1:
            raise ValueError(
                f"Repetition group '{orig_stim}' has inconsistent attended labels: {atts}"
            )

        merged.append({
            "eeg": eeg,
            "envL": envL,
            "envR": envR,
            "att": int(atts[0]),
            "dataset": group[0]["dataset"],
            "subject": group[0]["subject"],
            "trial_index": f"merged_{orig_stim}",
            "stim_L_name": group[0]["stim_L_name"],
            "stim_R_name": group[0]["stim_R_name"],
        })

    return long_trials + merged


# =============================================================================
# Window selection helpers
# =============================================================================
def _uniform_subsample_sorted(starts: np.ndarray, max_keep: int) -> np.ndarray:
    if len(starts) <= max_keep:
        return starts
    idx = np.linspace(0, len(starts) - 1, num=max_keep)
    idx = np.round(idx).astype(int)
    idx = np.unique(idx)
    return starts[idx]


def _compute_window_starts(
    T: int,
    win_len: int,
    win_step: int,
    min_gap: int,
    selection_mode: str = "step",
    max_windows_per_trial: int = None,
):
    """
    Build window starts for one trial.

    selection_mode:
      - "step": regular starts using win_step, then min_gap filter
      - "non_overlapping": starts at 0, win_len, 2*win_len, ...
      - "last_only": only one window at the end
    """
    if T < win_len:
        return np.asarray([], dtype=int)

    selection_mode = str(selection_mode).lower()

    if selection_mode == "step":
        starts = np.arange(0, T - win_len + 1, win_step, dtype=int)

    elif selection_mode == "non_overlapping":
        starts = np.arange(0, T - win_len + 1, win_len, dtype=int)

    elif selection_mode == "last_only":
        starts = np.asarray([T - win_len], dtype=int)

    else:
        raise ValueError(
            f"Unknown selection_mode='{selection_mode}'. "
            f"Use one of: step, non_overlapping, last_only."
        )

    if min_gap > 0 and len(starts) > 1:
        kept = [int(starts[0])]
        last = int(starts[0])
        for s in starts[1:]:
            s = int(s)
            if (s - last) >= min_gap:
                kept.append(s)
                last = s
        starts = np.asarray(kept, dtype=int)

    if max_windows_per_trial is not None and max_windows_per_trial > 0:
        starts = _uniform_subsample_sorted(starts, int(max_windows_per_trial))

    return starts.astype(int)


# =============================================================================
# Global standardization
# =============================================================================
class Standardizer:
    """
    Column-wise standardizer for arrays shaped (T, D).
    For EEG: D = channels
    For ENV: D = bands
    """
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.n = 0
        self._sum = None
        self._sumsq = None

    def partial_fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        sx = np.sum(X, axis=0, keepdims=True)
        sxx = np.sum(X * X, axis=0, keepdims=True)

        if self._sum is None:
            self._sum = sx
            self._sumsq = sxx
        else:
            self._sum += sx
            self._sumsq += sxx

        self.n += X.shape[0]
        return self

    def finalize(self):
        if self.n == 0:
            raise ValueError("Standardizer.finalize() called with no data.")

        self.mu = self._sum / self.n
        var = self._sumsq / self.n - self.mu * self.mu          #Var(X)=E[X²]−(E[X])²
        var[var < 1e-8] = 1.0
        self.sigma = np.sqrt(var)
        return self

    def transform(self, X):
        if self.mu is None or self.sigma is None:
            raise ValueError("Standardizer.transform() called before finalize().")

        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mu) / self.sigma).astype(np.float32)

def fit_global_standardizers_per_dataset(nwb_paths, cfg, sum_subbands=True, merge_repetitions=False):
    """
    Fit one EEG, envL, and envR standardizer per dataset key, using TRAIN subjects only.

    Returns:
      dataset_stds: dict
        {
          "DAS": {"eeg_std": ..., "envL_std": ..., "envR_std": ...},
          "DTU": {"eeg_std": ..., "envL_std": ..., "envR_std": ...},
          ...
        }
    """
    dataset_stds = {}

    for nwb_path in nwb_paths:
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            trials_df = nwb.trials.to_dataframe()
            subj_key = os.path.splitext(os.path.basename(str(nwb_path)))[0]

            subject_trial_dicts = []
            for _, tr_row in trials_df.iterrows():
                td = _load_data_from_trial(
                nwbfile=nwb,
                tr_row=tr_row,
                cfg=cfg,
                sum_subbands=sum_subbands,
            )
                td["subject"] = subj_key
                subject_trial_dicts.append(td)

            if merge_repetitions:
                subject_trial_dicts = _merge_repetition_trial_dicts(subject_trial_dicts)

            for td in subject_trial_dicts:
                ds_key = str(td["dataset"]).upper()

                if ds_key not in dataset_stds:
                    dataset_stds[ds_key] = {
                        "eeg_std": Standardizer(),
                        "envL_std": Standardizer(),
                        "envR_std": Standardizer(),
                    }

                dataset_stds[ds_key]["eeg_std"].partial_fit(td["eeg"])
                dataset_stds[ds_key]["envL_std"].partial_fit(td["envL"])
                dataset_stds[ds_key]["envR_std"].partial_fit(td["envR"])

    for ds_key in dataset_stds:
        dataset_stds[ds_key]["eeg_std"].finalize()
        dataset_stds[ds_key]["envL_std"].finalize()
        dataset_stds[ds_key]["envR_std"].finalize()

    return dataset_stds

def fit_global_standardizers(nwb_paths, cfg, sum_subbands=True, merge_repetitions=False):
    """
    Fit EEG, envL, and envR standardizers on provided subjects only.
    Intended usage: training subjects only for norm_mode='global'.

    EEG stats:
      - pooled over all training trials/time points
      - per EEG channel

    ENV stats:
      - pooled over all training trials/time points
      - envL and envR fitted independently
      - per envelope band
    """
    eeg_std = Standardizer()
    envL_std = Standardizer()
    envR_std = Standardizer()

    for nwb_path in nwb_paths:
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            trials_df = nwb.trials.to_dataframe()
            subj_key = os.path.splitext(os.path.basename(str(nwb_path)))[0]

            subject_trial_dicts = []
            for _, tr_row in trials_df.iterrows():
                td = _load_data_from_trial(
                nwbfile=nwb,
                tr_row=tr_row,
                cfg=cfg,
                sum_subbands=sum_subbands,
            )
                td["subject"] = subj_key
                subject_trial_dicts.append(td)

            if merge_repetitions:
                subject_trial_dicts = _merge_repetition_trial_dicts(subject_trial_dicts)

            for td in subject_trial_dicts:
                eeg_std.partial_fit(td["eeg"])
                envL_std.partial_fit(td["envL"])
                envR_std.partial_fit(td["envR"])

    eeg_std.finalize()
    envL_std.finalize()
    envR_std.finalize()
    return eeg_std, envL_std, envR_std



def fit_per_subject_standardizers(nwb_paths, cfg, sum_subbands=True, merge_repetitions=False):
    """
    Fit one EEG, one envL, and one envR standardizer per subject.

    Returns:
      subject_stds: dict
        {
          subj_key: {
            "eeg_std": Standardizer(),
            "envL_std": Standardizer(),
            "envR_std": Standardizer(),
          },
          ...
        }
    """
    subject_stds = {}

    for nwb_path in nwb_paths:
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            trials_df = nwb.trials.to_dataframe()
            subj_key = os.path.splitext(os.path.basename(str(nwb_path)))[0]

            eeg_std = Standardizer()
            envL_std = Standardizer()
            envR_std = Standardizer()

            subject_trial_dicts = []
            for _, tr_row in trials_df.iterrows():
                td = _load_data_from_trial(
                nwbfile=nwb,
                tr_row=tr_row,
                cfg=cfg,
                sum_subbands=sum_subbands,
            )
                td["subject"] = subj_key
                subject_trial_dicts.append(td)

            if merge_repetitions:
                subject_trial_dicts = _merge_repetition_trial_dicts(subject_trial_dicts)

            for td in subject_trial_dicts:
                eeg_std.partial_fit(td["eeg"])
                envL_std.partial_fit(td["envL"])
                envR_std.partial_fit(td["envR"])

            eeg_std.finalize()
            envL_std.finalize()
            envR_std.finalize()

            subject_stds[subj_key] = {
                "eeg_std": eeg_std,
                "envL_std": envL_std,
                "envR_std": envR_std,
            }

    return subject_stds

# =============================================================================
# Dataset
# =============================================================================
class AADDataset(Dataset):
    """
    PyTorch Dataset for auditory attention decoding (AAD) from EEG.

    Returns:
      eeg_win: (win_len, C_eeg)
      env_win: (2, win_len, B)
      att:     scalar
      meta:    dict with subject, dataset, trial_uid, trial_index, start
    """

    def __init__(
            self,
            nwb_paths,
            cfg,
            sum_subbands=True,
            split="train",
            eeg_std=None,
            envL_std=None,
            envR_std=None,
            subject_stds=None,
            dataset_stds=None,
            merge_repetitions=False,
        ):
        self.cfg = cfg
        self.sum_subbands = sum_subbands
        self.split = split

        self.fs = int(cfg["preprocessing"]["target_fs"])

        self.eeg_std = eeg_std
        self.envL_std = envL_std
        self.envR_std = envR_std
        self.subject_stds = subject_stds
        self.dataset_stds = dataset_stds
        self.merge_repetitions = merge_repetitions
        self.norm_mode = str(cfg["data"].get("norm_mode", "none")).lower()

        dw_cfg = cfg["DeepLearning"]["data_windows"]

        if split == "val":
            split_cfg = dw_cfg["val"]
            window_len_s = float(split_cfg["window_len_s"])
            window_step_s = float(split_cfg["window_step_s"])
            min_gap_s = float(split_cfg.get("min_gap_val", window_step_s))
            selection_mode = str(split_cfg.get("selection_mode", "step")).lower()
            max_windows_per_trial = split_cfg.get("max_windows_per_trial", None)
        else:
            split_cfg = dw_cfg["data"]
            window_len_s = float(split_cfg["window_len_s"])
            window_step_s = float(split_cfg["window_step_s"])
            min_gap_s = float(split_cfg.get("min_gap_train", window_step_s))
            selection_mode = str(split_cfg.get("selection_mode", "step")).lower()
            max_windows_per_trial = split_cfg.get("max_windows_per_trial", None)

        self.win_len = int(window_len_s * self.fs)
        self.win_step = int(window_step_s * self.fs)
        self.min_gap = int(min_gap_s * self.fs)
        self.selection_mode = selection_mode
        self.max_windows_per_trial = None if max_windows_per_trial is None else int(max_windows_per_trial)

        self.trials = []
        self.index = []
        self.sample_dataset_keys = []
        self.sample_subject_keys = []

        for nwb_path in nwb_paths:
            with NWBHDF5IO(nwb_path, "r") as io:
                nwb = io.read()
                trials_df = nwb.trials.to_dataframe()
                subj_key = os.path.splitext(os.path.basename(str(nwb_path)))[0]

                subject_trial_dicts = []
                for _, tr_row in trials_df.iterrows():
                    td = _load_data_from_trial(
                        nwbfile=nwb,
                        tr_row=tr_row,
                        cfg=self.cfg,
                        sum_subbands=self.sum_subbands,
                    )
                    td["subject"] = subj_key
                    subject_trial_dicts.append(td)

                if self.merge_repetitions:
                    subject_trial_dicts = _merge_repetition_trial_dicts(subject_trial_dicts)

                for td in subject_trial_dicts:
                    eeg = td["eeg"]
                    envL = td["envL"]
                    envR = td["envR"]
                    att = td["att"]
                    ds_key = td["dataset"]

                    trial_eeg_std = self.eeg_std
                    trial_envL_std = self.envL_std
                    trial_envR_std = self.envR_std

                    if self.norm_mode == "global_per_dataset":
                        if self.dataset_stds is None:
                            raise ValueError("norm_mode='global_per_dataset' but dataset_stds is None")

                        if ds_key not in self.dataset_stds:
                            raise KeyError(f"No dataset-specific standardizers found for dataset '{ds_key}'")

                        trial_eeg_std = self.dataset_stds[ds_key]["eeg_std"]
                        trial_envL_std = self.dataset_stds[ds_key]["envL_std"]
                        trial_envR_std = self.dataset_stds[ds_key]["envR_std"]

                    if self.norm_mode == "per_subject":
                        if self.subject_stds is None:
                            raise ValueError("norm_mode='per_subject' but subject_stds is None")

                        if subj_key not in self.subject_stds:
                            raise KeyError(f"No per-subject standardizers found for subject '{subj_key}'")

                        trial_eeg_std = self.subject_stds[subj_key]["eeg_std"]
                        trial_envL_std = self.subject_stds[subj_key]["envL_std"]
                        trial_envR_std = self.subject_stds[subj_key]["envR_std"]

                    elif self.norm_mode == "per_subject_train_global_val":
                        if self.split == "train":
                            if self.subject_stds is None:
                                raise ValueError(
                                    "norm_mode='per_subject_train_global_val' requires subject_stds for train split"
                                )

                            if subj_key not in self.subject_stds:
                                raise KeyError(f"No per-subject standardizers found for training subject '{subj_key}'")

                            trial_eeg_std = self.subject_stds[subj_key]["eeg_std"]
                            trial_envL_std = self.subject_stds[subj_key]["envL_std"]
                            trial_envR_std = self.subject_stds[subj_key]["envR_std"]

                        elif self.split == "val":
                            if self.eeg_std is None or self.envL_std is None or self.envR_std is None:
                                raise ValueError(
                                    "norm_mode='per_subject_train_global_val' requires global eeg_std/envL_std/envR_std for val split"
                                )

                            trial_eeg_std = self.eeg_std
                            trial_envL_std = self.envL_std
                            trial_envR_std = self.envR_std

                        else:
                            raise ValueError(
                                f"Unsupported split='{self.split}' for norm_mode='per_subject_train_global_val'"
                            )
                    
                    
                    eeg, envL, envR = _normalize_trial(
                        eeg=eeg,
                        envL=envL,
                        envR=envR,
                        norm_mode=self.norm_mode,
                        eeg_std=trial_eeg_std,
                        envL_std=trial_envL_std,
                        envR_std=trial_envR_std,
                    )

                    envelopes = np.stack([envL, envR], axis=0)  # (2, T, B)
                    T = len(eeg)

                    trial_uid = f"{subj_key}::{td['trial_index']}"

                    tdict = {
                        "eeg": eeg,
                        "env": envelopes,
                        "att": int(att),
                        "T": int(T),
                        "dataset": ds_key,
                        "subject": subj_key,
                        "trial_index": td["trial_index"],
                        "trial_uid": trial_uid,
                    }

                    trial_idx = len(self.trials)
                    self.trials.append(tdict)

                    starts = _compute_window_starts(
                        T=T,
                        win_len=self.win_len,
                        win_step=self.win_step,
                        min_gap=self.min_gap,
                        selection_mode=self.selection_mode,
                        max_windows_per_trial=self.max_windows_per_trial,
                    )

                    for s in starts:
                        self.index.append((trial_idx, int(s)))
                        self.sample_dataset_keys.append(ds_key)
                        self.sample_subject_keys.append(subj_key)

        print(f"{split} dataset: {len(self.index)} windows")
        print(f"{split} unique dataset keys: {sorted(set(self.sample_dataset_keys))[:10]}")
        print(f"{split} unique subjects: {sorted(set(self.sample_subject_keys))[:10]}")
        print(f"{split} win_len={self.win_len}, win_step={self.win_step}, min_gap={self.min_gap}")
        print(f"{split} selection_mode={self.selection_mode}, max_windows_per_trial={self.max_windows_per_trial}")
        print(f"{split} norm_mode={self.norm_mode}")
        print(f"{split} merge_repetitions={self.merge_repetitions}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        trial_idx, start = self.index[idx]
        tinfo = self.trials[trial_idx]

        end = start + self.win_len

        eeg_win = tinfo["eeg"][start:end]       # (T, C)
        env_win = tinfo["env"][:, start:end, :] # (2, T, B)
        att = tinfo["att"]

        meta = {
            "subject": tinfo["subject"],
            "dataset": tinfo["dataset"],
            "trial_uid": tinfo["trial_uid"],
            "trial_index": str(tinfo["trial_index"]),
            "start": int(start),
        }

        return (
            torch.tensor(eeg_win, dtype=torch.float32),
            torch.tensor(env_win, dtype=torch.float32),
            torch.tensor(att, dtype=torch.long),
            meta,
        )


# =============================================================================
# Subject/path helpers
# =============================================================================
def get_subject_list(cfg):
    subj_cfg = cfg["subjects"]
    key = cfg["use_subjects"]

    if isinstance(subj_cfg, dict):
        if key not in subj_cfg:
            raise KeyError(
                f"use_subjects='{key}' not found in cfg['subjects'] keys={list(subj_cfg.keys())}"
            )
        return list(subj_cfg[key])

    return list(subj_cfg)


def subject_to_paths(subjects):
    """
    Input: ["S1_DAS", "S12_DTU", ...]
    Output: NWB paths
    """
    all_paths = []
    for s in subjects:
        try:
            subj_id, ds_key = s.rsplit("_", 1)
        except ValueError:
            raise ValueError(
                f"Subject '{s}' must look like '<SUBJ>_<DATASET>' e.g. S1_DAS"
            )

        ds_key = ds_key.upper()
        if ds_key not in ("DAS", "DTU"):
            raise ValueError(f"Unknown dataset key '{ds_key}' in subject '{s}'")

        p = paths.subject_eegPP(subj_id, ds_key)
        all_paths.append(str(p))
    return all_paths


# =============================================================================
# Sampler helpers
# =============================================================================
def _get_sample_subject_keys(ds):
    for attr in ("sample_subject_keys", "sample_subject_ids", "sample_subjects"):
        if hasattr(ds, attr):
            return list(getattr(ds, attr))

    raise AttributeError(
        "AADDataset must expose per-sample subject IDs for subject-weighted sampling. "
        "Expected e.g. ds.sample_subject_keys with len == len(dataset)."
    )


def build_weighted_sampler_from_subject(ds):
    keys = _get_sample_subject_keys(ds)

    unique = sorted(set(keys))
    counts = {k: 0 for k in unique}
    for k in keys:
        counts[k] += 1

    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)
    return sampler, counts


def build_weighted_sampler_from_dataset(ds):
    if not hasattr(ds, "sample_dataset_keys"):
        raise AttributeError("AADDataset must expose sample_dataset_keys for dataset balancing")

    keys = list(ds.sample_dataset_keys)

    unique = sorted(set(keys))
    counts = {k: 0 for k in unique}
    for k in keys:
        counts[k] += 1

    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)
    return sampler, counts


def build_weighted_sampler_combined_dataset_subject(ds):
    if not hasattr(ds, "sample_dataset_keys"):
        raise AttributeError("AADDataset must expose sample_dataset_keys")

    dkeys = list(ds.sample_dataset_keys)
    skeys = _get_sample_subject_keys(ds)

    d_unique = sorted(set(dkeys))
    d_counts = {k: 0 for k in d_unique}
    for k in dkeys:
        d_counts[k] += 1

    s_unique = sorted(set(skeys))
    s_counts = {k: 0 for k in s_unique}
    for k in skeys:
        s_counts[k] += 1

    weights = torch.tensor(
        [1.0 / (d_counts[dk] * s_counts[sk]) for dk, sk in zip(dkeys, skeys)],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)

    return sampler, {"dataset": d_counts, "subject": s_counts}


# =============================================================================
# Dataloaders
# =============================================================================
def build_dataloaders(cfg, dl_cfg, train_subjects, val_subjects, normalization_bundle=None):
    train_paths = subject_to_paths(train_subjects)
    val_paths = subject_to_paths(val_subjects)

    norm_mode = str(cfg["data"].get("norm_mode", "none")).lower()
    if norm_mode not in {
        "none",
        "per_trial",
        "per_subject",
        "global",
        "per_subject_train_global_val",
        "global_per_dataset",
    }:
        raise ValueError(
            f"Unknown cfg['data']['norm_mode']={norm_mode}. "
            f"Use one of: 'none', 'per_trial', 'per_subject', 'global', "
            f"'per_subject_train_global_val', 'global_per_dataset'."
        )

    sum_subbands = bool(cfg["backward_model"].get("sum_subbands", True))
    print(f"[TARGET] sum_subbands={sum_subbands}")

    eeg_std, envL_std, envR_std = None, None, None
    train_subject_stds, val_subject_stds = None, None
    dataset_stds = None

    # ---------------------------------------------------------
    # Reuse provided normalization bundle if given
    # ---------------------------------------------------------
    if normalization_bundle is not None:
        print("[NORMALIZATION] Reusing provided normalization bundle")
        eeg_std = normalization_bundle.get("eeg_std", None)
        envL_std = normalization_bundle.get("envL_std", None)
        envR_std = normalization_bundle.get("envR_std", None)
        train_subject_stds = normalization_bundle.get("subject_stds", None)
        val_subject_stds = normalization_bundle.get("subject_stds", None)
        dataset_stds = normalization_bundle.get("dataset_stds", None)

    else:
        if norm_mode == "global":
            print("[NORMALIZATION] Fitting GLOBAL standardizers on TRAIN subjects only")
            eeg_std, envL_std, envR_std = fit_global_standardizers(
                nwb_paths=train_paths,
                cfg=cfg,
                sum_subbands=sum_subbands,
                merge_repetitions=False,
            )

        elif norm_mode == "per_subject":
            print("[NORMALIZATION] Fitting PER-SUBJECT standardizers on train subjects")
            train_subject_stds = fit_per_subject_standardizers(
                nwb_paths=train_paths,
                cfg=cfg,
                sum_subbands=sum_subbands,
                merge_repetitions=False,
            )

            print("[NORMALIZATION] Fitting PER-SUBJECT standardizers on val subjects")
            val_subject_stds = fit_per_subject_standardizers(
                nwb_paths=val_paths,
                cfg=cfg,
                sum_subbands=sum_subbands,
                merge_repetitions=False,
            )

        elif norm_mode == "per_subject_train_global_val":
            print("[NORMALIZATION] Fitting PER-SUBJECT standardizers on TRAIN subjects")
            train_subject_stds = fit_per_subject_standardizers(
                nwb_paths=train_paths,
                cfg=cfg,
                sum_subbands=sum_subbands,
                merge_repetitions=False,
            )

            print("[NORMALIZATION] Fitting GLOBAL standardizers on TRAIN subjects only (for val/test)")
            eeg_std, envL_std, envR_std = fit_global_standardizers(
                nwb_paths=train_paths,
                cfg=cfg,
                sum_subbands=sum_subbands,
                merge_repetitions=False,
            )

        elif norm_mode == "global_per_dataset":
            print("[NORMALIZATION] Fitting GLOBAL standardizers per dataset on TRAIN subjects only")
            dataset_stds = fit_global_standardizers_per_dataset(
                nwb_paths=train_paths,
                cfg=cfg,
                sum_subbands=sum_subbands,
                merge_repetitions=False,
            )
            print(f"[NORMALIZATION] Dataset-specific scalers fitted for: {sorted(dataset_stds.keys())}")

        else:
            print(f"[NORMALIZATION] Using norm_mode='{norm_mode}' (no fitted dataset-level standardizers)")

    train_merge = False
    val_merge = False

    train_ds = AADDataset(
        nwb_paths=train_paths,
        cfg=cfg,
        sum_subbands=sum_subbands,
        split="train",
        eeg_std=eeg_std,
        envL_std=envL_std,
        envR_std=envR_std,
        subject_stds=train_subject_stds if norm_mode in {"per_subject", "per_subject_train_global_val"} else None,
        dataset_stds=dataset_stds if norm_mode == "global_per_dataset" else None,
        merge_repetitions=train_merge,
    )

    val_ds = AADDataset(
        nwb_paths=val_paths,
        cfg=cfg,
        sum_subbands=sum_subbands,
        split="val",
        eeg_std=eeg_std,
        envL_std=envL_std,
        envR_std=envR_std,
        subject_stds=val_subject_stds if norm_mode == "per_subject" else None,
        dataset_stds=dataset_stds if norm_mode == "global_per_dataset" else None,
        merge_repetitions=val_merge,
    )

    overfit_cfg = dl_cfg["overfit_tiny"]
    if overfit_cfg["enable"]:
        n_windows = int(overfit_cfg["n_windows"])

        base_ds = AADDataset(
            nwb_paths=train_paths,
            cfg=cfg,
            sum_subbands=sum_subbands,
            split="train",
            eeg_std=eeg_std,
            envL_std=envL_std,
            envR_std=envR_std,
            subject_stds=train_subject_stds if norm_mode in {"per_subject", "per_subject_train_global_val"} else None,
            dataset_stds=dataset_stds if norm_mode == "global_per_dataset" else None,
            merge_repetitions=False,
        )

        n_take = min(n_windows, len(base_ds))
        idxs = list(range(n_take))

        train_ds = torch.utils.data.Subset(base_ds, idxs)
        val_ds = torch.utils.data.Subset(base_ds, idxs)

        print(f"[OVERFIT TEST] using EXACT same subset for train and val")
        print(f"[OVERFIT TEST] subject(s): {train_subjects}")
        print(f"[OVERFIT TEST] n_windows: {n_take}")

        sampler = None
        batch_size = min(int(dl_cfg["train"]["batch_size"]), n_take)
        num_epochs = int(overfit_cfg["max_epochs"])

    else:
        if dl_cfg["quick_test"]["enable"]:
            print("[QUICK TEST] enabled")
            num_epochs = int(dl_cfg["quick_test"]["max_epochs"])
        else:
            num_epochs = int(dl_cfg["train"]["num_epochs"])

        batch_size = int(dl_cfg["train"]["batch_size"])
        sampler_mode = str(dl_cfg["train"]["sampler_mode"]).lower()

        if sampler_mode == "none":
            sampler = None

        elif sampler_mode == "dataset":
            sampler, counts = build_weighted_sampler_from_dataset(train_ds)
            print(f"Train sample counts by dataset: {counts}")

        elif sampler_mode == "subject":
            sampler, counts = build_weighted_sampler_from_subject(train_ds)
            print(f"Train sample counts by subject: {counts}")

        elif sampler_mode in ("both", "combined"):
            sampler, counts = build_weighted_sampler_combined_dataset_subject(train_ds)
            print(f"Train sample counts by dataset: {counts['dataset']}")
            print(f"Train sample counts by subject: {counts['subject']}")

        else:
            raise ValueError(
                f"Unknown sampler_mode='{sampler_mode}'. Use none/dataset/subject/both."
            )

    num_workers = int(dl_cfg["train"]["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    eeg_sample, stim_sample, _, _ = train_ds[0]
    C_eeg = eeg_sample.shape[-1]
    C_stim = stim_sample.shape[-1]
    K = 2

    normalization_bundle_out = {
        "eeg_std": eeg_std,
        "envL_std": envL_std,
        "envR_std": envR_std,
        "subject_stds": train_subject_stds if train_subject_stds is not None else val_subject_stds,
        "dataset_stds": dataset_stds,
    }

    return (
        train_loader,
        train_eval_loader,
        val_loader,
        (C_eeg, C_stim, K),
        num_epochs,
        normalization_bundle_out,
    )