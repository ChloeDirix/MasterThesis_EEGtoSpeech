# Loaders/matlab_loader.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat

from Loaders.DataModels import Subject, Trial


class MatlabSubjectLoader:
    """
    Supports:
      - DAS trials format (RawData/FileHeader)
      - DTU continuous format (data.eeg + data.event.eeg) + sidecar *_plain.mat

    Normalization goals (both datasets):
      - Trial.index is the only trial identifier (no metadata["trial_id"])
      - metadata always includes at least:
          dataset, attended_ear, stim_L_name, stim_R_name, n_speakers
      - DTU additionally includes:
          attend_mf/attend_lr, stim_male_name/stim_female_name, trigger,
          acoustic_condition, start_sample/stop_sample
    """

    def __init__(self, mat_path: str, subject_id: str, debug: bool = True):
        self.mat_path = str(mat_path)
        self.subject_id = subject_id
        self.debug = debug

        self.mat = loadmat(self.mat_path, squeeze_me=True, struct_as_record=False)
        self._dbg(f"[init] {subject_id} loading from: {self.mat_path}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def load(self) -> Subject:
        items = self.find_trials(self.mat)

        self._dbg(f"[load] find_trials() returned {len(items)} item(s)")
        if not items:
            self._dbg("[load] ERROR: No items returned by find_trials()")
            return Subject(self.subject_id, [])

        # DTU/data-struct format
        if self._is_data_struct(items[0]):
            self._dbg("[load] Detected DTU DATA-STRUCT format. Parsing DTU...")
            trials_all = self._parse_dtu_trials(items[0])
            self._dbg(f"[load] Parsed {len(trials_all)} trial(s) (pre-validate)")

            trials_ok = []
            for t in trials_all:
                ok = t.validate()
                self._dbg(
                    f"[load] Trial {t.index} validate={ok} | eeg={getattr(t.eeg_raw,'shape',None)} "
                    f"| fs={t.fs_eeg_original} | channels={len(t.channels) if t.channels else None} "
                    f"| attended_ear={t.metadata.get('attended_ear') if t.metadata else None} "
                    f"| stim_L={t.metadata.get('stim_L_name') if t.metadata else None} "
                    f"| stim_R={t.metadata.get('stim_R_name') if t.metadata else None}"
                )
                if ok:
                    trials_ok.append(t)

            self._dbg(f"[load] Kept {len(trials_ok)} trial(s) (post-validate)")
            return Subject(self.subject_id, trials_ok)

        # DAS format
        self._dbg("[load] Detected DAS format (RawData/FileHeader). Parsing...")
        trials_ok = []
        for i, raw_trial in enumerate(items, start=1):
            try:
                trial = self._parse_trial(raw_trial, i)
            except Exception as e:
                self._dbg(f"[load] Trial {i} parse FAILED: {type(e).__name__}: {e}")
                continue

            ok = trial.validate()
            self._dbg(f"[load] Trial {i} validate={ok}")
            if ok:
                trials_ok.append(trial)

        self._dbg(f"[load] Kept {len(trials_ok)} DAS trial(s)")
        return Subject(self.subject_id, trials_ok)

    # ------------------------------------------------------------------
    # Trial discovery
    # ------------------------------------------------------------------
    def find_trials(self, mat) -> list:
        self._dbg(f"[find_trials] MAT keys: {list(mat.keys())}")

        if "trials" in mat:
            cand = mat["trials"]
            cand_key = "trials"
        elif "data" in mat:
            cand = mat["data"]
            cand_key = "data"
        else:
            cand, cand_key = None, None
            for k, v in mat.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray):
                    if cand is None or v.size > getattr(cand, "size", 0):
                        cand, cand_key = v, k
            if cand is None:
                raise ValueError("No candidate trial array found in .mat")

        self._dbg(f"[find_trials] Candidate key={cand_key} | type={type(cand)} | shape={getattr(cand,'shape',None)}")

        # DTU container
        if self._is_data_struct(cand):
            self._dbg("[find_trials] Candidate is DTU DATA-STRUCT (has eeg/fsample). Returning [cand].")
            if hasattr(cand, "_fieldnames"):
                self._dbg(f"[find_trials] data fields: {cand._fieldnames}")
            return [cand]

        # DAS array of trial structs
        arr = np.array(cand, dtype=object).flatten()
        self._dbg(f"[find_trials] Flattened candidate length={len(arr)}")

        trials = []
        for el in arr:
            if isinstance(el, (int, float, np.integer, np.floating, str)):
                continue
            if self._has_field(el, "RawData") or self._has_field(el, "FileHeader"):
                trials.append(el)

        self._dbg(f"[find_trials] DAS trial candidates found: {len(trials)}")
        return trials

    # ------------------------------------------------------------------
    # DAS trials parser
    # ------------------------------------------------------------------
    def _parse_trial(self, raw_trial, idx: int) -> Trial:
        rawdata = self.get_field(raw_trial, "RawData")
        filehdr = self.get_field(raw_trial, "FileHeader")

        eeg_raw = self.get_field(rawdata, "EegData")
        eeg_arr = np.squeeze(np.array(eeg_raw))

        channels = self.get_field(rawdata, "Channels")
        channel_names = [str(ch) for ch in np.atleast_1d(channels)]
        n_ch = len(channel_names)

        eeg_data = self.orient_eeg(eeg_arr, n_ch)
        fs_eeg_original = float(self.get_field(filehdr, "SampleRate"))

        stim_names_orig = [str(s) for s in np.atleast_1d(self.get_field(raw_trial, "stimuli"))]
        stim_names = [
            stim_names_orig[0].replace("hrtf", "dry"),
            stim_names_orig[1].replace("hrtf", "dry"),
        ]

        attended_ear = self.get_field(raw_trial, "attended_ear")

        # Normalized metadata schema
        meta = {
            "dataset": "DAS",
            "attended_ear": str(attended_ear),
            "stim_L_name": stim_names[0],
            "stim_R_name": stim_names[1],

            # Keep these for schema compatibility; may be unknown in DAS
            "attend_mf": None,
            "attend_lr": None,
            "stim_male_name": None,
            "stim_female_name": None,
            "trigger": None,
            "acoustic_condition": None,

            # assume dual-talker for DAS (adjust if your DAS can be single-speaker)
            "n_speakers": 2,

            # provenance for DTU only, but keep keys consistent
            "start_sample": None,
            "stop_sample": None,
        }

        return Trial(
            index=idx,
            eeg_raw=eeg_data,
            fs_eeg_original=fs_eeg_original,
            fs_eeg=fs_eeg_original,
            channels=channel_names,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # DTU parsing (data struct)
    # ------------------------------------------------------------------
    def _is_data_struct(self, obj) -> bool:
        return self._has_field(obj, "eeg") and self._has_field(obj, "fsample")

    def _parse_dtu_trials(self, data_obj) -> list[Trial]:
        """
        DTU: continuous EEG in data.eeg, events in data.event.eeg.sample/value.
        Trial metadata in expinfo (MATLAB table) -> loaded from *_plain.mat sidecar.

        This version matches the DTU Matlab preproc_script.m logic:
        - Raw event stream often contains 2 events per trial.
        - Matlab checks events(1:2:end) == expinfo.trigger
        - Trials are split at those (odd-indexed) event sample positions, in order.
        """
        eeg = np.array(self.get_field(data_obj, "eeg"))
        fs_val = self.to_scalar(self.get_field(data_obj, "fsample"))
        fs_eeg_original = float(fs_val)

        self._dbg(f"[DTU] data.eeg shape={eeg.shape}, fs={fs_eeg_original}")

        if eeg.ndim != 2:
            raise ValueError(f"[DTU] Expected eeg to be 2D (time x channels), got shape {eeg.shape}")

        n_ch = eeg.shape[1]
        channels = [f"ch{i+1}" for i in range(n_ch)]

        # --- events ---
        events = self.get_field(self.get_field(data_obj, "event"), "eeg")
        ev_sample = np.atleast_1d(np.array(self.get_field(events, "sample")).squeeze()).astype(np.int64)
        ev_value = np.atleast_1d(np.array(self.get_field(events, "value")).squeeze())
        ev_value = np.array([self.to_scalar(v) for v in ev_value], dtype=np.int64)

        self._dbg(f"[DTU] event samples={ev_sample.shape} values={ev_value.shape}")

        # --- expinfo from sidecar ---
        exp = self._load_dtu_plain_expinfo()
        trig = exp["trigger"].astype(np.int64)
        n_trials = len(trig)

        # -------------------------------------------------------
        # DTU event alignment (robust, mirrors Matlab behavior)
        # -------------------------------------------------------
        start_samples = None

        # Case A: raw-ish DTU events where odd events (Matlab 1:2:end) are triggers
        odd_ix = np.arange(0, len(ev_value), 2)  # Python 0,2,4,... == Matlab 1,3,5,...
        trig_stream = ev_value[odd_ix][:n_trials]
        samp_stream = ev_sample[odd_ix][:n_trials]

        if len(trig_stream) == n_trials and np.all(trig_stream == trig):
            start_samples = samp_stream.astype(np.int64)
            self._dbg("[DTU] Using odd-indexed event samples for trial starts (matched expinfo.trigger).")
        else:
            # Case B: already-preprocessed DTU output where event values are only attend_mf {1,2}
            ev_unique = set(np.unique(ev_value).tolist())
            if ev_unique.issubset({1, 2}) and len(ev_sample) >= n_trials:
                start_samples = ev_sample[:n_trials].astype(np.int64)
                self._dbg(
                    "[DTU] Event values are only {1,2}. Assuming preprocessed event stream; "
                    "using event samples in order as trial starts."
                )
            else:
                # Case C: fallback (value search). Works but can be wrong if triggers repeat.
                self._dbg(
                    "[DTU] WARNING: Could not match events(odd) to expinfo.trigger and events are not {1,2} only.\n"
                    "Falling back to trigger-value search; may create wrong/long trials if triggers repeat."
                )

                start_samples = np.full(n_trials, -1, dtype=np.int64)
                occ_counter = {}
                for i, tval in enumerate(trig):
                    tval = int(tval)
                    hits = np.where(ev_value == tval)[0]
                    if hits.size == 0:
                        self._dbg(f"[DTU] Trial {i+1}: trigger={tval} not found -> SKIP")
                        continue
                    k = occ_counter.get(tval, 0)
                    if k >= hits.size:
                        self._dbg(f"[DTU] Trial {i+1}: trigger={tval} occurrences={hits.size}, need {k+1} -> SKIP")
                        continue
                    start_samples[i] = int(ev_sample[hits[k]])
                    occ_counter[tval] = k + 1

        self._dbg(f"[DTU] aligned {np.sum(start_samples>=0)}/{n_trials} trials to event samples")

        # -------------------------------------------------------
        # Build trials by slicing continuous EEG between starts
        # -------------------------------------------------------
        trials: list[Trial] = []

        # next-valid-start lookup (handles -1 gaps if fallback was used)
        valid_idx = np.where(start_samples >= 0)[0]
        if valid_idx.size == 0:
            raise ValueError("[DTU] No valid start samples found. Check events vs expinfo.")

        for i in range(n_trials):
            start = int(start_samples[i])
            trigger_val = int(trig[i])

            if start < 0:
                # couldn't align this trial
                continue

            # stop = next valid start sample, else end of recording
            j = i + 1
            while j < n_trials and int(start_samples[j]) < 0:
                j += 1
            stop = int(start_samples[j]) if j < n_trials else eeg.shape[0]

            if stop <= start:
                stop = eeg.shape[0]

            eeg_seg = eeg[start:stop, :]
            eeg_seg = self.orient_eeg(eeg_seg, n_ch)

            # expinfo fields
            lr = int(exp["attend_lr"][i])      # 1=left, 2=right
            mf = int(exp["attend_mf"][i])      # 1=male, 2=female
            nsp = int(exp["n_speakers"][i])    # 1 or 2

            attended_ear = "left" if lr == 1 else "right" if lr == 2 else "unknown"
            stim_male = exp["wavfile_male"][i]
            stim_female = exp["wavfile_female"][i]

            # Derive which wav was on L/R
            stim_L = None
            stim_R = None
            if nsp == 1:
                # Single speaker: only attended talker present
                if mf == 1:
                    stim_L = stim_male if lr == 1 else None
                    stim_R = stim_male if lr == 2 else None
                elif mf == 2:
                    stim_L = stim_female if lr == 1 else None
                    stim_R = stim_female if lr == 2 else None
            else:
                # Two speakers present
                if mf == 1:
                    stim_L, stim_R = (stim_male, stim_female) if lr == 1 else (stim_female, stim_male)
                elif mf == 2:
                    stim_L, stim_R = (stim_female, stim_male) if lr == 1 else (stim_male, stim_female)

            meta = {
                "dataset": "DTU",
                "attended_ear": attended_ear,
                "stim_L_name": stim_L,
                "stim_R_name": stim_R,
                "n_speakers": nsp,

                "attend_mf": mf,
                "attend_lr": lr,
                "stim_male_name": stim_male,
                "stim_female_name": stim_female,
                "trigger": trigger_val,
                "acoustic_condition": int(exp["acoustic_condition"][i]),
                "start_sample": int(start),
                "stop_sample": int(stop),
            }

            trials.append(
                Trial(
                    index=i + 1,
                    eeg_raw=eeg_seg,
                    fs_eeg_original=fs_eeg_original,
                    fs_eeg=fs_eeg_original,
                    channels=channels,
                    metadata=meta,
                )
            )

        if not trials:
            raise ValueError("[DTU] No trials were built. Check triggers/events matching.")

        # Optional: duration stats to debug the "should be ~50s" expectation
        if self.debug:
            durs = [(t.metadata["stop_sample"] - t.metadata["start_sample"]) / fs_eeg_original for t in trials]
            self._dbg(
                f"[DTU] durations (sec): min={float(np.min(durs)):.2f}, "
                f"median={float(np.median(durs)):.2f}, max={float(np.max(durs)):.2f} | n={len(durs)}"
            )

        return trials


    def _load_dtu_plain_expinfo(self) -> dict:
        """
        Loads expinfo columns from <subject>_plain.mat sitting next to the original file.
        """
        p = Path(self.mat_path)
        plain = p.with_name(p.stem + "_plain.mat")

        if not plain.exists():
            raise FileNotFoundError(
                f"[DTU] expinfo is a MATLAB table (MCOS) and cannot be read by SciPy.\n"
                f"Create sidecar: {plain}\n"
                f"Use the MATLAB export script to write *_plain.mat with attend_mf/attend_lr/trigger/wavfiles."
            )

        m2 = loadmat(str(plain), squeeze_me=True, struct_as_record=False)
        required = [
            "attend_mf", "attend_lr", "trigger", "wavfile_male", "wavfile_female",
            "n_speakers", "acoustic_condition"
        ]
        missing = [k for k in required if k not in m2]
        if missing:
            raise KeyError(f"[DTU] {plain} missing keys: {missing}")

        def as_int_vec(x):
            x = np.atleast_1d(np.squeeze(np.array(x)))
            return np.array([self.to_scalar(v) for v in x], dtype=np.int64)

        def as_str_list(x):
            x = np.atleast_1d(np.squeeze(np.array(x, dtype=object)))
            out = []
            for v in x:
                v = self.to_scalar(v)
                if isinstance(v, (bytes, np.bytes_)):
                    v = v.decode(errors="ignore")
                s = str(v).strip()
                s = s.strip("[]").strip("'").strip('"')
                out.append(s)
            return out

        exp = {
            "attend_mf": as_int_vec(m2["attend_mf"]),
            "attend_lr": as_int_vec(m2["attend_lr"]),
            "trigger": as_int_vec(m2["trigger"]),
            "n_speakers": as_int_vec(m2["n_speakers"]),
            "acoustic_condition": as_int_vec(m2["acoustic_condition"]),
            "wavfile_male": as_str_list(m2["wavfile_male"]),
            "wavfile_female": as_str_list(m2["wavfile_female"]),
        }

        n = len(exp["trigger"])
        for k, v in exp.items():
            if len(v) != n:
                raise ValueError(f"[DTU] expinfo length mismatch: {k} has len {len(v)} vs trigger {n}")

        self._dbg(f"[DTU] Loaded expinfo from {plain} with {n} rows")
        return exp

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def orient_eeg(self, eeg_arr, n_ch: int | None):
        eeg_arr = np.asarray(eeg_arr)
        if eeg_arr.ndim != 2:
            raise ValueError(f"orient_eeg expects 2D array, got shape={eeg_arr.shape}")

        # want shape (time, channels)
        if n_ch is not None:
            if eeg_arr.shape[0] == n_ch:
                return eeg_arr.T
            if eeg_arr.shape[1] == n_ch:
                return eeg_arr
        return eeg_arr

    def get_field(self, obj, name):
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if isinstance(obj, np.ndarray) and obj.dtype.names:
            val = obj[name]
            if isinstance(val, np.ndarray) and val.size == 1:
                return val.item()
            return val
        if isinstance(obj, np.ndarray):
            for el in obj.flatten():
                try:
                    v = self.get_field(el, name)
                    if v is not None:
                        return v
                except Exception:
                    continue
        try:
            it = obj.item()
            return self.get_field(it, name)
        except Exception:
            pass
        raise KeyError(f"Field '{name}' not found in object of type {type(obj)}")

    def _has_field(self, obj, name: str) -> bool:
        try:
            self.get_field(obj, name)
            return True
        except Exception:
            return False

    def to_scalar(self, x):
        while isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()
        if hasattr(x, "_fieldnames"):
            for fname in ("value", "val", "fs", "fsample", "data", "x", "eeg"):
                if hasattr(x, fname):
                    return self.to_scalar(getattr(x, fname))
            return x
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return x

    def _dbg(self, msg: str):
        if self.debug:
            print(msg)
