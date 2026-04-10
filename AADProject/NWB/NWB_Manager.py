# NWB/NWB_Manager.py
from datetime import datetime, timezone
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.ecephys import ElectricalSeries


class NWBManager:
    def __init__(self, debug: bool = True, keep_only_dual_speaker: bool = True):
        self.debug = debug
        self.keep_only_dual_speaker = keep_only_dual_speaker

    # ----------------------------
    # Helpers for NWB-safe values
    # ----------------------------
    @staticmethod
    def _as_int(val, default=-1):
        """Convert val to int if possible, else default (never None)."""
        if val is None:
            return default
        try:
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            # handle strings / floats
            return int(float(val))
        except Exception:
            return default

    @staticmethod
    def _as_str(val, default=""):
        """Convert val to str, else default (never None)."""
        if val is None:
            return default
        try:
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            s = str(val)
            return s if s is not None else default
        except Exception:
            return default

    def save_subject(self, subject, out_path: str):
        n_trials = len(getattr(subject, "trials", []) or [])
        self._dbg(f"[NWB] Saving subject={getattr(subject, 'subject_id', None)} | trials={n_trials} | out={out_path}")

        if n_trials == 0:
            raise ValueError(f"Subject {getattr(subject, 'subject_id', None)} has 0 trials. Not writing NWB.")

        nwb = NWBFile(
            session_description="Auditory attention decoding",
            identifier=str(subject.subject_id),
            session_start_time=datetime.now(timezone.utc),
        )

        # Electrodes
        nwb.add_electrode_column(name="label", description="EEG channel name")

        device = nwb.create_device(name="EEG_cap")
        eeg_group = nwb.create_electrode_group(
            name="EEG_group",
            description="EEG electrodes",
            location="scalp",
            device=device
        )

        # ---- Ensure DAS trials have n_speakers=2 if missing ----
        for t in subject.trials:
            md = t.metadata or {}
            if md.get("dataset") in ("DAS", "classic") and md.get("n_speakers") is None:
                md["n_speakers"] = 2
            t.metadata = md

        
        if self.keep_only_dual_speaker:
            before = len(subject.trials)
            subject.trials = [
                t for t in subject.trials
                if self._as_int((t.metadata or {}).get("n_speakers"), default=-1) == 2
            ]
            self._dbg(f"[filter] kept {len(subject.trials)}/{before} dual-speaker trials")

            if not subject.trials:
                raise ValueError("[NWB] After filtering, no trials remain. Check n_speakers labeling.")

        channel_names = subject.trials[0].channels
        for i, ch_name in enumerate(channel_names):
            nwb.add_electrode(
                id=i,
                x=np.nan, y=np.nan, z=np.nan,
                imp=np.nan,
                location="scalp",
                group=eeg_group,
                label=str(ch_name)
            )

        all_electrodes = nwb.create_electrode_table_region(
            region=list(range(len(channel_names))),
            description="All EEG electrodes"
        )

        # Preprocessed EEG module
        eeg_proc_module = ProcessingModule(
            name="eeg_preprocessed",
            description="Filtered, rereferenced, and resampled EEG data."
        )
        nwb.add_processing_module(eeg_proc_module)

        # Trial columns
        trial_cols = [
            "trial_index",
            "dataset",
            "attended_ear",
            "stim_L_name",
            "stim_R_name",
            "n_speakers",
            "attend_mf",
            "attend_lr",
            "stim_male_name",
            "stim_female_name",
            "trigger",
            "acoustic_condition",
            "start_sample",
            "stop_sample",
        ]
        for col in trial_cols:
            try:
                nwb.add_trial_column(name=col, description=f"Trial metadata: {col}")
            except (ValueError, KeyError) as e:
                if self.debug:
                    print(f"[NWB] trial column '{col}' exists (skipping). ({type(e).__name__}: {e})")

        # Write trials
        for trial in subject.trials:
            # Raw EEG acquisition
            raw_series = TimeSeries(
                name=f"trial_{trial.index}_EEG_raw",
                data=trial.eeg_raw,
                unit="microvolts",
                starting_time=0.0,
                rate=float(trial.fs_eeg_original),
                description="Raw EEG signal"
            )
            nwb.add_acquisition(raw_series)

            # Preprocessed EEG processing module
            if getattr(trial, "eeg_PP", None) is not None:
                preproc_series = ElectricalSeries(
                    name=f"trial_{trial.index}_EEG_preprocessed",
                    data=trial.eeg_PP,
                    electrodes=all_electrodes,
                    starting_time=0.0,
                    rate=float(trial.fs_eeg),
                    description="Preprocessed EEG"
                )
                eeg_proc_module.add_data_interface(preproc_series)

            md = trial.metadata or {}

            # NWB trial times must be in seconds
            md = trial.metadata or {}
            fs0 = float(trial.fs_eeg_original)

            start_samp = md.get("start_sample", None)
            stop_samp  = md.get("stop_sample", None)

            if start_samp is not None and stop_samp is not None and int(stop_samp) > int(start_samp):
                start_time = float(int(start_samp)) / fs0
                stop_time  = float(int(stop_samp)) / fs0
            else:
                # fallback: per-trial local time
                start_time = 0.0
                stop_time = float(len(trial.eeg_raw)) / fs0


            # Never pass None: convert to safe defaults
            nwb.add_trial(
                trial_index=self._as_int(trial.index, default=-1),

                start_time=start_time,
                stop_time=stop_time,

                dataset=self._as_str(md.get("dataset"), default="unknown"),
                attended_ear=self._as_str(md.get("attended_ear"), default="unknown"),

                stim_L_name=self._as_str(md.get("stim_L_name"), default=""),
                stim_R_name=self._as_str(md.get("stim_R_name"), default=""),

                n_speakers=self._as_int(md.get("n_speakers"), default=-1),
                attend_mf=self._as_int(md.get("attend_mf"), default=-1),
                attend_lr=self._as_int(md.get("attend_lr"), default=-1),

                stim_male_name=self._as_str(md.get("stim_male_name"), default=""),
                stim_female_name=self._as_str(md.get("stim_female_name"), default=""),

                trigger=self._as_int(md.get("trigger"), default=-1),
                acoustic_condition=self._as_int(md.get("acoustic_condition"), default=-1),

                start_sample=self._as_int(md.get("start_sample"), default=-1),
                stop_sample=self._as_int(md.get("stop_sample"), default=-1),
            )

        with NWBHDF5IO(out_path, "w") as io:
            io.write(nwb)

        self._dbg("[NWB] Write complete.")

    def _dbg(self, msg: str):
        if self.debug:
            print(msg)
