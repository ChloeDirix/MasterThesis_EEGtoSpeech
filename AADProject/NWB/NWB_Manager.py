# NWB/NWB_Manager.py
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from datetime import datetime
from pynwb.ecephys import ElectricalSeries

class NWBManager:
    def save_subject(self, subject, out_path: str):
        """
        Save one subject (with trials) into a structured NWB file.
        Includes raw EEG, preprocessed EEG (if available), stimuli, and metadata.
        """
        nwb = NWBFile(
            session_description="Selective auditory attention task",
            identifier=subject.subject_id,
            session_start_time=datetime.now(),
        )

        eeg_proc_module = ProcessingModule(
            name="eeg_preprocessed",
            description="Filtered, rereferenced, and resampled EEG data."
        )
        nwb.add_processing_module(eeg_proc_module)

        #define columns
        custom_columns = ["attended_ear", "stim_L_name", "stim_R_name"]
        for col in custom_columns:
            nwb.add_trial_column(
                name=col,
                description=f"Custom metadata field: {col}"
            )

        # --- Add trials ---
        for trial in subject.trials:

            # 1️⃣ Raw EEG
            raw_series = TimeSeries(
                name=f"trial_{trial.index}_EEG_raw",
                data=trial.eeg_raw,
                unit="microvolts",
                starting_time=0.0,
                rate=float(trial.fs_eeg),
                description="Raw EEG signal"
            )
            nwb.add_acquisition(raw_series)

            # 2️⃣ Preprocessed EEG (if available)
            if hasattr(trial, "eeg_PP") and trial.eeg_PP is not None:
                preproc_series = TimeSeries(
                    name=f"trial_{trial.index}_EEG_preprocessed",
                    data=trial.eeg_PP,
                    unit="microvolts",
                    starting_time=0.0,
                    rate=float(trial.fs_eeg),
                    description="Preprocessed EEG (bandpass, reref, resampled)"
                )
                eeg_proc_module.add_data_interface(preproc_series)

            # # 3️⃣ Stimuli
            # if hasattr(trial, "stimuli") and trial.stimuli:
            #     for side, stim in trial.stimuli.items():
            #         stim_series = TimeSeries(
            #             name=f"trial_{trial.index}_stim_{side}",
            #             data=stim,
            #             unit="a.u.",
            #             starting_time=0.0,
            #             rate=trial.fs_stimuli.get(side, trial.fs_eeg),
            #             description=f"Stimulus envelope ({side})"
            #         )
            #         nwb.add_stimulus(stim_series)

            # clean_meta = {}
            # for k, v in trial.metadata.items():
            #     if isinstance(v, (list, dict, np.ndarray)):
            #         clean_meta[k] = str(v)
            #     else:
            #         clean_meta[k] = v


            # Add the trial
            nwb.add_trial(
                start_time=0.0,
                stop_time=len(trial.eeg_raw) / trial.fs_eeg,
                attended_ear=trial.metadata.get("attended_ear"),
                stim_L_name=trial.metadata.get("stim_names")[0],
                stim_R_name=trial.metadata.get("stim_names")[1],

            )

        # 5️⃣ Write to disk
        with NWBHDF5IO(out_path, "w") as io:
            io.write(nwb)
        print(f"✅ NWB file written for subject {subject.subject_id} -> {out_path}")
