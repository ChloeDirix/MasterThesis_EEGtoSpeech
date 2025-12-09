# NWB/NWB_Manager.py
from datetime import datetime

import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
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

        # Add custom electrode column for channel names
        nwb.add_electrode_column(
            name="label",
            description="EEG channel name"
        )

        #save electrodes
        device = nwb.create_device(name="EEG_cap")

        # Create an electrode group (required by NWB)
        eeg_group = nwb.create_electrode_group(
            name="EEG_group",
            description="EEG electrodes",
            location="scalp",
            device=device
        )
        # Add electrodes to the group
        channel_names = subject.trials[0].channels  # assumed present
        for i, ch_name in enumerate(channel_names):
            nwb.add_electrode(
                id=i,
                x=np.nan, y=np.nan, z=np.nan,
                imp=np.nan,
                location="scalp",
                group=eeg_group,  # ✔ correct type
                label=ch_name
            )

        # region referencing all electrodes
        all_electrodes = nwb.create_electrode_table_region(
            region=list(range(len(channel_names))),  # ✔ valid
            description="All EEG electrodes"
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
                preproc_series = ElectricalSeries(
                    name=f"trial_{trial.index}_EEG_preprocessed",
                    data=trial.eeg_PP,
                    electrodes=all_electrodes,  # ← link electrodes!
                    starting_time=0.0,
                    rate=float(trial.fs_eeg),
                    description="Preprocessed EEG (bandpass, reref, resampled)"
                )
                eeg_proc_module.add_data_interface(preproc_series)



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
