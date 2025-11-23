import os
from pathlib import Path
import yaml

class paths:
    """
    Centralized path manager for the project.
    Computes project root dynamically and exposes clean helpers
    for all folders and files.
    """

    # Root of the project (folder containing config.yaml)
    ROOT = Path(__file__).resolve().parent

    # --------------------
    # Base folders
    # --------------------
    DATA_DAS = ROOT / "Data_Das2019"
    RAW_EEG_DAS = DATA_DAS / "EEGData"
    STIM_DAS = DATA_DAS / "stimuli"

    DATA_INPUT_MODEL = ROOT / "Data_InputModel"
    ENVELOPES = DATA_INPUT_MODEL / "Envelopes"
    EEG_PP = DATA_INPUT_MODEL / "EEG_PP"

    RESULTS = ROOT / "Results"

    # Config file
    CONFIG_FILE = ROOT / "config.yaml"

    # --------------------
    # Helper methods
    # --------------------

    @staticmethod
    def load_config():
        """Load config.yaml as a dictionary."""
        with paths.CONFIG_FILE.open("r") as f:
            return yaml.safe_load(f)



    @staticmethod
    def subject_eegPP(subject_id: str):
        """
        Returns path to preprocessed EEG file for a given subject.
        Example: Paths.subject_eeg("S3") → Data_InputModel/EEG_PP/S3.nwb
        """
        return paths.EEG_PP / f"{subject_id}.nwb"

    @staticmethod
    def subject_eegPP_list(subject_ids: list):
        subj_paths=[]
        for subject in subject_ids:
            subj_path = paths.subject_eegPP(subject)
            subj_paths.append(subj_path)
        return subj_paths
    @staticmethod
    def subject_raw(subject_id: str):
        """Path to raw EEG file."""
        return paths.RAW_EEG_DAS / f"{subject_id}.mat"

    @staticmethod
    def envelope(filename: str):
        """Path to envelope file of a subject."""
        return paths.ENVELOPES / filename

    @staticmethod
    def stimulus(filename: str):
        """Path to a stimulus audio file."""
        return paths.STIM_DAS / filename

    @staticmethod
    def result_file(name: str):
        """Result file inside Results directory."""
        return paths.RESULTS / name

    @staticmethod
    def custom(*relative_parts):
        """
        Generic helper for manually specifying a path relative to project root.
        Example: Paths.custom("Data_InputModel", "S1.nwb")
        """
        return paths.ROOT.joinpath(*relative_parts)

