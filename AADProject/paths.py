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
    #ROOT = Path(__file__).resolve().parent
    ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
    # --------------------
    # Base folders
    # --------------------
    DATA_DAS = ROOT / "Data_Das2019"
    DATA_DTU = ROOT / "Data_DTU"

    RAW_EEG_DAS = DATA_DAS / "EEGData"
    STIM_DAS = DATA_DAS / "stimuli"

    RAW_EEG_DTU = DATA_DTU / "EEG"
    STIM_DTU= DATA_DTU / "AUDIO"

    DATA_INPUT_MODEL = ROOT / "Data_InputModel"
    ENVELOPES = DATA_INPUT_MODEL / "Envelopes"
    EEG_PP = DATA_INPUT_MODEL / "EEG_PP"

    # Config file
    CONFIG_FILE = ROOT / "config.yaml"

    RESULTS_LIN = ROOT / "Results_Lin"
    RESULTS_DL = ROOT / "Results_DL"

    # --------------------
    # Config loader
    # --------------------
    @staticmethod
    def load_config():
        """Load config.yaml as a dictionary."""
        with paths.CONFIG_FILE.open("r") as f:
            return yaml.safe_load(f)


    # --------------------
    # Subject paths
    # --------------------
    @staticmethod
    def subject_eegPP(subject_id: str,dataset=""):
        """
        Returns path to preprocessed EEG file for a given subject.
        Example: Paths.subject_eeg("S3") → Data_InputModel/EEG_PP/S3.nwb
        """
        if dataset=="":
            return paths.EEG_PP / f"{subject_id}.nwb"
        
        return paths.EEG_PP / f"{subject_id}_{dataset}.nwb"

    @staticmethod
    def subject_eegPP_list(subject_ids: list, dataset=""):
        return [paths.subject_eegPP(s,dataset) for s in subject_ids]

    @staticmethod
    def subject_raw(subject_id: str,dataset):
        """Path to raw EEG file in certain dataset."""
        if dataset=="DTU":
            return paths.RAW_EEG_DTU / f"{subject_id}.mat"
        elif dataset=="DAS":
            return paths.RAW_EEG_DAS / f"{subject_id}.mat"

    @staticmethod
    def envelope(filename: str):
        """Path to envelope file of a subject."""
        return paths.ENVELOPES / filename

    @staticmethod
    def stimulus(filename: str, dataset: str):
        """Path to a stimulus audio file."""
        if dataset=="DTU":
            return paths.STIM_DTU / filename
        elif dataset=="DAS":    
            return paths.STIM_DAS / filename

    # --------------------
    # Result paths
    # --------------------
    @staticmethod
    def result_file_lin_SS(name: str):
        """Result file inside Results directory."""
        dir=paths.RESULTS_LIN / "SS"
        os.makedirs(dir,exist_ok=True)
        return dir / name

    def result_file_lin_SI(name: str):
        """Result file inside Results directory."""
        dir = paths.RESULTS_LIN / "SI"
        os.makedirs(dir,exist_ok=True)
        return dir / name

    @staticmethod
    def result_file_DL(cfg):
        if cfg["DeepLearning"]["model"]["architecture"]=="wo_transformer":
            return paths.RESULTS_DL
        else:
            return f"{paths.RESULTS_DL}_Transformer"


    @staticmethod
    def custom(*relative_parts):
        """
        Generic helper for manually specifying a path relative to project root.
        Example: Paths.custom("Data_InputModel", "S1.nwb")
        """
        return paths.ROOT.joinpath(*relative_parts)


    # --------------------
    # Automatic run-numbered directory
    # --------------------
    @staticmethod
    def get_next_run_dir(base_dir):
        """
        Creates incrementing run folders: run_0001, run_0002, ...
        Returns the full path to the created directory.
        """
        os.makedirs(base_dir, exist_ok=True)

        existing = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
        ]

        run_numbers = []
        for d in existing:
            try:
                run_numbers.append(int(d.replace("run_", "")))
            except ValueError:
                pass

        next_num = max(run_numbers) + 1 if run_numbers else 1
        run_name = f"run_{next_num:04d}"

        new_run_dir = os.path.join(base_dir, run_name)
        os.makedirs(new_run_dir, exist_ok=True)
        return new_run_dir

    @staticmethod
    def save_config_copy(cfg: dict, run_dir: str):
        """Save a copy of the config to run_dir/config_used.yaml."""
        save_path = os.path.join(run_dir, "config_used.yaml")
        with open(save_path, "w") as f:
            yaml.safe_dump(cfg, f)
        return save_path
