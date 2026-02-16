from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np


# Represents one trial's EEG, stimuli, and metadata.
@dataclass
class Trial:

    index: int
    eeg_raw: Optional[np.ndarray] = None
    eeg_PP:Optional[np.ndarray]=None
    fs_eeg_original: Optional[float] = None
    fs_eeg: Optional[float] = None
    channels: Optional[list] = None
    stimuli: Dict[str, np.ndarray] = field(default_factory=dict)
    fs_stimuli_original: Dict[str, float] = field(default_factory=dict)
    fs_stimuli: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if self.eeg_raw is None or self.eeg_raw.ndim != 2:
            print(f"Trial {self.index}: invalid EEG data")
            return False
        if "attended_ear" not in self.metadata:
            print(f"Trial {self.index}: missing attended_ear metadata")
            return False
        return True


@dataclass
class Subject:
    subject_id: str
    trials: list[Trial] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
