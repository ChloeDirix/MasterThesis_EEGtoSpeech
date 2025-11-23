import torch
from torch.utils.data import Dataset


class PrecomputedAADDataset(Dataset):
    """
    Super fast dataset: loads all windows from a .pt file.
    No NWB. No NPZ. No preprocessing.
    """
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.eeg = data["eeg"]       # [N, T, C]
        self.stim = data["stim"]     # [N, 2, T, bands]
        self.att = data["att"]       # [N]

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx], self.stim[idx], self.att[idx]
