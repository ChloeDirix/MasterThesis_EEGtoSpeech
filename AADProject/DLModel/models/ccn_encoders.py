import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder_CNN(nn.Module):
    """
    Extremely lightweight EEG encoder to run fast on CPU.
    Input:  [B, T, C_eeg]
    Output: [B, D]
    """
    def __init__(self, input_dim, d_model=32, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(64, d_model)

    def forward(self, x):
        # [B, T, C] → [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.mean(dim=2)  # global average pooling
        return self.fc(x)


class AudioEncoder_CNN(nn.Module):
    """
    Lightweight CNN for stimulus encoding.
    """
    def __init__(self, input_dim, d_model=32, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(64, d_model)

    def forward(self, x):
        # [B, T, C] → [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.mean(dim=2)
        return self.fc(x)
