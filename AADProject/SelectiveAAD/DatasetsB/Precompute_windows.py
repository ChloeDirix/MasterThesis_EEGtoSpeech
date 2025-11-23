import torch
from torch.utils.data import DataLoader
from SelectiveAAD.datasets import SelectiveAADDataset   # Use your PATH A version
from paths import paths


def precompute_windows(nwb_paths, cfg, out_path, batch_size=32):
    """
    Extract all windows (EEG, stim, att) and save them to a .pt file.
    This removes NWB from the training loop entirely.
    """
    print(f"Precomputing windows from {len(nwb_paths)} subjects...")
    ds = SelectiveAADDataset(nwb_paths, cfg, multiband=True, split="precompute")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_eeg = []
    all_stim = []
    all_att = []

    for eeg, stim, att in loader:
        all_eeg.append(eeg)     # [B, T, C]
        all_stim.append(stim)   # [B, 2, T, bands]
        all_att.append(att)     # [B]

    eeg = torch.cat(all_eeg, dim=0)
    stim = torch.cat(all_stim, dim=0)
    att = torch.cat(all_att, dim=0)

    torch.save(
        {
            "eeg": eeg,   # [N, T, C]
            "stim": stim, # [N, 2, T, bands]
            "att": att,   # [N]
            "cfg": cfg,
        },
        out_path,
    )

    print(f"Saved {eeg.shape[0]} windows → {out_path}")


if __name__ == "__main__":
    cfg = paths.load_config()

    # TRAIN
    train_paths = [paths.subject_eegPP(s) for s in cfg["train_subjects"]]
    precompute_windows(train_paths, cfg, "train_windows.pt")

    # VALIDATION
    val_paths = [paths.subject_eegPP(s) for s in cfg["val_subjects"]]
    precompute_windows(val_paths, cfg, "val_windows.pt")
