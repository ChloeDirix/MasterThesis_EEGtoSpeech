import os
import torch

from DLModel.models.Type2_model import DirectCorrAADModel
from DLModel.models.Type3_model import CorrRankAADModel


def build_model_from_cfg(cfg, input_dims):
    """
    Build a bare model (not LightningModule) from config + input dims.
    """
    C_eeg, C_stim, K = input_dims
    dl_cfg = cfg["DeepLearning"]
    model_name = dl_cfg["modelType"]["name"]

    if K != 2:
        raise ValueError(f"Expected K=2 candidate stimuli, got K={K}")

    if model_name == "Type2":
        return DirectCorrAADModel(
            dl_cfg,
            eeg_input_dim=C_eeg,
            stim_input_dim=C_stim,
        )

    if model_name == "Type3":
        return CorrRankAADModel(
            dl_cfg,
            eeg_input_dim=C_eeg,
            stim_input_dim=C_stim,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


def freeze_all_except(module, trainable_prefixes):
    """
    trainable_prefixes: list of parameter-name prefixes to keep trainable.
    Example:
        freeze_all_except(model, ["eeg_encoder", "classifier"])
    """
    if isinstance(trainable_prefixes, str):
        trainable_prefixes = [trainable_prefixes]

    for name, p in module.named_parameters():
        keep_trainable = any(name.startswith(pref) for pref in trainable_prefixes)
        p.requires_grad = bool(keep_trainable)


def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_training_checkpoint(
    path,
    model,
    cfg,
    input_dims,
    train_subjects=None,
    val_subjects=None,
    eeg_std=None,
    envL_std=None,
    envR_std=None,
    subject_stds=None,
    dataset_stds=None,
    extra=None,
):
    """
    Save a reusable training checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "input_dims": input_dims,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "normalization_bundle": {
            "eeg_std": eeg_std,
            "envL_std": envL_std,
            "envR_std": envR_std,
            "subject_stds": subject_stds,
            "dataset_stds": dataset_stds,
        },
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_model_checkpoint(path, map_location="cpu"):
    """
    Returns:
        model, checkpoint_dict
    """
    ckpt = torch.load(path, map_location=map_location)
    model = build_model_from_cfg(ckpt["config"], ckpt["input_dims"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt