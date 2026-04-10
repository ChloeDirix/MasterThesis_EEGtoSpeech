import os
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class SaveValOutputsCallback(Callback):
    def __init__(self, save_dir: str, save_embeddings: bool = True, save_once: bool = True):
        super().__init__()
        self.save_dir = save_dir
        self.save_embeddings = save_embeddings
        self.save_once = save_once
        self._saved = False
        os.makedirs(self.save_dir, exist_ok=True)

    def _should_save_now(self, trainer) -> bool:
        if trainer.sanity_checking:
            return False
        if self.save_once and self._saved:
            return False

        max_epochs = trainer.max_epochs
        if max_epochs is not None and max_epochs > 0:
            if trainer.current_epoch >= (max_epochs - 1):
                return True

        if getattr(trainer, "should_stop", False):
            return True

        return False

    def _forward_and_score(self, pl_module, eeg, stim):
        """
        Backward-compatible scoring helper.

        Supports:
        1) old style: pl_module(eeg, stim) -> logits
        2) new style: pl_module(eeg) -> pred_stim, then score_candidates(pred_stim, stim)
        """
        # Try old interface first
        try:
            logits = pl_module(eeg, stim)
            pred_stim = None
            return pred_stim, logits
        except TypeError:
            pass

        # New interface
        pred_stim = pl_module(eeg)
        if not hasattr(pl_module, "model") or not hasattr(pl_module.model, "score_candidates"):
            raise AttributeError(
                "New-style model detected, but pl_module.model.score_candidates(...) is missing."
            )
        logits = pl_module.model.score_candidates(pred_stim, stim)
        return pred_stim, logits

    def _extract_embeddings(self, pl_module, eeg, stim, att, pred_stim=None):
        """
        Try to save something meaningful for both old and new models.
        """
        device = eeg.device
        b = torch.arange(stim.size(0), device=device)
        att_stim = stim[b, att]   # [B, T, C] assuming stim [B, 2, T, C]

        # Old-style models may expose get_pair_embeddings
        if hasattr(pl_module.model, "get_pair_embeddings"):
            try:
                z_eeg, z_stim = pl_module.model.get_pair_embeddings(eeg, att_stim)
                if z_eeg.ndim == 3 and z_stim.ndim == 3:
                    return torch.cat([z_eeg, z_stim], dim=1).detach().cpu()
                return z_eeg.detach().cpu()
            except Exception:
                pass

        # New-style model: save predicted stimulus directly
        if pred_stim is not None:
            return pred_stim.detach().cpu()

        return None

    def _compute_and_save(self, trainer, pl_module):
        val_loaders = trainer.val_dataloaders
        if not val_loaders:
            print("[SaveValOutputsCallback] No val dataloader found; skipping.")
            return
        val_loader = val_loaders[0] if isinstance(val_loaders, (list, tuple)) else val_loaders

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        preds, labels, embeds = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                eeg, stim, att = batch[0], batch[1], batch[2]

                eeg = eeg.to(device)
                stim = stim.to(device)
                att = att.to(device)

                pred_stim, logits = self._forward_and_score(pl_module, eeg, stim)

                if logits.ndim == 2 and logits.size(1) == 2:
                    pred = logits.argmax(dim=1)
                elif logits.ndim == 1:
                    pred = (logits > 0).long()
                else:
                    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

                preds.append(pred.detach().cpu())
                labels.append(att.detach().cpu())

                if self.save_embeddings:
                    emb = self._extract_embeddings(pl_module, eeg, stim, att, pred_stim=pred_stim)
                    if emb is not None:
                        embeds.append(emb)

        if was_training:
            pl_module.train()

        out = {
            "preds": torch.cat(preds).numpy(),
            "labels": torch.cat(labels).numpy(),
        }
        if self.save_embeddings and len(embeds) > 0:
            out["embeds"] = torch.cat(embeds).numpy()

        out_path = os.path.join(self.save_dir, "val_outputs.npz")
        np.savez(out_path, **out)

        self._saved = True
        print(f"[SaveValOutputsCallback] Saved {list(out.keys())} to {out_path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._should_save_now(trainer):
            self._compute_and_save(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if not self._saved:
            self._compute_and_save(trainer, pl_module)