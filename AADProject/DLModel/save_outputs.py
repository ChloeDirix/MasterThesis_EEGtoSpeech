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

                logits = pl_module(eeg, stim)

                if logits.ndim == 2 and logits.size(1) == 2:
                    pred = logits.argmax(dim=1)
                elif logits.ndim == 1:
                    pred = (logits > 0).long()
                else:
                    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

                preds.append(pred.detach().cpu())
                labels.append(att.detach().cpu())

                if self.save_embeddings:
                    b = torch.arange(stim.size(0), device=device)
                    att_stim = stim[b, att]  # expects stim [B, 2, T, ...]
                    z_eeg, z_stim = pl_module.model.get_pair_embeddings(eeg, att_stim)
                    embeds.append(torch.cat([z_eeg, z_stim], dim=1).detach().cpu())

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