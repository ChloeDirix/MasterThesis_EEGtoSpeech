# DLModel/plot_checks.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback


class PlotCallback(Callback):
    """
    Lightweight training-time diagnostics:
      - train/val accuracy curves
      - train/val loss curves
      - gradient norm curve (properly computed)

    No extra validation pass.
    """

    def __init__(self, save_dir="Results/plots", track_grad_norm: bool = True):
        super().__init__()
        self.save_dir = save_dir
        self.track_grad_norm = track_grad_norm
        os.makedirs(self.save_dir, exist_ok=True)

        # Metrics storage
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

        # Gradient tracking
        self.batch_grad_norms = []
        self.epoch_grad_norms = []

    # ----------------------------------------------------------
    # CORRECT place to measure gradients
    # ----------------------------------------------------------
    def on_after_backward(self, trainer, pl_module):
        if not self.track_grad_norm:
            return

        total_norm_sq = 0.0

        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2

        total_norm = total_norm_sq ** 0.5
        self.batch_grad_norms.append(total_norm)

    # ----------------------------------------------------------
    # End of training epoch
    # ----------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics

        if "train_acc" in logs:
            self.train_acc.append(float(logs["train_acc"]))

        if "train_loss" in logs:
            self.train_loss.append(float(logs["train_loss"]))

        # Aggregate gradient norms per epoch
        if self.track_grad_norm and len(self.batch_grad_norms) > 0:
            mean_grad = float(np.mean(self.batch_grad_norms))
            self.epoch_grad_norms.append(mean_grad)
            self.batch_grad_norms = []

    # ----------------------------------------------------------
    # End of validation epoch
    # ----------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics

        if "val_acc" in logs:
            self.val_acc.append(float(logs["val_acc"]))

        if "val_loss" in logs:
            self.val_loss.append(float(logs["val_loss"]))

    # ----------------------------------------------------------
    # End of training (generate plots)
    # ----------------------------------------------------------
    def on_train_end(self, trainer, pl_module):

        # --------------------------
        # Accuracy curves
        # --------------------------
        if len(self.train_acc) or len(self.val_acc):
            plt.figure()
            if len(self.train_acc):
                plt.plot(self.train_acc, label="Train Acc")
            if len(self.val_acc):
                plt.plot(self.val_acc, label="Val Acc")

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Curves")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "accuracy_curves.png"))
            plt.close()

        # --------------------------
        # Loss curves
        # --------------------------
        if len(self.train_loss) or len(self.val_loss):
            plt.figure()
            if len(self.train_loss):
                plt.plot(self.train_loss, label="Train Loss")
            if len(self.val_loss):
                plt.plot(self.val_loss, label="Val Loss")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "loss_curves.png"))
            plt.close()

        # --------------------------
        # Gradient norm curve
        # --------------------------
        if self.track_grad_norm and len(self.epoch_grad_norms):
            plt.figure()
            plt.plot(self.epoch_grad_norms)
            plt.xlabel("Epoch")
            plt.ylabel("Global L2 Gradient Norm")
            plt.title("Gradient Norm Over Training")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "gradient_norms.png"))
            plt.close()

        print(f"[PlotCallback] Saved training curves to: {self.save_dir}")