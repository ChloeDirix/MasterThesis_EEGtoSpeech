import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

class PlotCallback(Callback):

    def __init__(self, save_dir="Results/plots"):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.grad_norms = []

    # <-- FIXED: now a real class method
    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if "train_acc" in logs:
            self.train_acc.append(float(logs["train_acc"]))
        if "train_loss" in logs:
            self.train_loss.append(float(logs["train_loss"]))

        # Gradient Norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
        self.grad_norms.append(total_norm)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if "val_acc" in logs:
            self.val_acc.append(float(logs["val_acc"]))
        if "val_loss" in logs:
            self.val_loss.append(float(logs["val_loss"]))

    def on_train_end(self, trainer, pl_module):
        print("\nGenerating diagnostic plots...")

        # ------------------------------
        # 1. Accuracy curves
        # ------------------------------
        plt.figure()
        plt.plot(self.train_acc, label="Train Acc")
        plt.plot(self.val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "accuracy_curves.png"))
        plt.close()

        # ------------------------------
        # 2. Loss curves
        # ------------------------------
        plt.figure()
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "loss_curves.png"))
        plt.close()

        # ------------------------------
        # 3. Gradient norm curve
        # ------------------------------
        plt.figure()
        plt.plot(self.grad_norms)
        plt.xlabel("Epoch")
        plt.ylabel("L2 Gradient Norm")
        plt.title("Gradient Norm Over Training")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "gradient_norms.png"))
        plt.close()

        # ------------------------------
        # Prepare Loader
        # ------------------------------
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        preds = []
        labels = []
        embeds = []

        device = pl_module.device
        pl_module.eval()

        with torch.no_grad():
            for eeg, stim, att in val_loader:
                eeg = eeg.to(device)
                stim = stim.to(device)

                # Forward model
                logits = pl_module(eeg, stim)
                pred = logits.argmax(dim=1)

                # Extract embeddings for PCA
                z_eeg, z_stim = pl_module.model.get_embeddings(eeg, stim[:,0])  #only attended stim
                embeds.append(torch.cat([z_eeg, z_stim], dim=1).cpu())

                preds.append(pred.cpu())
                labels.append(att)

        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        embeds = torch.cat(embeds).numpy()

        # ------------------------------
        # 4. Prediction Histogram
        # ------------------------------
        plt.figure()
        plt.hist(preds, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
        plt.xticks([0, 1])
        plt.xlabel("Predicted Class")
        plt.ylabel("Count")
        plt.title("Prediction Distribution")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "prediction_histogram.png"))
        plt.close()

        # ------------------------------
        # 5. PCA of embeddings
        # ------------------------------
        pca = PCA(n_components=2)
        emb2 = pca.fit_transform(embeds)

        plt.figure()
        plt.scatter(emb2[:, 0], emb2[:, 1], c=labels, cmap="coolwarm", alpha=0.6)
        plt.colorbar(label="True Class")
        plt.title("PCA of EEG+Stimulus Embeddings")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "pca_embeddings.png"))
        plt.close()

        # ------------------------------
        # 6. Confusion Matrix
        # ------------------------------
        cm = confusion_matrix(labels, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Unatt", "Attended"],
                    yticklabels=["Unatt", "Attended"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"))
        plt.close()

        print(f"Saved all diagnostic plots to: {self.save_dir}")
