import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback


class PlotCallback(Callback):
    
    def __init__(
        self,
        plot_dir="Results/plots",
        history_dir="Results/posthoc",
        zoom_n_epochs: float = 5.0,
        subject_label: str | None = None,
        dataset_label: str | None = None,
        window_len_s: float | None = None,
        debug_keys: bool = False,
    ):
        super().__init__()
        self.plot_dir = plot_dir
        self.history_dir = history_dir
        self.zoom_n_epochs = float(zoom_n_epochs)
        self.subject_label = subject_label
        self.dataset_label = dataset_label
        self.window_len_s = window_len_s
        self.debug_keys = bool(debug_keys)

        self._printed_val_keys = False

        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        # Dense comparable curves only
        self.train_eval_acc_window = []
        self.train_eval_loss = []
        self.val_acc_window = []
        self.val_loss = []

        # Shared x-axis for dense validation events
        self.val_event_x = []

    def _get_validation_progress_x(self, trainer):
        """
        Returns fractional epoch progress, e.g.:
          0.2, 0.4, 0.6, 0.8, 1.0, ...
        when val_check_interval=0.2
        """
        epoch_idx = float(trainer.current_epoch)

        try:
            num_batches = trainer.num_training_batches
            completed = trainer.fit_loop.epoch_loop.batch_progress.current.completed

            if num_batches is None or num_batches <= 0:
                return epoch_idx

            frac = float(completed) / float(num_batches)
            frac = max(0.0, min(1.0, frac))
            return epoch_idx + frac
        except Exception:
            return epoch_idx

    @staticmethod
    def _subset_until_epoch(x, y, n_epochs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) == 0 or len(y) == 0:
            return np.asarray([]), np.asarray([])

        m = min(len(x), len(y))
        x = x[:m]
        y = y[:m]

        mask = x <= float(n_epochs)
        return x[mask], y[mask]

    @staticmethod
    def _finite_xy(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) == 0 or len(y) == 0:
            return np.asarray([]), np.asarray([])

        m = min(len(x), len(y))
        x = x[:m]
        y = y[:m]

        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def _title_prefix(self):
        parts = []
        if self.subject_label:
            parts.append(str(self.subject_label))
        if self.dataset_label:
            parts.append(str(self.dataset_label))
        if self.window_len_s is not None:
            parts.append(f"{self.window_len_s:g} s windows")
        return " — ".join(parts)

    def _full_title(self, base: str):
        prefix = self._title_prefix()
        return f"{prefix} — {base}" if prefix else base

    def _save_history_npz(self):
        os.makedirs(self.history_dir, exist_ok=True)

        np.savez(
            os.path.join(self.history_dir, "metric_history.npz"),
            val_event_x=np.asarray(self.val_event_x, dtype=float),
            train_eval_acc_window=np.asarray(self.train_eval_acc_window, dtype=float),
            val_acc_window=np.asarray(self.val_acc_window, dtype=float),
            train_eval_loss=np.asarray(self.train_eval_loss, dtype=float),
            val_loss=np.asarray(self.val_loss, dtype=float),
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        x_val = self._get_validation_progress_x(trainer)
        self.val_event_x.append(x_val)

        if self.debug_keys and not self._printed_val_keys:
            print("[PlotCallback] VAL callback_metrics keys:", sorted(logs.keys()))
            self._printed_val_keys = True

        te_acc = logs.get("train_eval_acc_window", None)
        v_acc = logs.get("val_acc_window", None)
        te_loss = logs.get("train_eval_loss", None)
        v_loss = logs.get("val_loss", None)

        self.train_eval_acc_window.append(float(te_acc) if te_acc is not None else np.nan)
        self.val_acc_window.append(float(v_acc) if v_acc is not None else np.nan)
        self.train_eval_loss.append(float(te_loss) if te_loss is not None else np.nan)
        self.val_loss.append(float(v_loss) if v_loss is not None else np.nan)

    def _plot_accuracy_dense(self):
        if not (
            np.isfinite(np.asarray(self.train_eval_acc_window, dtype=float)).any()
            or np.isfinite(np.asarray(self.val_acc_window, dtype=float)).any()
        ):
            return

        plt.figure(figsize=(8, 4.5))

        x_te, y_te = self._finite_xy(
            self.val_event_x[:len(self.train_eval_acc_window)],
            self.train_eval_acc_window,
        )
        x_v, y_v = self._finite_xy(
            self.val_event_x[:len(self.val_acc_window)],
            self.val_acc_window,
        )

        if len(y_te):
            plt.plot(x_te, y_te, marker="o", label="Train eval window accuracy")
        if len(y_v):
            plt.plot(x_v, y_v, marker="o", label="Validation window accuracy")

        plt.xlabel("Epoch progress")
        plt.ylabel("Accuracy")
        plt.title(self._full_title("Window Accuracy During Training"))
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "accuracy_dense.png"))
        plt.close()

    def _plot_accuracy_zoom(self):
        if not (
            np.isfinite(np.asarray(self.train_eval_acc_window, dtype=float)).any()
            or np.isfinite(np.asarray(self.val_acc_window, dtype=float)).any()
        ):
            return

        plt.figure(figsize=(8, 4.5))

        x_te, y_te = self._subset_until_epoch(
            self.val_event_x,
            self.train_eval_acc_window,
            self.zoom_n_epochs,
        )
        x_v, y_v = self._subset_until_epoch(
            self.val_event_x,
            self.val_acc_window,
            self.zoom_n_epochs,
        )

        x_te, y_te = self._finite_xy(x_te, y_te)
        x_v, y_v = self._finite_xy(x_v, y_v)

        if len(y_te):
            plt.plot(x_te, y_te, marker="o", label="Train eval window accuracy")
        if len(y_v):
            plt.plot(x_v, y_v, marker="o", label="Validation window accuracy")

        plt.xlabel("Epoch progress")
        plt.ylabel("Accuracy")
        plt.title(self._full_title(f"Window Accuracy During Training — First {self.zoom_n_epochs:g} Epochs"))
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "accuracy_zoom_first_epochs.png"))
        plt.close()

    def _plot_loss_dense(self):
        if not (
            np.isfinite(np.asarray(self.train_eval_loss, dtype=float)).any()
            or np.isfinite(np.asarray(self.val_loss, dtype=float)).any()
        ):
            return

        plt.figure(figsize=(8, 4.5))

        x_te, y_te = self._finite_xy(
            self.val_event_x[:len(self.train_eval_loss)],
            self.train_eval_loss,
        )
        x_v, y_v = self._finite_xy(
            self.val_event_x[:len(self.val_loss)],
            self.val_loss,
        )

        if len(y_te):
            plt.plot(x_te, y_te, marker="o", label="Train eval loss")
        if len(y_v):
            plt.plot(x_v, y_v, marker="o", label="Validation loss")

        plt.xlabel("Epoch progress")
        plt.ylabel("Loss")
        plt.title(self._full_title("Window Loss During Training"))
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "loss_dense.png"))
        plt.close()

    def _plot_loss_zoom(self):
        if not (
            np.isfinite(np.asarray(self.train_eval_loss, dtype=float)).any()
            or np.isfinite(np.asarray(self.val_loss, dtype=float)).any()
        ):
            return

        plt.figure(figsize=(8, 4.5))

        x_te, y_te = self._subset_until_epoch(
            self.val_event_x,
            self.train_eval_loss,
            self.zoom_n_epochs,
        )
        x_v, y_v = self._subset_until_epoch(
            self.val_event_x,
            self.val_loss,
            self.zoom_n_epochs,
        )

        x_te, y_te = self._finite_xy(x_te, y_te)
        x_v, y_v = self._finite_xy(x_v, y_v)

        if len(y_te):
            plt.plot(x_te, y_te, marker="o", label="Train eval loss")
        if len(y_v):
            plt.plot(x_v, y_v, marker="o", label="Validation loss")

        plt.xlabel("Epoch progress")
        plt.ylabel("Loss")
        plt.title(self._full_title(f"Window Loss During Training — First {self.zoom_n_epochs:g} Epochs"))
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "loss_zoom_first_epochs.png"))
        plt.close()

    def on_train_end(self, trainer, pl_module):
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        self._save_history_npz()
        self._plot_accuracy_dense()
        self._plot_accuracy_zoom()
        self._plot_loss_dense()
        self._plot_loss_zoom()

        print(f"[PlotCallback] Saved clean plots to: {self.plot_dir}")
        print(f"[PlotCallback] Saved metric history to: {self.history_dir}")