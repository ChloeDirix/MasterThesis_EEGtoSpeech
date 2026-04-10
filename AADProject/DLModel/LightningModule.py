import torch
import pytorch_lightning as pl

from DLModel.LogisticRankLoss import LogisticRankLoss
from DLModel.CorrelationLoss import CorrelationLoss
from DLModel.models.Type2_model import DirectCorrAADModel
from DLModel.models.Type3_model import CorrRankAADModel
from DLModel.models.Type4_model import TransformerRankAADModel
from DLModel.checkpoint_utils import freeze_all_except


def _safe_mean(x: torch.Tensor):
    if x.numel() == 0:
        return torch.tensor(float("nan"))
    return x.float().mean()


def _binary_metrics_from_preds_targets(preds: torch.Tensor, targets: torch.Tensor):
    """
    preds, targets: int tensors in {0,1}
    Returns:
      acc, bal_acc, recall0, recall1, mcc
    """
    preds = preds.long().view(-1)
    targets = targets.long().view(-1)

    acc = (preds == targets).float().mean()

    mask0 = (targets == 0)
    mask1 = (targets == 1)

    recall0 = _safe_mean((preds[mask0] == 0).float()) if mask0.any() else torch.tensor(float("nan"))
    recall1 = _safe_mean((preds[mask1] == 1).float()) if mask1.any() else torch.tensor(float("nan"))
    bal_acc = 0.5 * (recall0 + recall1)

    tp = ((preds == 1) & (targets == 1)).sum().float()
    tn = ((preds == 0) & (targets == 0)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den.item() > 0:
        mcc = (tp * tn - fp * fn) / den
    else:
        mcc = torch.tensor(0.0)

    return acc, bal_acc, recall0, recall1, mcc


class AADLightningModule(pl.LightningModule):
    def __init__(self, cfg, input_dims, pretrained_model=None):
        super().__init__()

        C_eeg, C_stim, K = input_dims
        self.dl_cfg = cfg["DeepLearning"]
        self.K = int(K)

        if self.K != 2:
            raise ValueError(
                f"AADLightningModule currently assumes exactly 2 candidate stimuli, got K={self.K}"
            )

        model_cfg = self.dl_cfg["modelType"]
        self.model_name = model_cfg["name"]
        self.loss_cfg = self.dl_cfg["loss"]

        if pretrained_model is not None:
            self.model = pretrained_model

            if self.model_name == "Type2":
                self.criterion = CorrelationLoss()

            elif self.model_name in {"Type3", "Type4"}:
                if self.loss_cfg["name"] == "logistic_rank":
                    self.criterion = LogisticRankLoss(
                        margin=float(self.loss_cfg.get("margin", 0.0))
                    )
                else:
                    raise ValueError(
                        f"Unsupported {self.model_name} loss: {self.loss_cfg['name']}"
                    )

            else:
                raise ValueError(f"Unknown model name: {self.model_name}")

        else:
            if self.model_name == "Type2":
                self.criterion = CorrelationLoss()
                self.model = DirectCorrAADModel(
                    self.dl_cfg,
                    eeg_input_dim=C_eeg,
                    stim_input_dim=C_stim,
                )

            elif self.model_name == "Type3":
                if self.loss_cfg["name"] == "logistic_rank":
                    self.criterion = LogisticRankLoss(
                        margin=float(self.loss_cfg.get("margin", 0.0))
                    )
                else:
                    raise ValueError(
                        f"Unsupported Type3 loss: {self.loss_cfg['name']}"
                    )

                self.model = CorrRankAADModel(
                    self.dl_cfg,
                    eeg_input_dim=C_eeg,
                    stim_input_dim=C_stim,
                )

            elif self.model_name == "Type4":
                if self.loss_cfg["name"] == "logistic_rank":
                    self.criterion = LogisticRankLoss(
                        margin=float(self.loss_cfg.get("margin", 0.0))
                    )
                else:
                    raise ValueError(
                        f"Unsupported Type4 loss: {self.loss_cfg['name']}"
                    )

                self.model = TransformerRankAADModel(
                    self.dl_cfg,
                    eeg_input_dim=C_eeg,
                    stim_input_dim=C_stim,
                )

            else:
                raise ValueError(f"Unknown model name: {self.model_name}")

            ft_cfg = self.dl_cfg.get("fine_tuning", {})
            freeze_prefixes = ft_cfg.get("trainable_prefixes", None)
            freeze_all_first = bool(ft_cfg.get("freeze_all_first", False))

            if freeze_all_first and freeze_prefixes:
                freeze_all_except(self.model, freeze_prefixes)

        # training buffers
        self._train_preds = []
        self._train_atts = []

        # train-eval buffers (dataloader_idx=0)
        self._train_eval_scores = []
        self._train_eval_preds = []
        self._train_eval_atts = []
        self._train_eval_trial_uids = []
        self._train_eval_losses = []

        # validation buffers (dataloader_idx=1)
        self._val_scores = []
        self._val_preds = []
        self._val_atts = []
        self._val_trial_uids = []
        self._val_subjects = []
        self._val_losses = []

    def forward(self, eeg, stim):
        return self.model(eeg, stim)

    def _compute_loss_and_scores(self, eeg, stim, att):
        att = att.long().view(-1)

        if self.model_name == "Type2":
            out = self.model(eeg, stim)
            pred = out["pred"]              # [B,T,C]
            candidates = out["candidates"]  # [B,K,T,C]
            scores = out["scores"]          # [B,K]

            b = torch.arange(pred.size(0), device=pred.device)
            attended = candidates[b, att]   # [B,T,C]

            loss = self.criterion(pred, attended)
            return loss, scores

        elif self.model_name in {"Type3", "Type4"}:
            scores = self.model(eeg, stim)  # [B,K]
            if scores.ndim != 2 or scores.size(1) != 2:
                raise ValueError(
                    f"Expected scores of shape [B,2], got {tuple(scores.shape)}"
                )

            loss = self.criterion(scores, att)
            return loss, scores

        else:
            raise RuntimeError(f"Unhandled model_name: {self.model_name}")

    def _compute_epoch_metrics_from_buffers(self, preds_list, atts_list, scores_list, trial_uids):
        if len(preds_list) == 0:
            return {}

        preds = torch.cat(preds_list)
        atts = torch.cat(atts_list)
        scores = torch.cat(scores_list)  # [Nw, 2]

        acc_window, _, _, _, _ = _binary_metrics_from_preds_targets(preds, atts)

        trial_score_sums = {}
        trial_counts = {}
        trial_target = {}

        for i in range(scores.size(0)):
            tid = trial_uids[i]
            sc = scores[i]
            y = int(atts[i].item())

            if tid not in trial_score_sums:
                trial_score_sums[tid] = sc.clone()
                trial_counts[tid] = 1
                trial_target[tid] = y
            else:
                trial_score_sums[tid] += sc
                trial_counts[tid] += 1

        trial_correct = []
        for tid in trial_score_sums.keys():
            mean_scores = trial_score_sums[tid] / float(trial_counts[tid])
            y = trial_target[tid]
            pred_t = int(torch.argmax(mean_scores).item())
            trial_correct.append(1.0 if pred_t == y else 0.0)

        if len(trial_correct) > 0:
            acc_trial = torch.tensor(trial_correct).float().mean()
        else:
            acc_trial = torch.tensor(float("nan"))

        return {
            "acc_window": acc_window,
            "acc_trial": acc_trial,
        }

    def training_step(self, batch, batch_idx):
        eeg, stim, att, meta = batch
        batch_size = eeg.size(0)

        loss, scores = self._compute_loss_and_scores(eeg, stim, att)

        att = att.long().view(-1)
        pred = scores.argmax(dim=1)

        self._train_preds.append(pred.detach().cpu())
        self._train_atts.append(att.detach().cpu())

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_end(self):
        if len(self._train_preds) == 0:
            return

        preds = torch.cat(self._train_preds)
        atts = torch.cat(self._train_atts)

        train_acc, _, _, _, _ = _binary_metrics_from_preds_targets(preds, atts)

        self.log("train_acc_window", train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(preds))

        self._train_preds.clear()
        self._train_atts.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        eeg, stim, att, meta = batch
        batch_size = eeg.size(0)

        loss, scores = self._compute_loss_and_scores(eeg, stim, att)

        att = att.long().view(-1)
        pred = scores.argmax(dim=1)

        if dataloader_idx == 0:
            self._train_eval_scores.append(scores.detach().cpu())
            self._train_eval_preds.append(pred.detach().cpu())
            self._train_eval_atts.append(att.detach().cpu())
            self._train_eval_trial_uids.extend(list(meta["trial_uid"]))
            self._train_eval_losses.append(loss.detach().cpu())

        elif dataloader_idx == 1:
            self._val_scores.append(scores.detach().cpu())
            self._val_preds.append(pred.detach().cpu())
            self._val_atts.append(att.detach().cpu())
            self._val_trial_uids.extend(list(meta["trial_uid"]))
            self._val_subjects.extend(list(meta["subject"]))
            self._val_losses.append(loss.detach().cpu())

        else:
            raise ValueError(f"Unexpected dataloader_idx={dataloader_idx}")

        return {"loss": loss.detach(), "batch_size": batch_size}

    def on_validation_epoch_end(self):
        # train-eval metrics
        if len(self._train_eval_preds) > 0:
            m = self._compute_epoch_metrics_from_buffers(
                preds_list=self._train_eval_preds,
                atts_list=self._train_eval_atts,
                scores_list=self._train_eval_scores,
                trial_uids=self._train_eval_trial_uids,
            )

            train_eval_loss = torch.stack(self._train_eval_losses).mean()

            self.log("train_eval_acc_window", m["acc_window"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(torch.cat(self._train_eval_preds)))
            self.log("train_eval_acc_trial", m["acc_trial"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(torch.cat(self._train_eval_preds)))
            self.log("train_eval_loss", train_eval_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(torch.cat(self._train_eval_preds)))

        # validation metrics
        if len(self._val_preds) > 0:
            m = self._compute_epoch_metrics_from_buffers(
                preds_list=self._val_preds,
                atts_list=self._val_atts,
                scores_list=self._val_scores,
                trial_uids=self._val_trial_uids,
            )

            val_loss = torch.stack(self._val_losses).mean()

            self.log("val_acc_window", m["acc_window"], prog_bar=True, on_step=False, on_epoch=True, batch_size=len(torch.cat(self._val_preds)))
            self.log("val_acc_trial", m["acc_trial"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(torch.cat(self._val_preds)))
            self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(torch.cat(self._val_preds)))

        # clear train-eval buffers
        self._train_eval_scores.clear()
        self._train_eval_preds.clear()
        self._train_eval_atts.clear()
        self._train_eval_trial_uids.clear()
        self._train_eval_losses.clear()

        # clear validation buffers
        self._val_scores.clear()
        self._val_preds.clear()
        self._val_atts.clear()
        self._val_trial_uids.clear()
        self._val_subjects.clear()
        self._val_losses.clear()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found. Check freezing configuration.")

        return torch.optim.AdamW(
            trainable_params,
            lr=float(self.dl_cfg["train"]["learning_rate"]),
            weight_decay=float(self.dl_cfg["train"]["weight_decay"]),
        )