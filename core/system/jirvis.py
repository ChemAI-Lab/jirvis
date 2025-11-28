from .base import BaseSystem
from typing import Any

from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
)
from torchmetrics import MetricCollection
import torch
import torch.nn.functional as F


class jirvis(BaseSystem):
    def __init__(self, config, **kwargs):
        super(jirvis, self).__init__(config)
        self.save_hyperparameters()

        num_labels = config.system.model.arch.num_labels

        # Separate metric collections for train and validation
        self.train_metrics = MetricCollection(
            {
                "micro_precision": MultilabelPrecision(
                    num_labels=num_labels, average="micro"
                ),
                "micro_recall": MultilabelRecall(
                    num_labels=num_labels, average="micro"
                ),
                "micro_f1": MultilabelF1Score(num_labels=num_labels, average="micro"),
                "macro_precision": MultilabelPrecision(
                    num_labels=num_labels, average="macro"
                ),
                "macro_recall": MultilabelRecall(
                    num_labels=num_labels, average="macro"
                ),
                "macro_f1": MultilabelF1Score(num_labels=num_labels, average="macro"),
            }
        )

        self.val_metrics = MetricCollection(
            {
                "micro_precision": MultilabelPrecision(
                    num_labels=num_labels, average="micro"
                ),
                "micro_recall": MultilabelRecall(
                    num_labels=num_labels, average="micro"
                ),
                "micro_f1": MultilabelF1Score(num_labels=num_labels, average="micro"),
                "macro_precision": MultilabelPrecision(
                    num_labels=num_labels, average="macro"
                ),
                "macro_recall": MultilabelRecall(
                    num_labels=num_labels, average="macro"
                ),
                "macro_f1": MultilabelF1Score(num_labels=num_labels, average="macro"),
            }
        )

        # Training EMR tracking
        self.train_emr_total = 0
        self.train_emr_correct = 0

        # Validation EMR tracking
        self.val_emr_total = 0
        self.val_emr_correct = 0

    def pre_process(self, batch):
        if hasattr(self, "data_transform") and self.data_transform is not None:
            batch = self.data_transform.pre_process(batch)
        return batch

    def forward(self, batch, **kwargs):
        inputs, _ = batch
        inputs = self.pre_process(inputs)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.pre_process(inputs)
        logits = self.model(inputs)

        loss = self.loss_func(logits, labels)

        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        # Update training metrics
        self.train_metrics.update(preds.int(), labels.int())

        # Update training EMR
        self.train_emr_correct += torch.all(preds == labels.bool(), dim=1).sum().item()
        self.train_emr_total += labels.size(0)

        self.log("train_loss", loss, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        inputs, labels = batch
        inputs = self.pre_process(inputs)
        logits = self.model(inputs)
        loss = self.loss_func(logits, labels)

        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        # Update validation metrics
        self.val_metrics.update(preds.int(), labels.int())

        # Update validation EMR
        self.val_emr_correct += torch.all(preds == labels.bool(), dim=1).sum().item()
        self.val_emr_total += labels.size(0)

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.pre_process(inputs)
        preds = self.model(inputs)

        loss = self.loss_func(preds, labels)
        self.log("test_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Compute and log training metrics
        train_computed = self.train_metrics.compute()
        self.log_dict(
            {f"train_{k}": v for k, v in train_computed.items()},
            prog_bar=False,
            sync_dist=True,
        )
        self.train_metrics.reset()

        # Compute and log training EMR
        train_emr = (
            self.train_emr_correct / self.train_emr_total
            if self.train_emr_total > 0
            else 0.0
        )
        self.log("train_emr", train_emr, prog_bar=False, sync_dist=True)

        # Reset training EMR counters
        self.train_emr_correct = 0
        self.train_emr_total = 0

        # add model checkpointing if loss/metric is better or best

    def on_validation_epoch_end(self):
        # Compute and log validation metrics
        val_computed = self.val_metrics.compute()
        self.log_dict(
            {f"val_{k}": v for k, v in val_computed.items()},
            prog_bar=False,
            sync_dist=True,
        )
        self.val_metrics.reset()

        # Compute and log validation EMR
        val_emr = (
            self.val_emr_correct / self.val_emr_total if self.val_emr_total > 0 else 0.0
        )
        self.log("val_emr", val_emr, prog_bar=False, sync_dist=True)

        # Reset validation EMR counters
        self.val_emr_correct = 0
        self.val_emr_total = 0

    def on_fit_start(self):
        self.train_metrics = self.train_metrics.to(self.device)
        self.val_metrics = self.val_metrics.to(self.device)

    def on_validation_epoch_start(self):
        if hasattr(self, "device"):
            self.train_metrics = self.train_metrics.to(self.device)
            self.val_metrics = self.val_metrics.to(self.device)
