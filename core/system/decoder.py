import pytorch_lightning as pl
import numpy as np
from .base import BaseSystem
from typing import Any


class MoleculeDecoder(BaseSystem):
    def __init__(self, config, **kwargs):
        super(MoleculeDecoder, self).__init__(config)
        self.save_hyperparameters()

    def pre_process(self, batch):
        if hasattr(self, "data_transform") and self.data_transform is not None:
            batch = self.data_transform.pre_process(batch)
        return batch

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        inputs, labels = batch
        inputs = self.pre_process(inputs)
        preds = self.model(inputs)

        loss = self.loss_func(preds, labels)

        self.log("val_loss", loss)
        return loss

    def forward(self, batch, **kwargs):
        inputs, _ = batch
        inputs = self.pre_process(inputs)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.pre_process(inputs)
        preds = self.model(inputs)

        loss = self.loss_func(preds, labels)

        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.pre_process(inputs)
        preds = self.model(inputs)

        loss = self.loss_func(preds, labels)

        self.log("test_loss", loss)
        return loss

    def compute_metrics(self, preds, gt):
        pass
