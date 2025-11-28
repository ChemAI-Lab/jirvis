import pytorch_lightning as pl
import numpy as np
from .base import BaseSystem
from typing import Any
import hydra

from sklearn.metrics import r2_score


class NMREncoder(BaseSystem):
    def __init__(self, config, **kwargs):
        self.tokenizer = hydra.utils.instantiate(config.system.model.tokenizer)
        config.system.model.arch.vocab_size = self.tokenizer.get_vocab_size()
        assert (
            config.system.model.arch.vocab_size is not None
        ), "Updating vocab size failed"
        # del self.tokenizer
        super(NMREncoder, self).__init__(config)
        self.save_hyperparameters()

    def pre_process(self, batch):
        if hasattr(self, "data_transform") and self.data_transform is not None:
            batch = self.data_transform.pre_process(batch)
        return batch

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        targets = batch["targets"]

        logits = self.forward(batch)
        loss = self.loss_func(logits, targets)

        self.compute_metrics(logits, targets, phase="val")

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def forward(self, batch, **kwargs):
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            scalar_values=batch["scalar_values"],
            scalar_mask=batch["scalar_mask"],
        )

        return output

    def training_step(self, batch, batch_idx):
        targets = batch["targets"]

        logits = self.forward(batch)
        loss = self.loss_func(logits, targets)

        self.compute_metrics(logits, targets, phase="train")

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        targets = batch["targets"]

        logits = self.forward(batch)
        loss = self.loss_func(logits, targets)

        self.compute_metrics(logits, targets, phase="test")

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def compute_metrics(self, preds, gt, phase="None"):
        # Detach and move to CPU just once
        preds = preds.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        # Vectorized metrics
        mse = np.mean((preds - gt) ** 2, axis=0)
        mae = np.mean(np.abs(preds - gt), axis=0)
        rmse = np.sqrt(mse)

        # R2 requires sklearn (no proper vectorized version)
        r2 = []

        for i in range(gt.shape[1]):
            r2.append(r2_score(gt[:, i], preds[:, i]))
        r2 = np.array(r2)

        # Log mean for easy tracking
        self.log(
            f"{phase}_mse", mse.mean(), on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            f"{phase}_mae", mae.mean(), on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            f"{phase}_rmse", rmse.mean(), on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(f"{phase}_r2", r2.mean(), on_step=True, on_epoch=False, prog_bar=True)

        return {
            "mse": mse.tolist(),
            "mae": mae.tolist(),
            "rmse": rmse.tolist(),
            "r2": r2.tolist(),
        }
