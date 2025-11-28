import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class DataBase(pl.LightningDataModule, ABC):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        # print("############Config############")
        # print(cfg)
        # print("############Config############")
        self.cfg = cfg
        self.batch_size = getattr(cfg, "batch_size", 32)
        self.num_workers = getattr(cfg, "num_workers", 8)

    def prepare_data(self):
        pass

    def _create_collate_fn(self):
        return None

    @abstractmethod
    def _count_total(self, path: str) -> int:
        """Return total number of items in the dataset at `path`."""
        raise NotImplementedError

    def _split_data(self, path):
        split_ratio = getattr(self.cfg, "split_ratio", [0.7, 0.15, 0.15])
        assert np.isclose(sum(split_ratio), 1.0), "Split ratios must sum to 1.0"

        train_indices, val_indices, test_indices = self.create_data_splits(
            path,
            train_ratio=split_ratio[0],
            val_ratio=split_ratio[1],
            test_ratio=split_ratio[2],
        )

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        print(
            f"Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}"
        )

    def create_data_splits(
        self,
        path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/val/test splits of molecule indices

        Returns:
            (train_indices, val_indices, test_indices)
        """

        # Get the total count
        n_total = self._count_total(path)

        # Create and shuffle indices
        indices = np.arange(n_total)
        np.random.shuffle(indices)

        # Calculate split points
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        print(
            f"Split sizes - Train: {len(train_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}"
        )

        return train_indices, val_indices, test_indices

    @property
    def train_dataset(self):
        raise NotImplementedError

    @property
    def val_dataset(self):
        raise NotImplementedError

    @property
    def test_dataset(self):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._create_collate_fn(),
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._create_collate_fn(),
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self._create_collate_fn(),
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )
