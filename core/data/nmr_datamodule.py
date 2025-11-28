from .base import DataBase


import numpy as np
import os
from .nmr_dataset import SimpleNMRDataset
import hydra
import torch
import h5py


class NMRData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(NMRData, self).__init__(cfg, **kwargs)
        self.root = os.path.expanduser(getattr(self.cfg, "data_root", "./data"))
        print(self.cfg)
        # self.k = getattr(self.cfg, 'k', 200)
        self.target_type = getattr(self.cfg, "target_type", "binary")

        self.nmr_path = self.root

        self.tokenizer = hydra.utils.instantiate(cfg.tokenizer)

        self._split_data(self.nmr_path)

    def _count_total(self, path):
        with h5py.File(path, "r") as f:
            n_total = len(f["molecules"])
        return n_total

    def _create_collate_fn(self):
        """Create a collate function that uses batch_tokenize"""

        def collate_fn(batch):
            # Extract texts and targets
            h_nmr_texts = [item["h_nmr_text"] for item in batch]
            c_nmr_texts = [item["c_nmr_text"] for item in batch]

            # Fix the warning by converting to numpy first
            targets = torch.tensor(
                np.array([item["targets"] for item in batch]), dtype=torch.float
            )

            # Batch tokenize - vocabulary is frozen (no updates)
            tokenized = self.tokenizer.batch_tokenize(
                h_nmr_texts, c_nmr_texts, update_vocab=False
            )

            return {
                "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    tokenized["attention_mask"], dtype=torch.long
                ),
                "scalar_values": torch.tensor(
                    tokenized["scalar_values"], dtype=torch.float
                ),
                "scalar_mask": torch.tensor(tokenized["scalar_mask"], dtype=torch.bool),
                "targets": targets,
            }

        return collate_fn

    @property
    def train_dataset(self):
        return SimpleNMRDataset(
            self.nmr_path,
            self.tokenizer,
            self.train_indices,
            self.target_type,
            phase="Train",
        )

    @property
    def val_dataset(self):
        return SimpleNMRDataset(
            self.nmr_path,
            self.tokenizer,
            self.val_indices,
            self.target_type,
            phase="Val",
        )

    @property
    def test_dataset(self):
        return SimpleNMRDataset(
            self.nmr_path,
            self.tokenizer,
            self.test_indices,
            self.target_type,
            phase="Test",
        )
