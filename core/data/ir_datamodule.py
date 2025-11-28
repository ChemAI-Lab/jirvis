from .base import DataBase

from sklearn.model_selection import train_test_split


import numpy as np
import os
from .ir_dataset import TestDataset, TrainDataset, ImageDataset


class IRData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(IRData, self).__init__(cfg, **kwargs)
        self.root = os.path.expanduser(getattr(self.cfg, "data_root", "./data"))
        print(self.cfg)
        self.k = getattr(self.cfg, "k", 200)
        self.model_name = getattr(self.cfg, "model_name", "ResNet50")
        self.use_catch_all = getattr(self.cfg, "use_catch_all", False)
        self.is_grayscale = getattr(self.cfg, "is_grayscale", True)

        # Paths to memory-mapped files
        self.images_path = os.path.join(self.root, "images.npy")
        self.labels_path = os.path.join(self.root, "labels.npy")

        # Load base dataset (grayscale determined by model name)
        self.label_indices = list(range(24))  # or pass dynamically via cfg
        self.base_dataset = ImageDataset(
            self.images_path,
            self.labels_path,
            label_indices=self.label_indices,
            use_catch_all=self.use_catch_all,
            is_grayscale=self.is_grayscale,
        )

        self._split_data(self.images_path)

    def _count_total(self, *args, **kwargs):
        return len(self.base_dataset)

    @property
    def train_dataset(self):
        return TrainDataset(
            dataset=self.base_dataset,
            balanced_indices=self.train_indices,
            model_name=self.model_name,
            use_catch_all=self.use_catch_all,
        )

    @property
    def val_dataset(self):
        return TestDataset(
            dataset=self.base_dataset,
            balanced_indices=self.val_indices,
            model_name=self.model_name,
            use_catch_all=self.use_catch_all,
        )

    @property
    def test_dataset(self):
        return TestDataset(
            dataset=self.base_dataset,
            balanced_indices=self.test_indices,
            model_name=self.model_name,
            use_catch_all=self.use_catch_all,
        )
