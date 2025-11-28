from .base import DataBase

import os
from .multimodal_dataset import MultiModalDataset
import hydra
import torch
import numpy as np
import h5py


class SpectroData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(SpectroData, self).__init__(cfg, **kwargs)
        self.root = os.path.expanduser(getattr(self.cfg, "image_data_root", "./data"))
        self.nmr_tok_root = os.path.expanduser(
            getattr(self.cfg, "nmr_data_root", "./data/spectro_enhanced.h5")
        )
        self.jirvis_embeddings_root = os.path.expanduser(
            getattr(
                self.cfg,
                "jirvis_embeddings_root",
                "./data/jirvis_embeddings_pytorchL.npy",
            )
        )
        self.nmr_embeddings_root = os.path.expanduser(
            getattr(
                self.cfg,
                "nmr_embeddings_root",
                "./data/nmr_embeddings_pytorchL.npy",
            )
        )
        print(self.cfg)

        # Get vars from config
        self.model_name = getattr(self.cfg, "model_name", "ResNet50")
        self.use_catch_all = getattr(self.cfg, "use_catch_all", False)
        self.is_grayscale = getattr(self.cfg, "is_grayscale", True)
        self.use_images = getattr(self.cfg, "use_images", False)
        self.use_text = getattr(self.cfg, "use_text", False)
        self.nmr_model_dim = getattr(self.cfg, "nmr_model_dim", 256)

        # Paths to memory-mapped images
        self.images_path = os.path.join(self.root, "images.npy")
        self.total_samples = getattr(self.cfg, "total_samples", 768200)

        self.nmr_tokenizer = hydra.utils.instantiate(cfg.tokenizer)

        self._split_data(self.images_path)
        
        
        # Create all datasets immediately during initialization
        self._train_dataset = MultiModalDataset(
            self.images_path,
            self.nmr_tok_root,
            self.jirvis_embeddings_root,
            self.nmr_embeddings_root,
            self.total_samples,
            is_grayscale=self.is_grayscale,
            use_images=self.use_images,
            use_text=self.use_text,
            nmr_model_dim=self.nmr_model_dim,
            phase="Train",
            indices=self.train_indices,
            shared_data=None,
        )
        
        self._val_dataset = MultiModalDataset(
            self.images_path,
            self.nmr_tok_root,
            self.jirvis_embeddings_root,
            self.nmr_embeddings_root,
            self.total_samples,
            is_grayscale=self.is_grayscale,
            use_images=self.use_images,
            use_text=self.use_text,
            nmr_model_dim=self.nmr_model_dim,
            phase="Val",
            indices=self.val_indices,
            shared_data=None,
        )
        
        self._test_dataset = MultiModalDataset(
            self.images_path,
            self.nmr_tok_root,
            self.jirvis_embeddings_root,
            self.nmr_embeddings_root,
            self.total_samples,
            is_grayscale=self.is_grayscale,
            use_images=self.use_images,
            use_text=self.use_text,
            nmr_model_dim=self.nmr_model_dim,
            phase="Test",
            indices=self.test_indices,
            shared_data=None,
        )

    def _count_total(self, *args, **kwargs):
        return self.total_samples

    def _create_collate_fn(self):
        def collate_fn(batch):
            samples = {}
            samples["selfies"] = [item["selfie"] for item in batch]
            samples["tokenized_selfies"] = torch.stack(
                [item["tokenized_selfies"] for item in batch], dim=0
            )

            # Handle both image and embedding
            if self.use_images:
                images = torch.stack(
                    [item["image"] for item in batch], dim=0
                )  # (B,3,512,512)
                samples["images"] = images
            else:
                embeddings = torch.stack(
                    [item["jirvis_embedding"] for item in batch], dim=0
                )  # (B,2048)
                samples["jirvis_embeddings"] = embeddings

            if self.use_text:
                # Text data
                h_nmr_texts = [item["h_nmr_text"] for item in batch]
                c_nmr_texts = [item["c_nmr_text"] for item in batch]

                # NMR tokenization
                tokenized = self.nmr_tokenizer.batch_tokenize(
                    h_nmr_texts, c_nmr_texts, update_vocab=False
                )

                samples["input_ids"] = torch.tensor(
                    tokenized["input_ids"], dtype=torch.long
                )
                samples["attention_mask"] = torch.tensor(
                    tokenized["attention_mask"], dtype=torch.long
                )
                samples["scalar_values"] = torch.tensor(
                    tokenized["scalar_values"], dtype=torch.float
                )
                samples["scalar_mask"] = torch.tensor(
                    tokenized["scalar_mask"], dtype=torch.bool
                )
            else:
                # NMR embeddings
                nmr_embeddings = torch.stack(
                    [item["nmr_embedding"] for item in batch], dim=0
                )
                samples["nmr_embeddings"] = nmr_embeddings

            return samples

        return collate_fn

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def get_idx_to_token(self):
        """Get index to token mapping from the dataset."""
        return self.train_dataset.get_idx_to_token()
