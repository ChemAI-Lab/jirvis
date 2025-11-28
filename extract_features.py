#!/usr/bin/env python3

import os

# Set strongest GPUs before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import numpy as np
from core.system.jirvis import jirvis
from core.data.multimodal_datamodule import SpectroData
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, checkpoint_path, config_path="configs/system/jirvis.yaml"):
        """
        Initialize feature extractor with trained jirvis model.

        Args:
            checkpoint_path: Path to trained jirvis checkpoint
            config_path: Path to jirvis config
        """
        # Load the trained model
        self.system = jirvis.load_from_checkpoint(checkpoint_path)
        self.system.eval()

        # Extract the ResNet feature extractor (before MLP)
        self.feature_model = self.system.model.features

        # Enable DataParallel for multi-GPU inference
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference")
            self.feature_model = torch.nn.DataParallel(self.feature_model)

        # Move to GPU and enable optimizations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_model = self.feature_model.to(device)
        self.device = device

        # Enable inference optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        print(f"Loaded jirvis model from {checkpoint_path}")
        print(f"Feature extractor ready - outputs 2048-dim vectors")
        print(f"Using device: {device}")

    def extract_features_batch(self, images):
        """
        Extract 2048-dim features from a batch of images.

        Args:
            images: torch.Tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor of shape (B, 2048)
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Convert grayscale to RGB if needed (same as jirvis)
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)

            # Extract features using ResNet backbone
            features = self.feature_model(images)  # (B, 2048, 1, 1)
            features = torch.flatten(features, 1)  # (B, 2048)

        return features.float()  # Convert back to float32

    def extract_from_dataset(self, dataset, output_path, max_samples=None):
        """
        Extract features from dataset in sequential order to maintain traceability.

        Args:
            dataset: PyTorch Dataset (not DataLoader to maintain order)
            output_path: Path to save extracted features (.npy)
            max_samples: Maximum number of samples to process (None for all)
        """
        # Use entire dataset if max_samples not specified
        total_samples = min(max_samples or len(dataset), len(dataset))

        print(f"Processing {total_samples} samples in sequential order...")
        print(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        # Larger batch size for better GPU utilization
        batch_size = 64  # Increased from 16

        # Pre-allocate numpy array for all features
        all_features = np.zeros((total_samples, 2048), dtype=np.float32)

        # Process samples sequentially with larger batches
        for i in tqdm(range(0, total_samples, batch_size), desc="Extracting features"):
            batch_end = min(i + batch_size, total_samples)
            batch_indices = list(range(i, batch_end))

            # Load batch data (only images needed for features)
            batch_images = []
            for idx in batch_indices:
                sample = dataset[idx]
                batch_images.append(sample["image"])

            # Stack images and move to device with non_blocking transfer
            images = torch.stack(batch_images).to(self.device, non_blocking=True)

            # Extract features
            features = self.extract_features_batch(images)
            features_np = features.cpu().numpy()

            # Write to pre-allocated array
            all_features[i:batch_end] = features_np

        # Save as numpy array - much faster and simpler
        np.save(output_path, all_features)

        print(f"Extracted {total_samples} feature vectors to {output_path}")
        print(f"Order maintained: features[i] = encoding of dataset[i]")
        print(f"Usage: features = np.load('{output_path}')")
        return output_path


def run_extraction():
    """Simple function to run feature extraction without hydra complexity."""
    # Set checkpoint path directly
    checkpoint_path = "/home/rudra/spectro/logs/jirvis/sim_ir/jirvis/checkpoints/epoch=47-acc=0.0000.ckpt"

    # Initialize feature extractor
    extractor = FeatureExtractor(checkpoint_path)

    # Create simple config for multimodal data using actual paths
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "data": {
                "name": "sim_spectro",
                "data": {
                    "image_data_root": "/home/rudra/Spectro/downloaded/data/final_data/ir_images_simulated/",
                    "nmr_data_root": "/home/rudra/Spectro/downloaded/data/final_data/nmr_text/spectro_data_tokenized.h5",
                    "total_samples": 768200,
                    "is_grayscale": True,
                    "model_name": "ResNet50",
                    "use_catch_all": False,
                    "split_ratio": [0.7, 0.15, 0.15],
                    "batch_size": 32,
                    "num_workers": 4,
                    "tokenizer": {
                        "_target_": "models.nmr_tokenizer.NMRTokenizer",
                        "max_length": 300,
                    },
                },
            }
        }
    )

    # Create the full dataset directly (not split by train/val/test)
    from core.data.multimodal_dataset import MultiModalDataset

    # Use the full dataset
    full_dataset = MultiModalDataset(
        images_path=config.data.data.image_data_root + "images.npy",
        spectro_h5_path=config.data.data.nmr_data_root,
        jirvis_embeddings_path="/tmp/dummy.npy",  # Not used since use_images=True
        total_samples=config.data.data.total_samples,
        is_grayscale=config.data.data.is_grayscale,
        use_images=True,  # We want to extract features FROM images
        phase="full",
    )

    # Extract features from entire dataset
    output_path = f"extracted_features_{config.data.name}.npy"
    extractor.extract_from_dataset(
        full_dataset, output_path, max_samples=None  # Process entire dataset
    )


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(config: DictConfig):
    run_extraction()


if __name__ == "__main__":
    main()
