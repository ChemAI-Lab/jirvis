#!/usr/bin/env python3

import os

# Set strongest GPUs before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import numpy as np
from core.system.nmr_enc import NMREncoder
from core.data.nmr_dataset import SimpleNMRDataset
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


class NMRFeatureExtractor:
    def __init__(self, checkpoint_path, config_path="configs/system/nmr_enc.yaml"):
        """
        Initialize feature extractor with trained NMR encoder model.

        Args:
            checkpoint_path: Path to trained nmr_enc checkpoint
            config_path: Path to nmr_enc config
        """
        # Load the trained model with device mapping for compatibility
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system = NMREncoder.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        self.system.eval()

        # Create a modified forward pass to extract embeddings before regression head
        self.original_model = self.system.model

        # Extract components we need
        self.token_embed = self.original_model.token_embed
        self.transformer = self.original_model.transformer
        self.dropout = self.original_model.dropout
        # Note: NOT using the regressor - that's what we want to extract features BEFORE

        # Move to GPU and enable optimizations
        self.token_embed = self.token_embed.to(device)
        self.transformer = self.transformer.to(device)
        self.dropout = self.dropout.to(device)
        self.device = device

        # Enable inference optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        print(f"Loaded NMR encoder model from {checkpoint_path}")
        print(
            f"Feature extractor ready - outputs {self.original_model.regressor.in_features}-dim vectors"
        )
        print(f"Using device: {device}")

    def extract_features_batch(
        self, input_ids, attention_mask, scalar_values, scalar_mask
    ):
        """
        Extract embeddings from a batch just before the regression head.

        Args:
            input_ids: torch.Tensor of shape (B, seq_len)
            attention_mask: torch.Tensor of shape (B, seq_len)
            scalar_values: torch.Tensor of shape (B, seq_len)
            scalar_mask: torch.Tensor of shape (B, seq_len)

        Returns:
            torch.Tensor of shape (B, d_model) - embeddings before regression
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Replicate the forward pass of NMRTransformerRegressor up to the regression head

            # Token embeddings
            x = self.token_embed(input_ids)  # [B, seq_len, d_model]

            # Create padding mask (True for padding tokens)
            padding_mask = ~attention_mask.bool()

            # Reshape scalar values to have 1 feature dimension
            scalars = scalar_values.unsqueeze(-1)  # [B, seq_len, 1]

            # Transformer (with scalar embeddings and positional encoding)
            x = self.transformer(
                x,
                src_key_padding_mask=padding_mask,
                scalars=scalars,
                scalar_mask=scalar_mask,
            )  # [B, seq_len, d_model]

            # Global average pooling (ignoring padding) - same as in original model
            mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # [B, d_model]

            # Apply dropout (matching original model)
            x = self.dropout(x)  # [B, d_model]

            # THIS IS WHERE WE STOP - we don't apply the regressor
            # x = self.regressor(x)  # [B, num_classes] <- NOT THIS

        return x.float()  # [B, d_model] - these are our features!

    def extract_from_dataset(self, dataset, output_path, max_samples=None):
        """
        Extract features from NMR dataset in sequential order to maintain traceability.

        Args:
            dataset: NMR Dataset (not DataLoader to maintain order)
            output_path: Path to save extracted features (.npy)
            max_samples: Maximum number of samples to process (None for all)
        """
        # Use entire dataset if max_samples not specified
        total_samples = min(max_samples or len(dataset), len(dataset))

        print(f"Processing {total_samples} NMR samples in sequential order...")
        print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        # Optimized batch size for transformer memory and throughput
        batch_size = 64  # Increased for better GPU utilization

        # Get feature dimension from the model
        feature_dim = self.original_model.regressor.in_features  # d_model = 256

        # Pre-allocate numpy array for all features
        all_features = np.zeros((total_samples, feature_dim), dtype=np.float32)

        # Get tokenizer from system
        tokenizer = self.system.tokenizer

        # Debug: Check vocabulary size compatibility
        model_vocab_size = self.original_model.token_embed.num_embeddings
        tokenizer_vocab_size = tokenizer.get_vocab_size()
        print(f"Model embedding layer vocab size: {model_vocab_size}")
        print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

        if model_vocab_size != tokenizer_vocab_size:
            print(
                f"WARNING: Vocabulary size mismatch! This could cause CUDA indexing errors."
            )
            print(f"Using model's vocab size ({model_vocab_size}) for bounds checking.")

        # Process samples sequentially with batching
        for i in tqdm(
            range(0, total_samples, batch_size), desc="Extracting NMR features"
        ):
            batch_end = min(i + batch_size, total_samples)
            batch_indices = list(range(i, batch_end))

            # Load batch data
            batch_input_ids = []
            batch_attention_masks = []
            batch_scalar_values = []
            batch_scalar_masks = []

            for idx in batch_indices:
                sample = dataset[idx]

                # Tokenize H and C NMR text separately (tokenizer handles combination internally)
                tokenized = tokenizer.tokenize(
                    sample["h_nmr_text"], sample["c_nmr_text"]
                )

                # Check for invalid token IDs that could cause CUDA indexing errors
                input_ids = tokenized["input_ids"]

                # Use model's vocabulary size for bounds checking (not tokenizer's)
                # Clamp any invalid token IDs to valid range
                input_ids = [
                    min(max(token_id, 0), model_vocab_size - 1)
                    for token_id in input_ids
                ]

                batch_input_ids.append(torch.tensor(input_ids))
                batch_attention_masks.append(torch.tensor(tokenized["attention_mask"]))
                batch_scalar_values.append(torch.tensor(tokenized["scalar_values"]))
                batch_scalar_masks.append(torch.tensor(tokenized["scalar_mask"]))

            # Stack tensors and move to device (no padding needed - tokenizer already handles it)
            input_ids = torch.stack(batch_input_ids).to(self.device, non_blocking=True)
            attention_mask = torch.stack(batch_attention_masks).to(
                self.device, non_blocking=True
            )
            scalar_values = torch.stack(batch_scalar_values).to(
                self.device, non_blocking=True
            )
            scalar_mask = torch.stack(batch_scalar_masks).to(
                self.device, non_blocking=True
            )

            # Clear CPU tensors to save memory
            del (
                batch_input_ids,
                batch_attention_masks,
                batch_scalar_values,
                batch_scalar_masks,
            )

            # Extract features
            features = self.extract_features_batch(
                input_ids, attention_mask, scalar_values, scalar_mask
            )
            features_np = features.cpu().numpy()

            # Clear GPU tensors to save memory
            del input_ids, attention_mask, scalar_values, scalar_mask, features
            torch.cuda.empty_cache()  # Free GPU memory

            # Write to pre-allocated array
            all_features[i:batch_end] = features_np

        # Save as numpy array
        np.save(output_path, all_features)

        print(
            f"Extracted {total_samples} feature vectors ({feature_dim}-dim) to {output_path}"
        )
        print(f"Order maintained: features[i] = encoding of dataset[i]")
        print(f"Usage: features = np.load('{output_path}')")
        return output_path


def run_extraction():
    """Run NMR feature extraction for nmr_enc/sim_nmr setup."""
    # Set checkpoint path as specified
    checkpoint_path = "/home/rudra/spectro/logs/nmr_enc/sim_nmr/nmr_enc/checkpoints/jirvis-epoch=35-best_acc=0.0000.ckpt"

    # Initialize feature extractor
    extractor = NMRFeatureExtractor(checkpoint_path)

    # Use the tokenizer that was saved with the trained model
    print("Using tokenizer from trained model (saved with checkpoint)")
    tokenizer = extractor.system.tokenizer

    # Create dataset for feature extraction (matching sim_nmr config)
    dataset = SimpleNMRDataset(
        h5_file_path="/home/rudra/Spectro/downloaded/data/final_data/nmr_text/nmr_text_data_with_hall_kier.h5",
        tokenizer=tokenizer,
        indices=None,  # Use all samples
        target_type="hall_kier",  # Match training config
        phase="full",
    )

    # Extract features from entire dataset
    output_path = "/home/rudra/Spectro/downloaded/data/final_data/nmr_text/nmr_embeddings_pytorchL.npy"
    extractor.extract_from_dataset(
        dataset, output_path, max_samples=None  # Process entire dataset
    )


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(config: DictConfig):
    run_extraction()


if __name__ == "__main__":
    main()
