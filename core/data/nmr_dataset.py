import h5py
from torch.utils.data import Dataset
import numpy as np
from typing import *
import time
from tqdm import tqdm


class SimpleNMRDataset(Dataset):
    """
    Simple PyTorch Dataset for NMR text data

    Args:
        h5_file_path: Path to enhanced H5 file
        tokenizer: NMRTokenizer instance
        indices: Specific indices to use (for train/val/test splits)
        target_type: 'binary' or 'numerical'
    """

    def __init__(
        self,
        h5_file_path: str,
        tokenizer,
        indices: Optional[np.ndarray] = None,
        target_type: str = "binary",
        phase: str = "base",
    ):
        self.h5_file_path = h5_file_path
        self.tokenizer = tokenizer
        self.target_type = target_type
        self.phase = phase

        print(f"Loading dataset from: {h5_file_path}")
        print(f"Target type: {target_type}")

        # Cache all samples in memory
        self.samples = self._load_samples_to_memory(indices)

        print(f"Dataset loaded with {len(self.samples):,} samples cached in RAM")

    def _load_samples_to_memory(
        self, indices: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Load all samples into memory from HDF5 file"""
        samples = []

        print("Reading samples from HDF5 file...")
        start_time = time.time()

        with h5py.File(self.h5_file_path, "r") as f:
            mol_ids = list(f["molecules"].keys())

            # Use provided indices or all
            if indices is not None:
                selected_mol_ids = [mol_ids[i] for i in indices]
            else:
                selected_mol_ids = mol_ids

            # Load each sample
            for mol_id in tqdm(
                selected_mol_ids,
                total=len(selected_mol_ids),
                desc=f"Preprocessing {self.phase} Text...",
            ):
                mol = f["molecules"][mol_id]

                # Extract data
                h_nmr_text = mol.attrs.get("h_nmr_text", "")
                c_nmr_text = mol.attrs.get("c_nmr_text", "")

                # Get targets based on target_type
                if self.target_type == "hall_kier":
                    target_key = "hall_kier"
                    # 1) fail if the attribute is missing
                    if target_key not in mol.attrs:
                        raise ValueError(
                            f"Missing `{target_key}` for molecule {mol_id}"
                        )
                    targets = mol.attrs[target_key]
                    # 2) fail if the length isn’t exactly 15
                    if len(targets) != 15:
                        raise ValueError(
                            f"`{target_key}` for molecule {mol_id} must be length 15, "
                            f"but got {len(targets)}. Hall-Kier targets should be a 15-element vector."
                        )
                else:
                    target_key = f"{self.target_type}_targets"
                    if target_key not in mol.attrs:
                        raise ValueError(
                            f"Missing `{target_key}` for molecule {mol_id}"
                        )
                    targets = mol.attrs[target_key]

                # Convert to appropriate types
                if isinstance(h_nmr_text, bytes):
                    h_nmr_text = h_nmr_text.decode("utf-8")
                if isinstance(c_nmr_text, bytes):
                    c_nmr_text = c_nmr_text.decode("utf-8")

                sample = {
                    "h_nmr_text": h_nmr_text,
                    "c_nmr_text": c_nmr_text,
                    "targets": targets.copy(),  # Make a copy to avoid reference issues
                }

                samples.append(sample)

        load_time = time.time() - start_time
        print(f"Loaded {len(samples):,} samples in {load_time:.2f} seconds")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample - now just returns from cached memory"""
        return self.samples[idx]


# class SimpleNMRDataset(Dataset):
#     """
#     Simple PyTorch Dataset for NMR text data

#     Args:
#         h5_file_path: Path to enhanced H5 file
#         tokenizer: NMRTokenizer instance
#         indices: Specific indices to use (for train/val/test splits)
#         target_type: 'binary' or 'numerical'
#     """

#     def __init__(
#         self,
#         h5_file_path: str,
#         tokenizer,
#         indices: Optional[np.ndarray] = None,
#         target_type: str = 'binary'
#     ):
#         self.h5_file_path = h5_file_path
#         self.tokenizer = tokenizer
#         self.target_type = target_type

#         # Get the molecule list once
#         with h5py.File(h5_file_path, 'r') as f:
#             self.mol_ids = list(f['molecules'].keys())

#         # Use provided indices or all
#         if indices is not None:
#             self.indices = indices
#         else:
#             self.indices = np.arange(len(self.mol_ids))

#         print(f"Dataset size: {len(self.indices):,} molecules")

#     def __len__(self) -> int:
#         return len(self.indices)

#     def __getitem__(self, idx: int) -> Dict:
#         # Get the actual molecule index
#         mol_idx = self.indices[idx]
#         mol_id = self.mol_ids[mol_idx]

#         # Load data from H5 file
#         with h5py.File(self.h5_file_path, 'r') as f:
#             mol = f['molecules'][mol_id]
#             h_nmr_text = mol.attrs.get('h_nmr_text', '')
#             c_nmr_text = mol.attrs.get('c_nmr_text', '')

#             # Get targets based on target_type
#             if self.target_type == 'hall_kier':
#                 target_key = 'hall_kier'
#                 # 1) fail if the attribute is missing
#                 if target_key not in mol.attrs:
#                     raise ValueError(f"Missing `{target_key}` for molecule {mol_id}")
#                 targets = mol.attrs[target_key]
#                 # 2) fail if the length isn’t exactly 15
#                 if len(targets) != 15:
#                     raise ValueError(
#                         f"`{target_key}` for molecule {mol_id} must be length 15, "
#                         f"but got {len(targets)}. Hall-Kier targets should be a 15-element vector."
#                     )
#             else:
#                 target_key = f'{self.target_type}_targets'
#                 if target_key not in mol.attrs:
#                     raise ValueError(f"Missing `{target_key}` for molecule {mol_id}")
#                 targets = mol.attrs[target_key]

#         return {
#             'h_nmr_text': h_nmr_text,
#             'c_nmr_text': c_nmr_text,
#             'targets': targets
#         }
