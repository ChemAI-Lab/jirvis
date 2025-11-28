import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class MultiModalDataset(Dataset):
    # Class-level cache for shared data
    _shared_data_cache = {}
    
    def __init__(
        self,
        images_path,
        spectro_h5_path,
        jirvis_embeddings_path,
        nmr_text_embeddings_path,
        total_samples,
        is_grayscale=True,
        use_images=False,
        use_text=False,
        nmr_model_dim=256,
        phase="base",
        indices=None,
        shared_data=None,
    ):
        """
        Multi-modal dataset with IR images and NMR text.

        Args:
            images_path (str): Path to the memory-mapped images file.
            spectro_h5_path (str): Path to the tokenized spectroscopy HDF5 file.
            total_samples (int): Total number of samples in the dataset.
            is_grayscale (bool): Whether the images are grayscale or RGB.
            indices (np.ndarray, optional): Subset of indices to use. If None, uses all samples.
            shared_data (dict, optional): Pre-loaded shared data to avoid redundant loading.
        """
        self.is_grayscale = is_grayscale
        self.total_samples = total_samples
        self.indices = indices
        self.images_path = images_path
        self.spectro_h5_path = spectro_h5_path
        self.jirvis_embeddings_path = jirvis_embeddings_path
        self.nmr_text_embeddings_path = nmr_text_embeddings_path
        self.use_images = use_images
        self.use_text = use_text
        self.nmr_model_dim = nmr_model_dim
        self.phase = phase
        
        # Use shared data if provided, otherwise load/cache data
        cache_key = f"{spectro_h5_path}_{jirvis_embeddings_path}_{nmr_text_embeddings_path}_{use_images}_{use_text}"
        
        if shared_data is not None:
            # Use pre-loaded shared data (preferred for multiprocessing)
            self._use_shared_data(shared_data)
            print(f"MultiModalDataset using pre-loaded shared data for {phase} phase")
        elif cache_key in self._shared_data_cache:
            # Use cached data
            self._use_shared_data(self._shared_data_cache[cache_key])
            print(f"MultiModalDataset using cached data for {phase} phase")
        else:
            # Load data and cache it
            self._load_data_and_cache(cache_key)
            print(f"MultiModalDataset loaded and cached data for {phase} phase")
            
        if self.use_images and not hasattr(self, 'transform'):
            self._get_augmementation()
        
        #print(f"MultiModalDataset initialized in {phase} phase")

    def _use_shared_data(self, shared_data):
        """Use pre-loaded shared data"""
        if self.use_images:
            self.images = shared_data['images']
        else:
            self.jirvis_embeddings = shared_data['jirvis_embeddings']
            
        if not self.use_text:
            self.nmr_embeddings = shared_data['nmr_embeddings']
            
        # Text data
        self.h_nmr_texts = shared_data['h_nmr_texts']
        self.c_nmr_texts = shared_data['c_nmr_texts']
        self.selfies = shared_data['selfies']
        self.tokenized_selfies = shared_data['tokenized_selfies']
        
    def _load_data_and_cache(self, cache_key):
        """Load data and cache it for sharing between datasets"""
        shared_data = {}
        
        if self.use_images:
            self._get_images()
            shared_data['images'] = self.images
        else:
            self._get_jirvis_embeddings()
            shared_data['jirvis_embeddings'] = self.jirvis_embeddings

        if not self.use_text:
            self._get_nmr_embeddings()
            shared_data['nmr_embeddings'] = self.nmr_embeddings

        self._load_spectro_data()
        shared_data['h_nmr_texts'] = self.h_nmr_texts
        shared_data['c_nmr_texts'] = self.c_nmr_texts
        shared_data['selfies'] = self.selfies
        shared_data['tokenized_selfies'] = self.tokenized_selfies
        
        # Cache the shared data
        self._shared_data_cache[cache_key] = shared_data

    def _get_augmementation(self):
        transforms = []
        # Add normalization based on model type
        if self.is_grayscale:
            # Use grayscale normalization (single channel)
            transforms.append(A.Normalize(mean=[0.485], std=[0.229]))
        else:
            # Use RGB normalization
            transforms.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        # Add ToTensorV2 to convert to PyTorch tensor
        transforms.append(ToTensorV2())

        self.transform = A.Compose(transforms)

    def _get_nmr_embeddings(self):
        # Load NMR embeddings into RAM for faster access
        #print(f"Loading NMR embeddings into RAM from {self.nmr_text_embeddings_path}")
        memmap_embeddings = np.memmap(
            self.nmr_text_embeddings_path,
            dtype=np.float32,
            mode="r",
            shape=(self.total_samples, self.nmr_model_dim),
        )
        # Copy to RAM for faster access
        self.nmr_embeddings = np.array(memmap_embeddings)
        print(f"NMR embeddings loaded: shape {self.nmr_embeddings.shape}, size: {self.nmr_embeddings.nbytes / 1e9:.2f} GB")

    def _load_spectro_data(self):
        """Load all NMR data into memory for fast access."""
        with h5py.File(self.spectro_h5_path, "r") as f:
            dataset = f["dataset"]

            # Load text data as lists of strings
            self.h_nmr_texts = [
                text.decode("utf-8") for text in dataset["h_nmr_text"][:]
            ]
            self.c_nmr_texts = [
                text.decode("utf-8") for text in dataset["c_nmr_text"][:]
            ]
            self.selfies = [text.decode("utf-8") for text in dataset["selfies"][:]]

            # Load tokenized SELFIES as numpy array for direct torch conversion
            self.tokenized_selfies = dataset["tokenized_selfies"][:].astype(np.int64)

            # Verify order matches image_index
            image_indices = dataset["image_index"][:]
            assert np.array_equal(
                image_indices, np.arange(len(image_indices))
            ), "Image indices are not in sequential order!"

    def _get_images(self):
        """Load memory-mapped images."""
        if self.is_grayscale:
            self.images = np.memmap(
                self.images_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.total_samples, 512, 512),
            )

        else:
            self.images = np.memmap(
                self.images_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.total_samples, 512, 512, 3),
            )

    def _get_jirvis_embeddings(self):
        # Load Jirvis embeddings into RAM for faster access
        #print(f"Loading Jirvis embeddings into RAM from {self.jirvis_embeddings_path}")
        memmap_embeddings = np.memmap(
            self.jirvis_embeddings_path,
            dtype=np.float32,
            mode="r",
            shape=(self.total_samples, 2048),
        )
        # Copy to RAM for faster access
        self.jirvis_embeddings = np.array(memmap_embeddings)
        #print(f"Jirvis embeddings loaded: shape {self.jirvis_embeddings.shape}, size: {self.jirvis_embeddings.nbytes / 1e9:.2f} GB")

    def __getitem__(self, idx):
        """
        Get a single sample with all modalities.

        Returns:
            dict: {
                'image': torch.Tensor,      # Shape:  (3, 512, 512), float32
                'h_nmr_text': str,          # H-NMR text
                'c_nmr_text': str,          # C-NMR text
                'tokenized_selfies': torch.Tensor  # Shape: (77,) int64
            }
        """
        # If using indices, map the idx to the actual data index
        if self.indices is not None:
            if idx >= len(self.indices):
                raise IndexError(
                    f"Index {idx} out of bounds for dataset subset with size {len(self.indices)}"
                )
            actual_idx = self.indices[idx]
        else:
            if idx >= self.total_samples:
                raise IndexError(
                    f"Index {idx} out of bounds for dataset with size {self.total_samples}"
                )
            actual_idx = idx

        sample = {
            "selfie": self.selfies[actual_idx],
            "tokenized_selfies": torch.from_numpy(self.tokenized_selfies[actual_idx]),
        }
        ##### WHEN DOING NMR EMBEDDINGS,ENSURE THE KEY IS 'nmr_embedding' #####
        if self.use_images:
            # Get image
            image = self.images[actual_idx]

            # Handle grayscale: convert to 3-channel
            if self.is_grayscale:
                image = self.grayscale_to_3_channel(image)

            sample["image"] = self.transform(image=image)[
                "image"
            ]  # (3, 512, 512), float32

        else:
            sample["jirvis_embedding"] = torch.from_numpy(self.jirvis_embeddings[actual_idx])

        if self.use_text:
            sample["h_nmr_text"] = self.h_nmr_texts[actual_idx]
            sample["c_nmr_text"] = self.c_nmr_texts[actual_idx]
        else:
            # If not using text, return the NMR embeddings
            sample["nmr_embedding"] = torch.from_numpy(self.nmr_embeddings[actual_idx])
        return sample

    def grayscale_to_3_channel(self, image):
        """
        Convert a single-channel grayscale image to 3-channel RGB.

        Args:
            image: numpy array of shape (H, W, 1)

        Returns:
            numpy array of shape (H, W, 3)
        """
        image = np.expand_dims(image, axis=-1)  # (H,W,1)
        return np.repeat(image, 3, axis=-1)  # (H,W,3)

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.total_samples

    def get_vocab_size(self):
        """Get vocabulary size for model configuration."""
        with h5py.File(self.spectro_h5_path, "r") as f:
            return f["meta"].attrs["num_tokens"]

    def get_max_seq_length(self):
        """Get maximum sequence length for model configuration."""
        with h5py.File(self.spectro_h5_path, "r") as f:
            return f["meta"].attrs["max_length"]

    def get_idx_to_token(self):
        """Get index to token mapping."""
        with h5py.File(self.spectro_h5_path, "r") as f:
            json_str = f["meta"].attrs["idx_to_token"]
            return json.loads(json_str)
