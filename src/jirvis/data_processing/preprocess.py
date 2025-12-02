"""
Module: data_loader

Output:
    - List[str]: List of image file paths, e.g., ['/path/to/image1.png', '/path/to/image2.png']
    - List[np.ndarray]: List of cropped images as NumPy arrays, e.g., [array(shape=(354, 715, 3), dtype=uint8), ...]
    - List[List[str]]: List of labels (functional groups) for each image, e.g., [['alkene'], ['alkyne', 'ketone']]

Usage:
    # Import DataLoader
    from data_loader import DataLoader, get_loader

    # Create an instance of DataLoader
    loader = get_loader(
        json_path="/path/to/selfies.json",
        image_base_path="/path/to/spectra_images/IR"
    )

    # Load data
    image_paths, images, labels = loader.get_data()

    # Output results
    print(f"Total images: {len(images)}")
    print(f"First image shape: {images[0].shape if images else 'No images'}")
"""

import json
import time
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import gc

# ------------------------------------------------------------- #
#                     DATA LOADER CLASS
# ------------------------------------------------------------- #


class DataLoader:
    """
    A class to load and preprocess spectroscopic image data.

    Attributes:
        json_path (str): Path to the JSON file containing metadata.
        image_base_path (str): Path to the folder containing spectroscopic images.
    """

    # ------------------------------------------------------------- #
    #                   INITIALIZATION AND SETUP
    # ------------------------------------------------------------- #

    def __init__(self, json_path: str, image_base_path: str, fg_key_path: str):
        """
        Initialize the DataLoader with metadata and image base path.

        Parameters:
            json_path (str): Path to the JSON file containing metadata.
            image_base_path (str): Path to the folder containing spectroscopic images.
        """
        self.json_path = json_path
        self.image_base_path = image_base_path
        self.data = self._load_json(json_path)
        self.fg_keys = self._load_json(fg_key_path)

    # ------------------------------------------------------------- #
    #                   PRIVATE HELPER FUNCTIONS
    # ------------------------------------------------------------- #

    def _load_json(self, json_path) -> List[Dict]:
        """
        Load metadata from the JSON file.

        Returns:
            List[Dict]: A list of metadata entries.
        """
        with open(json_path, "r") as f:
            return json.load(f)

    def _construct_image_path(self, sdbs_no: str) -> str:
        """
        Construct the file path for an image using its SDBS number.

        Parameters:
            sdbs_no (str): The SDBS number of the image.

        Returns:
            str: The full file path to the image.
        """
        return os.path.join(self.image_base_path, f"{sdbs_no}.png")

    def _load_single_image_old(self, image_path: str) -> np.ndarray:
        """
        Load a single image and crop it by height.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: The cropped image.
        """
        image = Image.open(image_path).convert("RGB")
        image_inv = 255 - np.array(image)
        image_inv = image_inv[90:-110, :]
        image = Image.fromarray(image_inv).resize((512, 512))
        final_image = np.array(image, dtype=np.uint8)

        return final_image

    def _load_single_image(
        self,
        image_path: str,
        crop_top: int = 90,
        crop_bottom: int = 110,
        crop_left: int = 0,
        crop_right: int = 290,
    ) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_inv = 255 - np.array(image)

        height, width, _ = image_inv.shape
        image_cropped = image_inv[
            crop_top : height - crop_bottom, crop_left : width - crop_right
        ]
        image = Image.fromarray(image_inv).resize((512, 512))
        final_image = np.array(image, dtype=np.uint8)
        return final_image

    # ------------------------------------------------------------- #
    #                   PUBLIC DATA LOADER
    # ------------------------------------------------------------- #

    def get_data(
        self,
    ) -> Tuple[List[str], List[np.ndarray], List[List[str]], np.ndarray]:
        """
        Get all image paths, cropped images, and their corresponding labels.

        Returns:
            Tuple containing:
                - List[str]: List of image file paths.
                - List[np.ndarray]: List of cropped images as NumPy arrays.
                - List[List[str]]: List of functional group labels for each image.
        """
        image_paths = []
        images = []
        labels = []

        for item in self.data:
            if "IR" in item["available_specs"]:
                sdbs_no = item["sdbs_no"]
                image_path = self._construct_image_path(sdbs_no)

                if os.path.exists(image_path):
                    try:
                        # Load and crop the image
                        image = self._load_single_image(image_path)
                        image_paths.append(image_path)
                        images.append(image)
                        labels.append(item["functional_groups"])

                        # Efficient memory management
                        del image
                        gc.collect()

                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        continue

        # Summary of loaded data
        print(f"\nDataset Summary:")
        print(f"Total images loaded: {len(images)}")
        print(
            f"Sample image shape: {images[0].shape if images else 'No images loaded'}"
        )
        print(f"Sample labels: {labels[0] if labels else 'No labels loaded'}")

        # Get multi-label one hot encoding of labels
        multi_label_ohe = []
        for mc_labels in labels:
            ohe_label = [0] * len(self.fg_keys.values())
            if mc_labels is None:
                multi_label_ohe.append(ohe_label)
                continue
            for label in mc_labels:
                val = self.fg_keys[label]
                ohe_label[val] = 1
            multi_label_ohe.append(ohe_label)
        print(f"Length of original labels: {len(labels)}")
        print(f"Length of OHE labels: {len(multi_label_ohe)}")
        return image_paths, images, labels, np.array(multi_label_ohe, dtype=np.uint8)


# ------------------------------------------------------------- #
#                   FACTORY FUNCTION
# ------------------------------------------------------------- #


def get_loader(json_path: str, image_base_path: str, fg_key_path: str) -> DataLoader:
    """
    Factory function to create a DataLoader instance.

    Parameters:
        json_path (str): Path to the JSON file containing metadata.
        image_base_path (str): Path to the folder containing spectroscopic images.

    Returns:
        DataLoader: An instance of the DataLoader class.
    """
    return DataLoader(json_path, image_base_path, fg_key_path)


# ------------------------------------------------------------- #
#                   DataLoader Caching
# ------------------------------------------------------------- #


class CachedDataLoader(DataLoader):
    """
    Extension of DataLoader that implements caching to disk.
    """

    def __init__(
        self,
        json_path: str,
        image_base_path: str,
        fg_key_path: str,
        cache_dir: str = "./cache",
    ):
        """
        Initialize the CachedDataLoader.

        Parameters:
            json_path (str): Path to the JSON file containing metadata
            image_base_path (str): Path to the folder containing spectroscopic images
            fg_key_path (str): Path to the functional group keys JSON
            cache_dir (str): Directory to store cached data
        """
        super().__init__(json_path, image_base_path, fg_key_path)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Cache file paths
        self.images_cache_path = os.path.join(cache_dir, "images.npy")
        self.labels_cache_path = os.path.join(cache_dir, "labels.npy")
        self.metadata_cache_path = os.path.join(cache_dir, "cached_metadata.json")

    def _save_to_cache(
        self,
        image_paths: List[str],
        images: List[np.ndarray],
        labels: List[List[str]],
        multi_label_ohe: np.ndarray,
    ) -> None:
        """
        Save the processed data to cache files.
        """
        assert len(images) == len(
            multi_label_ohe
        ), "Mismatch between images and labels!"

        try:
            # Save images and multi-label OHE
            # with h5py.File(self.images_cache_path, 'w') as h5f:
            #     images_dataset = h5f.create_dataset('images', data=np.stack(images), compression="gzip", compression_opts=4)
            #     images_dataset.attrs['length'] = len(images)

            #     h5f.create_dataset('multi_label_ohe', data=multi_label_ohe, compression="gzip", compression_opts=4)

            ### NEW METHOD ###
            np_images = np.stack(images, dtype=np.uint8)

            np_memmap = np.memmap(
                self.images_cache_path,
                dtype=np_images.dtype,
                mode="w+",
                shape=np_images.shape,
            )
            np_memmap[:] = np_images[:]
            np_memmap.flush()

            multi_label_ohe = multi_label_ohe.astype(np.uint8)

            ohe_memmap = np.memmap(
                self.labels_cache_path,
                dtype=multi_label_ohe.dtype,
                mode="w+",
                shape=multi_label_ohe.shape,
            )
            ohe_memmap[:] = multi_label_ohe[:]
            ohe_memmap.flush()
            del np_memmap
            del ohe_memmap

            print(f"Reading images as dtype: {np_images.dtype}")
            np_memmap = np.memmap(
                self.images_cache_path,
                dtype=np_images.dtype,
                mode="r",
                shape=np_images.shape,
            )
            print("Array1 equal:", np.array_equal(np_images, np_memmap))

            print(f"Reading ohe as dtype: {multi_label_ohe.dtype}")
            ohe_memmap = np.memmap(
                self.labels_cache_path,
                dtype=multi_label_ohe.dtype,
                mode="r",
                shape=multi_label_ohe.shape,
            )
            print("Array2 equal:", np.array_equal(multi_label_ohe, ohe_memmap))

            ### END NEW METHOD ###

            # Save metadata (paths and original labels)
            metadata = {
                "image_paths": image_paths,
                "labels": labels,
                "images_shape": np_images.shape,
                "num_images": len(images),
                "multi_label_ohe_shape": multi_label_ohe.shape,
            }
            with open(self.metadata_cache_path, "w") as f:
                json.dump(metadata, f)

            print(f"Successfully cached {len(images)} images")
            print(f"Multi-label OHE shape: {multi_label_ohe.shape}")
            print(
                f"Cache files: {os.path.basename(self.images_cache_path)}, {os.path.basename(self.metadata_cache_path)}"
            )

        except Exception as e:
            print(f"Error saving to cache: {e}")
            raise

    def _load_from_cache(
        self,
    ) -> Optional[Tuple[List[str], List[np.ndarray], List[List[str]], np.ndarray]]:
        """
        Try to load data from cache if it exists.

        Returns:
            Tuple of (image_paths, images, labels, multi_label_ohe) if cache exists, None otherwise
        """
        if not (
            os.path.exists(self.images_cache_path)
            and os.path.exists(self.metadata_cache_path)
        ):
            return None

        try:
            # Load images and multi-label OHE
            # with np.load(self.images_cache_path) as data:
            #     # Extract images and OHE
            #     # images = data['images']
            #     multi_label_ohe = data['multi_label_ohe']
            # with h5py.File(self.images_cache_path, 'r') as h5f:
            #     # images = h5f['images'][:]
            #     multi_label_ohe = h5f['multi_label_ohe'][:]

            # Load metadata
            with open(self.metadata_cache_path, "r") as f:
                metadata = json.load(f)

            ### NEW METHOD ###
            image_shape = tuple(metadata["images_shape"])
            label_shape = tuple(metadata["multi_label_ohe_shape"])

            images = np.memmap(
                self.images_cache_path, dtype=np.uint8, mode="r", shape=image_shape
            )
            multi_label_ohe = np.memmap(
                self.labels_cache_path, dtype=np.uint8, mode="r", shape=label_shape
            )
            ### END NEW METHOD ###

            return metadata["image_paths"], images, metadata["labels"], multi_label_ohe

        except Exception as e:
            print(f"Error loading from cache: {e}")
            raise

    def get_data(
        self, use_cache: bool = True, update_cache: bool = True
    ) -> Tuple[List[str], List[np.ndarray], List[List[str]], np.ndarray]:
        """
        Get data, either from cache or by processing images.

        Parameters:
            use_cache (bool): Whether to try loading from cache first
            update_cache (bool): Whether to update cache if processing new data

        Returns:
            Tuple containing:
                - List[str]: List of image file paths
                - List[np.ndarray]: List of cropped images as NumPy arrays
                - List[List[str]]: List of functional group labels for each image
                - np.ndarray: Multi-label one-hot encoded array
        """
        # Try loading from cache first
        if use_cache:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                return cached_data

        # If no cache or cache loading failed, process the data
        image_paths, images, labels, multi_label_ohe = super().get_data()

        # Update cache if requested
        if update_cache:
            self._save_to_cache(image_paths, images, labels, multi_label_ohe)

        return image_paths, images, labels, multi_label_ohe


def get_cached_loader(
    json_path: str, image_base_path: str, fg_key_path: str, cache_dir: str = "./cache"
) -> CachedDataLoader:
    """
    Factory function to create a CachedDataLoader instance.
    """
    time_start = time.time()
    tmp = CachedDataLoader(json_path, image_base_path, fg_key_path, cache_dir)
    time_end = time.time()
    print(f"Total Time: {time_end - time_start :.2f}")
    return tmp


def load_data(
    json_path: str,
    image_base_path: str,
    fg_key_path: str,
    cache_dir: str = "./ir_data_cache",
    use_cache: bool = True,
    update_cache: bool = False,
) -> Tuple[List[str], List[np.ndarray], List[List[str]], np.ndarray]:
    """
    Loading function to be used in training.
    """
    loader = get_cached_loader(json_path, image_base_path, fg_key_path, cache_dir)
    return loader.get_data(use_cache=use_cache, update_cache=update_cache)


def load_OHE(LABEL_PATH):
    labels = np.memmap(LABEL_PATH, dtype="uint8", mode="r", shape=(15283, 24))
    return labels[:]


if __name__ == "__main__":
    JSON_PATH = "/home/rudra/Spectro/raw_data/sdbs.json"
    IMAGE_BASE_PATH = "/home/rudra/Spectro/raw_data/spectra_images/IR"
    FG_KEY_PATH = "/home/rudra/Spectro/ir_model/fg_keys.json"
    CACHE_DIR = "./No_FP_cropped_ir_cache"

    # # Create loader

    loader = get_cached_loader(JSON_PATH, IMAGE_BASE_PATH, FG_KEY_PATH, CACHE_DIR)

    # # First run - will process images and create cache
    image_paths, images, labels, multi_label_ohe = loader.get_data(
        use_cache=False, update_cache=True
    )
    print(f"Loaded {len(images)} images")
    print(f"Multi-label OHE shape: {multi_label_ohe.shape}")

    # # Second run - will load from cache
    image_paths, images, labels, multi_label_ohe = loader.get_data(use_cache=True)
    import matplotlib.pyplot as plt

    image_np = images[10]
    plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_np)
    plt.axis("off")
    plt.savefig("./new_image.png")
    print(f"Loaded {len(images)} images from cache")
