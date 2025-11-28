from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import os
import torch


class ImageDataset(Dataset):
    def __init__(
        self,
        images_path,
        labels_path,
        label_indices,
        use_catch_all=True,
        is_grayscale=True,
    ):
        self.label_indices = label_indices
        self.use_catch_all = use_catch_all
        self.is_grayscale = is_grayscale

        # Calculate total samples from the labels file
        total_samples = os.path.getsize(labels_path) // (
            24 * np.dtype("uint8").itemsize
        )
        print(f"Total samples in dataset: {total_samples}")

        # Load data with calculated shape - now handling grayscale
        if is_grayscale:
            # For grayscale images (no channel dimension)
            self.images = np.memmap(
                images_path, dtype=np.uint8, mode="r", shape=(total_samples, 512, 512)
            )
        else:
            # For RGB images (with channel dimension)
            self.images = np.memmap(
                images_path,
                dtype=np.uint8,
                mode="r",
                shape=(total_samples, 512, 512, 3),
            )

        self.labels = np.memmap(
            labels_path, dtype=np.uint8, mode="r", shape=(total_samples, 24)
        )

        # If not using catch-all, filter out unlabeled data
        if not use_catch_all:
            valid_indices = []
            for idx in range(len(self.labels)):
                if np.sum(self.labels[idx][self.label_indices]) > 0:
                    valid_indices.append(idx)
            self.valid_indices = np.array(valid_indices)
        else:
            self.valid_indices = np.arange(len(self.labels))

        self.len = len(self.valid_indices)
        print(f"ImageDataset initialized with {self.len} valid indices")
        print(f"Images shape: {self.images.shape}, Grayscale mode: {is_grayscale}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with size {self.len}"
            )

        actual_idx = self.valid_indices[idx]
        image = self.images[actual_idx]

        # Handle grayscale images: add channel dimension if needed
        if self.is_grayscale:
            # Add a channel dimension for transformations (H,W) -> (H,W,1)
            image = np.expand_dims(image, axis=-1)
            # Convert to pseudo-RGB by duplicating the channel (H,W,1) -> (H,W,3)
            # This makes the image compatible with RGB-based transformations
            image = np.repeat(image, 3, axis=-1)

        # image = 255 - image  # Invert image

        # Get selected labels
        label = self.labels[actual_idx][self.label_indices]

        # Add catch-all label if enabled
        if self.use_catch_all:
            catch_all = 1 if np.sum(label) == 0 else 0
            label = np.append(label, catch_all)

        return image, label


class TrainDataset(ImageDataset):
    def __init__(
        self, dataset, balanced_indices, model_name="ResNet50", use_catch_all=True
    ):
        """
        Training dataset with data augmentation.

        Args:
            dataset: Base dataset instance
            balanced_indices: Indices for balanced sampling
            model_name: Name of the model architecture
            use_catch_all: Whether to use catch-all label for unlabeled data
        """
        self.base_dataset = dataset
        self.balanced_indices = balanced_indices
        self.use_catch_all = use_catch_all
        self.is_grayscale = dataset.is_grayscale
        # Create transformations list based on model type
        transforms = []
        if model_name == "ViT":
            print("Resizing Train Images to 224x224 (VIT Chosen)")
            transforms.append(A.Resize(224, 224))

        transforms.extend(
            [
                A.Rotate(limit=10, p=0.4),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    ],
                    p=0.4,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.4
                ),
                A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.4),
            ]
        )

        # Add normalization based on model type
        if self.is_grayscale:
            # Use grayscale normalization (single channel)
            # These values are approximations - consider calculating exact values from your dataset
            transforms.append(A.Normalize(mean=[0.485], std=[0.229]))
        else:
            # Use RGB normalization
            transforms.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        # Add ToTensorV2 to convert to PyTorch tensor
        transforms.append(ToTensorV2())

        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        actual_idx = self.balanced_indices[idx]
        image, label = self.base_dataset.__getitem__(actual_idx)

        transformed = self.transform(image=image)
        image = transformed["image"]

        # No need to permute channels as ToTensorV2 handles this
        return image, torch.from_numpy(label).float()

        return (
            torch.from_numpy(image).permute(2, 0, 1).float(),
            torch.from_numpy(label).float(),
        )


class TestDataset(ImageDataset):
    def __init__(
        self, dataset, balanced_indices, model_name="ResNet50", use_catch_all=True
    ):
        """
        Test/Validation dataset without augmentation.

        Args:
            dataset: Base dataset instance
            balanced_indices: Indices for balanced sampling
            model_name: Name of the model architecture
            use_catch_all: Whether to use catch-all label for unlabeled data
        """
        self.base_dataset = dataset
        self.balanced_indices = balanced_indices
        self.use_catch_all = use_catch_all
        self.is_grayscale = dataset.is_grayscale

        transforms = []
        if model_name == "ViT":
            print("Resizing Test/Val Images to 224x224 (VIT Chosen)")
            transforms.append(A.Resize(224, 224))

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

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        actual_idx = self.balanced_indices[idx]
        image, label = self.base_dataset.__getitem__(actual_idx)

        transformed = self.transform(image=image)
        image = transformed["image"]

        # No need to permute channels as ToTensorV2 handles this
        return image, torch.from_numpy(label).float()
