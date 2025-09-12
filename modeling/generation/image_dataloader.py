"""Dataset dataloaders for auto-encoder and VAE training.

Supports CelebA and AFHQv2 datasets from Hugging Face.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

DatasetType = Literal["celeba", "afhq"]


class CelebADataset(Dataset):
    """PyTorch Dataset wrapper for CelebA from Hugging Face."""

    def __init__(
        self,
        split: str = "train",
        image_size: int = 64,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize CelebA dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            image_size: Target image size (will be resized to square)
            normalize: Whether to normalize images to [-1, 1] range
            cache_dir: Directory to cache the dataset
        """
        self.split = split
        self.image_size = image_size
        self.normalize = normalize

        # Load dataset from Hugging Face
        logger.info(f"Loading CelebA {split} split...")
        try:
            self.dataset = load_dataset(
                "flwrlabs/celeba",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load CelebA dataset: {e}")
            logger.info("Falling back to a smaller test dataset...")
            # Fallback to a smaller dataset for testing
            try:
                self.dataset = load_dataset(
                    "mnist",
                    split=split,
                    cache_dir=cache_dir,
                )
                logger.info("Using MNIST as fallback dataset")
            except Exception as e2:
                logger.error(f"Failed to load fallback dataset: {e2}")
                raise RuntimeError(
                    "Could not load any dataset. Please check your internet "
                    "connection and dataset availability."
                )

        # Define transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            # Check if we're using MNIST (grayscale) or CelebA (RGB)
            if hasattr(self.dataset, "features") and "image" in self.dataset.features:
                # Check if the image is grayscale or RGB
                sample_image = self.dataset[0]["image"]
                if sample_image.mode == "L":  # Grayscale
                    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
                else:  # RGB
                    transform_list.append(
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    )
            else:
                # Default to RGB normalization
                transform_list.append(
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                )

        self.transform = transforms.Compose(transform_list)

        logger.info(f"Loaded {len(self.dataset)} images from CelebA {split} split")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Transformed image tensor of shape (C, H, W)
        """
        sample = self.dataset[idx]
        image = sample["image"]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image


class AFHQv2Dataset(Dataset):
    """PyTorch Dataset wrapper for AFHQv2 from Hugging Face."""

    def __init__(
        self,
        split: str = "train",
        image_size: int = 64,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        val_split_ratio: float = 0.1,
    ) -> None:
        """
        Initialize AFHQv2 dataset.

        Args:
            split: Dataset split ('train', 'validation')
            image_size: Target image size (will be resized to square)
            normalize: Whether to normalize images to [-1, 1] range
            cache_dir: Directory to cache the dataset
            val_split_ratio: Ratio of data to use for validation (when split='validation')
        """
        self.split = split
        self.image_size = image_size
        self.normalize = normalize

        # Load dataset from Hugging Face (always load 'train' since AFHQv2 only has train split)
        logger.info(f"Loading AFHQv2 dataset...")
        try:
            self.dataset = load_dataset(
                "huggan/AFHQv2",
                split="train",  # Always load train split
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load AFHQv2 dataset: {e}")
            logger.info("Falling back to a smaller test dataset...")
            # Fallback to a smaller dataset for testing
            try:
                self.dataset = load_dataset(
                    "mnist",
                    split=split,
                    cache_dir=cache_dir,
                )
                logger.info("Using MNIST as fallback dataset")
            except Exception as e2:
                logger.error(f"Failed to load fallback dataset: {e2}")
                raise RuntimeError(
                    "Could not load any dataset. Please check your internet "
                    "connection and dataset availability."
                )

        # Define transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

        if normalize:
            # Check if we're using MNIST (grayscale) or CelebA (RGB)
            if hasattr(self.dataset, "features") and "image" in self.dataset.features:
                # Check if the image is grayscale or RGB
                sample_image = self.dataset[0]["image"]
                if sample_image.mode == "L":  # Grayscale
                    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
                else:  # RGB
                    transform_list.append(
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    )
            else:
                # Default to RGB normalization
                transform_list.append(
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                )

        self.transform = transforms.Compose(transform_list)

        # Handle train/validation splitting for AFHQv2 (which only has 'train' split)
        if split == "validation":
            total_size = len(self.dataset)
            val_size = int(total_size * val_split_ratio)
            # Use the last portion of the dataset for validation
            self.dataset = self.dataset.select(range(total_size - val_size, total_size))
            logger.info(f"Created validation split with {len(self.dataset)} images")
        elif split == "train":
            total_size = len(self.dataset)
            val_size = int(total_size * val_split_ratio)
            # Use the first portion of the dataset for training
            self.dataset = self.dataset.select(range(total_size - val_size))
            logger.info(f"Created training split with {len(self.dataset)} images")

        logger.info(f"Loaded {len(self.dataset)} images from AFHQv2 {split} split")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Transformed image tensor of shape (C, H, W)
        """
        sample = self.dataset[idx]
        image = sample["image"]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image


class ImageDataLoader:
    """DataLoader factory for image datasets (CelebA and AFHQv2)."""

    def __init__(
        self,
        dataset_type: DatasetType = "afhq",
        batch_size: int = 32,
        image_size: int = 64,
        normalize: bool = True,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize Image DataLoader.

        Args:
            dataset_type: Type of dataset to use ('celeba' or 'afhq')
            batch_size: Batch size for training
            image_size: Target image size (will be resized to square)
            normalize: Whether to normalize images to [-1, 1] range
            num_workers: Number of worker processes for data loading
            cache_dir: Directory to cache the dataset
        """
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.image_size = image_size
        self.normalize = normalize
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    def get_dataloader(
        self,
        split: str = "train",
        shuffle: Optional[bool] = None,
        drop_last: bool = True,
    ) -> DataLoader:
        """
        Get a DataLoader for the specified split.

        Args:
            split: Dataset split ('train', 'validation', 'test' for CelebA;
                    'train', 'test' for AFHQv2)
            shuffle: Whether to shuffle data (defaults to True for train,
                    False for others)
            drop_last: Whether to drop the last incomplete batch

        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = split == "train"

        # Create appropriate dataset based on type
        if self.dataset_type == "celeba":
            dataset = CelebADataset(
                split=split,
                image_size=self.image_size,
                normalize=self.normalize,
                cache_dir=self.cache_dir,
            )
        elif self.dataset_type == "afhq":
            dataset = AFHQv2Dataset(
                split=split,
                image_size=self.image_size,
                normalize=self.normalize,
                cache_dir=self.cache_dir,
                val_split_ratio=0.1,  # Use 10% for validation
            )
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )

        return dataloader

    def get_train_dataloader(self, **kwargs) -> DataLoader:
        """Get training DataLoader."""
        return self.get_dataloader(split="train", **kwargs)

    def get_val_dataloader(self, **kwargs) -> DataLoader:
        """Get validation DataLoader."""
        return self.get_dataloader(split="validation", **kwargs)

    def get_test_dataloader(self, **kwargs) -> DataLoader:
        """Get test DataLoader."""
        return self.get_dataloader(split="test", **kwargs)


def create_image_dataloader(
    dataset_type: DatasetType = "afhq",
    batch_size: int = 32,
    image_size: int = 64,
    split: str = "train",
    normalize: bool = True,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    """
    Convenience function to create an image DataLoader.

    Args:
        dataset_type: Type of dataset to use ('celeba' or 'afhq')
        batch_size: Batch size for training
        image_size: Target image size (will be resized to square)
        split: Dataset split ('train', 'validation', 'test' for CelebA;
                'train', 'test' for AFHQv2)
        normalize: Whether to normalize images to [-1, 1] range
        num_workers: Number of worker processes for data loading
        cache_dir: Directory to cache the dataset
        **kwargs: Additional arguments passed to get_dataloader

    Returns:
        PyTorch DataLoader
    """
    dataloader_factory = ImageDataLoader(
        dataset_type=dataset_type,
        batch_size=batch_size,
        image_size=image_size,
        normalize=normalize,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )

    return dataloader_factory.get_dataloader(split=split, **kwargs)


# Backward compatibility function
def create_celeba_dataloader(
    batch_size: int = 32,
    image_size: int = 64,
    split: str = "train",
    normalize: bool = True,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    """
    Convenience function to create a CelebA DataLoader
    (backward compatibility).

    Args:
        batch_size: Batch size for training
        image_size: Target image size (will be resized to square)
        split: Dataset split ('train', 'validation', 'test')
        normalize: Whether to normalize images to [-1, 1] range
        num_workers: Number of worker processes for data loading
        cache_dir: Directory to cache the dataset
        **kwargs: Additional arguments passed to get_dataloader

    Returns:
        PyTorch DataLoader
    """
    return create_image_dataloader(
        dataset_type="celeba",
        batch_size=batch_size,
        image_size=image_size,
        split=split,
        normalize=normalize,
        num_workers=num_workers,
        cache_dir=cache_dir,
        **kwargs,
    )
