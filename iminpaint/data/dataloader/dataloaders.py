from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, sampler

from iminpaint.data.dataloader import data_utils


def create_train_val_loader_from_single_folder(
        path: Path,
        transform=data_utils.regular_data_transform(),
        train_percentage=0.85,
        shuffle=True,
        batch_size=16,
        num_workers=4,
        pin_memory=True) \
        -> (DataLoader, DataLoader):
    """Create train and validation dataloader from single image directory.
    
    Args:
        path: Path to image folder.
        transform: Torchvision data transformation.
        train_percentage: Percentage of data to use for training.
        shuffle: Shuffle data indices.
        batch_size: Number of samples in batch.
        num_workers: Number of workers in DataLoader.
        pin_memory: Whether to pin memory for DataLoaders.

    Returns:
        Train and validation DataLoader.
    """
    train_dataset = ImageFolder(str(path), transform=transform)
    val_dataset = ImageFolder(str(path), transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(train_percentage * num_train))

    if shuffle:
        np.random.seed(1)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler
    )
    return train_loader, val_loader
