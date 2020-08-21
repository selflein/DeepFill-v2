from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from iminpaint.data.dataloader import data_utils, datasets


def create_train_val_loader(
        path: Path,
        edges_path: Path,
        transform=data_utils.regular_data_transform(),
        transform_val=ToTensor(),
        train_percentage=0.85,
        shuffle=True,
        batch_size=16,
        num_workers=0,
        pin_memory=True) \
        -> (DataLoader, DataLoader):
    """Create train and validation dataloader from single image directory.

    Args:
        path: Path to image folder.
        edges_path: Path to folder where edge masks are stored.
        transform: Torchvision data transformation.
        transform_val: Torchvision data transformation for validation data.
        train_percentage: Percentage of data to use for training.
        shuffle: Shuffle data indices.
        batch_size: Number of samples in batch.
        num_workers: Number of workers in DataLoader.
        pin_memory: Whether to pin memory for DataLoaders.

    Returns:
        Train and validation DataLoader.
    """
    all_img_paths = list(path.glob('*.png'))

    num_train = len(all_img_paths)
    indices = list(range(num_train))
    split = int(np.floor(train_percentage * num_train))

    if shuffle:
        np.random.seed(1)
        np.random.shuffle(indices)

    train_dataset = datasets.InpaintingDataset(
        all_img_paths[:split],
        edge_masks_folder=edges_path,
        transforms=transform
    )

    val_dataset = datasets.InpaintingDataset(
        all_img_paths[split:],
        edge_masks_folder=edges_path,
        transforms=transform_val
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
