import os.path as osp

import pytest
from torch.utils.data import DataLoader
import albumentations as albu

from src.features.segmentation.dataset import SegmentationDataset


@pytest.fixture(scope="session", autouse=True)
def train_bubbles_data_path():
    path = r"..\data\weakly_segmented\bubbles_split\train"
    return path


@pytest.fixture(scope="session", autouse=True)
def val_bubbles_data_path():
    path = r"..\data\weakly_segmented\bubbles_split\valid"
    return path


@pytest.fixture(scope="session", autouse=True)
def train_dataset(train_bubbles_data_path):
    transform = albu.Compose([
        albu.Resize(256, 256)
    ])

    train_dataset = SegmentationDataset(
        images_dir=osp.join(train_bubbles_data_path, "images"),
        masks_dir=osp.join(train_bubbles_data_path, "masks"),
        augmentation_transform=transform
    )

    return train_dataset


@pytest.fixture(scope="session", autouse=True)
def val_dataset(val_bubbles_data_path):
    transform = albu.Compose([
        albu.Resize(256, 256)
    ])

    val_dataset = SegmentationDataset(
        images_dir=osp.join(val_bubbles_data_path, "images"),
        masks_dir=osp.join(val_bubbles_data_path, "masks"),
        augmentation_transform=transform
    )
    return val_dataset


@pytest.fixture(scope="session", autouse=True)
def train_dataloader(train_dataset):
    train_batch_size = 4
    pin_memory = False
    num_workers = 0
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=num_workers
    )
    return train_dataloader


@pytest.fixture(scope="session", autouse=True)
def val_dataloader(val_dataset):
    train_batch_size = 4
    pin_memory = False
    num_workers = 0
    val_dataloader = DataLoader(
        val_dataset, batch_size=train_batch_size, shuffle=False,
        pin_memory=pin_memory, num_workers=num_workers
    )
    return val_dataloader
