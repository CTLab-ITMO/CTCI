"""
Segmentation Dataset Module

This module provides class and functions for working with datasets used in the context of image segmentation tasks.
It includes a custom dataset class, `SegmentationDataset`, for loading image and mask pairs for binary segmentation
    tasks.
Additionally, it provides utility functions for creating transformation pipelines and constructing
    training and validation datasets based on a configuration.

Classes:
    - SegmentationDataset: A dataset class for loading image and mask pairs for binary segmentation tasks.

Functions:
    - get_transform_by_config: Creates a transformation pipeline based on the provided configuration.
    - get_train_dataset_by_config: Creates a training dataset based on the provided configuration and transformation.
    - get_val_dataset_by_config: Creates a validation dataset based on the provided configuration and transformation.
"""

import os
import os.path as osp

import cv2
import albumentations as albu
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.features.adele_utils import read_label
from src.models.utils.config import ConfigHandler


class SegmentationDataset(Dataset):
    """
    A dataset for bubble binary segmentation task.
    Dataset folder should have the structure as given below:

        data/
        |--train
        |    |--image
        |    |    |--image0.png
        |    |    |-- ...
        |    |--mask
        |         |--image0.png
        |         |-- ...
        |--val

    """

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            transform=None,
            augmentation_transform=None,
            use_adele=False,
            return_names=False
    ):
        """
        Initializes the SegmentationDataset.

        Args:
            images_dir (str): Path to the directory containing images.
            masks_dir (str): Path to the directory containing masks.
            transform (callable, optional): A function/transform to apply to the images.
            augmentation_transform (callable, optional): A transform to apply to the images and masks
                for data augmentation.
            use_adele (bool, optional): Whether to use ADELE correction for masks. Default is False.
            return_names (bool, optional): Whether to return image names along with images and masks. Default is False.
    `   """
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.augmentation_transform = augmentation_transform

        self.to_tensor = ToTensor()

        self.images_list = os.listdir(self.images_dir)
        self.masks_list = os.listdir(self.masks_dir)

        self.use_adele = use_adele
        self.return_names = return_names

        assert len(self.images_list) == len(self.masks_list), "some images or masks are missing"

    def _read_image_and_mask(self, image_name):
        """
        Reads an image and its corresponding mask.

        Args:
            image_name (str): Name of the image file.

        Returns:
            tuple: A tuple containing the image and its mask.
        """
        image = cv2.imread(osp.join(self.images_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(osp.join(self.masks_dir, image_name), cv2.IMREAD_GRAYSCALE)

        if self.use_adele:
            corrected = read_label(image_name)
            mask = mask + corrected

        return image, mask

    def _apply_transform(self, transform, image, mask):
        """
        Applies transformation to image and mask.

        Args:
            transform (callable): The transformation to apply.
            image (numpy.ndarray): The input image.
            mask (numpy.ndarray): The input mask.

        Returns:
            tuple: A tuple containing the transformed image and mask.
        """
        res = transform(image=image, mask=mask)
        return res['image'], res['mask']

    def __getitem__(self, idx):
        """
        Gets the item (image and mask) at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding mask.
        """
        image, mask = self._read_image_and_mask(self.images_list[idx])

        if self.transform:
            image, mask = self._apply_transform(self.transform, image, mask)
        if self.augmentation_transform:
            image, mask = self._apply_transform(self.augmentation_transform, image, mask)

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)  # to tensor is used for the mask as the task is binary segmentation

        if self.return_names:
            return image, mask, self.images_list[idx]

        return image, mask
    
    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.images_list)


def get_transform_by_config(config_handler: ConfigHandler):
    """
    Creates a transformation pipeline based on the provided configuration.

    Args:
        config_handler (ConfigHandler): An instance of ConfigHandler containing the configuration data.

    Returns:
        callable: A callable representing the transformation pipeline.
    """
    transform = albu.Compose([
        albu.Resize(config_handler.read('dataset', 'image_size', 'height'),
                    config_handler.read('dataset', 'image_size', 'width')),
        albu.CLAHE(always_apply=True),
        albu.Normalize(always_apply=True),
        albu.RandomCrop(config_handler.read('dataset', 'image_size', 'height'),
                        config_handler.read('dataset', 'image_size', 'width'), p=0.4),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5)
    ])
    return transform


def get_train_dataset_by_config(config_handler: ConfigHandler, transform):
    """
    Creates a training dataset based on the provided configuration and transformation.

    Args:
        config_handler (ConfigHandler): An instance of ConfigHandler containing the configuration.
        transform (callable): A callable representing the transformation pipeline.

    Returns:
        SegmentationDataset: An instance of SegmentationDataset representing the training dataset.
    """
    train_dataset = SegmentationDataset(
        images_dir=osp.join(config_handler.read('dataset', 'training_dataset_dirs')[0], "images"),
        masks_dir=osp.join(config_handler.read('dataset', 'training_dataset_dirs')[0], "masks"),
        augmentation_transform=transform
    )
    return train_dataset


def get_val_dataset_by_config(config_handler: ConfigHandler, transform):
    """
    Creates a validation dataset based on the provided configuration and transformation.

    Args:
        config_handler (ConfigHandler): An instance of ConfigHandler containing the configuration.
        transform (callable): A callable representing the transformation pipeline.

    Returns:
        SegmentationDataset: An instance of SegmentationDataset representing the validation dataset.
    """
    val_dataset = SegmentationDataset(
        images_dir=osp.join(config_handler.read('dataset', 'validation_dataset_dirs')[0], "images"),
        masks_dir=osp.join(config_handler.read('dataset', 'validation_dataset_dirs')[0], "masks"),
        augmentation_transform=transform
    )
    return val_dataset
