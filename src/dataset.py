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
import numpy as np
from torch.utils.data import Dataset


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
            images_folder: str,
            masks_folder: str,
            adele_dir: str = None,
            transform = None,
            return_names = False,
    ):
        """
        Initializes the SegmentationDataset.

        Args:
            images_folder (str): Path to the directory containing images.
            masks_folder (str): Path to the directory containing masks.
            transform (callable, optional): A function/transform to apply to the images.
            use_adele (bool, optional): Whether to use ADELE correction for masks. Default is False.
            return_names (bool, optional): Whether to return image names along with images and masks. Default is False.
    `   """
        super().__init__()

        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.transform = transform

        self.images_list = os.listdir(self.images_folder)
        self.masks_list = os.listdir(self.masks_folder)
        self.adele_dir = adele_dir
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
        image = cv2.imread(osp.join(self.images_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(osp.join(self.masks_folder, image_name), cv2.IMREAD_GRAYSCALE)

        return image, mask

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
            res = self.transform(image, mask)
            image, mask = res['image'], res['mask']

        if self.adele_dir and osp.isdir(self.adele_dir):
            correction = cv2.imread(
                osp.join(self.adele_dir, self.images_list[idx]),
                cv2.IMREAD_GRAYSCALE
                )
            mask = np.where(correction == 255, correction, mask)

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