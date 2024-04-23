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
        image = cv2.imread(osp.join(self.images_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(osp.join(self.masks_dir, image_name), cv2.IMREAD_GRAYSCALE)

        if self.use_adele:
            corrected = read_label(image_name)
            mask = mask + corrected

        return image, mask

    def _apply_transform(self, transform, image, mask):
        res = transform(image=image, mask=mask)
        return res['image'], res['mask']

    def __getitem__(self, idx):
        image, mask = self._read_image_and_mask(self.images_list[idx])

        if self.transform:
            image, mask = self._apply_transform(self.transform, image, mask)
        if self.augmentation_transform:
            image, mask = self._apply_transform(self.augmentation_transform, image, mask)

        # to tensor is used for the mask as the task is binary segmentation
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        if self.return_names:
            return image, mask, self.images_list[idx]

        return image, mask
    
    def __len__(self):
        return len(self.images_list)


def get_transform_by_config(config_handler: ConfigHandler):
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
    train_dataset = SegmentationDataset(
        images_dir=osp.join(config_handler.read('dataset', 'training_dataset_dirs')[0], "images"),
        masks_dir=osp.join(config_handler.read('dataset', 'training_dataset_dirs')[0], "masks"),
        augmentation_transform=transform
    )
    return train_dataset


def get_val_dataset_by_config(config_handler: ConfigHandler, transform):
    val_dataset = SegmentationDataset(
        images_dir=osp.join(config_handler.read('dataset', 'validation_dataset_dirs')[0], "images"),
        masks_dir=osp.join(config_handler.read('dataset', 'validation_dataset_dirs')[0], "masks"),
        augmentation_transform=transform
    )
    return val_dataset
