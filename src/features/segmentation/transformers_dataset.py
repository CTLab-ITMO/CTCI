import os
import os.path as osp

import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            image_transform=None,
            mask_transform=None,
            augmentation_transform=None,
            image_processor=None
    ):
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentation_transform = augmentation_transform
        self.image_processor = image_processor

        self.images_list = os.listdir(self.images_dir)
        self.masks_list = os.listdir(self.masks_dir)

        assert len(self.images_list) == len(self.masks_list), "some images or masks are missing"

    def _read_image_and_mask(self, image_name):
        image = cv2.imread(osp.join(self.images_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(osp.join(self.masks_dir, image_name), cv2.IMREAD_GRAYSCALE)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask) / 255

        return image, mask

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image, mask = self._read_image_and_mask(self.images_list[idx])

        if self.augmentation_transform:
            seed = torch.random.seed()
            torch.manual_seed(seed)
            image = self.augmentation_transform(image)
            mask = self.augmentation_transform(image)

        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs["pixel_values"], encoded_inputs["labels"]
