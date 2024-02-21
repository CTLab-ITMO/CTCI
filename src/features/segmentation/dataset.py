import os
import os.path as osp
import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    A dataset for bubble segmentation task. 
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
            image_transform=None,
            mask_transform=None
    ):
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.images_list = os.listdir(self.images_dir)
        self.masks_list = os.listdir(self.masks_dir)

        assert len(self.images_list) == len(self.masks_list), "some images or masks are missing"

    def _read_image_and_mask(self, image_name):
        image = cv2.imread(osp.join(self.images_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(osp.join(self.masks_dir, image_name), cv2.IMREAD_GRAYSCALE)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask) / 255  # TODO: make normalization

        return image, mask

    def __getitem__(self, idx):
        image, mask = self._read_image_and_mask(self.images_list[idx])

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
