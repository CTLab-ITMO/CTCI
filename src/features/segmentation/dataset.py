import os
import os.path as osp
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.features.adele_utils import read_label


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
            image_transform=None,
            mask_transform=None,
            augmentation_transform=None,
            use_adele=False,
            return_names=False
    ):
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
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

    def __getitem__(self, idx):
        image, mask = self._read_image_and_mask(self.images_list[idx])

        if self.augmentation_transform:
            augmentation_res = self.augmentation_transform(image=image, mask=mask)
            image = augmentation_res['image']
            mask = augmentation_res['mask']

        if self.image_transform:
            image = self.image_transform(image=image)['image']

        if self.mask_transform:
            mask = self.mask_transform(image=image)['image']

        # totensor is used for the mask as the task is binary segmentation
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        if self.return_names:
            return image, mask, self.images_list[idx]

        return image, mask
    
    def __len__(self):
        return len(self.images_list)