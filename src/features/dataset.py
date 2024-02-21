import os
import os.path as osp
import cv2
import torch
from torch.utils.data import Dataset

class BubblesDataset(Dataset):
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
            root='./data/',
            split='train',
            image_folder="image",
            mask_folder="mask",
            input_transform = None,
            ):
        """
        Args:
            root (str, optional): a folder, where dataset is located. Defaults to './data/'.
            split (str, optional): type of data reserved at certain split. Defaults to 'train'.
            image_folder (str, optional): folder name of images. Defaults to "image"
            mask_folder (str, optional): folder name of masks. Defaults to "mask"
            input_transform (_type_, optional): torchvision transform. Defaults to None.
        """
        super().__init__()

        self.data_path = osp.join(root, split)
        self.image_path = osp.join(self.data_path, image_folder)
        self.mask_path = osp.join(self.data_path, mask_folder)
        self.input_transform = input_transform

        self.images = os.listdir(self.image_path)

        assert len(self.images) == len(self.masks), "some images or masks are missing"


    def _read_image_and_mask(self, image_name):
        image = cv2.imread(osp.join(self.image_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BRG2RGB)
        mask = cv2.imread(osp.join(self.mask_path, image_name), cv2.IMREAD_GRAYSCALE)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask) / 255 #TODO: make normalization

        return image, mask


    def __getitem__(self, idx):
        image, mask = self._read_image_and_mask(self.images[idx])

        if self.input_transform:
            image = self.input_transform(image)
        
        return image, mask



