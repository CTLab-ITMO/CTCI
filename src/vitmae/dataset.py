import os

from PIL import Image
from torch.utils.data import Dataset


class BubblesDataset(Dataset):
    def __init__(self, images_dir, image_processor):
        self.images_dir = images_dir
        self.image_processor = image_processor

        self.image_list = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.images_dir, self.image_list[item]))
        inputs = self.image_processor(images=image, return_tensors="pt")

        return inputs