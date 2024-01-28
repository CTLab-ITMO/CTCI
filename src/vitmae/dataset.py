import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader


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


def init_datasets(
        train_images_dir,
        val_images_dir,
        image_processor
):
    train_dataset = BubblesDataset(
        images_dir=train_images_dir,
        image_processor=image_processor
    )
    val_dataset = BubblesDataset(
        images_dir=val_images_dir,
        image_processor=image_processor
    )

    return train_dataset, val_dataset


def init_dataloaders(
        train_dataset,
        val_dataset,
        batch_size_train=64,
        batch_size_val=32,
        pin_memory=True,
        num_workers=4
):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    return train_dataloader, val_dataloader
