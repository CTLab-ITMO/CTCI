import os.path as osp
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import AugmentationConfig, DataConfig
from src.dataset import SegmentationDataset
from src.transform import get_transforms


class CTCIDataModule(LightningDataModule):
    def __init__(self, cfg: DataConfig, aug_config: AugmentationConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self._train_transforms = get_transforms(aug_config.train)
        self._valid_transforms = get_transforms(aug_config.valid)

        self.save_hyperparameters(logger=False)

        self.data_path = self.cfg.data_dir

        self.data_train: Optional[SegmentationDataset] = None
        self.data_val: Optional[SegmentationDataset] = None
        self.data_test: Optional[SegmentationDataset] = None

        self.initialized: bool = False

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.data_train = SegmentationDataset(
                images_folder=osp.join(self.data_path, self.cfg.train_folder, 'images'),
                masks_folder=osp.join(self.data_path, self.cfg.train_folder, 'masks'),
                transform=self._train_transforms,
            )
            self.data_val = SegmentationDataset(
                images_folder=osp.join(self.data_path, self.cfg.valid_folder, 'images'),
                masks_folder=osp.join(self.data_path, self.cfg.valid_folder, 'masks'),
                transform=self._valid_transforms,
            )
        elif stage == 'test' and self.cfg.test_folder is not None:
            self.data_test = SegmentationDataset(
                images_folder=osp.join(self.data_path, self.cfg.test_folder, 'images'),
                masks_folder=osp.join(self.data_path, self.cfg.train_folder, 'masks'),
                transform=self._valid_transforms,
            )
        self.initialized = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )
