import os
import os.path as osp

import cv2
import torch
from torchvision.transforms import functional as vfunc
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.constants import PROJECT_ROOT
from src.dataset import SegmentationDataset
from src.config import DataConfig
from src.transform import get_transforms, tensor_to_cv_image
from tqdm import tqdm


class AdeleCallback(pl.Callback):
    def __init__(
            self,
            correction_dataloader,
            scales=(0.75, 1, 1.25),
            confidence_thresh=0.6,
            save_dir="data/corrected_masks",
    ):
        """
        Args:
            scales (list): List of scales to use for mask correction.
            confidence_thresh (float): Confidence threshold for mask correction.
            save_dir (str): Directory to save corrected masks.
        """
        super().__init__()
        self.correction_dataloader = correction_dataloader
        self.scales = scales
        self.confidence_thresh = confidence_thresh
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            tqdm_iterator = tqdm(self.correction_dataloader)
            tqdm_iterator.set_description("Running ADELE correction...")
            for batch in tqdm_iterator:
                images, targets, names = batch

                images = images.to(pl_module.device)
                targets = targets.to(pl_module.device)

                average_prediction = self._predict_average_on_scales(
                    pl_module=pl_module,
                    images=images,
                )

                corrected_masks = self._correct_mask(
                    targets,
                    average_prediction,
                )

                for mask, name in zip(corrected_masks, names):
                    file_path = os.path.join(self.save_dir, name)
                    cv2.imwrite(file_path, tensor_to_cv_image(mask) * 255)

    def _predict_on_scale(self, pl_module, images, scale):
        _, _, h, w = images.size()
        scaled = _interpolate_img(images, scale=scale, size=(h, w))
        res = pl_module(scaled)
        out = _interpolate_img(res, size=(h, w))
        return out

    def _predict_average_on_scales(self, pl_module, images):
        preds = []
        for scale in self.scales:
            with torch.no_grad():
                preds.append(self._predict_on_scale(pl_module, images, scale))
        preds = torch.stack(preds)
        return preds.mean(dim=0).squeeze(0)

    def _correct_mask(self, targets, averages):
        new_target = []
        for target, average in zip(targets, averages):
            new_target.append(torch.where(average > self.confidence_thresh, 1, target))
        new_target = torch.stack(new_target)
        return new_target


def _interpolate_img(image, size, scale=None):
    if scale:
        size = list(map(lambda x: int(x * scale), size))
        return vfunc.resize(image, size=size)
    return vfunc.resize(image, size=size)


def create_adele_dataloader(cfg: DataConfig, valid_aug_confid: DictConfig):
    dataset = SegmentationDataset(
            images_folder=osp.join(PROJECT_ROOT, cfg.data_dir, cfg.train_folder, 'images'),
            masks_folder=osp.join(PROJECT_ROOT, cfg.data_dir, cfg.train_folder, 'masks'),
            return_names=True,
            transform=get_transforms(valid_aug_confid),
        )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )
