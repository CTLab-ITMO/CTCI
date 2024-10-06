import os
import os.path as osp
import torch
from torchvision.transforms import functional as vfunc
import pytorch_lightning as pl

from src.dataset import SegmentationDataset
from src.config import DataConfig


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
        self.correction_dataloader = correction_dataloader
        self.scales = scales
        self.confidence_thresh = confidence_thresh
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            for batch in self.correction_dataloader:
                images, targets, names = batch

                average_prediction = self._predict_average_on_scales(
                    pl_module=pl_module,
                    batch=images,
                )

                corrected_masks = self._correct_mask(
                    targets,
                    average_prediction,
                )

                for mask, name in zip(corrected_masks, names):
                    file_path = os.path.join(self.save_dir, name)
                    torch.save(mask, file_path)

    def _predict_on_scale(self, pl_module, batch, scale):
        _, _, h, w = batch.size()
        scaled = _interpolate_img(batch, scale=scale, size=(h, w))
        res = pl_module.predict(scaled)
        out = _interpolate_img(res, size=(h, w))
        return out

    def _predict_average_on_scales(self, pl_module, batch):
        preds = []
        for scale in self.scales:
            preds.append(self._predict_on_scale(pl_module, batch, scale))
        preds = torch.stack(preds)
        return preds.mean(dim=0).squeeze(0)

    def _correct_mask(self, target, average):
        new_target = []
        for t in target:
            new_target.append(torch.where(average > self.confidence_thresh, 1, t))
        new_target = torch.stack(new_target)
        return new_target


def _interpolate_img(image, scale, size):
    if scale:
        size = list(map(lambda x: int(x * scale), size))
        return vfunc.resize(image, size=size)
    return vfunc.resize(image, size=size)


def create_adele_dataloader(cfg: DataConfig):
    dataset = SegmentationDataset(
            images_folder=osp.join(cfg.data_path, cfg.train_folder, 'images'),
            masks_folder=osp.join(cfg.data_path, cfg.train_folder, 'masks'),
            return_names=True,
        )
    return torch.utils.data.Dataloader(
        dataset,
        batch_size=8,
        shuffle=False,
    )