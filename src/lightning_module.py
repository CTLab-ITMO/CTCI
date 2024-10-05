from typing import Any, Dict

import hydra
import torch
from lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import MeanMetric

from src.config import ModuleConfig
from src.losses import get_losses
from src.metrics import get_classification_metrics, get_segmentation_metrics


class CTCILightningModule(LightningModule):
    def __init__(self, cfg: ModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.model = self._instantiate_model(self.cdf.model)

        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()
        self._seg_loss_fn = get_losses(cfg.seg_losses)
        self._cls_loss_fn = get_losses(cfg.cls_losses)
        self.thresh = cfg.threshold

        cls_metrics = get_classification_metrics(
            num_classes=cfg.num_classes,
            num_labels=cfg.num_classes,
            task='binary',
            average='macro',
            threshold=0.7,
        )
        seg_metrics = get_segmentation_metrics()

        self._val_cls_metrics = cls_metrics.clone(prefix='val_')
        self._test_cls_metrics = cls_metrics.clone(prefix='test_')
        self._val_seg_metrics = seg_metrics.clone(prefix='val_')
        self._test_seg_metrics = seg_metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: Tensor) -> Dict:
        images, targets = batch
        logits = self.model(images)
        loss = self._calculate_loss(logits, targets, 'train_')
        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss, 'logits': logits}

    def on_train_epoch_end(self) -> None:
        self.log(
            'mean_train_loss',
            self._train_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self._train_loss.reset()

    def validation_step(self, batch: Tensor) -> Tensor:
        images, targets = batch
        logits = self.model(images)
        loss = self._calculate_loss(logits, targets, 'valid_')
        self._valid_loss(loss)

        probs = torch.sigmoid(logits)
        self._val_cls_metrics(probs, targets)
        self._val_seg_metrics(probs, targets)
        preds = torch.zeros_like(probs)
        preds[probs > self.thresh] = 1
        return preds

    def on_validation_epoch_end(self) -> None:
        self.log(
            'mean_valid_loss',
            self._valid_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self._valid_loss.reset()

        self.log_dict(self._val_cls_metrics.compute(), on_epoch=True, prog_bar=True)
        self.log_dict(self._val_seg_metrics.compute(), on_epoch=True, prog_bar=True)
        self._val_cls_metrics.reset()
        self._val_seg_metrics.reset()

    def test_step(self, batch: Tensor) -> Tensor:
        images, targets = batch
        logits = self.model(images)

        probs = torch.sigmoid(logits)
        self._test_cls_metrics(probs, targets)
        self._test_seg_metrics(probs, targets)

        preds = torch.zeros_like(probs)
        preds[probs > self.thresh] = 1
        return preds

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_cls_metrics.compute(), on_epoch=True, prog_bar=True)
        self.log_dict(self._test_seg_metrics.compute(), on_epoch=True, prog_bar=True)
        self._test_cls_metrics.reset()
        self._test_seg_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.model.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def _instantiate_model(self, model_cfg) -> nn.Module:
        return hydra.utils.instantiate(model_cfg)

    def _calculate_loss(
        self,
        pred_masks_logits: Tensor,
        gt_masks: Tensor,
        prefix: str,
    ) -> Tensor:
        total_loss = 0
        for seg_loss in self._seg_loss_fn:
            loss = seg_loss.loss(pred_masks_logits, gt_masks)
            total_loss += seg_loss.weight * loss
            self.log(f'{prefix}{seg_loss.name}_loss', loss.item())
        for cls_loss in self._cls_loss_fn:
            loss = cls_loss.loss(pred_masks_logits, gt_masks)
            total_loss += cls_loss.weight * loss
            self.log(f'{prefix}{cls_loss.name}_loss', loss.item())
        return total_loss
