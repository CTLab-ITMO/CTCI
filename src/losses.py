import typing as tp
from dataclasses import dataclass
import hydra
from torch import nn
from src.config import LossConfig


@dataclass
class Loss:
    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: LossConfig) -> tp.List[Loss]:
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=hydra.utils.instantiate(loss_cfg.loss_fn),
        ) for loss_cfg in losses_cfg
    ]
