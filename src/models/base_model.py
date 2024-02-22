"""
an inspiration
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.net = None
        self.cls = None

    def set_loss_fn(self, loss_fn):
        pass

    def forward(self, x) -> torch.tensor:
        pass

    def predict(self, x):
        pass

    def _freeze_backbone() -> None:
        pass

    def _calc_loss_fn(self, input, target) -> torch.tensor:
        pass

    def _train_on_batch(self, input, target) -> torch.tensor:
        pass
