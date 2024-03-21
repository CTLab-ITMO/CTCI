import torch
import numpy as np
import torch.nn as nn
from src.models.unetr.decoder import UNETRDecoder

from src.models.twins.twins_arch import Twins

class TwinsUNETR(nn.Module):
    def __init__(self, image_processor=None, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = Twins()
        self.decoder = UNETRDecoder(
            feature_size=64,
            hidden_size=[64, 128, 256, 512]
        )
        self.image_processor = image_processor
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image):
        out, hidden_states = self.encoder(image)
        out = self.decoder(hidden_states, image)
        return out

    def _calc_loss_fn(self, image, target):
        return self.loss_fn(image, target)

    def train_on_batch(self, image, target):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss

    def val_on_batch(self, image, target):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss, outputs

    def predict(self, image):
        im = self.image_processor.process(image)
        return self.forward(im)
