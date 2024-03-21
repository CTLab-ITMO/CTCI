import torch
import numpy as np
import torch.nn as nn
from src.models.unetr.decoder import UNETRDecoder


class Swin(nn.Module):
    def __init__(self, net, image_processor=None, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = net.to(device)
        self.decoder = UNETRDecoder()
        self.image_processor = image_processor
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image):
        out = self.encoder(pixel_values=image, output_hidden_states=True)
        out = self.decoder(out.reshaped_hidden_states, image)
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
        im = self.image_processor(image, return_tensors="pt").pixel_values
        return self.forward(im)
